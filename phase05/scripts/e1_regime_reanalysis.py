"""Phase 05 / E1 -- retrospective no-refill regime analysis (CPU only).

Reads committed phase01-04 receipts, recomputes RIC/work/slot utilization from
raw depth traces where they exist, reconciles them with the published tables and
writes one row per (cell, baseline -> policy) comparison with an explicit
evidence class.  No model, no CUDA.  Missing values are NaN, never zero.

  CUDA_VISIBLE_DEVICES="" python phase05/scripts/e1_regime_reanalysis.py
  CUDA_VISIBLE_DEVICES="" python phase05/scripts/e1_regime_reanalysis.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P1 = "phase01/results/20260916_four_gpu_validation"
P2 = "phase02/oracle_upper_bound_results_20260916"
P3 = "phase03/predictor_tournament_results_20260916"
P4V = "phase04/results/20260917T060824Z_50910e6_bf16_resolution"
P4R = "phase04/results/20260917T065413Z_e4113e5_rank_stability_cpu"
EXTERNAL_ORACLE_RIC = Path("/home/hwlee/batch-loop/results")  # read-only, outside repo

EVIDENCE = ("TRACE_ONLY", "LATENCY_MODEL", "PHYSICAL_FORCED_DEPTH", "PHYSICAL_LIVE_EXIT")
NAN = float("nan")

COLUMNS = [
    "cell_id", "evidence_class", "phase", "runtime_kind", "model", "model_key", "kv_policy",
    "rmax", "dataset", "threshold", "context", "N", "B", "N_over_B", "num_tokens",
    "baseline", "policy", "is_n_eq_b_control", "trace_sha256", "prompt_sha256",
    "same_work_checksum", "depth_stats_source", "depth_mean", "depth_std", "depth_cv",
    "depth_p10", "depth_p50", "depth_p90", "token_loop_work", "ric_baseline", "ric_policy",
    "ric_reduction", "ric_ratio_baseline_over_policy", "slot_util_baseline", "slot_util_policy",
    "maxdepth_dist_baseline", "maxdepth_dist_policy", "mean_active_width_baseline",
    "mean_active_width_policy", "core_ms_baseline_median", "core_ms_policy_median",
    "core_ms_baseline_min", "core_ms_baseline_max", "core_ms_policy_min", "core_ms_policy_max",
    "core_reduction", "core_speedup", "wall_ms_baseline_median", "wall_ms_policy_median",
    "wall_ms_baseline_min", "wall_ms_baseline_max", "wall_ms_policy_min", "wall_ms_policy_max",
    "wall_reduction", "wall_speedup", "core_ms_per_call_baseline", "core_ms_per_call_policy",
    "core_tracking_ratio", "wall_tracking_ratio", "noncore_wall_fraction_baseline",
    "noncore_wall_fraction_policy", "selection_overhead_ms_policy", "selection_overhead_ratio",
    "model_time_ms_baseline", "model_time_ms_policy", "model_reduction", "model_speedup",
    "live_ric_minus_frozen_ric", "warmups", "measured_repeats", "reconciliation", "source_files",
    "source_rows", "notes",
]


# ---------------------------------------------------------------- arithmetic

def reduction(t_base: float, t_policy: float) -> float:
    """Normalized reduction (T_base - T_policy) / T_base; NaN if undefined."""
    if not (_finite(t_base) and _finite(t_policy)) or t_base <= 0:
        return NAN
    return (t_base - t_policy) / t_base


def speedup(t_base: float, t_policy: float) -> float:
    """Speedup T_base / T_policy; NaN if undefined. speedup = 1 / (1 - reduction)."""
    if not (_finite(t_base) and _finite(t_policy)) or t_policy <= 0:
        return NAN
    return t_base / t_policy


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fifo_groups(n: int, b: int) -> list[list[int]]:
    if n % b:
        raise ValueError("N must be divisible by B")
    return [list(range(i, i + b)) for i in range(0, n, b)]


def order_groups(order: list[int], b: int) -> list[list[int]]:
    return [order[i:i + b] for i in range(0, len(order), b)]


def oracle_groups(depths, b: int) -> list[list[int]]:
    """Sort by true depth, then request index (stable), chunk into B."""
    order = sorted(range(len(depths)), key=lambda i: (int(depths[i]), i))
    if len(order) % b:
        raise ValueError("N must be divisible by B")
    return order_groups(order, b)


def group_maxima(depths, groups) -> list[int]:
    return [max(int(depths[i]) for i in g) for g in groups]


def schedule_metrics(matrix: np.ndarray, groups_fn) -> dict:
    """matrix[T, N] of per-token depths; groups_fn(depths_t) -> list of groups.

    RIC = sum_t sum_m max_{i in m} d_i ; work = sum d_i ;
    slot utilization = work / sum_m |m| * max_{i in m} d_i.
    """
    ric = work = capacity = 0
    maxima_all: list[int] = []
    for t in range(matrix.shape[0]):
        depths = matrix[t]
        groups = groups_fn(depths)
        members = sorted(i for g in groups for i in g)
        if members != list(range(len(depths))):
            raise AssertionError("each request must appear exactly once per decode boundary")
        maxima = group_maxima(depths, groups)
        ric += sum(maxima)
        capacity += sum(len(g) * m for g, m in zip(groups, maxima))
        work += int(np.sum(depths))
        maxima_all.extend(maxima)
    return {"ric": ric, "work": work, "capacity": capacity,
            "slot_util": work / capacity if capacity else NAN,
            "maxdepth_dist": json.dumps(dict(sorted(Counter(maxima_all).items())), separators=(",", ":"))}


def depth_stats(values) -> dict:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {k: NAN for k in ("depth_mean", "depth_std", "depth_cv", "depth_p10", "depth_p50", "depth_p90")}
    mean, std = float(arr.mean()), float(arr.std())
    return {"depth_mean": mean, "depth_std": std, "depth_cv": std / mean if mean else NAN,
            "depth_p10": float(np.percentile(arr, 10)), "depth_p50": float(np.percentile(arr, 50)),
            "depth_p90": float(np.percentile(arr, 90))}


def depth_stats_from_hist(hist: dict[int, int]) -> dict:
    return depth_stats(np.repeat(np.array(list(hist.keys()), float), np.array(list(hist.values()), int)))


def cohort_checksum(matrix_tn: np.ndarray) -> str:
    """Same hash as phase02_runtime.oracle_ric._cohort_hash (request-major)."""
    digest = hashlib.sha256()
    for rid in range(matrix_tn.shape[1]):
        for t in range(matrix_tn.shape[0]):
            digest.update(f"{rid},{t},{int(matrix_tn[t, rid])}\n".encode())
    return digest.hexdigest()


def fifo_oracle_metrics(matrix: np.ndarray, b: int) -> tuple[dict, dict]:
    n = matrix.shape[1]
    return (schedule_metrics(matrix, lambda d: fifo_groups(n, b)),
            schedule_metrics(matrix, lambda d: oracle_groups(d, b)))


# ---------------------------------------------------------------- input registry

class Registry:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, dict] = {}

    def path(self, rel: str) -> Path:
        return self.root / rel

    def register(self, rel: str, *, rows=None, columns=None, role="", extra=None) -> str:
        p = self.path(rel) if not Path(rel).is_absolute() else Path(rel)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        entry = self.files.setdefault(str(rel), {"path": str(rel), "sha256": h, "bytes": p.stat().st_size,
                                                  "exists": True, "role": role})
        if rows is not None:
            entry["rows"] = rows
        if columns is not None:
            entry["columns"] = list(columns)
        if extra:
            entry.update(extra)
        return h

    def missing(self, rel: str, role: str, note: str):
        self.files[str(rel)] = {"path": str(rel), "exists": False, "role": role, "note": note}

    def csv(self, rel: str, role: str = "") -> list[dict]:
        with self.path(rel).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.register(rel, rows=len(rows), columns=reader.fieldnames, role=role)
        return rows

    def json(self, rel: str, role: str = "") -> dict:
        self.register(rel, role=role)
        return json.loads(self.path(rel).read_text())

    def jsonl_gz(self, rel: str, role: str = "") -> list[dict]:
        p = self.path(rel)
        opener = gzip.open if p.suffix == ".gz" else open
        raw = opener(p, "rb").read()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        self.register(rel, rows=len(rows), columns=sorted(rows[0]) if rows else [], role=role,
                      extra={"decompressed_sha256": hashlib.sha256(raw).hexdigest()})
        return rows


def fnum(x) -> float:
    if x is None or x == "" or str(x).lower() in ("nan", "none"):
        return NAN
    return float(x)


def base_row(**kw) -> dict:
    row = {c: NAN for c in COLUMNS}
    for c in ("notes", "reconciliation", "source_files", "source_rows", "maxdepth_dist_baseline",
              "maxdepth_dist_policy", "trace_sha256", "prompt_sha256", "same_work_checksum",
              "depth_stats_source", "context", "dataset", "baseline", "policy", "model_key"):
        row[c] = ""
    row.update(kw)
    if row["evidence_class"] not in EVIDENCE:
        raise ValueError(row["evidence_class"])
    if _finite(row["N"]) and _finite(row["B"]):
        row["N_over_B"] = float(row["N"]) / float(row["B"])
        row["is_n_eq_b_control"] = bool(int(row["N"]) == int(row["B"]))
    for a, b, red, ratio in (("ric_baseline", "ric_policy", "ric_reduction", "ric_ratio_baseline_over_policy"),
                             ("core_ms_baseline_median", "core_ms_policy_median", "core_reduction", "core_speedup"),
                             ("wall_ms_baseline_median", "wall_ms_policy_median", "wall_reduction", "wall_speedup"),
                             ("model_time_ms_baseline", "model_time_ms_policy", "model_reduction", "model_speedup")):
        if _finite(row[a]) and _finite(row[b]):
            row[red], row[ratio] = reduction(row[a], row[b]), speedup(row[a], row[b])
    rr = row["ric_reduction"]
    if _finite(rr) and rr != 0:
        if _finite(row["core_reduction"]):
            row["core_tracking_ratio"] = row["core_reduction"] / rr
        if _finite(row["wall_reduction"]):
            row["wall_tracking_ratio"] = row["wall_reduction"] / rr
    return row


def tag(th) -> str:
    return str(th).replace(".", "p")


# ---------------------------------------------------------------- trace inputs

class Traces:
    """Lazy loader of committed per-token depth traces -> matrix[T, N]."""

    def __init__(self, reg: Registry):
        self.reg, self.cache = reg, {}

    def rel(self, model: str, source: str, threshold: str) -> str:
        if model == "ouro":
            return f"{P2}/trace_inputs/ouro14_{source}_q{tag(threshold)}.jsonl.gz"
        return f"{P2}/trace_inputs/huginn_{source}_threshold_{tag(threshold)}_b1.jsonl.gz"

    def get(self, model, source, threshold) -> tuple[np.ndarray, str]:
        rel = self.rel(model, source, threshold)
        if rel not in self.cache:
            rows = self.reg.jsonl_gz(rel, role="raw B1 depth trace (forced-depth/trace input)")
            n = max(r["request_id"] for r in rows) + 1
            t = max(r["token_idx"] for r in rows) + 1
            m = np.zeros((t, n), dtype=np.int16)
            for r in rows:
                m[r["token_idx"], r["request_id"]] = r["depth"]
            if (m == 0).any():
                raise ValueError(f"incomplete trace {rel}")
            self.cache[rel] = (m, self.reg.files[rel]["decompressed_sha256"])
        return self.cache[rel]


# ---------------------------------------------------------------- phase02 trace sweeps

def phase02_trace_sweeps(reg, traces, cells, notes):
    for horizon in ("32tok", "64tok"):
        rel = f"{P2}/trace_sim_{horizon}/oracle_ric_sweep.csv"
        rows = reg.csv(rel, role=f"phase02 trace-only FIFO/Oracle RIC sweep ({horizon})")
        reg.json(f"{P2}/trace_sim_{horizon}/run_manifest.json", role="trace-sim manifest")
        reg.csv(f"{P2}/trace_sim_{horizon}/depth_distributions.csv", role="depth distribution table")
        mismatches = 0
        for idx, r in enumerate(rows, 1):
            n, b, t = int(r["N"]), int(r["B"]), int(r["num_tokens"])
            m, tsha = traces.get(r["model"], r["source"], r["threshold"])
            sub = m[:t, :n]
            fifo, orc = fifo_oracle_metrics(sub, b)
            checks = {"trace_sha": tsha == r["trace_sha256"], "fifo_ric": fifo["ric"] == int(r["fifo_ric"]),
                      "oracle_ric": orc["ric"] == int(r["oracle_ric"]),
                      "work": fifo["work"] == int(r["total_token_loop_work"]),
                      "checksum": cohort_checksum(sub) == r["same_work_checksum"]}
            ok = all(checks.values())
            mismatches += not ok
            ctx = "native uncapped prompts (Alpaca cap512 / ShareGPT cap2048)" if r["model"] == "ouro" else "prompt cap 64"
            cells.append(base_row(
                cell_id=f"p2trace_{horizon}_{r['model']}_{r['source']}_{r['threshold']}_N{n}_B{b}",
                evidence_class="TRACE_ONLY", phase="phase02", runtime_kind="trace-sim",
                model=r["model"], model_key="ouro14" if r["model"] == "ouro" else "huginn35",
                kv_policy="native-reference (trace source)" if r["model"] == "ouro" else "shared-physical-layer (trace source)",
                rmax=4 if r["model"] == "ouro" else 16, dataset=r["source"], threshold=float(r["threshold"]),
                context=ctx, N=n, B=b, num_tokens=t, baseline="FIFO", policy="ORACLE_DEPTH_SORT",
                trace_sha256=r["trace_sha256"], same_work_checksum=r["same_work_checksum"],
                depth_stats_source="recomputed from raw trace", **depth_stats(sub),
                token_loop_work=fifo["work"], ric_baseline=fifo["ric"], ric_policy=orc["ric"],
                slot_util_baseline=fifo["slot_util"], slot_util_policy=orc["slot_util"],
                maxdepth_dist_baseline=fifo["maxdepth_dist"], maxdepth_dist_policy=orc["maxdepth_dist"],
                reconciliation="OK" if ok else "MISMATCH:" + ",".join(k for k, v in checks.items() if not v),
                source_files=f"{rel};{traces.rel(r['model'], r['source'], r['threshold'])}",
                source_rows=f"{rel}:{idx}", notes="logical count only; not latency"))
        notes.append(f"phase02 trace_sim_{horizon}: {len(rows)} rows recomputed from raw traces, {mismatches} mismatches")


def external_oracle_ric(reg, notes):
    """Record (read-only) what exists outside the repo under results/oracle_ric_*."""
    repo64 = {(r["model"], r["source"], r["threshold"], r["N"], r["B"], r["fifo_ric"], r["oracle_ric"])
              for r in csv.DictReader(open(reg.path(f"{P2}/trace_sim_64tok/oracle_ric_sweep.csv")))}
    found = []
    if EXTERNAL_ORACLE_RIC.exists():
        for d in sorted(EXTERNAL_ORACLE_RIC.glob("oracle_ric_*")):
            f = d / "oracle_ric_sweep.csv"
            if not f.exists():
                reg.missing(str(f), "external oracle_ric sweep", "directory without sweep csv")
                continue
            rows = list(csv.DictReader(open(f)))
            keys = {(r["model"], r["source"], r["threshold"], r["N"], r["B"], r["fifo_ric"], r["oracle_ric"]) for r in rows}
            nt = sorted({r.get("num_tokens") for r in rows})
            reg.register(str(f), rows=len(rows), columns=list(rows[0]) if rows else [],
                         role="external read-only legacy oracle_ric sweep (not copied)",
                         extra={"num_tokens": nt, "rows_matching_repo_64tok_sweep": len(keys & repo64)})
            found.append(f"{d.name}: {len(rows)} rows, num_tokens={nt}, {len(keys & repo64)} match repo 64tok sweep")
    notes.append("phase02/trace_sim_64tok/oracle_ric_sweep.csv EXISTS in this branch (tracked, 192 rows). "
                 "External legacy sweeps: " + ("; ".join(found) if found else "none found"))


# ---------------------------------------------------------------- phase01

def phase01(reg, traces, cells, excluded, notes):
    audit = reg.csv(f"{P1}/ouro_trace_audit.csv", role="phase01 trace audit (depth histograms)")
    hist = {}
    for r in audit:
        key = (r["model"], r["dataset"], str(float(r["threshold"])))
        hist[key] = ({d: int(r[f"depth_{d}_tokens"]) for d in range(1, 5)}, r["trace_sha256"], int(r["fixed_decode_tokens"]))
    head = reg.csv(f"{P1}/ouro_headroom_fair.csv", role="phase01 trace-only fair headroom")
    reg.csv(f"{P1}/ouro_fairness_validation.csv", role="phase01 equal-work checks")
    by_key = defaultdict(dict)
    for idx, r in enumerate(head, 1):
        by_key[(r["run"], int(r["batch_size"]))][r["strategy"]] = (idx, r)
    recon_bad = 0
    for (run, b), strat in sorted(by_key.items()):
        fidx, f = strat["fifo"]
        mk, ds, th = f["model_key"], f["dataset"], str(float(f["exit_threshold"]))
        n, ntok = 256, int(f["num_selected_tokens"]) // 256
        h, tsha, fixed_tokens = hist[(mk, ds, th)]
        recomputed = {}
        if mk == "ouro14":
            m, rsha = traces.get("ouro", ds, f["exit_threshold"])
            recomputed = {"fifo": schedule_metrics(m[:ntok, :n], lambda d: fifo_groups(n, b)),
                          "oracle": schedule_metrics(m[:ntok, :n], lambda d: oracle_groups(d, b))}
        for s, (pidx, p) in sorted(strat.items()):
            if s == "fifo":
                continue
            ric_f, ric_p = int(f["total_capacity_slots"]) // b, int(p["total_capacity_slots"]) // b
            assert ric_f * b == int(f["total_capacity_slots"]) and ric_p * b == int(p["total_capacity_slots"])
            if mk == "ouro14":
                ok = (recomputed["fifo"]["ric"] == ric_f and recomputed["fifo"]["work"] == int(f["total_token_loop_work"])
                      and (s != "oracle" or recomputed["oracle"]["ric"] == ric_p))
                rec = "OK (recomputed from raw trace)" if ok else "MISMATCH"
                recon_bad += not ok
            else:
                rec = "NOT_RECOMPUTABLE: raw Ouro-2.6B trace not committed; capacity/B identity only"
            cells.append(base_row(
                cell_id=f"p1head_{run}_N{n}_B{b}_{s}", evidence_class="TRACE_ONLY", phase="phase01",
                runtime_kind="trace-sim (fair headroom)", model="ouro", model_key=mk, kv_policy="native-reference (trace source)",
                rmax=4, dataset=ds, threshold=float(th),
                context="Alpaca cap512" if ds == "alpaca" else "ShareGPT cap2048", N=n, B=b, num_tokens=ntok,
                baseline="FIFO", policy=s.upper() if s != "oracle" else "ORACLE_DEPTH_SORT", trace_sha256=tsha,
                depth_stats_source="phase01 trace audit histogram", **depth_stats_from_hist(h),
                token_loop_work=int(f["total_token_loop_work"]), ric_baseline=ric_f, ric_policy=ric_p,
                slot_util_baseline=fnum(f["slot_utilization"]), slot_util_policy=fnum(p["slot_utilization"]),
                maxdepth_dist_baseline=recomputed.get("fifo", {}).get("maxdepth_dist", ""),
                maxdepth_dist_policy=recomputed.get("oracle", {}).get("maxdepth_dist", "") if s == "oracle" else "",
                reconciliation=rec, source_files=f"{P1}/ouro_headroom_fair.csv",
                source_rows=f"{P1}/ouro_headroom_fair.csv:{fidx},{pidx}",
                notes="RIC = total_capacity_slots / B (all microbatches full); logical only"))
    notes.append(f"phase01 headroom: {len(by_key)} run x B groups; ouro14 recompute mismatches={recon_bad}")

    # hardware-calibrated replay = latency model
    rep = reg.csv(f"{P1}/ouro_hardware_replay_b4_b8.csv", role="phase01 hardware-calibrated trace replay (latency model)")
    grp = defaultdict(dict)
    for idx, r in enumerate(rep, 1):
        grp[(r["run"], int(r["batch_size"]))][r["strategy"]] = (idx, r)
    for (run, b), strat in sorted(grp.items()):
        fidx, f = strat["fifo"]
        mk = run.split("_")[0]
        for s, (pidx, p) in sorted(strat.items()):
            if s == "fifo":
                continue
            hf = by_key[(run, b)]
            ric_f = int(hf["fifo"][1]["total_capacity_slots"]) // b
            ric_p = int(hf[s][1]["total_capacity_slots"]) // b if s in hf else NAN
            row = base_row(
                cell_id=f"p1replay_{run}_N256_B{b}_{s}", evidence_class="LATENCY_MODEL", phase="phase01",
                runtime_kind="hardware-calibrated trace replay (not integrated runtime)", model="ouro", model_key=mk,
                kv_policy="native-reference", rmax=4, dataset=f["dataset"], threshold=float(f["exit_threshold"]),
                context="Alpaca cap512" if f["dataset"] == "alpaca" else "ShareGPT cap2048", N=256, B=b,
                num_tokens=int(f["selected_generated_tokens"]) // 256, baseline="FIFO",
                policy="ORACLE_DEPTH_SORT" if s == "oracle" else s.upper(), ric_baseline=ric_f, ric_policy=ric_p,
                model_time_ms_baseline=fnum(f["decode_replay_ms"]), model_time_ms_policy=fnum(p["decode_replay_ms"]),
                reconciliation="bct_speedup column matches recomputed BCT ratio" if
                abs(speedup(fnum(f["hardware_replay_bct_ms"]), fnum(p["hardware_replay_bct_ms"])) - fnum(p["bct_speedup_vs_fifo"])) < 1e-9
                else "MISMATCH bct_speedup", source_files=f"{P1}/ouro_hardware_replay_b4_b8.csv",
                source_rows=f"{P1}/ouro_hardware_replay_b4_b8.csv:{fidx},{pidx}",
                notes=("model output from fixed-depth native-reference calibration fits; decode_replay_ms excludes "
                       f"prefill anchor; BCT speedup incl. prefill = {fnum(p['bct_speedup_vs_fifo']):.4f}; not GPU-measured policy latency"))
            cells.append(row)

    # native E2E: never an early-exit baseline; excluded from all comparisons
    nat = reg.csv(f"{P1}/ouro_native_e2e.csv", role="phase01 native HF whole-model (post-hoc exit) E2E")
    for idx, r in enumerate(nat, 1):
        reason = ("OOM_PARTIAL native E2E cohort: excluded from throughput comparisons (capacity evidence only)"
                  if r["cohort_status"] != "COMPLETE" else
                  "native Ouro HF computes full recurrence and applies exit post hoc: not a physical early-exit "
                  "baseline and has no FIFO/Oracle pairing")
        excluded.append({"source_file": f"{P1}/ouro_native_e2e.csv", "source_row": idx, "model_key": r["model_key"],
                         "dataset": r["dataset"], "threshold": r["threshold"], "N": r["num_requests_target"],
                         "B": r["batch_size"], "status": r["cohort_status"], "evidence_kind": r["evidence_kind"],
                         "reason": reason})

    # calibration surface (fixed-depth native-reference) -> latency-model table only
    cal = reg.csv(f"{P1}/ouro_calibration.csv", role="phase01 fixed-depth native-reference calibration")
    for idx, r in enumerate(cal, 1):
        if r["status"] != "OK":
            excluded.append({"source_file": f"{P1}/ouro_calibration.csv", "source_row": idx, "model_key": r["model_key"],
                             "dataset": "", "threshold": "", "N": "", "B": r["batch_size"], "status": r["status"],
                             "evidence_kind": r["evidence_kind"],
                             "reason": f"calibration OOM at B={r['batch_size']} context={r['context_len']} depth={r['depth']}; "
                                       "retained as capacity evidence, no latency value"})
    fit = reg.csv(f"{P1}/ouro_calibration_fit.csv", role="phase01 calibration linear fits (latency model)")
    surface = []
    for idx, r in enumerate(fit, 1):
        b = int(r["batch_size"])
        surface.append({"evidence_class": "LATENCY_MODEL", "model_key": r["model_key"], "kv_policy": r["kv_policy"],
                        "B": b, "context_len": int(r["context_len"]), "recurrent_step_ms": fnum(r["recurrent_step_ms"]),
                        "recurrent_step_ms_per_row": fnum(r["recurrent_step_ms"]) / b,
                        "boundary_ms": fnum(r["boundary_ms"]), "fit_r2": fnum(r["fit_r2"]),
                        "full_depth_incremental_decode_ms": fnum(r["full_depth_incremental_decode_ms"]),
                        "source_row": f"{P1}/ouro_calibration_fit.csv:{idx}",
                        "note": "native HF fixed-depth fit, not survivor runtime; no bottleneck regime inferred"})
    for doc in ("RESULT_REVIEW.md", "review_manifest.json", "raw_source_inventory.json"):
        reg.register(f"{P1}/{doc}", role="phase01 review/provenance doc")
    return surface


# ---------------------------------------------------------------- physical helpers

def med_min_max(vals):
    vals = [v for v in vals if _finite(v)]
    if not vals:
        return NAN, NAN, NAN
    return float(np.median(vals)), float(min(vals)), float(max(vals))


def events_by_strategy(rows, rep_ok=lambda r: True):
    out = defaultdict(lambda: {"widths": [], "ms": []})
    for r in rows:
        if rep_ok(r):
            s = r["strategy"].upper()
            out[s]["widths"].append(r["active_batch_size"])
            out[s]["ms"].append(r["recurrent_pass_gpu_ms_including_kv"])
    return out


def width_cost_rows(cell_id, model, ev, evidence, src):
    rows = []
    for s, d in sorted(ev.items()):
        w, ms = np.array(d["widths"]), np.array(d["ms"])
        for width in sorted(set(w.tolist())):
            sel = ms[w == width]
            rows.append({"cell_id": cell_id, "evidence_class": evidence, "model": model, "strategy": s,
                         "active_width": int(width), "calls": int(sel.size), "median_call_ms": float(np.median(sel)),
                         "p10_call_ms": float(np.percentile(sel, 10)), "p90_call_ms": float(np.percentile(sel, 90)),
                         "source_file": src})
    return rows


# ---------------------------------------------------------------- phase02 physical

def phase02_physical(reg, traces, cells, excluded, width_rows, notes):
    rel = f"{P2}/physical_matrix.csv"
    matrix = reg.csv(rel, role="phase02 physical FIFO/Oracle matrix")
    reg.register(f"{P2}/RESULT_REVIEW.md", role="phase02 review")
    reg.register(f"{P2}/review_manifest.json", role="phase02 review manifest")
    used = set()
    for idx, mrow in enumerate(matrix, 1):
        run = mrow["run_dir"].replace("phase02/oracle_upper_bound_results_20260916/", "")
        used.add(Path(run).name)
        man = reg.json(f"{P2}/{run}/run_manifest.json", role="phase02 physical run manifest")
        summ = reg.csv(f"{P2}/{run}/summary.csv", role="phase02 physical per-repetition summary")
        ev_rows = reg.jsonl_gz(f"{P2}/{run}/recurrence_events.jsonl.gz", role="phase02 recurrence-event receipts")
        model = mrow["model"]
        n, b, t = int(mrow["N"]), int(mrow["B"]), int(mrow["num_tokens"])
        source = mrow["prompt_source"]
        m, tsha = traces.get(model, source, mrow["threshold"])
        sub = m[:t, :n]
        fifo, orc = fifo_oracle_metrics(sub, b)
        by = defaultdict(list)
        for r in summ:
            by[r["strategy"].upper()].append(r)
        core = {s: med_min_max([fnum(r["recurrent_pass_gpu_ms_including_kv"]) for r in v]) for s, v in by.items()}
        wall = {s: med_min_max([fnum(r["total_decode_wall_ms"]) for r in v]) for s, v in by.items()}
        calls = {s: sorted({int(r["recurrent_invocation_count"]) for r in v}) for s, v in by.items()}
        ev = events_by_strategy(ev_rows)
        cell_id = f"p2phys_{model}_{source}_{mrow['threshold']}_N{n}_B{b}"
        width_rows += width_cost_rows(cell_id, model, ev, "PHYSICAL_FORCED_DEPTH", f"{P2}/{run}/recurrence_events.jsonl.gz")
        checks = {
            "trace_sha": tsha == mrow["trace_sha256"],
            "fifo_ric": calls["FIFO"] == [fifo["ric"]] == [int(mrow["fifo_ric"])],
            "oracle_ric": calls["ORACLE"] == [orc["ric"]] == [int(mrow["oracle_ric"])],
            "work": fifo["work"] == int(mrow["total_token_loop_work"]),
            "event_calls": len(ev["FIFO"]["ms"]) == fifo["ric"] * len(by["FIFO"]),
            "core_median": abs(core["FIFO"][0] - fnum(mrow["fifo_median_recurrent_pass_gpu_ms_including_kv"])) < 1e-6
                           and abs(core["ORACLE"][0] - fnum(mrow["oracle_median_recurrent_pass_gpu_ms_including_kv"])) < 1e-6,
            "wall_median": abs(wall["FIFO"][0] - fnum(mrow["fifo_median_total_decode_wall_ms"])) < 1e-6
                           and abs(wall["ORACLE"][0] - fnum(mrow["oracle_median_total_decode_wall_ms"])) < 1e-6,
        }
        ok = all(checks.values())
        notes_txt = ("trace-forced frozen B1 depths, survivor-compacted no-refill runtime; outputs not quality-equivalent")
        if model == "ouro":
            notes_txt += ("; Ouro depth trace collected on uncapped native prompts but physical run caps prompts at 64 "
                          "tokens (phase02 review: 4/256 Alpaca, 80/256 ShareGPT truncated) -- FIFO/Oracle pair is matched")
        reps = int(man["repeats"])
        if reps < 5:
            notes_txt += f"; only {reps} measured repeats (<5 required for final scored cells)"
        cells.append(base_row(
            cell_id=cell_id, evidence_class="PHYSICAL_FORCED_DEPTH", phase="phase02",
            runtime_kind="survivor-compacted no-refill runtime, trace-forced depth", model=model,
            model_key="ouro14" if model == "ouro" else "huginn35", kv_policy=mrow["kv_backend"],
            rmax=4 if model == "ouro" else 16, dataset=source, threshold=float(mrow["threshold"]),
            context=f"prompt cap 64 (tokenized width {man.get('tokenized_prompt_width')}) + {t} decode tokens",
            N=n, B=b, num_tokens=t, baseline="FIFO", policy="ORACLE_DEPTH_SORT", trace_sha256=mrow["trace_sha256"],
            prompt_sha256=mrow["prompt_sha256"], same_work_checksum=mrow["same_work_checksum"],
            depth_stats_source="recomputed from raw trace (first N requests x first T tokens)", **depth_stats(sub),
            token_loop_work=fifo["work"], ric_baseline=fifo["ric"], ric_policy=orc["ric"],
            slot_util_baseline=fifo["slot_util"], slot_util_policy=orc["slot_util"],
            maxdepth_dist_baseline=fifo["maxdepth_dist"], maxdepth_dist_policy=orc["maxdepth_dist"],
            mean_active_width_baseline=float(np.mean(ev["FIFO"]["widths"])),
            mean_active_width_policy=float(np.mean(ev["ORACLE"]["widths"])),
            core_ms_baseline_median=core["FIFO"][0], core_ms_policy_median=core["ORACLE"][0],
            core_ms_baseline_min=core["FIFO"][1], core_ms_baseline_max=core["FIFO"][2],
            core_ms_policy_min=core["ORACLE"][1], core_ms_policy_max=core["ORACLE"][2],
            wall_ms_baseline_median=wall["FIFO"][0], wall_ms_policy_median=wall["ORACLE"][0],
            wall_ms_baseline_min=wall["FIFO"][1], wall_ms_baseline_max=wall["FIFO"][2],
            wall_ms_policy_min=wall["ORACLE"][1], wall_ms_policy_max=wall["ORACLE"][2],
            core_ms_per_call_baseline=core["FIFO"][0] / fifo["ric"], core_ms_per_call_policy=core["ORACLE"][0] / orc["ric"],
            noncore_wall_fraction_baseline=1 - core["FIFO"][0] / wall["FIFO"][0],
            noncore_wall_fraction_policy=1 - core["ORACLE"][0] / wall["ORACLE"][0],
            selection_overhead_ms_policy=NAN, warmups=int(man["warmups"]), measured_repeats=reps,
            reconciliation="OK" if ok else "MISMATCH:" + ",".join(k for k, v in checks.items() if not v),
            source_files=f"{rel};{P2}/{run}/summary.csv;{P2}/{run}/recurrence_events.jsonl.gz",
            source_rows=f"{rel}:{idx}", notes=notes_txt))
    for d in sorted(reg.path(f"{P2}/physical").glob("*/*")):
        if d.name in used:
            continue
        rel_d = str(d.relative_to(reg.root))
        man = reg.json(f"{rel_d}/run_manifest.json", role="phase02 non-scored physical run manifest")
        if (d / "failure_receipt.json").exists():
            fr = reg.json(f"{rel_d}/failure_receipt.json", role="phase02 failure receipt")
            reason = f"failed run ({fr['status']}): {fr['observation']}"
            status = fr["status"]
        else:
            reg.csv(f"{rel_d}/summary.csv", role="phase02 exploratory run summary")
            reason = (f"exploratory {man['repeats']}-repeat run superseded by the 5-repeat matrix cell; "
                      "kept for audit, not substituted")
            status = "EXPLORATORY_SUPERSEDED"
        excluded.append({"source_file": f"{rel_d}/run_manifest.json", "source_row": "", "model_key": man["model"],
                         "dataset": man["prompt_source"], "threshold": man["threshold"], "N": man["num_rows"],
                         "B": man["batch_size"], "status": status, "evidence_kind": man["evidence_kind"], "reason": reason})
    for d in sorted(reg.path(f"{P2}/validation").glob("*")):
        rel_d = str(d.relative_to(reg.root))
        man = reg.json(f"{rel_d}/run_manifest.json", role="phase02 validation manifest")
        excluded.append({"source_file": f"{rel_d}/run_manifest.json", "source_row": "", "model_key": "huginn",
                         "dataset": man.get("prompt_source", man.get("dataset", "")), "threshold": man.get("threshold", ""),
                         "N": man.get("num_rows", ""), "B": man.get("batch_size", ""), "status": "VALIDATION_ONLY",
                         "evidence_kind": man.get("evidence_kind", man.get("execution_kind", "")),
                         "reason": "small trace-collector/runtime agreement check (N4, 2 tokens); not a FIFO/Oracle comparison"})


# ---------------------------------------------------------------- phase03 physical

P3_DIRS = {  # dir -> (model, regime/labels dir, threshold, N request ids)
    "ouro_physical": ("ouro", "ouro_q0p2_n640_t32", 0.2, range(384, 640)),
    "huginn_physical": ("huginn", "huginn_threshold_0p12_n640_t32", 0.12, range(384, 640)),
    "ouro_q0p4_physical": ("ouro", "ouro_q0p4_n640_t32", 0.4, range(384, 640)),
    "huginn_threshold_0p26_physical": ("huginn", "huginn_threshold_0p26_n640_t32", 0.26, range(384, 640)),
    "ouro_n_eq_b_control": ("ouro", "ouro_q0p2_n640_t32", 0.2, range(384, 416)),
    "huginn_n_eq_b_control": ("huginn", "huginn_threshold_0p12_n640_t32", 0.12, range(384, 416)),
}


def phase03_physical(reg, cells, width_rows, notes):
    pm = reg.csv(f"{P3}/physical_matrix.csv", role="phase03 physical matrix")
    cm = reg.csv(f"{P3}/control_matrix.csv", role="phase03 N=B controls")
    sm = reg.csv(f"{P3}/secondary_matrix.csv", role="phase03 secondary physical")
    reg.csv(f"{P3}/predictor_tournament.csv", role="phase03 predictor tournament (trace RIC)")
    mbd = reg.csv(f"{P3}/microbatch_max_depth_distribution.csv", role="phase03 microbatch max-depth distribution")
    reg.csv(f"{P3}/active_width_curves.csv", role="phase03 active width curves")
    reg.register(f"{P3}/RESULT_REVIEW.md", role="phase03 review")
    reg.register(f"{P3}/review_manifest.json", role="phase03 review manifest")
    pub_dist = defaultdict(dict)
    for r in mbd:
        if int(r["microbatches"]):
            pub_dist[(r["regime"], r["strategy"])][int(r["max_depth"])] = int(r["microbatches"])
    for dname, (model, regime, th, ids) in P3_DIRS.items():
        lab_rel = f"{P3}/{regime}/depth_labels.npy"
        reg.register(lab_rel, role="phase03 frozen B1 live depth labels [T, 640]")
        reg.register(f"{P3}/{regime}/feature_manifest.json", role="phase03 feature manifest")
        labels = np.load(reg.path(lab_rel)).astype(int)
        sub = labels[:, list(ids)]
        man = reg.json(f"{P3}/{dname}/run_manifest.json", role="phase03 physical run manifest")
        summ = reg.csv(f"{P3}/{dname}/summary.csv", role="phase03 physical per-repetition summary")
        ev_rows = reg.jsonl_gz(f"{P3}/{dname}/recurrence_events.jsonl.gz", role="phase03 recurrence-event receipts")
        b, t = int(man["batch_size"]), int(man["num_tokens"])
        n = len(ids)
        fifo, orc = fifo_oracle_metrics(sub, b)
        meas = defaultdict(list)
        for i, r in enumerate(summ, 1):
            if r["run_kind"] == "measured":
                meas[r["strategy"]].append((i, r))
        base = "STAGED_FIFO" if "STAGED_FIFO" in meas else "FIFO"
        ev = events_by_strategy(ev_rows)
        cell_base = f"p3phys_{dname}_N{n}_B{b}"
        width_rows += width_cost_rows(cell_base, model, ev, "PHYSICAL_FORCED_DEPTH", f"{P3}/{dname}/recurrence_events.jsonl.gz")

        def stat(s, col):
            return med_min_max([fnum(r[col]) * 1000 for _, r in meas[s]])

        bcalls = sorted({int(r["recurrent_invocations"]) for _, r in meas[base]})
        for s in meas:
            if s == base:
                continue
            pcalls = sorted({int(r["recurrent_invocations"]) for _, r in meas[s]})
            checks = {"baseline_ric": bcalls == [fifo["ric"]], "work": {int(r["total_token_loop_work"]) for _, r in meas[s]} == {fifo["work"]}}
            if s == "ORACLE":
                checks["oracle_ric"] = pcalls == [orc["ric"]]
                if (regime, "ORACLE") in pub_dist and n == 256:
                    # published distribution is summed over the measured repetitions
                    reps = len(meas["ORACLE"])
                    per_rep = {k: v / reps for k, v in sorted(pub_dist[(regime, "ORACLE")].items())}
                    checks["pub_maxdepth_dist"] = per_rep == {int(k): v for k, v in json.loads(orc["maxdepth_dist"]).items()}
            # cross-check published medians
            pub = [r for r in pm if r["regime"] == regime and r["strategy"] == s] if n == 256 else []
            core_b, core_p = stat(base, "recurrent_gpu_s"), stat(s, "recurrent_gpu_s")
            wall_b, wall_p = stat(base, "decode_wall_s"), stat(s, "decode_wall_s")
            sel = med_min_max([(fnum(r["predictor_wall_s"]) + fnum(r["sort_wall_s"])) * 1000 for _, r in meas[s]])[0]
            if pub:
                checks["pub_wall_median"] = abs(fnum(pub[0]["median_decode_wall_s"]) * 1000 - wall_p[0]) < 1e-6
            if n == 32:
                c = [r for r in cm if r["model"] == model]
                checks["pub_control"] = bool(c) and abs(fnum(c[0]["oracle_median_wall_s"]) * 1000 - wall_p[0]) < 1e-6
            if dname in ("ouro_q0p4_physical", "huginn_threshold_0p26_physical"):
                c = [r for r in sm if r["regime"] == regime]
                if s == "ORACLE":
                    checks["pub_secondary"] = bool(c) and abs(fnum(c[0]["oracle_median_wall_s"]) * 1000 - wall_p[0]) < 1e-6
            ok = all(checks.values())
            mean_w_b = float(np.mean(ev[base]["widths"])) if ev[base]["widths"] else NAN
            mean_w_p = float(np.mean(ev[s]["widths"])) if ev[s]["widths"] else NAN
            policy = "ORACLE_DEPTH_SORT" if s == "ORACLE" else f"PREDICTOR_{s}"
            cells.append(base_row(
                cell_id=f"{cell_base}_{s}", evidence_class="PHYSICAL_FORCED_DEPTH", phase="phase03",
                runtime_kind="survivor-compacted no-refill runtime, trace-forced depth (held-out cohort)",
                model=model, model_key="ouro14" if model == "ouro" else "huginn35",
                kv_policy=ev_rows[0].get("kv_backend", "") or ("last-exited-depth-indexed" if model == "ouro" else "shared-physical-layer"),
                rmax=4 if model == "ouro" else 16, dataset="alpaca", threshold=th,
                context=f"prompt cap {man.get('prompt_cap')} + {t} decode tokens (held-out ids {ids.start}..{ids.stop - 1})",
                N=n, B=b, num_tokens=t, baseline=base, policy=policy, prompt_sha256=man["prompt_sha256"],
                same_work_checksum=man["same_work_checksum"], trace_sha256=man["test_labels_sha256"],
                depth_stats_source="recomputed from depth_labels.npy", **depth_stats(sub),
                token_loop_work=fifo["work"], ric_baseline=fifo["ric"], ric_policy=pcalls[0] if len(pcalls) == 1 else NAN,
                slot_util_baseline=fifo["slot_util"],
                slot_util_policy=fifo["work"] / (b * pcalls[0]) if len(pcalls) == 1 and s != "ORACLE" else orc["slot_util"] if s == "ORACLE" else NAN,
                maxdepth_dist_baseline=fifo["maxdepth_dist"], maxdepth_dist_policy=orc["maxdepth_dist"] if s == "ORACLE" else "",
                mean_active_width_baseline=mean_w_b, mean_active_width_policy=mean_w_p,
                core_ms_baseline_median=core_b[0], core_ms_policy_median=core_p[0],
                core_ms_baseline_min=core_b[1], core_ms_baseline_max=core_b[2], core_ms_policy_min=core_p[1], core_ms_policy_max=core_p[2],
                wall_ms_baseline_median=wall_b[0], wall_ms_policy_median=wall_p[0],
                wall_ms_baseline_min=wall_b[1], wall_ms_baseline_max=wall_b[2], wall_ms_policy_min=wall_p[1], wall_ms_policy_max=wall_p[2],
                core_ms_per_call_baseline=core_b[0] / fifo["ric"], core_ms_per_call_policy=core_p[0] / pcalls[0],
                noncore_wall_fraction_baseline=1 - core_b[0] / wall_b[0], noncore_wall_fraction_policy=1 - core_p[0] / wall_p[0],
                selection_overhead_ms_policy=sel, selection_overhead_ratio=sel / wall_p[0],
                warmups=int(man["warmups"]), measured_repeats=len(meas[s]),
                reconciliation="OK" if ok else "MISMATCH:" + ",".join(k for k, v in checks.items() if not v),
                source_files=f"{P3}/{dname}/summary.csv;{P3}/{dname}/recurrence_events.jsonl.gz;{lab_rel}",
                source_rows=f"{P3}/{dname}/summary.csv:" + ",".join(str(i) for i, _ in meas[base] + meas[s]),
                notes=("trace-forced frozen B1 labels, teacher-forced token ids; wall includes predictor/sort/staging"
                       + ("; Huginn baseline is STAGED_FIFO (common Prelude staging path)" if base == "STAGED_FIFO" else ""))))
    notes.append("phase03 physical: FIFO/ORACLE and predictor rows recomputed from labels and per-rep summaries")


# ---------------------------------------------------------------- phase04

def phase04(reg, cells, notes):
    for model in ("ouro", "huginn"):
        rel = f"{P4V}/{model}_fresh_validation/per_token_validation.csv"
        rows = reg.csv(rel, role="phase04 fresh BF16 live per-token validation")
        bsv_rel = f"{P4V}/{model}_fresh_validation/batch_shape_validation.csv"
        bsv = reg.csv(bsv_rel, role="phase04 live batch-shape validation (RIC)")
        reg.csv(f"{P4V}/{model}_fresh_validation/margin_bin_summary.csv", role="phase04 margin bins")
        ref = {}
        live = defaultdict(dict)
        for r in rows:
            key = (int(r["token_idx"]), int(r["request_id"]))
            ref[key] = int(r["reference_depth"])
            live[(int(r["B"]), r["order"])][key] = int(r["live_depth"])
        tmax = max(k[0] for k in ref) + 1
        n = max(k[1] for k in ref) + 1
        refm = np.array([[ref[(t, i)] for i in range(n)] for t in range(tmax)])
        orders = {"fifo": list(range(n))}
        for seed in (17, 29):
            o = list(range(n))
            random.Random(seed).shuffle(o)
            orders[f"permutation_seed_{seed}"] = o
        rmax = 4 if model == "ouro" else 16
        for b in (2, 4, 8):
            fifo, orc = fifo_oracle_metrics(refm, b)
            cells.append(base_row(
                cell_id=f"p4trace_{model}_N{n}_B{b}", evidence_class="TRACE_ONLY", phase="phase04",
                runtime_kind="trace-sim on canonical B1 BF16 depths", model=model,
                model_key="ouro14" if model == "ouro" else "huginn35",
                kv_policy="last-exited-depth-indexed" if model == "ouro" else "shared-physical-layer", rmax=rmax,
                dataset="phase04 fresh short-prompt cohort", threshold=NAN, context="prompt max_length 64 + 8 positions",
                N=n, B=b, num_tokens=tmax, baseline="FIFO", policy="ORACLE_DEPTH_SORT",
                depth_stats_source="reference_depth (canonical B1)", **depth_stats(refm),
                token_loop_work=fifo["work"], ric_baseline=fifo["ric"], ric_policy=orc["ric"],
                slot_util_baseline=fifo["slot_util"], slot_util_policy=orc["slot_util"],
                maxdepth_dist_baseline=fifo["maxdepth_dist"], maxdepth_dist_policy=orc["maxdepth_dist"],
                reconciliation="OK" if fifo["ric"] == int([x for x in bsv if int(x["B"]) == b and x["order"] == "fifo"][0]["frozen_B1_depth_grouping_RIC"]) else "MISMATCH frozen fifo RIC",
                source_files=rel, source_rows=f"{rel}:all reference_depth",
                notes="small N32/T8 diagnostic cohort; B1 depths are canonical profiling depths, not batch-invariant truth"))
            for oname, order in orders.items():
                lm = np.array([[live[(b, oname)][(t, i)] for i in range(n)] for t in range(tmax)])
                groups = order_groups(order, b)
                lmet = schedule_metrics(lm, lambda d: groups)
                fmet = schedule_metrics(refm, lambda d: groups)
                pub = [(i, x) for i, x in enumerate(bsv, 1) if int(x["B"]) == b and x["order"] == oname][0]
                ok = (lmet["ric"] == int(pub[1]["actual_live_RIC"]) and fmet["ric"] == int(pub[1]["frozen_B1_depth_grouping_RIC"])
                      and lmet["work"] == int(pub[1]["actual_live_work"]))
                cells.append(base_row(
                    cell_id=f"p4live_{model}_N{n}_B{b}_{oname}", evidence_class="PHYSICAL_LIVE_EXIT", phase="phase04",
                    runtime_kind="live BF16 batched exit (debug timing not scored)", model=model,
                    model_key="ouro14" if model == "ouro" else "huginn35",
                    kv_policy="last-exited-depth-indexed" if model == "ouro" else "shared-physical-layer", rmax=rmax,
                    dataset="phase04 fresh short-prompt cohort", threshold=NAN, context="prompt max_length 64 + 8 positions",
                    N=n, B=b, num_tokens=tmax, baseline=f"FROZEN_B1_DEPTH_{oname.upper()}", policy=f"LIVE_{oname.upper()}",
                    depth_stats_source="live_depth", **depth_stats(lm), token_loop_work=lmet["work"],
                    ric_baseline=NAN, ric_policy=lmet["ric"], slot_util_policy=lmet["slot_util"],
                    maxdepth_dist_policy=lmet["maxdepth_dist"], live_ric_minus_frozen_ric=lmet["ric"] - fmet["ric"],
                    reconciliation="OK" if ok else "MISMATCH live RIC/work",
                    source_files=f"{rel};{bsv_rel}", source_rows=f"{bsv_rel}:{pub[0]}",
                    notes=("measured live-exit RIC count only; no live Oracle grouping was executed and no timing is "
                           "scored; ric_reduction intentionally blank")))
    for f in ("rank_stability_summary.csv", "depth_delta_summary.csv", "pairwise_gap_stability.csv", "position_depth_range.csv"):
        reg.csv(f"{P4R}/{f}", role="phase04 CPU rank-stability receipts")
    for f in (f"{P4R}/RESULT_REVIEW.md", f"{P4V}/RESULT_REVIEW.md", f"{P4V}/CAUSE_REVIEW.md"):
        reg.register(f, role="phase04 review doc")
    notes.append("phase04: TRACE_ONLY FIFO/Oracle on canonical B1 depths and PHYSICAL_LIVE_EXIT live-RIC counts (no timing)")


# ---------------------------------------------------------------- outputs

def write_csv(path: Path, rows: list[dict], columns=None):
    columns = columns or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and math.isnan(r[k]) else r.get(k, "")) for k in columns})


def validate_cells(cells: list[dict]):
    for r in cells:
        assert r["evidence_class"] in EVIDENCE
        physical = r["evidence_class"].startswith("PHYSICAL")
        if not physical:
            for c in ("core_ms_baseline_median", "core_ms_policy_median", "wall_ms_baseline_median", "wall_ms_policy_median"):
                assert not _finite(r[c]), f"{r['cell_id']}: non-physical row carries measured {c}"
        if r["evidence_class"] != "LATENCY_MODEL":
            assert not _finite(r["model_time_ms_policy"]), f"{r['cell_id']}: model time on non-model row"
        if r["evidence_class"] == "LATENCY_MODEL":
            assert "runtime" not in r["runtime_kind"] or "not integrated" in r["runtime_kind"]


def plots(cells: list[dict], outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ouro": "#2a78d6", "huginn": "#eb6834", "ouro14": "#2a78d6", "ouro26": "#1baf7a", "huginn35": "#eb6834"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6})
    tr = [r for r in cells if r["phase"] == "phase02" and r["evidence_class"] == "TRACE_ONLY" and r["num_tokens"] == 64]
    markers = {4: "o", 8: "s", 16: "^", 32: "D"}

    th_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, per-model threshold rank

    def th_color(model, th):
        ths = {"ouro": [0.2, 0.4, 0.5, 0.7], "huginn": [0.12, 0.26, 0.35, 0.5]}[model]
        return th_colors[ths.index(th)]

    # 1) RIC reduction vs N/B, faceted by model (KV policy is model-specific: confounded)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, ("ouro", "huginn")):
        rows = [r for r in tr if r["model"] == model]
        for r in rows:
            ax.scatter(r["N_over_B"] * (1 + 0.04 * (math.log2(r["B"]) - 3.5)), 100 * r["ric_reduction"],
                       marker=markers[r["B"]], s=22, alpha=0.8, color=th_color(model, r["threshold"]),
                       facecolor="none" if r["dataset"] == "sharegpt" else None)
        phys = [r for r in cells if r["model"] == model and r["evidence_class"] == "PHYSICAL_FORCED_DEPTH" and r["policy"] == "ORACLE_DEPTH_SORT"]
        ax.scatter([r["N_over_B"] for r in phys], [100 * r["ric_reduction"] for r in phys], marker="*", s=110, color="black")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{model}: TRACE_ONLY 64-token sweep")
        ax.set_xlabel("N / B (ready pool / microbatch size)")
        ths = sorted({r["threshold"] for r in rows})
        ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=th_color(model, t), label=f"threshold {t:g}") for t in ths]
                  + [plt.Line2D([], [], marker=markers[b], ls="", color="gray", label=f"B={b}") for b in markers]
                  + [plt.Line2D([], [], marker="*", ls="", color="black", ms=10, label="physical cells (RIC, 32 tok)")]
                  + ([plt.Line2D([], [], marker="o", ls="", mfc="none", color="gray", label="hollow = ShareGPT")] if model == "ouro" else []),
                  fontsize=6.5, ncol=4, loc="upper left")
        ax.set_ylim(-2, 46)
    axes[0].set_ylabel("FIFO→Oracle RIC reduction (%)")
    fig.tight_layout()
    fig.savefig(outdir / "ric_reduction_vs_N_over_B.png", dpi=140)
    plt.close(fig)

    # 2) RIC reduction vs B at N=256 (phase01 headroom incl. B64 and Ouro-2.6B; phase02 64-tok for Huginn)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, mk in zip(axes, ("ouro14", "ouro26", "huginn35")):
        ph = "phase02" if mk == "huginn35" else "phase01"
        model = "huginn" if mk == "huginn35" else "ouro"
        rows = [r for r in cells if r["model_key"] == mk and r["N"] == 256 and r["evidence_class"] == "TRACE_ONLY"
                and r["policy"] == "ORACLE_DEPTH_SORT" and r["phase"] == ph and r["num_tokens"] in (64, 128)]
        for (src, th) in sorted({(r["dataset"], r["threshold"]) for r in rows}):
            s = sorted([r for r in rows if (r["dataset"], r["threshold"]) == (src, th)], key=lambda r: r["B"])
            ax.plot([r["B"] for r in s], [100 * r["ric_reduction"] for r in s], marker="o" if src == "alpaca" else "s",
                    ls="-" if src == "alpaca" else "--", lw=1.5, ms=4, color=th_color(model, th), label=f"{src} q={th:g}")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{mk}: N=256 RIC reduction vs B ({ph}, TRACE_ONLY)", fontsize=8.5)
        ax.set_xlabel("B")
        ax.legend(fontsize=6, ncol=4, loc="upper center")
        ax.set_ylim(0, 46)
    axes[0].set_ylabel("FIFO→Oracle RIC reduction (%)")
    fig.tight_layout()
    fig.savefig(outdir / "ric_reduction_vs_B.png", dpi=140)
    plt.close(fig)

    # 3) RIC reduction vs depth dispersion (CV), N256/B32 trace cells + physical
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for mk in ("ouro14", "ouro26", "huginn35"):
        s = [r for r in cells if r["model_key"] == mk and r["evidence_class"] == "TRACE_ONLY" and r["N"] == 256
             and r["B"] == 32 and r["policy"] == "ORACLE_DEPTH_SORT" and r["num_tokens"] in (64, 128)]
        ax.scatter([r["depth_cv"] for r in s], [100 * r["ric_reduction"] for r in s], color=colors[mk], s=28, label=f"{mk} trace N256/B32")
    s = [r for r in cells if r["evidence_class"] == "PHYSICAL_FORCED_DEPTH" and r["policy"] == "ORACLE_DEPTH_SORT" and r["N"] == 256]
    ax.scatter([r["depth_cv"] for r in s], [100 * r["ric_reduction"] for r in s], marker="*", s=120,
               color=[colors[r["model"]] for r in s], edgecolor="black", linewidth=0.6, label="physical cells (RIC, 32 tok)")
    ax.set_xlabel("depth coefficient of variation (std/mean)")
    ax.set_ylabel("FIFO→Oracle RIC reduction (%)")
    ax.set_title("RIC reduction vs depth dispersion (stratified by model)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "ric_reduction_vs_depth_cv.png", dpi=140)
    plt.close(fig)

    # 4) physical core / wall reduction vs RIC reduction
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    phys = [r for r in cells if r["evidence_class"] == "PHYSICAL_FORCED_DEPTH"]
    for ax, metric, lab in zip(axes, ("core_reduction", "wall_reduction"), ("recurrent Core GPU time", "decode wall")):
        ax.plot([-0.05, 0.4], [-0.05, 0.4], color="#999999", lw=1, ls=":", label="y = x (time ∝ invocations)")
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        for model in ("ouro", "huginn"):
            for ph, mkr in (("phase02", "o"), ("phase03", "s")):
                s = [r for r in phys if r["model"] == model and r["phase"] == ph]
                orc = [r for r in s if r["policy"] == "ORACLE_DEPTH_SORT"]
                pred = [r for r in s if r["policy"] != "ORACLE_DEPTH_SORT"]
                ax.scatter([100 * r["ric_reduction"] for r in orc], [100 * r[metric] for r in orc], marker=mkr, s=36,
                           color=colors[model], label=f"{model} {ph} Oracle")
                if pred:
                    ax.scatter([100 * r["ric_reduction"] for r in pred], [100 * r[metric] for r in pred], marker=mkr, s=36,
                               facecolor="none", edgecolor=colors[model], label=f"{model} {ph} predictor")
        ax.set_xlim(-5, 40)
        ax.set_ylim(-25, 40)
        ax.set_xlabel("FIFO→policy RIC reduction (%)")
        ax.set_title(f"PHYSICAL_FORCED_DEPTH: {lab} reduction")
    axes[0].set_ylabel("measured reduction (T_fifo − T_policy)/T_fifo (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "physical_reduction_vs_ric_reduction.png", dpi=140)
    plt.close(fig)


def summarize(cells, surface, excluded, width_rows, outdir: Path) -> dict:
    phys = [r for r in cells if r["evidence_class"] == "PHYSICAL_FORCED_DEPTH"]
    counts = Counter(r["evidence_class"] for r in cells)
    return {"counts_by_evidence_class": dict(counts), "excluded": len(excluded),
            "reconciliation_failures": [r["cell_id"] for r in cells if not r["reconciliation"].startswith(("OK", "NOT_RECOMPUTABLE", "bct"))],
            "physical_oracle": [{k: r[k] for k in ("cell_id", "ric_reduction", "core_reduction", "wall_reduction",
                                                  "core_tracking_ratio", "wall_tracking_ratio", "core_ms_per_call_baseline",
                                                  "core_ms_per_call_policy", "mean_active_width_baseline", "mean_active_width_policy")}
                                for r in phys]}


def run(root: Path, outdir: Path) -> dict:
    reg = Registry(root)
    traces = Traces(reg)
    cells, excluded, width_rows, notes = [], [], [], []
    phase02_trace_sweeps(reg, traces, cells, notes)
    external_oracle_ric(reg, notes)
    surface = phase01(reg, traces, cells, excluded, notes)
    phase02_physical(reg, traces, cells, excluded, width_rows, notes)
    phase03_physical(reg, cells, width_rows, notes)
    phase04(reg, cells, notes)
    validate_cells(cells)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "e1_regime_cells.csv", cells, COLUMNS)
    write_csv(outdir / "e1_excluded_rows.csv", excluded)
    write_csv(outdir / "e1_calibration_surface.csv", surface)
    write_csv(outdir / "e1_core_call_cost_by_width.csv", width_rows)
    plots(cells, outdir)
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "phase05/scripts/e1_regime_reanalysis.py"], text=True).strip())
    except Exception:  # pragma: no cover
        sha, dirty = "UNKNOWN", True
    import matplotlib
    import pandas
    manifest = {
        "execution_kind": "saved-data-analysis (CPU only; no model, no CUDA)",
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "source_head": sha, "script_uncommitted_or_dirty": dirty,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "notes": notes, "missing_inputs": [f for f in reg.files.values() if not f.get("exists", True)],
        "inputs": sorted(reg.files.values(), key=lambda f: f["path"]),
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(outdir.glob("*")) if p.suffix in (".csv", ".png")},
        "summary": summarize(cells, surface, excluded, width_rows, outdir),
    }
    (outdir / "e1_input_manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    return manifest


def smoke() -> None:
    """Tiny synthetic input: exercises arithmetic and the CSV/plot writers only."""
    m = np.array([[1, 4, 1, 4, 2, 2, 3, 1], [4, 4, 1, 1, 2, 3, 1, 1]])  # [T=2, N=8]
    fifo, orc = fifo_oracle_metrics(m, 2)
    assert fifo["ric"] == 4 + 4 + 2 + 3 + 4 + 1 + 3 + 1 and orc["ric"] == (1 + 2 + 3 + 4) + (1 + 1 + 3 + 4)
    rows = [base_row(cell_id="smoke", evidence_class="TRACE_ONLY", phase="smoke", runtime_kind="trace-sim",
                     model="ouro", N=8, B=2, num_tokens=2, ric_baseline=fifo["ric"], ric_policy=orc["ric"],
                     token_loop_work=fifo["work"], slot_util_baseline=fifo["slot_util"], slot_util_policy=orc["slot_util"],
                     **depth_stats(m))]
    validate_cells(rows)
    with tempfile.TemporaryDirectory() as d:
        write_csv(Path(d) / "smoke.csv", rows, COLUMNS)
        print(open(Path(d) / "smoke.csv").read().splitlines()[1][:200])
    print(f"SMOKE OK fifo_ric={fifo['ric']} oracle_ric={orc['ric']} work={fifo['work']} "
          f"reduction={reduction(fifo['ric'], orc['ric']):.4f} speedup={speedup(fifo['ric'], orc['ric']):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="run on a tiny synthetic input only")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "phase05/results/20261019_e1")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    man = run(args.root, args.output_dir)
    print(json.dumps({"counts": man["summary"]["counts_by_evidence_class"], "excluded": man["summary"]["excluded"],
                      "reconciliation_failures": man["summary"]["reconciliation_failures"], "notes": man["notes"]}, indent=1))


if __name__ == "__main__":
    main()
