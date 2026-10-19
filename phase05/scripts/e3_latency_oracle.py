"""E3 — latency-aware Oracle grouping vs FIFO and depth-sorted (RIC) Oracle at equal frozen work.

All Oracles read FUTURE frozen per-token depths (phase02 B=1 trace inputs) and are NOT deployable.
Physical runs force those depths (exit_source="trace"): evidence class PHYSICAL_FORCED_DEPTH, a
scheduling-mechanism test, never live-exit inference performance.

  replay   (CPU)  TRACE_ONLY RIC/work for every trace x N x B x T, plus LATENCY_MODEL predicted Core
                  time from the E2 H100 table (interpolated only inside measured B/L support).
  physical (GPU)  predeclared cells on the phase02 survivor runtime; per-call receipts for calibration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TRACES = ROOT / "phase02/oracle_upper_bound_results_20260916/trace_inputs"
POLICIES = ("fifo", "depth_oracle", "latency_oracle", "random")
EXHAUSTIVE_LIMIT = 200_000
PHYSICAL_CELLS = ((256, 8, 32), (256, 32, 32), (64, 8, 32), (64, 32, 32), (32, 32, 32))  # predeclared; last = N=B control


# ---------------------------------------------------------------- traces and context

def load_trace(path: Path) -> dict:
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    n = 1 + max(r["request_id"] for r in rows)
    t = 1 + max(r["token_idx"] for r in rows)
    depth = np.zeros((t, n), dtype=np.int64)
    seen = set()
    for r in rows:
        key = (r["request_id"], r["token_idx"])
        if key in seen:
            raise ValueError(f"duplicate {key}")
        seen.add(key)
        depth[r["token_idx"], r["request_id"]] = r["depth"]
    if len(seen) != n * t or (depth < 1).any():
        raise ValueError("incomplete trace")
    first = rows[0]
    huginn = first.get("model") == "huginn"  # phase02 Huginn B1 traces use a shorter schema
    meta = ({"family": "huginn", "dataset": "alpaca" if "alpaca" in path.name else "sharegpt",
             "threshold": first["threshold"], "rmax": 16, "exit_rule": first["exit_source"]} if huginn else
            {"family": first["family"], "dataset": first["dataset"], "threshold": first["exit_threshold"],
             "rmax": first["rmax"], "exit_rule": first["exit_rule"]})
    if depth.max() > meta["rmax"]:
        raise ValueError("depth above Rmax")
    return {"depth": depth, **meta,
            "sha256": hashlib.sha256(gzip.open(path, "rb").read()).hexdigest(), "path": str(path.relative_to(ROOT))}


def model_key(trace: dict) -> str:
    return "huginn" if trace["family"] == "huginn" else "ouro14"


# ---------------------------------------------------------------- E2 cost model

class CoreModel:
    """Median E2 Core CUDA-event time; linear in active width w and in context L, inside support only."""

    def __init__(self, raw_csv: Path):
        cells = defaultdict(list)
        for r in csv.DictReader(raw_csv.open()):
            if r["status"] == "OK":
                cells[(int(r["L"]), int(r["B"]))].append(float(r["core_ms"]))
        self.table = {k: statistics.median(v) for k, v in cells.items()}
        self.Ls = sorted({L for L, _ in self.table})
        self.source = str(raw_csv)
        self._memo: dict = {}

    def _at_L(self, w: float, L: int) -> float:
        bs = sorted(b for (l, b) in self.table if l == L)
        if not bs or w < bs[0] or w > bs[-1]:
            raise ValueError(f"w={w} outside measured B support {bs} at L={L}")
        hi = next(b for b in bs if b >= w)
        lo = max(b for b in bs if b <= w)
        if hi == lo:
            return self.table[(L, lo)]
        f = (w - lo) / (hi - lo)
        return (1 - f) * self.table[(L, lo)] + f * self.table[(L, hi)]

    def __call__(self, w: int, L: int) -> float:
        key = (w, L)
        if key not in self._memo:
            self._memo[key] = self._predict(w, L)
        return self._memo[key]

    def _predict(self, w: int, L: int) -> float:
        if L < self.Ls[0] or L > self.Ls[-1]:
            raise ValueError(f"L={L} outside measured support {self.Ls}")
        hi = next(x for x in self.Ls if x >= L)
        lo = max(x for x in self.Ls if x <= L)
        if hi == lo:
            return self._at_L(w, lo)
        f = (L - lo) / (hi - lo)
        return (1 - f) * self._at_L(w, lo) + f * self._at_L(w, hi)


_GROUP_MEMO: dict = {}


def group_cost(depths, L, model) -> float:
    key = (model, L, tuple(sorted(depths)))  # object key keeps the model alive: no id reuse
    if key not in _GROUP_MEMO:
        _GROUP_MEMO[key] = _group_cost(depths, L, model)
    return _GROUP_MEMO[key]


def _group_cost(depths, L, model) -> float:
    ds = sorted(depths, reverse=True)
    total, w = 0.0, len(ds)
    for r in range(1, ds[0] + 1):
        while w and ds[w - 1] < r:
            w -= 1
        total += model(w, L)
    return total


# ---------------------------------------------------------------- groupings (per decode step)

def fifo(d, B, **_):
    return [list(range(i, i + B)) for i in range(0, len(d), B)]


def depth_oracle(d, B, **_):
    order = sorted(range(len(d)), key=lambda i: (d[i], i))
    return [order[i:i + B] for i in range(0, len(d), B)]


def random_groups(d, B, seed=0, **_):
    order = list(range(len(d)))
    random.Random(seed).shuffle(order)
    return [order[i:i + B] for i in range(0, len(d), B)]


def latency_oracle(d, B, L=None, model=None, **_):
    """Minimize sum of predicted Core time. Exhaustive over depth-count vectors for 2 groups when small,
    else first-improvement swap local search from the depth-sorted grouping (never worse than it)."""
    groups = depth_oracle(d, B)
    if model is None or len(groups) == 1:
        return groups, "trivial"
    cost = lambda g: group_cost([d[i] for i in g], L, model)
    if len(groups) == 2:
        counts = Counter(d)
        values = sorted(counts)
        space = 1
        for v in values:
            space *= counts[v] + 1
        if space <= EXHAUSTIVE_LIMIT:
            best, best_c = None, None
            for c in itertools.product(*[range(counts[v] + 1) for v in values]):
                if sum(c) != B:
                    continue
                g1 = [v for v, k in zip(values, c) for _ in range(k)]
                g2 = [v for v in values for _ in range(counts[v] - c[values.index(v)])]
                total = group_cost(g1, L, model) + group_cost(g2, L, model)
                if best_c is None or total < best_c - 1e-12:
                    best, best_c = c, total
            pools = {v: [i for i in sorted(range(len(d)), key=lambda i: i) if d[i] == v] for v in values}
            g1 = [pools[v].pop(0) for v, k in zip(values, best) for _ in range(k)]
            g2 = [i for v in values for i in pools[v]]
            out = [sorted(g1), sorted(g2)]
            if sum(map(cost, out)) > sum(map(cost, groups)) + 1e-9:
                raise AssertionError("exhaustive solution worse than depth oracle")
            return out, "exhaustive_2group"
    costs = [cost(g) for g in groups]
    improved, iters = True, 0
    while improved and iters < 2000:
        improved, iters = False, iters + 1
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                da = {d[i]: i for i in groups[a]}
                db = {d[i]: i for i in groups[b]}
                for va, ia in da.items():
                    for vb, ib in db.items():
                        if va == vb:
                            continue
                        ga = [x for x in groups[a] if x != ia] + [ib]
                        gb = [x for x in groups[b] if x != ib] + [ia]
                        ca, cb = cost(ga), cost(gb)
                        if ca + cb < costs[a] + costs[b] - 1e-9:
                            groups[a], groups[b], costs[a], costs[b] = ga, gb, ca, cb
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return groups, f"local_search_swaps({iters})"


def plan_groups(policy, depth_tb, B, Ls, model, seed=0):
    out, solvers = [], Counter()
    for t, d in enumerate(depth_tb):
        d = [int(x) for x in d]
        if policy == "latency_oracle":
            g, s = latency_oracle(d, B, L=Ls[t], model=model)
            solvers[s.split("(")[0]] += 1
        elif policy == "random":
            g = random_groups(d, B, seed=seed * 1000 + t)
        else:
            g = {"fifo": fifo, "depth_oracle": depth_oracle}[policy](d, B)
        if sorted(i for x in g for i in x) != list(range(len(d))) or any(len(x) != B for x in g):
            raise AssertionError("grouping must cover every request exactly once with full cohorts")
        out.append(g)
    return out, dict(solvers)


def metrics(groups_t, depth_tb, Ls, model):
    ric = work = slots = 0
    pred = 0.0
    for t, groups in enumerate(groups_t):
        for g in groups:
            ds = [int(depth_tb[t][i]) for i in g]
            ric += max(ds)
            work += sum(ds)
            slots += len(ds) * max(ds)
            if model is not None:
                pred += group_cost(ds, Ls[t], model)
    return {"RIC": ric, "work": work, "slot_utilization": work / slots, "pred_core_ms": pred if model else None}


# ---------------------------------------------------------------- prompts / context width

def prompt_width(family: str, dataset: str, N: int) -> tuple[int, str]:
    from phase02_runtime.benchmark import _load_prompts
    from transformers import AutoTokenizer
    from phase02_runtime.huginn_adapter import HUGINN_SNAPSHOT
    from phase02_runtime.ouro_adapter import OURO14_SNAPSHOT
    tok = AutoTokenizer.from_pretrained(HUGINN_SNAPSHOT if family == "huginn" else OURO14_SNAPSHOT, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    prompts = _load_prompts(dataset, N, 1, False)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    digest = hashlib.sha256()
    for rid, p in enumerate(prompts):
        digest.update(f"{rid}:".encode() + p.encode() + b"\n")
    return int(enc["input_ids"].shape[1]), digest.hexdigest()


# ---------------------------------------------------------------- replay

def cmd_replay(args):
    models = {"ouro14": CoreModel(args.e2_root / "ouro14/e2_stage_latency_raw.csv"),
              "huginn": CoreModel(args.e2_root / "huginn/e2_stage_latency_raw.csv")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, widths = [], {}
    traces = sorted(TRACES.glob("*.jsonl.gz"))[: 1 if args.smoke else None]
    for path in traces:
        tr = load_trace(path)
        mk = model_key(tr)
        for N in (32, 64, 128, 256):
            key = (tr["family"], tr["dataset"], N)
            if key not in widths:
                widths[key] = prompt_width(tr["family"], tr["dataset"], N)
            W, phash = widths[key]
            for T in (32, 64):
                if T > tr["depth"].shape[0]:
                    continue
                depth_tb = tr["depth"][:T, :N]
                Ls = [W + t + 1 for t in range(T)]
                for B in sorted({b for b in (4, 8, 16, 32, 64) if b <= N} | {N}):  # B=N is the control
                    base = None
                    for policy in POLICIES:
                        status, note = "OK", ""
                        try:
                            g, solvers = plan_groups(policy, depth_tb, B, Ls, models[mk])
                            m = metrics(g, depth_tb, Ls, models[mk])
                        except ValueError as e:  # outside E2 support: model prediction missing, RIC still valid
                            g, solvers = plan_groups(policy if policy != "latency_oracle" else "depth_oracle",
                                                     depth_tb, B, Ls, None)
                            m = metrics(g, depth_tb, Ls, None)
                            status, note = "NO_MODEL_SUPPORT", str(e)[:120]
                            if policy == "latency_oracle":
                                m = {k: None for k in m}
                        if policy == "fifo":
                            base = m
                        rows.append({"trace": tr["path"], "trace_sha256": tr["sha256"], "family": tr["family"],
                                     "core_model": mk, "dataset": tr["dataset"], "threshold": tr["threshold"],
                                     "exit_rule": tr["exit_rule"], "N": N, "B": B, "T": T, "N_over_B": N // B,
                                     "context_width_W": W, "prompt_sha256": phash, "policy": policy, **m,
                                     "RIC_reduction_vs_fifo": (1 - m["RIC"] / base["RIC"]) if m["RIC"] else None,
                                     "pred_reduction_vs_fifo": (1 - m["pred_core_ms"] / base["pred_core_ms"])
                                     if m["pred_core_ms"] and base["pred_core_ms"] else None,
                                     "solver": json.dumps(solvers), "status": status, "note": note,
                                     "evidence_RIC": "TRACE_ONLY", "evidence_pred": "LATENCY_MODEL"})
        print(json.dumps({"trace": path.name, "rows": len(rows)}), flush=True)
    with (args.output_dir / "e3_replay.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.output_dir / "e3_replay_manifest.json").write_text(json.dumps({
        "evidence": "TRACE_ONLY (RIC/work) and LATENCY_MODEL (pred_core_ms from E2 medians, interpolation only inside support)",
        "core_models": {k: v.source for k, v in models.items()},
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "context_rule": "phase02 runtime pads all rows to the cohort-wide prompt width W (max_length 64); step t attends W+t+1 keys",
        "B_equal_N_rows": "control: grouping cannot change RIC"}, indent=2) + "\n")


# ---------------------------------------------------------------- physical

def run_groups(model_name, adapter, input_ids, prompt_valid, depth_tb, groups_t, receipts, tag):
    import torch
    from phase02_runtime.common import run_interval, timed
    device = adapter.device
    adapter.backend.instrumentation = "benchmark"  # KV updates without per-update events; Core timed by events
    torch.cuda.synchronize()
    pf = time.perf_counter()
    current = adapter.prefill(input_ids, prompt_valid)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - pf
    tot = defaultdict(float)
    ric = work = 0
    digest = hashlib.sha256()
    start = time.perf_counter()
    for t, groups in enumerate(groups_t):
        adapter.begin_token()
        nxt = torch.empty_like(current)
        for g in groups:
            rows = torch.tensor(g, dtype=torch.long, device=device)
            adapter.backend.set_active(rows)
            tokens = current.index_select(0, rows)
            (emb, state), pre_ms, _ = timed(device, lambda: adapter.prelude(tokens))
            forced = torch.tensor([int(depth_tb[t][i]) for i in g], dtype=torch.long, device=device)
            res = run_interval(adapter, emb, state, rows, rows, exit_source="trace", forced_depths=forced,
                               debug_ids=False, instrumentation="profile")
            adapter.backend.set_active(rows)
            out, coda_ms, _ = timed(device, lambda: adapter.coda(res.final_state))
            nxt.index_copy_(0, rows, out)
            tot["prelude_ms"] += pre_ms
            tot["core_ms"] += res.core_gpu_ms
            tot["gate_ms"] += res.gate_gpu_ms
            tot["compaction_ms"] += res.compaction_gpu_ms
            tot["exit_kv_ms"] += res.kv_exit_gpu_ms
            tot["coda_ms"] += coda_ms
            ric += len(res.events)
            work += int(forced.sum())
            for i, dd in zip(g, forced.tolist()):
                digest.update(f"{i},{t},{dd}\n".encode())
            if receipts is not None:
                for e in res.events:
                    receipts.write(json.dumps({"tag": tag, "t": t, "w": e["active_batch_size"],
                                               "core_ms": e["recurrent_pass_gpu_ms_including_kv"]}) + "\n")
        adapter.commit_token()
        current = nxt
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    return {"decode_wall_s": wall, "prefill_s": prefill_s, **tot, "host_other_ms": wall * 1000 - sum(tot.values()),
            "RIC": ric, "work": work, "work_checksum": digest.hexdigest(),
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30}


def cmd_physical(args):
    import torch
    from phase02_runtime.benchmark import _load_prompts
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    sys.path.insert(0, str(ROOT / "phase05/scripts"))
    from e2_stage_microbench import gpu_identity
    torch.set_num_threads(4)
    tr = load_trace(TRACES / args.trace)
    mk = model_key(tr)
    model = CoreModel(args.e2_root / f"{mk}/e2_stage_latency_raw.csv")
    gpu = gpu_identity()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    cls = HuginnAdapter if tr["family"] == "huginn" else OuroAdapter
    base = cls.load(1, state_init="native-seeded") if tr["family"] == "huginn" else cls.load(1)
    cells = PHYSICAL_CELLS[:1] if args.smoke else PHYSICAL_CELLS
    fields = ["cell", "N", "B", "T", "policy", "rep", "status", "decode_wall_s", "prefill_s", "prelude_ms", "core_ms",
              "gate_ms", "compaction_ms", "exit_kv_ms", "coda_ms", "host_other_ms", "RIC", "work", "work_checksum",
              "pred_core_ms", "peak_alloc_gib", "solver", "error"]
    summary = (out / "e3_physical_runs.csv").open("w", newline="")
    writer = csv.DictWriter(summary, fieldnames=fields)
    writer.writeheader()
    receipts = gzip.open(out / "e3_call_receipts.jsonl.gz", "wt")
    manifest = {"experiment": "E3 physical", "evidence_class": "PHYSICAL_FORCED_DEPTH (frozen B=1 trace depths; NOT live-exit)",
                "trace": tr["path"], "trace_sha256": tr["sha256"], "exit_rule_of_trace": tr["exit_rule"],
                "model": mk, "gpu": gpu, "argv": sys.argv, "cells": cells, "policies": POLICIES,
                "warmups": args.warmups, "reps": args.reps,
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
                "core_model": model.source, "torch": torch.__version__, "cells_detail": {}}
    for (N, B, T) in cells:
        if args.smoke:
            N, B, T = 8, 4, 2
        cell = f"N{N}_B{B}_T{T}"
        prompts = _load_prompts(tr["dataset"], N, 1, False)
        enc = base.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        ids, valid = enc["input_ids"].cuda(), enc["attention_mask"].cuda().bool()
        W = int(ids.shape[1])
        Ls = [W + t + 1 for t in range(T)]
        depth_tb = tr["depth"][:T, :N]
        plans = {}
        for policy in POLICIES:
            g, solvers = plan_groups(policy, depth_tb, B, Ls, model, seed=7)
            plans[policy] = (g, solvers, metrics(g, depth_tb, Ls, model))
        manifest["cells_detail"][cell] = {"W": W, "prompt_count": N, "plans": {p: {"solver": s, **m}
                                          for p, (g, s, m) in plans.items()}}
        schedule = [("fifo", -1 - k) for k in range(args.warmups)]
        for rep in range(args.reps):
            order = list(POLICIES)
            random.Random(f"e3:{cell}:{rep}").shuffle(order)
            schedule += [(p, rep) for p in order]
        for policy, rep in schedule:
            g, solvers, m = plans[policy]
            row = {"cell": cell, "N": N, "B": B, "T": T, "policy": policy, "rep": rep, "status": "OK",
                   "pred_core_ms": m["pred_core_ms"], "solver": json.dumps(solvers), "error": ""}
            adapter = None
            try:
                torch.cuda.reset_peak_memory_stats()
                adapter = (HuginnAdapter(base.model, base.tokenizer, N, state_init="native-seeded") if tr["family"] == "huginn"
                           else OuroAdapter(base.model, base.tokenizer, N))
                res = run_groups(mk, adapter, ids, valid, depth_tb, g, receipts if rep >= 0 else None,
                                 f"{cell}:{policy}:{rep}")
                if res["work"] != m["work"] or res["RIC"] != m["RIC"]:
                    raise AssertionError("physical RIC/work differ from the frozen plan")
                row.update(res)
            except torch.cuda.OutOfMemoryError as e:
                row["status"], row["error"] = "OOM", str(e).splitlines()[0][:200]
            except Exception as e:  # retained
                row["status"], row["error"] = "ERROR", f"{type(e).__name__}: {e}"[:200]
            finally:
                adapter = None
                torch.cuda.empty_cache()
            writer.writerow({k: row.get(k) for k in fields})
            summary.flush()
            print(json.dumps({k: row.get(k) for k in ("cell", "policy", "rep", "status", "decode_wall_s", "core_ms",
                                                      "pred_core_ms", "RIC")}), flush=True)
    summary.close()
    receipts.close()
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--e2-root", required=True, type=Path)
    r.add_argument("--output-dir", required=True, type=Path)
    r.add_argument("--smoke", action="store_true")
    q = sub.add_parser("physical")
    q.add_argument("--trace", required=True, help="file name under phase02 trace_inputs")
    q.add_argument("--e2-root", required=True, type=Path)
    q.add_argument("--output-dir", required=True, type=Path)
    q.add_argument("--warmups", type=int, default=2)
    q.add_argument("--reps", type=int, default=5)
    q.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    {"replay": cmd_replay, "physical": cmd_physical}[a.cmd](a)


if __name__ == "__main__":
    main()
