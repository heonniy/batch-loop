#!/usr/bin/env python3
"""v7 query-conditioned Context-KV sensitivity sweep (QUERY_CONDITIONED_RECURRENT_KV_PLAN.md).

  --prepare   CPU: freeze units (frozen v5 Controlled-Retrieval / HotpotQA-Distractor prompts, optional v6
              W1/W2 Bref prompts) and jobs; thresholds for the live-exit arm must be given here (clean-only
              calibration from an earlier run's REF receipts) or the LIVE jobs are not created.
  --execute   GPU workers (one model per worker, B=1); jobs with unmet dependencies wait.
  --worker    one worker process.
  --analyze   CPU: completeness (fail closed) + tables + findings (query_conditioned_analysis.py).

Job kinds (per unit = model x dataset x prompt UID):
  REF   untouched full-depth prefill, FULL fixed-depth frozen gold continuation, v7 validation gate, clean
        census at the prompt anchors and every generated query, source trajectory features, clean exit stats.
  CEN   chunk of Context sources: persistent accumulated eviction over the continuation (PRIMARY future
        outcome Y), clean-history probes at every t, query-conditioned probes at the anchors (all-loop,
        rec-only/plain-only for Huginn, preregistered one-loop masks).
  LIVE  label-blind source subset: FULL-LIVE x2 / FULL-REPLAY / FULL-FIXED and EVICT-LIVE / EVICT-REPLAY /
        EVICT-FIXED on the same frozen gold continuation + fixed-depth greedy answer exact match.
  V6F   v6 W1/W2 Bref prompts: anchor probes + clean census for every eligible source; the 128-step
        persistent-eviction outcome comes from the frozen v6 census (not recomputed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

PROTOCOL = "query-conditioned-v7-pilot-v1"
V5_RUN = ROOT / "results" / "20261030T164908Z_loopzip_failure_pilot20"
V6_RUN = ROOT / "results" / "20261031T070815Z_loopzip_v6_trajectory_pilot"
MODELS_V7 = ("ouro-2.6b", "huginn-0125")
DATASETS = ("controlled_retrieval", "hotpotqa_distractor")


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def git_dirty() -> bool:
    return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "kv_policy/kvp",
                                "kv_policy/scripts/query_conditioned_sweep.py"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip())


def _json(path: Path, value):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def live_subset(content_positions, anchor_attention, seed: int, n_hash: int, n_top: int) -> dict:
    """Label-blind LIVE subset: positional-stratified hash sample + top clean anchor attention.
    Inputs: Context positions, clean question-anchor attention (mean over loops). No outcome, evidence
    label or gold answer enters."""
    from kvp.role_main_candidates import choose_main_candidates
    pos = sorted(int(p) for p in content_positions)
    top = sorted(pos, key=lambda p: (-float(anchor_attention[p]), p))[:n_top]
    rest = [p for p in pos if p not in set(top)]
    hashed = [c["positions"][0] for c in choose_main_candidates(rest, seed, n_candidates=min(n_hash, len(rest)),
                                                                   window_tokens=1, position_bins=4)]
    return {"sources": sorted(set(top) | set(hashed)), "top_attention": top, "positional_hash": hashed}


# --------------------------------------------------------------------------------------------- prepare
def prepare(run: Path, cfg: dict):
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run directory")
    if git_dirty():
        raise SystemExit("Commit kvp/ and this script before freezing a run")
    run.mkdir(parents=True)
    v5 = json.loads((V5_RUN / "jobs.json").read_text())
    jobs, units = [], []
    for model in cfg["models"]:
        for ds in cfg["datasets"]:
            rows = sorted((j for j in v5 if j["model"] == model and j["dataset"] == ds), key=lambda j: j["uid"])
            rows = rows[cfg["skip_examples"]:cfg["skip_examples"] + cfg["n_examples"]]
            if len(rows) != cfg["n_examples"]:
                raise RuntimeError(f"Only {len(rows)} frozen v5 prompts for {model}/{ds}")
            for j in rows:
                unit = f"{model}__{ds}__{j['uid']}"
                content = j["content_positions"]
                base = {"model": model, "dataset": ds, "uid": j["uid"], "unit": unit, "v5_key": j["key"],
                        "prompt_len": len(j["prompt_ids"]), "n_targets": len(j["target_ids"])}
                units.append(base | {"n_content": len(content)})
                refk = f"REF__{unit}"
                jobs.append(base | {"kind": "REF", "key": refk})
                cs = cfg["chunk_size"][model]
                for k in range(0, len(content), cs):
                    jobs.append(base | {"kind": "CEN", "key": f"CEN__{unit}__{k // cs:03d}", "depends": refk,
                                        "sources": content[k:k + cs]})
                if cfg.get("live"):
                    jobs.append(base | {"kind": "LIVE", "key": f"LIVE__{unit}", "depends": refk})
    if cfg.get("v6_units"):
        v6 = json.loads((V6_RUN / "jobs.json").read_text())
        for ref in (j for j in v6 if j["kind"] == "Bref" and j["model"] in cfg["models"]):
            unit = f"{ref['model']}__v6{ref['workload']}__{ref['item_id']}"
            base = {"model": ref["model"], "dataset": f"v6_{ref['workload']}", "uid": ref["uid"], "unit": unit,
                    "v6_key": ref["key"], "item_id": ref["item_id"], "prompt_len": ref["prompt_len"]}
            meta = json.loads((V6_RUN / "raw" / f"{ref['key']}.json").read_text())
            elig = meta["eligible"]
            units.append(base | {"n_content": len(elig)})
            refk = f"V6R__{unit}"
            jobs.append(base | {"kind": "V6R", "key": refk})
            cs = cfg["chunk_size"][ref["model"]] * 2
            for k in range(0, len(elig), cs):
                jobs.append(base | {"kind": "V6F", "key": f"V6F__{unit}__{k // cs:03d}", "depends": refk,
                                    "sources": elig[k:k + cs]})
    if len({j["key"] for j in jobs}) != len(jobs):
        raise RuntimeError("Duplicate job keys")
    _json(run / "jobs.json", jobs)
    _json(run / "units.json", units)
    _json(run / "run_manifest.json", {"protocol": PROTOCOL, "code_sha": git_sha(), "config": cfg,
                                      "v5_source_run": str(V5_RUN), "v6_source_run": str(V6_RUN),
                                      "notes": [
                                          "Prompts/targets are the frozen v5 pilot20 Controlled Retrieval + HotpotQA "
                                          "evidence_excerpt prompts (seed 20261031); gold 'Answer: <x>\\n' continuation",
                                          "CEN sources are exhaustive over aligned Context tokens; no label enters selection",
                                          "LIVE subset = positional hash + top clean question-anchor attention (label-blind)",
                                          "Live thresholds are frozen from clean-only REF receipts of an earlier run",
                                          "Full-sized reference KV + read masks; no HBM claim",
                                      ]})
    from collections import Counter
    return {"jobs": len(jobs), "units": len(units), "kinds": dict(Counter(j["kind"] for j in jobs))}


# --------------------------------------------------------------------------------------------- GPU side
def _load(model_key):
    from kvp.models import eos_ids, load, make_runner
    spec, tok, model = load(model_key)
    return tok, make_runner(spec, tok, model).bb, eos_ids(spec, tok, model)


def _v5_job(job):
    v5 = {j["key"]: j for j in json.loads((V5_RUN / "jobs.json").read_text())}
    return v5[job["v5_key"]]


def _make_prompt(bb, tok, job):
    if "v5_key" in job:
        from loopzip_failure_screen import make_prompt
        j = _v5_job(job)
        return make_prompt(j), list(j["target_ids"]), j
    from kvp.loopzip_trajectory_data import build_trajectory_prompt
    item = next(x for x in json.loads((V6_RUN / "manifest.json").read_text()) if x["item_id"] == job["item_id"])
    meta = json.loads((V6_RUN / "raw" / f"{job['v6_key']}.json").read_text())
    p = build_trajectory_prompt(tok, bb.family, item["context"], item["instruction"], item["question"])
    return p, list(meta["ref"]["ids"]), meta


def live_spec(bb, cfg):
    from kvp.depth_eviction_probe import DepthProbeSpec
    if not cfg.get("live"):
        return None
    if bb.family == "ouro":
        return DepthProbeSpec("ouro", bb.R, float(cfg["live"]["ouro_q"]), min_depth=int(cfg["live"]["min_depth"]))
    return DepthProbeSpec("huginn", bb.R, float(cfg["live"]["huginn_cosine"]), min_depth=int(cfg["live"]["min_depth"]),
                          cosine_dtype=cfg["live"].get("cosine_dtype", "float32"))


class UnitCache:
    """One runner per unit, reused across consecutive jobs of the same unit in a worker."""

    def __init__(self):
        self.unit = None
        self.runner = None
        self.prompt = None
        self.targets = None

    def get(self, bb, tok, eos, job, cfg, horizon):
        from kvp.query_conditioned_probe import QueryConditionedRunner
        if self.unit == job["unit"]:
            return self.runner, self.prompt, self.targets
        self.runner = self.prompt = None
        p, targets, _ = _make_prompt(bb, tok, job)
        t0 = time.time()
        r = QueryConditionedRunner(bb, p, job["uid"], horizon=max(horizon, len(targets) + 1))
        r.t_prefill = time.time() - t0
        t0 = time.time()
        r.ref = r.reference(eos, frozen_ids=targets)
        r.t_reference = time.time() - t0
        self.unit, self.runner, self.prompt, self.targets = job["unit"], r, p, targets
        return r, p, targets


def _arr(rows, key):
    import numpy as np
    return np.stack([row[key].numpy() for row in rows]).astype("float32")


def evaluate(bb, tok, eos, job, cfg, run: Path, cache: UnitCache):
    import numpy as np
    import torch
    from kvp.query_conditioned_probe import PREREG_LOOPS
    raw = run / "raw"
    meta = {k: job[k] for k in ("key", "kind", "model", "dataset", "uid", "unit")} | {"sha": git_sha()}
    r, p, targets = cache.get(bb, tok, eos, job, cfg, cfg["horizon"])
    R, P, n = bb.R, r.P, len(r.ref_ids)
    dig = float(r.ref_H.norm(dim=-1).sum())
    if job["kind"] in ("REF", "V6R"):
        if job["kind"] == "REF" and r.ref["frozen_argmax_agreement"] is None:
            raise RuntimeError("Frozen gold continuation expected")
        if job["kind"] == "V6R":
            _, _, v6meta = _make_prompt(bb, tok, job)
            if r.ref["frozen_argmax_agreement"] < 1.0 or abs(dig - v6meta["ref_state_digest"]) > 1e-3 * max(1.0, dig):
                raise RuntimeError("Frozen v6 reference not reproduced")
        content = list(p.content_positions)
        va, vb = content[len(content) // 3], content[(2 * len(content)) // 3]
        t0 = time.time()
        val = r.validate_v7(va, vb, live_spec=live_spec(bb, cfg))
        t_val = time.time() - t0
        if not val["pass"]:
            raise RuntimeError(f"v7 validation failed: {val}")
        arrs = {}
        for name, col in r.anchors.items():
            for k, v in r.census(col).items():
                arrs[f"anchor_{name}_{k}"] = v.numpy().astype("float32")
        gen = [r.census(P + t) for t in range(n)]
        for k in ("attn", "attn_max", "delta", "delta_max", "vnorm"):
            arrs[f"gen_{k}"] = np.stack([g[k][:, :P].numpy() for g in gen]).astype("float32")
        arrs["gen_plain_attn"] = np.stack([g["plain_attn"][:, :P].numpy() for g in gen]).astype("float32")
        arrs["gen_plain_delta"] = np.stack([g["plain_delta"][:, :P].numpy() for g in gen]).astype("float32")
        sf = r.source_features()
        arrs |= {f"src_{k}": v.numpy().astype("float32") for k, v in sf.items()}
        for k in ("U", "F", "C", "h_norm"):
            arrs[f"prompt_{k}"] = np.stack([m[k].numpy() for m in r.prompt_metrics]).astype("float32")
            arrs[f"gen_traj_{k}"] = np.stack([m[k].numpy() for m in r.ref_metrics]).astype("float32")
        keys = sorted(r.prompt_exit[0])
        arrs["prompt_exit"] = np.array([[x[k] for k in keys] for x in r.prompt_exit], dtype="float32")
        arrs["gen_exit"] = np.array([[x[k] for k in keys] for x in r.ref_exit], dtype="float32")
        lp = r.ref_lp
        arrs["gen_entropy"] = (-(lp.exp() * lp).sum(-1)).cpu().numpy().astype("float32")
        arrs["gen_ref_nll"] = np.array([-float(r.last_prompt_lp[r.ref_ids[0]])] + [
            -float(lp[i][r.ref_ids[i + 1]]) for i in range(n - 1)], dtype="float32")
        np.savez_compressed(raw / f"{job['key']}.npz", **arrs)
        _json(raw / f"{job['key']}.json", meta | {
            "prompt_ids": list(p.token_ids), "roles": list(p.roles), "content_positions": content,
            "relevant_positions": list(p.relevant_positions), "question_positions": list(p.question_positions),
            "instruction_positions": list(p.instruction_positions), "anchors": r.anchors, "ref": r.ref,
            "target_ids": targets, "exit_keys": keys, "validation": val, "ref_state_digest": dig,
            "timing": {"prefill_s": r.t_prefill, "reference_s": r.t_reference, "validation_s": t_val},
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "prereg_loops": PREREG_LOOPS[bb.family]})
        return
    ref = json.loads((raw / f"{job['depends']}.json").read_text())
    if abs(dig - ref["ref_state_digest"]) > 1e-3 * max(1.0, dig):
        raise RuntimeError(f"Reference digest drift {dig} vs {ref['ref_state_digest']}")
    if job["kind"] in ("CEN", "V6F"):
        loops = PREREG_LOOPS[bb.family]
        probes = [("all", None, True)] + ([("rec", None, False), ("plain", (), True)] if bb.n_plain else []) \
            + [(f"loop{s}", (s,), False) for s in loops]
        N = len(job["sources"])
        # Third anchor (ORACLE, per source): the generated query t* with the largest clean attention to the source
        # (mean over loops) — the query that actually consumes the KV. Its column differs per source.
        anchors = list(r.anchors.items()) + [("future_maxattn", None)]
        fut_att = np.stack([r.local[P + t]["attn"][:, :P].cpu().numpy().mean(0) for t in range(n)])  # (n, P)
        fut_t = {int(s): int(fut_att[:, int(s)].argmax()) for s in job["sources"]}
        f32 = lambda *s: np.full(s, np.nan, dtype="float32")  # noqa: E731
        out = {}
        if job["kind"] == "CEN":
            out |= {f"acc_{k}": f32(N, n, R + 1) for k in ("E", "U_ev", "dircos")}
            out |= {f"acc_{k}": f32(N, n) for k in ("kl", "top1_changed", "nll_shift", "entropy_shift")}
            out |= {f"clean_{k}": f32(N, n, R + 1) for k in ("E", "U_ev", "dircos")}
            out |= {f"clean_{k}": f32(N, n) for k in ("kl", "top1_changed", "nll_shift", "entropy_shift")}
        A, Q = len(anchors), len(probes)
        out |= {f"q_{k}": f32(N, A, Q, R + 1) for k in ("E", "U_ev", "dircos", "delta_norm")}
        out |= {f"q_{k}": f32(N, A, Q) for k in ("kl", "top1_changed", "nll_shift", "entropy_shift")}
        exit_keys = ref["exit_keys"]
        out["q_exit"] = f32(N, A, Q, len(exit_keys), R)
        times = []
        for si, s in enumerate(job["sources"]):
            t0 = time.time()
            if job["kind"] == "CEN":
                acc = r.accumulated((s,))
                r._restore()
                for k in ("E", "U_ev", "dircos"):
                    out[f"acc_{k}"][si] = acc[k].numpy()
                for ti, tk in enumerate(acc["token"]):
                    for k in ("kl", "top1_changed", "nll_shift", "entropy_shift"):
                        out[f"acc_{k}"][si, ti] = np.nan if tk[k] is None else float(tk[k])
                cl = r.clean_probes((s,), range(1, n + 1))
                for t in range(1, n + 1):
                    for k in ("E", "U_ev", "dircos"):
                        out[f"clean_{k}"][si, t - 1] = cl[t][k].numpy()
                    for k in ("kl", "top1_changed", "nll_shift", "entropy_shift"):
                        v = cl[t]["token"][k]
                        out[f"clean_{k}"][si, t - 1] = np.nan if v is None else float(v)
            for ai, (aname, col) in enumerate(anchors):
                if col is None:
                    col = P + fut_t[int(s)]
                if s >= col:
                    continue
                for qi, (pname, lp, plain) in enumerate(probes):
                    o = r.query_probe(col, (s,), loops=lp, plain=plain)
                    for k in ("E", "U_ev", "dircos", "delta_norm"):
                        out[f"q_{k}"][si, ai, qi] = o[k].numpy()
                    for k in ("kl", "top1_changed", "nll_shift", "entropy_shift"):
                        v = o["token"][k]
                        out[f"q_{k}"][si, ai, qi] = np.nan if v is None else float(v)
                    out["q_exit"][si, ai, qi] = np.array([o["exit"][k] for k in exit_keys], dtype="float32")
            times.append(time.time() - t0)
        if job["kind"] == "CEN" and np.isnan(out["acc_E"][..., 1:]).any():
            raise RuntimeError("NaN in accumulated state error")
        out["future_t"] = np.array([fut_t[int(s)] for s in job["sources"]], dtype="int32")
        np.savez_compressed(raw / f"{job['key']}.npz", sources=np.array(job["sources"]), **out)
        _json(raw / f"{job['key']}.json", meta | {"sources": job["sources"], "n_tokens": n,
                                                  "anchors": [a for a, _ in anchors], "probes": [q[0] for q in probes],
                                                  "exit_keys": exit_keys, "per_source_s": times,
                                                  "reference_digest": dig,
                                                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        return
    if job["kind"] == "LIVE":
        from kvp.role_generation import score_free_answer
        spec = live_spec(bb, cfg)
        if spec is None:
            raise RuntimeError("LIVE job without frozen thresholds")
        qa = r.anchors["question_last"]
        att = r.census(qa)["attn"].mean(0)
        sub = live_subset(p.content_positions, att, int(_sha(job["unit"])[:8], 16),
                          cfg["live"]["n_hash"], cfg["live"]["n_top"])
        gold = _v5_job(job)["gold_answer"]
        t0 = time.time()
        full_live = r.decode((), spec, dynamic=True)
        full_live2 = r.decode((), spec, dynamic=True)
        full_replay = r.decode((), spec, forced_depths=full_live["depths"])
        full_fixed = r.decode((), r.spec)
        g_full = r.greedy((), cfg["live"]["greedy_tokens"], eos)
        arms = {"full_live": full_live, "full_live_repeat": full_live2, "full_replay": full_replay,
                "full_fixed": full_fixed,
                "full_greedy": score_free_answer(tok, gold, {"token_ids": g_full, "decoded_tokens": len(g_full)})
                | {"token_ids": g_full}}
        per = []
        for s in sub["sources"]:
            el = r.decode((s,), spec, dynamic=True)
            er = r.decode((s,), spec, forced_depths=full_live["depths"])
            ef = r.decode((s,), r.spec)
            g = r.greedy((s,), cfg["live"]["greedy_tokens"], eos)
            per.append({"source": int(s), "evict_live": el, "evict_replay": er, "evict_fixed": ef,
                        "evict_greedy": score_free_answer(tok, gold, {"token_ids": g, "decoded_tokens": len(g)})
                        | {"token_ids": g},
                        "in_top_attention": s in sub["top_attention"]})
        _json(raw / f"{job['key']}.json", meta | {"subset": sub, "spec": {"family": spec.family,
                                                                             "threshold": spec.threshold,
                                                                             "min_depth": spec.min_depth,
                                                                             "cosine_dtype": spec.cosine_dtype},
                                                  "arms": arms, "per_source": per, "gold_answer": gold,
                                                  "elapsed_s": time.time() - t0, "reference_digest": dig,
                                                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        return
    raise RuntimeError(f"Unknown job kind {job['kind']}")


def worker(run: Path, idx: int, model_filter: str | None):
    import torch
    m = json.loads((run / "run_manifest.json").read_text())
    if m["code_sha"] != git_sha():
        raise RuntimeError("Code SHA changed; use a NEW run directory")
    jobs = json.loads((run / "jobs.json").read_text())
    if model_filter:
        jobs = [j for j in jobs if j["model"] == model_filter]
    raw, claims, errors = (run / d for d in ("raw", "claims", "errors"))
    for d in (raw, claims, errors):
        d.mkdir(exist_ok=True)
    cur = tok = bb = eos = None
    cache = UnitCache()
    order = {"REF": 0, "V6R": 0, "CEN": 1, "LIVE": 2, "V6F": 3}
    while True:
        pend = [j for j in jobs if not (raw / f"{j['key']}.json").exists()
                and not (claims / f"{j['key']}.lock").exists()]
        if not pend:
            return
        ready = [j for j in pend if not j.get("depends") or (raw / f"{j['depends']}.json").exists()]
        if not ready:
            if any((errors / f"{j['depends']}.json").exists() for j in pend if j.get("depends")):
                raise RuntimeError("Dependency failed")
            time.sleep(15)
            continue
        # Prefer the cached unit (no re-prefill), then REF jobs, then chunks of the same model.
        ready.sort(key=lambda j: (j["model"] != cur, j["unit"] != cache.unit, order[j["kind"]], j["unit"], j["key"]))
        job = ready[0]
        try:
            fd = os.open(claims / f"{job['key']}.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"worker={idx} pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        try:
            if cur != job["model"]:
                tok = bb = None
                cache = UnitCache()
                torch.cuda.empty_cache()
                tok, bb, eos = _load(job["model"])
                cur = job["model"]
            torch.cuda.reset_peak_memory_stats()
            evaluate(bb, tok, eos, job, m["config"], run, cache)
            print(f"done {job['key']}", flush=True)
        except Exception:
            _json(errors / f"{job['key']}.json", {"key": job["key"], "traceback": traceback.format_exc()})
            cache = UnitCache()
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    for n in ("prepare", "execute", "analyze"):
        g.add_argument("--" + n, action="store_true")
    g.add_argument("--worker", type=int)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS_V7))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--model-filter")
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--skip-examples", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--chunk-ouro", type=int, default=48)
    ap.add_argument("--chunk-huginn", type=int, default=32)
    ap.add_argument("--v6-units", action="store_true")
    ap.add_argument("--live-ouro-q", type=float)
    ap.add_argument("--live-huginn-cosine", type=float)
    ap.add_argument("--live-min-depth", type=int, default=2)
    ap.add_argument("--live-n-hash", type=int, default=16)
    ap.add_argument("--live-n-top", type=int, default=8)
    ap.add_argument("--greedy-tokens", type=int, default=24)
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--gpu-models", default="")
    a = ap.parse_args()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    run = Path(a.run_dir).resolve()
    if a.prepare:
        live = None
        if a.live_ouro_q is not None or a.live_huginn_cosine is not None:
            if a.live_ouro_q is None or a.live_huginn_cosine is None:
                ap.error("Give both --live-ouro-q and --live-huginn-cosine")
            live = {"ouro_q": a.live_ouro_q, "huginn_cosine": a.live_huginn_cosine, "min_depth": a.live_min_depth,
                    "cosine_dtype": "float32", "n_hash": a.live_n_hash, "n_top": a.live_n_top,
                    "greedy_tokens": a.greedy_tokens}
        cfg = {"models": sp(a.models), "datasets": sp(a.datasets), "n_examples": a.n_examples,
               "skip_examples": a.skip_examples, "horizon": a.horizon,
               "chunk_size": {"ouro-2.6b": a.chunk_ouro, "huginn-0125": a.chunk_huginn},
               "v6_units": a.v6_units, "live": live}
        print(json.dumps(prepare(run, cfg), indent=2))
    elif a.worker is not None:
        worker(run, a.worker, a.model_filter)
    elif a.analyze:
        from query_conditioned_analysis import analyze
        print(json.dumps(analyze(run), indent=2, default=str))
    else:
        ids = sp(a.gpu_ids)
        filt = sp(a.gpu_models) if a.gpu_models else [""] * len(ids)
        (run / "logs").mkdir(exist_ok=True)
        ps = []
        for i, (gpu, mf) in enumerate(zip(ids, filt)):
            h = (run / "logs" / f"worker{i}.txt").open("a")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--run-dir", str(run)]
            if mf:
                cmd += ["--model-filter", mf]
            ps.append((subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                        stdout=h, stderr=subprocess.STDOUT), h))
        codes = [p.wait() for p, _ in ps]
        print(json.dumps({"worker_exit_codes": codes}))
        if any(codes):
            sys.exit(1)


if __name__ == "__main__":
    main()
