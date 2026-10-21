"""E4 physical — FIFO vs pre-core predictor grouping vs depth oracle on the pinned optimized CDB no-refill runtime.

Same model, resident cap N, cohort width B, cache, workload and frozen token/depth trace for every policy
(Phase-07 canonical B=1 test labels replayed with immediate consumption; CUDA graphs, paged FA3, async ON).
`predicted` sorts each wave by the frozen E2 predictor's expected depth for the CURRENT token (computed offline
from B=1 features); its online cost is NOT inside this wall time and is measured separately by `--overhead`
(GPU: feature gather + MLP + expected depth + argsort per decode step at width N). Evidence class:
PHYSICAL_FORCED_DEPTH (+ separately measured overhead). No live-exit claim (E0 parity failed).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from cdb_ext.cdb_common import MODELS, build_cdb, env_info, gpu_identity, load_model  # noqa: E402

PROMPTS = HERE.parents[1] / "inputs/test_prompts.json"
CELLS = ((256, 8, 32), (256, 32, 32), (256, 256, 32))  # predeclared; B=N is the native single-cohort control
POLICIES = ("fifo", "predicted", "depth_oracle")
WARMUPS, REPS = 2, 5


def overhead(dim: int, hidden: int, K: int, N: int, reps: int = 200) -> dict:
    import torch
    W1 = torch.randn(dim, max(hidden, 1), device="cuda", dtype=torch.float32)
    W2 = torch.randn(max(hidden, 1), K, device="cuda", dtype=torch.float32)
    feats = torch.randn(N, dim, device="cuda", dtype=torch.float32)
    src = torch.randn(N, dim, device="cuda", dtype=torch.bfloat16)  # stand-in for gathered pre-core states
    r = torch.arange(1, K + 1, device="cuda", dtype=torch.float32)
    ts = []
    for i in range(reps + 10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.record()
        feats.copy_(src)  # gather/convert features
        if hidden:
            p = torch.softmax(torch.nn.functional.gelu(feats @ W1) @ W2, -1)
            score = p @ r
        else:  # previous-depth policy: the score already exists; only the sort is paid
            score = feats[:, 0]
        order = torch.argsort(score, stable=True)
        e.record()
        e.synchronize()
        wall = (time.perf_counter() - t0) * 1000
        if i >= 10:
            ts.append((s.elapsed_time(e), wall))
    return {"dim": dim, "hidden": hidden, "K": K, "N": N, "gpu_ms_median": statistics.median(t[0] for t in ts),
            "sync_wall_ms_median": statistics.median(t[1] for t in ts),
            "params": dim * hidden + hidden * K if hidden else dim * K}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("ouro14", "huginn"))
    ap.add_argument("--lane", required=True, help="feature lane, e.g. ouro14_t0.2_alpaca")
    ap.add_argument("--features-root", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path, help="E2 test_predictions.npz")
    ap.add_argument("--pred-key", required=True)
    ap.add_argument("--feature-dim", required=True, type=int)
    ap.add_argument("--hidden", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    import torch
    from cdb_ext import grouping
    from cdb_ext.run import run_policy
    a.out.mkdir(parents=True, exist_ok=False)
    gpu, info = gpu_identity(), env_info()
    meta = np.load(a.features_root / a.lane / "test/meta.npz")
    z = np.load(a.predictions)
    P = z[a.pred_key]
    K = P.shape[-1]
    ev = P @ np.arange(1, K + 1)
    dataset = a.lane.split("_")[-1]
    texts = json.loads(PROMPTS.read_text())[dataset]["prompts"]
    grouping.install()
    model, tok = load_model(a.model)
    rows = []
    for N, B, T in CELLS:
        ids = [tok(t, add_special_tokens=True)["input_ids"][:64] for t in texts[:N]]
        depth = meta["depth"][:T, :N].astype(np.int64)
        scores = [[float(ev[t, i]) for t in range(T)] for i in range(N)]
        eng, _ = build_cdb(model, max_num_seqs=N, rmax=MODELS[a.model]["rmax"], delayed=False,
                           num_blocks=int(N * -(-(64 + T + 1 + 16) // 16) * 1.25) + 8)
        grouping.instrument_runner(eng)
        pols = ("fifo",) if B == N else POLICIES
        sched = [("fifo", -1 - k) for k in range(WARMUPS)]
        for rep in range(REPS):
            o = list(pols)
            random.Random(f"e4:{N}:{B}:{rep}").shuffle(o)
            sched += [(p, rep) for p in o]
        warm = True
        for pol, rep in sched:
            r = run_policy(eng, ids, depth, pol, B, seed=rep, warmup=warm, scores=scores)
            warm = False
            row = {"N": N, "B": B, "T": T, "policy": pol if B != N else "native_single_cohort", "rep": rep,
                   **{k: r[k] for k in ("wall_s", "decode_wall_s", "core_gpu_ms", "prelude_gpu_ms", "coda_gpu_ms",
                                        "RIC", "plan_RIC", "work", "work_hash", "mean_core_width", "n_errors")},
                   "errors": json.dumps(r["errors"])}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("N", "B", "policy", "rep", "decode_wall_s", "RIC", "n_errors")}), flush=True)
        eng = None
        torch.cuda.empty_cache()
    with (a.out / "e4_physical_runs.csv").open("w", newline="") as f:  # write raw rows before anything else can fail
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    oh = [overhead(a.feature_dim, a.hidden, K, 256), overhead(1, 0, 1, 256)]  # MLP predictor; prev-depth sort only
    (a.out / "RUN_MANIFEST.json").write_text(json.dumps({
        "evidence_class": "PHYSICAL_FORCED_DEPTH; predictor overhead measured separately (PHYSICAL microbenchmark)",
        "model": a.model, "lane": a.lane, "pred_key": a.pred_key, "cells": CELLS, "policies": POLICIES,
        "warmups": WARMUPS, "reps": REPS, "gpu": gpu, "env": info, "argv": sys.argv, "predictor_overhead": oh,
        "labels": "Phase-07 canonical B=1 test labels (eager runtime, ByteDance weights == upstream checkpoint tensors)"},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
