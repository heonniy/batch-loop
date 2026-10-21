"""[Copied from Phase-06 phase06/cdb_ext/run.py @ local commit 3e2779e; `scores` passthrough added.]

Run one grouping policy on the upstream CDB no-refill engine and verify the frozen-work contract."""

from __future__ import annotations

import hashlib
import time
from collections import Counter

import torch

from cdb_ext import grouping


def plan_ric(depth_tb, width: int, policy: str, seed: int, scores=None) -> int:
    """Frozen-plan RIC (sum over cohorts of max depth) for the same selector, CPU only."""
    class _It:
        def __init__(self, i, d, cur=1):
            self.state = type("S", (), {"request_id": f"request-{i}", "_synthetic_exit_cursor": cur})()
            self.synthetic_exit_depth = d
    ric = 0
    for t, row in enumerate(depth_tb):
        items = [_It(i, int(d) - 1, t + 1) for i, d in enumerate(row)]
        grouping.STATE.scores = scores
        for c in grouping.select(policy, items, width, seed * 100003 + t):
            ric += max(it.synthetic_exit_depth for it in c) + 1
    return ric


def run_policy(engine, input_ids, depth_tb, policy: str, width: int, seed: int = 0, warmup: bool = False, scores=None) -> dict:
    """depth_tb: [T, N] 1-indexed frozen depths. Immediate-consumption replay: 0-based exit step = depth-1."""
    T, N = depth_tb.shape
    exit_depths = [[int(depth_tb[t, i]) - 1 for t in range(T)] for i in range(N)]
    grouping.STATE.reset(policy, width, seed, scores)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = engine.generate_batch(input_ids, max_new_tokens=[T + 1] * N, eos_token_id=None, warmup=warmup,
                                    exit_depths=exit_depths)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    st = grouping.STATE
    stats = engine.last_stats
    gpu = st.gpu_ms()
    waves = st.waves
    errors = []
    # tokens per request
    gen = [len(getattr(o, "generated_tokens", getattr(o, "output_ids", []))) for o in outputs]
    if any(g != T + 1 for g in gen):
        errors.append(f"generated token counts {Counter(gen)} != {T + 1}")
    # work and exit histogram
    work = int(depth_tb.sum())
    if stats.recurrent_steps != work:
        errors.append(f"recurrent_steps {stats.recurrent_steps} != frozen work {work}")
    hist = {k + 1: v for k, v in stats.exit_depth_histogram.items()}
    if hist != dict(Counter(depth_tb.ravel().tolist())):
        errors.append("exit-depth histogram differs from the frozen trace")
    # RIC and no-refill widths
    ric = len(st.widths)
    if ric != stats.recurrent_batches:
        errors.append("recurrent launch count mismatch")
    plan = plan_ric(depth_tb, width, policy, seed, scores)
    grouping.STATE.scores = scores
    if ric != plan:
        errors.append(f"physical RIC {ric} != frozen plan {plan}")
    by_cohort = {}
    for w, c, s, width_ in st.widths:
        by_cohort.setdefault((w, c), []).append(width_)
    for (w, c), ws in by_cohort.items():
        if ws != sorted(ws, reverse=True) or ws[0] != len(waves[w][c]) or len(waves[w][c]) > width:
            errors.append(f"wave {w} cohort {c}: widths {ws} violate shrink-only/no-refill")
            break
    if stats.scheduler_refills:
        errors.append("scheduler refilled")
    # causality: each request once per wave, cursor advancing by one
    if len(waves) != T:
        errors.append(f"{len(waves)} waves != {T} decode steps")
    last = {}
    digest = hashlib.sha256()
    for w, cohorts in enumerate(waves):
        seen = Counter(rid for c in cohorts for rid, _, _ in c)
        if any(v != 1 for v in seen.values()) or len(seen) != N:
            errors.append(f"wave {w}: request multiplicity/coverage error")
            break
        for c in cohorts:
            for rid, cursor, step in c:
                if last.get(rid, 0) + 1 != cursor:
                    errors.append(f"request {rid}: cursor {cursor} after {last.get(rid)}")
                last[rid] = cursor
                if step != int(depth_tb[cursor - 1, rid]) - 1:
                    errors.append(f"request {rid} token {cursor - 1}: replay step {step} != trace")
    for rid, cursor, step in sorted(x for cohorts in waves for c in cohorts for x in c):
        digest.update(f"{rid},{cursor},{step}\n".encode())
    tokens = hashlib.sha256(str([list(getattr(o, "generated_tokens", [])) for o in outputs]).encode()).hexdigest()
    return {"policy": policy, "width": width, "N": N, "T": T, "wall_s": t1 - t0,
            "decode_wall_s": t1 - st.first_wave_t if st.first_wave_t else None,
            "core_gpu_ms": gpu["recurrent"], "prelude_gpu_ms": gpu["prelude"], "coda_gpu_ms": gpu["coda"],
            "RIC": ric, "plan_RIC": plan, "work": work, "work_hash": digest.hexdigest(), "output_token_hash": tokens,
            "mean_core_width": sum(x[3] for x in st.widths) / max(1, ric), "waves": len(waves),
            "recurrent_graph_hits": stats.recurrent_graph_hits, "recurrent_graph_captures": stats.recurrent_graph_captures,
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, "n_errors": len(errors), "errors": errors[:10],
            "widths": st.widths, "membership": waves}
