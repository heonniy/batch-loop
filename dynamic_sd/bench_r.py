"""Microbenchmark: wall time of ONE recurrent step (R call) vs number of rows, using the
upstream single-sequence blocks (eager, as in exp3).  Two layouts:
  same  : n consecutive positions at the SAME depth step (DtV verify-like; grouped path)
  wave  : n consecutive positions at DISTINCT steps (WFD wave-like; per-token varlen path)
Values are timing-only (KV contents are irrelevant for latency)."""
import argparse
import json
import os
import sys
import time

WF_ROOT = os.path.expanduser("~/dsd_work/wavefront-decoding")
sys.path.insert(0, WF_ROOT)
import torch  # noqa: E402

from experiments.exp2_spec_bench.run import build_blocks  # noqa: E402
from src.decoding.state import ActiveToken  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--T-total", type=int, required=True)
ap.add_argument("--prefix", type=int, default=600)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--out", required=True)
a = ap.parse_args()

blocks, wrapper = build_blocks(a.model, dtype=torch.bfloat16, device="cuda", T_total=a.T_total,
                               state_init="random", state_seed=0)
blocks.kv_budget_s = None
blocks.exit_threshold = None
blocks.exit_hidden_eps = None
ids = torch.randint(100, 20000, (1, a.prefix), device="cuda")
state, _ = blocks.prefill(ids, a.T_total, 1)
cache = state.kv_cache
P0 = state.prefix_len
res = {}
widths = [w for w in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if True]
for layout in ("same", "wave"):
    for n in widths:
        if layout == "wave" and n > a.T_total:
            continue
        toks = []
        for j in range(n):
            h, ss = blocks.prelude_forward(cache, 1000 + j, P0 + j)
            t = ActiveToken(token_id=1000 + j, position=P0 + j, role="verify")
            t.hidden, t.static_state = h, ss
            toks.append(t)
        base = [(t.hidden.clone(), None if t.static_state is None else t.static_state.clone()) for t in toks]

        def reset():
            for j, t in enumerate(toks):
                t.hidden = base[j][0].clone()
                t.static_state = base[j][1]
                t.step = 1 if layout == "same" else (a.T_total - 1 - j)
                t.should_exit = False
                t.prev_hidden = None
        try:
            for _ in range(3):
                reset(); blocks.advance_one_step(toks, cache)
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.iters):
                reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                blocks.advance_one_step(toks, cache)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            res[f"{layout}_{n}"] = {"median_ms": 1e3 * ts[len(ts) // 2], "p10_ms": 1e3 * ts[len(ts) // 10]}
            print(layout, n, f"{1e3*ts[len(ts)//2]:.2f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            res[f"{layout}_{n}"] = {"error": repr(e)[:200]}
            print(layout, n, "ERR", repr(e)[:200], flush=True)
        cache.drop_positions_at_and_after(P0)
json.dump({"model": a.model, "prefix": a.prefix, "gpu": torch.cuda.get_device_name(0), "res": res},
          open(a.out, "w"), indent=1)
