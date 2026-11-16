"""E1: oracle hardware headroom of depth-batching on the real core (GPU, opt-in).

For each (B, L, w): cohort prefilled at fixed depth D with FULL depth-indexed KV (every depth slot
holds real prompt K/V), then ONE physical step_many call advancing w rows per sequence at the
current position, depths 1..w (each row attends its own depth slot = the per-depth KV context a
depth-parallel solver would need). Inputs are the exact states h_0..h_{w-1} (oracle: recorded by a
serial pass first, which is NOT counted). Timed with CUDA events after warm-up:
  T_F(B)   = one depth for B sequences (w=1),  T_F(wB) = w depths for B sequences,
  S_oracle(w,B) = w * T_F(B) / T_F(wB)   (non-deployable ceiling; no predictor/correction/commit).
Also: serial D-step core time, coda time, peak/reserved HBM, host wall incl. Python row packing.
Cells that OOM are recorded as OOM. Writes <out>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import wf_engine as W
from .depth_diag import cohort_texts, env_info


def timed(fn, warm: int, reps: int):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev, wall = [], []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3)
        ev.append(s.elapsed_time(e))
    q = torch.tensor(ev)
    return dict(median_ms=float(q.median()), p10_ms=float(q.quantile(0.1)), p90_ms=float(q.quantile(0.9)),
                host_wall_median_ms=float(torch.tensor(wall).median()))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--batches", default="1,8,16,64,128")
    p.add_argument("--prompt-lens", default="256,1024")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    lm, tok, batch_cls, kind = W.load_model(a.model, dtype)
    D = a.depth
    widths = sorted({w for w in (1, 2, 4, 8, 16, 32, D) if w <= D})
    res = dict(config=vars(a) | env_info(a), widths=widths, cells=[])
    for L in [int(x) for x in a.prompt_lens.split(",")]:
        base = W.make_cohort(tok, cohort_texts("gsm8k3" if L <= 256 else "long", 64), L, 16)
        for B in [int(x) for x in a.batches.split(",")]:
            cell = dict(L=L, B=B)
            eng = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ids = base.repeat((B + 15) // 16, 1)[:B]
                eng = W.DepthEngine(lm, batch_cls, kind, D, B, L + 8)
                eng.prefill(ids)
                h0 = eng.prelude(ids[:, -1].tolist())
                ref = eng.serial(h0)                                  # oracle exact inputs
                t = {}
                for w in widths:
                    st = ref[:w].contiguous()
                    dids = torch.arange(1, w + 1)
                    t[w] = timed(lambda st=st, dids=dids: eng.step_many(st, dids), a.warmup, a.reps)
                cell["T_F"] = {str(w): t[w] for w in widths}
                cell["S_oracle"] = {str(w): w * t[1]["median_ms"] / t[w]["median_ms"] for w in widths}
                cell["S_oracle_host"] = {str(w): w * t[1]["host_wall_median_ms"] / t[w]["host_wall_median_ms"]
                                         for w in widths}
                cell["serial_D"] = timed(lambda: eng.serial(h0), 2, max(5, a.reps // 3))
                cell["coda"] = timed(lambda: eng.logits(ref[-1]), a.warmup, a.reps)
                cell["hbm_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
                cell["hbm_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
                cell["status"] = "ok"
            except torch.cuda.OutOfMemoryError as e:
                cell["status"] = "OOM"
                cell["error"] = str(e)[:300]
            del eng
            torch.cuda.empty_cache()
            res["cells"].append(cell)
            s = cell.get("S_oracle", {})
            print(f"L={L} B={B} {cell['status']} S_oracle={ {k: round(v, 2) for k, v in s.items()} }", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
