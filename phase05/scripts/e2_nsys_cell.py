"""E2 profiler subset: one (model, B, L) cell under Nsight Systems; each Core call inside an NVTX range.
Predeclared subset: every model at (B=1, L=512) and (B=32, L=512). Nsight Compute (DRAM bytes, achieved
FLOPs, occupancy) is not installed on this host; only Nsight Systems timeline evidence is available."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_stage_microbench import SEED_ROWS, Bench, long_prompts  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--L", type=int, required=True)
    a = p.parse_args()
    import torch
    bench = Bench(a.model)
    prompts = long_prompts(bench.tokenizer, a.L, SEED_ROWS)
    ad, tok = bench.build(a.B, prompts[: min(SEED_ROWS, a.B)])
    rows = torch.arange(a.B, device="cuda")
    ad.backend.instrumentation = "benchmark"
    for i in range(7):
        ad.begin_token()
        ad.backend.set_active(rows)
        emb, state = ad.prelude(tok)
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range("core_warmup" if i < 2 else "core"):
            ad.recurrent_step(state, emb, 1)
            torch.cuda.synchronize()
        bench.rollback(ad)


if __name__ == "__main__":
    main()
