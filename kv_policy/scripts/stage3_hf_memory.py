"""Measured KV residency of every policy with the policy-sized (compact) HF store.

vllm-rlt implements only FULL (last_exited) and strict LAST (shared); paper-style decode-only
LAST and AVERAGE are measured here instead. For each policy/mode the first frozen Stage-2 GSM8K
batch is generated with the compact store; we record the allocated KV bytes, CUDA peak memory and
wall time, and compare the generated tokens with the Stage-2 broadcast-layout receipts for the same
batch (same semantics, different attention input layout -> bf16 agreement, not bitwise identity).
The HF reference engine is not a serving engine: its timings are reported only for completeness.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kvp.jobs import ModelContext, batches, gpu_identity, protocol  # noqa: E402

CELLS = [("full", "decode"), ("last", "decode"), ("average", "decode"), ("last", "strict"), ("average", "strict")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage2-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = ModelContext(args.model)
    ctx.runner.store = "compact"
    items, _ = ctx.dataset("gsm8k")
    batch = batches(items)[0]
    proto = protocol(ctx.spec.family, "gsm8k")
    prompts, uids = [it["prompt_ids"] for it in batch], [it["uid"] for it in batch]
    s2 = Path(args.stage2_dir) / "raw"
    rep = {"model": args.model, "gpu": gpu_identity(), "batch": 0, "B": len(batch),
           "max_new_tokens": proto.max_new_tokens, "cells": []}
    bb = ctx.runner.bb
    plane = 2 * bb.H * bb.D * 2  # K+V bytes per token per layer plane (bf16)
    for pol, mode in CELLS:
        gc.collect()
        torch.cuda.empty_cache()
        res = ctx.runner.generate(prompts, uids, pol, mode, proto.max_new_tokens, set(), lambda b, x: False)
        ref_f = s2 / f"{args.model}__gsm8k__gen__{pol}-{mode}__b00.jsonl"
        agree = None
        if ref_f.exists():
            ref = {json.loads(ln)["uid"]: json.loads(ln) for ln in ref_f.read_text().splitlines()[1:]}
            pairs = [(ref[u]["gen_token_ids"], g) for u, g in zip(uids, res.gen_ids)]
            # stage-2 rows stop at a stop string/EOS; compare that prefix
            agree = sum(r == g[: len(r)] for r, g in pairs) / len(pairs)
        P, T = res.stats["P"], res.stats["T_alloc"]
        rec = bb.n_rec * plane * len(batch)
        analytic = {"full": rec * bb.R * T, "decode": rec * (bb.R * P + (T - P)), "strict": rec * T}
        rep["cells"].append({
            "policy": pol, "mode": mode, "P": P, "T_alloc": T,
            "kv_store_bytes_measured": res.stats["kv_store_bytes"],
            "recurrent_kv_bytes_analytic": analytic["full" if pol == "full" else mode],
            "non_recurrent_kv_bytes": bb.n_plain * plane * T * len(batch),
            "cuda_peak_bytes": res.stats["peak_mem_bytes"], "wall_s": res.stats["total_s"],
            "prefill_s": res.stats["prefill_s"], "decode_steps": res.stats["decode_steps"],
            "token_prefix_agreement_vs_stage2_broadcast": agree})
        print(rep["cells"][-1])
    Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
