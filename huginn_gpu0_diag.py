#!/usr/bin/env python3
"""Isolated single-H100 Huginn diagnosis for GPU0.

This file is intentionally additive: it does not change the four-GPU runners that
may already be in flight.  It answers three narrow questions:

1) Does latest-m4 actually reduce fixed-depth KV memory relative to full?
2) Does latest-m4-compress-s4 move the memory boundary as expected?
3) At the failing B=64/L=2048/D=16 point, is one-token generation already OOM
   (initial full-prompt/prefill dominated), or does OOM appear only after more
   decode tokens are requested?

It also performs a small B=1 adaptive-output comparison between latest-m4 and
latest-m4-compress-s4.  B>1 adaptive generation is deliberately NOT used as a
correctness reference because the upstream Huginn implementation explicitly
warns that adaptive-compute evaluation should use batch_size=1.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _synthetic_inputs(tokenizer, batch_size: int, context_len: int, device):
    import torch

    base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. Huginn recurrent inference memory probe. ",
        add_special_tokens=False,
    )
    if not base:
        base = [tokenizer.eos_token_id or 1]
    reps = (context_len + len(base) - 1) // len(base)
    ids = (base * reps)[:context_len]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _mem(torch) -> dict[str, float]:
    free_b, total_b = torch.cuda.mem_get_info()
    gib = 1024**3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
        "free_gib": free_b / gib,
        "total_gib": total_b / gib,
    }


def _clear(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _run_fixed_cell(model, tokenizer, *, batch_size: int, context_len: int, depth: int,
                    kv_policy: str, new_tokens: int, device) -> dict[str, Any]:
    import torch

    _clear(torch)
    inputs = _synthetic_inputs(tokenizer, batch_size, context_len, device)
    after_inputs = _mem(torch)
    row: dict[str, Any] = {
        "kind": "fixed_depth_memory",
        "batch_size": batch_size,
        "context_len": context_len,
        "depth": depth,
        "kv_policy": kv_policy,
        "new_tokens": new_tokens,
        "status": "OK",
        "after_inputs": after_inputs,
    }
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                num_steps=depth,
                cache_lookup_strategy=kv_policy,
            )
        torch.cuda.synchronize()
        row["generated_shape"] = list(out.sequences.shape if hasattr(out, "sequences") else out.shape)
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "OOM"
        row["error"] = str(exc)
    except Exception as exc:  # diagnostic must preserve unsupported/error cells
        row["status"] = f"ERROR:{type(exc).__name__}"
        row["error"] = str(exc)
    row["elapsed_s"] = time.perf_counter() - t0
    row.update(_mem(torch))
    del inputs
    if "out" in locals():
        del out
    _clear(torch)
    return row


def _adaptive_b1_compare(model, tokenizer, load_prompts, force_fixed_huginn_decode, *,
                         num_requests: int, prompt_cap: int, new_tokens: int,
                         threshold: float, depth: int, device) -> dict[str, Any]:
    import torch
    from transformers import GenerationConfig

    if not force_fixed_huginn_decode(model):
        raise RuntimeError("Huginn fixed-decode adapter could not find model._get_stops")
    prompts = load_prompts("alpaca", num_requests, 1)
    policies = ["latest-m4", "latest-m4-compress-s4"]
    results: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}

    for policy in policies:
        for request_id, prompt in enumerate(prompts):
            _clear(torch)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_cap).to(device)
            prompt_len = int(enc["input_ids"].shape[1])
            gc_cfg = GenerationConfig.from_model_config(model.config)
            gc_cfg.do_sample = False
            gc_cfg.max_new_tokens = new_tokens
            gc_cfg.min_new_tokens = new_tokens
            gc_cfg.return_dict_in_generate = True
            gc_cfg.pad_token_id = tokenizer.pad_token_id
            status = "OK"
            token_ids = None
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        generation_config=gc_cfg,
                        max_new_tokens=new_tokens,
                        tokenizer=tokenizer,
                        criterion="latent-diff",
                        exit_threshold=threshold,
                        num_steps=depth,
                        min_steps=1,
                        check_criterion_every_n_steps=1,
                        cache_lookup_strategy=policy,
                    )
                torch.cuda.synchronize()
                seq = out.sequences if hasattr(out, "sequences") else out
                token_ids = seq[0, prompt_len:].detach().cpu().tolist()
            except torch.cuda.OutOfMemoryError as exc:
                status = "OOM"
                error = str(exc)
            except Exception as exc:
                status = f"ERROR:{type(exc).__name__}"
                error = str(exc)
            rec = {
                "request_id": request_id,
                "policy": policy,
                "prompt_tokens": prompt_len,
                "generated_tokens": len(token_ids) if token_ids is not None else 0,
                "output_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() if token_ids is not None else None,
                "status": status,
                "elapsed_s": time.perf_counter() - t0,
                **_mem(torch),
            }
            if status != "OK":
                rec["error"] = error
            results[policy].append(rec)
            if "out" in locals():
                del out
            del enc
            _clear(torch)

    exact = []
    for a, b in zip(results[policies[0]], results[policies[1]]):
        comparable = a["status"] == "OK" and b["status"] == "OK"
        exact.append(bool(comparable and a["output_sha256"] == b["output_sha256"]))
    return {
        "kind": "adaptive_b1_policy_compare",
        "threshold": threshold,
        "rmax": depth,
        "fixed_decode_tokens": new_tokens,
        "policies": policies,
        "rows": results,
        "num_requests": num_requests,
        "num_exact": sum(exact),
        "exact_agreement": sum(exact) / len(exact) if exact else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.26)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from poc import force_fixed_huginn_decode, load_prompts

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_kw: dict[str, Any] = {"trust_remote_code": True}
    if args.revision:
        hf_kw["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **hf_kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        **({"revision": args.revision} if args.revision else {}),
    ).eval()
    torch.cuda.synchronize()
    model_load_mem = _mem(torch)

    rows: list[dict[str, Any]] = []

    # A. Prove/deny that latest-m4 is a memory-saving policy at a moderate cell.
    for policy in ["full", "latest-m4", "latest-m4-compress-s4"]:
        rows.append(_run_fixed_cell(
            model, tokenizer, batch_size=8, context_len=512, depth=args.depth,
            kv_policy=policy, new_tokens=1, device=device,
        ))

    # B. Context boundary under the actual compression policy.  One generated
    # token is deliberate: if this OOMs, the initial full-prompt forward alone
    # is already enough; later decode growth is not required to trigger OOM.
    for L in [512, 1024, 1536, 2048]:
        rows.append(_run_fixed_cell(
            model, tokenizer, batch_size=64, context_len=L, depth=args.depth,
            kv_policy="latest-m4-compress-s4", new_tokens=1, device=device,
        ))

    # C. Batch boundary at the long-context point.
    for B in [16, 32, 48, 64]:
        rows.append(_run_fixed_cell(
            model, tokenizer, batch_size=B, context_len=2048, depth=args.depth,
            kv_policy="latest-m4-compress-s4", new_tokens=1, device=device,
        ))

    # D. Decode-growth probe at cells expected to be less extreme.  Compare
    # new_tokens=1 vs 8.  A large peak increase here points to cache/decode
    # growth; similar peaks point toward the initial full-prompt forward.
    for B, L in [(16, 2048), (32, 1024)]:
        for nt in [1, 8]:
            rows.append(_run_fixed_cell(
                model, tokenizer, batch_size=B, context_len=L, depth=args.depth,
                kv_policy="latest-m4-compress-s4", new_tokens=nt, device=device,
            ))

    _json_dump(outdir / "memory_cells.json", {
        "model": args.model,
        "revision": args.revision or "default",
        "depth": args.depth,
        "model_load_memory": model_load_mem,
        "rows": rows,
        "interpretation_contract": {
            "one_token_oom": "initial full-prompt/prefill-side peak is sufficient to OOM; decode growth is not required",
            "one_token_ok_multi_token_oom": "decode/cache growth contributes materially after the initial full-prompt forward",
            "latest_m4_similar_to_full": "expected: latest-m4 is a missing-KV lookup policy, not recurrence-slot compression",
            "compress_s4_lower_peak": "expected: recurrence cache indices are reused modulo the 4-layer core",
        },
    })

    # E. Small semantic sanity, B=1 only.  This is deliberately separate from
    # any throughput claim.
    adaptive = _adaptive_b1_compare(
        model, tokenizer, load_prompts, force_fixed_huginn_decode,
        num_requests=4, prompt_cap=512, new_tokens=32,
        threshold=args.threshold, depth=args.depth, device=device,
    )
    _json_dump(outdir / "adaptive_b1_policy_compare.json", adaptive)

    summary = {
        "model": args.model,
        "revision": args.revision or "default",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "model_load_memory": model_load_mem,
        "num_memory_cells": len(rows),
        "memory_status_counts": {},
        "adaptive_b1_exact_agreement": adaptive["exact_agreement"],
        "evidence_kind": "HUGINN_GPU0_ISOLATED_DIAGNOSIS",
        "claim_boundary": "diagnostic only; not an integrated no-refill runtime or scheduler speed measurement",
    }
    for r in rows:
        summary["memory_status_counts"][r["status"]] = summary["memory_status_counts"].get(r["status"], 0) + 1
    _json_dump(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
