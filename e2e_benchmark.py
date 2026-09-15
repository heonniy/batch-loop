#!/usr/bin/env python3
"""Native whole-model E2E/BCT/throughput benchmark for looped LMs.

This benchmark intentionally measures the official/native generation path for a
model family.  It is separate from the trace-based no-refill scheduler replay.

Evidence labels matter:
- Ouro's current public HF implementation exposes exit depth post-hoc while still
  computing all recurrent steps, so threshold-dependent timing here is a native
  whole-model baseline, NOT physical early-exit speedup.
- Huginn exposes adaptive recurrent generation.  Batched adaptive execution is
  only trusted when the optional deterministic batch-vs-B1 parity check passes.

The script loads a model once, then sweeps thresholds and batch sizes to avoid
including repeated model-load cost in BCT/throughput measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from poc import load_prompts


def _csv_list(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _generated_lengths(sequences, attention_mask, eos_id, max_new_tokens: int) -> list[int]:
    # Inputs are left padded. The generated suffix always starts after padded input width.
    input_width = int(attention_mask.shape[1])
    suffix = sequences[:, input_width:]
    out = []
    for row in suffix:
        vals = row.tolist()
        if eos_id is not None and eos_id in vals:
            n = vals.index(eos_id) + 1
        else:
            n = min(len(vals), max_new_tokens)
        out.append(int(n))
    return out


def _prepare_batch(tokenizer, prompts: list[str], max_prompt_tokens: int, torch):
    t0 = time.perf_counter()
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    tok_s = time.perf_counter() - t0
    prompt_tokens = int(enc["attention_mask"].sum().item())
    t1 = time.perf_counter()
    enc = {k: v.to("cuda", non_blocking=False) for k, v in enc.items()}
    _sync(torch)
    h2d_s = time.perf_counter() - t1
    return enc, prompt_tokens, tok_s, h2d_s


def _generation_kwargs(args, tokenizer, model, threshold: float):
    from transformers import GenerationConfig

    if args.family == "ouro":
        kw: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.pad_token_id,
            "exit_threshold": float(threshold),
        }
        if args.fixed_output_tokens:
            kw["min_new_tokens"] = args.max_new_tokens
        return kw

    gc = GenerationConfig.from_model_config(model.config)
    gc.do_sample = False
    gc.max_new_tokens = args.max_new_tokens
    gc.return_dict_in_generate = True
    gc.pad_token_id = tokenizer.pad_token_id
    if args.fixed_output_tokens:
        gc.min_new_tokens = args.max_new_tokens
    return {
        "generation_config": gc,
        "tokenizer": tokenizer,
        "criterion": args.criterion,
        "exit_threshold": float(threshold),
        "num_steps": args.max_steps,
        "min_steps": args.min_steps,
        "check_criterion_every_n_steps": args.check_every,
        "cache_lookup_strategy": args.cache_lookup_strategy,
    }


def _run_generate(args, tokenizer, model, enc, threshold: float):
    kw = _generation_kwargs(args, tokenizer, model, threshold)
    return model.generate(**enc, **kw)


def _parity_check(args, tokenizer, model, prompts: list[str], threshold: float, batch_size: int, torch):
    n = min(args.parity_requests, batch_size, len(prompts))
    if n <= 1:
        return True, 1.0, "TRIVIAL_B1"
    sample = prompts[:n]
    enc_b, _, _, _ = _prepare_batch(tokenizer, sample, args.max_prompt_tokens, torch)
    _sync(torch)
    out_b = _run_generate(args, tokenizer, model, enc_b, threshold)
    _sync(torch)
    batch_rows = out_b.sequences.detach().cpu().tolist()

    single_rows = []
    for p in sample:
        enc_1, _, _, _ = _prepare_batch(tokenizer, [p], args.max_prompt_tokens, torch)
        _sync(torch)
        out_1 = _run_generate(args, tokenizer, model, enc_1, threshold)
        _sync(torch)
        single_rows.append(out_1.sequences[0].detach().cpu().tolist())

    # Compare generated suffixes rather than padded full inputs.
    matches = 0
    for p, br, sr in zip(sample, batch_rows, single_rows):
        plen = len(tokenizer(p, truncation=True, max_length=args.max_prompt_tokens)["input_ids"])
        # Batches are left-padded, so generated suffix is always the final max_new_tokens region
        # under fixed-output mode; without it, compare final decoded continuation strings.
        if args.fixed_output_tokens:
            bg = br[-args.max_new_tokens :]
            sg = sr[-args.max_new_tokens :]
            ok = bg == sg
        else:
            ok = tokenizer.decode(br, skip_special_tokens=True).endswith(
                tokenizer.decode(sr[plen:], skip_special_tokens=True)
            )
        matches += int(ok)
    frac = matches / n
    return frac == 1.0, frac, "EXACT_GENERATED_SUFFIX"


def benchmark(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    prompts = load_prompts(args.dataset, args.num_requests, args.seed, args.prompts_jsonl)
    thresholds = _csv_list(args.thresholds, float)
    batch_sizes = _csv_list(args.batch_sizes, int)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    _sync(torch)
    load_s = time.perf_counter() - t_load

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Warmup at B=1 for each threshold. Not included in measurements.
    for threshold in thresholds:
        enc, _, _, _ = _prepare_batch(tokenizer, [prompts[0]], args.max_prompt_tokens, torch)
        _sync(torch)
        _run_generate(args, tokenizer, model, enc, threshold)
        _sync(torch)

    for threshold in thresholds:
        for b in batch_sizes:
            parity_ok = None
            parity_fraction = None
            parity_kind = "NOT_RUN"
            if args.validate_parity:
                try:
                    parity_ok, parity_fraction, parity_kind = _parity_check(
                        args, tokenizer, model, prompts, threshold, b, torch
                    )
                except Exception as exc:  # preserve benchmarkability but do not hide invalidation
                    parity_ok = False
                    parity_fraction = float("nan")
                    parity_kind = f"ERROR:{type(exc).__name__}:{exc}"

            batch_times = []
            e2e_times = []
            tokenization_times = []
            h2d_times = []
            generated_tokens = 0
            prompt_tokens = 0
            completed_requests = 0
            failures = []
            torch.cuda.reset_peak_memory_stats()
            e2e_start = time.perf_counter()

            for start in range(0, len(prompts), b):
                group = prompts[start : start + b]
                if not group:
                    continue
                try:
                    enc, p_tok, tok_s, h2d_s = _prepare_batch(
                        tokenizer, group, args.max_prompt_tokens, torch
                    )
                    _sync(torch)
                    t0 = time.perf_counter()
                    out = _run_generate(args, tokenizer, model, enc, threshold)
                    _sync(torch)
                    model_s = time.perf_counter() - t0
                    lengths = _generated_lengths(
                        out.sequences,
                        enc["attention_mask"],
                        tokenizer.eos_token_id,
                        args.max_new_tokens,
                    )
                    g_tok = int(sum(lengths))
                    batch_times.append(model_s)
                    tokenization_times.append(tok_s)
                    h2d_times.append(h2d_s)
                    e2e_times.append(tok_s + h2d_s + model_s)
                    generated_tokens += g_tok
                    prompt_tokens += p_tok
                    completed_requests += len(group)
                except Exception as exc:
                    failures.append({
                        "start": start,
                        "batch_len": len(group),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    if args.fail_fast:
                        raise
            bct_e2e_s = time.perf_counter() - e2e_start
            model_s_total = float(sum(batch_times))
            accounted_e2e_s = float(sum(e2e_times))
            peak_gib = torch.cuda.max_memory_allocated() / (1024**3)

            if args.family == "ouro":
                adaptive_realized = False
                evidence_kind = "NATIVE_WHOLE_MODEL_POSTHOC_EXIT_BASELINE"
            else:
                adaptive_realized = bool(parity_ok) if args.validate_parity else None
                evidence_kind = (
                    "NATIVE_WHOLE_MODEL_ADAPTIVE_GENERATION_PARITY_CHECKED"
                    if parity_ok
                    else "NATIVE_WHOLE_MODEL_ADAPTIVE_GENERATION_PARITY_UNVERIFIED"
                )

            row = {
                "family": args.family,
                "model": args.model,
                "dataset": args.dataset,
                "threshold": threshold,
                "batch_size": b,
                "num_requests_target": len(prompts),
                "completed_requests": completed_requests,
                "failed_batches": len(failures),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "model_generation_s": model_s_total,
                "accounted_e2e_s": accounted_e2e_s,
                "offline_bct_e2e_s": bct_e2e_s,
                "model_generated_tokens_per_s": (generated_tokens / model_s_total) if model_s_total else math.nan,
                "e2e_generated_tokens_per_s": (generated_tokens / bct_e2e_s) if bct_e2e_s else math.nan,
                "request_throughput_per_s": (completed_requests / bct_e2e_s) if bct_e2e_s else math.nan,
                "mean_batch_model_ms": (1000.0 * np.mean(batch_times)) if batch_times else math.nan,
                "p50_batch_model_ms": (1000.0 * np.percentile(batch_times, 50)) if batch_times else math.nan,
                "p95_batch_model_ms": (1000.0 * np.percentile(batch_times, 95)) if batch_times else math.nan,
                "tokenization_s": float(sum(tokenization_times)),
                "h2d_s": float(sum(h2d_times)),
                "peak_gpu_allocated_gib": peak_gib,
                "model_load_s_excluded": load_s,
                "fixed_output_tokens": args.fixed_output_tokens,
                "parity_ok": parity_ok,
                "parity_fraction": parity_fraction,
                "parity_kind": parity_kind,
                "adaptive_compute_physically_realized": adaptive_realized,
                "evidence_kind": evidence_kind,
                "failures_json": json.dumps(failures, ensure_ascii=False),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    fields = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {output}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--family", required=True, choices=["ouro", "huginn"])
    p.add_argument("--dataset", default="alpaca", choices=["alpaca", "sharegpt"])
    p.add_argument("--prompts-jsonl", default=None)
    p.add_argument("--num-requests", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--batch-sizes", default="4,8,16,32,64")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--fixed-output-tokens", action="store_true", default=True)
    p.add_argument("--no-fixed-output-tokens", dest="fixed_output_tokens", action="store_false")
    p.add_argument("--validate-parity", action="store_true", default=True)
    p.add_argument("--no-validate-parity", dest="validate_parity", action="store_false")
    p.add_argument("--parity-requests", type=int, default=4)
    p.add_argument("--fail-fast", action="store_true")
    # Huginn-specific adaptive-generation controls.
    p.add_argument("--criterion", default="latent-diff")
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--min-steps", type=int, default=1)
    p.add_argument("--check-every", type=int, default=1)
    p.add_argument("--cache-lookup-strategy", default="full")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    benchmark(args)
