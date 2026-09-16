#!/usr/bin/env python3
"""Hardware calibration + trace replay for offline no-refill looped-LM batching.

This file deliberately distinguishes two evidence levels:

1) `calibrate` runs REAL model inference on the assigned GPU at fixed recurrent
   depths and measures wall-clock first-token and incremental decode latency.
2) `replay` combines those measured latency surfaces with observed per-token
   exit-depth traces to estimate FIFO/oracle/history no-refill throughput and
   batch-completion time (BCT).

The replay is hardware-calibrated, but it is NOT an integrated dynamic
batch-compaction runtime. Do not label replay numbers as an end-to-end engine
measurement. They are the bridge between slot-utilization headroom and a future
custom no-refill runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from poc import add_causal_predictors, order_token_group, read_traces


def csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def make_inputs(tokenizer, batch_size: int, context_len: int, device):
    import torch

    base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. Loop inference benchmark. ",
        add_special_tokens=False,
    )
    if not base:
        base = [tokenizer.eos_token_id or 1]
    reps = (context_len + len(base) - 1) // len(base)
    ids = (base * reps)[:context_len]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_model(model_name: str, revision: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        **kwargs,
    ).eval()
    return model, tokenizer


def generation_kwargs(
    family: str,
    model,
    depth: int,
    new_tokens: int,
    tokenizer,
    cache_lookup_strategy: str,
) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if family == "ouro":
        # Ouro's public forward always iterates self.model.total_ut_steps.  For
        # calibration only, physically shorten the loop to the requested fixed
        # depth and select that final state.
        model.model.total_ut_steps = int(depth)
        if hasattr(model, "config"):
            model.config.total_ut_steps = int(depth)
        if hasattr(model.model, "config"):
            model.model.config.total_ut_steps = int(depth)
        kw["exit_at_step"] = int(depth) - 1
    elif family == "huginn":
        # No adaptive criterion -> Huginn follows the regular HF generation path
        # while forwarding this fixed recurrence count into model.forward.
        kw["num_steps"] = int(depth)
        kw["cache_lookup_strategy"] = cache_lookup_strategy
    else:
        raise ValueError(f"unknown family={family}")
    return kw


def timed_generate(model, inputs, kwargs: dict[str, Any]) -> float:
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        _ = model.generate(**inputs, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def calibrate(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("calibrate requires CUDA")
    model, tokenizer = load_model(args.model, args.revision or None)
    device = torch.device("cuda")
    batches = csv_ints(args.batch_sizes)
    contexts = csv_ints(args.context_lengths)
    depths = csv_ints(args.depths)

    rows: list[dict[str, Any]] = []
    for L in contexts:
        for B in batches:
            inputs = make_inputs(tokenizer, B, L, device)
            for depth in depths:
                kv_policy = "native-reference" if args.family == "ouro" else args.cache_lookup_strategy
                row = {
                    "model": args.model,
                    "family": args.family,
                    "revision": args.revision or "default",
                    "batch_size": B,
                    "context_len": L,
                    "depth": depth,
                    "rmax": max(depths),
                    "kv_policy": kv_policy,
                    "decode_tokens": args.decode_tokens,
                    "evidence_kind": "H100_FIXED_DEPTH_CALIBRATION",
                    "status": "OK",
                }
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    # Warm once.  The timed calls are fresh generate() calls so
                    # cache state cannot leak across repetitions.
                    _ = timed_generate(
                        model,
                        inputs,
                        generation_kwargs(
                            args.family, model, depth, 2, tokenizer, args.cache_lookup_strategy
                        ),
                    )
                    first_times = []
                    multi_times = []
                    for _rep in range(args.repeats):
                        first_times.append(
                            timed_generate(
                                model,
                                inputs,
                                generation_kwargs(
                                    args.family, model, depth, 1, tokenizer, args.cache_lookup_strategy
                                ),
                            )
                        )
                        multi_times.append(
                            timed_generate(
                                model,
                                inputs,
                                generation_kwargs(
                                    args.family,
                                    model,
                                    depth,
                                    args.decode_tokens,
                                    tokenizer,
                                    args.cache_lookup_strategy,
                                ),
                            )
                        )
                    first_ms = float(np.median(first_times))
                    multi_ms = float(np.median(multi_times))
                    incr = (multi_ms - first_ms) / max(1, args.decode_tokens - 1)
                    row.update(
                        first_token_ms=first_ms,
                        multi_token_ms=multi_ms,
                        incremental_decode_ms=float(incr),
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row.update(status="OOM", first_token_ms=np.nan, multi_token_ms=np.nan, incremental_decode_ms=np.nan)
                except Exception as exc:
                    row.update(status=f"ERROR:{type(exc).__name__}:{exc}", first_token_ms=np.nan, multi_token_ms=np.nan, incremental_decode_ms=np.nan)
                finally:
                    row.update(
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / (1024**3),
                        peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / (1024**3),
                    )
                rows.append(row)
                print(row, flush=True)

    raw = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)

    fits = []
    ok = raw[(raw.status == "OK") & raw.incremental_decode_ms.notna()].copy()
    for (B, L), g in ok.groupby(["batch_size", "context_len"]):
        if g.depth.nunique() < 2:
            continue
        x = g.depth.to_numpy(float)
        y = g.incremental_decode_ms.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        yhat = intercept + slope * x
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        # Full-depth first-token time is used as a measured prefill+first-output
        # anchor for BCT replay; it is intentionally kept separate from decode.
        maxd = int(g.depth.max())
        gd = g[g.depth == maxd].iloc[0]
        fits.append(
            {
                "model": args.model,
                "family": args.family,
                "revision": args.revision or "default",
                "kv_policy": "native-reference" if args.family == "ouro" else args.cache_lookup_strategy,
                "rmax": max(depths),
                "batch_size": int(B),
                "context_len": int(L),
                "boundary_ms": max(0.0, float(intercept)),
                "recurrent_step_ms": max(0.0, float(slope)),
                "fit_r2": r2,
                "full_depth": maxd,
                "full_depth_first_token_ms": float(gd.first_token_ms),
                "full_depth_incremental_decode_ms": float(gd.incremental_decode_ms),
            }
        )
    fit_path = out.with_name(out.stem + "_fit.csv")
    pd.DataFrame(fits).to_csv(fit_path, index=False)
    print(f"wrote {out} and {fit_path}")


def nearest_context(fit: pd.DataFrame, context: int) -> int:
    vals = sorted(set(int(x) for x in fit.context_len.unique()))
    return min(vals, key=lambda x: abs(x - context))


def interp_step_ms(fit: pd.DataFrame, context_len: int, active_batch: int) -> float:
    sub = fit[fit.context_len == context_len].sort_values("batch_size")
    xs = sub.batch_size.to_numpy(float)
    ys = sub.recurrent_step_ms.to_numpy(float)
    if len(xs) == 0:
        raise ValueError(f"no calibration for context {context_len}")
    return float(np.interp(float(active_batch), xs, ys, left=ys[0], right=ys[-1]))


def boundary_ms(fit: pd.DataFrame, context_len: int, batch_size: int) -> float:
    sub = fit[fit.context_len == context_len].sort_values("batch_size")
    xs = sub.batch_size.to_numpy(float)
    ys = sub.boundary_ms.to_numpy(float)
    return float(np.interp(float(batch_size), xs, ys, left=ys[0], right=ys[-1]))


def first_token_ms(fit: pd.DataFrame, context_len: int, batch_size: int) -> float:
    sub = fit[fit.context_len == context_len].sort_values("batch_size")
    xs = sub.batch_size.to_numpy(float)
    ys = sub.full_depth_first_token_ms.to_numpy(float)
    return float(np.interp(float(batch_size), xs, ys, left=ys[0], right=ys[-1]))


def replay_microbatch(depths: np.ndarray, context: int, fit: pd.DataFrame, B: int) -> float:
    L = nearest_context(fit, context)
    total = boundary_ms(fit, L, B)
    for r in range(1, int(depths.max()) + 1):
        active = int(np.sum(depths >= r))
        total += interp_step_ms(fit, L, active)
    return float(total)


def replay(args):
    traces = read_traces(args.traces)
    traces = add_causal_predictors(traces, args.recent_k)
    fit = pd.read_csv(args.fit)
    batches = csv_ints(args.batch_sizes)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    seed = args.seed

    all_rows = []
    for (source_file, model), raw in traces.groupby(["source_file", "model"], dropna=False):
        metadata = {
            key: raw.iloc[0][key]
            for key in ("family", "dataset", "exit_threshold", "kv_policy", "rmax")
            if key in raw.columns
        }
        model_fit = fit[fit.model == model]
        if model_fit.empty:
            # Allow calibration file to contain a local snapshot path while trace
            # contains the Hub id, as long as only one model is calibrated.
            if fit.model.nunique() == 1:
                model_fit = fit
            else:
                print(f"[skip] no latency fit for {model}")
                continue
        run = Path(str(source_file)).stem
        for B in batches:
            # Common fair subset is frozen once per token index before ordering.
            bases = []
            for token_idx, g in raw.groupby("token_idx", sort=True):
                base = g.sort_values("request_id")
                n_full = len(base) // B
                if n_full:
                    bases.append((int(token_idx), base.iloc[: n_full * B].copy()))
            for strategy in strategies:
                rng = random.Random(seed + B * 1009 + sum(ord(c) for c in strategy))
                decode_ms = 0.0
                prefill_anchor_ms = 0.0
                ntokens = 0
                nmb = 0
                for token_idx, base in bases:
                    ordered = order_token_group(base, strategy, rng)
                    assert set(ordered.index) == set(base.index)
                    n_full = len(ordered) // B
                    for bi in range(n_full):
                        mb = ordered.iloc[bi * B : (bi + 1) * B]
                        depths = mb.depth.to_numpy(int)
                        context = int(np.median(mb.prompt_tokens.to_numpy(int) + token_idx)) if "prompt_tokens" in mb else 128
                        # Token 0 is the prefill/first-output boundary in the
                        # original generation trace.  Use measured full-depth
                        # first-token time once rather than double-counting a
                        # synthetic adaptive decode step.
                        if token_idx == 0:
                            L = nearest_context(model_fit, context)
                            prefill_anchor_ms += first_token_ms(model_fit, L, B)
                        else:
                            decode_ms += replay_microbatch(depths, context, model_fit, B)
                        ntokens += len(mb)
                        nmb += 1
                total_ms = prefill_anchor_ms + decode_ms
                all_rows.append(
                    {
                        "run": run,
                        "model": model,
                        **metadata,
                        "batch_size": B,
                        "strategy": strategy,
                        "selected_generated_tokens": ntokens,
                        "num_microbatches": nmb,
                        "prefill_first_token_anchor_ms": prefill_anchor_ms,
                        "decode_replay_ms": decode_ms,
                        "hardware_replay_bct_ms": total_ms,
                        "hardware_replay_tokens_per_s": (1000.0 * ntokens / total_ms) if total_ms > 0 else np.nan,
                        "evidence_kind": "HARDWARE_CALIBRATED_TRACE_REPLAY_NOT_INTEGRATED_RUNTIME",
                    }
                )

    out = pd.DataFrame(all_rows)
    fifo = out[out.strategy == "fifo"][["run", "batch_size", "hardware_replay_bct_ms", "hardware_replay_tokens_per_s"]].rename(
        columns={"hardware_replay_bct_ms": "fifo_bct_ms", "hardware_replay_tokens_per_s": "fifo_tokens_per_s"}
    )
    out = out.merge(fifo, on=["run", "batch_size"], how="left")
    out["bct_speedup_vs_fifo"] = out.fifo_bct_ms / out.hardware_replay_bct_ms
    out["throughput_speedup_vs_fifo"] = out.hardware_replay_tokens_per_s / out.fifo_tokens_per_s
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out[out.strategy.isin(["fifo", "oracle", "prev", "recent", "seqmean"])].to_string(index=False))
    print(f"wrote {path}")


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("--model", required=True)
    c.add_argument("--family", required=True, choices=["ouro", "huginn"])
    c.add_argument("--revision", default="")
    c.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    c.add_argument("--context-lengths", default="32,128,512")
    c.add_argument("--depths", required=True)
    c.add_argument("--cache-lookup-strategy", default="full")
    c.add_argument("--decode-tokens", type=int, default=6)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--output", required=True)
    c.set_defaults(func=calibrate)

    r = sub.add_parser("replay")
    r.add_argument("--traces", nargs="+", required=True)
    r.add_argument("--fit", required=True)
    r.add_argument("--batch-sizes", default="4,8,16,32,64")
    r.add_argument("--strategies", default="fifo,prev,recent,seqmean,oracle")
    r.add_argument("--recent-k", type=int, default=4)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--output", required=True)
    r.set_defaults(func=replay)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
