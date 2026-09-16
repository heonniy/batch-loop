#!/usr/bin/env python3
"""Trace + simulate offline no-refill batching for looped language models.

Phase-1 goal: measure whether recurrent-depth-compatible micro-batch formation can
remove the idle-slot valley of pure no-refill execution.

This file intentionally separates TRACE COLLECTION from SCHEDULER SIMULATION.
The simulation never changes model outputs: it only regroups already-observed
per-token exit depths at token boundaries.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")[:180]


def force_fixed_huginn_decode(model) -> bool:
    """Disable Huginn's private stop tokens for an exact-token benchmark.

    The pinned remote implementation terminates on three hard-coded tokens and
    does not honor GenerationConfig.min_new_tokens.  Returning an empty tensor
    from its stop-token helper is the smallest process-local adaptation that
    makes the fixed-decode workload contract real; model source stays untouched.
    """
    if not hasattr(model, "_get_stops"):
        return False
    if not hasattr(model, "_batch_loop_original_get_stops"):
        model._batch_loop_original_get_stops = model._get_stops

        def _no_stop_tokens(_self, _generation_config, _tokenizer, _model_kwargs):
            import torch

            return torch.empty(0, dtype=torch.long)

        model._get_stops = types.MethodType(_no_stop_tokens, model)
    return True


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_text(row: dict[str, Any]) -> str:
    """Best-effort prompt extraction for small public PoC datasets."""
    instruction = row.get("instruction")
    inp = row.get("input")
    if isinstance(instruction, str) and instruction.strip():
        if isinstance(inp, str) and inp.strip() and inp.strip() != instruction.strip():
            return f"{instruction.strip()}\n\n{inp.strip()}"
        return instruction.strip()

    for key in ("prompt", "question", "text", "query"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    conv = row.get("conversations") or row.get("conversation") or row.get("messages")
    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from", msg.get("role", ""))).lower()
            val = msg.get("value", msg.get("content"))
            if role in {"human", "user"} and isinstance(val, str) and val.strip():
                return val.strip()
    raise ValueError(f"Could not extract prompt from row keys={list(row.keys())[:20]}")


def load_prompts(dataset: str, num_requests: int, seed: int, prompts_jsonl: str | None = None) -> list[str]:
    if prompts_jsonl:
        rows: list[dict[str, Any]] = []
        with open(prompts_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        prompts = [_first_text(r) for r in rows]
    else:
        from datasets import load_dataset

        if dataset == "alpaca":
            ds = load_dataset("tatsu-lab/alpaca", split="train")
        elif dataset == "sharegpt":
            # Small, easy-to-load processed ShareGPT variant. The parser is generic.
            ds = load_dataset("ewof/sharegpt-instruct-unfiltered-deduped", split="train")
        else:
            raise ValueError(f"Unknown dataset={dataset}. Use alpaca/sharegpt or --prompts-jsonl.")
        prompts = []
        for row in ds:
            try:
                prompts.append(_first_text(dict(row)))
            except ValueError:
                continue

    rng = random.Random(seed)
    rng.shuffle(prompts)
    if len(prompts) < num_requests:
        raise ValueError(f"Only {len(prompts)} prompts available, requested {num_requests}")
    return prompts[:num_requests]


def ouro_exit_depth_from_gate_logits(gate_logits: list[float], threshold: float) -> int:
    """Match Ouro's cumulative exit-PDF selection. Returns 1-based depth."""
    gates = np.asarray(gate_logits, dtype=np.float64)
    lambdas = 1.0 / (1.0 + np.exp(-gates))
    remaining = 1.0
    pdf = []
    for i, lam in enumerate(lambdas):
        if i < len(lambdas) - 1:
            p = lam * remaining
            remaining *= 1.0 - lam
        else:
            p = remaining
        pdf.append(float(p))
    cumulative = np.cumsum(pdf)
    hits = np.flatnonzero(cumulative >= threshold)
    return int(hits[0] + 1) if len(hits) else len(gate_logits)


def collect_ouro(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("collect-ouro requires a CUDA GPU")

    prompts = load_prompts(args.dataset, args.num_requests, args.seed, args.prompts_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    if not hasattr(model, "model") or not hasattr(model.model, "early_exit_gate"):
        raise RuntimeError("Expected Ouro model.model.early_exit_gate but it was not found")

    total_steps = int(getattr(model.model, "total_ut_steps", 4))
    output_path = Path(args.output)
    receipt_path = output_path.with_suffix(".requests.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.append:
        output_path.unlink()
    if receipt_path.exists() and not args.append:
        receipt_path.unlink()

    if not 0 <= args.start_request_id <= len(prompts):
        raise ValueError(f"start_request_id={args.start_request_id} outside 0..{len(prompts)}")
    selected_prompts = prompts[args.start_request_id :]
    with torch.inference_mode():
        for request_id, prompt in enumerate(
            tqdm(selected_prompts, desc=f"Ouro {args.model}"), start=args.start_request_id
        ):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to("cuda")
            prompt_len = int(enc["input_ids"].shape[1])
            gate_calls: list[torch.Tensor] = []

            def hook(_module, _inputs, output):
                gate_calls.append(output.detach().float().cpu())

            handle = model.model.early_exit_gate.register_forward_hook(hook)
            try:
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    exit_threshold=float(args.exit_threshold),
                    pad_token_id=tokenizer.pad_token_id,
                )
            finally:
                handle.remove()

            sequences = out.sequences.detach().cpu()
            generated = sequences[0, prompt_len:]
            gen_len = int(generated.numel())
            num_forward_groups = len(gate_calls) // total_steps
            if num_forward_groups < gen_len:
                raise RuntimeError(
                    f"Captured only {len(gate_calls)} gate calls ({num_forward_groups} forward groups) "
                    f"for {gen_len} generated tokens"
                )

            rows = []
            for token_idx in range(gen_len):
                group = gate_calls[token_idx * total_steps : (token_idx + 1) * total_steps]
                gate_logits = [float(g[0, -1, 0].item()) for g in group]
                depth = ouro_exit_depth_from_gate_logits(gate_logits, float(args.exit_threshold))
                rows.append(
                    {
                        "model": args.model,
                        "family": "ouro",
                        "dataset": args.dataset,
                        "request_id": request_id,
                        "token_idx": token_idx,
                        "depth": depth,
                        "max_depth": total_steps,
                        "generated_token_id": int(generated[token_idx].item()),
                        "prompt_tokens": prompt_len,
                        "exit_rule": "learned_gate_posthoc",
                        "exit_threshold": float(args.exit_threshold),
                        "kv_policy": "native-reference",
                        "rmax": total_steps,
                        "fixed_decode_tokens": True,
                        "seed": args.seed,
                    }
                )
            _write_jsonl(output_path, rows, append=True)
            _write_jsonl(
                receipt_path,
                [{
                    "request_id": request_id,
                    "prompt_tokens": prompt_len,
                    "generated_tokens": gen_len,
                    "trace_rows": len(rows),
                    "eos_observed": bool(tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated.tolist()),
                    "hit_decode_cap": gen_len == args.max_new_tokens,
                    "kv_policy": "native-reference",
                    "rmax": total_steps,
                    "status": "OK",
                }],
                append=True,
            )

    print(f"wrote {output_path}")


def collect_huginn(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    _seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("collect-huginn requires a CUDA GPU")

    prompts = load_prompts(args.dataset, args.num_requests, args.seed, args.prompts_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    if not force_fixed_huginn_decode(model):
        raise RuntimeError("Huginn fixed-decode adapter could not find model._get_stops")

    output_path = Path(args.output)
    receipt_path = output_path.with_suffix(".requests.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.append:
        output_path.unlink()
    if receipt_path.exists() and not args.append:
        receipt_path.unlink()

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.do_sample = False
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.min_new_tokens = args.max_new_tokens
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True
    generation_config.pad_token_id = tokenizer.pad_token_id

    # Huginn's own source warns that faithful adaptive-compute evaluation should use batch_size=1.
    if not 0 <= args.start_request_id <= len(prompts):
        raise ValueError(f"start_request_id={args.start_request_id} outside 0..{len(prompts)}")
    selected_prompts = prompts[args.start_request_id :]
    with torch.inference_mode():
        for request_id, prompt in enumerate(
            tqdm(selected_prompts, desc=f"Huginn {args.model}"), start=args.start_request_id
        ):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to("cuda")
            prompt_len = int(enc["input_ids"].shape[1])
            out = model.generate(
                **enc,
                generation_config=generation_config,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                criterion=args.criterion,
                exit_threshold=("auto" if args.exit_threshold == "auto" else float(args.exit_threshold)),
                num_steps=args.max_steps,
                min_steps=args.min_steps,
                check_criterion_every_n_steps=args.check_every,
                cache_lookup_strategy=args.cache_lookup_strategy,
            )

            scores = list(out.scores or [])
            sequences = out.sequences.detach().cpu()
            generated = sequences[0, prompt_len : prompt_len + len(scores)]
            rows = []
            for token_idx, score_entry in enumerate(scores):
                # Custom Huginn GenerateDecoderOnlyOutput stores:
                # scores[token_idx] = [compute_steps_per_seq, exit_values_per_seq]
                compute_steps_per_seq = score_entry[0]
                exit_values_per_seq = score_entry[1]
                depth = int(compute_steps_per_seq[0])
                rows.append(
                    {
                        "model": args.model,
                        "family": "huginn",
                        "dataset": args.dataset,
                        "request_id": request_id,
                        "token_idx": token_idx,
                        "depth": depth,
                        "max_depth": int(args.max_steps),
                        "generated_token_id": int(generated[token_idx].item()) if token_idx < len(generated) else None,
                        "prompt_tokens": prompt_len,
                        "exit_rule": args.criterion,
                        "exit_threshold": args.exit_threshold,
                        "exit_values": exit_values_per_seq[0],
                        "kv_policy": args.cache_lookup_strategy,
                        "rmax": int(args.max_steps),
                        "fixed_decode_tokens": True,
                        "huginn_private_stop_tokens_suppressed": True,
                        "seed": args.seed,
                    }
                )
            _write_jsonl(output_path, rows, append=True)
            generated_len = int(sequences.shape[1] - prompt_len)
            generated_ids = sequences[0, prompt_len:].tolist()
            _write_jsonl(
                receipt_path,
                [{
                    "request_id": request_id,
                    "prompt_tokens": prompt_len,
                    "generated_tokens": generated_len,
                    "trace_rows": len(rows),
                    "eos_observed": bool(tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated_ids),
                    "hit_decode_cap": generated_len == args.max_new_tokens,
                    "kv_policy": args.cache_lookup_strategy,
                    "rmax": int(args.max_steps),
                    "status": "OK" if len(rows) == generated_len else "TRACE_LENGTH_MISMATCH",
                    "huginn_private_stop_tokens_suppressed": True,
                }],
                append=True,
            )

    print(f"wrote {output_path}")


def read_traces(paths: list[str]) -> pd.DataFrame:
    expanded: list[str] = []
    for p in paths:
        matched = glob.glob(p)
        expanded.extend(matched if matched else [p])
    rows: list[dict[str, Any]] = []
    for path in expanded:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["source_file"] = os.path.basename(path)
                    rows.append(row)
    if not rows:
        raise ValueError("No trace records found")
    df = pd.DataFrame(rows)
    needed = {"request_id", "token_idx", "depth", "model"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing trace fields: {missing}")
    df["request_id"] = df["request_id"].astype(int)
    df["token_idx"] = df["token_idx"].astype(int)
    df["depth"] = df["depth"].astype(int)
    return df


def add_causal_predictors(df: pd.DataFrame, recent_k: int) -> pd.DataFrame:
    df = df.sort_values(["request_id", "token_idx"]).copy()
    df["prev_pred"] = df.groupby("request_id")["depth"].shift(1)
    df["seqmean_pred"] = (
        df.groupby("request_id")["depth"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["recent_pred"] = (
        df.groupby("request_id")["depth"]
        .apply(lambda s: s.shift(1).rolling(recent_k, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    # Token 0 has no causal history; all such requests get the same key so no information is leaked.
    for c in ("prev_pred", "seqmean_pred", "recent_pred"):
        df[c] = df[c].fillna(0.0)
    return df


def order_token_group(g: pd.DataFrame, strategy: str, rng: random.Random) -> pd.DataFrame:
    if strategy == "fifo":
        return g.sort_values("request_id")
    if strategy == "random":
        idx = list(g.index)
        rng.shuffle(idx)
        return g.loc[idx]
    if strategy == "oracle":
        return g.sort_values(["depth", "request_id"])
    if strategy == "prev":
        return g.sort_values(["prev_pred", "request_id"])
    if strategy == "recent":
        return g.sort_values(["recent_pred", "request_id"])
    if strategy == "seqmean":
        return g.sort_values(["seqmean_pred", "request_id"])
    raise ValueError(strategy)


def simulate_strategy(
    df: pd.DataFrame,
    batch_size: int,
    strategy: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    rng = random.Random(seed + batch_size * 1009 + sum(ord(c) for c in strategy))
    total_work = 0
    total_capacity = 0
    total_idle = 0
    num_batches = 0
    depth_stds = []
    active_by_loop: dict[int, list[float]] = defaultdict(list)
    worst: dict[str, Any] | None = None

    for token_idx, g in df.groupby("token_idx", sort=True):
        ordered = order_token_group(g, strategy, rng)
        n_full = len(ordered) // batch_size
        if n_full == 0:
            continue
        ordered = ordered.iloc[: n_full * batch_size]
        for bi in range(n_full):
            mb = ordered.iloc[bi * batch_size : (bi + 1) * batch_size]
            depths = mb["depth"].to_numpy(dtype=int)
            max_d = int(depths.max())
            work = int(depths.sum())
            capacity = int(batch_size * max_d)
            idle = capacity - work
            util = work / capacity if capacity else 1.0
            total_work += work
            total_capacity += capacity
            total_idle += idle
            num_batches += 1
            depth_stds.append(float(np.std(depths)))
            for r in range(1, max_d + 1):
                active_by_loop[r].append(float(np.mean(depths >= r)))
            if worst is None or (1.0 - util) > worst["idle_ratio"]:
                worst = {
                    "token_idx": int(token_idx),
                    "batch_index": int(bi),
                    "depths": depths.tolist(),
                    "idle_ratio": float(1.0 - util),
                }

    if total_capacity == 0:
        metrics = {
            "batch_size": batch_size,
            "strategy": strategy,
            "slot_utilization": np.nan,
            "idle_slot_ratio": np.nan,
            "num_microbatches": 0,
            "mean_depth_std": np.nan,
            "total_token_loop_work": 0,
            "total_capacity_slots": 0,
        }
    else:
        metrics = {
            "batch_size": batch_size,
            "strategy": strategy,
            "slot_utilization": total_work / total_capacity,
            "idle_slot_ratio": total_idle / total_capacity,
            "num_microbatches": num_batches,
            "mean_depth_std": float(np.mean(depth_stds)),
            "total_token_loop_work": total_work,
            "total_capacity_slots": total_capacity,
        }

    curve = pd.DataFrame(
        [
            {
                "batch_size": batch_size,
                "strategy": strategy,
                "loop_step": r,
                "mean_active_fraction": float(np.mean(vals)),
                "num_microbatch_samples": len(vals),
            }
            for r, vals in sorted(active_by_loop.items())
        ]
    )
    return metrics, curve, worst


def predictor_metrics(df: pd.DataFrame, recent_k: int) -> pd.DataFrame:
    rows = []
    for name, col in (("prev", "prev_pred"), (f"recent{recent_k}", "recent_pred"), ("seqmean", "seqmean_pred")):
        valid = df[df["token_idx"] > 0].copy()
        if valid.empty:
            continue
        y = valid["depth"].to_numpy(float)
        p = valid[col].to_numpy(float)
        corr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan
        pred_int = np.rint(p).astype(int)
        rows.append(
            {
                "predictor": name,
                "n": len(valid),
                "mae": float(np.mean(np.abs(y - p))),
                "exact_accuracy": float(np.mean(pred_int == y.astype(int))),
                "pearson": corr,
            }
        )
    return pd.DataFrame(rows)


def plot_run(run_df: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = [s for s in ("fifo", "random", "prev", "recent", "seqmean", "oracle") if s in set(summary.strategy)]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for strategy in strategies:
        s = summary[summary.strategy == strategy].sort_values("batch_size")
        ax.plot(s.batch_size, s.slot_utilization, marker="o", label=strategy)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Micro-batch size")
    ax.set_ylabel("Slot utilization")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "slot_utilization.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for strategy in strategies:
        s = summary[summary.strategy == strategy].sort_values("batch_size")
        ax.plot(s.batch_size, s.idle_slot_ratio, marker="o", label=strategy)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Micro-batch size")
    ax.set_ylabel("Idle-slot ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "idle_slot_ratio.png", dpi=180)
    plt.close(fig)

    # Valley curves at representative batch sizes.
    for b in [16, 32, 64, 128, 256]:
        c = curves[curves.batch_size == b]
        if c.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for strategy in [s for s in ("fifo", "prev", "recent", "seqmean", "oracle") if s in set(c.strategy)]:
            x = c[c.strategy == strategy].sort_values("loop_step")
            ax.plot(x.loop_step, x.mean_active_fraction, marker="o", label=strategy)
        ax.set_xlabel("Recurrent loop step within one token interval")
        ax.set_ylabel("Mean active fraction of a live micro-batch")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"valley_example_b{b}.png", dpi=180)
        plt.close(fig)

    # Trace heatmap: request x token depth for a small subset.
    pivot = run_df.pivot_table(index="request_id", columns="token_idx", values="depth", aggfunc="first")
    pivot = pivot.iloc[: min(48, len(pivot)), : min(64, pivot.shape[1])]
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
        ax.set_xlabel("Generated token index")
        ax.set_ylabel("Request")
        fig.colorbar(im, ax=ax, label="Exit depth")
        fig.tight_layout()
        fig.savefig(out_dir / "depth_trace_heatmap.png", dpi=180)
        plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)
    df = read_traces(args.traces)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # source_file separates different thresholds/configurations even if model+dataset match.
    run_cols = ["source_file", "model"]
    all_summary = []
    all_curves = []
    all_predict = []

    for run_key, run_df0 in df.groupby(run_cols, dropna=False):
        source_file, model = run_key
        run_df = add_causal_predictors(run_df0, args.recent_k)
        run_name = _slug(f"{Path(str(source_file)).stem}__{model}")
        run_out = output_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        pred = predictor_metrics(run_df, args.recent_k)
        pred.insert(0, "run", run_name)
        pred.to_csv(run_out / "predictability.csv", index=False)
        all_predict.append(pred)

        summaries = []
        curves = []
        worst_examples: dict[str, Any] = {}
        for b in batch_sizes:
            for strategy in strategies:
                metrics, curve, worst = simulate_strategy(run_df, b, strategy, args.seed)
                metrics.update({"run": run_name, "model": model, "source_file": source_file})
                summaries.append(metrics)
                if not curve.empty:
                    curve.insert(0, "run", run_name)
                    curve["model"] = model
                    curves.append(curve)
                if worst is not None and strategy in {"fifo", "oracle"}:
                    worst_examples[f"{strategy}_b{b}"] = worst

        summary = pd.DataFrame(summaries)
        curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

        # Explicit oracle headroom over FIFO for each B.
        fifo = summary[summary.strategy == "fifo"].set_index("batch_size")["slot_utilization"]
        oracle = summary[summary.strategy == "oracle"].set_index("batch_size")["slot_utilization"]
        headroom = (oracle - fifo).rename("oracle_headroom_pp") * 100.0
        summary = summary.merge(headroom.reset_index(), on="batch_size", how="left")

        summary.to_csv(run_out / "summary.csv", index=False)
        if not curve_df.empty:
            curve_df.to_csv(run_out / "valley_curves.csv", index=False)
        with (run_out / "worst_examples.json").open("w", encoding="utf-8") as f:
            json.dump(worst_examples, f, indent=2)
        plot_run(run_df, summary, curve_df, run_out)

        all_summary.append(summary)
        if not curve_df.empty:
            all_curves.append(curve_df)

    pd.concat(all_summary, ignore_index=True).to_csv(output_root / "summary.csv", index=False)
    if all_predict:
        pd.concat(all_predict, ignore_index=True).to_csv(output_root / "predictability.csv", index=False)
    if all_curves:
        pd.concat(all_curves, ignore_index=True).to_csv(output_root / "valley_curves.csv", index=False)

    # Console go/no-go table.
    summary_all = pd.concat(all_summary, ignore_index=True)
    cols = ["run", "batch_size", "strategy", "slot_utilization", "idle_slot_ratio", "oracle_headroom_pp"]
    print(summary_all[cols].to_string(index=False))
    print(f"\nwrote analysis to {output_root}")


def synthetic(args: argparse.Namespace) -> None:
    """Generate a tiny trace where depth-aware grouping has known large headroom."""
    rows = []
    depths = [2, 8, 3, 7, 2, 8, 3, 7] * 8
    for t in range(args.tokens):
        rot = t % 4
        for rid, d in enumerate(depths):
            rows.append(
                {
                    "model": "synthetic-loop",
                    "family": "synthetic",
                    "dataset": "synthetic",
                    "request_id": rid,
                    "token_idx": t,
                    "depth": max(1, min(8, d + ((rid + rot) % 3 - 1))),
                    "max_depth": 8,
                    "exit_rule": "synthetic",
                    "exit_threshold": 0,
                }
            )
    _write_jsonl(args.output, rows)
    print(f"wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common_collect = argparse.ArgumentParser(add_help=False)
    common_collect.add_argument("--model", required=True)
    common_collect.add_argument("--dataset", default="alpaca", choices=["alpaca", "sharegpt"])
    common_collect.add_argument("--prompts-jsonl", default=None)
    common_collect.add_argument("--num-requests", type=int, default=64)
    common_collect.add_argument("--max-new-tokens", type=int, default=32)
    common_collect.add_argument("--max-prompt-tokens", type=int, default=512)
    common_collect.add_argument("--seed", type=int, default=1)
    common_collect.add_argument("--output", required=True)
    common_collect.add_argument("--append", action="store_true")
    common_collect.add_argument(
        "--start-request-id",
        type=int,
        default=0,
        help="Resume from this deterministic shuffled-prompt index; use with --append.",
    )

    po = sub.add_parser("collect-ouro", parents=[common_collect])
    po.add_argument("--exit-threshold", type=float, default=0.2)
    po.set_defaults(func=collect_ouro)

    ph = sub.add_parser("collect-huginn", parents=[common_collect])
    ph.add_argument("--max-steps", type=int, default=16)
    ph.add_argument("--criterion", default="latent-diff", choices=["latent-diff", "cosine", "entropy-diff", "kl", "minp-kl", "argmax-stability"])
    ph.add_argument("--exit-threshold", default="0.12")
    ph.add_argument("--min-steps", type=int, default=1, help="0-based minimum check step; 1 => earliest exit after 2 recurrences")
    ph.add_argument("--check-every", type=int, default=1)
    ph.add_argument("--cache-lookup-strategy", default="full")
    ph.set_defaults(func=collect_huginn)

    pa = sub.add_parser("analyze")
    pa.add_argument("--traces", nargs="+", required=True)
    pa.add_argument("--batch-sizes", default="4,8,16,32,64,128,256")
    pa.add_argument("--strategies", default="fifo,random,prev,recent,seqmean,oracle")
    pa.add_argument("--recent-k", type=int, default=4)
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--output-dir", required=True)
    pa.set_defaults(func=analyze)

    ps = sub.add_parser("synthetic")
    ps.add_argument("--tokens", type=int, default=16)
    ps.add_argument("--output", default="results/traces/synthetic.jsonl")
    ps.set_defaults(func=synthetic)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
