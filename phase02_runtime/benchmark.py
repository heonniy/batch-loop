#!/usr/bin/env python3
"""Bounded physical no-refill M0/M1/M2 runner, restricted to GPUs 6 and 7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from phase02_runtime.common import run_interval, timed
from phase02_runtime.huginn_adapter import HuginnAdapter
from phase02_runtime.ouro_adapter import OuroAdapter


def _gpu_identity(model_key: str) -> dict:
    physical = 6 if model_key == "ouro" else 7
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical):
        raise RuntimeError(f"{model_key} requires only physical GPU {physical}; got CUDA_VISIBLE_DEVICES={visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("runner must expose exactly one CUDA device")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical),
         "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    if int(query.split(",", 1)[0]) != physical:
        raise RuntimeError("nvidia-smi physical GPU binding mismatch")
    return {"physical_gpu": physical, "visible_device": 0, "gpu_query": query}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trace(path: Path, num_rows: int, num_tokens: int) -> list[list[int]]:
    matrix = [[0] * num_tokens for _ in range(num_rows)]
    seen = set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            req, token = int(row["request_id"]), int(row["token_idx"])
            if req >= num_rows or token >= num_tokens:
                continue
            if (req, token) in seen:
                raise ValueError(f"duplicate trace record {(req, token)}")
            seen.add((req, token))
            matrix[req][token] = int(row["depth"])
    if len(seen) != num_rows * num_tokens:
        raise ValueError(f"trace incomplete: {len(seen)} / {num_rows * num_tokens}")
    return matrix


def _synthetic_depths(model_key: str, num_rows: int, num_tokens: int, seed: int) -> list[list[int]]:
    rmax = 4 if model_key == "ouro" else 16
    choices = [1, 2, 3, 4] if model_key == "ouro" else [1, 2, 4, 8, 12, 16]
    rng = random.Random(seed)
    matrix = [[choices[(row + token) % len(choices)] for token in range(num_tokens)] for row in range(num_rows)]
    rng.shuffle(matrix)
    assert all(1 <= d <= rmax for row in matrix for d in row)
    return matrix


def _prompts(num_rows: int, heterogeneous: bool) -> list[str]:
    base = "Describe the next step of this short reasoning task. "
    return [f"Request {i}: " + base * (1 + (i % 4 if heterogeneous else 0)) for i in range(num_rows)]


def _strategy_batches(depths: list[int], batch_size: int, strategy: str) -> list[list[int]]:
    order = list(range(len(depths)))
    if strategy == "oracle":
        order.sort(key=lambda row: (depths[row], row))
    elif strategy not in {"fifo", "full"}:
        raise ValueError(strategy)
    if len(order) % batch_size:
        raise ValueError("M1/M2 require complete microbatches")
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def _new_adapter(model_key: str, loaded, tokenizer, num_rows: int, state_init: str):
    return (
        OuroAdapter(loaded, tokenizer, num_rows) if model_key == "ouro"
        else HuginnAdapter(loaded, tokenizer, num_rows, state_init=state_init)
    )


def _run_once(args, loaded, tokenizer, input_ids, prompt_valid, depths_matrix,
              strategy: str, repetition: int, run_kind: str) -> tuple[dict, list[dict]]:
    adapter = _new_adapter(args.model, loaded, tokenizer, args.num_rows, args.state_init)
    device = adapter.device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    prefill_next, prefill_gpu_ms, prefill_wall_ms = timed(
        device, lambda: adapter.prefill(input_ids, prompt_valid)
    )
    current_tokens = prefill_next
    events = []
    total_core = total_gate = total_compaction = total_kv = total_exit_kv = 0.0
    total_prelude = total_coda = total_wall = 0.0
    total_work = 0
    num_intervals = 0
    start_run = time.perf_counter()

    for token_idx in range(args.num_tokens):
        adapter.begin_token()
        depths = [depths_matrix[row][token_idx] for row in range(args.num_rows)]
        if strategy == "full":
            depths = [adapter.rmax] * args.num_rows
        batches = _strategy_batches(depths, args.batch_size, strategy)
        next_tokens = torch.empty_like(current_tokens)
        for microbatch_id, row_ids in enumerate(batches):
            rows = torch.tensor(row_ids, dtype=torch.long, device=device)
            adapter.backend.set_active(rows)
            tokens = current_tokens.index_select(0, rows)
            (embedded, state), prelude_gpu_ms, prelude_wall_ms = timed(
                device, lambda: adapter.prelude(tokens)
            )
            prelude_kv_ms = adapter.backend.pop_update_gpu_ms()
            forced = torch.tensor(depths, dtype=torch.long, device=device).index_select(0, rows)
            result = run_interval(
                adapter, embedded, state, rows, rows,
                exit_source="live" if args.exit_source == "live" else "trace",
                forced_depths=None if args.exit_source == "live" else forced,
                threshold=args.threshold, debug_ids=args.debug_ids,
            )
            adapter.backend.set_active(rows)
            out_tokens, coda_gpu_ms, coda_wall_ms = timed(
                device, lambda: adapter.coda(result.final_state)
            )
            coda_kv_ms = adapter.backend.pop_update_gpu_ms()
            next_tokens.index_copy_(0, rows, out_tokens)
            work = int(result.exit_depths.sum().item())
            total_work += work
            num_intervals += 1
            total_core += result.core_gpu_ms
            total_gate += result.gate_gpu_ms
            total_compaction += result.compaction_gpu_ms
            total_kv += result.kv_gpu_ms + prelude_kv_ms + coda_kv_ms
            total_exit_kv += result.kv_exit_gpu_ms
            total_prelude += prelude_gpu_ms
            total_coda += coda_gpu_ms
            total_wall += prelude_wall_ms + result.wall_ms + coda_wall_ms
            for event in result.events:
                events.append({
                    "run_kind": run_kind, "strategy": strategy, "repetition": repetition,
                    "token_idx": token_idx, "microbatch_id": microbatch_id,
                    "exit_source": args.exit_source, "model": args.model,
                    "kv_backend": adapter.kv_policy, "initial_B": len(row_ids),
                    "prelude_gpu_ms": prelude_gpu_ms, "coda_gpu_ms": coda_gpu_ms,
                    **event,
                })
        adapter.commit_token()
        current_tokens = next_tokens

    torch.cuda.synchronize(device)
    elapsed_wall_ms = (time.perf_counter() - start_run) * 1000
    generated = args.num_rows * args.num_tokens
    summary = {
        "run_kind": run_kind, "strategy": strategy, "repetition": repetition,
        "model": args.model, "exit_source": args.exit_source,
        "kv_backend": adapter.kv_policy, "threshold": args.threshold,
        "num_rows": args.num_rows, "batch_size": args.batch_size,
        "num_tokens": args.num_tokens, "generated_tokens": generated,
        "num_intervals": num_intervals, "total_token_loop_work": total_work,
        "prefill_gpu_ms_excluded": prefill_gpu_ms,
        "prefill_wall_ms_excluded": prefill_wall_ms,
        "recurrent_pass_gpu_ms_including_kv": total_core,
        "gate_gpu_ms": total_gate, "compaction_gpu_ms": total_compaction,
        "kv_update_gpu_ms_within_stages": total_kv,
        "exit_kv_gpu_ms": total_exit_kv,
        "prelude_gpu_ms": total_prelude, "coda_gpu_ms": total_coda,
        "instrumented_stage_wall_ms": total_wall,
        "total_decode_wall_ms": elapsed_wall_ms,
        "generated_tokens_per_decode_wall_s": generated * 1000 / elapsed_wall_ms,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "actual_kv_writes": adapter.backend.actual_writes,
        "last_exited_fill_operations": adapter.backend.fill_operations,
        "shared_kv_overwrites": adapter.backend.overwrites,
    }
    if args.model == "ouro" and run_kind == "m0":
        summary["last_exited_fill_receipts"] = adapter.backend.fill_receipts
    return summary, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["ouro", "huginn"])
    parser.add_argument("--stage", required=True, choices=["m0", "m1", "m2", "live"])
    parser.add_argument("--num-rows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--exit-source", choices=["trace", "live"], default="trace")
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--state-init", choices=["zero", "native-seeded"], default="zero")
    parser.add_argument("--heterogeneous-prompts", action="store_true")
    parser.add_argument("--debug-ids", action="store_true")
    parser.add_argument("--emit-live-trace", action="store_true",
                        help="write measured B=1 live exit depths for later forced-depth timing")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_rows < 1 or args.batch_size < 1 or args.num_tokens < 1 or args.repeats < 1:
        parser.error("rows, B, tokens and repeats must be positive")
    if args.num_rows % args.batch_size:
        parser.error("num-rows must be divisible by batch-size")
    if args.stage == "m0" and (args.num_rows, args.batch_size, args.num_tokens) != (4, 4, 1):
        parser.error("M0 is fixed to 4 rows, B4, one token")
    if args.stage == "live" and args.exit_source != "live":
        parser.error("live stage requires --exit-source live")
    if args.exit_source == "live" and args.stage not in {"live", "m0"}:
        parser.error("live-exit timing must be a separately labelled stage")
    if args.emit_live_trace and (args.stage != "live" or args.batch_size != 1 or
                                 args.repeats != 1 or not args.debug_ids):
        parser.error("--emit-live-trace requires live stage, B=1, one repeat, and --debug-ids")
    if args.model == "huginn" and args.stage == "m1" and args.trace_file is None:
        print("[scope] Huginn M1 forced depths are synthetic until B1 shared-KV trace exists", flush=True)

    identity = _gpu_identity(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    trace_info = (
        {"path": str(args.trace_file.resolve()), "sha256": _sha256(args.trace_file)}
        if args.trace_file else None
    )
    manifest = {
        "source_sha": source_sha, "argv": sys.argv, "gpu": identity,
        "model": args.model,
        "model_revision": OuroAdapter.model_revision if args.model == "ouro" else HuginnAdapter.model_revision,
        "kv_backend": OuroAdapter.kv_policy if args.model == "ouro" else HuginnAdapter.kv_policy,
        "exit_source": args.exit_source,
        "forced_depth_source": (
            "not-applicable-live" if args.exit_source == "live"
            else "provided-trace" if trace_info else "deterministic-synthetic"
        ),
        "trace": trace_info, "threshold": args.threshold, "seed": args.seed,
        "state_init": args.state_init, "prompt_source": "deterministic-synthetic-text",
        "heterogeneous_prompts": args.heterogeneous_prompts,
        "num_rows": args.num_rows, "batch_size": args.batch_size,
        "num_tokens": args.num_tokens, "repeats": args.repeats, "warmups": args.warmups,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "evidence_kind": "PHYSICAL_SURVIVOR_COMPACTION_NOT_QUALITY_EVALUATION",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.trace_file:
        depths_matrix = _load_trace(args.trace_file, args.num_rows, args.num_tokens)
    elif args.stage == "m0":
        depths_matrix = [[d] for d in [1, 2, 4 if args.model == "ouro" else 16, 2]]
    else:
        depths_matrix = _synthetic_depths(args.model, args.num_rows, args.num_tokens, args.seed)

    adapter_cls = OuroAdapter if args.model == "ouro" else HuginnAdapter
    initial = adapter_cls.load(args.num_rows, state_init=args.state_init) if args.model == "huginn" else adapter_cls.load(args.num_rows)
    loaded, tokenizer = initial.model, initial.tokenizer
    prompts = _prompts(args.num_rows, args.heterogeneous_prompts)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to("cuda")
    prompt_valid = encoded["attention_mask"].to("cuda").bool()
    manifest["tokenized_prompt_width"] = int(input_ids.shape[1])
    manifest["prompt_token_counts"] = prompt_valid.sum(dim=1).cpu().tolist()
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    strategies = ["fifo"] if args.stage in {"m0", "live"} else ["fifo", "oracle"]
    if args.stage == "m0":
        strategies = ["fifo"]
    summaries = []
    live_trace: dict[tuple[int, int], int] = {}
    with (args.output_dir / "recurrence_events.jsonl").open("w") as event_file:
        for repetition in range(-args.warmups, args.repeats):
            for strategy in strategies:
                kind = "warmup" if repetition < 0 else args.stage
                summary, events = _run_once(
                    args, loaded, tokenizer, input_ids, prompt_valid,
                    depths_matrix, strategy, repetition, kind,
                )
                print(json.dumps({k: v for k, v in summary.items() if k != "last_exited_fill_receipts"}), flush=True)
                if repetition >= 0:
                    summaries.append(summary)
                    for event in events:
                        event_file.write(json.dumps(event) + "\n")
                        if args.emit_live_trace:
                            for request_id in event["exited_request_ids"]:
                                key = (int(request_id), int(event["token_idx"]))
                                if key in live_trace:
                                    raise AssertionError(f"duplicate live exit {key}")
                                live_trace[key] = int(event["recurrence"])

    if args.emit_live_trace:
        if len(live_trace) != args.num_rows * args.num_tokens:
            raise AssertionError("live trace has missing request-token exits")
        with (args.output_dir / "live_exit_depth_trace.jsonl").open("w") as stream:
            for (request_id, token_idx), depth in sorted(live_trace.items()):
                stream.write(json.dumps({
                    "request_id": request_id, "token_idx": token_idx, "depth": depth,
                    "model": args.model, "threshold": args.threshold,
                    "exit_source": "live-b1-shared-kv" if args.model == "huginn" else "live-b1-last-exited-kv",
                    "source_sha": source_sha,
                }) + "\n")

    if args.exit_source == "trace" and len(strategies) == 2:
        for repetition in range(args.repeats):
            pair = [row for row in summaries if row["repetition"] == repetition]
            if len(pair) != 2 or len({row["total_token_loop_work"] for row in pair}) != 1:
                raise AssertionError("FIFO/Oracle did not execute identical token-loop work")
    fields = [key for key in summaries[0] if key != "last_exited_fill_receipts"]
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)
    if args.stage == "m0" and args.model == "ouro":
        (args.output_dir / "ouro_kv_fill_receipts.json").write_text(
            json.dumps(summaries[0].get("last_exited_fill_receipts", []), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
