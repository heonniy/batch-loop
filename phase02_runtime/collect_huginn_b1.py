"""B=1 shared-KV Huginn live exit traces without timing instrumentation.

This collector uses the same model adapter, exit criterion, token-boundary KV
commit, and fixed decode horizon as the physical benchmark. It deliberately
does not report latency: its sole output is an exit-depth work-control trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from phase02_runtime.benchmark import _load_prompts
from phase02_runtime.huginn_adapter import HuginnAdapter


def _gpu_identity() -> str:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7" or torch.cuda.device_count() != 1:
        raise RuntimeError("Huginn B1 trace collector requires only physical GPU7")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", "7", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not query.startswith("7,"):
        raise RuntimeError("physical GPU7 identity mismatch")
    return query


@torch.inference_mode()
def collect_one(adapter: HuginnAdapter, input_ids: torch.Tensor, prompt_valid: torch.Tensor,
                num_tokens: int, threshold: float, trace_path: Path) -> dict:
    current_tokens = adapter.prefill(input_ids, prompt_valid)
    device = adapter.device
    n = input_ids.shape[0]
    histogram: Counter[int] = Counter()
    boundary_means: list[float] = []
    start = time.perf_counter()
    with trace_path.open("w", encoding="utf-8") as stream:
        for token_idx in range(num_tokens):
            adapter.begin_token()
            next_tokens = torch.empty_like(current_tokens)
            depths = []
            for request_id in range(n):
                row = torch.tensor([request_id], dtype=torch.long, device=device)
                adapter.backend.set_active(row)
                embedded, state = adapter.prelude(current_tokens.index_select(0, row))
                exit_depth = 0
                for recurrence in range(1, adapter.rmax + 1):
                    after = adapter.recurrent_step(state, embedded, recurrence)
                    should_exit = adapter.live_exit(state, after, row, recurrence, threshold)
                    state = after
                    if bool(should_exit.item()):
                        exit_depth = recurrence
                        break
                if not 1 <= exit_depth <= adapter.rmax:
                    raise AssertionError("B1 request never exited")
                adapter.backend.on_exit(row, exit_depth)
                next_tokens.index_copy_(0, row, adapter.coda(state))
                histogram[exit_depth] += 1
                depths.append(exit_depth)
                stream.write(json.dumps({
                    "request_id": request_id, "token_idx": token_idx, "depth": exit_depth,
                    "model": "huginn", "threshold": threshold,
                    "exit_source": "live-b1-shared-kv",
                }) + "\n")
            adapter.commit_token()
            current_tokens = next_tokens
            boundary_means.append(statistics.mean(depths))
            if (token_idx + 1) % 8 == 0 or token_idx + 1 == num_tokens:
                stream.flush()
                print(f"threshold={threshold:g} token_boundaries={token_idx + 1}/{num_tokens} "
                      f"mean_depth_last={boundary_means[-1]:.2f} elapsed_s={time.perf_counter() - start:.1f}",
                      flush=True)
    count = sum(histogram.values())
    if count != n * num_tokens:
        raise AssertionError("trace record count mismatch")
    return {
        "threshold": threshold, "request_count": n, "decode_tokens": num_tokens,
        "trace_rows": count, "depth_histogram": dict(sorted(histogram.items())),
        "mean_depth": sum(depth * freq for depth, freq in histogram.items()) / count,
        "boundary_mean_depth_min": min(boundary_means),
        "boundary_mean_depth_max": max(boundary_means),
        "elapsed_wall_s": time.perf_counter() - start,
        "trace_file": trace_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--num-rows", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--dataset", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_rows < 1 or args.num_tokens < 1 or len(set(args.thresholds)) != len(args.thresholds):
        parser.error("positive rows/tokens and unique thresholds required")
    gpu = _gpu_identity()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    prompts = _load_prompts(args.dataset, args.num_rows, args.seed, heterogeneous=True)
    digest = hashlib.sha256()
    for request_id, prompt in enumerate(prompts):
        digest.update(f"{request_id}:{prompt}\n".encode())
    manifest = {
        "execution_kind": "exit-trace-collection", "status": "RUNNING",
        "source_sha": source_sha, "gpu": gpu,
        "model_revision": HuginnAdapter.model_revision,
        "kv_backend": HuginnAdapter.kv_policy,
        "exit_rule": "latent-diff convergence, B1 shared-KV",
        "Rmax": HuginnAdapter.rmax, "dataset": args.dataset,
        "dataset_revision": "UNKNOWN (local cached Hugging Face dataset)",
        "prompt_sha256": digest.hexdigest(), "seed": args.seed,
        "prompt_max_tokens": 64, "state_init": "native-seeded",
        "num_rows": args.num_rows, "num_tokens": args.num_tokens,
        "thresholds_requested": args.thresholds, "completed": [],
        "not_a_timing_result": True,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    initial = HuginnAdapter.load(args.num_rows, state_init="native-seeded")
    model, tokenizer = initial.model, initial.tokenizer
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = encoded["input_ids"].cuda()
    prompt_valid = encoded["attention_mask"].cuda().bool()
    manifest["tokenized_prompt_width"] = int(input_ids.shape[1])
    manifest["prompt_token_counts"] = prompt_valid.sum(dim=1).cpu().tolist()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for threshold in args.thresholds:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        adapter = HuginnAdapter(model, tokenizer, args.num_rows, state_init="native-seeded")
        name = f"huginn_{args.dataset}_threshold_{str(threshold).replace('.', 'p')}_b1.jsonl"
        summary = collect_one(adapter, input_ids, prompt_valid, args.num_tokens, threshold,
                              args.output_dir / name)
        manifest["completed"].append(summary)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["status"] = "COMPLETE"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
