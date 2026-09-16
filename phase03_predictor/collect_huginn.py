"""Collect frozen causal Huginn features and B=1 shared-KV live depth labels.

Only physical GPU7 is permitted. Every feature is materialized before the
current recurrent interval; current depth is written to a separate label file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phase02_runtime.benchmark import _load_prompts
from phase02_runtime.huginn_adapter import HuginnAdapter


SCALARS = (
    "token_position", "context_length", "previous_depth", "previous_available",
    "depth_mean_2", "depth_std_2", "depth_min_2", "depth_max_2",
    "depth_mean_4", "depth_std_4", "depth_min_4", "depth_max_4",
    "depth_mean_8", "depth_std_8", "depth_min_8", "depth_max_8",
    "depth_trend", "sampled_surprisal", "entropy", "top1_probability",
    "top1_top2_margin", "previous_trajectory_length",
) + tuple(f"previous_convergence_{step + 1}" for step in range(16))


def gpu_identity() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "7" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Huginn Phase-3 collector requires CUDA_VISIBLE_DEVICES=7 and one CUDA device")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", "7", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 4 or fields[0] != "7" or "H100" not in fields[2]:
        raise RuntimeError(f"physical GPU7 identity mismatch: {query}")
    return {"physical_index": 7, "uuid": fields[1], "name": fields[2],
            "memory_total": fields[3], "CUDA_VISIBLE_DEVICES": visible, "cuda_ordinal": 0}


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_features(history: list[int]) -> list[float]:
    output: list[float] = []
    for width in (2, 4, 8):
        recent = np.asarray(history[-width:], dtype=np.float32)
        output.extend((float(recent.mean()), float(recent.std()),
                       float(recent.min()), float(recent.max())) if len(recent) else (0.0,) * 4)
    return output


def _uncertainty(logits: torch.Tensor, token: torch.Tensor) -> list[float]:
    values = logits.float().flatten()
    log_z = torch.logsumexp(values, dim=0)
    probs = torch.softmax(values, dim=0)
    top = torch.topk(probs, 2).values
    entropy = log_z - (probs * values).sum()
    return [float(log_z - values[int(token.item())]), float(entropy),
            float(top[0]), float(top[0] - top[1])]


@torch.inference_mode()
def collect(args: argparse.Namespace) -> None:
    identity = gpu_identity()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prompts = _load_prompts(args.dataset, args.num_rows, args.seed, heterogeneous=True)
    prompt_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    prompt_digest = hashlib.sha256()
    with gzip.open(args.output_dir / "prompts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index, prompt in enumerate(prompts):
            prompt_digest.update(f"{index}:{prompt}\n".encode())
            stream.write(json.dumps({"request_id": index, "prompt": prompt,
                                     "sha256": prompt_hashes[index]}, ensure_ascii=False) + "\n")
    split = {"train": list(range(0, args.train_rows)),
             "validation": list(range(args.train_rows, args.train_rows + args.validation_rows)),
             "test": list(range(args.train_rows + args.validation_rows, args.num_rows))}
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    initial = HuginnAdapter.load(args.num_rows, state_init="native-seeded")
    model, tokenizer = initial.model, initial.tokenizer
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = encoded["input_ids"].cuda()
    prompt_valid = encoded["attention_mask"].cuda().bool()
    hidden_size = int(model.transformer.wte.weight.shape[1])
    torch.manual_seed(args.seed)
    projection = torch.randn(hidden_size, args.projection_dim, device="cuda", dtype=torch.float32)
    projection /= hidden_size ** 0.5
    np.save(args.output_dir / "projection.npy", projection.cpu().numpy())
    manifest = {
        "execution_kind": "exit-trace-collection", "status": "RUNNING",
        "source_sha": source_sha, "argv": sys.argv, "gpu": identity,
        "model_revision": HuginnAdapter.model_revision, "model": "huginn-3.5b",
        "dataset": args.dataset, "dataset_revision": "UNKNOWN (local cached Hugging Face dataset)",
        "prompt_sha256": prompt_digest.hexdigest(), "prompt_hashes": prompt_hashes,
        "prompt_cap": 64, "tokenized_prompt_width": int(input_ids.shape[1]),
        "prompt_token_counts": prompt_valid.sum(dim=1).cpu().tolist(),
        "request_split": split, "split_unit": "request", "seed": args.seed,
        "threshold": args.threshold, "num_rows": args.num_rows,
        "num_tokens": args.num_tokens, "rmax": initial.rmax,
        "kv_backend": initial.kv_policy, "state_init": "native-seeded",
        "feature_schema_version": 1, "scalar_names": SCALARS,
        "embedding_projection": "frozen seeded Gaussian 1/sqrt(hidden), shared with Prelude",
        "projection_dim": args.projection_dim,
        "feature_causality": "all features written before current-token recurrence; label separate",
        "completed_token_boundaries": 0,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shape = (args.num_tokens, args.num_rows)
    scalar_arr = np.lib.format.open_memmap(args.output_dir / "scalars.npy", mode="w+", dtype="float32",
                                           shape=(*shape, len(SCALARS)))
    embedding_arr = np.lib.format.open_memmap(args.output_dir / "embedding.npy", mode="w+", dtype="float16",
                                              shape=(*shape, args.projection_dim))
    prelude_arr = np.lib.format.open_memmap(args.output_dir / "prelude.npy", mode="w+", dtype="float16",
                                            shape=(*shape, args.projection_dim))
    tokens_arr = np.lib.format.open_memmap(args.output_dir / "token_ids.npy", mode="w+", dtype="int32", shape=shape)
    labels_arr = np.lib.format.open_memmap(args.output_dir / "depth_labels.npy", mode="w+", dtype="int16", shape=shape)
    adapter = HuginnAdapter(model, tokenizer, args.num_rows, state_init="native-seeded")
    current_tokens = adapter.prefill(input_ids, prompt_valid)
    previous_logits = adapter.prefill_logits[:, -1, :]
    history: list[list[int]] = [[] for _ in range(args.num_rows)]
    previous_trajectories: list[list[float]] = [[] for _ in range(args.num_rows)]
    began = time.perf_counter()
    for token_idx in range(args.num_tokens):
        adapter.begin_token()
        next_tokens = torch.empty_like(current_tokens)
        next_logits = torch.empty_like(previous_logits)
        for request_id in range(args.num_rows):
            row = torch.tensor([request_id], dtype=torch.long, device=adapter.device)
            adapter.backend.set_active(row)
            token = current_tokens.index_select(0, row)
            embedded, state = adapter.prelude(token)
            current_embedding = model.transformer.wte(token)
            if model.emb_scale != 1:
                current_embedding = current_embedding * model.emb_scale
            embedded_proj = current_embedding.float().flatten() @ projection
            prelude_proj = embedded.float().flatten() @ projection
            prev_depths = history[request_id]
            trajectory = previous_trajectories[request_id]
            prompt_length = int(manifest["prompt_token_counts"][request_id])
            scalar = [float(token_idx), float(prompt_length + token_idx),
                      float(prev_depths[-1]) if prev_depths else 0.0,
                      float(bool(prev_depths))]
            scalar += _history_features(prev_depths)
            scalar += [float(prev_depths[-1] - prev_depths[-2]) if len(prev_depths) > 1 else 0.0]
            scalar += _uncertainty(previous_logits[request_id], token)
            scalar += [float(len(trajectory))]
            scalar += trajectory[:16] + [0.0] * (16 - len(trajectory))
            if len(scalar) != len(SCALARS):
                raise AssertionError("scalar schema mismatch")
            scalar_arr[token_idx, request_id] = scalar
            embedding_arr[token_idx, request_id] = embedded_proj.cpu().numpy().astype(np.float16)
            prelude_arr[token_idx, request_id] = prelude_proj.cpu().numpy().astype(np.float16)
            tokens_arr[token_idx, request_id] = int(token.item())
            current_trajectory: list[float] = []
            for recurrence in range(1, adapter.rmax + 1):
                after = adapter.recurrent_step(state, embedded, recurrence)
                ratio = ((after.float() - state.float()).norm(dim=-1) /
                         after.float().norm(dim=-1).clamp_min(1e-12)).mean()
                current_trajectory.append(float(ratio.item()))
                state = after
                if recurrence == adapter.rmax or current_trajectory[-1] < args.threshold:
                    break
            adapter.backend.on_exit(row, recurrence)
            logits = adapter.coda_logits(state)
            next_logits.index_copy_(0, row, logits[:, -1, :])
            next_tokens.index_copy_(0, row, logits.argmax(dim=-1))
            labels_arr[token_idx, request_id] = recurrence
            history[request_id].append(recurrence)
            previous_trajectories[request_id] = current_trajectory
        adapter.commit_token()
        current_tokens = next_tokens
        previous_logits = next_logits
        manifest["completed_token_boundaries"] = token_idx + 1
        if (token_idx + 1) % 4 == 0 or token_idx + 1 == args.num_tokens:
            for array in (scalar_arr, embedding_arr, prelude_arr, tokens_arr, labels_arr):
                array.flush()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"completed={token_idx + 1}/{args.num_tokens} elapsed_s={time.perf_counter() - began:.1f}",
                  flush=True)
    manifest["status"] = "COMPLETE"
    manifest["elapsed_wall_s"] = time.perf_counter() - began
    manifest["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    manifest["artifact_sha256"] = {name: _digest_file(args.output_dir / name) for name in
                                    ("prompts.jsonl.gz", "projection.npy", "scalars.npy",
                                     "embedding.npy", "prelude.npy", "token_ids.npy", "depth_labels.npy")}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("alpaca", "sharegpt"), default="alpaca")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--test-rows", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.train_rows, args.validation_rows, args.test_rows, args.num_tokens, args.projection_dim) < 1:
        parser.error("all counts must be positive")
    args.num_rows = args.train_rows + args.validation_rows + args.test_rows
    collect(args)


if __name__ == "__main__":
    main()
