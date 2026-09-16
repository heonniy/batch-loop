"""Collect self-consistent Ouro B=1 gate labels and causal pre-recurrence features.

The user explicitly corrected Phase-3 ownership to physical GPU6=Ouro and
GPU7=Huginn. The repository's GPU8 instruction is overridden for this run;
the manifest records that provenance. Chunking bounds depth-indexed KV HBM.
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
from phase02_runtime.ouro_adapter import OuroAdapter
from phase03_predictor.collect_huginn import SCALARS, _digest_file, _history_features, _uncertainty


def gpu_identity() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "6" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Ouro Phase-3 collector requires CUDA_VISIBLE_DEVICES=6 and one CUDA device")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", "6", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 4 or fields[0] != "6" or "H100" not in fields[2]:
        raise RuntimeError(f"physical GPU6 identity mismatch: {query}")
    return {"physical_index": 6, "uuid": fields[1], "name": fields[2],
            "memory_total": fields[3], "CUDA_VISIBLE_DEVICES": visible, "cuda_ordinal": 0,
            "ownership_override": "user correction in current task: GPU6=Ouro, GPU7=Huginn"}


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
    initial = OuroAdapter.load(min(args.chunk_size, args.num_rows))
    model, tokenizer = initial.model, initial.tokenizer
    hidden_size = int(model.model.embed_tokens.weight.shape[1])
    torch.manual_seed(args.seed)
    projection = torch.randn(hidden_size, args.projection_dim, device="cuda", dtype=torch.float32)
    projection /= hidden_size ** 0.5
    np.save(args.output_dir / "projection.npy", projection.cpu().numpy())
    manifest = {
        "execution_kind": "exit-trace-collection", "status": "RUNNING",
        "source_sha": source_sha, "argv": sys.argv, "gpu": identity,
        "model_revision": OuroAdapter.model_revision, "model": "ouro-1.4b",
        "dataset": args.dataset, "dataset_revision": "UNKNOWN (local cached Hugging Face dataset)",
        "prompt_sha256": prompt_digest.hexdigest(), "prompt_hashes": prompt_hashes,
        "prompt_cap": 64, "request_split": split, "split_unit": "request", "seed": args.seed,
        "threshold": args.threshold, "num_rows": args.num_rows,
        "num_tokens": args.num_tokens, "rmax": initial.rmax,
        "kv_backend": initial.kv_policy, "chunk_size": args.chunk_size,
        "feature_schema_version": 1, "scalar_names": SCALARS,
        "embedding_projection": "frozen seeded Gaussian 1/sqrt(hidden)",
        "representation_type": "previous_exit_hidden_projection; token0 zero fallback",
        "projection_dim": args.projection_dim,
        "feature_causality": "all features written before current-token recurrence; label separate",
        "completed_chunks": [],
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shape = (args.num_tokens, args.num_rows)
    scalar_arr = np.lib.format.open_memmap(args.output_dir / "scalars.npy", mode="w+", dtype="float32",
                                           shape=(*shape, len(SCALARS)))
    embedding_arr = np.lib.format.open_memmap(args.output_dir / "embedding.npy", mode="w+", dtype="float16",
                                              shape=(*shape, args.projection_dim))
    representation_arr = np.lib.format.open_memmap(args.output_dir / "prelude.npy", mode="w+", dtype="float16",
                                                   shape=(*shape, args.projection_dim))
    tokens_arr = np.lib.format.open_memmap(args.output_dir / "token_ids.npy", mode="w+", dtype="int32", shape=shape)
    labels_arr = np.lib.format.open_memmap(args.output_dir / "depth_labels.npy", mode="w+", dtype="int16", shape=shape)
    began = time.perf_counter()
    for start in range(0, args.num_rows, args.chunk_size):
        stop = min(start + args.chunk_size, args.num_rows)
        chunk_prompts = prompts[start:stop]
        encoded = tokenizer(chunk_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        input_ids = encoded["input_ids"].cuda()
        prompt_valid = encoded["attention_mask"].cuda().bool()
        adapter = OuroAdapter(model, tokenizer, stop - start)
        current_tokens = adapter.prefill(input_ids, prompt_valid)
        previous_logits = adapter.prefill_logits[:, -1, :]
        previous_hidden = torch.zeros(stop - start, args.projection_dim, dtype=torch.float32, device="cuda")
        history: list[list[int]] = [[] for _ in range(stop - start)]
        previous_trajectories: list[list[float]] = [[] for _ in range(stop - start)]
        prompt_lengths = prompt_valid.sum(dim=1).cpu().tolist()
        for token_idx in range(args.num_tokens):
            adapter.begin_token()
            next_tokens = torch.empty_like(current_tokens)
            next_logits = torch.empty_like(previous_logits)
            next_hidden = torch.empty_like(previous_hidden)
            for local_id in range(stop - start):
                request_id = start + local_id
                row = torch.tensor([local_id], dtype=torch.long, device=adapter.device)
                adapter.backend.set_active(row)
                token = current_tokens.index_select(0, row)
                embedded, state = adapter.prelude(token)
                embedded_proj = embedded.float().flatten() @ projection
                prev_depths = history[local_id]
                trajectory = previous_trajectories[local_id]
                scalar = [float(token_idx), float(prompt_lengths[local_id] + token_idx),
                          float(prev_depths[-1]) if prev_depths else 0.0,
                          float(bool(prev_depths))]
                scalar += _history_features(prev_depths)
                scalar += [float(prev_depths[-1] - prev_depths[-2]) if len(prev_depths) > 1 else 0.0]
                scalar += _uncertainty(previous_logits[local_id], token)
                scalar += [float(len(trajectory))]
                scalar += trajectory[:16] + [0.0] * (16 - len(trajectory))
                if len(scalar) != len(SCALARS):
                    raise AssertionError("scalar schema mismatch")
                scalar_arr[token_idx, request_id] = scalar
                embedding_arr[token_idx, request_id] = embedded_proj.cpu().numpy().astype(np.float16)
                representation_arr[token_idx, request_id] = previous_hidden[local_id].cpu().numpy().astype(np.float16)
                tokens_arr[token_idx, request_id] = int(token.item())
                current_trajectory: list[float] = []
                for recurrence in range(1, adapter.rmax + 1):
                    after = adapter.recurrent_step(state, embedded, recurrence)
                    lam = torch.sigmoid(model.model.early_exit_gate(after).squeeze()).float()
                    current_trajectory.append(float(lam.item()))
                    should_exit = adapter.live_exit(state, after, row, recurrence, args.threshold)
                    state = after
                    if bool(should_exit.item()):
                        break
                adapter.backend.on_exit(row, recurrence)
                logits = adapter.coda_logits(state)
                next_logits.index_copy_(0, row, logits[:, -1, :])
                next_tokens.index_copy_(0, row, logits.argmax(dim=-1))
                next_hidden[local_id] = state.float().flatten() @ projection
                labels_arr[token_idx, request_id] = recurrence
                history[local_id].append(recurrence)
                previous_trajectories[local_id] = current_trajectory
            adapter.commit_token()
            current_tokens = next_tokens
            previous_logits = next_logits
            previous_hidden = next_hidden
            if (token_idx + 1) % 8 == 0 or token_idx + 1 == args.num_tokens:
                print(f"chunk={start}:{stop} tokens={token_idx + 1}/{args.num_tokens} "
                      f"elapsed_s={time.perf_counter() - began:.1f}", flush=True)
        manifest["completed_chunks"].append({"start": start, "stop": stop,
                                               "prompt_token_counts": prompt_lengths})
        for array in (scalar_arr, embedding_arr, representation_arr, tokens_arr, labels_arr):
            array.flush()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        del adapter, input_ids, prompt_valid, current_tokens, previous_logits, previous_hidden
        torch.cuda.empty_cache()
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
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--test-rows", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.train_rows, args.validation_rows, args.test_rows, args.num_tokens,
           args.projection_dim, args.chunk_size) < 1:
        parser.error("all counts must be positive")
    args.num_rows = args.train_rows + args.validation_rows + args.test_rows
    collect(args)


if __name__ == "__main__":
    main()
