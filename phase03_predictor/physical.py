"""Staged, fixed-work physical FIFO/predictor/Oracle survivor tournament.

Huginn uses a common Prelude stage for every strategy. Ouro has no Prelude,
but uses the same pre-group embedding stage for all strategies. Exit labels are
frozen B=1 live labels from the held-out feature cohort; recurrent work is
identical across strategies. The model still physically executes only survivor
rows. Frozen token IDs are teacher-forced at every boundary to keep the input
trajectory identical despite strategy-specific KV/output drift. Predictor
feature extraction, CPU head inference, and sorting are inside
the prefill-excluded decode wall timer.
"""

from __future__ import annotations

import argparse
import csv
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

from phase02_runtime.common import run_interval
from phase02_runtime.huginn_adapter import HuginnAdapter
from phase02_runtime.ouro_adapter import OuroAdapter
from phase03_predictor.collect_huginn import SCALARS, _history_features
from phase03_predictor.tournament import Head, assemble_candidate_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_identity(model_key: str) -> dict:
    physical = 7 if model_key == "huginn" else 6
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical) or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"{model_key} Phase-3 physical requires only GPU{physical}")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical), "--query-gpu=index,uuid,name,memory.total",
         "--format=csv,noheader"], text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 4 or fields[0] != str(physical) or "H100" not in fields[2]:
        raise RuntimeError(f"physical GPU{physical} identity mismatch: {query}")
    result = {"physical_index": physical, "uuid": fields[1], "name": fields[2],
              "memory_total": fields[3], "CUDA_VISIBLE_DEVICES": visible, "cuda_ordinal": 0}
    if model_key == "ouro":
        result["ownership_override"] = "user correction in current task: GPU6=Ouro, GPU7=Huginn"
    return result


def _load_prompts(path: Path, ids: list[int]) -> list[str]:
    selected = {}
    wanted = set(ids)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["request_id"] in wanted:
                prompt = record["prompt"]
                if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record["sha256"]:
                    raise RuntimeError("prompt receipt hash mismatch")
                selected[record["request_id"]] = prompt
    if set(selected) != wanted:
        raise RuntimeError("held-out prompt pool incomplete")
    return [selected[rid] for rid in ids]


def _top_three(path: Path) -> list[str]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ranked = sorted((row for row in rows if row["split"] == "test"),
                    key=lambda row: (-float(row["oracle_ric_recovery"]),
                                     float(row["prediction_cpu_s"]), row["candidate"]))
    if len(ranked) != 10 or len({row["candidate"] for row in ranked}) != 10:
        raise RuntimeError("full C0-C9 held-out tournament is required")
    return [row["candidate"] for row in ranked[:3]]


class Predictor:
    def __init__(self, candidate: str, tournament_root: Path, feature_manifest: dict):
        self.candidate = candidate
        self.feature_manifest = feature_manifest
        self.head = None
        self.mean = self.std = None
        if candidate != "C0":
            checkpoint = torch.load(tournament_root / f"{candidate}_checkpoint.pt",
                                    map_location="cpu", weights_only=False)
            self.head = Head(checkpoint["input_dim"], checkpoint["output_dim"],
                             checkpoint["config"]["width"])
            self.head.load_state_dict(checkpoint["state_dict"])
            self.head.eval()
            self.mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            self.std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    def score(self, scalar: np.ndarray, embedding: np.ndarray, representation: np.ndarray) -> np.ndarray:
        if self.candidate == "C0":
            return scalar[:, 2].copy()
        matrix = assemble_candidate_features(scalar, embedding, representation,
                                             self.feature_manifest)[self.candidate]
        x = np.ascontiguousarray((matrix - self.mean) / self.std, dtype=np.float32)
        with torch.inference_mode():
            out = self.head(torch.from_numpy(x))
            if self.candidate == "C7":
                out = 1 + torch.sigmoid(out).sum(dim=-1)
            else:
                out = out.squeeze(-1)
        score = out.numpy()
        if not np.isfinite(score).all():
            raise AssertionError("non-finite physical predictor score")
        return score


def _need_trajectory(candidate: str) -> bool:
    return candidate in {"C2", "C6", "C7", "C8", "C9"}


def _uncertainty_batch(logits: torch.Tensor, token_ids: torch.Tensor) -> np.ndarray:
    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1)
    probs = torch.exp(values - log_z[:, None])
    top = torch.topk(values, 2, dim=-1).values
    surprisal = log_z - values.gather(1, token_ids.long()).squeeze(1)
    entropy = log_z - (probs * values).sum(dim=-1)
    top_probs = torch.exp(top - log_z[:, None])
    return torch.stack((surprisal, entropy, top_probs[:, 0],
                        top_probs[:, 0] - top_probs[:, 1]), dim=-1).cpu().numpy()


def _features(args, adapter, current_tokens, previous_logits, histories,
              previous_trajectories, previous_hidden, staged_embedded,
              prompt_lengths, projection, candidate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(histories)
    scalar = np.zeros((n, len(SCALARS)), dtype=np.float32)
    needs_uncertainty = candidate in {"C3", "C6", "C7", "C8", "C9"}
    uncertainty = (_uncertainty_batch(previous_logits, current_tokens) if needs_uncertainty
                   else np.zeros((n, 4), dtype=np.float32))
    for request_id in range(n):
        history = histories[request_id]
        trajectory = previous_trajectories[request_id]
        fields = [float(len(history)), float(prompt_lengths[request_id] + len(history)),
                  float(history[-1]) if history else 0.0, float(bool(history))]
        fields += _history_features(history)
        fields += [float(history[-1] - history[-2]) if len(history) > 1 else 0.0]
        fields += uncertainty[request_id].tolist()
        fields += [float(len(trajectory))]
        fields += trajectory[:16] + [0.0] * (16 - len(trajectory))
        scalar[request_id] = fields
    dimension = projection.shape[1]
    embedding = np.zeros((n, dimension), dtype=np.float32)
    representation = np.zeros((n, dimension), dtype=np.float32)
    if candidate in {"C4", "C6", "C7", "C8", "C9"} or (candidate == "C5" and args.model == "ouro"):
        if args.model == "huginn":
            raw = adapter.model.transformer.wte(current_tokens)
            if adapter.model.emb_scale != 1:
                raw = raw * adapter.model.emb_scale
        else:
            raw = staged_embedded
        embedding = (raw.float().squeeze(1) @ projection).cpu().numpy()
    if candidate in {"C5", "C6", "C7", "C8", "C9"}:
        representation = ((staged_embedded.float().squeeze(1) @ projection).cpu().numpy()
                          if args.model == "huginn" else previous_hidden.cpu().numpy())
    return scalar, embedding, representation


@torch.inference_mode()
def _run_once(args, loaded, tokenizer, input_ids, prompt_valid, labels, frozen_tokens, prompt_lengths,
              projection, feature_manifest, predictors, strategy, repetition, event_stream) -> dict:
    n, batch_size = labels.shape[1], args.batch_size
    adapter = (HuginnAdapter(loaded, tokenizer, n, state_init="native-seeded") if args.model == "huginn"
               else OuroAdapter(loaded, tokenizer, n))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    current_tokens = adapter.prefill(input_ids, prompt_valid)
    prefill_match = float((current_tokens.squeeze(1).cpu().numpy() == frozen_tokens[0]).mean())
    previous_logits = adapter.prefill_logits[:, -1, :]
    previous_hidden = torch.zeros((n, projection.shape[1]), dtype=torch.float32, device="cuda")
    histories: list[list[int]] = [[] for _ in range(n)]
    previous_trajectories: list[list[float]] = [[] for _ in range(n)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    total_core = total_stage = total_predictor = total_sort = 0.0
    actual_ric = total_work = 0
    expected_ric = 0
    matching_next_tokens = 0
    work_digest = hashlib.sha256()
    for token_idx in range(args.num_tokens):
        current_tokens = torch.from_numpy(np.ascontiguousarray(frozen_tokens[token_idx])).to(
            device="cuda", dtype=torch.long).unsqueeze(1)
        adapter.begin_token()
        stage_start = time.perf_counter()
        width = int(adapter.model.transformer.wte.weight.shape[1] if args.model == "huginn"
                    else adapter.model.model.embed_tokens.weight.shape[1])
        embedded_all = torch.empty((n, 1, width), dtype=torch.bfloat16, device="cuda")
        state_all = torch.empty_like(embedded_all)
        for first in range(0, n, batch_size):
            rows = torch.arange(first, first + batch_size, dtype=torch.long, device="cuda")
            adapter.backend.set_active(rows)
            embedded, state = adapter.prelude(current_tokens.index_select(0, rows))
            embedded_all.index_copy_(0, rows, embedded)
            state_all.index_copy_(0, rows, state)
        torch.cuda.synchronize()
        total_stage += time.perf_counter() - stage_start
        if strategy in {"STAGED_FIFO", "FIFO"}:
            order = np.arange(n)
        elif strategy == "ORACLE":
            order = np.lexsort((np.arange(n), labels[token_idx]))
        else:
            predict_start = time.perf_counter()
            scalar, embedding, representation = _features(
                args, adapter, current_tokens, previous_logits, histories,
                previous_trajectories, previous_hidden, embedded_all, prompt_lengths,
                projection, strategy,
            )
            scores = predictors[strategy].score(scalar, embedding, representation)
            total_predictor += time.perf_counter() - predict_start
            sort_start = time.perf_counter()
            order = np.lexsort((np.arange(n), scores))
            total_sort += time.perf_counter() - sort_start
        expected_ric += int(labels[token_idx, order].reshape(-1, batch_size).max(axis=1).sum())
        next_tokens = torch.empty_like(current_tokens)
        next_logits = torch.empty_like(previous_logits)
        next_hidden = torch.empty_like(previous_hidden)
        trajectory_sink: dict[int, list[float]] | None = {} if _need_trajectory(strategy) else None
        for microbatch_id, first in enumerate(range(0, n, batch_size)):
            row_ids = order[first:first + batch_size]
            rows = torch.tensor(row_ids, dtype=torch.long, device="cuda")
            forced = torch.tensor(labels[token_idx, row_ids].astype(np.int64), device="cuda")
            result = run_interval(
                adapter, embedded_all.index_select(0, rows), state_all.index_select(0, rows),
                rows, rows, exit_source="trace", forced_depths=forced, debug_ids=False,
                trajectory_sink=trajectory_sink,
                trajectory_kind=("huginn_convergence" if args.model == "huginn" else "ouro_gate")
                if trajectory_sink is not None else None,
            )
            actual_ric += len(result.events)
            total_core += result.core_gpu_ms / 1000
            total_work += int(result.exit_depths.sum())
            adapter.backend.set_active(rows)
            logits = adapter.coda_logits(result.final_state)
            next_tokens.index_copy_(0, rows, logits.argmax(dim=-1))
            next_logits.index_copy_(0, rows, logits[:, -1, :])
            if args.model == "ouro":
                next_hidden.index_copy_(0, rows, result.final_state.float().squeeze(1) @ projection)
            if repetition >= 0:
                for event in result.events:
                    event_stream.write(json.dumps({
                        "strategy": strategy, "repetition": repetition, "token_idx": token_idx,
                        "microbatch_id": microbatch_id, "row_ids": row_ids.tolist(), **event,
                    }) + "\n")
        for request_id in range(n):
            depth = int(labels[token_idx, request_id])
            histories[request_id].append(depth)
            work_digest.update(f"{request_id},{token_idx},{depth}\n".encode())
            if trajectory_sink is not None:
                previous_trajectories[request_id] = trajectory_sink[request_id]
        adapter.commit_token()
        if token_idx + 1 < args.num_tokens:
            matching_next_tokens += int((next_tokens.squeeze(1).cpu().numpy() ==
                                         frozen_tokens[token_idx + 1]).sum())
        current_tokens = next_tokens
        previous_logits = next_logits
        previous_hidden = next_hidden
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if actual_ric != expected_ric:
        raise AssertionError("physical RIC differs from frozen label RIC")
    if total_work != int(labels.sum()):
        raise AssertionError("physical token-loop work differs from frozen labels")
    return {
        "strategy": strategy, "repetition": repetition, "run_kind": "warmup" if repetition < 0 else "measured",
        "num_rows": n, "batch_size": batch_size, "num_tokens": args.num_tokens,
        "recurrent_invocations": actual_ric, "total_token_loop_work": total_work,
        "same_work_checksum": work_digest.hexdigest(),
        "prefill_token_match_fraction": prefill_match,
        "generated_next_token_match_fraction": (
            matching_next_tokens / (n * (args.num_tokens - 1)) if args.num_tokens > 1 else float("nan")),
        "recurrent_gpu_s": total_core, "staging_wall_s": total_stage,
        "predictor_wall_s": total_predictor, "sort_wall_s": total_sort,
        "decode_wall_s": elapsed, "generated_tokens_per_s": n * args.num_tokens / elapsed,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "kv_backend": adapter.kv_policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("huginn", "ouro"), required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--tournament-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control", action="store_true", help="N=B=32 FIFO/Oracle control")
    args = parser.parse_args()
    torch.set_num_threads(8)
    identity = _gpu_identity(args.model)
    feature_manifest = json.loads((args.feature_root / "feature_manifest.json").read_text())
    if feature_manifest["status"] != "COMPLETE":
        raise RuntimeError("frozen feature cohort incomplete")
    if not feature_manifest["model"].startswith(args.model):
        raise RuntimeError("feature model mismatch")
    ids = feature_manifest["request_split"]["test"]
    if args.control:
        ids = ids[:args.batch_size]
    if len(ids) % args.batch_size:
        raise RuntimeError("held-out physical rows must make full microbatches")
    labels = np.asarray(np.load(args.feature_root / "depth_labels.npy", mmap_mode="r")[:args.num_tokens, ids],
                        dtype=np.int16)
    frozen_tokens = np.asarray(np.load(args.feature_root / "token_ids.npy", mmap_mode="r")[:args.num_tokens, ids],
                               dtype=np.int32)
    if labels.shape != (args.num_tokens, len(ids)) or np.any(labels < 1):
        raise RuntimeError("held-out frozen labels incomplete")
    prompts = _load_prompts(args.feature_root / "prompts.jsonl.gz", ids)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    adapter_cls = HuginnAdapter if args.model == "huginn" else OuroAdapter
    initial = (adapter_cls.load(len(ids), state_init="native-seeded") if args.model == "huginn"
               else adapter_cls.load(len(ids)))
    loaded, tokenizer = initial.model, initial.tokenizer
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = encoded["input_ids"].cuda()
    prompt_valid = encoded["attention_mask"].cuda().bool()
    prompt_lengths = prompt_valid.sum(dim=1).cpu().tolist()
    projection = torch.from_numpy(np.load(args.feature_root / "projection.npy")).cuda()
    top_three = [] if args.control else _top_three(args.tournament_root / "predictor_tournament.csv")
    predictors = {candidate: Predictor(candidate, args.tournament_root, feature_manifest)
                  for candidate in top_three}
    fifo = "STAGED_FIFO" if args.model == "huginn" else "FIFO"
    strategies = [fifo, *top_three, "ORACLE"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "execution_kind": "real-no-refill-runtime", "status": "RUNNING",
        "source_sha": source_sha, "source_dirty": dirty, "argv": sys.argv,
        "gpu": identity, "model": args.model, "model_revision": adapter_cls.model_revision,
        "kv_backend": adapter_cls.kv_policy, "feature_manifest_sha256": _sha256(args.feature_root / "feature_manifest.json"),
        "tournament_csv_sha256": (None if args.control else
                                   _sha256(args.tournament_root / "predictor_tournament.csv")),
        "test_request_ids": ids, "prompt_sha256": hashlib.sha256("".join(
            f"{rid}:{prompt}\n" for rid, prompt in zip(ids, prompts)).encode()).hexdigest(),
        "test_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "test_token_ids_sha256": hashlib.sha256(frozen_tokens.tobytes()).hexdigest(),
        "prompt_cap": 64, "num_tokens": args.num_tokens, "batch_size": args.batch_size,
        "warmups": args.warmups, "repeats": args.repeats,
        "top_three_by_test_RIC": top_three, "strategies": strategies,
        "same_staging_path_for_all_strategies": True,
        "physical_labels": "frozen B1 live depths; trace-forced identical work",
        "token_trajectory_policy": "frozen B1 token IDs teacher-forced at each boundary",
        "quality_equivalence_claimed": False,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    with gzip.open(args.output_dir / "recurrence_events.jsonl.gz", "wt", encoding="utf-8") as events:
        for repetition in range(-args.warmups, args.repeats):
            for strategy in strategies:
                torch.cuda.reset_peak_memory_stats()
                summary = _run_once(args, loaded, tokenizer, input_ids, prompt_valid,
                                    labels, frozen_tokens, prompt_lengths, projection, feature_manifest,
                                    predictors, strategy, repetition, events)
                summaries.append(summary)
                print(json.dumps({key: summary[key] for key in
                                  ("strategy", "repetition", "recurrent_invocations",
                                   "decode_wall_s", "predictor_wall_s")}), flush=True)
    checksums = {row["same_work_checksum"] for row in summaries}
    if len(checksums) != 1:
        raise AssertionError("strategies did not execute identical request/token work")
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    manifest["status"] = "COMPLETE"
    manifest["same_work_checksum"] = next(iter(checksums))
    manifest["summary_sha256"] = _sha256(args.output_dir / "summary.csv")
    manifest["events_sha256"] = _sha256(args.output_dir / "recurrence_events.jsonl.gz")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
