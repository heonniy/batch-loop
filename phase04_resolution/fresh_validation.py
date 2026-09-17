"""Predeclared fresh-cohort BF16 batch-dependent live-depth validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

from phase02_runtime.common import run_interval
from phase02_runtime.huginn_adapter import HuginnAdapter, HUGINN_SNAPSHOT
from phase02_runtime.ouro_adapter import OuroAdapter, OURO14_SNAPSHOT
from phase04_runtime.data_contract import effective_token_hashes, sha256_file
from phase04_runtime.validate import _adapter, _reference, gpu_identity


def _fresh_prompts() -> list[str]:
    base = "Describe the next step of this short reasoning task. "
    return [f"Validation request {1000 + index}: " + base * (1 + index % 4)
            for index in range(32)]


def _margin_bin(value: float) -> str:
    if value < 0.001:
        return "[0,0.001)"
    if value < 0.005:
        return "[0.001,0.005)"
    if value < 0.02:
        return "[0.005,0.02)"
    return "[0.02,infinity)"


def _orders() -> dict[str, list[int]]:
    base = list(range(32))
    result = {"fifo": base}
    for seed in (17, 29):
        shuffled = base.copy()
        random.Random(seed).shuffle(shuffled)
        result[f"permutation_seed_{seed}"] = shuffled
    return result


@torch.inference_mode()
def _batched(model_key, model, tokenizer, prompts, known_tokens, threshold, batch_size, order):
    n = len(prompts)
    adapter = _adapter(model_key, model, tokenizer, list(range(n)), 1, "benchmark")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    adapter.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
    records = {}
    total_ric = total_work = frozen_ric = 0
    for token_idx, tokens in enumerate(known_tokens):
        adapter.begin_token()
        for first in range(0, n, batch_size):
            selected = order[first:first + batch_size]
            if len(selected) != batch_size:
                raise RuntimeError("primary validation requires complete microbatches")
            rows = torch.tensor(selected, dtype=torch.long, device="cuda")
            adapter.backend.set_active(rows)
            ids = torch.tensor([[tokens[rid]] for rid in selected], dtype=torch.long, device="cuda")
            embedded, state = adapter.prelude(ids)
            result = run_interval(adapter, embedded, state, rows, rows,
                                  exit_source="live", threshold=threshold,
                                  instrumentation="benchmark", debug_ids=True)
            adapter.backend.set_active(rows)
            next_ids = adapter.coda_logits(result.final_state).argmax(dim=-1)
            depths = result.exit_depths.tolist()
            total_ric += max(depths)
            total_work += sum(depths)
            for local, rid in enumerate(selected):
                scores = [event["exit_score"][event["active_request_ids"].index(rid)]
                          for event in result.events if rid in event["active_request_ids"]]
                margins = [event["exit_margin"][event["active_request_ids"].index(rid)]
                           for event in result.events if rid in event["active_request_ids"]]
                records[(token_idx, rid)] = {"depth": depths[local],
                                             "next_token": int(next_ids[local]),
                                             "scores": scores, "margins": margins}
        adapter.commit_token()
    return records, total_ric, total_work


def validate(model_key: str, role: str, old_manifest_path: Path, output_dir: Path) -> dict:
    gpu = gpu_identity(model_key, role)
    torch.set_num_threads(8)
    old = json.loads(old_manifest_path.read_text())
    if old["status"] != "PASS_EXACT_SAVED_REPLAY" or old["model"] != model_key:
        raise RuntimeError("old hash source must be the verified model packet")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    adapter_type = HuginnAdapter if model_key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if model_key == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("fresh validation must use primary BF16 precision")
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if sha256_file(imported) != old["imported_model_source_sha256"]:
        raise RuntimeError("imported model source changed")
    prompts = _fresh_prompts()
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    hashes = effective_token_hashes(encoded["input_ids"], encoded["attention_mask"])
    old_hashes = {row["sha256"] for row in old["cohort_effective_inputs"]}
    if len(set(hashes)) != 32 or set(hashes) & old_hashes:
        raise RuntimeError("fresh cohort is duplicated within itself or overlaps old Gate A")
    reference, known, _ = _reference(model_key, model, tokenizer, prompts, 8,
                                     old["threshold"], 1, int(encoded["input_ids"].shape[1]))
    for record in reference["records"].values():
        record.pop("logits", None)
    reference.pop("states", None)
    orders = _orders()
    summary_rows = []
    raw_rows = []
    bin_rows = []
    denominator = 32 * 8
    for batch_size in (2, 4, 8):
        for order_name, order in orders.items():
            if sorted(order) != list(range(32)):
                raise RuntimeError("fixed order is not a permutation")
            actual, ric, work = _batched(model_key, model, tokenizer, prompts, known,
                                         old["threshold"], batch_size, order)
            frozen_ric = 0
            for token_idx in range(8):
                for first in range(0, 32, batch_size):
                    frozen_ric += max(reference["records"][(token_idx, rid)]["depth"]
                                      for rid in order[first:first + batch_size])
            depth_flips = token_flips = near_count = 0
            bins = {name: {"positions": 0, "depth_flips": 0, "next_token_flips": 0}
                    for name in ("[0,0.001)", "[0.001,0.005)", "[0.005,0.02)", "[0.02,infinity)")}
            for token_idx in range(8):
                for rid in range(32):
                    before = reference["records"][(token_idx, rid)]
                    after = actual[(token_idx, rid)]
                    d_flip = int(before["depth"] != after["depth"])
                    n_flip = int(before["next_token"] != after["next_token"])
                    depth_flips += d_flip
                    token_flips += n_flip
                    margin = min(abs(value) for value in before["margins"])
                    near_count += int(margin < 0.005)
                    name = _margin_bin(margin)
                    bins[name]["positions"] += 1
                    bins[name]["depth_flips"] += d_flip
                    bins[name]["next_token_flips"] += n_flip
                    raw_rows.append({"model": model_key, "B": batch_size, "order": order_name,
                                     "token_idx": token_idx, "request_id": rid,
                                     "reference_depth": before["depth"], "live_depth": after["depth"],
                                     "depth_flip": d_flip, "reference_next_token": before["next_token"],
                                     "live_next_token": after["next_token"], "next_token_flip": n_flip,
                                     "reference_min_abs_margin": margin,
                                     "margin_bin": name,
                                     "reference_exit_margin": before["margins"][-1],
                                     "live_exit_margin": after["margins"][-1],
                                     "reference_scores": json.dumps(before["scores"]),
                                     "live_scores": json.dumps(after["scores"])})
            summary_rows.append({"model": model_key, "B": batch_size, "order": order_name,
                                 "positions": denominator, "depth_flips": depth_flips,
                                 "depth_flip_rate": depth_flips / denominator,
                                 "next_token_flips": token_flips,
                                 "next_token_flip_rate": token_flips / denominator,
                                 "reference_margin_lt_0.005": near_count,
                                 "reference_B1_work": sum(row["depth"] for row in reference["records"].values()),
                                 "actual_live_work": work,
                                 "actual_live_RIC": ric,
                                 "frozen_B1_depth_grouping_RIC": frozen_ric})
            for name, values in bins.items():
                bin_rows.append({"model": model_key, "B": batch_size, "order": order_name,
                                 "margin_bin": name, **values})
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("batch_shape_validation.csv", summary_rows),
                       ("per_token_validation.csv", raw_rows),
                       ("margin_bin_summary.csv", bin_rows)):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    snapshot = HUGINN_SNAPSHOT if model_key == "huginn" else OURO14_SNAPSHOT
    manifest = {"status": "FRESH_VALIDATION_RECORDED", "model": model_key,
                "source_sha": source_sha,
                "protocol": "phase04/BF16_NUMERICAL_VALIDATION_PROTOCOL_20260917.md",
                "argv": sys.argv, "gpu": gpu, "precision": "torch.bfloat16",
                "model_revision": old["model_revision"],
                "model_config_sha256": hashlib.sha256(json.dumps(
                    model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
                "imported_model_source_sha256": sha256_file(imported),
                "tokenizer_snapshot_sha256": sha256_file(snapshot / "tokenizer_config.json"),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "transformers": importlib.import_module("transformers").__version__,
                "prompt_source": "fresh Validation request 1000..1031 synthetic correctness cohort",
                "prompt_cap": 64, "cohort_tokenized_width": int(encoded["input_ids"].shape[1]),
                "decode_positions": 8, "EOS_handling": "not applied",
                "input_policy": "B1-known token IDs teacher-forced into each live batched run",
                "effective_input_hashes": hashes,
                "old_counterexample_manifest_sha256": sha256_file(old_manifest_path),
                "threshold": old["threshold"], "kv_policy": old["kv_policy"],
                "B_values": [2, 4, 8], "orderings": orders,
                "reference_positions": denominator,
                "reference_B1_total_work": sum(row["depth"] for row in reference["records"].values()),
                "reference_margin_lt_0.005": sum(
                    min(abs(value) for value in row["margins"]) < 0.005
                    for row in reference["records"].values()),
                "raw_artifacts": ["batch_shape_validation.csv", "per_token_validation.csv",
                                  "margin_bin_summary.csv"]}
    (output_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "model": model_key,
                      "conditions": len(summary_rows),
                      "depth_flips": [row["depth_flips"] for row in summary_rows],
                      "next_token_flips": [row["next_token_flips"] for row in summary_rows]}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    validate(args.model, args.role, args.old_manifest, args.output_dir)


if __name__ == "__main__":
    main()
