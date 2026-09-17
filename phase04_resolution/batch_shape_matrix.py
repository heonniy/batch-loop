"""One-token live-gate batch-shape matrix from fixed B1 per-request histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import torch

from phase02_runtime.common import run_interval
from phase02_runtime.huginn_adapter import HuginnAdapter
from phase02_runtime.ouro_adapter import OuroAdapter
from phase04_runtime.data_contract import sha256_file
from phase04_runtime.validate import gpu_identity
from phase04_resolution.counterexamples import _tensor_sha256


def _new_adapter(model_key, model, tokenizer, request_ids):
    return (HuginnAdapter(model, tokenizer, len(request_ids), state_init="native-seeded",
                          identity_seed=1, canonical_ids=torch.tensor(request_ids, device="cuda"),
                          capture_state_hashes=True)
            if model_key == "huginn" else OuroAdapter(model, tokenizer, len(request_ids)))


@torch.inference_mode()
def _canonical_histories(model_key, model, tokenizer, packet, request_ids, token_idx, bf16):
    cohort = packet["cohort_effective_inputs"]
    known = packet["input_tokens_by_boundary"]
    adapters = {}
    for rid in request_ids:
        adapter = _new_adapter(model_key, model, tokenizer, [rid])
        ids = torch.tensor([cohort[rid]["input_ids"]], dtype=torch.long, device="cuda")
        mask = torch.tensor([cohort[rid]["attention_mask"]], dtype=torch.bool, device="cuda")
        first = adapter.prefill(ids, mask)
        if bf16 and int(first[0]) != known[0][rid]:
            raise RuntimeError(f"B1 prefill token failed frozen replay: {rid}")
        for step in range(token_idx):
            adapter.begin_token()
            rows = torch.tensor([0], dtype=torch.long, device="cuda")
            adapter.backend.set_active(rows)
            current = torch.tensor([[known[step][rid]]], dtype=torch.long, device="cuda")
            embedded, state = adapter.prelude(current)
            result = run_interval(adapter, embedded, state, rows,
                                  torch.tensor([rid], dtype=torch.long, device="cuda"),
                                  exit_source="live", threshold=packet["threshold"],
                                  instrumentation="profile")
            adapter.backend.set_active(rows)
            next_id = int(adapter.coda_logits(result.final_state).argmax(dim=-1)[0])
            adapter.commit_token()
            if bf16 and next_id != known[step + 1][rid]:
                raise RuntimeError(f"B1 previous token failed frozen replay: {(step, rid)}")
        adapters[rid] = adapter
    return adapters


def _merge(model_key, model, tokenizer, snapshots, copies):
    logical = [logical_id for logical_id, _ in copies]
    if len(logical) != len(set(logical)):
        raise ValueError("matrix requires distinct logical request identities")
    adapter = _new_adapter(model_key, model, tokenizer, logical)
    backend = adapter.backend
    original = snapshots[copies[0][1]].backend
    slots = set(original.history)
    if any(set(snapshots[source].backend.history) != slots for _, source in copies):
        raise RuntimeError("frozen history slot sets differ")
    backend.key_valid = torch.cat([snapshots[source].backend.key_valid for _, source in copies], dim=0)
    backend.phase = "ready"
    backend.history = {}
    if model_key == "huginn":
        length = original.history_length
        if any(snapshots[source].backend.history_length != length for _, source in copies):
            raise RuntimeError("frozen Huginn history lengths differ")
        backend.history_length = backend.history_capacity = length
    for slot in slots:
        pair = []
        for kind in (0, 1):
            values = [snapshots[source].backend.history[slot][kind] for _, source in copies]
            if model_key == "huginn":
                values = [value[:, :, :backend.history_length, :] for value in values]
            merged = torch.cat(values, dim=0)
            for local, original_value in enumerate(values):
                if not torch.equal(merged[local], original_value[0]):
                    raise RuntimeError("frozen target/filler KV changed during merge")
            pair.append(merged)
        backend.history[slot] = tuple(pair)
    return adapter


def _history_digest(snapshot) -> str:
    digest = hashlib.sha256()
    backend = snapshot.backend
    digest.update(_tensor_sha256(backend.key_valid).encode())
    for slot in sorted(backend.history, key=repr):
        digest.update(repr(slot).encode())
        for value in backend.history[slot]:
            if hasattr(backend, "history_length"):
                value = value[:, :, :backend.history_length, :]
            digest.update(_tensor_sha256(value).encode())
    return digest.hexdigest()


@torch.inference_mode()
def _one(model_key, model, tokenizer, packet, snapshots, copies, token_idx, clone_h0=False):
    adapter = _merge(model_key, model, tokenizer, snapshots, copies)
    backend = adapter.backend
    backend.begin_token()
    rows = torch.arange(len(copies), dtype=torch.long, device="cuda")
    backend.set_active(rows)
    token_ids = torch.tensor([[packet["input_tokens_by_boundary"][token_idx][source]]
                              for _, source in copies], dtype=torch.long, device="cuda")
    embedded, state = adapter.prelude(token_ids)
    duplicate_prelude_equal = None
    if clone_h0:
        if len(copies) != 2 or copies[0][1] != copies[1][1]:
            raise ValueError("duplicate control must clone one source into two rows")
        duplicate_prelude_equal = bool(torch.equal(embedded[0], embedded[1]))
        # Explicit same-state diagnostic; the second logical ID remains distinct.
        state[1].copy_(state[0])
    request_ids = torch.tensor([logical for logical, _ in copies], dtype=torch.long, device="cuda")
    result = run_interval(adapter, embedded, state, rows, request_ids,
                          exit_source="live", threshold=packet["threshold"],
                          instrumentation="profile", debug_ids=True)
    backend.set_active(rows)
    next_ids = adapter.coda_logits(result.final_state).argmax(dim=-1)
    out = []
    for local, (logical, _) in enumerate(copies):
        scores = [event["exit_score"][event["active_request_ids"].index(logical)]
                  for event in result.events if logical in event["active_request_ids"]]
        margins = [event["exit_margin"][event["active_request_ids"].index(logical)]
                   for event in result.events if logical in event["active_request_ids"]]
        h0_hash = next((digest for (rid, position), digest in getattr(adapter, "state_hashes", {}).items()
                        if rid == logical and position != "prefill"), None)
        out.append({"request_id": logical, "depth": int(result.exit_depths[local]),
                    "next_token": int(next_ids[local]), "scores": scores, "margins": margins,
                    "h0_sha256": h0_hash,
                    "final_hidden_sha256": _tensor_sha256(result.final_state[local])})
    return {"rows": out, "duplicate_prelude_equal": duplicate_prelude_equal,
            "active_widths": [event["active_batch_size"] for event in result.events]}


def _selected(target, fillers, batch_size, position, permuted):
    if batch_size == 1:
        return [target]
    chosen = fillers[:batch_size - 1]
    if permuted:
        chosen = list(reversed(chosen))
    result = list(chosen)
    result.insert(position, target)
    return result


def matrix(model_key: str, role: str, manifest_path: Path, target: int,
           token_idx: int, fp32: bool, output_dir: Path) -> dict:
    gpu = gpu_identity(model_key, role)
    torch.set_num_threads(8)
    packet = json.loads(manifest_path.read_text())
    if packet["model"] != model_key or packet["status"] != "PASS_EXACT_SAVED_REPLAY":
        raise RuntimeError("matrix requires the verified frozen model packet")
    if not any(row["canonical_request_id"] == target and row["token_idx"] == token_idx
               for row in packet["target_rows"]):
        raise ValueError("target is not in frozen packet")
    adapter_type = HuginnAdapter if model_key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if model_key == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if sha256_file(imported) != packet["imported_model_source_sha256"]:
        raise RuntimeError("imported model source changed")
    if fp32:
        model.float()
    length = sum(packet["cohort_effective_inputs"][target]["attention_mask"])
    mixed_group = list(range((target // 8) * 8, (target // 8 + 1) * 8))
    fillers = {
        "original_mixed": [rid for rid in mixed_group if rid != target],
        "equal_length": [rid for rid, row in enumerate(packet["cohort_effective_inputs"])
                         if rid != target and sum(row["attention_mask"]) == length][:7],
    }
    if any(len(values) < 7 for values in fillers.values()):
        raise RuntimeError("insufficient frozen fillers for B8 control")
    needed = sorted({target} | set(fillers["original_mixed"]) | set(fillers["equal_length"]))
    snapshots = _canonical_histories(model_key, model, tokenizer, packet, needed, token_idx, not fp32)
    baseline = _one(model_key, model, tokenizer, packet, snapshots, [(target, target)], token_idx)["rows"][0]
    original = next(row for row in packet["target_rows"]
                    if row["canonical_request_id"] == target and row["token_idx"] == token_idx)
    if not fp32 and (baseline["depth"] != original["reference_depth"] or
                     baseline["next_token"] != original["reference_next_token"]):
        raise RuntimeError("matrix B1 baseline does not match frozen reference")
    cached = {(target,): {"rows": [baseline]}}
    lines = []
    for padding, pool in fillers.items():
        for batch_size in (1, 2, 4, 8):
            for label, position in (("first", 0), ("middle", batch_size // 2),
                                    ("last", batch_size - 1)):
                for order in ("fifo", "fixed_permutation"):
                    selected = _selected(target, pool, batch_size, position,
                                         order == "fixed_permutation")
                    key = tuple(selected)
                    if key not in cached:
                        cached[key] = _one(model_key, model, tokenizer, packet, snapshots,
                                           [(rid, rid) for rid in selected], token_idx)
                    row = next(row for row in cached[key]["rows"] if row["request_id"] == target)
                    common = min(len(row["scores"]), len(baseline["scores"]))
                    lines.append({"model": model_key, "precision": "FP32" if fp32 else "BF16",
                                  "target_request_id": target, "token_idx": token_idx,
                                  "padding_condition": padding, "B": batch_size,
                                  "target_position": label, "order": order,
                                  "selected_request_ids": json.dumps(selected),
                                  "target_local_row": selected.index(target),
                                  "target_depth": row["depth"], "B1_depth": baseline["depth"],
                                  "depth_flip": int(row["depth"] != baseline["depth"]),
                                  "target_next_token": row["next_token"],
                                  "B1_next_token": baseline["next_token"],
                                  "next_token_flip": int(row["next_token"] != baseline["next_token"]),
                                  "min_absolute_target_margin": min(abs(m) for m in row["margins"]),
                                  "max_common_gate_score_delta": max(
                                      (abs(row["scores"][i] - baseline["scores"][i])
                                       for i in range(common)), default=0.0),
                                  "h0_same_as_B1": ("NA" if row["h0_sha256"] is None else
                                                    str(row["h0_sha256"] == baseline["h0_sha256"])),
                                  "target_KV_merge_exact": True})
    duplicate = _one(model_key, model, tokenizer, packet, snapshots,
                     [(target, target), (1000 + target, target)], token_idx, clone_h0=True)
    left, right = duplicate["rows"]
    duplicate_result = {
        "bookkeeping_ids_distinct": left["request_id"] != right["request_id"],
        "history_copied_from_same_target": True, "current_input_token_identical": True,
        "Huginn_h0_explicitly_cloned_for_diagnostic": model_key == "huginn",
        "prelude_embedding_exact_equal": duplicate["duplicate_prelude_equal"],
        "depth_exact_equal": left["depth"] == right["depth"],
        "next_token_exact_equal": left["next_token"] == right["next_token"],
        "gate_scores_exact_equal": left["scores"] == right["scores"],
        "final_hidden_exact_equal": left["final_hidden_sha256"] == right["final_hidden_sha256"],
        "left": left, "right": right,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "batch_shape_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=lines[0].keys())
        writer.writeheader()
        writer.writerows(lines)
    (output_dir / "duplicate_target_invariant.json").write_text(json.dumps(duplicate_result, indent=2) + "\n")
    summary = {"status": "MATRIX_RECORDED", "model": model_key,
               "precision": "FP32" if fp32 else "BF16",
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "gpu": gpu, "imported_model_source_sha256": sha256_file(imported),
               "counterexample_manifest_sha256": sha256_file(manifest_path),
               "target_request_id": target, "token_idx": token_idx,
               "frozen_history_source": "per-request B1 live execution, same pinned precision and policy",
               "target_B1_history_sha256": _history_digest(snapshots[target]),
               "filler_ids": fillers, "threshold": packet["threshold"],
               "baseline": baseline, "grid_rows": len(lines),
               "unique_live_batch_compositions": len(cached),
               "depth_flip_grid_rows": sum(row["depth_flip"] for row in lines),
               "next_token_flip_grid_rows": sum(row["next_token_flip"] for row in lines),
               "duplicate_target_invariant": "duplicate_target_invariant.json",
               "matrix": "batch_shape_matrix.csv",
               "note": "Repeated grid rows are controls, not independent samples; no fresh validation rate or speed claim."}
    (output_dir / "matrix_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "model": model_key,
                      "precision": summary["precision"], "grid_rows": len(lines),
                      "depth_flip_grid_rows": summary["depth_flip_grid_rows"],
                      "next_token_flip_grid_rows": summary["next_token_flip_grid_rows"]}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-request", required=True, type=int)
    parser.add_argument("--token-idx", required=True, type=int)
    parser.add_argument("--diagnostic-fp32", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    matrix(args.model, args.role, args.manifest, args.target_request, args.token_idx,
           args.diagnostic_fp32, args.output_dir)


if __name__ == "__main__":
    main()
