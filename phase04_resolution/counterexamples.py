"""Freeze exact Phase-04 Gate-A counterexamples after hash-checked replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import torch

from phase02_runtime.benchmark import _prompts
from phase02_runtime.huginn_adapter import HuginnAdapter
from phase02_runtime.ouro_adapter import OuroAdapter
from phase04_runtime.data_contract import effective_token_hashes, sha256_file
from phase04_runtime.validate import _compact, _reference, gpu_identity


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "phase04/results/20260917T052824Z_2ff6ab1"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _history_signature(backend) -> dict:
    if backend.key_valid is None:
        raise RuntimeError("history has no valid-token mask")
    items = []
    for slot in sorted(backend.history, key=repr):
        key, value = backend.history[slot]
        items.append({"slot": repr(slot), "key_shape": list(key.shape),
                      "key_sha256": _tensor_sha256(key),
                      "value_shape": list(value.shape),
                      "value_sha256": _tensor_sha256(value)})
    return {"key_valid": backend.key_valid[0].bool().tolist(), "slots": items}


def _mismatch_map(validation: dict) -> dict[tuple[int, int], list[dict]]:
    result: dict[tuple[int, int], list[dict]] = {}
    for row in validation["decision_mismatches"]:
        result.setdefault((row["token_idx"], row["request_id"]), []).append(row)
    return result


def _match_old_decisions(old: dict, reference: dict, compact: dict) -> None:
    observed = []
    for key, expected in reference["records"].items():
        actual = compact["records"][key]
        for field in ("depth", "next_token"):
            if expected[field] != actual[field]:
                observed.append((key[0], key[1], field, expected[field], actual[field]))
    saved = [(row["token_idx"], row["request_id"], row["field"],
              row["reference"], row["compact"]) for row in old["decision_mismatches"]]
    if observed != saved:
        raise RuntimeError(f"saved decision mismatch set did not replay: {observed} != {saved}")
    for row in old["decision_mismatches"]:
        key = (row["token_idx"], row["request_id"])
        replay = reference["records"][key]
        if len(replay["scores"]) != len(row["reference_scores"]) or any(
                abs(a - b) > 1e-7 for a, b in zip(replay["scores"], row["reference_scores"])):
            raise RuntimeError(f"saved reference gate trajectory did not replay: {key}")


def freeze(model_key: str, role: str, output_dir: Path) -> dict:
    gpu = gpu_identity(model_key, role)
    torch.set_num_threads(8)
    old_dir = OLD / f"gate_a_{model_key}_gpu{gpu['physical_index']}_n32_b8_t8_bf16"
    old_policy = json.loads((old_dir / "policy_manifest.json").read_text())
    old_validation = json.loads((old_dir / "runtime_validation.json").read_text())
    old_rng = json.loads((old_dir / "rng_identity_report.json").read_text())
    if old_policy["dtype"] != "torch.bfloat16" or old_validation["status"] != "FAIL":
        raise RuntimeError("counterexample source is not the saved primary BF16 failure")
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter_type = HuginnAdapter if model_key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if model_key == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if sha256_file(imported) != old_policy["actually_imported_source_sha256"]:
        raise RuntimeError("imported model source differs from frozen Gate-A source")
    prompts = _prompts(32, heterogeneous=True)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    hashes = effective_token_hashes(encoded["input_ids"], encoded["attention_mask"])
    if hashes != old_policy["effective_input_hashes"]:
        raise RuntimeError("effective token IDs differ from saved Gate-A hashes")
    mismatches = _mismatch_map(old_validation)
    wanted = set(mismatches) | {(0, rid) for rid in range(32)}
    histories = {}

    def before_token(token_idx, request_id, adapter):
        if (token_idx, request_id) in wanted:
            histories[(token_idx, request_id)] = _history_signature(adapter.backend)

    reference, input_tokens, ref_adapters = _reference(
        model_key, model, tokenizer, prompts, 8, old_policy["threshold"], 1,
        old_policy["cohort_tokenized_width"], before_token=before_token)
    compact, _ = _compact(model_key, model, tokenizer, prompts, input_tokens,
                          old_policy["threshold"], 1, 8, "fifo", capture_events=True)
    _match_old_decisions(old_validation, reference, compact)
    controls = sorted(
        ((min(abs(margin) for margin in reference["records"][(0, rid)]["margins"]), (0, rid))
         for rid in range(32) if (0, rid) not in mismatches), reverse=True,
    )[:2]
    targets = sorted(set(mismatches) | {key for _, key in controls})
    packet = []
    for token_idx, request_id in targets:
        key = (token_idx, request_id)
        ref = reference["records"][key]
        bat = compact["records"][key]
        first = (request_id // 8) * 8
        events = compact["events"][(token_idx, first)]
        adapter = ref_adapters[request_id][0]
        current_hashes = {f"{rid}:{pos}": digest for (rid, pos), digest in
                          getattr(adapter, "state_hashes", {}).items()}
        saved_hashes = {name: digest for name, digest in old_rng["reference_state_hashes"].items()
                        if name.startswith(f"{request_id}:")}
        for name, digest in saved_hashes.items():
            if current_hashes.get(name) != digest:
                raise RuntimeError(f"saved h0 identity did not replay: {name}")
        packet.append({
            "kind": "decision_mismatch" if key in mismatches else "far_margin_control",
            "canonical_request_id": request_id, "token_idx": token_idx,
            "effective_input_sha256": hashes[request_id],
            "effective_input_ids": encoded["input_ids"][request_id].tolist(),
            "effective_attention_mask": encoded["attention_mask"][request_id].tolist(),
            "previous_token_ids": [input_tokens[t][request_id] for t in range(token_idx)],
            "current_input_token_id": input_tokens[token_idx][request_id],
            "reference_depth": ref["depth"], "compact_depth": bat["depth"],
            "reference_next_token": ref["next_token"], "compact_next_token": bat["next_token"],
            "reference_scores": ref["scores"], "compact_scores": bat["scores"],
            "reference_margins": ref["margins"], "compact_margins": bat["margins"],
            "min_absolute_reference_margin": min(abs(m) for m in ref["margins"]),
            "h0_identity_hashes": saved_hashes,
            "prior_history": histories[key],
            "reference_active_row_order": [[request_id] for _ in ref["scores"]],
            "compact_initial_group_order": list(range(first, first + 8)),
            "compact_active_row_order": [event["active_request_ids"] for event in events],
            "saved_decision_receipts": mismatches.get(key, []),
        })
    result = {
        "status": "PASS_EXACT_SAVED_REPLAY", "model": model_key,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "old_gate_a_source_sha": old_policy["source_sha"],
        "old_policy_manifest_sha256": sha256_file(old_dir / "policy_manifest.json"),
        "old_runtime_validation_sha256": sha256_file(old_dir / "runtime_validation.json"),
        "model_revision": old_policy["model_revision"],
        "model_config_sha256": old_policy["model_config_sha256"],
        "imported_model_source_sha256": sha256_file(imported),
        "tokenizer_snapshot_sha256": old_policy["tokenizer_snapshot_sha256"],
        "gpu": gpu, "threshold": old_policy["threshold"], "rmax": old_policy["rmax"],
        "kv_policy": old_policy["kv_policy"], "dtype": old_policy["dtype"],
        "cohort_tokenized_width": old_policy["cohort_tokenized_width"],
        "cohort_effective_inputs": [
            {"request_id": rid, "input_ids": encoded["input_ids"][rid].tolist(),
             "attention_mask": encoded["attention_mask"][rid].tolist(), "sha256": hashes[rid]}
            for rid in range(32)],
        "input_tokens_by_boundary": input_tokens,
        "target_rows": packet,
        "original_evidence": str(old_dir.relative_to(ROOT)),
    }
    (output_dir / "COUNTEREXAMPLE_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "model": model_key,
                      "mismatch_targets": len(mismatches), "controls": len(controls)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    freeze(args.model, args.role, args.output_dir)


if __name__ == "__main__":
    main()
