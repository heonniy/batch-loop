"""Trace a frozen mismatch token through Prelude, live recurrence and Coda."""

from __future__ import annotations

import argparse
import csv
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
from phase04_resolution.trace_prefill import TraceBook, _diff


def _new_adapter(model_key, model, tokenizer, ids):
    return (HuginnAdapter(model, tokenizer, len(ids), state_init="native-seeded", identity_seed=1,
                          canonical_ids=torch.tensor(ids, device="cuda"))
            if model_key == "huginn" else OuroAdapter(model, tokenizer, len(ids)))


def _score_for(events, request_id):
    return [float(event["exit_score"][event["active_request_ids"].index(request_id)])
            for event in events if request_id in event["active_request_ids"]]


@torch.inference_mode()
def _run_target(model_key, model, tokenizer, manifest, target, token_idx, packed,
                book: TraceBook, *, full_name=None, full_occurrence=None,
                require_saved_decision=True):
    cohort = manifest["cohort_effective_inputs"]
    known = manifest["input_tokens_by_boundary"]
    requests = list(range(len(cohort))) if packed else [target]
    target_packet = next(row for row in manifest["target_rows"]
                         if row["canonical_request_id"] == target and row["token_idx"] == token_idx)
    ids = torch.tensor([cohort[rid]["input_ids"] for rid in requests], dtype=torch.long, device="cuda")
    mask = torch.tensor([cohort[rid]["attention_mask"] for rid in requests], dtype=torch.bool, device="cuda")
    adapter = _new_adapter(model_key, model, tokenizer, requests)
    current = adapter.prefill(ids, mask)
    if require_saved_decision and not packed and int(current[0]) != known[0][target]:
        raise RuntimeError("frozen B1 first token did not replay")
    captured = None
    for current_idx in range(token_idx + 1):
        adapter.begin_token()
        groups = [requests[first:first + 8] for first in range(0, len(requests), 8)] if packed else [requests]
        for selected in groups:
            physical_rows = torch.tensor([requests.index(rid) for rid in selected],
                                         dtype=torch.long, device="cuda")
            adapter.backend.set_active(physical_rows)
            token_ids = torch.tensor([[known[current_idx][rid]] for rid in selected],
                                     dtype=torch.long, device="cuda")
            tracing = current_idx == token_idx and target in selected
            if tracing:
                book.begin(selected.index(target), len(selected), torch.ones(1, dtype=torch.bool),
                           full_name=full_name, full_occurrence=full_occurrence)
            embedded, state = adapter.prelude(token_ids)

            def before(recurrence, active_requests, state_before, runtime_adapter):
                del recurrence, state_before, runtime_adapter
                if not tracing:
                    return
                values = active_requests.tolist()
                book.active = target in values
                if book.active:
                    book.width = len(values)
                    book.local = values.index(target)

            result = run_interval(adapter, embedded, state, physical_rows,
                                  torch.tensor(selected, dtype=torch.long, device="cuda"),
                                  exit_source="live", threshold=manifest["threshold"],
                                  instrumentation="profile", debug_ids=True,
                                  debug_pre_hook=before if tracing else None)
            adapter.backend.set_active(physical_rows)
            if tracing:
                book.active = True
                book.width = len(selected)
                book.local = selected.index(target)
            next_ids = adapter.coda_logits(result.final_state).argmax(dim=-1)
            if tracing:
                captured = book.end()
                local = selected.index(target)
                expected_depth = target_packet["compact_depth" if packed else "reference_depth"]
                expected_next = target_packet["compact_next_token" if packed else "reference_next_token"]
                if require_saved_decision and (
                        int(result.exit_depths[local]) != expected_depth or
                        int(next_ids[local]) != expected_next):
                    raise RuntimeError("frozen target decision did not replay")
                expected_scores = target_packet["compact_scores" if packed else "reference_scores"]
                observed_scores = _score_for(result.events, target)
                if require_saved_decision and (len(expected_scores) != len(observed_scores) or any(
                        abs(a - b) > 1e-7 for a, b in zip(expected_scores, observed_scores))):
                    raise RuntimeError("frozen target gate scores did not replay")
            if require_saved_decision and not packed and current_idx < token_idx and (
                    int(next_ids[0]) != known[current_idx + 1][target]):
                raise RuntimeError("frozen B1 prior token did not replay")
        adapter.commit_token()
    if captured is None:
        raise RuntimeError("target was never traced")
    return captured


def trace(model_key: str, role: str, manifest_path: Path, target: int, token_idx: int,
          fp32: bool, output_dir: Path, limit: int) -> dict:
    gpu = gpu_identity(model_key, role)
    torch.set_num_threads(8)
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS_EXACT_SAVED_REPLAY" or manifest["model"] != model_key:
        raise RuntimeError("not a verified same-model frozen packet")
    if not any(row["canonical_request_id"] == target and row["token_idx"] == token_idx
               for row in manifest["target_rows"]):
        raise ValueError("target token is not in the frozen packet")
    adapter_type = HuginnAdapter if model_key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if model_key == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if sha256_file(imported) != manifest["imported_model_source_sha256"]:
        raise RuntimeError("imported model source changed")
    if fp32:
        model.float()
    book = TraceBook(model, model_key, limit)
    ref_events, _ = _run_target(model_key, model, tokenizer, manifest, target, token_idx,
                                False, book, require_saved_decision=not fp32)
    bat_events, _ = _run_target(model_key, model, tokenizer, manifest, target, token_idx,
                                True, book, require_saved_decision=not fp32)
    valid = torch.ones(1, dtype=torch.bool)
    lines = []
    first_difference = None
    for sequence, (old, new) in enumerate(zip(ref_events, bat_events)):
        if (old["name"], old["occurrence"]) != (new["name"], new["occurrence"]):
            break
        out = _diff(old["output"], new["output"], valid)
        inp = _diff(old["input"], new["input"], valid)
        line = {"event": sequence, "module": old["name"], "occurrence": old["occurrence"],
                "module_type": old["module_type"],
                "input_exact_equal": None if inp is None else inp["exact_equal"],
                "input_max_abs": None if inp is None else inp["max_abs"],
                "output_exact_equal": out["exact_equal"],
                "output_max_abs": out["max_abs"], "output_relative_l2": out["relative_l2"],
                "output_B1_sha256": out["left_sha256"],
                "output_packed_sha256": out["right_sha256"]}
        lines.append(line)
        if first_difference is None and not out["exact_equal"]:
            first_difference = line
    if not lines:
        raise RuntimeError("no comparable target-row module events")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "first_divergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=lines[0].keys())
        writer.writeheader()
        writer.writerows(lines)
    isolation = None
    if first_difference is not None and first_difference["module_type"] in {"Linear", "Embedding"}:
        name, occurrence = first_difference["module"], first_difference["occurrence"]
        _, ref_full = _run_target(model_key, model, tokenizer, manifest, target, token_idx,
                                  False, book, full_name=name, full_occurrence=occurrence,
                                  require_saved_decision=not fp32)
        _, bat_full = _run_target(model_key, model, tokenizer, manifest, target, token_idx,
                                  True, book, full_name=name, full_occurrence=occurrence,
                                  require_saved_decision=not fp32)
        if ref_full is None or bat_full is None:
            raise RuntimeError("isolated operator capture failed")
        module = book.modules[name]
        with torch.inference_mode():
            ref_again = module(ref_full["input"].cuda()).detach().cpu()
            bat_again = module(bat_full["input"].cuda()).detach().cpu()
        group = list(range((target // 8) * 8, (target // 8 + 1) * 8))
        local = group.index(target)
        input_delta = _diff(ref_full["input"][0], bat_full["input"][local], valid)
        output_delta = _diff(ref_again[0], bat_again[local], valid)
        torch.save({"B1_input": ref_full["input"], "packed_input": bat_full["input"],
                    "B1_output": ref_again, "packed_output": bat_again,
                    "module": name, "occurrence": occurrence}, output_dir / "isolated_operator_inputs.pt")
        isolation = {"module": name, "occurrence": occurrence,
                     "B1_input_sha256": _tensor_sha256(ref_full["input"]),
                     "packed_input_sha256": _tensor_sha256(bat_full["input"]),
                     "target_input_difference": input_delta,
                     "target_output_difference": output_delta,
                     "B1_forward_reproduced": bool(torch.equal(ref_full["output"], ref_again)),
                     "packed_forward_reproduced": bool(torch.equal(bat_full["output"], bat_again)),
                     "isolated_inputs": "isolated_operator_inputs.pt"}
    book.close()
    receipt = {"status": "FIRST_DECODE_DIFFERENCE_IDENTIFIED" if first_difference else "NO_DECODE_DIFFERENCE_IN_CAPTURED_EVENTS",
               "model": model_key, "target_request_id": target, "token_idx": token_idx,
               "precision": str(next(model.parameters()).dtype),
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "counterexample_manifest_sha256": sha256_file(manifest_path),
               "gpu": gpu, "imported_model_source_sha256": sha256_file(imported),
               "B1_event_count": len(ref_events), "packed_event_count": len(bat_events),
               "comparable_event_count": len(lines), "capture_limit": limit,
               "first_difference": first_difference, "isolated_operator": isolation}
    (output_dir / "per_layer_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "model": model_key,
                      "first_difference": first_difference["module"] if first_difference else None,
                      "isolated": isolation is not None}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-request", required=True, type=int)
    parser.add_argument("--token-idx", required=True, type=int)
    parser.add_argument("--diagnostic-fp32", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-events", type=int, default=180)
    args = parser.parse_args()
    trace(args.model, args.role, args.manifest, args.target_request, args.token_idx,
          args.diagnostic_fp32, args.output_dir, args.max_events)


if __name__ == "__main__":
    main()
