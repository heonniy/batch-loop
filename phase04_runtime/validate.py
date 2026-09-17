"""Bounded same-policy B=1 reference versus compact live-exit GPU validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from phase02_runtime.benchmark import _prompts
from phase02_runtime.common import run_interval
from phase02_runtime.huginn_adapter import HuginnAdapter, HUGINN_SNAPSHOT
from phase02_runtime.ouro_adapter import OuroAdapter, OURO14_SNAPSHOT
from phase04_runtime.data_contract import effective_token_hashes, sha256_file


ASSIGNMENT = Path(__file__).resolve().parents[1] / "phase04/GPU_ASSIGNMENT_20260917.json"


def gpu_identity(model_key: str, role: str) -> dict:
    assignment = json.loads(ASSIGNMENT.read_text())["device_map"]
    if role not in assignment or not role.startswith(model_key + "_"):
        raise ValueError("role is not an authorized model/GPU assignment")
    physical = assignment[role]["physical_index"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical) or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly the mapped physical GPU must be visible")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", str(physical), "--query-gpu=index,uuid,name,memory.total",
         "--format=csv,noheader"], text=True).strip()
    fields = [part.strip() for part in query.split(",")]
    if len(fields) != 4 or int(fields[0]) != physical or fields[1] != assignment[role]["uuid"]:
        raise RuntimeError(f"mapped GPU UUID/index mismatch: {query}")
    if "H100" not in fields[2]:
        raise RuntimeError("authorized GPU is not the expected H100")
    return {"physical_index": physical, "uuid": fields[1], "name": fields[2],
            "memory_total": fields[3], "CUDA_VISIBLE_DEVICES": visible, "cuda_ordinal": 0,
            "role": role, "assignment_sha256": sha256_file(ASSIGNMENT)}


def _adapter(model_key: str, model, tokenizer, ids: list[int], seed: int, mode: str):
    if model_key == "huginn":
        return HuginnAdapter(model, tokenizer, len(ids), state_init="native-seeded",
                             identity_seed=seed, canonical_ids=torch.tensor(ids, device="cuda"),
                             instrumentation=mode, capture_state_hashes=True)
    return OuroAdapter(model, tokenizer, len(ids), instrumentation=mode)


def _capture(store: dict, token_idx: int):
    def hook(recurrence, request_ids, after, adapter):
        del adapter
        for local, request_id in enumerate(request_ids.tolist()):
            store[(token_idx, int(request_id), recurrence)] = after[local].detach().float().cpu()
    return hook


@torch.inference_mode()
def _reference(model_key: str, model, tokenizer, prompts: list[str], tokens: int,
               threshold: float, seed: int, cohort_width: int | None) -> tuple[dict, list[list[int]], list]:
    records = {}
    input_tokens = [[0] * len(prompts) for _ in range(tokens)]
    adapters = []
    prefill_logits = {}
    for request_id, prompt in enumerate(prompts):
        adapter = _adapter(model_key, model, tokenizer, [request_id], seed, "profile")
        encoded = tokenizer([prompt], return_tensors="pt",
                            padding="max_length" if cohort_width is not None else True,
                            truncation=True, max_length=cohort_width or 64)
        current = adapter.prefill(encoded["input_ids"].cuda(),
                                  encoded["attention_mask"].cuda().bool())
        prefill_logits[request_id] = adapter.prefill_logits[0, -1].detach().cpu().float()
        adapters.append((adapter, current))
    states = {}
    for token_idx in range(tokens):
        for request_id, (adapter, current) in enumerate(adapters):
            row = torch.tensor([0], dtype=torch.long, device="cuda")
            input_tokens[token_idx][request_id] = int(current.item())
            adapter.begin_token()
            adapter.backend.set_active(row)
            embedded, state = adapter.prelude(current)
            scores = []
            margins = []
            for depth in range(1, adapter.rmax + 1):
                after = adapter.recurrent_step(state, embedded, depth)
                states[(token_idx, request_id, depth)] = after[0].detach().float().cpu()
                mask = adapter.live_exit(state, after, row, depth, threshold)
                scores.append(float(adapter.last_exit_decision.score[0]))
                margins.append(float(adapter.last_exit_decision.margin[0]))
                state = after
                if bool(mask.item()):
                    adapter.backend.on_exit(row, depth)
                    break
            adapter.backend.set_active(row)
            logits = adapter.coda_logits(state)[:, -1, :]
            next_token = logits.argmax(dim=-1).view(1, 1)
            adapter.commit_token()
            records[(token_idx, request_id)] = {
                "depth": depth, "next_token": int(next_token.item()),
                "scores": scores, "margins": margins, "logits": logits[0].detach().cpu().float(),
            }
            adapters[request_id] = (adapter, next_token)
    return {"records": records, "states": states, "prefill_logits": prefill_logits}, input_tokens, adapters


@torch.inference_mode()
def _compact(model_key: str, model, tokenizer, prompts: list[str], input_tokens: list[list[int]],
             threshold: float, seed: int, batch_size: int, ordering: str) -> tuple[dict, object]:
    n = len(prompts)
    adapter = _adapter(model_key, model, tokenizer, list(range(n)), seed, "profile")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    adapter.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
    prefill_logits = {request_id: adapter.prefill_logits[request_id, -1].detach().cpu().float()
                      for request_id in range(n)}
    records = {}
    states = {}
    widths = {}
    for token_idx, known in enumerate(input_tokens):
        adapter.begin_token()
        order = list(range(n)) if ordering == "fifo" else list(reversed(range(n)))
        for first in range(0, n, batch_size):
            selected = order[first:first + batch_size]
            rows = torch.tensor(selected, dtype=torch.long, device="cuda")
            adapter.backend.set_active(rows)
            ids = torch.tensor([known[rid] for rid in selected], dtype=torch.long,
                               device="cuda").unsqueeze(1)
            embedded, initial_state = adapter.prelude(ids)
            result = run_interval(
                adapter, embedded, initial_state, rows, rows, exit_source="live",
                threshold=threshold, instrumentation="profile", debug_ids=True,
                debug_hook=_capture(states, token_idx),
            )
            adapter.backend.set_active(rows)
            logits = adapter.coda_logits(result.final_state)[:, -1, :]
            next_ids = logits.argmax(dim=-1)
            widths[(token_idx, first)] = [event["active_batch_size"] for event in result.events]
            for local, request_id in enumerate(selected):
                records[(token_idx, request_id)] = {
                    "depth": int(result.exit_depths[local]),
                    "next_token": int(next_ids[local]),
                    "logits": logits[local].detach().cpu().float(),
                    "scores": [float(event["exit_score"][event["active_request_ids"].index(request_id)])
                               for event in result.events if request_id in event["active_request_ids"]],
                    "margins": [float(event["exit_margin"][event["active_request_ids"].index(request_id)])
                                for event in result.events if request_id in event["active_request_ids"]],
                }
        adapter.commit_token()
    return {"records": records, "states": states, "widths": widths,
            "prefill_logits": prefill_logits}, adapter


def _kv_compare(reference_adapters: list, compact_adapter, atol: float, rtol: float
                ) -> tuple[float, float, int, list]:
    mismatches = []
    maximum = 0.0
    maximum_relative_l2 = 0.0
    structural_mismatches = 0
    for request_id, (reference, _) in enumerate(reference_adapters):
        if set(reference.backend.history) != set(compact_adapter.backend.history):
            structural_mismatches += 1
            mismatches.append({"request_id": request_id, "reason": "KV slot set mismatch"})
            continue
        for slot, ref_pair in reference.backend.history.items():
            bat_pair = compact_adapter.backend.history[slot]
            ref_valid = reference.backend.key_valid[0].bool()
            bat_valid = compact_adapter.backend.key_valid[request_id].bool()
            for kind, (ref, bat) in enumerate(zip(ref_pair, bat_pair)):
                ref_seq = ref[0, :, :len(ref_valid), :][:, ref_valid, :].float()
                bat_seq = bat[request_id, :, :len(bat_valid), :][:, bat_valid, :].float()
                if ref_seq.shape != bat_seq.shape:
                    structural_mismatches += 1
                    mismatches.append({"request_id": request_id, "slot": str(slot),
                                       "kind": kind, "reason": "shape mismatch"})
                    continue
                error = (ref_seq - bat_seq).abs()
                maximum = max(maximum, float(error.max()))
                maximum_relative_l2 = max(maximum_relative_l2,
                                          float(torch.linalg.vector_norm(error) /
                                                torch.linalg.vector_norm(ref_seq).clamp_min(1e-12)))
                if not torch.allclose(ref_seq, bat_seq, atol=atol, rtol=rtol) and len(mismatches) < 20:
                    mismatches.append({"request_id": request_id, "slot": str(slot),
                                       "kind": kind, "max_abs_error": float(error.max())})
    return maximum, maximum_relative_l2, structural_mismatches, mismatches


def _compare(reference: dict, compact: dict, reference_adapters: list, compact_adapter,
             atol: float, rtol: float) -> dict:
    mismatches = []
    max_hidden = max_logits = max_gate = 0.0
    max_hidden_relative_l2 = max_logit_relative_l2 = max_prefill_relative_l2 = 0.0
    depth_disagreements = token_disagreements = 0
    decision_mismatches = []
    min_abs_gate_margin = float("inf")
    near_threshold_scores = []
    near_threshold_count = 0
    prefill_errors = {}
    for request_id, old in reference["prefill_logits"].items():
        new = compact["prefill_logits"][request_id]
        error = (old - new).abs()
        prefill_errors[request_id] = float(error.max())
        max_prefill_relative_l2 = max(max_prefill_relative_l2,
                                      float(torch.linalg.vector_norm(error) /
                                            torch.linalg.vector_norm(old).clamp_min(1e-12)))
        if not torch.allclose(old, new, atol=atol, rtol=rtol) and len(mismatches) < 100:
            mismatches.append({"request_id": request_id, "field": "prefill_logits",
                               "max_abs_error": float(error.max())})
    for key, expected in reference["records"].items():
        actual = compact["records"][key]
        for field in ("depth", "next_token"):
            if actual[field] != expected[field]:
                if field == "depth":
                    depth_disagreements += 1
                else:
                    token_disagreements += 1
                decision_mismatches.append({"token_idx": key[0], "request_id": key[1],
                                            "field": field, "reference": expected[field],
                                            "compact": actual[field],
                                            "reference_scores": expected["scores"],
                                            "compact_scores": actual["scores"],
                                            "reference_margins": expected["margins"],
                                            "compact_margins": actual["margins"]})
            if actual[field] != expected[field] and len(mismatches) < 100:
                mismatches.append({"token_idx": key[0], "request_id": key[1],
                                   "field": field, "reference": expected[field], "compact": actual[field],
                                   "reference_exit_margin": expected["margins"][-1]})
        logits_error = (actual["logits"] - expected["logits"]).abs()
        max_logits = max(max_logits, float(logits_error.max()))
        max_logit_relative_l2 = max(max_logit_relative_l2,
                                    float(torch.linalg.vector_norm(logits_error) /
                                          torch.linalg.vector_norm(expected["logits"]).clamp_min(1e-12)))
        if not torch.allclose(actual["logits"], expected["logits"], atol=atol, rtol=rtol) and len(mismatches) < 100:
            mismatches.append({"token_idx": key[0], "request_id": key[1],
                               "field": "logits", "max_abs_error": float(logits_error.max())})
        for depth, (old, new) in enumerate(zip(expected["scores"], actual["scores"]), 1):
            max_gate = max(max_gate, abs(old - new))
            min_abs_gate_margin = min(min_abs_gate_margin, abs(expected["margins"][depth - 1]))
            if abs(expected["margins"][depth - 1]) <= 2 * abs(old - new):
                near_threshold_count += 1
                if len(near_threshold_scores) < 100:
                    near_threshold_scores.append({"token_idx": key[0], "request_id": key[1],
                                                  "recurrence": depth, "reference_score": old,
                                                  "compact_score": new,
                                                  "reference_margin": expected["margins"][depth - 1],
                                                  "same_exit_decision": expected["depth"] == actual["depth"]})
            if abs(old - new) > atol + rtol * abs(old) and len(mismatches) < 100:
                mismatches.append({"token_idx": key[0], "request_id": key[1],
                                   "field": "gate_score", "recurrence": depth,
                                   "reference": old, "compact": new})
    for key, old in reference["states"].items():
        new = compact["states"].get(key)
        if new is None:
            if len(mismatches) < 100:
                mismatches.append({"token_idx": key[0], "request_id": key[1],
                                   "field": "state_missing", "recurrence": key[2]})
            continue
        error = (old - new).abs()
        max_hidden = max(max_hidden, float(error.max()))
        max_hidden_relative_l2 = max(max_hidden_relative_l2,
                                     float(torch.linalg.vector_norm(error) /
                                           torch.linalg.vector_norm(old).clamp_min(1e-12)))
        if not torch.allclose(old, new, atol=atol, rtol=rtol) and len(mismatches) < 100:
            mismatches.append({"token_idx": key[0], "request_id": key[1],
                               "field": "hidden", "recurrence": key[2],
                               "max_abs_error": float(error.max())})
    h0_mismatches = []
    if getattr(compact_adapter, "identity", None) is not None:
        for request_id, (adapter, _) in enumerate(reference_adapters):
            for (_, position), digest in adapter.state_hashes.items():
                if compact_adapter.state_hashes.get((request_id, position)) != digest:
                    h0_mismatches.append({"request_id": request_id, "position": position})
    kv_max, kv_relative_l2, kv_structural, kv_mismatches = _kv_compare(
        reference_adapters, compact_adapter, atol, rtol)
    strict_status = "PASS" if not mismatches and not h0_mismatches and not kv_mismatches else "FAIL"
    semantic_pass = (depth_disagreements == 0 and token_disagreements == 0 and
                     not h0_mismatches and kv_structural == 0 and
                     max_hidden_relative_l2 <= 0.03 and max_logit_relative_l2 <= 0.03 and
                     max_prefill_relative_l2 <= 0.03 and kv_relative_l2 <= 0.03 and
                     max_gate <= 0.02)
    return {"status": "PASS" if semantic_pass else "FAIL",
            "strict_elementwise_status": strict_status,
            "semantic_release_rule": "phase04/GATE_A_PARITY_RULE.md",
            "mismatch_count": len(mismatches), "first_mismatches": mismatches[:30],
            "h0_mismatches": h0_mismatches[:30], "kv_mismatches": kv_mismatches,
            "max_hidden_abs_error": max_hidden, "max_logit_abs_error": max_logits,
            "max_prefill_logit_abs_error_by_request": prefill_errors,
            "max_hidden_relative_l2": max_hidden_relative_l2,
            "max_logit_relative_l2": max_logit_relative_l2,
            "max_prefill_logit_relative_l2": max_prefill_relative_l2,
            "max_kv_relative_l2": kv_relative_l2,
            "kv_structural_mismatch_count": kv_structural,
            "depth_disagreements": depth_disagreements,
            "next_token_disagreements": token_disagreements,
            "decision_mismatches": decision_mismatches,
            "first_decision_divergence": (min(decision_mismatches,
                                              key=lambda item: (item["token_idx"], item["request_id"]))
                                          if decision_mismatches else None),
            "min_absolute_reference_gate_margin": min_abs_gate_margin,
            "near_threshold_score_count": near_threshold_count,
            "near_threshold_scores": near_threshold_scores,
            "max_gate_abs_error": max_gate, "max_kv_abs_error": kv_max,
            "active_widths": {f"{token},{first}": values for (token, first), values in compact["widths"].items()}}


def validate(args: argparse.Namespace) -> dict:
    torch.set_num_threads(8)
    gpu = gpu_identity(args.model, args.role)
    if args.num_rows < 1 or args.batch_size < 1 or args.num_tokens < 1:
        raise ValueError("rows, B, and tokens must be positive")
    if args.batch_size > args.num_rows:
        raise ValueError("batch size cannot exceed cohort size")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.perf_counter()
    adapter_type = HuginnAdapter if args.model == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if args.model == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    if args.diagnostic_fp32:
        model.float()
    imported_module = importlib.import_module(model.__class__.__module__)
    imported_path = Path(imported_module.__file__).resolve()
    snapshot = HUGINN_SNAPSHOT if args.model == "huginn" else OURO14_SNAPSHOT
    snapshot_source = snapshot / ("raven_modeling_minimal.py" if args.model == "huginn" else "modeling_ouro.py")
    config = model.config.to_dict()
    prompts = _prompts(args.num_rows, heterogeneous=not args.homogeneous_prompts)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    policy = {"model": args.model, "model_revision": adapter_type.model_revision,
              "source_sha": source_sha, "argv": sys.argv, "gpu": gpu,
              "cached_model_source": str(snapshot_source),
              "cached_model_source_sha256": sha256_file(snapshot_source),
              "actually_imported_source": str(imported_path),
              "actually_imported_source_sha256": sha256_file(imported_path),
              "model_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
              "tokenizer_snapshot_sha256": sha256_file(snapshot / "tokenizer_config.json"),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": importlib.import_module("transformers").__version__,
              "dtype": str(next(model.parameters()).dtype),
              "attention_backend": "SDPA math head96" if args.model == "huginn" else "eager",
              "threshold": args.threshold, "rmax": initial.rmax, "kv_policy": initial.kv_policy,
              "h0_policy": "request-keyed truncated normal" if args.model == "huginn" else "not applicable",
              "test_time_noise": getattr(model.config, "test_time_noise", None),
              "prompt_source": ("distinct synthetic equal-length diagnostic probes"
                                if args.homogeneous_prompts else "distinct synthetic mixed-length correctness probes"),
              "diagnostic_fp32": args.diagnostic_fp32,
              "prompt_cap": 64, "reference_padding": args.reference_padding,
              "semantic_release_rule_sha256": sha256_file(
                  Path(__file__).resolve().parents[1] / "phase04/GATE_A_PARITY_RULE.md"),
              "cohort_tokenized_width": int(encoded["input_ids"].shape[1]),
              "effective_input_hashes": effective_token_hashes(
                  encoded["input_ids"], encoded["attention_mask"]),
              "fixed_horizon": args.num_tokens, "EOS_handling": "not applied in correctness probe"}
    (args.output_dir / "policy_manifest.json").write_text(json.dumps(policy, indent=2) + "\n")
    reference, known_tokens, reference_adapters = _reference(
        args.model, model, tokenizer, prompts, args.num_tokens, args.threshold, args.seed,
        int(encoded["input_ids"].shape[1]) if args.reference_padding == "cohort" else None)
    compact, compact_adapter = _compact(
        args.model, model, tokenizer, prompts, known_tokens, args.threshold, args.seed,
        args.batch_size, args.ordering)
    comparison = _compare(reference, compact, reference_adapters, compact_adapter,
                          args.atol, args.rtol)
    comparison.update({"execution_kind": "real-no-refill-runtime-correctness",
                       "model": args.model, "num_rows": args.num_rows, "batch_size": args.batch_size,
                       "num_tokens": args.num_tokens, "ordering": args.ordering,
                       "reference_padding": args.reference_padding,
                       "input_policy": "B1-generated token IDs teacher-forced into live compact gate",
                       "depth_policy": "live threshold; never forced",
                       "atol": args.atol, "rtol": args.rtol,
                       "elapsed_wall_s": time.perf_counter() - started,
                       "reference_work": sum(row["depth"] for row in reference["records"].values()),
                       "compact_work": sum(row["depth"] for row in compact["records"].values()),
                       "source_sha": source_sha, "gpu": gpu})
    (args.output_dir / "runtime_validation.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (args.output_dir / "rng_identity_report.json").write_text(json.dumps({
        "status": "PASS" if not comparison["h0_mismatches"] else "FAIL",
        "model": args.model, "seed": args.seed,
        "reference_state_hashes": {f"{rid}:{pos}": digest
                                   for adapter, _ in reference_adapters
                                   for (rid, pos), digest in getattr(adapter, "state_hashes", {}).items()},
        "compact_state_hashes": {f"{rid}:{pos}": digest
                                 for (rid, pos), digest in getattr(compact_adapter, "state_hashes", {}).items()},
        "mismatches": comparison["h0_mismatches"],
    }, indent=2) + "\n")
    print(json.dumps({"status": comparison["status"], "model": args.model,
                      "role": args.role, "rows": args.num_rows, "tokens": args.num_tokens,
                      "mismatches": comparison["mismatch_count"],
                      "h0_mismatches": len(comparison["h0_mismatches"]),
                      "kv_mismatches": len(comparison["kv_mismatches"]),
                      "elapsed_s": round(comparison["elapsed_wall_s"], 1)}), flush=True)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("huginn", "ouro"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-rows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ordering", choices=("fifo", "reverse"), default="fifo")
    parser.add_argument("--reference-padding", choices=("native", "cohort"), default="cohort")
    parser.add_argument("--homogeneous-prompts", action="store_true")
    parser.add_argument("--diagnostic-fp32", action="store_true")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.threshold is None:
        args.threshold = 0.12 if args.model == "huginn" else 0.2
    validate(args)


if __name__ == "__main__":
    main()
