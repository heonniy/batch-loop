"""Reproduce the first Huginn decode SDPA difference with frozen Q/K/V/mask."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from phase02_runtime.huginn_adapter import HuginnAdapter
from phase04_runtime.data_contract import sha256_file
from phase04_runtime.validate import gpu_identity
from phase04_resolution.counterexamples import _tensor_sha256
from phase04_resolution.trace_decode import _run_target
from phase04_resolution.trace_prefill import TraceBook, _diff


def _record_attention(book: TraceBook, destination: dict):
    def observe(q, k, v, mask, output):
        if not book.active or destination:
            return
        destination.update({"local": book.local, "width": book.width,
                            "q": q.detach().cpu().clone(),
                            "k": k.detach().cpu().clone(),
                            "v": v.detach().cpu().clone(),
                            "mask": mask.detach().cpu().clone(),
                            "output": output.detach().cpu().clone()})
    return observe


def diagnose(manifest_path: Path, target: int, token_idx: int, fp32: bool,
             output_dir: Path) -> dict:
    gpu = gpu_identity("huginn", "huginn_primary")
    torch.set_num_threads(8)
    manifest = json.loads(manifest_path.read_text())
    if manifest["model"] != "huginn" or manifest["status"] != "PASS_EXACT_SAVED_REPLAY":
        raise RuntimeError("expected verified Huginn counterexample packet")
    initial = HuginnAdapter.load(1, state_init="native-seeded")
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if sha256_file(imported) != manifest["imported_model_source_sha256"]:
        raise RuntimeError("imported Huginn source changed")
    if fp32:
        model.float()
    book = TraceBook(model, "huginn", limit=0)
    attention = model.transformer.core_block[1].attn
    reference = {}
    attention._phase04_sdpa_observer = _record_attention(book, reference)
    try:
        _run_target("huginn", model, tokenizer, manifest, target, token_idx, False,
                    book, require_saved_decision=not fp32)
        packed = {}
        attention._phase04_sdpa_observer = _record_attention(book, packed)
        _run_target("huginn", model, tokenizer, manifest, target, token_idx, True,
                    book, require_saved_decision=not fp32)
    finally:
        del attention._phase04_sdpa_observer
        book.close()
    if not reference or not packed:
        raise RuntimeError("target SDPA call not captured in both executions")
    b1, bat = 0, packed["local"]
    field_differences = {field: _diff(reference[field][b1], packed[field][bat],
                                     torch.ones(1, dtype=torch.bool))
                         for field in ("q", "k", "v", "mask", "output")}
    with torch.inference_mode(), sdpa_kernel(backends=[SDPBackend.MATH]):
        direct_ref = F.scaled_dot_product_attention(
            reference["q"].cuda(), reference["k"].cuda(), reference["v"].cuda(),
            attn_mask=reference["mask"].cuda(), dropout_p=0.0, is_causal=False).cpu()
        direct_bat = F.scaled_dot_product_attention(
            packed["q"].cuda(), packed["k"].cuda(), packed["v"].cuda(),
            attn_mask=packed["mask"].cuda(), dropout_p=0.0, is_causal=False).cpu()
    output_reproduced = (bool(torch.equal(reference["output"], direct_ref)),
                         bool(torch.equal(packed["output"], direct_bat)))
    direct_delta = _diff(direct_ref[b1], direct_bat[bat], torch.ones(1, dtype=torch.bool))
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"B1": reference, "packed": packed, "direct_B1_output": direct_ref,
                "direct_packed_output": direct_bat}, output_dir / "isolated_attention_inputs.pt")
    result = {"status": "ISOLATED_SDPA_REPRODUCED" if all(output_reproduced)
              else "SDPA_DIRECT_REPLAY_DIFFERS_FROM_ATTACHED_CALL",
              "model": "huginn", "target_request_id": target, "token_idx": token_idx,
              "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "counterexample_manifest_sha256": sha256_file(manifest_path),
              "gpu": gpu, "precision": str(next(model.parameters()).dtype),
              "imported_model_source_sha256": sha256_file(imported),
              "operator": "torch.nn.functional.scaled_dot_product_attention",
              "backend": "SDPBackend.MATH", "dropout_p": 0.0, "is_causal": False,
              "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "target_field_differences": field_differences,
              "direct_output_difference": direct_delta,
              "direct_reproduces_attached_B1": output_reproduced[0],
              "direct_reproduces_attached_packed": output_reproduced[1],
              "B1_input_hashes": {field: _tensor_sha256(reference[field])
                                  for field in ("q", "k", "v", "mask")},
              "packed_input_hashes": {field: _tensor_sha256(packed[field])
                                      for field in ("q", "k", "v", "mask")},
              "isolated_inputs": "isolated_attention_inputs.pt"}
    (output_dir / "attention_operator_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "precision": result["precision"],
                      "q_equal": field_differences["q"]["exact_equal"],
                      "k_equal": field_differences["k"]["exact_equal"],
                      "v_equal": field_differences["v"]["exact_equal"],
                      "mask_equal": field_differences["mask"]["exact_equal"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-request", type=int, default=7)
    parser.add_argument("--token-idx", type=int, default=0)
    parser.add_argument("--diagnostic-fp32", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    diagnose(args.manifest, args.target_request, args.token_idx,
             args.diagnostic_fp32, args.output_dir)


if __name__ == "__main__":
    main()
