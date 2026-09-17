"""Bounded target-row first-divergence trace in the frozen Gate-A prefill."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
from pathlib import Path

import torch

from phase02_runtime.huginn_adapter import HuginnAdapter
from phase02_runtime.ouro_adapter import OuroAdapter
from phase04_runtime.data_contract import sha256_file
from phase04_runtime.validate import gpu_identity
from phase04_resolution.counterexamples import _tensor_sha256


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for part in value:
            found = _first_tensor(part)
            if found is not None:
                return found
    return None


def _traceable(name: str, module: torch.nn.Module, model_key: str) -> bool:
    root = "model.model." if model_key == "ouro" else "model.transformer."
    # named_modules() on a CausalLM uses these names without the local variable prefix.
    root = root.removeprefix("model.")
    if not (name.startswith(root) or name == "lm_head"):
        return False
    kind = module.__class__.__name__
    return (isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) or
            any(word in kind for word in ("Norm", "Attention", "MLP", "Block", "DecoderLayer")))


class TraceBook:
    def __init__(self, model: torch.nn.Module, model_key: str, limit: int = 180):
        self.events = []
        self.active = False
        self.local = 0
        self.width = 0
        self.valid = None
        self.limit = limit
        self.occurrences = {}
        self.full_name = None
        self.full_occurrence = None
        self.full = None
        self.modules = {}
        self.handles = []
        for name, module in model.named_modules():
            if _traceable(name, module, model_key):
                self.modules[name] = module
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def collect(module, args, output):
            if not self.active:
                return
            tensor = _first_tensor(output)
            if tensor is None or tensor.ndim < 2 or tensor.shape[0] != self.width:
                return
            number = self.occurrences.get(name, 0)
            self.occurrences[name] = number + 1
            if self.full_name == name and self.full_occurrence == number:
                source = _first_tensor(args)
                if source is not None:
                    self.full = {"name": name, "occurrence": number,
                                 "input": source.detach().cpu().clone(),
                                 "output": tensor.detach().cpu().clone()}
            if len(self.events) >= self.limit:
                return
            source = _first_tensor(args)
            select = lambda value: value[self.local].detach().cpu().clone()
            self.events.append({"name": name, "occurrence": number,
                                "input": select(source) if source is not None and source.ndim >= 2 and
                                source.shape[0] == self.width else None,
                                "output": select(tensor), "module_type": module.__class__.__name__})
        return collect

    def begin(self, local: int, width: int, valid: torch.Tensor, *, full_name=None, full_occurrence=None):
        self.events = []
        self.occurrences = {}
        self.local, self.width = local, width
        self.valid = valid.detach().cpu().bool()
        self.full_name, self.full_occurrence, self.full = full_name, full_occurrence, None
        self.active = True

    def end(self):
        self.active = False
        return self.events, self.full

    def close(self):
        for handle in self.handles:
            handle.remove()


def _diff(left: torch.Tensor | None, right: torch.Tensor | None, valid: torch.Tensor):
    if left is None or right is None or left.shape != right.shape:
        return None
    if left.ndim >= 2 and left.shape[0] == len(valid):
        left, right = left[valid], right[valid]
    error = left.float() - right.float()
    return {"exact_equal": bool(torch.equal(left, right)),
            "max_abs": float(error.abs().max()) if error.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(error) /
                                 torch.linalg.vector_norm(left.float()).clamp_min(1e-12)),
            "left_sha256": _tensor_sha256(left), "right_sha256": _tensor_sha256(right)}


@torch.inference_mode()
def _one_prefill(model_key, model, tokenizer, ids, mask, canonical_ids, target,
                 book: TraceBook, *, full_name=None, full_occurrence=None):
    rows = len(canonical_ids)
    local = canonical_ids.index(target)
    adapter = (HuginnAdapter(model, tokenizer, rows, state_init="native-seeded", identity_seed=1,
                             canonical_ids=torch.tensor(canonical_ids, device="cuda"))
               if model_key == "huginn" else OuroAdapter(model, tokenizer, rows))
    book.begin(local, rows, mask[local], full_name=full_name, full_occurrence=full_occurrence)
    try:
        adapter.prefill(ids.cuda(), mask.cuda().bool())
    finally:
        events, full = book.end()
    return events, full


def trace(model_key: str, role: str, manifest_path: Path, target: int, packed_size: int,
          diagnostic_fp32: bool, output_dir: Path, limit: int) -> dict:
    gpu = gpu_identity(model_key, role)
    torch.set_num_threads(8)
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS_EXACT_SAVED_REPLAY" or manifest["model"] != model_key:
        raise RuntimeError("counterexample packet is not a verified same-model replay")
    rows = manifest["cohort_effective_inputs"]
    if target not in {row["canonical_request_id"] for row in manifest["target_rows"]}:
        raise ValueError("target not frozen in counterexample manifest")
    if packed_size < 2 or packed_size > len(rows):
        raise ValueError("controlled packed size must be 2..32")
    ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
    mask = torch.tensor([row["attention_mask"] for row in rows], dtype=torch.long)
    adapter_type = HuginnAdapter if model_key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if model_key == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if sha256_file(imported) != manifest["imported_model_source_sha256"]:
        raise RuntimeError("model source changed since frozen counterexample")
    if diagnostic_fp32:
        model.float()
    book = TraceBook(model, model_key, limit)
    first = (target // packed_size) * packed_size
    selected = list(range(first, min(first + packed_size, len(rows))))
    if len(selected) != packed_size or target not in selected:
        raise ValueError("target is absent from selected full packed group")
    ref_events, _ = _one_prefill(model_key, model, tokenizer, ids[target:target+1],
                                 mask[target:target+1], [target], target, book)
    packed_events, _ = _one_prefill(model_key, model, tokenizer, ids[selected],
                                    mask[selected], selected, target, book)
    if [(row["name"], row["occurrence"]) for row in ref_events] != [
            (row["name"], row["occurrence"]) for row in packed_events]:
        raise RuntimeError("B1/packed module event order differs; inspect separately")
    valid = mask[target].bool()
    lines = []
    first_difference = None
    for sequence, (old, new) in enumerate(zip(ref_events, packed_events)):
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
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "first_divergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=lines[0].keys())
        writer.writeheader()
        writer.writerows(lines)
    isolation = None
    if first_difference is not None and first_difference["module_type"] in {"Linear", "Embedding"}:
        name, occurrence = first_difference["module"], first_difference["occurrence"]
        _, ref_full = _one_prefill(model_key, model, tokenizer, ids[target:target+1],
                                   mask[target:target+1], [target], target, book,
                                   full_name=name, full_occurrence=occurrence)
        _, packed_full = _one_prefill(model_key, model, tokenizer, ids[selected],
                                      mask[selected], selected, target, book,
                                      full_name=name, full_occurrence=occurrence)
        if ref_full is None or packed_full is None:
            raise RuntimeError("first operator capture failed")
        module = book.modules[name]
        with torch.inference_mode():
            again_ref = module(ref_full["input"].cuda()).detach().cpu()
            again_packed = module(packed_full["input"].cuda()).detach().cpu()
        local = selected.index(target)
        input_delta = _diff(ref_full["input"][0], packed_full["input"][local], valid)
        output_delta = _diff(again_ref[0], again_packed[local], valid)
        torch.save({"B1_input": ref_full["input"], "packed_input": packed_full["input"],
                    "B1_output": again_ref, "packed_output": again_packed,
                    "module": name, "occurrence": occurrence}, output_dir / "isolated_operator_inputs.pt")
        isolation = {"module": name, "occurrence": occurrence,
                     "B1_input_sha256": _tensor_sha256(ref_full["input"]),
                     "packed_input_sha256": _tensor_sha256(packed_full["input"]),
                     "target_input_difference": input_delta,
                     "target_output_difference": output_delta,
                     "B1_forward_reproduced": bool(torch.equal(ref_full["output"], again_ref)),
                     "packed_forward_reproduced": bool(torch.equal(packed_full["output"], again_packed)),
                     "isolated_inputs": "isolated_operator_inputs.pt"}
    book.close()
    receipt = {"status": "FIRST_PREFILL_DIFFERENCE_IDENTIFIED" if first_difference else "NO_PREFILL_DIFFERENCE_IN_CAPTURED_EVENTS",
               "model": model_key, "target_request_id": target, "precision": str(next(model.parameters()).dtype),
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "counterexample_manifest_sha256": sha256_file(manifest_path),
               "gpu": gpu, "imported_model_source_sha256": sha256_file(imported),
               "packed_rows": selected, "B1_rows": [target],
               "effective_input_sha256": rows[target]["sha256"],
               "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "captured_events": len(lines), "capture_limit": limit,
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
    parser.add_argument("--packed-size", type=int, default=32)
    parser.add_argument("--diagnostic-fp32", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-events", type=int, default=180)
    args = parser.parse_args()
    trace(args.model, args.role, args.manifest, args.target_request, args.packed_size,
          args.diagnostic_fp32, args.output_dir, args.max_events)


if __name__ == "__main__":
    main()
