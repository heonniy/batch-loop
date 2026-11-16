"""Collect real fixed-depth recurrent hidden-state traces (GPU, explicit opt-in).

Uses the EXISTING phase02_runtime model adapters, without changing them.
These adapters are an internal experimental baseline and not guaranteed to
match original HF-native generation: native parity is an explicit E0 gate.

WARNING: This collector always runs the full adapter.rmax recurrent steps and
commits its *native adapter KV policy*. Do not use its traces as a full-loop-KV
reference, a parallel execution benchmark, or a quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch


def _load_records(path: Path, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item.get("prompt"), str) or not item["prompt"]:
                    raise ValueError("each JSONL row needs a nonempty prompt")
                rows.append(item)
                if len(rows) == n:
                    break
    if len(rows) != n:
        raise ValueError(f"expected {n} prompts, got {len(rows)}")
    teacher_flags = [("decode_token_ids" in r) for r in rows]
    if any(teacher_flags) and not all(teacher_flags):
        raise ValueError("all rows must have decode_token_ids, or none")
    return rows


@torch.inference_mode()
def collect(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_gpu:
        raise RuntimeError("Explicit --allow-gpu required; no GPU jobs are launched by default")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.batch < 1 or args.decode_steps < 1 or args.prompt_cap < 1:
        raise ValueError("batch/decode_steps/prompt_cap must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.family == "huginn":
        from phase02_runtime.huginn_adapter import HuginnAdapter as Adapter

        options: dict[str, Any] = {
            "state_init": "native-seeded",
            "identity_seed": args.seed,
            "canonical_ids": torch.arange(args.batch, device="cuda", dtype=torch.long),
            "instrumentation": "benchmark",
        }
    elif args.family == "ouro14":
        from phase02_runtime.ouro_adapter import OuroAdapter as Adapter
        options = {"instrumentation": "benchmark"}
    else:
        raise ValueError(args.family)

    if args.snapshot:
        options["snapshot"] = Path(args.snapshot)
    adapter = Adapter.load(num_rows=args.batch, **options)
    records = _load_records(args.input_jsonl, args.batch)
    tokenizer = adapter.tokenizer
    tokens_in = tokenizer(
        [r["prompt"] for r in records],
        padding=True,
        truncation=True,
        max_length=args.prompt_cap,
        return_tensors="pt",
    )
    token_ids = tokens_in["input_ids"].to("cuda")
    mask = tokens_in["attention_mask"].to("cuda").bool()

    teacher = "decode_token_ids" in records[0]
    if teacher:
        for i, r in enumerate(records):
            ids = r["decode_token_ids"]
            if not isinstance(ids, list) or len(ids) < args.decode_steps:
                raise ValueError(f"row {i} needs >= {args.decode_steps} decode token IDs")
            if any(not isinstance(v, int) or v < 0 for v in ids[:args.decode_steps]):
                raise ValueError(f"invalid decode_token_ids in row {i}")
        forced = torch.tensor(
            [r["decode_token_ids"][:args.decode_steps] for r in records],
            dtype=torch.long,
            device="cuda",
        )
    else:
        forced = None

    initial_next = adapter.prefill(token_ids, mask)
    next_ids = forced[:, :1] if teacher else initial_next

    all_states: list[torch.Tensor] = []
    final_predictions: list[torch.Tensor] = []
    executed_token_ids: list[torch.Tensor] = []

    for time_idx in range(args.decode_steps):
        executed_token_ids.append(next_ids.squeeze(1).detach().cpu())
        adapter.begin_token()
        embedded, h = adapter.prelude(next_ids)
        # Store BF16 exactly as available in the model runtime; no conversion
        # to FP32 before recording, avoiding an incorrect 'high precision' claim.
        step_states = [h[:, -1, :].detach().to(device="cpu", dtype=torch.bfloat16)]
        for depth in range(1, adapter.rmax + 1):
            h = adapter.recurrent_step(h, embedded, recurrence=depth)
            step_states.append(h[:, -1, :].detach().to(device="cpu", dtype=torch.bfloat16))
        logits = adapter.coda_logits(h)
        predicted = logits.argmax(dim=-1)
        final_predictions.append(predicted.squeeze(1).detach().cpu())
        adapter.commit_token()
        all_states.append(torch.stack(step_states))
        if time_idx + 1 < args.decode_steps:
            next_ids = forced[:, time_idx + 1:time_idx + 2] if teacher else predicted

    states = torch.stack(all_states)  # [T,D+1,B,H]
    payload: dict[str, Any] = {
        "states": states,
        "token_ids": torch.stack(executed_token_ids),
        "prediction_ids": torch.stack(final_predictions),
        "metadata": {
            "source": "phase02_runtime adapter, full adapter.rmax decode",
            "family": args.family,
            "model_revision": adapter.model_revision,
            "kv_policy": adapter.kv_policy,
            "rmax": adapter.rmax,
            "num_rows": args.batch,
            "decode_steps": args.decode_steps,
            "prompt_cap": args.prompt_cap,
            "seed": args.seed,
            "teacher_forced": teacher,
            "precision": "bf16",
            "prefetch": "off/no external prefetch",
            "author_native_parity": "UNVALIDATED",
            "gpu_name": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "purpose": "E2 offline diagnostics only; not physical speedup",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return {
        "status": "trace_saved_not_parity_validated",
        "output": str(args.output),
        "shape": list(states.shape),
        "metadata": payload["metadata"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["huginn", "ouro14"], required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--prompt-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), indent=2, default=str))


if __name__ == "__main__":
    main()
