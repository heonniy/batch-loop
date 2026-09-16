#!/usr/bin/env python3
"""Bounded B=1 Huginn KV-policy output and HBM probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from poc import force_fixed_huginn_decode, load_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--policies", default="full,latest-m4")
    ap.add_argument("--dataset", default="alpaca", choices=["alpaca", "sharegpt"])
    ap.add_argument("--num-requests", type=int, default=4)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.26)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    prompts = load_prompts(args.dataset, args.num_requests, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    if not force_fixed_huginn_decode(model):
        raise RuntimeError("Huginn fixed-decode adapter could not find model._get_stops")

    gc = GenerationConfig.from_model_config(model.config)
    gc.do_sample = False
    gc.max_new_tokens = args.max_new_tokens
    gc.min_new_tokens = args.max_new_tokens
    gc.return_dict_in_generate = True
    gc.pad_token_id = tokenizer.pad_token_id

    rows = []
    outputs: dict[str, list[list[int] | None]] = {}
    for policy in [x.strip() for x in args.policies.split(",") if x.strip()]:
        policy_outputs: list[list[int] | None] = []
        for request_id, prompt in enumerate(prompts):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to("cuda")
            prompt_width = int(enc["input_ids"].shape[1])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            status = "OK"
            token_ids = None
            try:
                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        generation_config=gc,
                        max_new_tokens=args.max_new_tokens,
                        tokenizer=tokenizer,
                        criterion="latent-diff",
                        exit_threshold=args.threshold,
                        num_steps=args.max_steps,
                        min_steps=1,
                        check_criterion_every_n_steps=1,
                        cache_lookup_strategy=policy,
                    )
                torch.cuda.synchronize()
                token_ids = out.sequences[0, prompt_width:].detach().cpu().tolist()
            except torch.cuda.OutOfMemoryError as exc:
                status = f"OOM:{exc}"
                torch.cuda.empty_cache()
            except Exception as exc:
                status = f"ERROR:{type(exc).__name__}:{exc}"
            digest = (
                hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()
                if token_ids is not None
                else None
            )
            row = {
                "model": args.model,
                "model_revision": args.model_revision or "default",
                "family": "huginn",
                "dataset": args.dataset,
                "request_id": request_id,
                "batch_size": 1,
                "threshold": args.threshold,
                "kv_policy": policy,
                "rmax": args.max_steps,
                "prompt_tokens": prompt_width,
                "generated_tokens": len(token_ids) if token_ids is not None else 0,
                "output_sha256": digest,
                "status": status,
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
                "evidence_kind": "HUGINN_B1_KV_POLICY_OUTPUT_PROBE",
                "huginn_private_stop_tokens_suppressed": True,
            }
            rows.append(row)
            policy_outputs.append(token_ids)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        outputs[policy] = policy_outputs

    reference = next(iter(outputs), None)
    comparison = []
    if reference is not None:
        for policy, vals in outputs.items():
            comparable = [
                a is not None and b is not None
                for a, b in zip(outputs[reference], vals)
            ]
            exact = [
                bool(ok and a == b)
                for ok, a, b in zip(comparable, outputs[reference], vals)
            ]
            comparison.append(
                {
                    "reference_policy": reference,
                    "candidate_policy": policy,
                    "num_requests": len(vals),
                    "num_comparable": sum(comparable),
                    "num_exact": sum(exact),
                    "exact_agreement": (sum(exact) / sum(comparable)) if sum(comparable) else None,
                }
            )

    payload = {
        "contract": {
            "model": args.model,
            "model_revision": args.model_revision or "default",
            "dataset": args.dataset,
            "threshold": args.threshold,
            "rmax": args.max_steps,
            "fixed_decode_tokens": args.max_new_tokens,
            "batch_size": 1,
        },
        "rows": rows,
        "comparison": comparison,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
