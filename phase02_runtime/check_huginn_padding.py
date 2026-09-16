#!/usr/bin/env python3
"""GPU7-only padded-batch/B1 equivalence check for the repaired D=96 path."""

from __future__ import annotations

import json
import os
import subprocess

import torch

from phase02_runtime.huginn_adapter import HuginnAdapter


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7" or torch.cuda.device_count() != 1:
        raise RuntimeError("Huginn padding check requires only physical GPU7")
    gpu = subprocess.check_output(
        ["nvidia-smi", "-i", "7", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
    ).strip()
    prompts = [
        "Explain two plus two.",
        "Write one short sentence about the color of the sky at noon.",
        "List three steps to make tea, taking care to distinguish boiling water from steeping tea leaves.",
        "In a single paragraph, compare how a small seed grows into a tree and how a short idea becomes a detailed research plan.",
    ]
    batch = HuginnAdapter.load(4, state_init="zero")
    tokenizer = batch.tokenizer
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    torch.manual_seed(1)
    batch_tokens = batch.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
    assert batch.prefill_logits is not None
    checks = []
    for i, prompt in enumerate(prompts):
        single = HuginnAdapter(batch.model, tokenizer, 1, state_init="zero")
        one = tokenizer(prompt, return_tensors="pt")
        one_tokens = single.prefill(one["input_ids"].cuda(), one["attention_mask"].cuda().bool())
        assert single.prefill_logits is not None
        left = batch.prefill_logits[i].float()
        right = single.prefill_logits[0].float()
        normalized_rms = float((left - right).square().mean().sqrt() / right.square().mean().sqrt().clamp_min(1e-12))
        top1_equal = int(batch_tokens[i].item()) == int(one_tokens[0].item())
        checks.append({
            "row": i, "prompt_tokens": int(one["attention_mask"].sum()),
            "batch_token_id": int(batch_tokens[i].item()), "b1_token_id": int(one_tokens[0].item()),
            "top1_equal": top1_equal, "normalized_logit_rms": normalized_rms,
        })
        if not top1_equal or normalized_rms > 0.01:
            raise AssertionError(f"padded batch differs from B1 for row {i}: {checks[-1]}")
    print(json.dumps({
        "gpu": gpu, "model_revision": HuginnAdapter.model_revision,
        "attention_backend": "explicit-additive-mask-sdpa-math-D96",
        "state_init": "zero", "batch_size": 4,
        "batched_prompt_width": int(encoded["input_ids"].shape[1]), "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
