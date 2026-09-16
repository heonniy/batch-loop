#!/usr/bin/env python3
"""Interruptible sustained GPU inference, not an experiment measurement.

Start one process per physical GPU with CUDA_VISIBLE_DEVICES set externally.
Create ``--stop-file`` or send SIGTERM to stop after the current iteration.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path


_STOP = False


def _request_stop(_signal: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B")
    parser.add_argument("--revision", default="37bc7e8ff85a90d813dc096383405f521b6a67b0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.batch_size < 1 or args.prompt_tokens < 1 or args.decode_tokens < 1:
        parser.error("batch-size, prompt-tokens and decode-tokens must be positive")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import Cache

    # The pinned Ouro remote module assigns these containers. Transformers
    # 4.54's read-only base properties otherwise prevent cache construction.
    for attribute in ("key_cache", "value_cache"):
        if isinstance(getattr(Cache, attribute, None), property):
            delattr(Cache, attribute)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU per burn process")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    prompt = "Warm-up inference for a GPU burn-in job. " * 100
    encoded = tokenizer(
        [prompt] * args.batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.prompt_tokens,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    print(json.dumps({
        "event": "loaded",
        "model": args.model,
        "revision": args.revision,
        "gpu_name": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "prompt_width": int(encoded["input_ids"].shape[1]),
        "decode_tokens": args.decode_tokens,
        "stop_file": str(args.stop_file),
    }), flush=True)

    iteration = 0
    total_seconds = 0.0
    with torch.inference_mode():
        while not _STOP and not args.stop_file.exists():
            start = time.perf_counter()
            output = model.generate(
                **encoded,
                max_new_tokens=args.decode_tokens,
                min_new_tokens=args.decode_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                exit_threshold=0.5,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            iteration += 1
            total_seconds += elapsed
            if iteration == 1 or iteration % args.log_every == 0:
                print(json.dumps({
                    "event": "progress",
                    "iteration": iteration,
                    "last_seconds": round(elapsed, 3),
                    "total_seconds": round(total_seconds, 3),
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "output_width": int(output.shape[1]),
                }), flush=True)
            del output

    print(json.dumps({"event": "stopped", "iterations": iteration}), flush=True)


if __name__ == "__main__":
    main()
