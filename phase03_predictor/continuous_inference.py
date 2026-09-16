"""Run repeated live-exit inference with one loaded model on its assigned GPU."""

from __future__ import annotations

import argparse
import json
import time

import torch

from phase02_runtime.benchmark import _gpu_identity, _prompts, _run_once
from phase02_runtime.huginn_adapter import HuginnAdapter
from phase02_runtime.ouro_adapter import OuroAdapter


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    identity = _gpu_identity(args.model)
    adapter_type = HuginnAdapter if args.model == "huginn" else OuroAdapter
    initial = (adapter_type.load(args.num_rows, state_init="native-seeded")
               if args.model == "huginn" else adapter_type.load(args.num_rows))
    model, tokenizer = initial.model, initial.tokenizer
    prompts = _prompts(args.num_rows, heterogeneous=True)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to("cuda")
    prompt_valid = encoded["attention_mask"].to("cuda").bool()
    # The FIFO ordering is fixed; run_interval obtains exit depths from the live model.
    unused_trace = [[1] * args.num_tokens for _ in range(args.num_rows)]
    print(json.dumps({"event": "model_ready", "model": args.model, "gpu": identity,
                      "num_rows": args.num_rows, "batch_size": args.batch_size,
                      "num_tokens": args.num_tokens, "threshold": args.threshold,
                      "duration_sec": args.duration_sec}), flush=True)
    started = time.monotonic()
    cycle = 0
    while args.duration_sec == 0 or time.monotonic() - started < args.duration_sec:
        summary, events = _run_once(args, model, tokenizer, input_ids, prompt_valid,
                                    unused_trace, "fifo", cycle, "continuous_inference")
        del events
        print(json.dumps({"event": "cycle_complete", "model": args.model,
                          "gpu": identity["physical_gpu"], "cycle": cycle,
                          "elapsed_sec": round(time.monotonic() - started, 1),
                          "generated_tokens": summary["generated_tokens"],
                          "decode_wall_sec": round(summary["total_decode_wall_ms"] / 1000, 3),
                          "recurrent_invocations": summary["recurrent_invocation_count"],
                          "peak_reserved_gib": round(summary["peak_reserved_gib"], 2)}), flush=True)
        cycle += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ouro", "huginn"), required=True)
    parser.add_argument("--num-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--duration-sec", type=float, default=86400,
                        help="maximum wall duration; 0 means until stopped")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.num_rows < 1 or args.batch_size < 1 or args.num_tokens < 1 or args.duration_sec < 0:
        parser.error("rows, batch size, token count must be positive; duration must be nonnegative")
    if args.num_rows % args.batch_size:
        parser.error("num-rows must be divisible by batch-size")
    if args.threshold is None:
        args.threshold = 0.26 if args.model == "huginn" else 0.4
    args.state_init = "native-seeded" if args.model == "huginn" else "zero"
    args.exit_source = "live"
    args.debug_ids = False
    run(args)


if __name__ == "__main__":
    main()
