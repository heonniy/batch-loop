#!/usr/bin/env python3
"""Only the TWO prioritized 10/29 KV research questions.

Track ouro   = Do all historical tokens need their own first-loop KV?
Track huginn = Find useful Full>Last accuracy gaps across depth/CoT/context.

This is a focused, fail-closed interface to the validated generic H200
job/worker machinery. No GPU starts without explicit --execute.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = {"ouro": "ouro_token", "huginn": "huginn_hard"}


def command(args, python: str = sys.executable) -> list[str]:
    phase = PHASE[args.track]
    if args.track == "ouro" and args.prompt_style != "default":
        raise ValueError("Only Huginn supports nondefault prompt styles")
    cmd = [python, str(HERE / "kv_knob_sweep.py"),
           "--phase", phase,
           "--task", args.task,
           "--sample", args.sample,
           "--prompt-style", args.prompt_style,
           "--run-dir", args.run_dir,
           "--datasets", args.datasets,
           "--batch-size-gsm8k", str(args.batch_size_gsm8k),
           "--batch-size-math500", str(args.batch_size_math500),
           "--gpu-ids", args.gpu_ids]
    if args.models:
        cmd.extend(["--models", args.models])
    if args.policies:
        cmd.extend(["--policies", args.policies])
    if args.execute:
        cmd.append("--execute")
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=sorted(PHASE))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task", choices=("nll", "gen", "both"), default="nll")
    ap.add_argument("--sample", choices=("pilot", "main"), default="pilot")
    ap.add_argument("--models", default="")
    ap.add_argument("--policies", default="")
    ap.add_argument("--datasets", default="gsm8k,math500")
    ap.add_argument("--prompt-style", choices=("default", "reason", "verify", "distract512"),
                    default="default")
    ap.add_argument("--batch-size-gsm8k", type=int, default=8)
    ap.add_argument("--batch-size-math500", type=int, default=4)
    ap.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--execute", action="store_true", help="OPT-IN remote H200 work")
    args = ap.parse_args()
    try:
        cmd = command(args)
    except ValueError as e:
        ap.error(str(e))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
