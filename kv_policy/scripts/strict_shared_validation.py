#!/usr/bin/env python3
"""Prepare or execute paired FULL and strict-terminal KV validation on H200 workers.

Reuses the vetted HF reference engine.  Does not import Parcae inference code:
the BeNice! `compress-s1` policy is a semantic comparison, not a drop-in
backend for Huginn/Ouro checkpoints.

Usage:
  python scripts/strict_shared_validation.py --run-dir results/strict_pilot --batching pilot
  python scripts/strict_shared_validation.py --run-dir results/strict_pilot --execute --gpu-ids 0,1,2,3,4,5,6,7

For full samples pass --batching main with a DIFFERENT run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
MODELS = "huginn-0125,ouro-1.4b,ouro-2.6b"
POLICIES = "full-decode,full-strict,last-strict"


def validate(run: Path) -> dict:
    jobs = json.loads((run / "jobs.json").read_text())
    expected = {("full", "decode"), ("full", "strict"), ("last", "strict")}
    got = {(j["policy"], j["mode"]) for j in jobs}
    if got != expected:
        raise RuntimeError(f"Unexpected policies: {got}")
    if {j["model"] for j in jobs} != set(MODELS.split(",")):
        raise RuntimeError("Expected exactly the three pinned models")
    names = [f"{j['model']}__{j['dataset']}__{j['task']}__{j['policy']}-{j['mode']}"
             + (f"-{j['batching']}" if j.get("batching", "main") != "main" else "")
             + f"__b{j['batch']:02d}.jsonl" for j in jobs]
    finished = [x for x in names if (run / "raw" / x).exists()]
    errors = sorted(str(x) for x in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    return {"jobs": len(jobs), "finished": len(finished), "missing": [x for x in names if x not in finished],
            "errors": errors, "policies": sorted(map(list, expected))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batching", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    if not (run / "jobs.json").exists():
        cmd = [args.python, str(HERE / "make_quality_jobs.py"), "--run-dir", str(run),
               "--models", MODELS, "--batching", args.batching, "--policies", POLICIES]
        subprocess.run(cmd, cwd=PARENT, check=True)
    else:
        manifest = json.loads((run / "run_manifest.json").read_text())
        if manifest.get("policies") != [list(x) for x in [("full", "decode"), ("full", "strict"), ("last", "strict")]]:
            raise SystemExit("Existing run has a different policy manifest: refusing to reuse.")
    if args.execute:
        processes = []
        for rank, gpu in enumerate(args.gpu_ids.split(",")):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            log = (run / "logs" / f"worker{rank}_stdout.log").open("a")
            cmd = [args.python, str(HERE / "quality_worker.py"), "--run-dir", str(run), "--worker", str(rank)]
            processes.append((subprocess.Popen(cmd, cwd=PARENT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT), log))
        statuses = [p.wait() for p, _ in processes]
        for _, log in processes:
            log.close()
        if any(statuses):
            raise SystemExit(f"Workers exited abnormally: {statuses}")
    result = validate(run)
    (run / "strict_validation_completeness.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}, indent=2))
    if args.execute:
        if result["missing"] or result["errors"]:
            raise SystemExit("Incomplete run or errors: fix/retry before analysis")
        subprocess.run([args.python, str(HERE / "analyze_quality.py"), "--run-dir", str(run)],
                       cwd=PARENT, check=True)
        print("Quality tables saved under", run / "analysis")


if __name__ == "__main__":
    main()
