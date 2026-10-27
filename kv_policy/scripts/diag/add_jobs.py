"""Append quality jobs (scripts/make_quality_jobs.py semantics, pilot batching) to <run>/jobs.json.

Generates into a scratch directory, then merges jobs (skipping duplicates), prompts and subset
manifests, so several (models, policies) groups can share one run directory and running workers
pick the new jobs up (they re-read jobs.json)."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--batching", default="pilot")
    args = ap.parse_args()
    run = Path(args.run_dir)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "scripts/make_quality_jobs.py", "--models", args.models, "--policies", args.policies,
                        "--batching", args.batching, "--run-dir", tmp], check=True)
        new = json.loads((Path(tmp) / "jobs.json").read_text())
        (run / "prompts").mkdir(parents=True, exist_ok=True)
        for f in (Path(tmp) / "prompts").iterdir():
            shutil.copy(f, run / "prompts" / f.name)
        for name in ("subsets.json", "run_manifest.json"):
            shutil.copy(Path(tmp) / name, run / f"{name[:-5]}__{args.models.replace(',', '+')}__{len(new)}.json")
    old = json.loads((run / "jobs.json").read_text()) if (run / "jobs.json").exists() else []
    key = lambda j: (j["model"], j["dataset"], j["task"], j["policy"], j["mode"], j["batch"], j.get("batching", "main"))  # noqa: E731
    have = {key(j) for j in old}
    merged = old + [j for j in new if key(j) not in have]
    merged.sort(key=lambda j: -j["est_cost"])
    tmpf = run / ".jobs.json.tmp"
    tmpf.write_text(json.dumps(merged, indent=0))
    tmpf.replace(run / "jobs.json")
    print(f"+{len(merged) - len(old)} jobs -> {run / 'jobs.json'} ({len(merged)} total)")


if __name__ == "__main__":
    main()
