"""Create a Stage-2 run directory: frozen prompts/targets/batches, subset manifests and job list."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from kvp.jobs import BATCH_SIZE, QUALITY_POLICIES, batches, prepare_items, sha  # noqa: E402
from kvp.specs import MODELS  # noqa: E402
from kvp.manifest import environment  # noqa: E402

DATASETS = ("gsm8k", "math500")
TASKS = ("gen", "nll")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--policies", default=",".join(f"{p}-{m}" for p, m in QUALITY_POLICIES),
                    help="comma-separated policy-mode pairs, e.g. full-decode,last-convert")
    args = ap.parse_args()
    run = Path(args.run_dir)
    policies = [tuple(x.split("-")) for x in args.policies.split(",")]
    (run / "prompts").mkdir(parents=True, exist_ok=True)
    jobs, subsets = [], {}
    for mk in args.models.split(","):
        spec = MODELS[mk]
        tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, trust_remote_code=True)
        for ds in DATASETS:
            items, meta = prepare_items(mk, ds, tok)
            subsets[ds] = meta["dataset_meta"]
            with open(run / "prompts" / f"{mk}__{ds}.jsonl", "w") as f:
                f.write(json.dumps({"protocol": meta["protocol"], "batch_size": BATCH_SIZE[ds]}) + "\n")
                for it in items:
                    f.write(json.dumps({k: it[k] for k in ("uid", "batch", "batch_pos", "prompt", "target",
                                                           "target_tokenization")} |
                                       {"prompt_sha256": sha(it["prompt"]), "prompt_tokens": len(it["prompt_ids"]),
                                        "prompt_ids": it["prompt_ids"], "target_ids": it["target_ids"]}) + "\n")
            nb = len(batches(items))
            longest = max(len(it["prompt_ids"]) for it in items)
            for task in TASKS:
                for pol, mode in policies:
                    for b in range(nb):
                        cost = (longest if mode == "strict" else longest / 20) + (
                            1024 if ds == "math500" else 256)
                        cost *= {"huginn": 1.6, "ouro": 1.0}[spec.family] * (2 if mk == "ouro-2.6b" else 1)
                        jobs.append({"model": mk, "dataset": ds, "task": task, "policy": pol, "mode": mode,
                                     "batch": b, "est_cost": round(cost, 1)})
    rng = random.Random(20261022)
    rng.shuffle(jobs)  # randomise GPU<->(model, policy) pairing; workers then claim dynamically
    jobs.sort(key=lambda j: -j["est_cost"])  # stable: long jobs first, shuffled within ties
    (run / "jobs.json").write_text(json.dumps(jobs, indent=0))
    (run / "subsets.json").write_text(json.dumps(subsets, indent=1))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    (run / "run_manifest.json").write_text(json.dumps(
        {"source_sha": git, "command": " ".join(sys.argv), "n_jobs": len(jobs),
         "batch_size": BATCH_SIZE, "policies": policies,
         "models": {k: MODELS[k].__dict__ for k in args.models.split(",")}, "environment": environment()},
        indent=1))
    print(f"{len(jobs)} jobs -> {run}")


if __name__ == "__main__":
    main()
