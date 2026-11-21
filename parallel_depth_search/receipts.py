"""Build an auditable ledger receipt from REAL executed job reports + artifacts (CPU).

usage: python -m parallel_depth_search.receipts --run-dir RUN --node NODE --metrics metrics.json
          [--state executed] [--split hardware|pilot|dev|test] [--method TEXT] [--reason TEXT]
          [--jobs job_id,...]  (default: every job_logs/*.execution.json of that node)
GPU-hours = sum of executed job wall-clock seconds / 3600 (one GPU per job).
Artifacts = sha256 of every file under results/<NODE>/ (and listed extra files).
Literature = parallel_depth_search/literature_checks.json entries for the node's refs
(arXiv abstract + research-wiki card, read 2026-11-21; verification depth recorded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .control import program

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(run_dir, node_id, metrics, *, state="executed", split="pilot", method="", reason="",
          job_ids=None, extra_files=()):
    run = Path(run_dir).resolve()
    node = next(n for n in program()["nodes"] if n["id"] == node_id)
    lit = json.loads((ROOT / "literature_checks.json").read_text())
    checks = [dict(paper_id=p, url=lit[p]["url"], version=lit[p]["version"], claim=lit[p]["claim"],
                   transfer_caveat=lit[p]["transfer_caveat"], verification_depth=lit[p]["verification_depth"],
                   wiki_card=lit[p].get("wiki_card_path")) for p in node["refs"]]
    reps = []
    for f in sorted((run / "job_logs").glob("*.execution.json")):
        r = json.loads(f.read_text())
        if r["node_id"] == node_id and (job_ids is None or r["job_id"] in job_ids):
            reps.append(r)
    ok = [r for r in reps if r.get("exit_code") == 0]
    gpu_hours = round(sum(r["elapsed_seconds"] for r in ok) / 3600, 4)
    files = sorted(p for p in (run / "results" / node_id).rglob("*") if p.is_file())
    files += [Path(x) for x in extra_files]
    art = {str(p.relative_to(run)) if str(p).startswith(str(run)) else str(p): sha(p) for p in files}
    art_digest = hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest() if art else ""
    git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    models = sorted({str(r["argv"][r["argv"].index("--model") + 1]) for r in ok if "--model" in r["argv"]})
    gpus = sorted({r["gpu_name"] for r in ok})
    trial = f"{node_id}-{hashlib.sha256((art_digest + json.dumps(sorted(r['job_id'] for r in ok))).encode()).hexdigest()[:12]}"
    rec = dict(node_id=node_id, trial_id=trial, state=state, source_git_sha=git,
               command=[" ".join(r["argv"]) for r in ok] or ["(cpu) see metrics.provenance"],
               data_split=split, model=",".join(models) or "n/a", seed=20261121, method=method or node["method"],
               literature_checks=checks, metrics=metrics,
               hardware=dict(gpu_name=gpus[0] if gpus else ("CPU" if state == "executed" else ""),
                             gpu_names=gpus, gpu_indices=sorted({r["gpu_index"] for r in ok}),
                             jobs_ok=len(ok), jobs_failed=len(reps) - len(ok)),
               artifacts=dict(sha256=art_digest, files=art,
                              failed_jobs=[dict(job_id=r["job_id"], exit_code=r.get("exit_code"), error=r.get("error"))
                                           for r in reps if r.get("exit_code") != 0]),
               gpu_hours=gpu_hours if state == "executed" else 0)
    if reason:
        rec["reason"] = reason
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--state", default="executed")
    p.add_argument("--split", default="pilot")
    p.add_argument("--method", default="")
    p.add_argument("--reason", default="")
    p.add_argument("--jobs", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = json.loads(Path(a.metrics).read_text())
    rec = build(a.run_dir, a.node, m, state=a.state, split=a.split, method=a.method, reason=a.reason,
                job_ids=set(a.jobs.split(",")) if a.jobs else None)
    if a.state == "executed" and rec["gpu_hours"] <= 0:
        rec["gpu_hours"] = 1e-4        # CPU-only node: nominal, flagged
        rec["metrics"]["cpu_only_nominal_gpu_hours"] = True
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("trial_id", "state", "gpu_hours", "data_split")}))


if __name__ == "__main__":
    main()
