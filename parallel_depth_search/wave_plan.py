"""Build a run_jobs plan for an explicit list of nodes (CPU; launches nothing).

Unlike make_jobs.generate (eligibility = all prerequisites already ingested), a wave may contain
nodes whose prerequisites are executing in the SAME wave, to keep 8 GPUs busy. Such nodes are
recorded in `concurrent_with_prerequisites`; their results are ingested and interpreted only after
the prerequisite receipts (fail-closed: a failed prerequisite invalidates them). Extra jobs (e.g.
e2e_decoder cells chosen from evidence) are appended from a JSON list of {node_id, argv, fields}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .control import init_run, program, propose
from .make_jobs import jobs_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--nodes", default="")
    p.add_argument("--extra", default="", help="JSON file with extra jobs")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    init_run(a.run_dir)
    run = Path(a.run_dir).resolve()
    prog = {n["id"]: n for n in program()["nodes"]}
    state = propose(a.run_dir)
    eligible = {x["id"] for x in state["eligible_next"]}
    jobs, concurrent = [], []
    nodes = [x for x in a.nodes.split(",") if x]
    extra = json.loads(Path(a.extra).read_text()) if a.extra else []
    for nid in nodes + sorted({e["node_id"] for e in extra} - set(nodes)):
        if nid not in eligible:
            concurrent.append(nid)
    dropped = []
    for nid in nodes:
        found = jobs_for(prog[nid])
        # gpu_diagnostic records metrics after EVERY round, so a config that differs only in
        # --rounds is fully contained in the max-rounds job of the same config: keep that one.
        best = {}
        for j in found:
            f = dict(j.get("fields", {}))
            if "rounds" not in f:
                best[j["job_id"]] = j
                continue
            key = json.dumps({k: v for k, v in f.items() if k != "rounds"}, sort_keys=True)
            if key not in best or f["rounds"] > best[key]["fields"]["rounds"]:
                if key in best:
                    dropped.append(best[key]["job_id"])
                best[key] = j
            else:
                dropped.append(j["job_id"])
        jobs.extend(best.values())
    for e in extra:
        argv = [sys.executable] + e["argv"] if e["argv"][0] == "-m" else e["argv"]
        jid = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
        jobs.append({"job_id": jid, "node_id": e["node_id"], "argv": argv,
                     "timeout_s": e.get("timeout_s", 14400), "metrics_status": "NOT_RUN",
                     "reference_mode": e.get("reference_mode", "e2e_free_running"), "fields": e.get("fields", {})})
    for j in jobs:
        outdir = run / "results" / j["node_id"]
        if "__OUT__" in " ".join(j["argv"]):
            j["argv"] = [x.replace("__OUT__", str(outdir)) for x in j["argv"]]
        elif "--out" not in j["argv"]:
            j["argv"] += ["--out", str(outdir / j["job_id"])]
        j["papers_to_check"] = prog[j["node_id"]]["refs"]
    # longest first (LPT) so the wave finishes evenly
    jobs.sort(key=lambda j: -j.get("timeout_s", 0))
    plan = {"status": "PLANNED_ONLY_NO_GPU_JOB_LAUNCHED", "run_dir": str(run), "jobs": jobs,
            "nodes": nodes + [e["node_id"] for e in extra], "concurrent_with_prerequisites": concurrent,
            "dropped_as_round_prefix_of_kept_job": dropped}
    Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"jobs": len(jobs), "concurrent_with_prerequisites": concurrent}))


if __name__ == "__main__":
    main()
