"""Deterministic evidence-gated job list generator. Does NOT launch GPUs.

Only methods with executable adapters are emitted. Nodes requiring new code
must be implemented by the coding agent, with a paper review, before running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .control import init_run, program, propose


def job_spec(node, model, depth, B, L, *, cohort="gsm8k3", offset=0, steps=32,
             warmup=1, predictor="constant", corrector="jacobi", rounds=4,
             allocation="full", width=0, scale=1.0, seed=20261121):
    argv = [sys.executable, "-m", "parallel_depth_search.gpu_diagnostic",
            "--model", model, "--depth", str(depth), "--batch", str(B),
            "--prompt-len", str(L), "--cohort", cohort, "--offset", str(offset),
            "--decode-steps", str(steps), "--warmup", str(warmup),
            "--predictor", predictor, "--corrector", corrector,
            "--rounds", str(rounds), "--allocation", allocation,
            "--width", str(width), "--scale", str(scale),
            "--seed", str(seed), "--allow-gpu"]
    fields = dict(node=node, model=model, depth=depth, batch=B, length=L,
                  offset=offset, steps=steps, warmup=warmup, predictor=predictor,
                  corrector=corrector, rounds=rounds, allocation=allocation,
                  width=width, scale=scale, seed=seed)
    id_ = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]
    return {"job_id": id_, "node_id": node, "argv": argv,
            "timeout_s": 10800 if depth >= 16 else 5400,
            "reference_mode": "exact_AR_teacher_forced_diagnostic",
            "metrics_status": "NOT_RUN",
            "fields": fields}


def jobs_for(n, preset="pilot"):
    id_ = n["id"]
    jobs = []
    # Target two distinct architectures; small B=16 diagnostic before adding
    # H200-offline B64/128 cells. Large batch/length may OOM; record exceptions.
    machines = [("huginn-3.5b", 32), ("ouro-1.4b", 4), ("ouro-2.6b", 4)]
    def append(model, D, **opts):
        b = opts.pop("B", 16); L = opts.pop("L", 256)
        if D < opts.get("warmup", 1):
            return
        jobs.append(job_spec(id_, model, D, b, L, **opts))
    if id_ in ("G0_NATIVE", "G1_WIDTH"):
        for model, D in machines:
            if id_ == "G0_NATIVE":
                depths = "8,16,32" if model == "huginn-3.5b" else "2,3,4"
                argv = [sys.executable, "-m", "parallel_depth_poc.e0_native_parity",
                        "--model", model, "--depths", depths,
                        "--out", f"__OUT__/e0_{model}.json", "--allow-gpu"]
            else:
                argv = [sys.executable, "-m", "parallel_depth_poc.e1_oracle_width",
                        "--model", model, "--depth", str(D),
                        "--prompt-lens", "256", "--out", f"__OUT__/e1_{model}.json",
                        "--allow-gpu"]
            jid = hashlib.sha256((" ".join(argv)).encode()).hexdigest()[:16]
            jobs.append({"job_id": jid, "node_id": id_, "argv": argv,
                         "timeout_s": 10800, "metrics_status": "NOT_RUN",
                         "reference_mode": "native_parity_or_oracle_hardware",
                         "fields": {"model": model, "depth": D, "baseline": id_}})
        return jobs
    if id_ in ("G0_KV", "G1_DYNAMICS"):
        for model, D in machines:
            if id_ == "G1_DYNAMICS":
                argv = [sys.executable, "-m", "parallel_depth_poc.depth_diag",
                        "--model", model, "--depth", str(D), "--batch", "16",
                        "--prompt-len", "256", "--decode-steps", "64",
                        "--out", f"__OUT__/dynamics_{model}", "--allow-gpu"]
                jid = hashlib.sha256((" ".join(argv)).encode()).hexdigest()[:16]
                jobs.append({"job_id": jid, "node_id": id_, "argv": argv, "timeout_s": 10800,
                             "reference_mode": "oracle_teacher_forced",
                             "metrics_status": "NOT_RUN", "fields": {"model": model, "depth": D}})
            else:
                append(model, D, predictor="constant", corrector="jacobi", rounds=2)
        return jobs
    if id_ == "G0_DATA":
        return []
    if id_ in ("P1_BASELINE", "P2_WARMUP", "P3_GEOMETRIC", "P4_HISTORY"):
        for model, D in machines:
            if id_ == "P1_BASELINE":
                configs = [(1, x, "jacobi", 2) for x in ("constant","linear","prev_shift")]
            elif id_ == "P2_WARMUP":
                configs = [(m, "linear", "jacobi", 2) for m in (1,2,3,4,8) if m <= D]
            elif id_ == "P3_GEOMETRIC":
                configs = [(m, x, "jacobi", 2) for m in (2,3) if m <= D for x in ("geometric","secant")]
            else:
                configs = [(1, "prev_shift", c, 4) for c in ("jacobi","anderson")]
            for m,pred,cor,k in configs:
                append(model, D, warmup=m, predictor=pred, corrector=cor, rounds=k)
        return jobs
    if id_ in ("C1_JACOBI","C2_ANDERSON","C3_NEWTON"):
        for model,D in machines:
            if id_ == "C1_JACOBI":
                configs = [("jacobi",x,1.0) for x in (1,2,4,8) if x < D] + [("damped",4,1.0)]
            elif id_ == "C2_ANDERSON":
                configs = [("anderson",k,1.0) for k in (2,4,8) if k < D]
            else:
                configs = [("newton_identity",k,1.0) for k in (2,4)] + [
                    ("newton_scaled",4,a) for a in (0.0,0.5,0.75,1.25)]
            for cor,k,a in configs:
                append(model,D,corrector=cor,rounds=k,scale=a,predictor="constant")
        return jobs
    if id_ in ("A1_FULL_VS_SPARSE","A2_INFLUENCE"):
        for model,D in machines:
            if id_ == "A1_FULL_VS_SPARSE":
                configs = [("full",0),("prefix",min(4,D-1)),("suffix",min(4,D-1)),("alternating",min(8,D-1))]
            else:
                configs = [("residual_topk",min(w,D-1)) for w in (4,8,16)]
            for pol,w in configs:
                append(model,D,corrector="jacobi",rounds=4,allocation=pol,width=w)
        return jobs
    # Correct numerical implementations must be added before emitting jobs for
    # C4_SCAN_LOW_RANK, C5_PARAREAL variants, A3_BLOCKS, V1, V2, J1, F1, F2.
    return []


def generate(run_dir, *, phase=None, node=None, max_jobs=256):
    init_run(run_dir)
    p = program()
    state = propose(run_dir)
    eligible_ids = {x["id"] for x in state["eligible_next"]}
    ready = [n for n in p["nodes"] if n["id"] in eligible_ids
             and (phase is None or n["phase"] == phase)
             and (node is None or n["id"] == node)]
    items, manual = [], []
    for n in ready:
        found = jobs_for(n)
        if not found:
            manual.append({"node_id": n["id"], "reason": "Needs paper-grounded agent implementation or split audit before execution."})
        items.extend(found)
    items = items[:max_jobs]
    for j in items:
        # Out stem is local to this run, not a hard-coded institute path.
        stem = str(Path(run_dir).resolve() / "results" / j["node_id"] / j["job_id"])
        if "__OUT__" in " ".join(j["argv"]):
            j["argv"] = [x.replace("__OUT__", str(Path(run_dir).resolve() / "results" / j["node_id"])) for x in j["argv"]]
        else:
            j["argv"] += ["--out", stem]
        j["papers_to_check"] = next(n["refs"] for n in ready if n["id"] == j["node_id"])
    return {"status": "PLANNED_ONLY_NO_GPU_JOB_LAUNCHED", "run_dir": str(Path(run_dir).resolve()),
            "jobs": items, "manual_nodes": manual, "excluded_until_evidence": state["blocked"],
            "eligible_nodes": [n["id"] for n in ready]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True, help="JSON plan file")
    p.add_argument("--phase")
    p.add_argument("--node")
    p.add_argument("--max-jobs", type=int, default=256)
    a = p.parse_args()
    plan = generate(a.run_dir, phase=a.phase, node=a.node, max_jobs=a.max_jobs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "job_count": len(plan["jobs"]),
                      "manual_nodes": plan["manual_nodes"], "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
