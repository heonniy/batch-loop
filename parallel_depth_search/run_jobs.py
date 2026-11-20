"""Opt-in, non-preemptive per-GPU local job executor.

Runs only on the institute/server host with assigned devices. Never kills,
pauses, resumes, restores or steals other GPU processes. DOES NOT perform
automatic scientific promotion: raw receipts require agent review/curation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def gpu_inventory():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError("nvidia-smi unavailable: cannot certify H200 allocation") from e
    GPUs = []
    for line in p.stdout.splitlines():
        cols = [t.strip() for t in line.split(",")]
        if len(cols) != 4:
            raise RuntimeError("unrecognized nvidia-smi GPU inventory")
        GPUs.append({"index": int(cols[0]), "uuid": cols[1], "name": cols[2],
                     "memory_used_mb": int(cols[3])})
    return GPUs


def active_compute_uuids():
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError("cannot inspect existing compute processes; fail closed") from e
    result = set()
    for line in p.stdout.splitlines():
        cols = [t.strip() for t in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            result.add(cols[1])
    return result


def check_devices(indices, require_h200=True):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset inherited CUDA_VISIBLE_DEVICES to avoid index/remapping errors")
    inv = {r["index"]: r for r in gpu_inventory()}
    used = active_compute_uuids()
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("GPU ids must be distinct and nonempty")
    for i in indices:
        if i not in inv:
            raise RuntimeError(f"GPU {i} not visible")
        d = inv[i]
        if require_h200 and "H200" not in d["name"]:
            raise RuntimeError(f"GPU {i} is not an H200: {d['name']}")
        if d["uuid"] in used or d["memory_used_mb"] > 1024:
            raise RuntimeError(f"GPU {i} has an existing process or high memory use; do not preempt")
    return [inv[i] for i in indices]


def run_manifest(plan_path, run_dir, gpu_ids, wall_limit_hours, *, allowed):
    if not allowed:
        raise RuntimeError("GPU host execution explicitly gated; pass --allow-gpu --authorized")
    if wall_limit_hours <= 0 or wall_limit_hours > 48:
        raise ValueError("wall-clock budget must be in (0,48]")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("status") != "PLANNED_ONLY_NO_GPU_JOB_LAUNCHED":
        raise ValueError("not a preregistered diagnostic job manifest")
    devices = check_devices(gpu_ids)
    # Do not silently run a generated manifest for a different directory.
    run = Path(run_dir).resolve()
    if Path(plan["run_dir"]).resolve() != run:
        raise ValueError("manifest run_dir mismatch")
    jobs = plan["jobs"]
    if not jobs:
        return {"status": "NOT_RUN", "reason": "no eligible implemented GPU job"}
    for job in jobs:
        if not isinstance(job.get("argv"), list) or len(job["argv"]) < 3:
            raise ValueError("malformed argv")
        if job["argv"][1:3] != ["-m", "parallel_depth_search.gpu_diagnostic"] and not (
           job["argv"][1] == "-m" and job["argv"][2] in {
            "parallel_depth_poc.e0_native_parity",
            "parallel_depth_poc.e1_oracle_width",
            "parallel_depth_poc.depth_diag"}):
            raise ValueError("unreviewed job module; refuse to run")
        if not all(isinstance(s, str) for s in job["argv"]):
            raise ValueError("argv must be literal string list")
    run.mkdir(parents=True, exist_ok=True)
    (run / "job_logs").mkdir(exist_ok=True)
    deadline = time.monotonic() + 3600 * wall_limit_hours
    lock = threading.Lock()
    pending = iter(jobs)
    results = []

    def worker(device):
        while True:
            with lock:
                if time.monotonic() > deadline:
                    break
                job = next(pending, None)
            if job is None:
                break
            jid = job["job_id"]
            report = run / "job_logs" / (jid + ".execution.json")
            if report.exists():
                existing = json.loads(report.read_text(encoding="utf-8"))
                if existing.get("exit_code") == 0:
                    with lock:
                        results.append({"job_id": jid, "skipped": "previous_success"})
                    continue
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(device["index"])
            env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
            env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
            env["HF_DATASETS_OFFLINE"] = env.get("HF_DATASETS_OFFLINE", "1")
            env["PYTHONHASHSEED"] = "0"
            argv = list(job["argv"])
            log = run / "job_logs" / (jid + ".log")
            begin = time.monotonic()
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            exit_code = None
            issue = None
            try:
                # This executor never calls shell, process scanning during a
                # run, burner scripts, or remote commands.
                with open(log, "w", encoding="utf-8") as fp:
                    remaining = max(1, min(float(job.get("timeout_s", 3600)),
                                           deadline - time.monotonic()))
                    proc = subprocess.run(argv, stdout=fp, stderr=subprocess.STDOUT,
                                          env=env, timeout=remaining, check=False)
                    exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                issue = "timeout"
            except Exception as exc:
                issue = "launch_failed:" + repr(exc)
            rec = {
                "job_id": jid, "node_id": job["node_id"],
                "gpu_index": device["index"], "gpu_name": device["name"],
                "gpu_uuid": device["uuid"], "started": started,
                "elapsed_seconds": time.monotonic() - begin,
                "exit_code": exit_code, "error": issue,
                "argv": argv, "log": str(log),
                "research_status": "EXECUTION_ONLY_NOT_SCIENTIFICALLY_VALIDATED",
                "ingested_into_research_ledger": False
            }
            temp = report.with_suffix(".tmp")
            temp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, report)
            with lock:
                results.append(rec)
            if issue == "timeout":
                # Cease this worker, but never touch somebody else's process.
                break

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device) for device in devices]
        for f in futures:
            f.result()
    return {"status": "LOCAL_GPU_JOBS_COMPLETE_OR_STOPPED", "jobs_requested": len(jobs),
            "jobs_attempted": len(results),
            "success": sum(r.get("exit_code") == 0 for r in results),
            "failures": [r for r in results if r.get("exit_code") not in (0, None)],
            "receipts": str(run / "job_logs"), "warning": "No science result automatically accepted"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, help="from make_jobs.py")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    p.add_argument("--wall-hours", type=float, default=12.0)
    p.add_argument("--allow-gpu", action="store_true")
    p.add_argument("--authorized", action="store_true",
                   help="confirm these idle GPUs are assigned to this experiment")
    a = p.parse_args()
    ids = [int(t.strip()) for t in a.gpu_ids.split(",") if t.strip()]
    result = run_manifest(a.plan, a.run_dir, ids, a.wall_hours,
                          allowed=a.allow_gpu and a.authorized)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
