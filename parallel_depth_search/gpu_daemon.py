"""Persistent single-GPU worker over an ordered, append-only queue of run_jobs plans.

usage: python -m parallel_depth_search.gpu_daemon --gpu G --queue RUN/queue.txt --run-dir RUN \
          --allow-gpu --authorized
queue.txt lists plan paths (one per line, appended as the adaptive campaign proceeds); each plan
<p>.json is claimed through <p>.claims (same flock ledger as run_jobs --claim-file), so daemons on
different GPUs and plain run_jobs executors never run a job twice. Before every job the daemon
fails closed: it waits while ANY process carrying CUDA_VISIBLE_DEVICES=G exists (e.g. a job of an
older executor still loading its model) or the device shows foreign memory/compute use; it never
kills or pre-empts anything. Exits when RUN/STOP exists. Execution reports match run_jobs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from .run_jobs import active_compute_uuids, claim_next, gpu_inventory


def ours_on_gpu(g: int) -> bool:
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            env = (d / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if f"CUDA_VISIBLE_DEVICES={g}".encode() in env:
            return True
    return False


def device_idle(g: int) -> bool:
    inv = {r["index"]: r for r in gpu_inventory()}
    d = inv[g]
    return d["uuid"] not in active_compute_uuids() and d["memory_used_mb"] <= 1024 and not ours_on_gpu(g)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    p.add_argument("--authorized", action="store_true")
    a = p.parse_args()
    if not (a.allow_gpu and a.authorized):
        raise SystemExit("explicit --allow-gpu --authorized required")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("unset CUDA_VISIBLE_DEVICES for the daemon itself")
    run = Path(a.run_dir).resolve()
    (run / "job_logs").mkdir(parents=True, exist_ok=True)
    name = next(r["name"] for r in gpu_inventory() if r["index"] == a.gpu)
    uuid = next(r["uuid"] for r in gpu_inventory() if r["index"] == a.gpu)
    if "H200" not in name:
        raise SystemExit(f"GPU {a.gpu} is not an H200: {name}")
    while not (run / "STOP").exists():
        if not device_idle(a.gpu):
            time.sleep(20)
            continue
        job = None
        for line in Path(a.queue).read_text().splitlines():
            plan_path = line.strip()
            if not plan_path:
                continue
            plan = json.loads(Path(plan_path).read_text())
            if Path(plan["run_dir"]).resolve() != run:
                continue
            job = claim_next(plan["jobs"], plan_path[:-5] + ".claims" if plan_path.endswith(".json") else plan_path + ".claims")
            if job:
                break
        if job is None:
            time.sleep(60)
            continue
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONHASHSEED="0")
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            env.setdefault(k, "1")
        jid = job["job_id"]
        log = run / "job_logs" / (jid + ".log")
        begin, started, issue, code = time.monotonic(), dt.datetime.now(dt.timezone.utc).isoformat(), None, None
        try:
            with open(log, "w", encoding="utf-8") as fp:
                code = subprocess.run(job["argv"], stdout=fp, stderr=subprocess.STDOUT, env=env,
                                      timeout=float(job.get("timeout_s", 3600)), check=False).returncode
        except subprocess.TimeoutExpired:
            issue = "timeout"
        except Exception as exc:                     # noqa: BLE001
            issue = "launch_failed:" + repr(exc)
        rec = {"job_id": jid, "node_id": job["node_id"], "gpu_index": a.gpu, "gpu_name": name, "gpu_uuid": uuid,
               "started": started, "elapsed_seconds": time.monotonic() - begin, "exit_code": code, "error": issue,
               "argv": job["argv"], "log": str(log), "executor": "gpu_daemon",
               "research_status": "EXECUTION_ONLY_NOT_SCIENTIFICALLY_VALIDATED", "ingested_into_research_ledger": False}
        tmp = run / "job_logs" / (jid + ".execution.tmp")
        tmp.write_text(json.dumps(rec, indent=2) + "\n")
        os.replace(tmp, run / "job_logs" / (jid + ".execution.json"))


if __name__ == "__main__":
    main()
