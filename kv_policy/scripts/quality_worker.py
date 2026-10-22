"""Single-GPU Stage-2 worker: claims jobs from <run>/jobs.json until none remain.

Claims are lock files (O_CREAT|O_EXCL), so any number of workers can share one run directory and a
crashed run can be resumed (finished outputs are skipped; stale claims of failed jobs are reported
in <run>/errors and must be cleared explicitly before a retry).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kvp.jobs import ModelContext, job_name, run_job  # noqa: E402


def claim(run: Path, name: str) -> bool:
    (run / "claims").mkdir(exist_ok=True)
    try:
        fd = os.open(run / "claims" / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.environ.get('CUDA_VISIBLE_DEVICES')} {os.getpid()} {time.time()}\n".encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker", type=int, required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "raw"
    jobs = json.loads((run / "jobs.json").read_text())
    ctx = None
    log = open(run / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    while True:
        jobs = json.loads((run / "jobs.json").read_text())  # re-read: control jobs may be appended
        pending = [j for j in jobs if not (out / f"{job_name(j)}.jsonl").exists()
                   and not (run / "claims" / f"{job_name(j)}.lock").exists()]
        if not pending:
            break
        if ctx is not None:  # prefer the resident model, else the longest remaining job
            same = [j for j in pending if j["model"] == ctx.key]
            pending = same or pending
        job = pending[0]
        name = job_name(job)
        if not claim(run, name):
            continue
        try:
            if ctx is None or ctx.key != job["model"]:
                ctx = None
                torch.cuda.empty_cache()
                t = time.time()
                ctx = ModelContext(job["model"])
                log.write(f"loaded {job['model']} in {time.time() - t:.1f}s\n")
            meta = run_job(ctx, job, out)
            log.write(f"done {name} {meta['wall_s']:.1f}s peak={meta['engine_stats']['peak_mem_bytes']/2**30:.1f}GiB\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (run / "errors").mkdir(exist_ok=True)
            (run / "errors" / f"{name}.json").write_text(json.dumps(
                {"job": job, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "traceback": traceback.format_exc()}))
            log.write(f"FAILED {name}\n{traceback.format_exc()}\n")
            torch.cuda.empty_cache()
    log.write("no pending jobs; exit\n")


if __name__ == "__main__":
    main()
