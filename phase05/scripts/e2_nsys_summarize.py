"""Summarize Nsight Systems NVTX range kernel time vs range wall (GPU busy fraction, launches per Core)."""
import csv
import json
import sys
from pathlib import Path

rows = []
for d in sorted(Path(sys.argv[1]).glob("*_nvtx_kern_sum.csv")):
    cell = d.name.replace("_nvtx_kern_sum.csv", "")
    kern = [r for r in csv.DictReader(d.open()) if r.get("NVTX Range", "") == ":core"]
    proj = [r for r in csv.DictReader((d.parent / f"{cell}_nvtx_sum.csv").open()) if r.get("Range", "") == ":core"]
    if not kern or not proj:
        continue
    instances = int(proj[0]["Instances"])
    wall_ns = float(proj[0]["Total Time (ns)"])
    kern_ns = sum(float(r["Total Time (ns)"]) for r in kern)
    launches = sum(int(r["Kern Inst"]) for r in kern)
    rows.append({"cell": cell, "core_calls": instances, "range_wall_ms_per_call": wall_ns / instances / 1e6,
                 "kernel_ms_per_call": kern_ns / instances / 1e6, "gpu_busy_fraction": kern_ns / wall_ns,
                 "kernel_launches_per_call": launches / instances,
                 "mean_kernel_us": kern_ns / launches / 1e3})
with (Path(sys.argv[1]) / "e2_nsys_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps(rows, indent=1))
