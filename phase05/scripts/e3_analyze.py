"""Summarize E3 physical forced-depth runs and the latency-model calibration (CPU only).

Calibration rule (predeclared in E3_LATENCY_AWARE_ORACLE.md): a LATENCY_MODEL prediction for a cell is
accepted only if the median measured FIFO Core time is within +-10% of the predicted FIFO Core time.
Physical numbers are PHYSICAL_FORCED_DEPTH (frozen B=1 trace depths), never live-exit performance.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_latency_oracle import CoreModel  # noqa: E402

TOL = 0.10


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("e3_root", type=Path)
    p.add_argument("--e2-root", required=True, type=Path)
    a = p.parse_args()
    rows, calls = [], []
    for lane in sorted((a.e3_root / "physical").glob("*")):
        man = json.loads((lane / "RUN_MANIFEST.json").read_text())
        model = CoreModel(a.e2_root / f"{man['model']}/e2_stage_latency_raw.csv")
        runs = [r for r in csv.DictReader((lane / "e3_physical_runs.csv").open()) if int(r["rep"]) >= 0]
        by = defaultdict(list)
        for r in runs:
            by[(r["cell"], r["policy"])].append(r)
        for (cell, policy), rs in sorted(by.items()):
            ok = [r for r in rs if r["status"] == "OK"]
            fifo = [r for r in by[(cell, "fifo")] if r["status"] == "OK"]
            med = lambda xs, k: statistics.median(float(x[k]) for x in xs) if xs else None
            core, wall = [float(r["core_ms"]) for r in ok], [float(r["decode_wall_s"]) for r in ok]
            plan = man["cells_detail"][cell]["plans"][policy]
            fplan = man["cells_detail"][cell]["plans"]["fifo"]
            row = {"trace": man["trace"], "model": man["model"], "cell": cell, "N": rs[0]["N"], "B": rs[0]["B"],
                   "T": rs[0]["T"], "W": man["cells_detail"][cell]["W"], "policy": policy, "runs": len(rs),
                   "ok": len(ok), "failures": ";".join(sorted({r["status"] for r in rs if r["status"] != "OK"})),
                   "RIC": plan["RIC"], "work": plan["work"], "slot_utilization": plan["slot_utilization"],
                   "RIC_reduction_vs_fifo": 1 - plan["RIC"] / fplan["RIC"],
                   "pred_core_ms": plan["pred_core_ms"],
                   "pred_reduction_vs_fifo": 1 - plan["pred_core_ms"] / fplan["pred_core_ms"],
                   "core_ms_median": med(ok, "core_ms"), "core_ms_q25": q(core, .25) if core else None,
                   "core_ms_q75": q(core, .75) if core else None,
                   "wall_s_median": med(ok, "decode_wall_s"), "wall_s_min": min(wall, default=None),
                   "wall_s_max": max(wall, default=None),
                   "prelude_ms_median": med(ok, "prelude_ms"), "coda_ms_median": med(ok, "coda_ms"),
                   "compaction_ms_median": med(ok, "compaction_ms"), "host_other_ms_median": med(ok, "host_other_ms"),
                   "core_ms_per_call": med(ok, "core_ms") / plan["RIC"] if ok else None,
                   "solver": rs[0]["solver"], "evidence_physical": "PHYSICAL_FORCED_DEPTH",
                   "evidence_pred": "LATENCY_MODEL"}
            if ok and fifo:
                row["core_reduction_vs_fifo"] = 1 - med(ok, "core_ms") / med(fifo, "core_ms")
                row["wall_reduction_vs_fifo"] = 1 - med(ok, "decode_wall_s") / med(fifo, "decode_wall_s")
                row["wall_speedup_vs_fifo"] = med(fifo, "decode_wall_s") / med(ok, "decode_wall_s")
                row["all_reps_faster_than_all_fifo"] = max(wall) < min(float(r["decode_wall_s"]) for r in fifo)
                row["core_reduction_over_RIC_reduction"] = (row["core_reduction_vs_fifo"] / row["RIC_reduction_vs_fifo"]
                                                           if row["RIC_reduction_vs_fifo"] > 0.01 else None)
                fp = fplan["pred_core_ms"]
                row["fifo_pred_over_measured"] = fp / med(fifo, "core_ms")
                row["calibration_accepted"] = abs(row["fifo_pred_over_measured"] - 1) <= TOL
            rows.append(row)
        W = {c: d["W"] for c, d in man["cells_detail"].items()}
        with gzip.open(lane / "e3_call_receipts.jsonl.gz", "rt") as f:
            for line in f:
                e = json.loads(line)
                cell = e["tag"].split(":")[0]
                L = W[cell] + e["t"] + 1
                try:
                    pred = model(e["w"], L)
                except ValueError:
                    pred = None
                calls.append({"model": man["model"], "trace": man["trace"], "cell": cell, "policy": e["tag"].split(":")[1],
                              "w": e["w"], "L": L, "measured_ms": e["core_ms"], "pred_ms": pred})
    get = {(r["trace"], r["cell"], r["policy"]): r for r in rows}
    for r in rows:
        if r["policy"] == "latency_oracle":
            d = get[(r["trace"], r["cell"], "depth_oracle")]
            r["latency_minus_depth_wall_reduction"] = (r.get("wall_reduction_vs_fifo") or 0) - (d.get("wall_reduction_vs_fifo") or 0)
            r["latency_minus_depth_core_reduction"] = (r.get("core_reduction_vs_fifo") or 0) - (d.get("core_reduction_vs_fifo") or 0)
    out = a.e3_root
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "e3_oracle_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    cal = defaultdict(list)
    for c in calls:
        if c["pred_ms"]:
            cal[(c["model"], c["w"] >= 16)].append(c["measured_ms"] / c["pred_ms"])
    call_summary = {f"{m}/{'w>=16' if big else 'w<16'}": {"calls": len(v), "median_measured_over_pred": statistics.median(v),
                                                         "p10": q(v, .1), "p90": q(v, .9)} for (m, big), v in cal.items()}
    (out / "e3_call_calibration.json").write_text(json.dumps(call_summary, indent=2) + "\n")
    plot(rows, calls, out)
    print(json.dumps(call_summary, indent=1))


def plot(rows, calls, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"depth_oracle": "#2a78d6", "latency_oracle": "#eb6834", "random": "#8a8984"}
    traces = sorted({r["trace"] for r in rows})
    fig, axes = plt.subplots(1, len(traces), figsize=(4.2 * len(traces), 3.8), sharey=True)
    for ax, tr in zip(axes, traces):
        cells = sorted({r["cell"] for r in rows if r["trace"] == tr}, key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_")[1][1:])))
        for i, pol in enumerate(("depth_oracle", "latency_oracle", "random")):
            ys = [100 * (next((r.get("wall_reduction_vs_fifo") for r in rows if r["trace"] == tr and r["cell"] == c and r["policy"] == pol), None) or 0) for c in cells]
            ax.bar([x + (i - 1) * 0.27 for x in range(len(cells))], ys, width=0.25, color=colors[pol], label=pol)
        ax.set_xticks(range(len(cells)), [c.replace("_T32", "") for c in cells], rotation=30, fontsize=7)
        ax.axhline(0, color="#8a8984", lw=1)
        ax.set_title(Path(tr).name.replace(".jsonl.gz", ""), fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("decode-wall reduction vs FIFO (%)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("E3 PHYSICAL_FORCED_DEPTH: median of 5 reps (frozen trace depths, not live exit)", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(out / "e3_physical_wall_reduction.png", dpi=150, facecolor="#fcfcfb")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axes, ("ouro14", "huginn")):
        xs = [c["pred_ms"] for c in calls if c["model"] == m and c["pred_ms"] and c["policy"] == "fifo"]
        ys = [c["measured_ms"] for c in calls if c["model"] == m and c["pred_ms"] and c["policy"] == "fifo"]
        ax.scatter(xs, ys, s=4, alpha=0.3, color="#2a78d6")
        if xs:
            lim = [min(xs + ys), max(xs + ys)]
            ax.plot(lim, lim, color="#8a8984", lw=1)
        ax.set_title(f"{m}: per-call Core, FIFO", fontsize=10)
        ax.set_xlabel("E2-predicted ms")
        ax.set_ylabel("E3 measured ms")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "e3_call_calibration.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
