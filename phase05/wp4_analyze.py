"""Summarize WP4 under the preregistered rules (CPU only)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from phase05.wp3_predictor import EVAL_B
from phase05.wp4_gpu_serving import POLICIES

COLORS = {"history_last": "#eb6834", "predictor_host": "#e87ba4", "predictor_gpu": "#2a78d6", "oracle_b1": "#008300"}
MARKERS = {"history_last": "v", "predictor_host": "D", "predictor_gpu": "o", "oracle_b1": "*"}


def cells(lane_dir: Path) -> list[dict]:
    m = json.loads((lane_dir / "run_manifest.json").read_text())
    groups = defaultdict(list)
    for r in csv.DictReader((lane_dir / "runs.csv").open()):
        if r["kind"] == "scored":
            groups[(int(r["cohort"]), int(r["B"]), r["policy"])].append(r)
    rows = []
    for (cohort, b, policy), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "OK"]
        fifo = [r for r in groups[(cohort, b, "fifo")] if r["status"] == "OK"]
        w = [float(r["decode_wall_s"]) for r in ok]
        fw = [float(r["decode_wall_s"]) for r in fifo]
        med = lambda xs, k: statistics.median(float(x[k]) for x in xs) if xs else None
        h = 1 - statistics.median(w) / statistics.median(fw) if w and fw else None
        ric = 1 - med(ok, "live_RIC") / med(fifo, "live_RIC") if ok and fifo else None
        rows.append({"model": m["model"], "dataset": m["dataset"], "predictor": m["predictor"], "cohort": cohort,
                     "B": b, "policy": policy, "runs": len(rs), "ok": len(ok),
                     "wall_median_s": med(ok, "decode_wall_s"), "wall_min_s": min(w, default=None),
                     "wall_max_s": max(w, default=None), "headroom_vs_fifo": h, "ric_saving_vs_fifo": ric,
                     "wall_per_ric_translation": h / ric if h is not None and ric and ric >= 0.02 else None,
                     "all_faster_than_all_fifo": bool(w and fw and max(w) < min(fw)),
                     "wins_cell": bool(policy == "predictor_gpu" and h is not None and h >= 0.05 and max(w) < min(fw)),
                     "live_work_vs_fifo_rel": med(ok, "live_work") / med(fifo, "live_work") - 1 if ok and fifo else None,
                     "peak_gib_max": max((float(r["peak_allocated_gib"]) for r in ok), default=None)})
    get = {(r["cohort"], r["B"], r["policy"]): r for r in rows}
    for r in rows:
        o, hl = get[(r["cohort"], r["B"], "oracle_b1")], get[(r["cohort"], r["B"], "history_last")]
        r["recovery_of_oracle"] = r["headroom_vs_fifo"] / o["headroom_vs_fifo"] if o["headroom_vs_fifo"] else None
        r["gain_vs_history_last"] = 1 - r["wall_median_s"] / hl["wall_median_s"]
        if r["policy"] == "predictor_gpu":
            r["gpu_faster_than_host"] = r["wall_median_s"] < get[(r["cohort"], r["B"], "predictor_host")]["wall_median_s"]
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lanes = sorted({(r["model"], r["dataset"]) for r in rows})
    fig, axes = plt.subplots(1, len(lanes), figsize=(4.2 * len(lanes), 3.8), sharey=True)
    for ax, lane in zip(axes, lanes):
        ax.axhline(0, color="#8a8984", lw=1)
        ax.axhline(5, color="#8a8984", lw=1, ls=":")
        for policy in ("history_last", "predictor_host", "predictor_gpu", "oracle_b1"):
            for cohort, style in ((0, "-"), (1, "--")):
                pts = sorted((r["B"], 100 * r["headroom_vs_fifo"]) for r in rows if (r["model"], r["dataset"]) == lane
                             and r["policy"] == policy and r["cohort"] == cohort)
                if pts:
                    ax.plot(*zip(*pts), color=COLORS[policy], marker=MARKERS[policy], ms=6, lw=2, ls=style,
                            label=f"{policy} (c{cohort})")
        pred = next(r["predictor"] for r in rows if (r["model"], r["dataset"]) == lane)
        ax.set_xscale("log", base=2)
        ax.set_xticks(EVAL_B, [str(b) for b in EVAL_B])
        ax.set_title(f"{lane[0]} · {lane[1]} · {pred}", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("cohort size B")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("decode-wall saving vs FIFO (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("WP4 fresh cohorts: GPU (sync-free) vs host feature serving of the same frozen predictors; median of 3",
                 fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = [r for d in sorted((args.result_dir / "physical").glob("lane_*")) for r in cells(d)]
    out = args.result_dir / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "cell_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
        writer.writeheader()
        writer.writerows(rows)
    blocks = defaultdict(dict)
    for r in rows:
        if r["policy"] == "predictor_gpu":
            blocks[r["model"]].setdefault(f"{r['dataset']}/cohort{r['cohort']}", [])
            if r["wins_cell"]:
                blocks[r["model"]][f"{r['dataset']}/cohort{r['cohort']}"].append(r["B"])
    gpu = [r for r in rows if r["policy"] == "predictor_gpu"]
    trans = lambda p: [r["wall_per_ric_translation"] for r in rows if r["policy"] == p and r["wall_per_ric_translation"] is not None]
    faster = sum(r["gpu_faster_than_host"] for r in gpu) / len(gpu)
    hypothesis = {"gpu_faster_than_host_fraction": faster,
                  "mean_translation_host": statistics.mean(trans("predictor_host")) if trans("predictor_host") else None,
                  "mean_translation_gpu": statistics.mean(trans("predictor_gpu")) if trans("predictor_gpu") else None}
    hypothesis["status"] = ("supported" if faster >= 0.75 and hypothesis["mean_translation_gpu"] is not None
                            and hypothesis["mean_translation_host"] is not None
                            and hypothesis["mean_translation_gpu"] > hypothesis["mean_translation_host"] else "not supported")
    verdict = {m: {"practical_predictor": "GO" if len(b) == 4 and all(len(v) >= 2 for v in b.values()) else "NO-GO",
                   "winning_B_per_block": b} for m, b in blocks.items()}
    audits = {d.name: json.loads((d / "equivalence_audit.json").read_text())["status"]
              for d in sorted((args.result_dir / "audit").glob("lane_*")) if (d / "equivalence_audit.json").exists()}
    decision = {"rule": "phase05/WP4_PREREGISTRATION.md", "gate0_equivalence": audits, "verdict": verdict,
                "serving_cost_hypothesis": hypothesis}
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    plot(rows, out / "headroom_vs_B.png")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
