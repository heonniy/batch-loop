"""Summarize WP3 sealed-test physical runs under the preregistered rules (CPU only).

Also computes, only now (after the physical run), the frozen test-label lockstep RIC
recovery of every candidate checkpoint, as an offline reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase05.wp3_predictor import COHORT, EVAL_B, PHYSICAL_POLICIES, lockstep_recovery

COLORS = {"history_last": "#eb6834", "predictor": "#2a78d6", "oracle_b1": "#008300"}
MARKERS = {"history_last": "v", "predictor": "o", "oracle_b1": "*"}


def lane_cells(lane_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((lane_dir / "run_manifest.json").read_text())
    runs = [r for r in csv.DictReader((lane_dir / "runs.csv").open()) if r["kind"] == "scored"]
    groups = defaultdict(list)
    for r in runs:
        groups[(int(r["cohort"]), int(r["B"]), r["policy"])].append(r)
    rows = []
    for (cohort, b, policy), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "OK"]
        fifo = [r for r in groups[(cohort, b, "fifo")] if r["status"] == "OK"]
        walls = [float(r["decode_wall_s"]) for r in ok]
        fifo_walls = [float(r["decode_wall_s"]) for r in fifo]
        med = lambda xs, k: statistics.median(float(x[k]) for x in xs) if xs else None
        headroom = 1 - statistics.median(walls) / statistics.median(fifo_walls) if walls and fifo_walls else None
        rows.append({
            "model": manifest["model"], "dataset": manifest["dataset"], "predictor": manifest["predictor"],
            "cohort": cohort, "B": b, "policy": policy, "runs": len(rs), "ok": len(ok),
            "failures": ";".join(sorted({r["status"] for r in rs if r["status"] != "OK"})),
            "wall_median_s": med(ok, "decode_wall_s"), "wall_min_s": min(walls, default=None),
            "wall_max_s": max(walls, default=None), "headroom_vs_fifo": headroom,
            "all_faster_than_all_fifo": bool(walls and fifo_walls and max(walls) < min(fifo_walls)),
            "wins_cell": bool(policy == "predictor" and headroom is not None and headroom >= 0.05
                              and walls and fifo_walls and max(walls) < min(fifo_walls)),
            "live_RIC_median": med(ok, "live_RIC"), "live_work_median": med(ok, "live_work"),
            "live_work_vs_fifo_rel": (med(ok, "live_work") / med(fifo, "live_work") - 1) if ok and fifo else None,
            "feature_and_predict_s_median": med(ok, "feature_and_predict_s"), "sort_s_median": med(ok, "sort_s"),
            "overhead_share": (med(ok, "feature_and_predict_s") + med(ok, "sort_s")) / med(ok, "decode_wall_s") if ok else None,
            "depth_flips_median": med(ok, "depth_flips"), "next_token_flips_median": med(ok, "next_token_flips"),
            "peak_gib_max": max((float(r["peak_allocated_gib"]) for r in ok), default=None)})
    get = {(r["cohort"], r["B"], r["policy"]): r for r in rows}
    for r in rows:
        o = get.get((r["cohort"], r["B"], "oracle_b1"))
        h = get.get((r["cohort"], r["B"], "history_last"))
        r["recovery_of_oracle"] = (r["headroom_vs_fifo"] / o["headroom_vs_fifo"]
                                   if o and o["headroom_vs_fifo"] and r["headroom_vs_fifo"] is not None else None)
        r["gain_vs_history_last"] = (1 - r["wall_median_s"] / h["wall_median_s"]) if h and r["wall_median_s"] else None
    return manifest, rows


def offline_test(feature_root: Path, selection_root: Path) -> list[dict]:
    """Frozen test-label recovery for every committed candidate (opened only post hoc)."""
    import torch
    from phase03_predictor.physical import Predictor
    from phase03_predictor.tournament import assemble_candidate_features
    manifest = json.loads((selection_root / "feature_manifest.json").read_text())
    d = feature_root / "wp3_test"
    scal = np.load(d / "scalars.npy").astype(np.float32)
    emb = np.load(d / "embedding.npy").astype(np.float32)
    rep = np.load(d / "prelude.npy").astype(np.float32)
    labels = np.load(d / "depth_labels.npy").astype(np.int64)
    out = []
    names = ["C0"] + sorted(p.name.split("_")[0] for p in selection_root.glob("C*_checkpoint.pt"))
    for c in names:
        pred = Predictor(c, selection_root, manifest)
        scores = np.stack([pred.score(scal[t], emb[t], rep[t]) for t in range(scal.shape[0])])
        rec = lockstep_recovery(labels, scores)
        out.append({"candidate": c, **{f"test_recovery_B{b}": v for b, v in rec.items()},
                    "test_mean_recovery": float(np.nanmean(list(rec.values())))})
    del torch
    return out


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lanes = sorted({(r["model"], r["dataset"]) for r in rows})
    fig, axes = plt.subplots(1, len(lanes), figsize=(4.2 * len(lanes), 3.8), sharey=True)
    for ax, lane in zip(axes, lanes):
        ax.axhline(0, color="#8a8984", lw=1)
        ax.axhline(5, color="#8a8984", lw=1, ls=":")
        for policy in ("history_last", "predictor", "oracle_b1"):
            for cohort, style in ((0, "-"), (1, "--")):
                pts = sorted((r["B"], 100 * r["headroom_vs_fifo"]) for r in rows
                             if (r["model"], r["dataset"]) == lane and r["policy"] == policy
                             and r["cohort"] == cohort and r["headroom_vs_fifo"] is not None)
                if pts:
                    ax.plot(*zip(*pts), color=COLORS[policy], marker=MARKERS[policy], ms=6, lw=2, ls=style,
                            label=f"{policy} (cohort {cohort})")
        pred = next(r["predictor"] for r in rows if (r["model"], r["dataset"]) == lane)
        ax.set_xscale("log", base=2)
        ax.set_xticks(EVAL_B, [str(b) for b in EVAL_B])
        ax.set_title(f"{lane[0]} · {lane[1]} · predictor {pred}", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("cohort size B")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("decode-wall saving vs FIFO (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("WP3 sealed test, live no-refill: median of 3; solid/dashed = test cohort 0/1; dotted = 5% rule",
                 fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows, offline, verdict_blocks = [], [], defaultdict(dict)
    for lane_dir in sorted((args.result_dir / "physical").glob("lane_*")):
        manifest, lane_rows = lane_cells(lane_dir)
        rows += lane_rows
        for cohort in (0, 1):
            wins = [r["B"] for r in lane_rows if r["policy"] == "predictor" and r["cohort"] == cohort and r["wins_cell"]]
            verdict_blocks[manifest["model"]][f"{manifest['dataset']}/cohort{cohort}"] = wins
        for row in offline_test(args.result_dir / "features" / lane_dir.name, args.result_dir / "selection" / lane_dir.name):
            offline.append({"model": manifest["model"], "dataset": manifest["dataset"],
                            "selected": row["candidate"] == manifest["predictor"], **row})
    out = args.result_dir / "analysis"
    out.mkdir(exist_ok=True)
    for name, data in (("cell_summary.csv", rows), ("offline_test_recovery.csv", offline)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    verdict = {model: {"practical_predictor": "GO" if len(blocks) == 4 and all(len(w) >= 2 for w in blocks.values())
                       else "NO-GO", "winning_B_per_block": blocks}
               for model, blocks in verdict_blocks.items()}
    (out / "decision.json").write_text(json.dumps({"rule": "phase05/WP3_PREREGISTRATION.md", "verdict": verdict},
                                                  indent=2) + "\n")
    plot(rows, out / "headroom_vs_B.png")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
