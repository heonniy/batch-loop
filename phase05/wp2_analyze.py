"""Summarize WP2 lanes under the preregistered decision rules (CPU only)."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from phase05.wp2_headroom import BATCHES, POLICIES, policy_order

GO_B = (8, 16, 32, 64)
COLORS = dict(zip(POLICIES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
MARKERS = dict(zip(POLICIES, ("o", "s", "^", "v", "D", "*")))


def _lane(path: Path) -> dict:
    manifest = json.loads((path / "run_manifest.json").read_text())
    runs = list(csv.DictReader((path / "runs.csv").open()))
    b1 = defaultdict(dict)
    with gzip.open(path / "b1_reference.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            b1[int(row["token_idx"])][int(row["request_id"])] = int(row["b1_depth"])
    n, t = manifest["N"], manifest["T"]
    depth = [[b1[i][r] for r in range(n)] for i in range(t)]
    status = [json.loads(line) for line in (path / "status.jsonl").open()]
    b1_wall = next(s["decode_wall_s"] for s in status if s["event"] == "b1_reference_done")
    return {"path": path, "manifest": manifest, "runs": runs, "b1_depth": depth, "b1_wall": b1_wall}


def frozen_lockstep_ric(depth: list[list[int]], b: int, policy: str) -> int:
    """Structural: every policy grouped and costed on frozen B1 depths (history = B1 history)."""
    n = len(depth[0])
    total = 0
    for t in range(len(depth)):
        order = policy_order(policy, t, n, depth[:t], depth, 0)
        total += sum(max(depth[t][r] for r in order[i:i + b]) for i in range(0, n, b))
    return total


def summarize(lane: dict) -> list[dict]:
    m = lane["manifest"]
    cells = defaultdict(list)
    for run in lane["runs"]:
        if run["kind"] in ("scored", "control_no_composition"):
            cells[(int(run["B"]), run["policy"])].append(run)
    rows = []
    for (b, policy), runs in sorted(cells.items()):
        ok = [r for r in runs if r["status"] == "OK"]
        fifo_ok = [r for r in cells.get((b, "fifo"), []) if r["status"] == "OK"]
        walls = [float(r["decode_wall_s"]) for r in ok]
        fifo_walls = [float(r["decode_wall_s"]) for r in fifo_ok]
        med = lambda rs, k: statistics.median(float(r[k]) for r in rs) if rs else None
        row = {"model": m["model"], "dataset": ("alpaca" if "alpaca" in m["dataset"] else "sharegpt"), "lane": m["gpu"]["lane"],
               "B": b, "policy": policy, "runs": len(runs), "ok": len(ok),
               "failures": ";".join(sorted({r["status"] for r in runs if r["status"] != "OK"})),
               "wall_median_s": med(ok, "decode_wall_s"),
               "wall_min_s": min(walls) if walls else None, "wall_max_s": max(walls) if walls else None,
               "headroom_vs_fifo": (1 - statistics.median(walls) / statistics.median(fifo_walls))
               if walls and fifo_walls else None,
               "all_faster_than_all_fifo": bool(walls and fifo_walls and max(walls) < min(fifo_walls)),
               "live_RIC_median": med(ok, "live_RIC"), "live_work_median": med(ok, "live_work"),
               "live_work_vs_fifo": (med(ok, "live_work") - med(fifo_ok, "live_work")) if ok and fifo_ok else None,
               "core_row_slots_median": med(ok, "core_row_slots"),
               "depth_flips_median": med(ok, "depth_flips"), "next_token_flips_median": med(ok, "next_token_flips"),
               "scheduler_cpu_s_median": med(ok, "scheduler_cpu_s"),
               "tokens_per_s_median": med(ok, "tokens_per_s"), "peak_gib_max": max(
                   (float(r["peak_allocated_gib"]) for r in ok), default=None),
               "frozen_b1_lockstep_RIC": frozen_lockstep_ric(lane["b1_depth"], b, policy)
               if b < m["N"] else None}
        rows.append(row)
    return rows


def decide(rows: list[dict]) -> dict:
    by_model = defaultdict(dict)
    detail = []
    for lane_key in sorted({(r["model"], r["dataset"]) for r in rows}):
        lane_rows = [r for r in rows if (r["model"], r["dataset"]) == lane_key]
        get = {(r["B"], r["policy"]): r for r in lane_rows}
        present = []
        causal = [p for p in POLICIES if p not in ("fifo", "random", "oracle_b1")]
        best = max(causal, key=lambda p: statistics.mean(
            get[(b, p)]["headroom_vs_fifo"] or 0 for b in BATCHES if (b, p) in get))
        for b in BATCHES:
            o = get.get((b, "oracle_b1"))
            if not o or o["headroom_vs_fifo"] is None:
                continue
            ok = o["headroom_vs_fifo"] >= 0.05 and o["all_faster_than_all_fifo"]
            if ok and b in GO_B:
                present.append(b)
            h = get[(b, best)]["headroom_vs_fifo"]
            detail.append({"model": lane_key[0], "dataset": lane_key[1], "B": b,
                           "oracle_headroom": o["headroom_vs_fifo"], "oracle_present": ok,
                           "best_causal_policy": best, "best_causal_headroom": h,
                           "history_recovery": (h / o["headroom_vs_fifo"]) if o["headroom_vs_fifo"] > 0 else None})
        by_model[lane_key[0]][lane_key[1]] = {"oracle_present_B": present, "best_causal_policy": best}
    verdict = {}
    for model, datasets in by_model.items():
        go = len(datasets) == 2 and all(len([b for b in d["oracle_present_B"] if b in GO_B]) >= 2
                                        for d in datasets.values())
        verdict[model] = {"WP3": "GO" if go else "NO-GO", "per_dataset": datasets}
    return {"rule": "phase05/WP2_PREREGISTRATION.md", "verdict": verdict, "cells": detail}


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lanes = sorted({(r["model"], r["dataset"]) for r in rows})
    fig, axes = plt.subplots(1, len(lanes), figsize=(4.2 * len(lanes), 3.8), sharey=True)
    for ax, lane in zip(axes, lanes):
        ax.axhline(0, color="#8a8984", lw=1)
        ax.axhline(5, color="#8a8984", lw=1, ls=":")
        for policy in POLICIES[1:]:
            pts = sorted((r["B"], 100 * r["headroom_vs_fifo"]) for r in rows
                         if (r["model"], r["dataset"]) == lane and r["policy"] == policy
                         and r["B"] in BATCHES and r["headroom_vs_fifo"] is not None)
            if pts:
                ax.plot(*zip(*pts), color=COLORS[policy], marker=MARKERS[policy], ms=6, lw=2, label=policy)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
        ax.set_title(f"{lane[0]} · {lane[1]}", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("cohort size B")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("decode-wall saving vs FIFO (%)")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("WP2 live no-refill regrouping: median of 3 repeats; dotted line = 5% GO threshold",
                 fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    lanes = [_lane(p) for p in sorted(args.result_dir.glob("lane_*")) if (p / "runs.csv").exists()]
    rows = [row for lane in lanes for row in summarize(lane)]
    out = args.result_dir / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "cell_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    decision = decide(rows)
    decision["b1_control_decode_wall_s"] = {f'{l["manifest"]["model"]}/{("alpaca" if "alpaca" in l["manifest"]["dataset"] else "sharegpt")}':
                                            l["b1_wall"] for l in lanes}
    decision["input_sha256"] = {str(p.relative_to(args.result_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for l in lanes for p in (l["path"] / "runs.csv", l["path"] / "b1_reference.csv.gz")}
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    plot(rows, out / "headroom_vs_B.png")
    print(json.dumps(decision["verdict"], indent=2))


if __name__ == "__main__":
    main()
