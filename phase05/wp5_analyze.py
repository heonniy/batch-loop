"""Summarize WP5 under the preregistered rules (CPU + one GPU-free offline scoring pass)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase05.wp3_predictor import EVAL_B, lockstep_recovery

COLORS = {"history_last": "#eb6834", "predictor_gpu": "#e87ba4", "ranker_gpu": "#2a78d6", "oracle_b1": "#008300"}
MARKERS = {"history_last": "v", "predictor_gpu": "D", "ranker_gpu": "o", "oracle_b1": "*"}


def cells(lane_dir: Path) -> list[dict]:
    m = json.loads((lane_dir / "run_manifest.json").read_text())
    groups = defaultdict(list)
    for r in csv.DictReader((lane_dir / "runs.csv").open()):
        if r["kind"] == "scored":
            groups[(int(r["cohort"]), int(r["B"]), r["policy"])].append(r)
    rows = []
    for (c, b, p), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "OK"]
        fifo = [r for r in groups[(c, b, "fifo")] if r["status"] == "OK"]
        w, fw = [float(r["decode_wall_s"]) for r in ok], [float(r["decode_wall_s"]) for r in fifo]
        med = lambda xs, k: statistics.median(float(x[k]) for x in xs) if xs else None
        h = 1 - statistics.median(w) / statistics.median(fw) if w and fw else None
        ric = 1 - med(ok, "live_RIC") / med(fifo, "live_RIC") if ok and fifo else None
        rows.append({"model": m["model"], "dataset": m["dataset"], "cohort": c, "B": b, "policy": p,
                     "runs": len(rs), "ok": len(ok), "wall_median_s": med(ok, "decode_wall_s"),
                     "wall_min_s": min(w, default=None), "wall_max_s": max(w, default=None),
                     "headroom_vs_fifo": h, "ric_saving_vs_fifo": ric,
                     "wall_per_ric_translation": h / ric if h is not None and ric and ric >= 0.02 else None,
                     "all_faster_than_all_fifo": bool(w and fw and max(w) < min(fw)),
                     "wins_cell": bool(p == "ranker_gpu" and h is not None and h >= 0.05 and max(w) < min(fw)),
                     "live_work_vs_fifo_rel": med(ok, "live_work") / med(fifo, "live_work") - 1 if ok and fifo else None})
    get = {(r["cohort"], r["B"], r["policy"]): r for r in rows}
    for r in rows:
        o = get[(r["cohort"], r["B"], "oracle_b1")]
        r["recovery_of_oracle"] = r["headroom_vs_fifo"] / o["headroom_vs_fifo"] if o["headroom_vs_fifo"] else None
        if r["policy"] == "ranker_gpu":
            r["ranker_faster_than_predictor"] = r["wall_median_s"] < get[(r["cohort"], r["B"], "predictor_gpu")]["wall_median_s"]
    return rows


def offline_test(feature_root: Path, ranker_root: Path, wp3_selection_root: Path, wp3_feature_root: Path) -> dict:
    """Post-hoc frozen wp5_test recovery of R and of the WP3 predictor (same B1 labels)."""
    import torch
    from phase03_predictor.physical import Predictor
    from phase05.wp5_ranker import GpuRanker
    d = feature_root / "wp5_test"
    labels = np.load(d / "depth_labels.npy").astype(np.int64)
    tokens = np.load(d / "token_ids.npy").astype(np.int64)
    scal = np.load(d / "scalars.npy").astype(np.float32)
    ranker = GpuRanker(ranker_root / "ranker_checkpoint.pt")
    fulls = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ranker.full_names}
    r_scores = []
    with torch.inference_mode():
        for t in range(labels.shape[0]):
            cur = torch.as_tensor(tokens[t], device="cuda")
            prev = torch.as_tensor(tokens[t - 1], device="cuda") if t else torch.full_like(cur, ranker.vocab)
            r_scores.append(ranker.score(torch.as_tensor(scal[t], device="cuda"), cur, prev,
                                         {n: torch.as_tensor(np.asarray(f[t], dtype=np.float32), device="cuda")
                                          for n, f in fulls.items()}).cpu().numpy())
    sel = json.loads((wp3_selection_root / "selection.json").read_text())
    pred = Predictor(sel["winner"], wp3_selection_root, json.loads((wp3_selection_root / "feature_manifest.json").read_text()))
    emb, rep = np.load(d / "embedding.npy").astype(np.float32), np.load(d / "prelude.npy").astype(np.float32)
    p_scores = np.stack([pred.score(scal[t], emb[t], rep[t]) for t in range(labels.shape[0])])
    rec = lambda s: {b: v for b, v in lockstep_recovery(labels, np.asarray(s)).items()}
    rr, pr = rec(r_scores), rec(p_scores)
    return {"ranker": rr, "ranker_mean": float(np.nanmean(list(rr.values()))),
            "wp3_predictor": pr, "wp3_predictor_mean": float(np.nanmean(list(pr.values()))),
            "wp3_predictor_name": sel["winner"]}


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lanes = sorted({(r["model"], r["dataset"]) for r in rows})
    fig, axes = plt.subplots(1, len(lanes), figsize=(4.2 * len(lanes), 3.8), sharey=True)
    for ax, lane in zip(axes, lanes):
        ax.axhline(0, color="#8a8984", lw=1)
        ax.axhline(5, color="#8a8984", lw=1, ls=":")
        for p in ("history_last", "predictor_gpu", "ranker_gpu", "oracle_b1"):
            for c, style in ((0, "-"), (1, "--")):
                pts = sorted((r["B"], 100 * r["headroom_vs_fifo"]) for r in rows
                             if (r["model"], r["dataset"]) == lane and r["policy"] == p and r["cohort"] == c)
                if pts:
                    ax.plot(*zip(*pts), color=COLORS[p], marker=MARKERS[p], ms=6, lw=2, ls=style, label=f"{p} (c{c})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(EVAL_B, [str(b) for b in EVAL_B])
        ax.set_title(f"{lane[0]} · {lane[1]}", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("cohort size B")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("decode-wall saving vs FIFO (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("WP5 sealed wp5_test: ranker R vs WP3 predictor (both GPU-served); median of 3", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--wp3-root", required=True, type=Path)
    args = parser.parse_args()
    rows, offline = [], {}
    for lane_dir in sorted((args.result_dir / "physical").glob("lane_*")):
        rows += cells(lane_dir)
        offline[lane_dir.name] = offline_test(args.result_dir / "features" / lane_dir.name,
                                              args.result_dir / "ranker" / lane_dir.name,
                                              args.wp3_root / "selection" / lane_dir.name,
                                              args.wp3_root / "features" / lane_dir.name)
    out = args.result_dir / "analysis"
    out.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "cell_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    blocks = defaultdict(dict)
    for r in rows:
        if r["policy"] == "ranker_gpu":
            blocks[r["model"]].setdefault(f"{r['dataset']}/cohort{r['cohort']}", [])
            if r["wins_cell"]:
                blocks[r["model"]][f"{r['dataset']}/cohort{r['cohort']}"].append(r["B"])
    rk = [r for r in rows if r["policy"] == "ranker_gpu"]
    faster = sum(r["ranker_faster_than_predictor"] for r in rk) / len(rk)
    offline_better = all(v["ranker_mean"] > v["wp3_predictor_mean"] for v in offline.values())
    audits = {d.name: json.loads((d / "ranker_audit.json").read_text())["status"]
              for d in sorted((args.result_dir / "audit").glob("lane_*")) if (d / "ranker_audit.json").exists()}
    decision = {"rule": "phase05/WP5_PREREGISTRATION.md", "gate0": audits,
                "verdict": {m: {"practical_ranker": "GO" if len(b) == 4 and all(len(v) >= 2 for v in b.values()) else "NO-GO",
                                "winning_B_per_block": b} for m, b in blocks.items()},
                "stronger_ranker_hypothesis": {"ranker_faster_than_predictor_fraction": faster,
                                               "offline_test_better_in_all_lanes": offline_better,
                                               "status": "supported" if faster >= 0.75 and offline_better else "not supported"},
                "offline_test_recovery": offline}
    (out / "decision.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
    plot(rows, out / "headroom_vs_B.png")
    print(json.dumps({k: decision[k] for k in ("gate0", "verdict", "stronger_ranker_hypothesis")}, indent=2, default=float))


if __name__ == "__main__":
    main()
