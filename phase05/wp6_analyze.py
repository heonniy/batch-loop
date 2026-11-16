"""Summarize WP6 studies under the preregistered rules (CPU only)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def lane_rows(lane_dir: Path) -> list[dict]:
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
        rows.append({"study": m["study"], "model": m["model"], "dataset": m["dataset"], "cohort": c, "B": b,
                     "policy": p, "runs": len(rs), "ok": len(ok), "wall_median_s": med(ok, "decode_wall_s"),
                     "wall_min_s": min(w, default=None), "wall_max_s": max(w, default=None),
                     "headroom_vs_fifo": h, "ric_saving_vs_fifo": ric,
                     "all_faster_than_all_fifo": bool(w and fw and max(w) < min(fw)),
                     "wins_cell": bool(p.startswith("ranker_") and h is not None and h >= 0.05 and max(w) < min(fw)),
                     "live_work_vs_fifo_rel": med(ok, "live_work") / med(fifo, "live_work") - 1 if ok and fifo else None,
                     "active_tokens": med(ok, "active_tokens")})
    get = {(r["cohort"], r["B"], r["policy"]): r for r in rows}
    for r in rows:
        o = get[(r["cohort"], r["B"], "oracle_b1")]
        r["recovery_of_oracle"] = r["headroom_vs_fifo"] / o["headroom_vs_fifo"] if o["headroom_vs_fifo"] else None
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#4a3aa7"]
    panels = sorted({(r["study"], r["model"], r["dataset"]) for r in rows})
    cols = 4
    nrows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 3.6 * nrows), sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, panel in zip(axes.flat, panels):
        pol = [p for p in dict.fromkeys(r["policy"] for r in rows if (r["study"], r["model"], r["dataset"]) == panel)
               if p != "fifo"]
        colors = {p: ("#008300" if p == "oracle_b1" else "#8a8984" if p == "history_last" else None) for p in pol}
        rk = [p for p in pol if colors[p] is None]
        colors.update({p: palette[i] for i, p in enumerate(rk)})
        ax.axhline(0, color="#8a8984", lw=1)
        ax.axhline(5, color="#8a8984", lw=1, ls=":")
        for p in pol:
            for c, style in ((0, "-"), (1, "--")):
                pts = sorted((r["B"], 100 * r["headroom_vs_fifo"]) for r in rows
                             if (r["study"], r["model"], r["dataset"]) == panel and r["policy"] == p and r["cohort"] == c)
                if pts:
                    ax.plot(*zip(*pts), color=colors[p], marker="o" if p.startswith("ranker") else "*" if p == "oracle_b1" else "v",
                            ms=5, lw=2, ls=style, label=p if c == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.set_title(f"{panel[0]} · {panel[1]} · {panel[2]}", fontsize=9, color="#0b0b0b")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=6, loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel("decode-wall saving vs FIFO (%)")
    fig.suptitle("WP6: median of 3; solid/dashed = cohort 0/1; dotted = 5% rule", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = [r for d in sorted((args.result_dir / "physical").glob("*/lane_*")) for r in lane_rows(d)]
    out = args.result_dir / "analysis"
    out.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "cell_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    verdicts = {}
    for (study, model, policy) in sorted({(r["study"], r["model"], r["policy"]) for r in rows if r["policy"].startswith("ranker_")}):
        blocks = defaultdict(list)
        for r in rows:
            if (r["study"], r["model"], r["policy"]) == (study, model, policy):
                blocks[f"{r['dataset']}/cohort{r['cohort']}"]
                if r["wins_cell"]:
                    blocks[f"{r['dataset']}/cohort{r['cohort']}"].append(r["B"])
        verdicts[f"{study}/{model}/{policy}"] = {
            "GO" if len(blocks) == 4 and all(len(v) >= 2 for v in blocks.values()) else "NO-GO": dict(blocks)}
    get = {(r["study"], r["model"], r["dataset"], r["cohort"], r["B"], r["policy"]): r for r in rows}
    extra = {}
    cross = [(r["headroom_vs_fifo"], get[(r["study"], r["model"], r["dataset"], r["cohort"], r["B"], "ranker_same")]["headroom_vs_fifo"])
             for r in rows if r["policy"] == "ranker_cross"]
    for model in ("ouro", "huginn"):
        pairs = [(r["headroom_vs_fifo"], get[(r["study"], r["model"], r["dataset"], r["cohort"], r["B"], "ranker_same")]["headroom_vs_fifo"])
                 for r in rows if r["policy"] == "ranker_cross" and r["model"] == model]
        if pairs:
            extra[f"transfer_retention_{model}"] = statistics.mean(c / s for c, s in pairs if s and s > 0.01)
        nt = [r["wall_median_s"] < get[(r["study"], r["model"], r["dataset"], r["cohort"], r["B"], "ranker_same")]["wall_median_s"]
              for r in rows if r["policy"] == "ranker_notraj" and r["model"] == model]
        if nt:
            extra[f"notraj_faster_than_same_fraction_{model}"] = sum(nt) / len(nt)
    if "notraj_faster_than_same_fraction_huginn" in extra:
        extra["trajectory_cost_hypothesis"] = ("supported" if extra["notraj_faster_than_same_fraction_huginn"] >= 0.75
                                               else "not supported")
    audits = {str(p.relative_to(args.result_dir)): json.loads(p.read_text())["status"]
              for p in sorted((args.result_dir / "audit").rglob("ranker_audit.json"))}
    decision = {"rule": "phase05/WP6_PREREGISTRATION.md", "gate0": audits, "verdicts": verdicts, **extra}
    (out / "decision.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
    plot(rows, out / "headroom_vs_B.png")
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__":
    main()
