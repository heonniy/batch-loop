"""E4 — offline grouping utility of predictor outputs on frozen test cohorts (TRACE_ONLY).

Lockstep no-refill decode boundary: at every token step the N resident requests are partitioned into N/B
cohorts by ascending score (ties by request id). Policies:
  fifo, static_first_true (diagnostic: true first-token depth, fixed), prev_depth, predicted expected depth
  E[D] and deep-probability P(D>=r*) from each saved E2 model, dynamic_true_oracle (not deployable).
Metrics: RIC, RIC recovery=(RIC_FIFO-RIC_pol)/(RIC_FIFO-RIC_oracle) (NaN when the denominator is 0), slot
utilization = work / sum(|m| * max d), mean survivor width per recurrent launch, and for PDF predictors the
expected-cohort-max calibration error E_pred[max] - true max (independence assumption).
No resident KV swapping is modelled; all cohorts are resident-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import lockstep_ric, recovery  # noqa: E402

N_GRID = (64, 128, 256)
B_GRID = (4, 8, 16, 32, 64)


def cohorts(score_t, N, B):
    ids = np.arange(N)
    order = ids if score_t is None else np.lexsort((ids, score_t))
    return [order[i:i + B] for i in range(0, N, B)]


def evaluate(depth, score, probs, N, B):
    work = slots = launches = width_sum = 0
    ric = 0
    emax_err = []
    for t in range(depth.shape[0]):
        for m in cohorts(None if score is None else score[t, :N], N, B):
            d = depth[t, m]
            mx = int(d.max())
            ric += mx
            work += int(d.sum())
            slots += len(m) * mx
            for r in range(1, mx + 1):
                launches += 1
                width_sum += int((d >= r).sum())
            if probs is not None:
                surv = np.cumsum(probs[t, m][:, ::-1], 1)[:, ::-1]  # P(D >= r), r = 1..K
                emax = float((1 - np.prod(1 - surv, 0)).sum())
                emax_err.append(emax - mx)
    return {"RIC": ric, "work": work, "slot_utilization": work / slots, "mean_survivor_width": width_sum / launches,
            "expected_max_error_mean": float(np.mean(emax_err)) if emax_err else None,
            "expected_max_abs_error": float(np.mean(np.abs(emax_err))) if emax_err else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe_root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for lane_dir in sorted(p for p in a.probe_root.iterdir() if (p / "test_predictions.npz").exists()):
        z = np.load(lane_dir / "test_predictions.npz")
        depth = z["depth"]
        K = int(depth.max()) if "huginn" not in lane_dir.name else 16
        policies = {"fifo": (None, None), "static_first_true": (np.broadcast_to(depth[:1], depth.shape).astype(float), None),
                    "prev_depth": (z["prev"].astype(float), None), "dynamic_true_oracle": (depth.astype(float), None)}
        for key in z.files:
            if "|" not in key:
                continue
            P = z[key].astype(np.float64)
            Kp = P.shape[-1]
            ev = P @ np.arange(1, Kp + 1)
            policies[f"pred_E[D]:{key}"] = (ev, P)
            rstar = int(np.ceil(np.median(depth)))
            policies[f"pred_P(D>={rstar}):{key}"] = (P[..., rstar - 1:].sum(-1), P)
        for N in N_GRID:
            if N > depth.shape[1]:
                continue
            d = depth[:, :N]
            for B in B_GRID:
                if B >= N:
                    continue
                res = {name: evaluate(d, None if s is None else s[:, :N], None if p is None else p[:, :N], N, B)
                       for name, (s, p) in policies.items()}
                f, o = res["fifo"]["RIC"], res["dynamic_true_oracle"]["RIC"]
                for name, r in res.items():
                    rows.append({"lane": lane_dir.name, "N": N, "B": B, "policy": name, **r,
                                 "RIC_reduction_vs_fifo": 1 - r["RIC"] / f, "RIC_recovery": recovery(f, r["RIC"], o),
                                 "evidence_class": "TRACE_ONLY"})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "e4_grouping_utility.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    plot(rows, a.out)
    print(json.dumps({"rows": len(rows)}))


def plot(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lanes = sorted({r["lane"] for r in rows})
    fig, axes = plt.subplots(1, len(lanes), figsize=(3.8 * len(lanes), 3.6), sharey=True, squeeze=False)
    for ax, lane in zip(axes[0], lanes):
        sel = [r for r in rows if r["lane"] == lane and r["N"] == 256]
        best_pred = max({r["policy"] for r in sel if r["policy"].startswith("pred_E[D]")},
                        key=lambda p: np.nanmean([r["RIC_recovery"] for r in sel if r["policy"] == p]), default=None)
        for pol, c in (("prev_depth", "#eb6834"), ("static_first_true", "#8a8984"), (best_pred, "#2a78d6"),
                       ("dynamic_true_oracle", "#008300")):
            if pol is None:
                continue
            pts = sorted((r["B"], 100 * r["RIC_reduction_vs_fifo"]) for r in sel if r["policy"] == pol)
            ax.plot(*zip(*pts), marker="o", ms=4, lw=2, color=c, label=pol.split(":")[0] + (" (best E2)" if pol == best_pred else ""))
        ax.set_xscale("log", base=2)
        ax.set_title(lane, fontsize=7)
        ax.set_xlabel("B (N=256)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("RIC reduction vs FIFO (%) — TRACE_ONLY")
    axes[0][-1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "e4_ric_reduction_vs_B.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
