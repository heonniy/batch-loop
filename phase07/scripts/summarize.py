"""Phase-07 summary tables and plots from the committed raw outputs (no new measurement).

Inputs (under the run directory): probes/*/probe_results.csv + test_predictions.npz, analysis/e1/*.csv,
analysis/e4/e4_grouping_utility.csv, e4_physical/*/{e4_physical_runs.csv,RUN_MANIFEST.json}, e0/*.json,
and the committed Phase-03 tournament (phase03/predictor_tournament_results_20260916).
Outputs: analysis/summary/*.csv, tables.md and plots (calibration/confusion, overhead vs RIC recovery,
oracle gap by B/model). All recoveries are (FIFO-X)/(FIFO-Oracle) on the stated cohort.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P03 = ROOT / "phase03/predictor_tournament_results_20260916"
PRIMARY = "all_precore|h64|w0"  # predeclared physical predictor (E2 all pre-core features, MLP-64, unweighted)


def read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def write(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ms(v):
    v = [float(x) for x in v]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def md(rows, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if r[c] not in (None, "") else "—" for c in cols) + " |")
    return "\n".join(out)


def e2_summary(run):
    g = defaultdict(list)
    for f in sorted(glob.glob(str(run / "probes/*/probe_results.csv"))):
        for r in read(f):
            g[(r["lane"], r["stage"], r["features"], r["hidden"], r["class_weighted"])].append(r)
    rows = []
    for (lane, stage, feat, h, w), v in g.items():
        row = {"lane": lane, "stage": stage, "features": feat, "hidden": int(h), "class_weighted": w, "seeds": len(v),
               "dim": int(v[0]["dim"]), "params": int(v[0]["params"]),
               "n_test_survivors": v[0].get("n_test_survivors") or None}
        for k in ("nll", "brier", "ece", "acc", "mae_expected", "kendall_tau_b", "RIC_recovery_B8", "RIC_recovery_B16",
                  "RIC_recovery_B32"):
            if v[0].get(k) not in (None, ""):
                row[k], row[k + "_sd"] = ms(x[k] for x in v)
        rows.append(row)
    return rows


def physical(run):
    out = []
    for d in sorted((run / "e4_physical").iterdir()):
        man = json.loads((d / "RUN_MANIFEST.json").read_text())
        oh = {o["hidden"]: o for o in man["predictor_overhead"]}
        g = defaultdict(list)
        for r in read(d / "e4_physical_runs.csv"):
            if int(r["rep"]) >= 0:
                g[(int(r["N"]), int(r["B"]), int(r["T"]), r["policy"])].append(r)
        for (N, B, T, pol), v in sorted(g.items()):
            if pol == "native_single_cohort":
                continue
            f, o = g[(N, B, T, "fifo")], g[(N, B, T, "depth_oracle")]
            med = lambda rr: st.median(float(x["decode_wall_s"]) for x in rr)
            tf, to, tp = med(f), med(o), med(v)
            ric = lambda rr: int(rr[0]["RIC"])
            # measured predictor cost per decode step (synchronised wall, gather+MLP+E[D]+argsort at width N)
            add = T * oh[64]["sync_wall_ms_median"] / 1000 if pol == "predicted" else 0.0
            nat = g.get((N, N, T, "native_single_cohort")) or [r for (k, rr) in g.items() if k[3] == "native_single_cohort" for r in rr]
            out.append({"lane": d.name, "N": N, "B": B, "T": T, "policy": pol, "reps": len(v),
                        "decode_wall_median_s": tp, "decode_wall_min_s": min(float(x["decode_wall_s"]) for x in v),
                        "decode_wall_max_s": max(float(x["decode_wall_s"]) for x in v),
                        "RIC": ric(v), "work_hash_unique": len({x["work_hash"] for x in v}),
                        "n_errors": sum(int(x["n_errors"]) for x in v),
                        "RIC_recovery": (ric(f) - ric(v)) / (ric(f) - ric(o)),
                        "wall_reduction_vs_fifo": (tf - tp) / tf, "wall_recovery": (tf - tp) / (tf - to),
                        "predictor_overhead_s": add, "net_wall_reduction_vs_fifo": (tf - tp - add) / tf,
                        "net_wall_recovery": (tf - tp - add) / (tf - to),
                        "native_B_eq_N_wall_median_s": med(nat) if nat else None,
                        "evidence_class": "PHYSICAL_FORCED_DEPTH"})
    return out


def phase03_compare(run, e4rows, phys):
    """Matched cohort: N=256 held-out requests, B=32, Ouro q0.2 / Huginn tau0.12, Alpaca, Ouro-1.4B."""
    p3 = [r for r in read(P03 / "predictor_tournament.csv") if r["split"] == "test" and r["batch_size"] == "32"]
    p3phys = read(P03 / "physical_matrix.csv")
    lanes = {"ouro": ("ouro14_t0.2_alpaca", "ouro_q0p2_n640_t32"), "huginn": ("huginn_t0.12_alpaca", "huginn_threshold_0p12_n640_t32")}
    rows = []
    for model, (lane, regime) in lanes.items():
        for c in sorted({r["candidate"] for r in p3 if r["model"] == model and r["regime"] == regime}):
            r = next(r for r in p3 if r["model"] == model and r["regime"] == regime and r["candidate"] == c)
            rows.append({"model": model, "source": "phase03", "predictor": c, "cohort": "N256/B32/T32 test",
                         "trace_RIC_recovery": float(r["oracle_ric_recovery"]), "kendall_tau_b": float(r["kendall_tau_b"]),
                         "exact_acc": float(r["exact_depth_accuracy"])})
        sel = [r for r in e4rows if r["lane"] == lane and r["N"] == "256" and r["B"] == "32"]
        for pol, label in (("prev_depth", "P07 prev-depth (≈C0)"), ("pred_E[D]:cheap|h64|w0", "P07 cheap MLP (history, ≈C1)"),
                           ("pred_E[D]:cheap+tok|h64|w0", "P07 cheap+token-lookup MLP"),
                           ("pred_E[D]:emb|h64|w0", "P07 emb MLP (≈C4)"),
                           ("pred_E[D]:prevh|h64|w0", "P07 prev-token final state MLP"),
                           ("pred_E[D]:pre|h64|w0", "P07 Huginn Prelude MLP (≈C5)"),
                           ("pred_E[D]:all_precore|h0|w0", "P07 all pre-core linear"),
                           (f"pred_E[D]:{PRIMARY}", "P07 all pre-core MLP-64 (primary)")):
            m = [r for r in sel if r["policy"] == pol]
            if m:
                rows.append({"model": model, "source": "phase07", "predictor": label, "cohort": "N256/B32/T64 test",
                             "trace_RIC_recovery": float(m[0]["RIC_recovery"])})
        for r in p3phys:
            if r["model"] == model and r["regime"] == regime:  # wall timer includes predictor/staging (Phase-03 runtime)
                rows.append({"model": model, "source": "phase03 physical", "predictor": r["strategy"], "cohort": "N256/B32/T32 test",
                             "physical_RIC_recovery": float(r["ric_recovery"]), "wall_recovery": float(r["wall_recovery"]),
                             "median_decode_wall_s": float(r["median_decode_wall_s"]),
                             "predictor_plus_sort_wall_s": float(r["median_predictor_wall_s"]) + float(r["median_sort_wall_s"])})
        for p in phys:
            if p["lane"] == lane and p["B"] == 32:
                rows.append({"model": model, "source": "phase07 physical", "predictor": p["policy"], "cohort": "N256/B32/T32 test",
                             "physical_RIC_recovery": p["RIC_recovery"], "wall_recovery": p["wall_recovery"],
                             "net_wall_recovery": p["net_wall_recovery"], "wall_reduction_vs_fifo": p["wall_reduction_vs_fifo"]})
    return rows


def plots(run, out, e2rows, e4rows, phys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, ink, mut = "#fcfcfb", "#1d1d1b", "#8a8984"
    lanes = sorted({r["lane"] for r in e2rows})
    # 1. reliability + confusion for the primary predictor
    fig, axes = plt.subplots(2, len(lanes), figsize=(2.6 * len(lanes), 5.2), squeeze=False)
    for j, lane in enumerate(lanes):
        z = np.load(run / "probes" / lane / "test_predictions.npz")
        P, y = z[PRIMARY].reshape(-1, z[PRIMARY].shape[-1]), z["depth"].ravel()
        conf, pred = P.max(1), P.argmax(1) + 1
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [(pred[idx == b] == y[idx == b]).mean() for b in range(10) if (idx == b).sum() >= 20]
        ax = axes[0][j]
        ax.plot([0, 1], [0, 1], color=mut, lw=1, ls="--")
        ax.plot(xs, ys, marker="o", ms=4, lw=2, color="#2a78d6")
        ax.set_title(lane, fontsize=7, color=ink)
        ax.set_xlabel("confidence", fontsize=7)
        if j == 0:
            ax.set_ylabel("accuracy (reliability)", fontsize=7)
        K = P.shape[1]
        lo = max(1, int(np.percentile(y, 1)))
        hi = int(np.percentile(y, 99.5))
        cm = np.zeros((hi - lo + 1, hi - lo + 1))
        for t, p in zip(np.clip(y, lo, hi), np.clip(pred, lo, hi)):
            cm[t - lo, p - lo] += 1
        cm /= cm.sum(1, keepdims=True).clip(1)
        ax = axes[1][j]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, extent=(lo - .5, hi + .5, hi + .5, lo - .5))
        ax.set_xlabel("predicted depth", fontsize=7)
        if j == 0:
            ax.set_ylabel("true depth (row-normalised)", fontsize=7)
        for a in axes[:, j]:
            a.tick_params(labelsize=6)
    fig.suptitle(f"E2 primary predictor {PRIMARY}: reliability (top) and confusion (bottom), test split", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "e2_calibration_confusion.png", dpi=150, facecolor=bg)
    plt.close(fig)
    # 2. predictor cost vs trace RIC recovery (N256/B32) per lane
    fig, ax = plt.subplots(figsize=(6.4, 4))
    fam = {"ouro14": "#2a78d6", "ouro26": "#7b3fbf", "huginn": "#eb6834"}
    for lane in lanes:
        sel = [r for r in e4rows if r["lane"] == lane and r["N"] == "256" and r["B"] == "32" and r["policy"].startswith("pred_E[D]")]
        par = {(r["features"], r["hidden"]): r["params"] for r in e2rows if r["lane"] == lane and r["stage"] == "E2_precore"}
        pts = []
        for r in sel:
            f, h, w = r["policy"].split(":", 1)[1].split("|")
            if w == "w0" and (f, int(h[1:])) in par:
                pts.append((par[(f, int(h[1:]))], float(r["RIC_recovery"])))
        c = fam[next(k for k in fam if lane.startswith(k))]
        ax.scatter(*zip(*pts), s=14, color=c, alpha=.7, label=lane, edgecolors=bg, linewidths=.5)
    ax.set_xscale("log")
    ax.set_xlabel("predictor parameters (≈ MACs / token)")
    ax.set_ylabel("trace RIC recovery, N256/B32")
    ax.legend(frameon=False, fontsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Predictor cost vs grouping utility (TRACE_ONLY). Measured GPU cost of MLP-64 ≈0.15 ms/step at N=256", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e4_overhead_vs_recovery.png", dpi=150, facecolor=bg)
    plt.close(fig)
    # 3. oracle gap by B / model (N=256): oracle RIC reduction and primary predictor reduction
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for lane in lanes:
        c = fam[next(k for k in fam if lane.startswith(k))]
        ls = "-" if "alpaca" in lane else ":"
        for pol, alpha in (("dynamic_true_oracle", 1.0), (f"pred_E[D]:{PRIMARY}", .55)):
            pts = sorted((int(r["B"]), 100 * float(r["RIC_reduction_vs_fifo"])) for r in e4rows
                         if r["lane"] == lane and r["N"] == "256" and r["policy"] == pol)
            ax.plot(*zip(*pts), color=c, ls=ls, alpha=alpha, lw=2 if alpha == 1 else 1.5, marker="o", ms=3,
                    label=f"{lane} {'oracle' if alpha == 1 else 'predictor'}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cohort width B (N=256)")
    ax.set_ylabel("RIC reduction vs FIFO (%)")
    ax.legend(frameon=False, fontsize=5, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Oracle headroom vs primary pre-core predictor (TRACE_ONLY)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "e4_oracle_gap_by_B.png", dpi=150, facecolor=bg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    out = a.run / "analysis/summary"
    out.mkdir(parents=True, exist_ok=True)
    e2 = e2_summary(a.run)
    e4 = read(a.run / "analysis/e4/e4_grouping_utility.csv")
    phys = physical(a.run)
    cmp_ = phase03_compare(a.run, e4, phys)
    write(out / "e2_e3_probe_summary.csv", e2)
    write(out / "e4_physical_summary.csv", phys)
    write(out / "phase03_comparison.csv", cmp_)
    e0 = []
    for f in sorted((a.run / "e0").glob("e0_parity_*.json")):
        r = json.loads(f.read_text())
        for B, v in r["widths"].items():
            e0.append({"file": f.name, "B": B, **{k: v[k] for k in v if not isinstance(v[k], dict)},
                       "verdict": r.get("gate_semantic") or r.get("live_parity")})
    write(out / "e0_parity_summary.csv", e0)
    plots(a.run, out, e2, e4, phys)
    print(json.dumps({"e2_rows": len(e2), "physical_rows": len(phys), "phase03_rows": len(cmp_), "e0_rows": len(e0)}))


if __name__ == "__main__":
    main()
