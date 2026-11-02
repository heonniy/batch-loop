#!/usr/bin/env python3
"""Loop-wise attention phenomenology (CPU, from frozen v7 REF receipts): how a looped model's attention over the
prompt changes from recurrence 1 to R for the SAME query, with the SAME weights.

For every generated query t (gold continuation) and the question anchor, per loop r:
  entropy of the head-mean attention over prompt tokens; mass on evidence tokens, on the first 4 tokens (sink), on
  the 16 most recent prompt tokens (local), on question tokens; Spearman between the attention vector at loop r and at
  loop R (how much the ranking still moves); and the loop at which the evidence mass peaks.
Also the per-source "loop-profile shape" statistics used by the v7/v8 analyses.

Usage: loop_attention_phenomenology.py <v7 run dir> [<v7 run dir> ...]  -> <first run>/analysis/loop_phenomenology.*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from query_conditioned_analysis import _boot, _ci, _f, _json, _spearman  # noqa: E402


def _entropy(w):
    w = w / max(1e-12, w.sum())
    return float(-(w * np.log(w + 1e-12)).sum())


def per_unit(run: Path, ref: dict) -> list[dict]:
    raw = run / "raw"
    m = json.loads((raw / f"{ref['key']}.json").read_text())
    z = np.load(raw / f"{ref['key']}.npz")
    P = len(m["prompt_ids"])
    roles = np.array(m["roles"])
    ev = roles == "relevant_content"
    qmask = roles == "question"
    content = np.array(m["content_positions"])
    ga = z["gen_attn"]                                   # (n, R, P)
    qa_raw = z["anchor_question_last_attn"]              # (R, anchor col)
    qa = np.zeros((qa_raw.shape[0], P), dtype=qa_raw.dtype)
    qa[:, :qa_raw.shape[1]] = qa_raw                     # (R, P)
    n, R, _ = ga.shape
    rows = []
    for kind, A in (("generated", ga), ("question_anchor", qa[None])):
        for t in range(A.shape[0]):
            W = A[t]                                     # (R, P)
            last = W[-1]
            for r in range(R):
                w = W[r]
                cw = w[content]
                rows.append({"model": ref["model"], "dataset": ref["dataset"], "uid": ref["uid"], "query": kind, "t": t,
                             "loop": r + 1, "loop_frac": (r + 1) / R, "R": R,
                             "entropy_content": _entropy(cw), "mass_total": float(w.sum()),
                             "mass_evidence": float(w[ev].sum() / max(1e-12, w[content].sum())),
                             "mass_sink4": float(w[:4].sum() / max(1e-12, w.sum())),
                             "mass_local16": float(w[max(0, P - 16):P].sum() / max(1e-12, w.sum())),
                             "mass_question": float(w[qmask].sum() / max(1e-12, w.sum())),
                             "rho_vs_last": _spearman(cw, last[content]),
                             "rho_vs_prev": _spearman(cw, W[r - 1][content]) if r > 0 else np.nan,
                             "top1_is_evidence": float(ev[content][int(cw.argmax())]),
                             "top8_evidence_frac": float(ev[content][np.argsort(-cw)[:8]].mean())})
    return rows


def main(runs: list[Path]):
    rows = []
    for run in runs:
        jobs = json.loads((run / "jobs.json").read_text())
        for ref in [j for j in jobs if j["kind"] == "REF" and (run / "raw" / f"{j['key']}.npz").exists()]:
            rows += per_unit(run, ref)
    df = pd.DataFrame(rows)
    out = runs[0] / "analysis"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "loop_phenomenology_long.parquet", index=False)
    md = ["# Loop-wise attention phenomenology (same query, same weights, recurrence 1..R)", "",
          f"Runs: {', '.join(r.name for r in runs)}; prompts: " + ", ".join(
              f"{m}/{d}: {g.uid.nunique()}" for (m, d), g in df.groupby(['model', 'dataset'])), ""]
    # loop-binned summary for generated queries
    bins = [0, 0.126, 0.26, 0.51, 0.76, 1.0]
    df["loop_bin"] = pd.cut(df.loop_frac, bins, labels=["≤1/8", "1/8–1/4", "1/4–1/2", "1/2–3/4", "3/4–1"], include_lowest=True)
    md += ["## Generated queries (gold continuation): per-loop statistics, mean over queries and prompts", "",
           "| model | dataset | loop bin | entropy over Context | evidence mass (of Context) | sink mass (first 4) | local mass (last 16) | "
           "question mass | ρ(loop r, loop R) | ρ(loop r, loop r−1) | top-1 is evidence | top-8 evidence frac |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = []
    for (m, d, b), g in df[df["query"] == "generated"].groupby(["model", "dataset", "loop_bin"], observed=True):
        r = {"model": m, "dataset": d, "loop_bin": str(b), **{k: float(g[k].mean()) for k in (
            "entropy_content", "mass_evidence", "mass_sink4", "mass_local16", "mass_question", "rho_vs_last", "rho_vs_prev",
            "top1_is_evidence", "top8_evidence_frac")}}
        summ.append(r)
        md.append(f"| {m} | {d} | {b} | {_f(r['entropy_content'])} | {_f(r['mass_evidence'])} | {_f(r['mass_sink4'])} | {_f(r['mass_local16'])} | "
                  f"{_f(r['mass_question'])} | {_f(r['rho_vs_last'])} | {_f(r['rho_vs_prev'])} | {_f(r['top1_is_evidence'])} | {_f(r['top8_evidence_frac'])} |")
    md.append("")
    # exact per-loop curves (Ouro: 4 loops; Huginn: selected loops)
    md += ["## Exact per-loop curves (generated queries): evidence mass and ρ(loop r, loop R)", "",
           "| model | dataset | loop | evidence mass [CI over prompts] | ρ vs last [CI] | entropy | top-8 evidence frac |", "|---|---|---|---|---|---|---|"]
    curves = []
    for (m, d), g in df[df["query"] == "generated"].groupby(["model", "dataset"]):
        R = int(g.R.iloc[0])
        loops = range(1, R + 1) if R <= 8 else [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32]
        for r in loops:
            gg = g[g.loop == r]
            per = gg.groupby("uid").mass_evidence.mean().tolist()
            rho = gg.groupby("uid").rho_vs_last.mean().tolist()
            c = {"model": m, "dataset": d, "loop": r, "evidence_mass": float(np.mean(per)), "evidence_ci": _boot(per),
                 "rho_vs_last": float(np.nanmean(rho)), "rho_ci": _boot(rho), "entropy": float(gg.entropy_content.mean()),
                 "top8_evidence_frac": float(gg.top8_evidence_frac.mean())}
            curves.append(c)
            md.append(f"| {m} | {d} | {r} | {_f(c['evidence_mass'])}{_ci(c['evidence_ci'])} | {_f(c['rho_vs_last'])}{_ci(c['rho_ci'])} | "
                      f"{_f(c['entropy'])} | {_f(c['top8_evidence_frac'])} |")
    md.append("")
    # peak loop of evidence mass per query
    md += ["## Where in the recurrence does a query's evidence attention peak? (fraction of generated queries by peak-loop bin)", "",
           "| model | dataset | ≤1/8 | 1/8–1/4 | 1/4–1/2 | 1/2–3/4 | 3/4–1 | mean peak loop / R |", "|---|---|---|---|---|---|---|---|"]
    peaks = []
    for (m, d), g in df[df["query"] == "generated"].groupby(["model", "dataset"]):
        pk = g.groupby(["uid", "t"]).apply(lambda x: x.loc[x.mass_evidence.idxmax()].loop_frac)
        hist = pd.cut(pk, bins, labels=["≤1/8", "1/8–1/4", "1/4–1/2", "1/2–3/4", "3/4–1"], include_lowest=True).value_counts(normalize=True)
        peaks.append({"model": m, "dataset": d, "hist": {str(k): float(v) for k, v in hist.items()}, "mean_peak_frac": float(pk.mean())})
        md.append(f"| {m} | {d} | " + " | ".join(_f(hist.get(k, 0.0), 2) for k in ["≤1/8", "1/8–1/4", "1/4–1/2", "1/2–3/4", "3/4–1"]) + f" | {_f(pk.mean())} |")
    md.append("")
    # question anchor vs generated at the last loop: do they look at the same tokens?
    # cross-query agreement: question anchor vs generated queries, and loop-1 vs loop-R of the same query
    md += ["## Cross-query / cross-loop agreement of Context attention (Spearman over Context tokens)", "",
           "| model | dataset | anchor(loop R) vs generated t*(loop R) | anchor(loop 1) vs generated t*(loop R) | generated t*: loop 1 vs loop R | "
           "generated t=1 (format) vs t* at loop R |", "|---|---|---|---|---|---|"]
    xq = []
    for run in runs:
        jobs = json.loads((run / "jobs.json").read_text())
        for ref in [j for j in jobs if j["kind"] == "REF" and (run / "raw" / f"{j['key']}.npz").exists()]:
            m = json.loads((run / "raw" / f"{ref['key']}.json").read_text())
            z = np.load(run / "raw" / f"{ref['key']}.npz")
            content = np.array(m["content_positions"])
            ga = z["gen_attn"]
            P = len(m["prompt_ids"])
            qa_raw = z["anchor_question_last_attn"]
            qa = np.zeros((qa_raw.shape[0], P)); qa[:, :qa_raw.shape[1]] = qa_raw
            ev = np.array(m["roles"]) == "relevant_content"
            # consuming query = generated query with the largest evidence mass at loop R
            tstar = int(ga[:, -1, ev].sum(1).argmax()) if ev.any() else 0
            xq.append({"model": ref["model"], "dataset": ref["dataset"],
                       "aR_tR": _spearman(qa[-1, content], ga[tstar, -1, content]), "a1_tR": _spearman(qa[0, content], ga[tstar, -1, content]),
                       "t1_tR": _spearman(ga[tstar, 0, content], ga[tstar, -1, content]), "f_tR": _spearman(ga[0, -1, content], ga[tstar, -1, content])})
    X = pd.DataFrame(xq)
    for (m, d), g in X.groupby(["model", "dataset"]):
        md.append(f"| {m} | {d} | {_f(g.aR_tR.mean())}{_ci(_boot(g.aR_tR.tolist()))} | {_f(g.a1_tR.mean())} | {_f(g.t1_tR.mean())}{_ci(_boot(g.t1_tR.tolist()))} | {_f(g.f_tR.mean())} |")
    md.append("")
    _json(out / "loop_phenomenology.json", {"bins": summ, "curves": curves, "peaks": peaks})
    (out / "LOOP_PHENOMENOLOGY.md").write_text("\n".join(md))
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main([Path(a).resolve() for a in sys.argv[1:]])
