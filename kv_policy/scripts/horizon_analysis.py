#!/usr/bin/env python3
"""Decode-horizon growth of KV-eviction damage, looped vs standard (CPU).

Inputs: a v8 HOR run (looped models: plane/token/mixed arms over the frozen v6 128-token continuations), the frozen v6
census (looped single-token eviction curves), and a standard-model v6 run (single-token census + token policies).
Output: <hor_run>/analysis/HORIZON_FINDINGS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from query_conditioned_analysis import _boot, _ci, _f, _json  # noqa: E402
from query_conditioned_sweep import V6_RUN  # noqa: E402

BINS = [(0, 8, "t 1–8"), (8, 32, "t 9–32"), (32, 128, "t 33–128")]


def hor_rows(run: Path) -> pd.DataFrame:
    jobs = json.loads((run / "jobs.json").read_text())
    rows = []
    for j in (x for x in jobs if x["kind"] == "HOR"):
        f = run / "raw" / f"{j['key']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        z = np.load(run / "raw" / f"{j['key']}.npz")
        for ai, r in enumerate(d["rows"]):
            kl = z["kl"][ai]
            n = len(kl)
            row = {"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "n": n, **{k: v for k, v in r.items()}}
            for lo, hi, lab in BINS:
                row[lab] = float(np.mean(kl[lo:min(hi, n)])) if n > lo else np.nan
            row["kl_curve"] = kl.tolist()
            rows.append(row)
    return pd.DataFrame(rows)


def looped_single_token_curves() -> pd.DataFrame:
    """Frozen v6 census: single-token eviction KL(t) for every eligible source (looped models)."""
    jobs = json.loads((V6_RUN / "jobs.json").read_text())
    rows = []
    for c in (x for x in jobs if x["kind"] == "B"):
        f = V6_RUN / "raw" / f"{c['key']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        meta = json.loads((V6_RUN / "raw" / f"{c['depends']}.json").read_text())
        roles = meta["roles"]
        for si, s in enumerate(z["sources"]):
            kl = z["acc_kl"][si]
            n = int(np.isfinite(kl).sum())
            if roles[int(s)] not in ("content", "relevant_content", "irrelevant_content"):
                continue
            row = {"model": c["model"], "dataset": f"v6_{c['workload']}", "uid": c["uid"], "source": int(s), "n": n}
            for lo, hi, lab in BINS:
                row[lab] = float(np.nanmean(kl[lo:hi])) if n > lo else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def standard_curves(std_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = json.loads((std_run / "jobs.json").read_text())
    cen, pol = [], []
    for j in (x for x in jobs if x["kind"] == "V6U"):
        f = std_run / "raw" / f"{j['key']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        z = np.load(std_run / "raw" / f"{j['key']}.npz")
        n = z["cen_kl"].shape[1]
        for si, s in enumerate(z["sources"]):
            kl = z["cen_kl"][si]
            row = {"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "source": int(s), "n": n}
            for lo, hi, lab in BINS:
                row[lab] = float(np.nanmean(kl[lo:hi])) if n > lo else np.nan
            cen.append(row)
        for r in d["policy_rows"]:
            pol.append({"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "n": n, **r})
    return pd.DataFrame(cen), pd.DataFrame(pol)


def growth_table(df: pd.DataFrame, md: list, title: str, group_cols: list):
    md += [f"## {title}", "", "| " + " | ".join(group_cols) + " | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | "
           "share of prompts with late > early |", "|" + "---|" * (len(group_cols) + 6)]
    out = []
    for key, g in df.groupby(group_cols):
        per = g.groupby("uid")[["t 1–8", "t 9–32", "t 33–128"]].mean()
        per = per[per["t 33–128"].notna()]
        if per.empty:
            continue
        ratio = (per["t 33–128"] / per["t 1–8"].clip(lower=1e-6)).tolist()
        r = {"key": key, "n": len(g), "early": float(per["t 1–8"].mean()), "mid": float(per["t 9–32"].mean()), "late": float(per["t 33–128"].mean()),
             "ratio_median": float(np.median(ratio)), "ratio_ci": _boot([np.log10(x) for x in ratio]),
             "share_growing": float((per["t 33–128"] > per["t 1–8"]).mean()), "n_prompts": len(per)}
        out.append(r)
        ci = r["ratio_ci"]
        md.append("| " + " | ".join(str(k) for k in (key if isinstance(key, tuple) else (key,))) + f" | {len(g)} | {_f(r['early'])} | {_f(r['mid'])} | "
                  f"{_f(r['late'])} | {_f(r['ratio_median'])}" + (f" [10^{ci[0]:.2f},10^{ci[1]:.2f}]" if ci else "") + f" | {_f(r['share_growing'], 2)} ({len(per)} prompts) |")
    md.append("")
    return out


def analyze(hor_run: Path, std_run: Path | None):
    out = hor_run / "analysis"
    out.mkdir(exist_ok=True)
    md = ["# Decode-horizon growth of KV damage: looped (Huginn R=32, Ouro R=4) vs standard (Qwen2.5-3B)", "",
          "Outcome: KL(p_full || p_policy) at generated step t of the model's OWN frozen greedy 128-token continuation "
          "(v6 W1 continuation / W2 summarise-explain prompts), teacher-forced; binned over t. 'late/early' = mean KL at "
          "t 33–128 over mean KL at t 1–8 (per prompt, median; CI on log10). Growth > 1 means damage accumulates through the "
          "generated history; < 1 means the model recovers.", ""]
    res = {}
    hor = hor_rows(hor_run)
    if len(hor):
        hor.drop(columns=["kl_curve"]).to_csv(out / "horizon_long.csv", index=False)
        hor["arm"] = hor.axis + ":" + hor.plane_policy + "@" + hor.scope + ("+" + hor.token_score).where(hor.token_score != "none", "")
        res["looped_arms"] = growth_table(hor, md, "Looped models: plane compression (prompt-only vs prompt+generated), token eviction (30% kept), mixed",
                                          ["model", "dataset", "arm"])
    lo = looped_single_token_curves()
    if len(lo):
        res["looped_single"] = growth_table(lo, md, "Looped models: single Context-token eviction (frozen v6 census, mean over sources)", ["model", "dataset"])
        # top-5% harmful sources only
        top = lo[lo.groupby(["model", "dataset", "uid"])["t 1–8"].transform(lambda s: s >= s.quantile(0.95))]
        res["looped_single_top5"] = growth_table(top, md, "Looped models: single-token eviction, top-5% sources by early damage", ["model", "dataset"])
    if std_run is not None and std_run.exists():
        cen, pol = standard_curves(std_run)
        if len(cen):
            res["std_single"] = growth_table(cen, md, "Standard model: single Context-token eviction (mean over sources)", ["model", "dataset"])
            top = cen[cen.groupby(["model", "dataset", "uid"])["t 1–8"].transform(lambda s: s >= s.quantile(0.95))]
            res["std_single_top5"] = growth_table(top, md, "Standard model: single-token eviction, top-5% sources by early damage", ["model", "dataset"])
        if len(pol):
            pol = pol.rename(columns={"kl_early8": "t 1–8", "kl_late": "t 33–128"})
            pol["t 9–32"] = np.nan
            pol = pol[pol.retention == 0.3]
            res["std_policies"] = growth_table(pol, md, "Standard model: token policies at 30% kept (kl_late = second half of the continuation)", ["model", "dataset", "policy"])
    _json(out / "horizon_summary.json", res)
    (out / "HORIZON_FINDINGS.md").write_text("\n".join(md))
    return {"hor_rows": len(hor), "looped_single": len(lo)}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None), indent=2, default=str))
