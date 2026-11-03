#!/usr/bin/env python3
"""v8 CPU analysis: two-axis KV budgets (tokens x loop planes) for looped transformers.

Writes <run>/analysis/{completeness.json, tax_long.csv, tax_summary.csv, pln_long.csv, pln_summary.json,
PLANE_BUDGET_FINDINGS.md}. All comparisons are paired within prompt; CIs are prompt bootstraps.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from query_conditioned_analysis import _boot, _ci, _f, _json, _lexical, _spearman, _tokenizers  # noqa: E402


def completeness(run: Path) -> dict:
    jobs = json.loads((run / "jobs.json").read_text())
    raw = run / "raw"
    missing = [j["key"] for j in jobs if not (raw / f"{j['key']}.json").exists()
               or (j["kind"] in ("REF", "PLN", "HOR") and not (raw / f"{j['key']}.npz").exists())]
    errs = sorted(p.stem for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    problems = []
    cov = defaultdict(set)
    for j in jobs:
        if j["kind"] == "PLN" and (raw / f"{j['key']}.json").exists():
            cov[j["depends"]].update(j["sources"])
    for j in jobs:
        if j["kind"] == "REF" and (raw / f"{j['key']}.json").exists():
            m = json.loads((raw / f"{j['key']}.json").read_text())
            if not (m["validation_v7"]["pass"] and m["validation_planes"]["pass"]):
                problems.append(f"validation failed {j['key']}")
            want = {s for jj in jobs if jj.get("depends") == j["key"] and jj["kind"] == "PLN" for s in jj["sources"]}
            if cov[j["key"]] != want:
                problems.append(f"PLN coverage {j['key']}: {len(cov[j['key']])}/{len(want)}")
    out = {"expected": len(jobs), "completed": len(jobs) - len(missing), "missing": missing, "errors": errs,
           "problems": problems, "complete": not missing and not errs and not problems}
    _json(run / "analysis" / "completeness.json", out)
    return out


# ------------------------------------------------------------------------------------------- TAX
def tax_table(run: Path) -> pd.DataFrame:
    jobs = json.loads((run / "jobs.json").read_text())
    rows = []
    for j in (x for x in jobs if x["kind"] == "TAX"):
        f = run / "raw" / f"{j['key']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for r in d["rows"]:
            rows.append({"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "split": j["split"], "R": d["R"],
                         "full_em": float(d["full"]["exact_match"]), "full_nll": d["full"]["nll_future"], **r,
                         "budget": r["budget"] if r["axis"] != "natural" else -1.0,
                         "exact_match": float(r["exact_match"]), "budget_used": r["planes_kept"] / r["planes_total"]})
    return pd.DataFrame(rows)


def _arm_name(r):
    if r.axis == "natural":
        return f"natural:{r.plane_policy}"
    if r.axis == "token":
        return f"token:{r.token_score}"
    if r.axis == "plane":
        return f"plane:{r.plane_policy}" + ("" if r.semantics == "nearest" else f"[{r.semantics}]")
    return f"mixed:{r.token_score}+{r.plane_policy}"


def tax_summary(df: pd.DataFrame, out: Path, md: list) -> pd.DataFrame:
    df = df.assign(arm=df.apply(_arm_name, axis=1))
    res = []
    for (m, ds, sp, b), g in df.groupby(["model", "dataset", "split", "budget"]):
        rnd = g[g.arm.isin(("token:random_11", "token:random_29"))].groupby("uid").delta_nll_future.mean()
        rnd_em = g[g.arm.isin(("token:random_11", "token:random_29"))].groupby("uid").exact_match.mean()
        for arm, ga in g.groupby("arm"):
            per = ga.set_index("uid")
            dn = per.delta_nll_future
            paired = [dn[u] - rnd[u] for u in per.index if u in rnd.index]
            paired_em = [per.exact_match[u] - rnd_em[u] for u in per.index if u in rnd_em.index]
            res.append({"model": m, "dataset": ds, "split": sp, "budget": b, "arm": arm, "axis": per.axis.iloc[0],
                        "n_prompts": len(per), "budget_used": float(per.budget_used.mean()),
                        "dnll_mean": float(dn.mean()), "dnll_ci": _boot(dn.tolist()), "kl_mean": float(per.kl_mean.mean()),
                        "em": float(per.exact_match.mean()), "full_em": float(per.full_em.mean()),
                        "dnll_vs_random": float(np.mean(paired)) if paired else np.nan, "dnll_vs_random_ci": _boot(paired),
                        "em_vs_random": float(np.mean(paired_em)) if paired_em else np.nan})
    S = pd.DataFrame(res)
    S.to_csv(out / "tax_summary.csv", index=False)
    md += ["## Two-axis equal-plane-budget policies (gold ΔNLL_future, greedy EM; paired vs random-token eviction with full planes)", "",
           "Budget = kept planes / (Context tokens × R). token:* = keep budget·|C| tokens with ALL planes; plane:* = keep ALL "
           "tokens with budget·R planes each (last-k / first+last / evenly spaced / settle-adaptive allocation / most-attended "
           "loops); mixed:* = √budget tokens × √budget planes. Scores: qa_rR = question-anchor loop-R attention, recv_rR = "
           "all later prompt queries loop-R, la2_rR = 2-token format lookahead, def4_rR = deferred after 4 generated tokens, "
           "fut_oracle_rR = all generated queries (oracle). '[latest]' = read the latest kept plane ≤ r instead of the nearest.", ""]
    for (m, ds, sp), g in S.groupby(["model", "dataset", "split"]):
        md += [f"### {m} / {ds} / {sp} — {int(g.n_prompts.max())} prompts, FULL EM {_f(g.full_em.iloc[0], 2)}", "",
               "| budget | arm | used | ΔNLL mean [CI] | KL | EM | ΔNLL − random [CI] | EM − random |", "|---|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values(["budget", "axis", "arm"]).iterrows():
            md.append(f"| {r.budget} | {r.arm} | {_f(r.budget_used, 2)} | {_f(r.dnll_mean)}{_ci(r.dnll_ci)} | {_f(r.kl_mean)} | "
                      f"{_f(r.em, 2)} | {_f(r.dnll_vs_random)}{_ci(r.dnll_vs_random_ci)} | {_f(r.em_vs_random, 2)} |")
        md.append("")
    # best arm per cell/budget and the axis comparison
    md += ["### Best arm per cell and budget (by mean ΔNLL; deployable arms only = no def4/fut_oracle)", "",
           "| model | dataset | split | budget | best deployable arm | ΔNLL | EM | best plane-only | ΔNLL | best token-only (deployable) | ΔNLL | oracle token-only ΔNLL |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, ds, sp, b), g in S.groupby(["model", "dataset", "split", "budget"]):
        if b < 0:
            continue
        dep = g[~g.arm.str.contains("def4|fut_oracle")]
        best = dep.loc[dep.dnll_mean.idxmin()]
        pl = g[g.axis == "plane"]
        bp = pl.loc[pl.dnll_mean.idxmin()] if len(pl) else None
        tk = dep[(dep.axis == "token") & ~dep.arm.str.contains("random")]
        bt = tk.loc[tk.dnll_mean.idxmin()] if len(tk) else None
        orc = g[g.arm == "token:fut_oracle_rR"]
        md.append(f"| {m} | {ds} | {sp} | {b} | {best.arm} | {_f(best.dnll_mean)} | {_f(best.em, 2)} | "
                  f"{bp.arm if bp is not None else '—'} | {_f(bp.dnll_mean) if bp is not None else '—'} | "
                  f"{bt.arm if bt is not None else '—'} | {_f(bt.dnll_mean) if bt is not None else '—'} | "
                  f"{_f(orc.dnll_mean.iloc[0]) if len(orc) else '—'} |")
    md.append("")
    return S


# ------------------------------------------------------------------------------------------- PLN
def pln_table(run: Path) -> pd.DataFrame:
    jobs = json.loads((run / "jobs.json").read_text())
    raw = run / "raw"
    cfg = json.loads((run / "run_manifest.json").read_text())["config"]
    toks = _tokenizers(cfg["models"])
    rows = []
    for ref in (x for x in jobs if x["kind"] == "REF"):
        if not (raw / f"{ref['key']}.npz").exists():
            continue
        m = json.loads((raw / f"{ref['key']}.json").read_text())
        z = np.load(raw / f"{ref['key']}.npz")
        R = z["qa"].shape[0]
        recv, ga, settle, U, F = z["recv"], z["gen_attn"], z["settle"], z["prompt_U"], z["prompt_F"]
        exit_depth = z["exit_depth"] if "exit_depth" in z.files else None
        roles = m["roles"]
        tok = toks[ref["model"]]
        for c in (x for x in jobs if x.get("depends") == ref["key"] and x["kind"] == "PLN"):
            if not (raw / f"{c['key']}.npz").exists():
                continue
            zc = np.load(raw / f"{c['key']}.npz")
            for si, s in enumerate(zc["sources"]):
                s = int(s)
                prof = ga[:, :, s].mean(0)                       # (R,) mean over generated queries
                tstar = int(ga[:, :, s].mean(1).argmax())
                row = {"model": ref["model"], "dataset": ref["dataset"], "uid": ref["uid"], "split": ref["split"], "source": s,
                       "role": roles[s], "is_evidence": float(roles[s] == "relevant_content"), "R": R,
                       "settle": int(settle[s]), "settle_frac": float(settle[s]) / (R - 1), "F_mid": float(F[s][R // 2]),
                       "exit_depth": int(exit_depth[s]) if exit_depth is not None else np.nan,
                       "F_last_minus1": float(F[s][R - 1]), "U_mean": float(np.nanmean(U[s][1:])), "U_last": float(U[s][R]),
                       "U_late_mean": float(np.nanmean(U[s][R // 2 + 1:])),
                       "recv_mean": float(recv[:, s].mean()), "recv_rR": float(recv[-1, s]), "recv_r1": float(recv[0, s]),
                       "recv_lastloop_frac": float(recv[-1, s] / max(1e-12, recv[:, s].sum())),
                       "fut_attn_mean": float(prof.mean()), "fut_attn_rR": float(prof[-1]), "fut_attn_r1": float(prof[0]),
                       "fut_attn_at_t": float(ga[tstar, :, s].mean()),
                       "fut_lastloop_frac": float(prof[-1] / max(1e-12, prof.sum())),
                       "fut_earlyloop_frac": float(prof[:max(1, R // 4)].sum() / max(1e-12, prof.sum())),
                       **{k: v for k, v in _lexical(tok, m["prompt_ids"][s]).items()}}
                for v in c["variants"]:
                    for k in ("dnll", "kl_mean", "kl_max", "first_dnll"):
                        row[f"{v}_{k}"] = float(zc[f"{v}_{k}"][si])
                rows.append(row)
    return pd.DataFrame(rows)


PLN_FEATS = {"position/lexical": ["source", "tok_chars", "tok_word_start", "tok_punct", "tok_digit", "tok_upper"],
             "attention (received, loop-R)": ["recv_mean", "recv_rR", "recv_r1", "recv_lastloop_frac"],
             "own trajectory (settle/U)": ["settle_frac", "F_mid", "U_mean", "U_last", "U_late_mean"],
             "future-query loop profile (oracle)": ["fut_attn_mean", "fut_attn_rR", "fut_attn_r1", "fut_lastloop_frac", "fut_earlyloop_frac"]}


def pln_summary(df: pd.DataFrame, out: Path, md: list):
    md += ["## Per-token loop-plane compression census (one Context token at a time; gold ΔNLL / KL at fixed depth)", "",
           "| model | dataset | split | variant | n tokens | median KL | mean KL | p95 KL | share of total KL in top 5% | mean ΔNLL | "
           "frac ΔNLL>0.01 | evidence mean KL | non-evidence mean KL |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for (m, ds, sp), g in df.groupby(["model", "dataset", "split"]):
        for v in ("last", "first", "half", "exit"):
            c = f"{v}_kl_mean"
            if c not in g.columns or g[c].isna().all():
                continue
            kl = g[c]
            k = max(1, int(round(0.05 * len(g))))
            r = {"model": m, "dataset": ds, "split": sp, "variant": v, "n": int(kl.notna().sum()), "kl_median": float(kl.median()),
                 "kl_mean": float(kl.mean()), "kl_p95": float(kl.quantile(.95)), "top5_share": float(g.nlargest(k, c)[c].sum() / max(1e-12, kl.sum())),
                 "dnll_mean": float(g[f"{v}_dnll"].mean()), "frac_dnll_gt_001": float((g[f"{v}_dnll"] > 0.01).mean()),
                 "kl_evidence": float(g[g.is_evidence == 1][c].mean()), "kl_other": float(g[g.is_evidence == 0][c].mean())}
            res.append(r)
            md.append(f"| {m} | {ds} | {sp} | {v} | {r['n']} | {_f(r['kl_median'])} | {_f(r['kl_mean'])} | {_f(r['kl_p95'])} | {_f(r['top5_share'])} | "
                      f"{_f(r['dnll_mean'])} | {_f(r['frac_dnll_gt_001'])} | {_f(r['kl_evidence'])} | {_f(r['kl_other'])} |")
    md.append("")
    if "exit_kl_mean" in df.columns:
        md += ["### Exit-truncated planes (what native early exit would leave) vs last-only, per token", "",
               "| model | dataset | split | n | mean exit-depth / R | mean KL exit-truncated | mean KL last-only | frac tokens exit KL > last KL | "
               "ρ(exit depth, exit KL) |", "|---|---|---|---|---|---|---|---|---|"]
        for (m, ds, sp), g in df.groupby(["model", "dataset", "split"]):
            g = g[g.exit_kl_mean.notna()]
            if g.empty:
                continue
            md.append(f"| {m} | {ds} | {sp} | {len(g)} | {_f(g.exit_depth.mean() / g.R.iloc[0]) if 'exit_depth' in g.columns else '—'} | "
                      f"{_f(g.exit_kl_mean.mean())} | {_f(g.last_kl_mean.mean())} | {_f((g.exit_kl_mean > g.last_kl_mean).mean())} | "
                      f"{_f(np.nanmean([_spearman(gg.exit_depth, gg.exit_kl_mean) for _, gg in g.groupby('uid')])) if 'exit_depth' in g.columns else '—'} |")
        md.append("")
    # within-prompt correlations of plane damage with feature families
    md += ["### Within-prompt Spearman of per-token plane damage (last-only KL) with clean features (mean over prompts [CI])", "",
           "| model | dataset | split | feature | ρ (last-only KL) | ρ (first-only KL) | ρ (token-eviction KL, for reference: fut_attn_at_t) |", "|---|---|---|---|---|---|---|"]
    uni = []
    for (m, ds, sp), g in df.groupby(["model", "dataset", "split"]):
        for fam, feats in PLN_FEATS.items():
            for f in feats:
                r1 = [_spearman(gg[f], gg.last_kl_mean) for _, gg in g.groupby("uid")]
                r2 = [_spearman(gg[f], gg.first_kl_mean) for _, gg in g.groupby("uid")] if "first_kl_mean" in g.columns else [np.nan]
                uni.append({"model": m, "dataset": ds, "split": sp, "family": fam, "feature": f, "rho_last": float(np.nanmean(r1)),
                            "ci_last": _boot(r1), "rho_first": float(np.nanmean(r2))})
                md.append(f"| {m} | {ds} | {sp} | {f} ({fam}) | {_f(np.nanmean(r1))}{_ci(_boot(r1))} | {_f(np.nanmean(r2))} | |")
    md.append("")
    # LOPO ridge: feature families for plane damage
    md += ["### Leave-one-UID-out ridge ranking of per-token plane damage (log10 last-only KL): feature families", "",
           "| model | dataset | split | features | mean ρ [CI] | LIFT over position/lexical [CI] |", "|---|---|---|---|---|---|"]
    lopo = []
    for (m, ds, sp), g in df.groupby(["model", "dataset", "split"]):
        if g.uid.nunique() < 3:
            continue
        g = g.assign(y=np.log10(g.last_kl_mean.clip(lower=1e-9)))
        base = None
        sets = [("position/lexical", PLN_FEATS["position/lexical"]),
                ("+ attention received", PLN_FEATS["position/lexical"] + PLN_FEATS["attention (received, loop-R)"]),
                ("+ own trajectory", PLN_FEATS["position/lexical"] + PLN_FEATS["own trajectory (settle/U)"]),
                ("+ attention + trajectory", PLN_FEATS["position/lexical"] + PLN_FEATS["attention (received, loop-R)"] + PLN_FEATS["own trajectory (settle/U)"]),
                ("+ attention + trajectory + future loop profile (oracle)", sum(PLN_FEATS.values(), []))]
        for name, feats in sets:
            rhos = []
            for uid in g.uid.unique():
                tr, te = g[g.uid != uid], g[g.uid == uid]
                X = np.log10(np.abs(tr[feats].to_numpy(float)) + 1e-9)
                mu, sd = X.mean(0), X.std(0) + 1e-12
                Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
                w = np.linalg.solve(Xs.T @ Xs + np.eye(Xs.shape[1]), Xs.T @ tr.y.to_numpy())
                Xt = np.log10(np.abs(te[feats].to_numpy(float)) + 1e-9)
                rhos.append(_spearman(np.c_[np.ones(len(te)), (Xt - mu) / sd] @ w, te.y.to_numpy()))
            if base is None:
                base = rhos
            lift = [b - a for a, b in zip(base, rhos)]
            lopo.append({"model": m, "dataset": ds, "split": sp, "features": name, "mean_rho": float(np.nanmean(rhos)), "ci": _boot(rhos),
                         "lift": float(np.nanmean(lift)), "lift_ci": _boot(lift)})
            md.append(f"| {m} | {ds} | {sp} | {name} | {_f(np.nanmean(rhos))}{_ci(_boot(rhos))} | {_f(np.nanmean(lift))}{_ci(_boot(lift))} |")
    md.append("")
    _json(out / "pln_summary.json", {"census": res, "univariate": uni, "lopo": lopo})
    return res


def kv_similarity(run: Path, out: Path, md: list):
    """How redundant are the loop planes? Mean cosine of K and V at plane r to the terminal plane, per model (pooled over
    Context tokens and prompts), by loop and by recurrent layer; plus the fraction of tokens whose K at plane r is
    within cos>0.99 of the terminal plane."""
    jobs = json.loads((run / "jobs.json").read_text())
    acc = {}
    for j in (x for x in jobs if x["kind"] == "REF"):
        f = run / "raw" / f"{j['key']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if "k_cos_to_last" not in z.files:
            continue
        content = json.loads((run / "raw" / f"{j['key']}.json").read_text())["content_positions"]
        for key in ("k_cos_to_last", "v_cos_to_last", "k_cos_to_prev", "v_cos_to_prev"):
            a = z[key][:, :, content]                       # (n_rec, R, |C|)
            acc.setdefault((j["model"], key), []).append(a)
    if not acc:
        return
    md += ["## Loop-plane redundancy: cosine of a Context token's K/V at plane r to its terminal plane R (pooled; head-mean)", "",
           "| model | quantity | " + " | ".join(f"r={r}" for r in (1, 2, 3, 4, 8, 16, 24, 28, 31)) + " | frac tokens cos(K_r,K_R)>0.99 at r=R/2 | per-layer at r=R/2 |",
           "|---|---|" + "---|" * 11]
    res = []
    for (m, key), arrs in sorted(acc.items()):
        A = np.concatenate(arrs, axis=2)                    # (n_rec, R, N)
        R = A.shape[1]
        by_loop = A.mean(axis=(0, 2))
        half = R // 2 - 1
        frac = float((A[:, half].mean(0) > 0.99).mean()) if "to_last" in key else np.nan
        per_layer = A[:, half].mean(1).tolist()
        res.append({"model": m, "quantity": key, "by_loop": by_loop.tolist(), "frac_gt_099_half": frac, "per_layer_half": per_layer})
        md.append(f"| {m} | {key} | " + " | ".join(_f(by_loop[r - 1], 3) if r <= R else "—" for r in (1, 2, 3, 4, 8, 16, 24, 28, 31))
                  + f" | {_f(frac)} | {', '.join(_f(x, 3) for x in per_layer)} |")
    md.append("")
    _json(out / "kv_plane_similarity.json", res)


def gsm_summary(run: Path, out: Path, md: list):
    """Reasoning arm: paired greedy GSM8K accuracy under plane policies (prompt-only vs prompt+generated)."""
    jobs = json.loads((run / "jobs.json").read_text())
    rows = []
    for j in (x for x in jobs if x["kind"] == "GSM"):
        f = run / "raw" / f"{j['key']}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        full = next(r for r in d["rows"] if r["arm"] == "full")
        for r in d["rows"]:
            rows.append({"model": j["model"], "uid": j["uid"], "arm": r["arm"], "correct": float(r["correct"]), "flex": float(r["flex"]),
                         "full_correct": float(full["correct"]), "n_tokens": r["n_tokens"], "finish": r["finish"],
                         "identical": float(bool(r.get("identical_to_full", r["arm"] == "full"))),
                         "first_div": r.get("first_divergence", np.nan), "R": d["R"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out / "gsm_long.csv", index=False)
    md += ["## Reasoning arm: zero-shot GSM8K greedy accuracy under plane policies (paired with the model's FULL generation)", "",
           "lastK@prompt = prompt tokens keep their last K planes; @prompt+generated = every generated token is also converted to its "
           "last K planes after it finishes (what a real cache would hold); first1 = first plane only (negative control); "
           "token_random50 = half of the question tokens evicted (full planes). Δacc vs FULL [95% problem bootstrap]; "
           "'identical' = greedy token sequence identical to FULL.", "",
           "| model | arm | n | acc | FULL acc | Δacc [CI] | flexible-extract acc | identical generations | median first divergence (tokens) | mean length |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for (m, arm), g in df.groupby(["model", "arm"]):
        d = (g.correct - g.full_correct).tolist()
        r = {"model": m, "arm": arm, "n": len(g), "acc": float(g.correct.mean()), "full_acc": float(g.full_correct.mean()),
             "delta": float(np.mean(d)), "ci": _boot(d), "flex": float(g.flex.mean()), "identical": float(g.identical.mean()),
             "first_div_median": float(np.nanmedian(g.first_div)) if g.first_div.notna().any() else np.nan, "len": float(g.n_tokens.mean())}
        res.append(r)
        md.append(f"| {m} | {arm} | {len(g)} | {_f(r['acc'], 2)} | {_f(r['full_acc'], 2)} | {_f(r['delta'], 2)}{_ci(r['ci'])} | {_f(r['flex'], 2)} | "
                  f"{_f(r['identical'], 2)} | {_f(r['first_div_median'])} | {_f(r['len'])} |")
    md.append("")
    _json(out / "gsm_summary.json", res)
    return res


def settle_hist(run: Path, md: list):
    jobs = json.loads((run / "jobs.json").read_text())
    acc = defaultdict(lambda: None)
    for j in (x for x in jobs if x["kind"] == "REF"):
        f = run / "raw" / f"{j['key']}.json"
        if not f.exists():
            continue
        h = np.array(json.loads(f.read_text())["settle_hist"])
        acc[j["model"]] = h if acc[j["model"]] is None else acc[j["model"]] + h
    md += ["## Settle depth of Context tokens (first plane within 10% of the terminal state; pooled over prompts)", ""]
    for m, h in acc.items():
        frac = h / h.sum()
        cum = np.cumsum(frac)
        md.append(f"- {m}: mean settle plane {float((np.arange(len(h)) * frac).sum()):.2f} of {len(h) - 1}; "
                  f"share settled by plane {len(h) // 4 - 1}: {cum[max(0, len(h) // 4 - 1)]:.2f}, by plane {len(h) // 2 - 1}: {cum[len(h) // 2 - 1]:.2f}; "
                  f"histogram {h.tolist()}")
    md.append("")


def analyze(run: Path) -> dict:
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    comp = completeness(run)
    md = [f"# v8 two-axis KV budget — findings ({run.name})", "",
          f"Completeness: {comp['completed']}/{comp['expected']} jobs, errors {len(comp['errors'])}, problems {len(comp['problems'])}, "
          f"complete={comp['complete']}", ""]
    settle_hist(run, md)
    kv_similarity(run, out, md)
    gsm_summary(run, out, md)
    tax = tax_table(run)
    if len(tax):
        tax.to_csv(out / "tax_long.csv", index=False)
        tax_summary(tax, out, md)
    pln = pln_table(run)
    if len(pln):
        pln.to_csv(out / "pln_long.csv", index=False)
        pln_summary(pln, out, md)
    (out / "PLANE_BUDGET_FINDINGS.md").write_text("\n".join(md))
    return {"complete": comp["complete"], "tax_rows": len(tax), "pln_rows": len(pln), "missing": len(comp["missing"]),
            "errors": comp["errors"], "problems": comp["problems"]}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]).resolve()), indent=2, default=str))
