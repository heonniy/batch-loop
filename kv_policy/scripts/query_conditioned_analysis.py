#!/usr/bin/env python3
"""v7 CPU analysis: completeness (fail closed), per-source long table, floor/matched-pair/gain/prediction/exit
summaries and QUERY_CONDITIONED_FINDINGS.md. Usage: query_conditioned_sweep.py --analyze --run-dir <run>.

Feature groups (all per (unit, Context source i); never uses future outcomes as features):
  A  deployable clean prompt-query attention/position: question-anchor attention (mean/last/first loop, max head),
     all-later-prompt-query reception (v6 recv_mean/last_query), age, relative position, lexical token features.
  B  A + source's own clean prefill trajectory (U mean, settle loop, C mean).
  C  A + value norm + closed-form local attention-output perturbation at the anchor (mean/last/first loop).
  D  C + query-conditioned intervention responses at the anchor (OFFLINE ORACLE): all-loop probe KL / E_R,
     one-loop probe E_R and gain, rec-only/plain-only (Huginn).
  F  C + FUTURE generated-query clean attention / local perturbation (ORACLE: not available at prompt time).
  S  C + cheap surrogates (loop-weighted local perturbation, attention x anchor update profile).
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TOP_FRAC = 0.10
LOOKAHEAD_K = (1, 2, 3, 4, 6, 8)


def _json(path: Path, value):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=float))
    tmp.replace(path)


def _boot(vals, n=2000, seed=0):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    d = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return [d[int(.025 * n)], d[int(.975 * n) - 1]]


def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}g}"


def _ci(ci):
    return f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""


def _spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return np.nan
    return float(a[ok].rank().corr(b[ok].rank()))


# ------------------------------------------------------------------------------------------- completeness
def completeness(run: Path) -> dict:
    jobs = json.loads((run / "jobs.json").read_text())
    raw = run / "raw"
    missing = [j["key"] for j in jobs if not (raw / f"{j['key']}.json").exists()
               or (j["kind"] != "LIVE" and not (raw / f"{j['key']}.npz").exists())]
    errs = sorted(p.stem for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    problems = []
    cov = defaultdict(set)
    for j in jobs:
        if j["kind"] in ("CEN", "V6F") and (raw / f"{j['key']}.json").exists():
            cov[j["depends"]].update(j["sources"])
    for j in jobs:
        if j["kind"] in ("REF", "V6R") and (raw / f"{j['key']}.json").exists():
            m = json.loads((raw / f"{j['key']}.json").read_text())
            if not m["validation"]["pass"]:
                problems.append(f"validation failed {j['key']}: {m['validation']['failures']}")
            want = set(m["content_positions"]) if j["kind"] == "REF" else set(
                x["sources"] for x in [] ) or {s for jj in jobs if jj.get("depends") == j["key"] for s in jj["sources"]}
            if cov[j["key"]] != want:
                problems.append(f"source coverage incomplete for {j['key']}: {len(cov[j['key']])}/{len(want)}")
    out = {"expected": len(jobs), "completed": len(jobs) - len(missing), "missing": missing, "errors": errs,
           "problems": problems, "complete": not missing and not errs and not problems}
    _json(run / "completeness.json", out)
    return out


# ------------------------------------------------------------------------------------------- long table
def _lexical(tok, tid: int) -> dict:
    s = tok.convert_ids_to_tokens(int(tid))
    txt = tok.decode([int(tid)])
    core = txt.strip()
    return {"tok_chars": len(core), "tok_word_start": float(txt[:1].isspace() or s[:1] in ("Ġ", "▁")),
            "tok_punct": float(bool(core) and all(not c.isalnum() for c in core)),
            "tok_digit": float(any(c.isdigit() for c in core)), "tok_upper": float(core[:1].isupper()),
            "tok_str": txt}


def _settle(F, thr=0.10):
    idx = np.where(F < thr)[0]
    return int(idx[0]) if len(idx) else len(F) - 1


def _tokenizers(models):
    from transformers import AutoTokenizer
    from kvp.specs import MODELS
    return {m: AutoTokenizer.from_pretrained(MODELS[m].repo, revision=MODELS[m].revision, trust_remote_code=True)
            for m in models}


def would_exit(ex: np.ndarray, keys: list, family: str, thr: float, min_depth: int) -> int:
    """Offline model-native exit depth from recorded exit statistics (Ouro cumulative mass; Huginn fp32 distance)."""
    R = ex.shape[-1]
    if family == "ouro":
        cum = ex[keys.index("cumulative")]
        for r in range(1, R + 1):
            if r >= min_depth and (r == R or (thr < 1 and cum[r - 1] >= thr)):
                return r
        return R
    d = ex[keys.index("distance_fp32")]
    for r in range(1, R + 1):
        if r >= min_depth and (d[r - 1] < thr or r == R):
            return r
    return R


def source_table(run: Path) -> pd.DataFrame:
    jobs = json.loads((run / "jobs.json").read_text())
    cfg = json.loads((run / "run_manifest.json").read_text())["config"]
    raw = run / "raw"
    toks = _tokenizers(cfg["models"])
    v6_cache = {}
    rows = []
    for ref in (j for j in jobs if j["kind"] in ("REF", "V6R")):
        if not (raw / f"{ref['key']}.npz").exists():
            continue
        m = json.loads((raw / f"{ref['key']}.json").read_text())
        z = np.load(raw / f"{ref['key']}.npz")
        fam = "ouro" if ref["model"].startswith("ouro") else "huginn"
        R = z["prompt_U"].shape[1] - 1
        P = len(m["prompt_ids"])
        n = m["ref"]["n"]
        tok = toks[ref["model"]]
        roles = m["roles"]
        content = m["content_positions"]
        qa, qb = m["anchors"]["question_last"], m["anchors"]["prompt_last"]
        keys = m["exit_keys"]
        A = {k: z[f"anchor_question_last_{k}"] for k in ("attn", "attn_max", "delta", "delta_max", "vnorm")}
        B = {k: z[f"anchor_prompt_last_{k}"] for k in ("attn", "attn_max", "delta", "delta_max", "vnorm")}
        has_plain = z["anchor_question_last_plain_attn"].sum() > 0
        gen_attn, gen_delta = z["gen_attn"], z["gen_delta"]            # (n, R, P)
        Uq = z["prompt_U"][qa]                                          # anchor's own clean update profile (R+1)
        loopw = np.arange(R, 0, -1, dtype=float) / R                    # early loops weighted more
        # v6 outcome lookup
        if ref["kind"] == "V6R":
            from query_conditioned_sweep import V6_RUN
            if "jobs" not in v6_cache:
                v6_cache["jobs"] = json.loads((V6_RUN / "jobs.json").read_text())
            v6y = {}
            for c in (j for j in v6_cache["jobs"] if j.get("depends") == ref["v6_key"]):
                zz = np.load(V6_RUN / "raw" / f"{c['key']}.npz")
                for si, s in enumerate(zz["sources"]):
                    v6y[int(s)] = {"kl_mean": float(np.nanmean(zz["acc_kl"][si])),
                                   "kl_max": float(np.nanmax(zz["acc_kl"][si])),
                                   "nll_shift_mean": float(np.nanmean(zz["acc_nll_shift"][si])),
                                   "E_R_mean": float(np.nanmean(zz["acc_E"][si][:, R])),
                                   "kl_early8": float(np.nanmean(zz["acc_kl"][si][:8]))}
        chunks = [j for j in jobs if j.get("depends") == ref["key"]]
        for c in chunks:
            if not (raw / f"{c['key']}.npz").exists():
                continue
            zc = np.load(raw / f"{c['key']}.npz")
            cm = json.loads((raw / f"{c['key']}.json").read_text())
            probes, anchors = cm["probes"], cm["anchors"]
            for si, s in enumerate(zc["sources"]):
                s = int(s)
                if s >= qa:
                    continue   # v6 units list Instruction/Question tokens too; anchor features are undefined there
                lex = _lexical(tok, m["prompt_ids"][s])
                row = {"model": ref["model"], "dataset": ref["dataset"], "uid": ref["uid"], "unit": ref["unit"],
                       "source": s, "token_id": int(m["prompt_ids"][s]), "role": roles[s], "P": P, "n_targets": n,
                       "age": P - s, "rel_pos": (s - content[0]) / max(1, content[-1] - content[0]),
                       "is_evidence": float(roles[s] == "relevant_content"), **lex}
                # A: deployable prompt-query attention
                for pre, X in (("qa", A), ("qb", B)):
                    a = X["attn"][:, s]
                    row |= {f"{pre}_attn_mean": float(a.mean()), f"{pre}_attn_last": float(a[-1]),
                            f"{pre}_attn_first": float(a[0]), f"{pre}_attn_maxloop": float(a.max()),
                            f"{pre}_attn_maxhead": float(X["attn_max"][:, s].max()),
                            f"{pre}_attn_slope": float(a[-1] - a[0]),
                            f"{pre}_delta_mean": float(X["delta"][:, s].mean()),
                            f"{pre}_delta_last": float(X["delta"][-1, s]), f"{pre}_delta_first": float(X["delta"][0, s]),
                            f"{pre}_delta_maxhead": float(X["delta_max"][:, s].max()),
                            f"{pre}_delta_loopw": float((X["delta"][:, s] * loopw).sum() / loopw.sum()),
                            f"{pre}_attn_x_U": float((a * np.nan_to_num(Uq[1:])).sum()),
                            f"{pre}_vnorm_mean": float(X["vnorm"][:, s].mean())}
                if has_plain:
                    row |= {"qa_plain_attn": float(z["anchor_question_last_plain_attn"][:, s].mean()),
                            "qa_plain_delta": float(z["anchor_question_last_plain_delta"][:, s].mean())}
                row |= {"recv_mean": float(z["src_recv_mean"][:, s].mean()), "recv_last_loop": float(z["src_recv_mean"][-1, s]),
                        "recv_max": float(z["src_recv_max"][:, s].max()), "lastq_mean": float(z["src_last_query"][:, s].mean())}
                # B: own trajectory
                row |= {"src_U_mean": float(np.nanmean(z["prompt_U"][s][1:])), "src_U_last": float(z["prompt_U"][s][-1]),
                        "src_settle": _settle(z["prompt_F"][s]), "src_C_mean": float(np.nanmean(z["prompt_C"][s]))
                        if np.isfinite(z["prompt_C"][s]).any() else 0.0}
                # F: future oracle; L: 2-token format lookahead (the continuation always starts "Answer:" so the
                # queries at t=1,2 are known at prompt time and cost two extra B=1 steps -> deployable)
                ga, gd = gen_attn[:, :, s], gen_delta[:, :, s]
                tstar = int(ga.mean(1).argmax())
                prof = ga[tstar] / max(1e-12, ga[tstar].sum())
                row |= {"fut_attn_mean": float(ga.mean()), "fut_attn_max_t": float(ga.mean(1).max()),
                        "fut_attn_last": float(ga[:, -1].mean()), "fut_delta_mean": float(gd.mean()),
                        "fut_delta_max_t": float(gd.mean(1).max()), "fut_delta_last": float(gd[:, -1].mean()),
                        "fut_lastloop_frac": float(prof[-1]), "fut_firstloop_frac": float(prof[0]),
                        "fut_loop_entropy": float(-(prof * np.log(prof + 1e-12)).sum()),
                        "la1_attn": float(ga[0].mean()), "la2_attn": float(ga[min(1, n - 1)].mean()),
                        "la12_attn_max": float(ga[:2].mean(1).max()), "la1_delta": float(gd[0].mean()),
                        "la2_delta": float(gd[min(1, n - 1)].mean()), "la12_delta_max": float(gd[:2].mean(1).max()),
                        "qa_lastloop_frac": float(A["attn"][-1, s] / max(1e-12, A["attn"][:, s].sum()))}
                for k in LOOKAHEAD_K:
                    kk = min(k, n)
                    row[f"lk{k}_attn_max"] = float(ga[:kk].mean(1).max())
                    row[f"lk{k}_delta_max"] = float(gd[:kk].mean(1).max())
                    row[f"lk{k}_attn_lastloop_max"] = float(ga[:kk, -1].max())
                row |= {f"fut_prof_r{r + 1}": float(prof[r]) for r in range(R)}
                row |= {f"fut_attn_r{r + 1}": float(ga[tstar, r]) for r in range(R)}
                row |= {f"la12_attn_r{r + 1}": float(ga[:2, r].max()) for r in range(R)}
                row |= {f"qa_attn_r{r + 1}": float(A["attn"][r, s]) for r in range(R)}
                qprof = A["attn"][:, s] / max(1e-12, A["attn"][:, s].sum())
                row |= {f"qa_prof_r{r + 1}": float(qprof[r]) for r in range(R)}
                # Y
                if ref["kind"] == "REF":
                    kl = zc["acc_kl"][si]
                    nll = zc["acc_nll_shift"][si]
                    row |= {"kl_mean": float(np.nanmean(kl)), "kl_max": float(np.nanmax(kl)),
                            "kl_first": float(kl[0]), "nll_shift_mean": float(np.nanmean(nll)),
                            "nll_shift_max": float(np.nanmax(nll)), "top1_any": float(np.nanmax(zc["acc_top1_changed"][si])),
                            "E_R_mean": float(np.nanmean(zc["acc_E"][si][:, R])), "E_R_last": float(zc["acc_E"][si][-1, R]),
                            "clean_kl_mean": float(np.nanmean(zc["clean_kl"][si])),
                            "hist_kl_gap": float(np.nanmean(kl) - np.nanmean(zc["clean_kl"][si]))}
                else:
                    y = v6y.get(s)
                    if y is None:
                        continue
                    row |= y
                # D: anchor probes
                amap = {"question_last": ("qa", qa), "prompt_last": ("qb", qb), "future_maxattn": ("qf", None)}
                for ai, an in enumerate(anchors):
                    pre, col = amap[an]
                    if col is None:
                        col = P + int(zc["future_t"][si])
                        row["fut_t"] = col - P
                        row["fut_attn_at_t"] = float(gen_attn[col - P, :, s].mean())
                        row["fut_delta_at_t"] = float(gen_delta[col - P, :, s].mean())
                    if s >= col:
                        continue
                    for qi, pn in enumerate(probes):
                        E = zc["q_E"][si, ai, qi]
                        if not np.isfinite(E[1:]).any():
                            continue
                        row[f"{pre}_{pn}_E_R"] = float(E[R])
                        row[f"{pre}_{pn}_kl"] = float(zc["q_kl"][si, ai, qi])
                        row[f"{pre}_{pn}_top1"] = float(zc["q_top1_changed"][si, ai, qi])
                        row[f"{pre}_{pn}_dn_R"] = float(zc["q_delta_norm"][si, ai, qi][R])
                        if pn.startswith("loop"):
                            sl = int(pn[4:])
                            row[f"{pre}_{pn}_E_s"] = float(E[sl])
                            row[f"{pre}_{pn}_gain"] = float(E[R] / (E[sl] + 1e-12))
                        row[f"{pre}_{pn}_dircos_mean"] = float(np.nanmean(zc["q_dircos"][si, ai, qi][1:]))
                        if pn == "all":
                            ex = zc["q_exit"][si, ai, qi]
                            row[f"{pre}_all_exit_cum3" if fam == "ouro" else f"{pre}_all_exit_dist_last"] = (
                                float(ex[keys.index("cumulative")][2]) if fam == "ouro"
                                else float(ex[keys.index("distance_fp32")][-1]))
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- feature sets
A_FEATS = ["qa_attn_mean", "qa_attn_last", "qa_attn_first", "qa_attn_maxhead", "qa_attn_slope", "recv_mean",
           "recv_last_loop", "recv_max", "lastq_mean", "age", "rel_pos", "tok_chars", "tok_word_start",
           "tok_punct", "tok_digit", "tok_upper"]
B_FEATS = ["src_U_mean", "src_U_last", "src_settle", "src_C_mean"]
C_FEATS = ["qa_delta_mean", "qa_delta_last", "qa_delta_first", "qa_delta_maxhead", "qa_vnorm_mean"]
S_FEATS = ["qa_delta_loopw", "qa_attn_x_U"]
L_FEATS = ["la1_attn", "la2_attn", "la12_attn_max", "la1_delta", "la2_delta", "la12_delta_max"]
F_FEATS = ["fut_attn_mean", "fut_attn_max_t", "fut_attn_last", "fut_delta_mean", "fut_delta_max_t", "fut_delta_last"]


def d_feats(df: pd.DataFrame, pre: str = "qa") -> list:
    return [c for c in df.columns if c.startswith(pre + "_") and (c.endswith("_E_R") or c.endswith("_kl") or
                                                                  c.endswith("_gain") or c.endswith("_dn_R"))
            and not c.startswith(pre + "_attn") and not c.startswith(pre + "_delta")]


def _ridge_lopo(g: pd.DataFrame, feats: list, y: str, lam=1.0):
    feats = [f for f in feats if f in g.columns]
    rhos, aurocs = [], []
    top = g.groupby("uid")[y].transform(lambda s: s >= s.quantile(1 - TOP_FRAC))
    for uid in g.uid.unique():
        tr, te = g[g.uid != uid], g[g.uid == uid]
        X = np.log10(np.abs(tr[feats].to_numpy(float)) + 1e-9) if False else tr[feats].to_numpy(float)
        X = np.nan_to_num(X)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ tr[y].to_numpy())
        Xt = np.nan_to_num(te[feats].to_numpy(float))
        pred = np.c_[np.ones(len(te)), (Xt - mu) / sd] @ w
        rhos.append(_spearman(pred, te[y].to_numpy()))
        lab = top[te.index].to_numpy()
        if 0 < lab.sum() < len(lab):
            r = pd.Series(pred).rank().to_numpy()
            aurocs.append(float((r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())))
        else:
            aurocs.append(np.nan)
    return rhos, aurocs


def _logfeat(df: pd.DataFrame) -> pd.DataFrame:
    """Rank-robust transform: log10 of magnitude features (attention, delta, KL, E) so ridge is not dominated by
    heavy tails; position/lexical features untouched."""
    out = df.copy()
    for c in df.columns:
        if any(c.startswith(p) for p in ("qa_", "qb_", "recv", "lastq", "fut_", "src_U")) and df[c].dtype != object:
            v = df[c].to_numpy(float)
            out[c] = np.log10(np.abs(v) + 1e-9)
    return out


def prediction(st: pd.DataFrame, out: Path, md: list) -> list:
    st = st[st.kl_mean.notna()].copy()
    st["y"] = np.log10(st.kl_mean.clip(lower=1e-9))
    st["y_nll"] = st.nll_shift_mean
    L = _logfeat(st)
    D = d_feats(st)
    G = d_feats(st, "qf")
    G1 = [c for c in G if "_loop" in c and not c.endswith("_kl")]          # one-loop probes: E_s, E_R, gain only
    G1g = [c for c in G if c.endswith("_gain")]                             # scale-free propagation gains only
    G2 = [c for c in G if c.startswith("qf_all") or c.endswith("_kl")]     # all-loop probe = within-query damage at t*
    FF = F_FEATS + [c for c in ("fut_attn_at_t", "fut_delta_at_t") if c in st.columns]
    sets = [("A attention/position/lexical", A_FEATS), ("B = A + source trajectory", A_FEATS + B_FEATS),
            ("C = A + value/local-perturbation", A_FEATS + C_FEATS), ("S = C + cheap surrogates", A_FEATS + C_FEATS + S_FEATS),
            ("L = C + 2-token format lookahead (deployable, +2 steps)", A_FEATS + C_FEATS + L_FEATS),
            ("D = C + prompt-anchor query probes (oracle)", A_FEATS + C_FEATS + D),
            ("F = C + future-query census (oracle)", A_FEATS + C_FEATS + FF),
            ("G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle)", A_FEATS + C_FEATS + FF + G1),
            ("G1g = F + propagation gains only at t* (oracle)", A_FEATS + C_FEATS + FF + G1g),
            ("G2 = F + all-loop probe at t* (≈ within-query damage component)", A_FEATS + C_FEATS + FF + G2),
            ("G = F + all future-query probes (oracle)", A_FEATS + C_FEATS + FF + G),
            ("D+F", A_FEATS + C_FEATS + D + FF)]
    res = []
    for (m, ds), g in L.groupby(["model", "dataset"]):
        if g.uid.nunique() < 3:
            continue
        per = {}
        for name, feats in sets:
            rhos, au = _ridge_lopo(g, feats, "y")
            per[name] = rhos
            res.append({"model": m, "dataset": ds, "features": name, "n_feats": len([f for f in feats if f in g.columns]),
                        "mean_spearman": float(np.nanmean(rhos)), "ci": _boot(rhos),
                        "mean_auroc_top10": float(np.nanmean(au)), "auroc_ci": _boot(au), "n_prompts": len(rhos)})
        for a, b in (("A attention/position/lexical", "B = A + source trajectory"),
                     ("A attention/position/lexical", "C = A + value/local-perturbation"),
                     ("C = A + value/local-perturbation", "S = C + cheap surrogates"),
                     ("C = A + value/local-perturbation", "L = C + 2-token format lookahead (deployable, +2 steps)"),
                     ("L = C + 2-token format lookahead (deployable, +2 steps)", "F = C + future-query census (oracle)"),
                     ("C = A + value/local-perturbation", "D = C + prompt-anchor query probes (oracle)"),
                     ("C = A + value/local-perturbation", "F = C + future-query census (oracle)"),
                     ("F = C + future-query census (oracle)", "G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle)"),
                     ("F = C + future-query census (oracle)", "G1g = F + propagation gains only at t* (oracle)"),
                     ("F = C + future-query census (oracle)", "G2 = F + all-loop probe at t* (≈ within-query damage component)"),
                     ("F = C + future-query census (oracle)", "D+F")):
            lift = [y - x for x, y in zip(per[a], per[b]) if np.isfinite(x) and np.isfinite(y)]
            res.append({"model": m, "dataset": ds, "features": f"LIFT {b.split(' ')[0]} − {a.split(' ')[0]}",
                        "mean_spearman": float(np.mean(lift)) if lift else np.nan, "ci": _boot(lift), "n_prompts": len(lift)})
    _json(out / "prediction_lopo.json", res)
    md += ["## H3 — leave-one-question-UID-out prediction of future eviction damage (log10 mean KL over the gold continuation)",
           "", "Ridge on log-magnitude features; Spearman within each held-out prompt, mean [95% prompt bootstrap]. "
           "D and F are offline oracles (query recomputation / future-query attention); S is the cheap one-pass surrogate.", "",
           "| model | dataset | features | mean ρ [CI] | AUROC top-10% harm |", "|---|---|---|---|---|"]
    for r in res:
        md.append(f"| {r['model']} | {r['dataset']} | {r['features']} | {_f(r['mean_spearman'])}{_ci(r['ci'])} | "
                  f"{_f(r.get('mean_auroc_top10'))}{_ci(r.get('auroc_ci'))} |")
    md.append("")
    return res


# ------------------------------------------------------------------------------------------- univariate
def univariate(st: pd.DataFrame, out: Path, md: list):
    st = st[st.kl_mean.notna()]
    feats = ["qa_attn_mean", "qa_attn_last", "qa_attn_maxhead", "recv_mean", "lastq_mean", "qb_attn_mean",
             "qa_delta_mean", "qa_delta_last", "qa_delta_loopw", "qa_vnorm_mean", "src_U_mean", "src_settle",
             "fut_attn_mean", "fut_attn_max_t", "fut_delta_mean", "fut_delta_max_t", "fut_attn_at_t", "fut_delta_at_t",
             "qa_all_kl", "qa_all_E_R", "qa_all_dn_R", "qb_all_kl", "qb_all_E_R", "qf_all_kl", "qf_all_E_R", "qf_all_dn_R",
             "qf_loop1_gain", "qa_loop1_gain", "clean_kl_mean", "la1_attn", "la2_attn", "la12_attn_max", "la12_delta_max",
             "fut_lastloop_frac", "fut_loop_entropy", "qa_lastloop_frac"]
    res = []
    for (m, ds), g in st.groupby(["model", "dataset"]):
        for f in feats:
            if f not in g.columns:
                continue
            rhos = [_spearman(gg[f], gg.kl_mean) for _, gg in g.groupby("uid")]
            rhos_n = [_spearman(gg[f], gg.nll_shift_mean) for _, gg in g.groupby("uid")]
            res.append({"model": m, "dataset": ds, "feature": f, "rho_kl": float(np.nanmean(rhos)), "ci_kl": _boot(rhos),
                        "rho_nll": float(np.nanmean(rhos_n)), "ci_nll": _boot(rhos_n)})
    pd.DataFrame(res).to_csv(out / "univariate_within_prompt.csv", index=False)
    md += ["## Within-prompt Spearman of single features with future damage (mean over prompts [95% CI])", "",
           "| model | dataset | feature | ρ(mean KL) | ρ(mean NLL shift) |", "|---|---|---|---|---|"]
    for r in res:
        md.append(f"| {r['model']} | {r['dataset']} | {r['feature']} | {_f(r['rho_kl'])}{_ci(r['ci_kl'])} | "
                  f"{_f(r['rho_nll'])}{_ci(r['ci_nll'])} |")
    md.append("")
    return res


# ------------------------------------------------------------------------------------------- floor & gain
def floor_and_gain(st: pd.DataFrame, out: Path, md: list):
    res = {}
    md += ["## Query-conditioned probe magnitudes vs the numerical floor", "",
           "Sources binned by clean question-anchor attention (within prompt deciles). The lowest decile serves as the "
           "empirical 'near-zero perturbation' floor: in BF16 any mask changes rounding, so hidden-state E has a floor "
           "even when the removed attention mass is ~0; the question is whether higher-attention sources rise above it.", "",
           "| model | dataset | attention decile | n | median qa all-loop KL | median E_R | median ‖Δh_R‖ | median top-1 flip |",
           "|---|---|---|---|---|---|---|---|"]
    for (m, ds), g in st.groupby(["model", "dataset"]):
        g = g[g.qa_all_kl.notna()].copy()
        if g.empty:
            continue
        g["dec"] = g.groupby("uid").qa_attn_mean.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
        tab = []
        for d, gg in g.groupby("dec"):
            r = {"decile": int(d), "n": len(gg), "kl_med": float(gg.qa_all_kl.median()), "E_R_med": float(gg.qa_all_E_R.median()),
                 "dn_med": float(gg.qa_all_dn_R.median()), "top1": float(gg.qa_all_top1.mean())}
            tab.append(r)
            md.append(f"| {m} | {ds} | {d} | {len(gg)} | {_f(r['kl_med'])} | {_f(r['E_R_med'])} | {_f(r['dn_med'])} | {_f(r['top1'])} |")
        res[f"{m}/{ds}"] = tab
    md.append("")
    # one-loop gain by masked loop
    md += ["## One-loop query masks: propagation gain E_R / E_s by masked loop s (median over sources; IQR)", "",
           "| model | dataset | anchor | s | n | median E_s | median E_R | median gain | IQR gain | median KL | p90 KL |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    gain = []
    for (m, ds), g in st.groupby(["model", "dataset"]):
        for pre, label in (("qa", "question anchor"), ("qf", "future consuming query t*")):
            for c in sorted((c for c in g.columns if c.startswith(pre + "_loop") and c.endswith("_gain")), key=lambda c: int(c[7:-5])):
                s = int(c[7:-5])
                gg = g[g[c].notna()]
                if gg.empty:
                    continue
                q = gg[c].quantile([.25, .5, .75])
                r = {"model": m, "dataset": ds, "anchor": label, "s": s, "n": len(gg), "E_s_med": float(gg[f"{pre}_loop{s}_E_s"].median()),
                     "E_R_med": float(gg[f"{pre}_loop{s}_E_R"].median()), "gain_med": float(q[.5]), "gain_iqr": [float(q[.25]), float(q[.75])],
                     "kl_med": float(gg[f"{pre}_loop{s}_kl"].median()), "kl_p90": float(gg[f"{pre}_loop{s}_kl"].quantile(.9))}
                gain.append(r)
                md.append(f"| {m} | {ds} | {label} | {s} | {len(gg)} | {_f(r['E_s_med'])} | {_f(r['E_R_med'])} | {_f(r['gain_med'])} | "
                          f"[{_f(r['gain_iqr'][0])},{_f(r['gain_iqr'][1])}] | {_f(r['kl_med'])} | {_f(r['kl_p90'])} |")
    md.append("")
    _json(out / "floor_and_gain.json", {"floor": res, "gain": gain})
    return res, gain


# ------------------------------------------------------------------------------------------- matched pairs
def matched_pairs(st: pd.DataFrame, out: Path, md: list, seed=20261102):
    """Within-prompt same-role pairs matched on clean question-anchor attention (|log ratio| ≤ 0.35, |Δpos| ≤ 16);
    then additionally matched on local perturbation δ (|log ratio| ≤ 0.35). Dispersion of future damage within pairs
    and whether the query-conditioned probe predicts the within-pair ordering."""
    rng = random.Random(seed)
    rows = []
    md += ["## H1/H2 — attention-matched and local-perturbation-matched source pairs", "",
           "Pairs are same-role, |Δpos|≤16, within prompt. 'attention' = clean question-anchor attention (deployable); "
           "'future' = the consuming generated query's clean attention (oracle). δ = closed-form local attention-output "
           "perturbation at the same query. If Y dispersion stays large under future-attn+δ matching, the damage is not "
           "a local/attention quantity at the consuming query either (propagation / recurrence or later history). Pairs where both "
           "members have mean KL < 1e-3 (BF16 floor) are excluded as uninformative.", "",
           "| model | dataset | pairing | n pairs | median |log10 Y ratio| | frac >10× | probe-KL (anchor of the pairing) sign agreement with Y [CI] | "
           "future-attn sign agreement [CI] |", "|---|---|---|---|---|---|---|---|"]
    for (m, ds), g in st.groupby(["model", "dataset"]):
        g = g[g.kl_mean.notna() & g.qa_all_kl.notna()]
        if "fut_attn_at_t" not in g.columns:
            g = g.assign(fut_attn_at_t=g.fut_attn_max_t, fut_delta_at_t=g.fut_delta_max_t)
        for pairing in ("attention-matched", "attention+δ-matched", "future-attn-matched", "future-attn+δ-matched",
                        "random same-role"):
            disp, agree_probe, agree_fut, npairs = [], [], [], 0
            for uid, gg in g.groupby("uid"):
                gg = gg.sort_values("source")
                pos = gg.source.to_numpy()
                fut = pairing.startswith("future")
                la = np.log((gg.fut_attn_at_t if fut else gg.qa_attn_mean).to_numpy() + 1e-9)
                ld = np.log((gg.fut_delta_at_t if fut else gg.qa_delta_mean).to_numpy() + 1e-9)
                y = gg.kl_mean.to_numpy()
                pk = (gg.qf_all_kl if fut and "qf_all_kl" in gg.columns else gg.qa_all_kl).to_numpy()
                fa = gg.fut_attn_mean.to_numpy()
                role = gg.role.to_numpy()
                used = set()
                cands = []
                for i in range(len(pos)):
                    for j in range(i + 1, len(pos)):
                        if pos[j] - pos[i] > 16:
                            break
                        if role[i] != role[j]:
                            continue
                        if pairing == "random same-role":
                            cands.append((rng.random(), i, j))
                        elif abs(la[i] - la[j]) <= 0.35 and ("δ" not in pairing or abs(ld[i] - ld[j]) <= 0.35):
                            cands.append((abs(la[i] - la[j]), i, j))
                cands.sort()
                for _, i, j in cands:
                    if i in used or j in used:
                        continue
                    if max(y[i], y[j]) < 1e-3:   # both at the BF16 floor: ordering is noise, not information
                        continue
                    used |= {i, j}
                    npairs += 1
                    disp.append(abs(math.log10((y[i] + 1e-9) / (y[j] + 1e-9))))
                    if y[i] != y[j]:
                        agree_probe.append(float((pk[i] > pk[j]) == (y[i] > y[j])))
                        agree_fut.append(float((fa[i] > fa[j]) == (y[i] > y[j])))
            r = {"model": m, "dataset": ds, "pairing": pairing, "n_pairs": npairs,
                 "median_abs_log10_ratio": float(np.median(disp)) if disp else np.nan,
                 "frac_over_10x": float(np.mean([d > 1 for d in disp])) if disp else np.nan,
                 "probe_sign_agreement": float(np.mean(agree_probe)) if agree_probe else np.nan, "probe_ci": _boot(agree_probe),
                 "future_attn_sign_agreement": float(np.mean(agree_fut)) if agree_fut else np.nan, "future_ci": _boot(agree_fut)}
            rows.append(r)
            md.append(f"| {m} | {ds} | {pairing} | {npairs} | {_f(r['median_abs_log10_ratio'])} | {_f(r['frac_over_10x'])} | "
                      f"{_f(r['probe_sign_agreement'])}{_ci(r['probe_ci'])} | {_f(r['future_attn_sign_agreement'])}{_ci(r['future_ci'])} |")
    md.append("")
    pd.DataFrame(rows).to_csv(out / "matched_pairs.csv", index=False)
    return rows


# ------------------------------------------------------------------------------------------- heavy tail / evidence
def tail_and_evidence(st: pd.DataFrame, md: list):
    md += ["## Outcome structure: heavy tail and evidence labels (post-hoc stratification only)", "",
           "| model | dataset | n sources | share of total KL in top 5% | mean KL evidence | mean KL non-evidence | "
           "evidence share among top-5% | mean ΔNLL_future evidence |", "|---|---|---|---|---|---|---|---|"]
    out = []
    for (m, ds), g in st.groupby(["model", "dataset"]):
        g = g[g.kl_mean.notna()]
        k = max(1, int(round(0.05 * len(g))))
        top = g.nlargest(k, "kl_mean")
        r = {"model": m, "dataset": ds, "n": len(g), "top5_share": float(top.kl_mean.sum() / max(1e-12, g.kl_mean.sum())),
             "kl_evidence": float(g[g.is_evidence == 1].kl_mean.mean()), "kl_other": float(g[g.is_evidence == 0].kl_mean.mean()),
             "evidence_in_top5": float(top.is_evidence.mean()), "nll_evidence": float(g[g.is_evidence == 1].nll_shift_mean.mean())}
        out.append(r)
        md.append(f"| {m} | {ds} | {len(g)} | {_f(r['top5_share'])} | {_f(r['kl_evidence'])} | {_f(r['kl_other'])} | "
                  f"{_f(r['evidence_in_top5'])} | {_f(r['nll_evidence'])} |")
    md.append("")
    return out


# ------------------------------------------------------------------------------------------- loop profiles
def loop_profiles(st: pd.DataFrame, out: Path, md: list):
    """Which recurrent loops read the harmful KV? Normalised attention-by-loop profile at the consuming query t*
    and at the question anchor, top-5% harm sources vs the rest (within prompt)."""
    md += ["## Loop profile of clean attention to the source (normalised over loops): top-5% harm vs rest", "",
           "| model | dataset | group | n | query | first loop | last loop | peak loop | entropy |", "|---|---|---|---|---|---|---|---|---|"]
    res = []
    for (m, ds), g in st.groupby(["model", "dataset"]):
        g = g[g.kl_mean.notna()].copy()
        g["top"] = g.groupby("uid").kl_mean.transform(lambda s: s >= s.quantile(0.95))
        for pre, label in (("fut", "consuming query t*"), ("qa", "question anchor")):
            cols = sorted((c for c in g.columns if c.startswith(pre + "_prof_r") and g[c].notna().any()),
                          key=lambda c: int(c.split("_r")[-1]))
            if not cols:
                continue
            for grp, gg in g.groupby("top"):
                prof = gg[cols].to_numpy(float).mean(0)
                r = {"model": m, "dataset": ds, "group": "top5" if grp else "rest", "n": len(gg), "query": label,
                     "profile": prof.tolist(), "first": float(prof[0]), "last": float(prof[-1]),
                     "peak": int(prof.argmax()) + 1, "entropy": float(-(prof * np.log(prof + 1e-12)).sum())}
                res.append(r)
                md.append(f"| {m} | {ds} | {r['group']} | {len(gg)} | {label} | {_f(r['first'])} | {_f(r['last'])} | {r['peak']} | {_f(r['entropy'])} |")
    md.append("")
    _json(out / "loop_profiles.json", res)
    return res


# ------------------------------------------------------------------------------------------- lookahead curve
def lookahead_curve(st: pd.DataFrame, out: Path, md: list):
    """Deferred-eviction diagnostic: if the eviction decision is taken after the first k generated tokens (their
    clean attention is then available at no extra cost), how much of the future-query oracle is recovered?
    LOPO ridge C + {max over t<=k of generated-query attention / local perturbation}. k=0 is C."""
    st = st[st.kl_mean.notna()].copy()
    st["y"] = np.log10(st.kl_mean.clip(lower=1e-9))
    L = _logfeat(st)
    res = []
    md += ["## Deferred-eviction lookahead curve: LOPO ranking with the first k generated queries' clean attention", "",
           "| model | dataset | k | mean ρ [CI] | AUROC top-10% | LIFT over k=0 [CI] | univariate ρ(max attn t≤k) |",
           "|---|---|---|---|---|---|---|"]
    for (m, ds), g in L.groupby(["model", "dataset"]):
        if g.uid.nunique() < 3:
            continue
        base, _ = _ridge_lopo(g, A_FEATS + C_FEATS, "y")
        res.append({"model": m, "dataset": ds, "k": 0, "mean_spearman": float(np.nanmean(base)), "ci": _boot(base)})
        md.append(f"| {m} | {ds} | 0 (C) | {_f(np.nanmean(base))}{_ci(_boot(base))} | — | — | — |")
        for k in LOOKAHEAD_K:
            feats = A_FEATS + C_FEATS + [f"lk{k}_attn_max", f"lk{k}_delta_max", f"lk{k}_attn_lastloop_max"]
            rhos, au = _ridge_lopo(g, feats, "y")
            lift = [b - a for a, b in zip(base, rhos) if np.isfinite(a) and np.isfinite(b)]
            uni = float(np.nanmean([_spearman(gg[f"lk{k}_attn_max"], gg.y) for _, gg in g.groupby("uid")]))
            r = {"model": m, "dataset": ds, "k": k, "mean_spearman": float(np.nanmean(rhos)), "ci": _boot(rhos),
                 "auroc": float(np.nanmean(au)), "lift": float(np.mean(lift)), "lift_ci": _boot(lift), "uni_rho": uni}
            res.append(r)
            md.append(f"| {m} | {ds} | {k} | {_f(r['mean_spearman'])}{_ci(r['ci'])} | {_f(r['auroc'])} | {_f(r['lift'])}{_ci(r['lift_ci'])} | {_f(uni)} |")
    md.append("")
    _json(out / "lookahead_curve.json", res)
    return res


# ------------------------------------------------------------------------------------------- loop-resolved importance
def loop_resolved(st: pd.DataFrame, out: Path, md: list):
    """Which loop's attention is the better importance signal? Within-prompt Spearman of attention at loop r with Y,
    for the consuming query t* (oracle), the 2-token lookahead (deployable) and the question anchor (deployable)."""
    md += ["## Loop-resolved attention importance: ρ(attention at loop r, Y) within prompt (mean over prompts)", "",
           "| model | dataset | query | " + " | ".join(f"r={r}" for r in (1, 2, 3, 4, 8, 16, 24, 32)) + " | mean over r |",
           "|---|---|---|" + "---|" * 9]
    res = []
    for (m, ds), g in st.groupby(["model", "dataset"]):
        g = g[g.kl_mean.notna()]
        R = 4 if m.startswith("ouro") else 32
        for pre, label, meanc in (("fut_attn", "consuming query t* (oracle)", "fut_attn_at_t"),
                                  ("la12_attn", "2-token lookahead", "la12_attn_max"), ("qa_attn", "question anchor", "qa_attn_mean")):
            rho = {}
            for r in range(1, R + 1):
                c = f"{pre}_r{r}"
                if c in g.columns:
                    rho[r] = float(np.nanmean([_spearman(gg[c], gg.kl_mean) for _, gg in g.groupby("uid")]))
            mean_r = float(np.nanmean([_spearman(gg[meanc], gg.kl_mean) for _, gg in g.groupby("uid")]))
            res.append({"model": m, "dataset": ds, "query": label, "rho_by_loop": rho, "rho_mean_over_loops": mean_r})
            md.append(f"| {m} | {ds} | {label} | " + " | ".join(_f(rho.get(r)) for r in (1, 2, 3, 4, 8, 16, 24, 32)) + f" | {_f(mean_r)} |")
    md.append("")
    _json(out / "loop_resolved_importance.json", res)
    return res


# ------------------------------------------------------------------------------------------- exit arms
def exit_arms(run: Path, st: pd.DataFrame, out: Path, md: list):
    from kvp.query_conditioned_analysis import summarize_live_replay
    jobs = json.loads((run / "jobs.json").read_text())
    raw = run / "raw"
    live = [j for j in jobs if j["kind"] == "LIVE" and (raw / f"{j['key']}.json").exists()]
    if not live:
        md += ["## Gate D — live exit arms", "", "No LIVE jobs in this run (thresholds not frozen yet).", ""]
        return None
    rows = []
    for j in live:
        d = json.loads((raw / f"{j['key']}.json").read_text())
        a = d["arms"]
        P, n = j["prompt_len"], len(a["full_live"]["depths"])
        R = 4 if j["model"].startswith("ouro") else 32
        pad = [R] * P
        # summarize_live_replay indexes depths by token position 0..P+n-2 (the n-th generated query scores nothing)
        dep = lambda arm: pad + list(arm["depths"][:n - 1])  # noqa: E731
        for s in d["per_source"]:
            try:
                sm = summarize_live_replay(prompt_len=P, target_len=n, native_max_depth=R,
                                           full_live={**a["full_live"], "depths": dep(a["full_live"])},
                                           full_replay={**a["full_replay"], "depths": dep(a["full_replay"])},
                                           full_fixed={**a["full_fixed"], "depths": dep(a["full_fixed"])},
                                           evicted_live={**s["evict_live"], "depths": dep(s["evict_live"])},
                                           evicted_replay={**s["evict_replay"], "depths": dep(s["evict_replay"])},
                                           evicted_fixed={**s["evict_fixed"], "depths": dep(s["evict_fixed"])})
            except ValueError as e:
                rows.append({"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "source": s["source"], "error": str(e)})
                continue
            rows.append({"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "source": s["source"],
                         "clean_mean_depth": float(np.mean(sm["clean_live_depths"])), "frac_changed": sm["fraction_changed"],
                         "n_up": sm["n_depth_increased"], "n_down": sm["n_depth_decreased"], "total_depth_delta": sm["total_depth_delta"],
                         "fixed_penalty": sm["fixed_future_nll_penalty"], "replay_penalty": sm["replay_future_nll_penalty"],
                         "live_penalty": sm["live_future_nll_penalty"],
                         "live_minus_replay": sm["live_minus_replay_nll_penalty_descriptive"],
                         "em_full": float(a["full_greedy"]["exact_match"]), "em_evict": float(s["evict_greedy"]["exact_match"]),
                         "in_top_attention": s["in_top_attention"],
                         "full_live_repeat_identical": a["full_live"]["depths"] == a["full_live_repeat"]["depths"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "exit_arms.csv", index=False)
    md += ["## Gate D — native dynamic exit under post-prefill single-token eviction (label-blind subset)", ""]
    if "error" in df.columns and df.error.notna().any():
        md.append(f"Parity failures: {int(df.error.notna().sum())} rows (see exit_arms.csv).")
    ok = df[df.error.isna()] if "error" in df.columns else df
    if ok.empty or "fixed_penalty" not in ok.columns:
        md.append("No parity-passing live rows.")
        return rows
    md += ["| model | dataset | n (prompt×source) | clean mean depth | frac queries depth-changed | mean Σdepth Δ | "
           "mean fixed ΔNLL | mean replay ΔNLL | mean live ΔNLL | live−replay [CI] | EM full→evict |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, ds), g in ok.groupby(["model", "dataset"]):
        md.append(f"| {m} | {ds} | {len(g)} | {_f(g.clean_mean_depth.mean())} | {_f(g.frac_changed.mean())} | "
                  f"{_f(g.total_depth_delta.mean())} | {_f(g.fixed_penalty.mean())} | {_f(g.replay_penalty.mean())} | "
                  f"{_f(g.live_penalty.mean())} | {_f(g.live_minus_replay.mean())}{_ci(_boot(g.live_minus_replay.tolist()))} | "
                  f"{_f(g.em_full.mean())}→{_f(g.em_evict.mean())} |")
    md.append("")
    # relation of depth change to damage
    md += ["| model | dataset | ρ(frac depth-changed, fixed ΔNLL) | ρ(Σdepth Δ, fixed ΔNLL) | frac Σdepth Δ>0 | frac <0 |", "|---|---|---|---|---|---|"]
    for (m, ds), g in ok.groupby(["model", "dataset"]):
        md.append(f"| {m} | {ds} | {_f(_spearman(g.frac_changed, g.fixed_penalty))} | {_f(_spearman(g.total_depth_delta, g.fixed_penalty))} | "
                  f"{_f((g.total_depth_delta > 0).mean())} | {_f((g.total_depth_delta < 0).mean())} |")
    md.append("")
    return rows


# ------------------------------------------------------------------------------------------- clean exit calibration
def clean_exit_calibration(run: Path, out: Path, md: list):
    """Clean-only exit statistics for threshold calibration (no eviction outcome enters)."""
    jobs = json.loads((run / "jobs.json").read_text())
    raw = run / "raw"
    res = {}
    for ref in (j for j in jobs if j["kind"] == "REF" and (raw / f"{j['key']}.npz").exists()):
        z = np.load(raw / f"{ref['key']}.npz")
        m = json.loads((raw / f"{ref['key']}.json").read_text())
        keys = m["exit_keys"]
        fam = "ouro" if ref["model"].startswith("ouro") else "huginn"
        for phase in ("prompt_exit", "gen_exit"):
            ex = z[phase]  # (tokens, keys, R)
            if fam == "ouro":
                cum = ex[:, keys.index("cumulative"), :]
                for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                    dep = [would_exit(e, keys, fam, q, 2) for e in ex]
                    res.setdefault(f"{ref['model']}/{phase}/q={q}", []).extend(dep)
                res.setdefault(f"{ref['model']}/{phase}/cum_r1", []).extend(cum[:, 0].tolist())
                res.setdefault(f"{ref['model']}/{phase}/cum_r2", []).extend(cum[:, 1].tolist())
                res.setdefault(f"{ref['model']}/{phase}/cum_r3", []).extend(cum[:, 2].tolist())
            else:
                for thr in (1e-2, 3e-3, 1e-3, 3e-4):
                    dep = [would_exit(e, keys, fam, thr, 2) for e in ex]
                    res.setdefault(f"{ref['model']}/{phase}/cos={thr}", []).extend(dep)
    summ = {k: {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                "frac_at_max": float(np.mean(np.array(v) == np.max(v))) if "cum" not in k else None,
                "hist": {str(int(d)): int(c) for d, c in zip(*np.unique(np.round(v), return_counts=True))} if "cum" not in k else None}
            for k, v in res.items()}
    _json(out / "clean_exit_calibration.json", summ)
    md += ["## Clean-only exit calibration (for freezing live thresholds; no eviction data used)", "",
           "| model / phase / threshold | n tokens | mean depth | frac at R | histogram |", "|---|---|---|---|---|"]
    for k, v in summ.items():
        if v["hist"] is not None:
            md.append(f"| {k} | {v['n']} | {_f(v['mean'])} | {_f(v['frac_at_max'])} | {v['hist']} |")
    md.append("")
    return summ


# ------------------------------------------------------------------------------------------- per-prompt reports
def per_prompt_reports(st: pd.DataFrame, exit_rows, out: Path):
    """ONE readable report per unit: the most harmful sources with every feature family side by side, the tiny-effect
    flag, matched pairs and the live-exit subset (if present)."""
    d = out / "per_prompt"
    d.mkdir(exist_ok=True)
    ex = pd.DataFrame(exit_rows) if exit_rows else None
    for unit, g in st.groupby("unit"):
        g = g.sort_values("kl_mean", ascending=False)
        m = [f"# {unit}", "", f"sources {len(g)}, prompt tokens {int(g.P.iloc[0])}, gold continuation {int(g.n_targets.iloc[0])} tokens; "
             f"evidence tokens {int(g.is_evidence.sum())}", "",
             "Tiny-effect flags: anchor-probe KL < 1e-4 is at the BF16 floor; E_R ≈ 0.013–0.016 for any mask.", "",
             "| rank | pos | tok | role | Y mean KL | ΔNLL | qa attn | qa δ | lookahead attn (t≤2) | future attn @t* (t*) | "
             "future δ @t* | qa probe KL | qf probe KL | qf E_R | qf loop1 gain | src U mean |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, (_, r) in enumerate(g.head(15).iterrows(), 1):
            m.append(f"| {k} | {int(r.source)} | `{str(r.tok_str).strip()}` | {r.role[:4]} | {_f(r.kl_mean)} | {_f(r.nll_shift_mean)} | "
                     f"{_f(r.qa_attn_mean, 2)} | {_f(r.qa_delta_mean, 2)} | {_f(r.get('la12_attn_max'), 2)} | "
                     f"{_f(r.get('fut_attn_at_t'), 2)} ({_f(r.get('fut_t'), 2)}) | {_f(r.get('fut_delta_at_t'), 2)} | "
                     f"{_f(r.get('qa_all_kl'), 2)} | {_f(r.get('qf_all_kl'), 2)} | {_f(r.get('qf_all_E_R'), 2)} | "
                     f"{_f(r.get('qf_loop1_gain'), 2)} | {_f(r.src_U_mean, 2)} |")
        m += ["", "Within-prompt Spearman with Y: " + ", ".join(
            f"{c}={_f(_spearman(g[c], g.kl_mean), 2)}" for c in ("qa_attn_mean", "qa_delta_mean", "la12_attn_max", "fut_attn_at_t",
                                                                 "fut_delta_at_t", "qa_all_kl", "qf_all_kl", "src_U_mean") if c in g.columns), ""]
        if ex is not None and "uid" in ex.columns:
            e = ex[(ex.uid == g.uid.iloc[0]) & (ex.model == g.model.iloc[0])]
            if len(e):
                m += ["## Live exit subset", "", "| pos | top-attn pick | clean mean depth | frac depth changed | Σdepth Δ | fixed ΔNLL | replay ΔNLL | live ΔNLL | EM full→evict |",
                      "|---|---|---|---|---|---|---|---|---|"]
                for _, r in e.sort_values("fixed_penalty", ascending=False).iterrows():
                    if "error" in e.columns and isinstance(r.get("error"), str):
                        m.append(f"| {int(r.source)} | parity error: {r.error} |")
                        continue
                    m.append(f"| {int(r.source)} | {r.in_top_attention} | {_f(r.clean_mean_depth)} | {_f(r.frac_changed)} | {int(r.total_depth_delta)} | "
                             f"{_f(r.fixed_penalty)} | {_f(r.replay_penalty)} | {_f(r.live_penalty)} | {int(r.em_full)}→{int(r.em_evict)} |")
                m.append("")
        (d / f"{unit}.md").write_text("\n".join(m))


# ------------------------------------------------------------------------------------------- main
def analyze(run: Path) -> dict:
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    comp = completeness(run)
    md = [f"# v7 query-conditioned sensitivity — findings ({run.name})", "",
          f"Completeness: {comp['completed']}/{comp['expected']} jobs, errors {len(comp['errors'])}, "
          f"problems {len(comp['problems'])}, complete={comp['complete']}", ""]
    st = source_table(run)
    st.to_parquet(out / "source_long.parquet", index=False)
    st.drop(columns=[c for c in st.columns if c.endswith("_dircos_mean")]).to_csv(out / "source_long.csv", index=False)
    md += [f"Sources in table: {len(st)} over {st.unit.nunique()} units; per model/dataset: "
           + ", ".join(f"{m}/{d}: {len(g)} sources / {g.uid.nunique()} prompts" for (m, d), g in st.groupby(['model', 'dataset'])), ""]
    main = st[st.dataset.isin(("controlled_retrieval", "hotpotqa_distractor"))]
    if not main.empty:
        tail_and_evidence(main, md)
        floor_and_gain(main, out, md)
        univariate(main, out, md)
        matched_pairs(main, out, md)
        loop_profiles(main, out, md)
        loop_resolved(main, out, md)
        prediction(main, out, md)
        lookahead_curve(main, out, md)
    v6 = st[~st.dataset.isin(("controlled_retrieval", "hotpotqa_distractor"))]
    if not v6.empty:
        md += ["# Generalisation arm: v6 W1/W2 prompts (128-token greedy continuation outcome from the frozen v6 census)", ""]
        floor_and_gain(v6, out / "v6", md) if (out / "v6").mkdir(exist_ok=True) or True else None
        univariate(v6, out / "v6", md)
        prediction(v6, out / "v6", md)
    ex = exit_arms(run, st, out, md)
    clean_exit_calibration(run, out, md)
    per_prompt_reports(st, ex, out)
    (out / "QUERY_CONDITIONED_FINDINGS.md").write_text("\n".join(md))
    return {"complete": comp["complete"], "n_sources": len(st), "units": int(st.unit.nunique()),
            "missing": len(comp["missing"]), "errors": comp["errors"], "problems": comp["problems"]}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]).resolve()), indent=2, default=str))
