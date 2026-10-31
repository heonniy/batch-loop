#!/usr/bin/env python3
"""LoopZip v6 remaining analyses (CPU): Stage A pattern labels (dev-frozen, holdout-validated),
Stage B audit artifacts, Stage C matched/role/LOPO/example-selection analyses, and the Stage C
mechanism-control / readout / Controlled-Retrieval summaries from the controls run.

Usage: loopzip_trajectory_stagec.py --pilot-run <v6 pilot> --controls-run <v6 controls>
Writes <pilot>/analysis/{stagea_*, source_metadata.csv, exhaustive_eligibility_audit.json,
first_token_and_cache_parity.json, execution_profile.json, stagec_*} and <controls>/analysis/*,
plus STAGEA_STAGEC_FINDINGS.md in the pilot analysis directory.

Dev/holdout split rule (fixed here, applied to manifest order which was frozen before outputs):
Stage A: first 20 of the 40 A prompts per workload = dev, last 20 = holdout.
Stage B/C: first 2 of the 4 B prompts per workload = dev, last 2 = holdout.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loopzip_trajectory_analysis import _boot, _json, _nanmean  # noqa: E402

F_SETTLE = 0.10      # a-priori: state within 10% (relative L2) of the terminal state
U_RISE = 0.10        # a-priori: an update >10% larger than the previous one counts as a rise


def _split(run: Path):
    man = json.loads((run / "manifest.json").read_text())
    sp = {}
    for w in ("W1", "W2"):
        for pool, k in (("A", 20), ("B", 2)):
            items = [x["item_id"] for x in man if x["workload"] == w and x["pool"] == pool]
            for i, it in enumerate(items):
                sp[it] = "dev" if i < k else "holdout"
    return sp


def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}g}"


# ------------------------------------------------------------------------------------------- Stage A
def stage_a_patterns(run: Path, out: Path, md: list):
    split = _split(run)
    jobs = [j for j in json.loads((run / "jobs.json").read_text()) if j["kind"] == "A"]
    rows = []
    for j in jobs:
        z = np.load(run / "raw" / f"{j['key']}.npz")
        meta = json.loads((run / "raw" / f"{j['key']}.json").read_text())
        for phase, pre, ids, roles in (("prompt", "prompt_", meta["prompt_ids"], meta["roles"]),
                                       ("generated", "gen_", meta["ref"]["ids"], None)):
            U, Fd = z[pre + "U"], z[pre + "F"]
            n, R1 = U.shape
            R = R1 - 1
            below = Fd[:, 1:] < F_SETTLE
            # settle depth: first r after which F stays below the threshold through R
            settle = np.full(n, R, dtype=float)
            for t in range(n):
                ok = np.where(~below[t])[0]
                settle[t] = (ok.max() + 2) if ok.size else 1
            settle = np.minimum(settle, R)
            rises = (U[:, 2:] > U[:, 1:-1] * (1 + U_RISE)).sum(1)
            for t in range(n):
                rows.append({"model": j["model"], "workload": j["workload"], "uid": j["uid"],
                             "split": split[j["item_id"]], "phase": phase, "t": t,
                             "role": roles[t] if roles else "generated", "token_id": int(ids[t]),
                             "settle_rel": settle[t] / R, "n_rises": int(rises[t]),
                             "U1": float(U[t, 1]), "UR": float(U[t, R]), "F1": float(Fd[t, 1])})
    df = pd.DataFrame(rows)
    # thresholds frozen on DEV: tertiles of relative settle depth per model
    th = {m: np.quantile(g["settle_rel"], [1 / 3, 2 / 3]).tolist()
          for m, g in df[df.split == "dev"].groupby("model")}

    degenerate = {m: th[m][0] == th[m][1] for m in th}

    def label(r):
        if r["n_rises"] >= 2:
            return "nonmonotonic"
        if degenerate[r["model"]]:  # coarse R (Ouro R=4): dev tertiles coincide -> exact settle loop
            return f"settles_at_r/R={r['settle_rel']:.2f}"
        a, b = th[r["model"]]
        return "early_settling" if r["settle_rel"] <= a else ("mid_settling" if r["settle_rel"] <= b
                                                               else "late_settling")
    df["label"] = df.apply(label, axis=1)
    df.to_csv(out / "stagea_token_patterns.csv", index=False)
    props = (df.groupby(["model", "split", "label"]).size() / df.groupby(["model", "split"]).size()
             ).rename("share").reset_index()
    role_props = (df.groupby(["model", "split", "role", "label"]).size() /
                  df.groupby(["model", "split", "role"]).size()).rename("share").reset_index()
    # repeatability of token-identity structure: same token ID in dev vs holdout prompts
    rep = {}
    for m, g in df.groupby("model"):
        a = g[g.split == "dev"].groupby("token_id").settle_rel.mean()
        b = g[g.split == "holdout"].groupby("token_id").settle_rel.mean()
        cnt_a = g[g.split == "dev"].groupby("token_id").size()
        cnt_b = g[g.split == "holdout"].groupby("token_id").size()
        common = [t for t in a.index.intersection(b.index) if cnt_a[t] >= 3 and cnt_b[t] >= 3]
        rho = pd.Series(a[common]).rank().corr(pd.Series(b[common]).rank()) if len(common) > 5 else None
        # variance share of settle depth: between prompts vs between roles vs residual
        tot = g.settle_rel.var()
        by_prompt = g.groupby("uid").settle_rel.transform("mean").var() / tot if tot else None
        by_role = g.groupby("role").settle_rel.transform("mean").var() / tot if tot else None
        by_tok = g.groupby("token_id").settle_rel.transform("mean").var() / tot if tot else None
        rep[m] = {"n_common_token_ids": len(common), "token_id_settle_spearman_dev_vs_holdout": rho,
                  "var_share_prompt": by_prompt, "var_share_role": by_role, "var_share_token_id_insample": by_tok,
                  "dev_thresholds_settle_rel": th[m]}
    _json(out / "stagea_patterns.json", {"F_SETTLE": F_SETTLE, "U_RISE": U_RISE, "thresholds": th,
                                         "label_share": props.to_dict("records"), "repeatability": rep})
    role_props.to_csv(out / "stagea_label_share_by_role.csv", index=False)
    md += ["## Stage A — descriptive trajectory patterns (dev-frozen thresholds, holdout check)", "",
           f"Settle depth = first loop after which F_t(r) < {F_SETTLE} through R (a-priori); a token with ≥2 "
           f"update rises (U(r+1) > {1 + U_RISE:.1f}·U(r)) is labeled nonmonotonic. Early/mid/late cut points are "
           "the DEV tertiles of relative settle depth per model; shares are then compared on HOLDOUT prompts. "
           "If the dev tertiles coincide (Ouro, R=4), tokens are labeled by their exact settle loop instead.", "",
           "| model | label | dev share | holdout share |", "|---|---|---|---|"]
    pv = props.pivot_table(index=["model", "label"], columns="split", values="share").reset_index()
    for _, r in pv.iterrows():
        md.append(f"| {r['model']} | {r['label']} | {_f(r.get('dev'))} | {_f(r.get('holdout'))} |")
    md += ["", "| model | dev cut points (settle r/R) | Spearman token-ID settle depth dev↔holdout (n IDs) | "
           "var share: prompt | role | token ID (in-sample) |", "|---|---|---|---|---|---|"]
    for m, e in rep.items():
        md.append(f"| {m} | {[round(x, 3) for x in e['dev_thresholds_settle_rel']]} | "
                  f"{_f(e['token_id_settle_spearman_dev_vs_holdout'])} ({e['n_common_token_ids']}) | "
                  f"{_f(e['var_share_prompt'])} | {_f(e['var_share_role'])} | {_f(e['var_share_token_id_insample'])} |")
    md += ["", "Generated vs prompt and role-wise label shares are in stagea_label_share_by_role.csv.",
           "Ouro per-loop readout is reported in the controls section below (Huginn: not model-native).", ""]
    return df


# ------------------------------------------------------------------------------------------- Stage B/C
def _lexical(tok, tid: int) -> dict:
    s = tok.convert_ids_to_tokens(int(tid))
    txt = tok.decode([int(tid)])
    core = txt.strip()
    return {"tok_chars": len(core), "tok_word_start": float(txt[:1].isspace() or s[:1] in ("Ġ", "▁")),
            "tok_punct": float(bool(core) and all(not c.isalnum() for c in core)),
            "tok_digit": float(any(c.isdigit() for c in core)), "tok_upper": float(core[:1].isupper())}


def source_table(run: Path):
    from loopzip_trajectory_sweep import _tokenizers
    toks = _tokenizers(("ouro-2.6b", "huginn-0125"))
    split = _split(run)
    jobs = json.loads((run / "jobs.json").read_text())
    recs = []
    for j in (j for j in jobs if j["kind"] == "B"):
        z = np.load(run / "raw" / f"{j['key']}.npz")
        zr = np.load(run / "raw" / f"{j['depends']}.npz")
        rm = json.loads((run / "raw" / f"{j['depends']}.json").read_text())
        P = len(rm["roles"])
        R = z["acc_E"].shape[2] - 1
        n = z["acc_E"].shape[1]
        for si, s in enumerate(z["sources"].astype(int)):
            lq = zr["src_last_query"][:, s]
            recs.append({"model": j["model"], "workload": j["workload"], "uid": j["uid"], "item_id": j["item_id"],
                         "split": split[j["item_id"]], "source": int(s), "token_id": rm["prompt_ids"][s],
                         **_lexical(toks[j["model"]][1], rm["prompt_ids"][s]),
                         "role": rm["roles"][s], "position": int(s), "position_pct": s / (P - 1), "age": P - s,
                         "prompt_len": P, "n_tokens": n,
                         "attn_recv_mean": float(zr["src_recv_mean"][:, s].mean()),
                         "attn_recv_max": float(zr["src_recv_max"][:, s].max()),
                         "attn_last_query_mean": float(lq.mean()), "attn_last_query_last": float(lq[-1]),
                         "attn_last_query_loopvar": float(lq.var()),
                         "src_U1": float(zr["prompt_U"][s, 1]), "src_U_mean": _nanmean(zr["prompt_U"][s, 1:]),
                         "src_UR": float(zr["prompt_U"][s, R]), "src_F1": float(zr["prompt_F"][s, 1]),
                         "src_C_mean": _nanmean(zr["prompt_C"][s, 1:R]),
                         "kl_mean": _nanmean(z["acc_kl"][si]),
                         "kl_late": _nanmean(z["acc_kl"][si, 64:]) if n > 64 else None,
                         "E_R_mean": _nanmean(z["acc_E"][si, :, R])})
    return pd.DataFrame(recs)


def audit_artifacts(run: Path, out: Path, st: pd.DataFrame, md: list):
    jobs = json.loads((run / "jobs.json").read_text())
    elig, parity, prof = {}, {}, defaultdict(list)
    for j in (j for j in jobs if j["kind"] == "Bref"):
        m = json.loads((run / "raw" / f"{j['key']}.json").read_text())
        roles = m["roles"]
        cens = st[(st.uid == j["uid"]) & (st.model == j["model"])]
        elig[j["key"]] = {"prompt_len": len(roles), "eligible": len(m["eligible"]),
                          "excluded_other": sum(r == "other" for r in roles),
                          "by_role": {r: roles.count(r) for r in sorted(set(roles))},
                          "census_sources": int(cens.source.nunique()),
                          "census_complete": sorted(cens.source.tolist()) == sorted(m["eligible"]),
                          "ref_tokens": m["ref"]["n"], "ref_stop": m["ref"]["stop"]}
        parity[j["key"]] = m["validation"] | {"first_token_lp": m["ref"]["first_token_lp"]}
    for j in jobs:
        m = json.loads((run / "raw" / f"{j['key']}.json").read_text())
        prof[(j["model"], j["kind"])].append({"peak_gpu_bytes": m.get("peak_gpu_bytes"),
                                              "per_source_s": np.mean(m["per_source_s"]) if "per_source_s" in m else None,
                                              "prefill_s": m.get("timing", {}).get("prefill_s")})
    profile = {f"{m}/{k}": {"jobs": len(v), "max_peak_gpu_gb": max(x["peak_gpu_bytes"] for x in v) / 1e9,
                            "mean_per_source_s": _nanmean([x["per_source_s"] for x in v if x["per_source_s"]]) if
                            any(x["per_source_s"] for x in v) else None,
                            "mean_prefill_s": _nanmean([x["prefill_s"] for x in v if x["prefill_s"]]) if
                            any(x["prefill_s"] for x in v) else None} for (m, k), v in prof.items()}
    _json(out / "exhaustive_eligibility_audit.json", elig)
    _json(out / "first_token_and_cache_parity.json", parity)
    _json(out / "execution_profile.json", profile | {"hardware": "H200 x8, BF16, B=1, one model per worker"})
    st.to_csv(out / "source_metadata.csv", index=False)
    worst = max(max(v[k] for k in v if isinstance(v[k], float) and k not in ("masked_max_E", "first_token_lp"))
                for v in parity.values())
    md += ["## Stage B audit", "",
           f"{len(elig)} Stage-B prompts; census complete for all: {all(e['census_complete'] for e in elig.values())}; "
           f"eligible sources {sum(e['eligible'] for e in elig.values())}, excluded 'other' tokens "
           f"{sum(e['excluded_other'] for e in elig.values())}. Worst G0 parity value over prompts: {worst:.3g}.", ""]


def matched_pairs(st: pd.DataFrame, out: Path, md: list):
    rows, null = [], []
    rng = random.Random(0)
    for (m, uid, role), g in st.groupby(["model", "uid", "role"]):
        g = g.sort_values("position")
        pct = g.attn_recv_mean.rank(pct=True)
        rec = g.assign(apct=pct).to_dict("records")
        for i in range(len(rec)):
            for k in range(i + 1, len(rec)):
                a, b = rec[i], rec[k]
                if abs(a["position"] - b["position"]) > 4:
                    break
                if abs(a["apct"] - b["apct"]) <= 0.10:
                    lr = abs(np.log10(max(a["kl_mean"], 1e-12) / max(b["kl_mean"], 1e-12)))
                    rows.append({"model": m, "uid": uid, "role": role, "a": a["source"], "b": b["source"],
                                 "attn_stratum": "high" if (a["apct"] + b["apct"]) / 2 > 2 / 3 else
                                 ("low" if (a["apct"] + b["apct"]) / 2 < 1 / 3 else "mid"),
                                 "abs_log10_kl_ratio": lr, "kl_a": a["kl_mean"], "kl_b": b["kl_mean"]})
        for _ in range(200):
            if len(rec) < 2:
                break
            a, b = rng.sample(rec, 2)
            null.append({"model": m, "uid": uid, "abs_log10_kl_ratio":
                         abs(np.log10(max(a["kl_mean"], 1e-12) / max(b["kl_mean"], 1e-12)))})
    pr, nl = pd.DataFrame(rows), pd.DataFrame(null)
    pr.to_csv(out / "stagec_matched_pairs.csv", index=False)
    md += ["## Stage C — matched-source comparisons (same prompt, same role, |Δposition| ≤ 4, "
           "attention-received percentile within 0.10)", "",
           "|log10 KL ratio| for matched pairs vs random same-role pairs from the same prompt; "
           "per-prompt medians, then mean [95% CI] over prompts.", "",
           "| model | attention stratum | matched pairs | prompts | matched median | random same-role median | "
           "share of matched pairs >10× apart |", "|---|---|---|---|---|---|---|"]
    res = []
    for (m, s), g in pr.groupby(["model", "attn_stratum"]):
        med = g.groupby("uid").abs_log10_kl_ratio.median()
        nul = nl[nl.model == m].groupby("uid").abs_log10_kl_ratio.median()
        ci = _boot(med.tolist())
        res.append({"model": m, "stratum": s, "pairs": len(g), "matched": float(med.mean()), "ci": ci,
                    "random": float(nul.mean()), "gt10x": float((g.abs_log10_kl_ratio > 1).mean())})
        md.append(f"| {m} | {s} | {len(g)} | {g.uid.nunique()} | {med.mean():.3f}"
                  + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else "") + f" | {nul.mean():.3f} | "
                  f"{(g.abs_log10_kl_ratio > 1).mean():.3f} |")
    md += ["", "Interpretation guard: matched sources are adjacent tokens with similar received attention; residual "
           "dispersion is NOT by itself recurrence-specific (value content, token identity, and local syntax "
           "are uncontrolled).", ""]
    return res


def role_support(st: pd.DataFrame, out: Path, md: list):
    rows = []
    for (m, uid), g in st.groupby(["model", "uid"]):
        c, i = g[g.role == "content"], g[g.role != "content"]
        if i.empty:
            continue
        pos_overlap = max(0, min(c.position.max(), i.position.max()) - max(c.position.min(), i.position.min()))
        lo, hi = max(c.attn_recv_mean.min(), i.attn_recv_mean.min()), min(c.attn_recv_mean.max(), i.attn_recv_mean.max())
        cc, ii = c[(c.attn_recv_mean >= lo) & (c.attn_recv_mean <= hi)], i[(i.attn_recv_mean >= lo) & (i.attn_recv_mean <= hi)]
        rows.append({"model": m, "uid": uid, "position_overlap_tokens": int(pos_overlap),
                     "attn_common_support_content": len(cc), "attn_common_support_instr_q": len(ii),
                     "kl_content_in_support": _nanmean(cc.kl_mean), "kl_instr_q_in_support": _nanmean(ii.kl_mean)})
    d = pd.DataFrame(rows)
    d.to_csv(out / "stagec_role_support.csv", index=False)
    md += ["## Stage C — role comparison support", "",
           f"Position overlap between Content and Instruction/Question sources: "
           f"{int(d.position_overlap_tokens.max())} tokens in every prompt (Instruction/Question always follow "
           "Context), so there is NO common position/age support and no role-causality claim is made. Within "
           "the common range of received attention only (descriptive):", "",
           "| model | prompts | content n | instr/question n | mean KL content | mean KL instr/question |",
           "|---|---|---|---|---|---|"]
    for m, g in d.groupby("model"):
        md.append(f"| {m} | {len(g)} | {g.attn_common_support_content.sum()} | {g.attn_common_support_instr_q.sum()} | "
                  f"{_f(g.kl_content_in_support.mean())} | {_f(g.kl_instr_q_in_support.mean())} |")
    md.append("")


ATTN_FEATS = ["position_pct", "attn_recv_mean", "attn_recv_max", "attn_last_query_mean",
              "attn_last_query_last", "attn_last_query_loopvar", "is_instr", "is_question"]
LEX_FEATS = ["tok_chars", "tok_word_start", "tok_punct", "tok_digit", "tok_upper"]
CONTROL_FEATS = ATTN_FEATS + LEX_FEATS
REC_FEATS = ["src_U1", "src_U_mean", "src_UR", "src_F1", "src_C_mean"]


def lopo(st: pd.DataFrame, out: Path, md: list):
    """Descriptive leave-one-PROMPT-out ridge on log mean-KL; NOT the Stage D heldout test."""
    st = st.assign(is_instr=(st.role == "instruction").astype(float), is_question=(st.role == "question").astype(float),
                   y=np.log10(st.kl_mean.clip(lower=1e-9)))
    res = []
    for m, g in st.groupby("model"):
        for name, feats in (("attention/position/role", ATTN_FEATS), ("lexical", LEX_FEATS),
                            ("control (attn+lexical)", CONTROL_FEATS), ("recurrence", REC_FEATS),
                            ("control+recurrence", CONTROL_FEATS + REC_FEATS)):
            rhos = []
            for uid in g.uid.unique():
                tr, te = g[g.uid != uid], g[g.uid == uid]
                X = tr[feats].to_numpy(float)
                mu, sd = X.mean(0), X.std(0) + 1e-12
                Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
                w = np.linalg.solve(Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1]), Xs.T @ tr.y.to_numpy())
                pred = np.c_[np.ones(len(te)), (te[feats].to_numpy(float) - mu) / sd] @ w
                rhos.append(pd.Series(pred).rank().corr(pd.Series(te.y.to_numpy()).rank()))
            res.append({"model": m, "features": name, "mean_spearman": float(np.mean(rhos)), "ci": _boot(rhos),
                        "per_prompt": rhos})
        # paired lift of adding recurrence features to the full control set, per held-out prompt
        c = next(r for r in res if r["model"] == m and r["features"] == "control (attn+lexical)")["per_prompt"]
        cr = next(r for r in res if r["model"] == m and r["features"] == "control+recurrence")["per_prompt"]
        lift = [b - a for a, b in zip(c, cr)]
        res.append({"model": m, "features": "LIFT: control+recurrence − control", "mean_spearman": float(np.mean(lift)),
                    "ci": _boot(lift), "per_prompt": lift})
    _json(out / "stagec_lopo_descriptive.json", res)
    md += ["## Stage C — descriptive leave-one-prompt-out ranking of source harm (NOT Stage D)", "",
           "Ridge on log10 mean KL with clean-prefill features; Spearman within the held-out prompt. The 16 "
           "census prompts are exhausted for case/threshold choices, so this is not an out-of-sample claim.", "",
           "| model | features | mean within-prompt Spearman [95% CI over prompts] |", "|---|---|---|"]
    for r in res:
        ci = r["ci"]
        md.append(f"| {r['model']} | {r['features']} | {r['mean_spearman']:.3f}" +
                  (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else "") + " |")
    md.append("")
    return res


def example_selection(st: pd.DataFrame, pairs_csv: Path, out: Path, md: list):
    sel = []
    for (m, uid), g in st[st.split == "dev"].groupby(["model", "uid"]):
        g = g.sort_values("kl_mean")
        for lab, r in (("smallest", g.iloc[0]), ("median", g.iloc[len(g) // 2]), ("largest", g.iloc[-1])):
            sel.append({"model": m, "uid": uid, "label": lab, "source": int(r.source), "role": r.role,
                        "token_id": int(r.token_id), "kl_mean": r.kl_mean, "attn_recv_pct":
                        float(g.attn_recv_mean.rank(pct=True)[r.name])})
    pd.DataFrame(sel).to_csv(out / "stagec_dev_examples.csv", index=False)
    conf = []
    for (m, sp), g in st.groupby(["model", "split"]):
        per = g.groupby("uid").apply(lambda x: pd.Series({
            "max_over_median": x.kl_mean.max() / x.kl_mean.median(),
            "top5_share": np.sort(x.kl_mean.values)[::-1][:max(1, len(x) // 20)].sum() / x.kl_mean.sum(),
            "largest_attn_pct": float(x.attn_recv_mean.rank(pct=True)[x.kl_mean.idxmax()])}))
        conf.append({"model": m, "split": sp, **{k: float(per[k].mean()) for k in per}})
    pr = pd.read_csv(pairs_csv)
    split = st.drop_duplicates("uid").set_index("uid").split
    pr["split"] = pr.uid.map(split)
    disc = pr.groupby(["model", "split"]).apply(lambda x: (x.abs_log10_kl_ratio > 1).mean()).rename("gt10x")
    md += ["## Stage C — example selection protocol (dev 2 prompts / holdout 2 prompts per model×workload)", "",
           "Largest/median/smallest sources are picked on DEV only (stagec_dev_examples.csv). Holdout confirms "
           "whether the heavy-tail and discordant-pair statistics carry over:", "",
           "| model | split | max/median KL | top-5% KL share | attention pct of largest source | matched pairs >10× |",
           "|---|---|---|---|---|---|"]
    for c in conf:
        md.append(f"| {c['model']} | {c['split']} | {c['max_over_median']:.1f} | {c['top5_share']:.3f} | "
                  f"{c['largest_attn_pct']:.2f} | {_f(disc.get((c['model'], c['split'])))} |")
    md.append("")


# ------------------------------------------------------------------------------------------- controls
def controls(ctl: Path, md: list):
    out = ctl / "analysis"
    out.mkdir(exist_ok=True)
    jobs = json.loads((ctl / "jobs.json").read_text())
    missing = [j["key"] for j in jobs if not (ctl / "raw" / f"{j['key']}.json").exists()]
    errs = [p.stem for p in (ctl / "errors").glob("*.json")] if (ctl / "errors").exists() else []
    _json(ctl / "completeness.json", {"complete": not missing and not errs, "missing": missing, "errors": errs,
                                      "jobs": len(jobs)})
    md += ["## Stage C — mechanism controls (clean history, small labeled diagnostics)", "",
           f"controls run complete={not missing and not errs} ({len(jobs) - len(missing)}/{len(jobs)} jobs).", ""]
    rows = []
    for j in (j for j in jobs if j["kind"] == "CTL" and (ctl / "raw" / f"{j['key']}.json").exists()):
        m = json.loads((ctl / "raw" / f"{j['key']}.json").read_text())
        R = m["R"]
        for r in m["rows"]:
            E = np.array(r["E"])
            rows.append({"model": j["model"], "workload": j["workload"], "uid": j["uid"], "label": r["label"],
                         "source": r["source"], "t": r["t"], "arm": r["arm"], "loop": r["loop"], "seed": r["seed"],
                         "kl": r["kl"], "top1_changed": r["top1_changed"], "E_R": float(E[R]),
                         "E_at_loop": float(E[r["loop"]]) if r["loop"] else None,
                         "E_R_over_E_loop": float(E[R] / E[r["loop"]]) if r["loop"] and E[r["loop"]] > 0 else None,
                         "delta_norm": r.get("delta_norm"), "R": R})
    d = pd.DataFrame(rows)
    d.to_csv(out / "mechanism_controls_long.csv", index=False)
    if not d.empty:
        md += ["Per case (source × t), all at query token t with pristine FULL history. One-loop mask = recurrent "
               "layers of loop r only; 'own delta' injects the one-loop mask's h_r change (should reproduce it up "
               "to BF16 rounding); 'random' injects a random direction with the same norm.", "",
               "KL is heavy-tailed across cases, so medians are shown next to means.", "",
               "| model | arm | n | mean KL | median KL | median E(t,R) |", "|---|---|---|---|---|---|"]
        for (m, a), g in d.groupby(["model", "arm"]):
            md.append(f"| {m} | {a} | {len(g)} | {g.kl.mean():.4g} | {g.kl.median():.4g} | {g.E_R.median():.4g} |")
        md += ["", "| model | loop r | one-loop mask KL mean / median | random-matched KL median | "
               "median random/mask KL ratio | median E(R)/E(r) after one-loop mask | own-delta≈mask: median rel. gap, "
               "Spearman |", "|---|---|---|---|---|---|---|"]
        piv = []
        for (m, L), g in d[d.loop.notna()].groupby(["model", "loop"]):
            key = ["uid", "source", "t"]
            mk = g[g.arm == "one_loop_mask"].set_index(key)
            ow = g[g.arm == "inject_own_delta"].set_index(key)
            rn = g[g.arm == "inject_random_matched"].groupby(key).kl.mean()
            ratio = (rn / mk.kl.clip(lower=1e-12)).median()
            gap = float(((ow.kl - mk.kl).abs() / mk.kl.clip(lower=1e-9)).median())
            rho = float(ow.kl.rank().corr(mk.kl.rank()))
            piv.append({"model": m, "loop": int(L), "mask_kl_mean": mk.kl.mean(), "mask_kl_median": mk.kl.median(),
                        "own_kl_mean": ow.kl.mean(), "rand_kl_median": rn.median(), "ratio": float(ratio),
                        "growth": float(mk.E_R_over_E_loop.median()), "own_vs_mask_median_rel": gap,
                        "own_vs_mask_spearman": rho})
            md.append(f"| {m} | {int(L)} | {mk.kl.mean():.3g} / {mk.kl.median():.3g} | {rn.median():.3g} | {ratio:.3g} | "
                      f"{mk.E_R_over_E_loop.median():.3g} | {gap:.3g}, {rho:.3f} |")
        # additivity: sum of one-loop KLs vs all-loop (Ouro: all loops probed; Huginn: subset only)
        add = []
        for (m, uid, s, t), g in d.groupby(["model", "uid", "source", "t"]):
            one = g[g.arm == "one_loop_mask"].kl.sum()
            allr = g[g.arm == "rec_only_all_loops"].kl.mean()
            allp = g[g.arm == "all_loops_plain"].kl.mean()
            add.append({"model": m, "sum_one_loop_kl": one, "rec_only_all_kl": allr, "all_plain_kl": allp})
        ad = pd.DataFrame(add)
        md += ["", "| model | Σ one-loop KL (probed loops) | recurrent-only all-loop KL | all-loop + plain KL |",
               "|---|---|---|---|"]
        for m, g in ad.groupby("model"):
            md.append(f"| {m} | {g.sum_one_loop_kl.mean():.4g} | {g.rec_only_all_kl.mean():.4g} | {g.all_plain_kl.mean():.4g} |")
        md += ["", "Huginn probes 7 of 32 loops, so its Σ is not a full additive decomposition.", ""]
        _json(out / "mechanism_controls_summary.json", {"per_loop": piv, "additivity": ad.groupby("model").mean().reset_index()
                                                         .to_dict("records")})
    # Ouro readout
    ro = [j for j in jobs if j["kind"] == "RO" and (ctl / "raw" / f"{j['key']}.npz").exists()]
    if ro:
        kl, ag = defaultdict(list), defaultdict(list)
        for j in ro:
            z = np.load(ctl / "raw" / f"{j['key']}.npz")
            kl[j["workload"]].append(z["kl_to_final"])
            ag[j["workload"]].append(z["top1_agree"])
        md += ["## Stage A supplement — Ouro-native per-loop readout (generated tokens, lm_head(h_r))", "",
               "| workload | units | KL(p_r‖p_R) r=1 | r=2 | r=3 | top-1 = final at r=1 | r=2 | r=3 |",
               "|---|---|---|---|---|---|---|---|"]
        summ = {}
        for w in sorted(kl):
            K, A = np.concatenate(kl[w]), np.concatenate(ag[w])
            summ[w] = {"kl": K.mean(0).tolist(), "top1": A.mean(0).tolist(), "units": len(kl[w])}
            md.append(f"| {w} | {len(kl[w])} | " + " | ".join(f"{x:.3g}" for x in K.mean(0)[:3]) + " | " +
                      " | ".join(f"{x:.3f}" for x in A.mean(0)[:3]) + " |")
        _json(out / "ouro_readout_summary.json", summ)
        md.append("")
    # CR auxiliary
    cr = [j for j in jobs if j["kind"] == "CR" and (ctl / "raw" / f"{j['key']}.npz").exists()]
    if cr:
        recs = []
        for j in cr:
            z = np.load(ctl / "raw" / f"{j['key']}.npz")
            ev = set(j["evidence_positions"])
            rm = json.loads((ctl / "raw" / f"{j['depends']}.json").read_text())
            for si, s in enumerate(z["sources"].astype(int)):
                recs.append({"model": j["model"], "uid": j["uid"], "source": int(s), "evidence": s in ev,
                             "role": rm["roles"][s], "kl_mean": _nanmean(z["acc_kl"][si]), "n_tokens": rm["ref"]["n"]})
        c = pd.DataFrame(recs)
        c.to_csv(out / "cr_aux_per_source.csv", index=False)
        md += ["## Auxiliary — v5 Controlled Retrieval persistent census (evidence labels post hoc only)", "",
               "| model | prompts | sources | evidence | mean KL evidence | mean KL non-evidence | evidence share of top-5% |"
               " median evidence KL percentile |", "|---|---|---|---|---|---|---|---|"]
        for m, g in c.groupby("model"):
            g = g.assign(pct=g.groupby("uid").kl_mean.rank(pct=True))
            top = g[g.groupby("uid").kl_mean.rank(ascending=False, pct=True) <= .05]
            md.append(f"| {m} | {g.uid.nunique()} | {len(g)} | {int(g.evidence.sum())} | "
                      f"{g[g.evidence].kl_mean.mean():.4g} | {g[~g.evidence].kl_mean.mean():.4g} | "
                      f"{top.evidence.mean():.3f} | {g[g.evidence].pct.median():.3f} |")
        md.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-run", required=True)
    ap.add_argument("--controls-run")
    a = ap.parse_args()
    run = Path(a.pilot_run).resolve()
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    md = ["# LoopZip v6 — Stage A patterns and Stage C analyses", "",
          "Descriptive pilot (4 prompts per model × workload in Stage B/C). Bootstrap over prompts. "
          "Full-sized reference KV + read masks: no HBM/latency claims.", ""]
    stage_a_patterns(run, out, md)
    st = source_table(run)
    audit_artifacts(run, out, st, md)
    matched_pairs(st, out, md)
    role_support(st, out, md)
    lopo(st, out, md)
    example_selection(st, out / "stagec_matched_pairs.csv", out, md)
    if a.controls_run:
        controls(Path(a.controls_run).resolve(), md)
    (out / "STAGEA_STAGEC_FINDINGS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
