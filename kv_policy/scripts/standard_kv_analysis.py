#!/usr/bin/env python3
"""CPU analysis for the standard-transformer control arm (standard_kv_census.py): per-source long table, heavy tail,
layer-resolved attention importance (the layer analogue of loop-resolved importance), LOPO feature-family ranking,
lookahead/deferred curve and the equal-budget token policy table. Output: <run>/analysis/STANDARD_FINDINGS.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from query_conditioned_analysis import _boot, _ci, _f, _json, _ridge_lopo, _spearman  # noqa: E402

LOOKAHEAD_K = (1, 2, 3, 4, 6, 8)


def long_table(run: Path) -> pd.DataFrame:
    from transformers import AutoTokenizer
    from standard_kv_census import STD_MODELS
    from query_conditioned_analysis import _lexical
    jobs = json.loads((run / "jobs.json").read_text())
    toks = {}
    rows = []
    for j in jobs:
        f = run / "raw" / f"{j['key']}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        z = np.load(run / "raw" / f"{j['key']}.npz")
        if j["model"] not in toks:
            repo, rev = STD_MODELS[j["model"]]
            toks[j["model"]] = AutoTokenizer.from_pretrained(repo, revision=rev)
        tok = toks[j["model"]]
        qa, ga = z["qa"], z["gen_attn"]                      # (L, P), (n, L, P)
        L, P = qa.shape
        n = ga.shape[0]
        q4 = max(1, L // 4)
        roles = m["roles"]
        content = m["content_positions"]
        for si, s in enumerate(z["sources"]):
            s = int(s)
            kl = z["cen_kl"][si]
            a = qa[:, s]
            g = ga[:, :, s]                                   # (n, L)
            tstar = int(g.mean(1).argmax())
            row = {"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "split": j["split"], "source": s, "P": P, "n_targets": n,
                   "role": roles[s], "is_evidence": float(roles[s] == "relevant_content"), "age": P - s,
                   "rel_pos": (s - content[0]) / max(1, content[-1] - content[0]), **_lexical(tok, m["prompt_ids"][s]),
                   "kl_mean": float(np.nanmean(kl)), "kl_max": float(np.nanmax(kl)), "nll_shift_mean": float(np.nanmean(z["cen_nll_shift"][si])),
                   "qa_attn_mean": float(a.mean()), "qa_attn_last": float(a[-1]), "qa_attn_first": float(a[0]),
                   "qa_attn_early": float(a[:q4].mean()), "qa_attn_late": float(a[-q4:].mean()), "qa_attn_maxlayer": float(a.max()),
                   "fut_attn_mean": float(g.mean()), "fut_attn_at_t": float(g[tstar].mean()), "fut_attn_late": float(g[:, -q4:].mean()),
                   "fut_attn_early": float(g[:, :q4].mean()), "fut_attn_last": float(g[:, -1].mean()), "fut_t": tstar}
            row |= {f"qa_attn_l{l + 1}": float(a[l]) for l in range(L)}
            row |= {f"fut_attn_l{l + 1}": float(g[tstar, l]) for l in range(L)}
            row |= {f"la12_attn_l{l + 1}": float(g[:2, l].max()) for l in range(L)}
            for k in LOOKAHEAD_K:
                kk = min(k, n)
                row[f"lk{k}_attn_max"] = float(g[:kk].mean(1).max())
                row[f"lk{k}_attn_late_max"] = float(g[:kk, -q4:].mean(1).max())
            rows.append(row)
    return pd.DataFrame(rows)


A_FEATS = ["qa_attn_mean", "qa_attn_last", "qa_attn_first", "qa_attn_early", "qa_attn_late", "qa_attn_maxlayer", "age", "rel_pos",
           "tok_chars", "tok_word_start", "tok_punct", "tok_digit", "tok_upper"]
F_FEATS = ["fut_attn_mean", "fut_attn_at_t", "fut_attn_late", "fut_attn_early", "fut_attn_last"]


def _logfeat(df):
    out = df.copy()
    for c in df.columns:
        if (c.startswith("qa_attn") or c.startswith("fut_attn") or c.startswith("lk")) and df[c].dtype != object:
            out[c] = np.log10(np.abs(df[c].to_numpy(float)) + 1e-9)
    return out


def analyze(run: Path) -> dict:
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    jobs = json.loads((run / "jobs.json").read_text())
    done = [j for j in jobs if (run / "raw" / f"{j['key']}.json").exists()]
    errs = sorted(p.stem for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    st = long_table(run)
    st.to_csv(out / "source_long.csv", index=False)
    md = [f"# Standard-transformer control arm — findings ({run.name})", "",
          f"Completed units {len(done)}/{len(jobs)}, errors {len(errs)}; sources {len(st)}", "",
          "Model: Qwen2.5-3B (36 layers, non-looped). Same frozen prompts/targets as the looped-model runs; eviction = masking "
          "the token for every later query at every layer; no loop planes exist.", ""]
    # heavy tail
    md += ["## Outcome structure (single-token eviction over the gold continuation)", "",
           "| dataset | split | n sources | top-5% share of total KL | mean KL evidence | mean KL other | evidence share in top-5% |", "|---|---|---|---|---|---|---|"]
    for (ds, sp), g in st.groupby(["dataset", "split"]):
        k = max(1, int(round(0.05 * len(g))))
        top = g.nlargest(k, "kl_mean")
        md.append(f"| {ds} | {sp} | {len(g)} | {_f(top.kl_mean.sum() / max(1e-12, g.kl_mean.sum()))} | {_f(g[g.is_evidence == 1].kl_mean.mean())} | "
                  f"{_f(g[g.is_evidence == 0].kl_mean.mean())} | {_f(top.is_evidence.mean())} |")
    md.append("")
    # layer-resolved importance
    L = int(max(int(c[len("qa_attn_l"):]) for c in st.columns if c.startswith("qa_attn_l") and c[len("qa_attn_l"):].isdigit()))
    sel = sorted(set([1, 2, 4, 8, 12, 18, 24, 30, 36, L]))
    md += ["## Layer-resolved attention importance: within-prompt ρ(attention at layer l, future damage), mean over prompts", "",
           "| dataset | split | query | " + " | ".join(f"l={l}" for l in sel) + " | mean over layers | early quarter | late quarter |",
           "|---|---|---|" + "---|" * (len(sel) + 3)]
    lr = []
    for (ds, sp), g in st.groupby(["dataset", "split"]):
        for pre, label, meanc, early, late in (("fut_attn_l", "consuming query t* (oracle)", "fut_attn_at_t", "fut_attn_early", "fut_attn_late"),
                                               ("la12_attn_l", "2-token lookahead", "lk2_attn_max", None, "lk2_attn_late_max"),
                                               ("qa_attn_l", "question anchor", "qa_attn_mean", "qa_attn_early", "qa_attn_late")):
            rho = {l: float(np.nanmean([_spearman(gg[f"{pre}{l}"], gg.kl_mean) for _, gg in g.groupby("uid")])) for l in sel if f"{pre}{l}" in g.columns}
            mr = float(np.nanmean([_spearman(gg[meanc], gg.kl_mean) for _, gg in g.groupby("uid")]))
            er = float(np.nanmean([_spearman(gg[early], gg.kl_mean) for _, gg in g.groupby("uid")])) if early else np.nan
            lt = float(np.nanmean([_spearman(gg[late], gg.kl_mean) for _, gg in g.groupby("uid")]))
            lr.append({"dataset": ds, "split": sp, "query": label, "rho_by_layer": rho, "mean": mr, "early": er, "late": lt})
            md.append(f"| {ds} | {sp} | {label} | " + " | ".join(_f(rho.get(l)) for l in sel) + f" | {_f(mr)} | {_f(er)} | {_f(lt)} |")
    md.append("")
    # full per-layer curve for the consuming query (all layers)
    md += ["### Full per-layer curve, consuming query t* and question anchor (all layers; pooled dev+holdout)", "",
           "| dataset | query | " + " | ".join(str(l) for l in range(1, L + 1)) + " |", "|---|---|" + "---|" * L]
    for ds, g in st.groupby("dataset"):
        for pre, label in (("fut_attn_l", "t*"), ("qa_attn_l", "anchor")):
            vals = [float(np.nanmean([_spearman(gg[f"{pre}{l}"], gg.kl_mean) for _, gg in g.groupby("uid")])) for l in range(1, L + 1)]
            md.append(f"| {ds} | {label} | " + " | ".join(_f(v, 2) for v in vals) + " |")
    md.append("")
    # LOPO prediction: A vs A+lookahead vs A+future
    md += ["## Leave-one-UID-out ranking of damage (log10 mean KL): prompt-time vs lookahead vs future-query oracle", "",
           "| dataset | split | features | mean ρ [CI] | AUROC top-10% | LIFT over A [CI] |", "|---|---|---|---|---|---|"]
    Lg = _logfeat(st.assign(y=np.log10(st.kl_mean.clip(lower=1e-9))))
    pred = []
    for (ds, sp), g in Lg.groupby(["dataset", "split"]):
        if g.uid.nunique() < 3:
            continue
        base, au0 = _ridge_lopo(g, A_FEATS, "y")
        sets = [("A prompt-anchor attention (layers) + position + lexical", A_FEATS, base, au0)]
        for k in LOOKAHEAD_K:
            rhos, au = _ridge_lopo(g, A_FEATS + [f"lk{k}_attn_max", f"lk{k}_attn_late_max"], "y")
            sets.append((f"A + lookahead k={k}", None, rhos, au))
        rhos, au = _ridge_lopo(g, A_FEATS + F_FEATS, "y")
        sets.append(("A + future-query census (oracle)", None, rhos, au))
        for name, _, rhos, au in sets:
            lift = [b - a for a, b in zip(base, rhos)]
            pred.append({"dataset": ds, "split": sp, "features": name, "rho": float(np.nanmean(rhos)), "ci": _boot(rhos),
                         "auroc": float(np.nanmean(au)), "lift": float(np.nanmean(lift)), "lift_ci": _boot(lift)})
            md.append(f"| {ds} | {sp} | {name} | {_f(np.nanmean(rhos))}{_ci(_boot(rhos))} | {_f(np.nanmean(au))} | {_f(np.nanmean(lift))}{_ci(_boot(lift))} |")
    md.append("")
    # policies
    rows = []
    for j in done:
        d = json.loads((run / "raw" / f"{j['key']}.json").read_text())
        for r in d["policy_rows"]:
            rows.append({"dataset": j["dataset"], "split": j["split"], "uid": j["uid"], "full_em": float(d["full_exact_match"]), **r,
                         "exact_match": float(r["exact_match"])})
    pol = pd.DataFrame(rows)
    pol.to_csv(out / "policy_long.csv", index=False)
    md += ["## Equal-budget token retention policies (gold ΔNLL_future, greedy EM; paired vs random)", "",
           "qa_* = question-anchor attention (last layer / mean / early quarter / late quarter / max layer); la2 = 2-token format "
           "lookahead; def4 = deferred after 4 generated tokens; fut_oracle = all generated queries (oracle); *_early/_late = layer quarter.", "",
           "| dataset | split | keep | policy | ΔNLL mean [CI] | EM | ΔNLL − random [CI] | EM − random |", "|---|---|---|---|---|---|---|---|"]
    summ = []
    for (ds, sp, keep), g in pol.groupby(["dataset", "split", "retention"]):
        rnd = g[g.policy.str.startswith("random")].groupby("uid").delta_nll_future.mean()
        rnd_em = g[g.policy.str.startswith("random")].groupby("uid").exact_match.mean()
        for pname, gp in g.groupby("policy"):
            per = gp.set_index("uid")
            paired = [per.delta_nll_future[u] - rnd[u] for u in per.index]
            pem = [per.exact_match[u] - rnd_em[u] for u in per.index]
            r = {"dataset": ds, "split": sp, "keep": keep, "policy": pname, "dnll": float(per.delta_nll_future.mean()), "ci": _boot(per.delta_nll_future.tolist()),
                 "em": float(per.exact_match.mean()), "full_em": float(per.full_em.mean()), "vs_random": float(np.mean(paired)), "vs_random_ci": _boot(paired),
                 "em_vs_random": float(np.mean(pem))}
            summ.append(r)
            md.append(f"| {ds} | {sp} | {keep} | {pname} | {_f(r['dnll'])}{_ci(r['ci'])} | {_f(r['em'], 2)} | {_f(r['vs_random'])}{_ci(r['vs_random_ci'])} | {_f(r['em_vs_random'], 2)} |")
    md.append("")
    _json(out / "standard_summary.json", {"layer_resolved": lr, "prediction": pred, "policies": summ})
    (out / "STANDARD_FINDINGS.md").write_text("\n".join(md))
    return {"units": len(done), "expected": len(jobs), "errors": errs, "sources": len(st)}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]).resolve()), indent=2, default=str))
