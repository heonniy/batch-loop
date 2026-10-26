"""E1 analysis -> analysis/e1_depth_nll.csv, e1_exit_distribution.csv, e1_kv_geometry.csv,
e1_kv_substitution_effects.csv and e1_training_signal_report.md.

All cross-model statements are within-model (relative depth, normalised improvement); raw norms and
absolute losses are never compared across architectures. CIs: 95% bootstrap over problems/prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import zlib
from collections import defaultdict
from pathlib import Path

MODELS = ("huginn-0125", "ouro-1.4b", "ouro-2.6b")
N_BOOT = 1000


def boot(vals, key):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(zlib.crc32(key.encode()))
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(N_BOOT))
    return st.mean(vals), ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT) - 1], len(vals)


def spearman(x, y):
    if len(x) < 4:
        return float("nan")
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    a, b = rk(x), rk(y)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def ent(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    recs = [json.loads(p.read_text()) for p in sorted((run / "e1" / "raw").glob("e1__*.json"))]
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    md = ["# E1 — training-depth signal characterisation under FULL", "",
          f"Run `{run}`; {len(recs)} item records. Gold teacher forcing on the frozen GSM8K-200 / MATH500-200 targets, "
          "B=1, BF16. Ouro per-step predictions are its native exit heads (lm_head(norm h_r)); Huginn depth-r predictions "
          "use the coda readout of the loop-r latent (its own adaptive-compute path). Exit probabilities are model outputs, "
          "not evidence of training exposure.", ""]

    # ------------------------------------------------ A. depth NLL
    dn_rows = []
    md += ["## A. Gold next-token loss by recurrent depth (token-weighted; normalised improvement = (NLL_1 − NLL_r)/(NLL_1 − NLL_R))", ""]
    for m in MODELS:
        for ds in ("gsm8k", "math500"):
            g = [r for r in recs if r["job"]["model"] == m and r["job"]["dataset"] == ds]
            if not g:
                continue
            R = g[0]["R"]
            ntok = sum(r["n"] for r in g)
            tw = [sum(sum(r["depth_nll"][d]) for r in g) / ntok for d in range(R)]
            acc = [sum(sum(r["depth_top1"][d]) for r in g) / ntok for d in range(R)]
            en = [sum(sum(r["depth_entropy"][d]) for r in g) / ntok for d in range(R)]
            for d in range(R):
                m_, lo, hi, n = boot([st.mean(r["depth_nll"][d]) for r in g], f"dn{m}{ds}{d}")
                dn_rows.append({"model": m, "dataset": ds, "depth": d + 1, "rel_depth": (d + 1) / R, "nll_tokw": tw[d],
                                "nll_problem_mean": m_, "ci_lo": lo, "ci_hi": hi, "top1": acc[d], "entropy": en[d],
                                "norm_improvement": (tw[0] - tw[d]) / (tw[0] - tw[-1]) if tw[0] != tw[-1] else float("nan"),
                                "problems": n, "tokens": ntok})
            show = list(range(R)) if R <= 8 else [0, 1, 3, 7, 15, 23, 27, 30, 31]
            md.append(f"- **{m} {ds}** ({len(g)} problems, {ntok} tokens): " + "; ".join(
                f"r={d + 1}: NLL {tw[d]:.3f}, top-1 {acc[d]:.3f}, norm.impr {((tw[0] - tw[d]) / (tw[0] - tw[-1])):.2f}" for d in show))
            best = min(range(R), key=lambda d: tw[d])
            md.append(f"  - lowest NLL at r={best + 1} ({tw[best]:.3f}); final-step NLL {tw[-1]:.3f}"
                      + (f" — final step is NOT the best (non-monotone depth curve)" if best != R - 1 else ""))
    write_csv(out / "e1_depth_nll.csv", dn_rows)
    md.append("")

    # ------------------------------------------------ B. Ouro exit distribution
    ex_rows = []
    md += ["## B. Ouro ACT exit distribution (official algorithm: sigmoid gates, remaining mass on the last step)", ""]
    for m in ("ouro-1.4b", "ouro-2.6b"):
        for ds in ("gsm8k", "math500"):
            g = [r for r in recs if r["job"]["model"] == m and r["job"]["dataset"] == ds]
            if not g:
                continue
            R = g[0]["R"]
            for seg, key in (("prompt", "prompt_exit_pdf"), ("target", "target_exit_pdf")):
                pdfs = [p for r in g for p in r[key]]
                mean = [st.mean(p[i] for p in pdfs) for i in range(R)]
                eexit = [sum((i + 1) * p[i] for i in range(R)) for p in pdfs]
                ents = [ent(p) for p in pdfs]
                argm = [max(range(R), key=lambda i: p[i]) + 1 for p in pdfs]
                row = {"model": m, "dataset": ds, "segment": seg, "positions": len(pdfs),
                       **{f"p_step{i + 1}": mean[i] for i in range(R)},
                       "expected_exit_mean": st.mean(eexit), "expected_exit_std": st.pstdev(eexit),
                       "expected_exit_median": st.median(eexit), "pdf_entropy_mean": st.mean(ents),
                       **{f"mode_step{i + 1}_frac": sum(a == i + 1 for a in argm) / len(argm) for i in range(R)},
                       "inference_exit_step_threshold1.0": R}
                if seg == "target":
                    fin = [r["depth_nll"][-1][k] for r in g for k in range(r["n"])]
                    imp = [r["depth_nll"][0][k] - r["depth_nll"][-1][k] for r in g for k in range(r["n"])]
                    per = []
                    for r in g:
                        ee = [sum((i + 1) * p[i] for i in range(R)) for p in r["target_exit_pdf"]]
                        per.append(spearman(ee, r["depth_nll"][-1]))
                    row["spearman_exit_vs_final_nll_pooled"] = spearman(eexit, fin)
                    row["spearman_exit_vs_depth_improvement_pooled"] = spearman(eexit, imp)
                    mm, lo, hi, n = boot(per, f"exitnll{m}{ds}")
                    row.update({"spearman_exit_vs_final_nll_within_problem": mm, "ci_lo": lo, "ci_hi": hi})
                ex_rows.append(row)
                md.append(f"- {m} {ds} {seg}: mean pdf " + ", ".join(f"p{i + 1}={mean[i]:.3f}" for i in range(R))
                          + f"; E[exit] {st.mean(eexit):.2f} ± {st.pstdev(eexit):.2f}; pdf entropy {st.mean(ents):.2f} nats; "
                          + "mode step " + ", ".join(f"{i + 1}: {row[f'mode_step{i + 1}_frac']:.2f}" for i in range(R))
                          + (f"; within-problem Spearman(E[exit], final NLL) {row['spearman_exit_vs_final_nll_within_problem']:+.3f} "
                             f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]" if seg == "target" else ""))
    md += ["", "Inference with the checkpoint default `early_exit_threshold=1.0` exits at the last step (cumulative mass "
           "reaches 1 only there); the fixed-R4 evaluation therefore matches default inference.", ""]
    write_csv(out / "e1_exit_distribution.csv", ex_rows)

    # ------------------------------------------------ C. KV geometry
    geo_rows = []
    md += ["## C. Cross-depth K/V geometry under FULL (layer-mean of ||X^(r) − X^(R)|| / ||X^(R)|| and cos; within-model only)", ""]
    for m in MODELS:
        g = [r for r in recs if r["job"]["model"] == m]
        if not g:
            continue
        loops = g[0]["geo_loops"]
        for seg in ("prompt", "cont"):
            for metric in ("K_dist", "V_dist", "K_cos", "V_cos"):
                key = f"{metric}|{seg}"
                mats = [r["kv_geometry"][key] for r in g if key in r["kv_geometry"]]
                L = len(mats[0])
                for l in range(L):
                    for i, a in enumerate(loops):
                        for j, b in enumerate(loops):
                            geo_rows.append({"model": m, "segment": seg, "metric": metric, "layer": l, "loop_r": a, "loop_s": b,
                                             "mean": st.mean(x[l][i][j] for x in mats), "items": len(mats)})
            for metric in ("K_dist", "V_dist", "K_cos", "V_cos"):
                key = f"{metric}|{seg}"
                mats = [r["kv_geometry"][key] for r in g if key in r["kv_geometry"]]
                L = len(mats[0])
                vs_last = [st.mean(st.mean(x[l][i][-1] for l in range(L)) for x in mats) for i in range(len(loops))]
                md.append(f"- {m} {seg} {metric} vs terminal (layer mean): " + ", ".join(
                    f"r={a}: {v:.3f}" for a, v in zip(loops, vs_last)))
    write_csv(out / "e1_kv_geometry.csv", geo_rows)
    md.append("")

    # ------------------------------------------------ D. association (Ouro)
    md += ["## D. Ouro: effect of terminal-substituting ONE historical token's K/V vs that token's features", "",
           "Effect = mean over the next 32 gold predictions after the cut of KL(p_FULL ‖ p_FULL−i), inside an otherwise "
           "FULL cache. Source i = cache column of token x_i (prompt or continuation, always before the window). "
           "Spearman correlations are computed within each prompt and averaged; [95% bootstrap CI over prompts].", ""]
    d_rows = []
    feats = ("p_step1", "p_last", "expected_exit", "pdf_entropy", "nll_final", "nll_improve_1_to_R", "kv_dist_K",
             "kv_dist_V", "recency", "in_prompt")
    for m in ("ouro-1.4b", "ouro-2.6b"):
        g = [r for r in recs if r["job"]["model"] == m and "D" in r and "sources" in r["D"]]
        if not g:
            continue
        per = defaultdict(list)
        shuffled = []
        dup, allt = [], []
        for r in g:
            src = [x for x in r["D"]["sources"] if "col" in x and x.get("exit_pdf") and x.get("depth_nll")]
            dup += [x["kl_mean"] for x in r["D"]["sources"] if x["name"] == "dup_full"]
            allt += [x["kl_mean"] for x in r["D"]["sources"] if x["name"] == "all_terminal"]
            R = len(src[0]["exit_pdf"]) if src else 4
            for x in src:
                p = x["exit_pdf"]
                f = {"p_step1": p[0], "p_last": p[-1], "expected_exit": sum((i + 1) * p[i] for i in range(R)),
                     "pdf_entropy": ent(p), "nll_final": x["depth_nll"][-1],
                     "nll_improve_1_to_R": x["depth_nll"][0] - x["depth_nll"][-1], "kv_dist_K": x["kv_dist_K"],
                     "kv_dist_V": x["kv_dist_V"], "recency": -x["dist_to_window"], "in_prompt": float(x["in_prompt"])}
                d_rows.append({"model": m, "dataset": r["job"]["dataset"], "uid": r["job"]["uid"], "col": x["col"],
                               "effect_kl": x["kl_mean"], "effect_dnll": x["dnll_mean"], **f})
            eff = [x["kl_mean"] for x in src]
            for fn in feats:
                vals = [row[fn] for row in d_rows[-len(src):]]
                per[fn].append(spearman(vals, eff))
            rng = random.Random(zlib.crc32(r["job"]["uid"].encode()))
            sh = eff[:]
            rng.shuffle(sh)
            shuffled.append(spearman([row["expected_exit"] for row in d_rows[-len(src):]], sh))
        md.append(f"- **{m}** ({len(g)} prompts; negative control duplicate-FULL row effect max {max(dup):.1e}; "
                  f"positive reference all-terminal effect median {st.median(allt):.3g}; single-token effect median "
                  f"{st.median(x['effect_kl'] for x in d_rows if x['model'] == m):.3g}, max {max(x['effect_kl'] for x in d_rows if x['model'] == m):.3g})")
        for fn in feats:
            mm, lo, hi, n = boot(per[fn], f"D{m}{fn}")
            md.append(f"  - Spearman(effect, {fn}) = {mm:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={n})")
        mm, lo, hi, n = boot(shuffled, f"Dsh{m}")
        md.append(f"  - negative control: Spearman(effect shuffled within prompt, expected_exit) = {mm:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    write_csv(out / "e1_kv_substitution_effects.csv", d_rows)
    md += ["", "Correlations are associations within these checkpoints; they do not show that ACT training causes the sensitivity.", ""]
    (out / "e1_training_signal_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
