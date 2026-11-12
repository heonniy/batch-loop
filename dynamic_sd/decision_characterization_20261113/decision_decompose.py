#!/usr/bin/env python3
"""Stage A — decompose the measured same-prefix one-round oracle into nested policy classes.

Classes (per cell = model x exit regime x precision):
  S     best single fixed action (AR or DtV(d,K)), calibrated on CALIBRATION prompts
  A     {AR, best fixed SD action DtV(d0,K0)}                      (speculate or not)
  K     {AR} U {DtV(d0,K) for all K}                               (adaptive length, fixed d0)
  d     {AR} U {DtV(d,K0) for all d}                               (adaptive draft recurrence, fixed K0)
  Joint {AR} U all DtV(d,K)
S c A c K c Joint and S c A c d c Joint.

Evaluation prompts are disjoint from calibration prompts (2-fold over prompts, each prompt
evaluated once). Within a class, the per-prefix choice is a DESCRIPTIVE same-prefix oracle:
chosen on two timing repeats, scored on the held-out third (3-fold over repeats). This is not
a deployable policy and not prompt-level generalization of the per-prefix choice; only S/d0/K0
are calibrated out-of-sample. Rates are aggregate committed tokens / aggregate measured ms.

Usage: decision_decompose.py --cf-dir DIR [--manifest MANIFEST.sha256] --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cf_receipts import (ReceiptError, assert_disjoint, bootstrap_ratio, manifest_check, parse_action,
                         prompt_split, read_meta, read_rows, selection_bias, tok_ms, validate)

CLASSES = ("S", "A", "K", "d", "Joint")
RATIOS = (("A", "S"), ("K", "A"), ("d", "A"), ("Joint", "K"), ("Joint", "d"), ("Joint", "S"),
          ("K", "S"), ("d", "S"))


def calibrate(cell, prompts):
    tot = defaultdict(lambda: [0, 0.0])
    for p in cell["prefixes"].values():
        if p["meta"]["prompt_idx"] not in prompts:
            continue
        for a, ts in p["actions"].items():
            t, m = tok_ms(ts)
            tot[a][0] += t
            tot[a][1] += m
    rate = {a: v[0] / v[1] for a, v in tot.items()}
    S = max(sorted(rate), key=rate.get)
    sd = {a: r for a, r in rate.items() if a != "AR"}
    SD0 = max(sorted(sd), key=sd.get)
    _, d0, K0 = parse_action(SD0)
    grid = cell["actions"]
    classes = {
        "S": [S],
        "A": sorted({"AR", SD0}),
        "K": ["AR"] + [a for a in grid if a != "AR" and parse_action(a)[1] == d0],
        "d": ["AR"] + [a for a in grid if a != "AR" and parse_action(a)[2] == K0],
        "Joint": list(grid),
    }
    for small, big in (("S", "A"), ("A", "K"), ("A", "d"), ("K", "Joint"), ("d", "Joint")):
        assert set(classes[small]) <= set(classes[big]), (small, big)
    return {"S": S, "SD0": SD0, "d0": d0, "K0": K0, "calib_rate": rate, "classes": classes}


def choose(prefs, cands, r, select):
    """Per-prefix action choice on the timing repeats != r.
    select='aggregate': maximise the class's aggregate Σtok/Σms (Dinkelbach: pick argmax t - λ m,
                        λ = resulting aggregate rate, iterate to a fixed point). Nested classes are
                        then nested in-sample (Joint >= K >= A >= S on the training repeats).
    select='ratio':     argmax of the per-prefix t/m (what oracle_frontier.py / 5c6bbea3 used);
                        NOT aggregate-optimal, superset classes can score lower."""
    train = {pid: {a: tok_ms([x for x in p["actions"][a] if x["repeat"] != r]) for a in cands}
             for pid, p in prefs.items()}
    if len(cands) == 1:
        return {pid: cands[0] for pid in prefs}
    if select == "ratio":
        return {pid: max(cands, key=lambda a: tm[a][0] / tm[a][1]) for pid, tm in train.items()}
    lam = 0.0
    for _ in range(100):
        pick = {pid: max(cands, key=lambda a: tm[a][0] - lam * tm[a][1]) for pid, tm in train.items()}
        T = sum(train[pid][a][0] for pid, a in pick.items())
        M = sum(train[pid][a][1] for pid, a in pick.items())
        new = T / M
        if abs(new - lam) < 1e-12:
            break
        lam = new
    return pick


def evaluate(cell, cal, prompts, reps, select="aggregate"):
    """Per-prompt (tok, ms) for each class + pick statistics, on evaluation prompts."""
    per_prompt = {c: defaultdict(lambda: [0, 0.0]) for c in CLASSES}
    picks = {c: Counter() for c in CLASSES}
    detail = {c: {"rounds": 0, "ar": 0, "n_acc": 0, "K_eff": 0, "rejected_rounds": 0, "rows": 0}
              for c in CLASSES}
    prefs = {pid: p for pid, p in cell["prefixes"].items() if p["meta"]["prompt_idx"] in prompts}
    chosen = {(c, r): choose(prefs, cal["classes"][c], r, select) for c in CLASSES for r in reps}
    for pid, p in prefs.items():
        pr = p["meta"]["prompt_idx"]
        for c in CLASSES:
            for r in reps:
                pick = chosen[(c, r)][pid]
                trial = next(x for x in p["actions"][pick] if x["repeat"] == r)
                per_prompt[c][pr][0] += trial["committed_tokens"]
                per_prompt[c][pr][1] += trial["wall_ms"]
                picks[c][pick] += 1
                dd = detail[c]
                dd["rounds"] += 1
                if pick == "AR":
                    dd["ar"] += 1
                    dd["rows"] += trial["info"]["depth"]
                else:
                    _, d, K = parse_action(pick)
                    dd["n_acc"] += trial["info"]["n_acc"]
                    dd["K_eff"] += K
                    dd["rejected_rounds"] += int(trial["info"]["n_acc"] < K)
                    dd["rows"] += sum(trial["info"]["depths"])
    return per_prompt, picks, detail


def analyze_cell(path, manifest=None, n_boot=2000, select="aggregate"):
    sha, listed = manifest_check(path, manifest)
    rows = read_rows(path)
    meta, meta_path = read_meta(path)
    cell = validate(rows, expected_repeats=3)
    prompts_all = sorted({p["meta"]["prompt_idx"] for p in cell["prefixes"].values()})
    folds = prompt_split(prompts_all, 2)
    reps = cell["repeats"]
    pp = {c: {} for c in CLASSES}
    picks = {c: Counter() for c in CLASSES}
    detail = {c: Counter() for c in CLASSES}
    fold_cal = []
    for f in range(2):
        evalp, calp = set(folds[f]), set(folds[1 - f])
        assert_disjoint(calp, evalp)
        cal = calibrate(cell, calp)
        fold_cal.append({k: cal[k] for k in ("S", "SD0", "d0", "K0", "classes")}
                        | {"calib_prompts": sorted(calp), "eval_prompts": sorted(evalp)})
        per_prompt, pk, dt = evaluate(cell, cal, evalp, reps, select)
        for c in CLASSES:
            for pr, v in per_prompt[c].items():
                assert pr not in pp[c]
                pp[c][pr] = tuple(v)
            picks[c].update(pk[c])
            detail[c].update(dt[c])
    rates = {c: sum(v[0] for v in pp[c].values()) / sum(v[1] for v in pp[c].values()) for c in CLASSES}
    ratios = {}
    for a, b in RATIOS:
        pairs = {pr: (*pp[a][pr], *pp[b][pr]) for pr in pp[a]}
        pt, lo, hi = bootstrap_ratio(pairs, n=n_boot)
        ratios[f"{a}/{b}"] = {"point": pt, "lo": lo, "hi": hi}
    # per-prompt heterogeneity of the incremental d and K effects
    het = {}
    for a, b in (("Joint", "K"), ("K", "A"), ("d", "A")):
        vals = sorted((pp[a][pr][0] / pp[a][pr][1]) / (pp[b][pr][0] / pp[b][pr][1]) for pr in pp[a])
        het[f"{a}/{b}"] = {"min": vals[0], "median": vals[len(vals) // 2], "max": vals[-1],
                           "frac_gt_1.01": sum(v > 1.01 for v in vals) / len(vals)}
    ar_rate = None
    tot = [0, 0.0]
    for p in cell["prefixes"].values():
        t, m = tok_ms(p["actions"]["AR"])
        tot[0] += t; tot[1] += m
    ar_rate = tot[0] / tot[1]
    det = {}
    for c in CLASSES:
        d = detail[c]
        sd_rounds = d["rounds"] - d["ar"]
        det[c] = {"frac_AR": d["ar"] / d["rounds"],
                  "mean_n_acc_sd": d["n_acc"] / sd_rounds if sd_rounds else None,
                  "mean_K_sd": d["K_eff"] / sd_rounds if sd_rounds else None,
                  "frac_rejected_sd": d["rejected_rounds"] / sd_rounds if sd_rounds else None,
                  "core_rows_per_round": d["rows"] / d["rounds"],
                  "picks": dict(picks[c].most_common())}
    return {
        "file": str(path), "sha256": sha, "manifest_sha256": listed, "meta_file": meta_path,
        "exit_policy": meta.get("exit_policy"), "model": meta.get("args", {}).get("model"),
        "precision": meta.get("args", {}).get("dtype"), "source_commit": meta.get("batch_loop_commit"),
        "audit": cell["audit"], "selection_bias": selection_bias(cell),
        "folds": fold_cal, "rate_tok_per_ms": rates, "AR_all_prefixes_tok_per_ms": ar_rate,
        "class_vs_AR": {c: rates[c] / ar_rate for c in CLASSES},
        "ratios": ratios, "per_prompt_heterogeneity": het, "class_detail": det,
        "claim_scope": "DERIVED FROM RECEIPTS. Per-prefix choice = same-prefix descriptive oracle "
                       "(timing-repeat cross-fit); S/d0/K0 calibrated on disjoint prompts. B=1 eager one-round; "
                       "not a rollout, no controller cost, not held-out generalization of the choice.",
    }


def screen(res):
    jk = res["ratios"]["Joint/K"]
    ka = res["ratios"]["K/A"]
    da = res["ratios"]["d/A"]
    out = []
    out.append("Joint/K ≤1% or CI∋1 → adaptive d low priority" if (jk["point"] <= 1.01 or jk["lo"] <= 1.0)
               else "Joint/K >1% with CI>1 → adaptive d adds over adaptive K")
    out.append("K/A negligible (≤1% or CI∋1)" if (ka["point"] <= 1.01 or ka["lo"] <= 1.0) else "K/A material")
    out.append("d/A negligible (≤1% or CI∋1)" if (da["point"] <= 1.01 or da["lo"] <= 1.0) else "d/A material")
    return out


def fmt(r):
    return f"{r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", default="ouro_fixed,ouro_tau0.4,ouro_tau0.1,ouro_tau0.4_bf16,huginn_fixed,huginn_eps0.1")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, failures, ratio_mode = {}, {}, {}
    for c in a.cells.split(","):
        path = Path(a.cf_dir) / f"{c}.jsonl.gz"
        try:
            results[c] = analyze_cell(path, a.manifest, a.boot, "aggregate")
            results[c]["screen"] = screen(results[c])
            ratio_mode[c] = analyze_cell(path, a.manifest, a.boot, "ratio")
        except ReceiptError as e:
            failures[c] = str(e)
    # cross-regime paired comparisons (same prompt ids within a model)
    cross = {}
    for live, fixed in (("ouro_tau0.4", "ouro_fixed"), ("ouro_tau0.1", "ouro_fixed"),
                        ("huginn_eps0.1", "huginn_fixed")):
        if live in results and fixed in results:
            cross[f"{live}-{fixed}"] = paired_cross(Path(a.cf_dir), live, fixed, results, a.boot)
    json.dump({"cells": results, "cells_ratio_select": ratio_mode, "validation_failures": failures,
               "cross_regime": cross},
              open(out / "decomposition.json", "w"), indent=1)
    md = render(results, failures, cross, ratio_mode)
    (out / "decomposition.md").write_text(md)
    print(md)


def paired_cross(cf_dir, live, fixed, results, n_boot, select="aggregate"):
    """Recompute per-prompt class totals for both cells and bootstrap the difference of ratios
    over the shared prompt ids (distinct native target trajectories: descriptive, not causal)."""
    pp = {}
    for cell_name in (live, fixed):
        path = cf_dir / f"{cell_name}.jsonl.gz"
        cell = validate(read_rows(path), expected_repeats=3)
        prompts_all = sorted({p["meta"]["prompt_idx"] for p in cell["prefixes"].values()})
        folds = prompt_split(prompts_all, 2)
        acc = {c: {} for c in CLASSES}
        for f in range(2):
            evalp, calp = set(folds[f]), set(folds[1 - f])
            cal = calibrate(cell, calp)
            per_prompt, _, _ = evaluate(cell, cal, evalp, cell["repeats"], select)
            for c in CLASSES:
                acc[c].update({k: tuple(v) for k, v in per_prompt[c].items()})
        pp[cell_name] = acc
    shared = sorted(set(pp[live]["S"]) & set(pp[fixed]["S"]))
    out = {"shared_prompts": len(shared)}
    rng = random.Random(1)
    for a_, b_ in (("Joint", "S"), ("Joint", "K"), ("K", "A"), ("d", "A")):
        def ratio(cell_pp, ks):
            na = sum(cell_pp[a_][k][0] for k in ks); ma = sum(cell_pp[a_][k][1] for k in ks)
            nb = sum(cell_pp[b_][k][0] for k in ks); mb = sum(cell_pp[b_][k][1] for k in ks)
            return (na / ma) / (nb / mb)
        point = ratio(pp[live], shared) - ratio(pp[fixed], shared)
        bs = []
        for _ in range(n_boot):
            ks = [rng.choice(shared) for _ in shared]
            bs.append(ratio(pp[live], ks) - ratio(pp[fixed], ks))
        bs.sort()
        out[f"{a_}/{b_}"] = {"live": ratio(pp[live], shared), "fixed": ratio(pp[fixed], shared),
                             "diff": point, "lo": bs[int(0.025 * n_boot)], "hi": bs[int(0.975 * n_boot) - 1]}
    return out


def render(results, failures, cross, ratio_mode=None):
    L = ["# Stage A — nested decision decomposition (DERIVED FROM RECEIPTS)", "",
         "Source: committed `followup_20261112/results_20261112/cf/*.jsonl.gz` (SOURCE MEASURED at 5c6bbea3, "
         "B=1 eager one-round). S/d0/K0 calibrated on the other prompt fold; per-prefix choice inside a class "
         "is a same-prefix descriptive oracle (2 timing repeats pick, held-out repeat scores); the pick maximises the "
         "class aggregate Σtok/Σms (Dinkelbach). "
         "Rates = Σ committed tokens / Σ measured ms. CIs: prompt-cluster bootstrap.", ""]
    if failures:
        L += ["**Validation failures:** " + "; ".join(f"{k}: {v}" for k, v in failures.items()), ""]
    L += ["## Audit", "", "| cell | rows | prompts | prefixes kept / total | excluded (inexact) | inexact trials | "
          "selection bias kept vs excluded (mean ref D, mean k) |", "|---|---:|---:|---:|---:|---:|---|"]
    for c, r in results.items():
        a = r["audit"]
        sb = r["selection_bias"]
        sbs = "-" if not sb["excluded"] else (f"D {sb['kept']['mean_ref_D']:.2f} vs {sb['excluded']['mean_ref_D']:.2f}; "
                                              f"k {sb['kept']['mean_k']:.0f} vs {sb['excluded']['mean_k']:.0f}")
        L.append(f"| {c} | {a['rows']} | {a['prompts']} | {a['prefixes_kept']} / {a['prefixes_total']} | "
                 f"{a['prefixes_excluded_inexact']} | {a['inexact_trials']} | {sbs} |")
    L += ["", "## Calibrated fixed actions (per prompt fold)", "", "| cell | fold | S | d0,K0 (best SD) |", "|---|---:|---|---|"]
    for c, r in results.items():
        for i, f in enumerate(r["folds"]):
            L.append(f"| {c} | {i} | {f['S']} | d{f['d0']},K{f['K0']} |")
    L += ["", "## Nested class rates and incremental ratios", "",
          "| cell | S vs AR | A/S | K/A | d/A | Joint/K | Joint/d | Joint/S |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, r in results.items():
        R = r["ratios"]
        L.append(f"| {c} | {r['class_vs_AR']['S']:.3f} | {fmt(R['A/S'])} | {fmt(R['K/A'])} | {fmt(R['d/A'])} | "
                 f"{fmt(R['Joint/K'])} | {fmt(R['Joint/d'])} | {fmt(R['Joint/S'])} |")
    L += ["", "## Screen (plan's decision rule, per cell)", ""]
    for c, r in results.items():
        L.append(f"- **{c}**: " + "; ".join(r["screen"]))
    L += ["", "## Per-prompt heterogeneity (ratio per evaluation prompt)", "",
          "| cell | Joint/K median [min, max], frac>1.01 | K/A median [min, max], frac>1.01 | d/A median [min, max], frac>1.01 |",
          "|---|---|---|---|"]
    for c, r in results.items():
        h = r["per_prompt_heterogeneity"]
        cells = [f"{h[k]['median']:.3f} [{h[k]['min']:.3f}, {h[k]['max']:.3f}], {h[k]['frac_gt_1.01']:.2f}"
                 for k in ("Joint/K", "K/A", "d/A")]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## Class behaviour (evaluation prompts)", "",
          "| cell | class | frac AR | mean K (SD rounds) | mean accepted | frac rejected rounds | core rows/round | top picks |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for c, r in results.items():
        for k in CLASSES:
            d = r["class_detail"][k]
            f = lambda x: "-" if x is None else f"{x:.2f}"
            top = ", ".join(f"{a}:{n}" for a, n in list(d["picks"].items())[:4])
            L.append(f"| {c} | {k} | {d['frac_AR']:.2f} | {f(d['mean_K_sd'])} | {f(d['mean_n_acc_sd'])} | "
                     f"{f(d['frac_rejected_sd'])} | {d['core_rows_per_round']:.1f} | {top} |")
    if cross:
        L += ["", "## Cross-regime (live − fixed), paired over shared prompt ids — descriptive, distinct targets", "",
              "| pair | metric | live | fixed | diff [95% CI] |", "|---|---|---:|---:|---:|"]
        for k, v in cross.items():
            for m in ("Joint/S", "Joint/K", "K/A", "d/A"):
                x = v[m]
                L.append(f"| {k} (n={v['shared_prompts']}) | {m} | {x['live']:.3f} | {x['fixed']:.3f} | "
                         f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] |")
    if ratio_mode:
        L += ["", "## Sensitivity: per-prefix ratio-argmax selection (rule used by oracle_frontier.py at 5c6bbea3)", "",
              "Not aggregate-optimal: a superset class can score below its subset. Shown only for comparability.", "",
              "| cell | A/S | K/A | d/A | Joint/K | Joint/d | Joint/S |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c, r in ratio_mode.items():
            R = r["ratios"]
            L.append(f"| {c} | " + " | ".join(fmt(R[k]) for k in ("A/S", "K/A", "d/A", "Joint/K", "Joint/d", "Joint/S")) + " |")
    L += ["", "*Caveat:* one-round reward with different K advances different token counts; these ratios are "
          "not multi-round speedups. No controller cost. Eager B=1 (FP32 uses SDPA fallback attention)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
