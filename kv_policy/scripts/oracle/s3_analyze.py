"""Stage 3 analysis: single-token restoration values, equal-budget selection policies, Go/No-Go.

Decision rules (fixed in code before any Stage-3 result was inspected; HUGINN_R32_KV_ORACLE_PLAN.md):
  noise_w(prompt) = max over 3 latent-perturbed LAST rows |NLL_w(LAST_alt) - NLL_w(LAST)| (window w).
  (b) "meaningful positive restoration value": a candidate with V_A > noise_A AND V_B > noise_B
      (selected on A, reproduced on the held-out window B); report the fraction of prompts with >= 1.
  (c) "candidate-oracle sets improve loss at equal KV budget": on held-out window B, the mean over
      prompts of [NLL_B(LAST) - NLL_B(ORACLE+FILL@f)] AND of [NLL_B(RANDOM@f) - NLL_B(ORACLE+FILL@f)]
      (RANDOM averaged over its 5 seeds) both have a 95% prompt-bootstrap CI above 0 for some f.
  GO           = Stage-1 parity PASS and (b) holds for >= half of the prompts and (c) holds.
  CONDITIONAL  = (c) holds but neither CONV-DIST/CONV-CONS nor RECENT beats RANDOM on window B (CI > 0)
                 at that budget.
  NO-GO        = otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import zlib
from pathlib import Path

N_BOOT = 2000
POLICIES = ("random", "recent", "conv_dist", "conv_cons", "oracle_fill", "oracle_only")


def wmean(v, a, b):
    return sum(v[a:b]) / (b - a)


def boot(vals: list[float], key: str) -> tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(zlib.crc32(key.encode()))
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(N_BOOT))
    return st.mean(vals), ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT) - 1]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage1-report", default="results/20261023T075134Z_huginn_r32_oracle_stage1/analysis/stage1_parity.md")
    args = ap.parse_args()
    run = Path(args.run_dir)
    recs = [json.loads(p.read_text()) for p in sorted((run / "stage3" / "raw").glob("s3__*.json"))]
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    W = recs[0]["W"]

    cand_rows, pareto, per_prompt = [], [], []
    for r in recs:
        ds, uid = r["job"]["dataset"], r["job"]["uid"]
        nz = r["noise_scale"]
        anchors = [x for x in r["phase1"] + r["phase2"] if x["label"] == "__anchor__"]
        a_vals = [wmean(x["nll"], 0, W) for x in anchors]
        singles = [x for x in r["phase1"] if x["label"] != "__anchor__"]
        cmeta = {c["col"]: c for c in r["candidates"]}
        for x in singles:
            c = int(x["label"].split(":")[1])
            m = cmeta[c]
            cand_rows.append({"dataset": ds, "uid": uid, "col": c, "in_prompt": m["in_prompt"],
                              "dist_from_window": m["dist_from_window"], "tags": "+".join(m["tags"]),
                              "dist_to_terminal": m["dist_to_terminal"], "consecutive": m["consecutive"],
                              "V_A": x["V_A"], "V_B": x["V_B"], "VKL_A": x["VKL_A"], "VKL_B": x["VKL_B"],
                              "noise_A": nz["A"], "noise_B": nz["B"],
                              "above_noise_A": x["V_A"] > nz["A"], "above_noise_both": x["V_A"] > nz["A"] and x["V_B"] > nz["B"]})
        # phase 2: improvement vs the anchor of the same batch (V_w), and vs mean RANDOM
        p2 = {x["label"]: x for x in r["phase2"] if x["label"] != "__anchor__"}
        fracs = sorted({float(lbl.split("@")[1]) for lbl in p2})
        for f in fracs:
            rnd = [p2[f"random{s}@{f}"] for s in range(5)]
            for pol in POLICIES:
                if pol == "random":
                    vA, vB = st.mean(x["V_A"] for x in rnd), st.mean(x["V_B"] for x in rnd)
                    kA, kB = st.mean(x["VKL_A"] for x in rnd), st.mean(x["VKL_B"] for x in rnd)
                    k = rnd[0]["k"]
                else:
                    x = p2[f"{pol}@{f}"]
                    vA, vB, kA, kB, k = x["V_A"], x["V_B"], x["VKL_A"], x["VKL_B"], x["k"]
                rb = r["bytes"]
                rec_bytes = r["H"] * rb["rec_terminal_per_token"] + k * (rb["rec_full_per_token"] - rb["rec_terminal_per_token"])
                pareto.append({"dataset": ds, "uid": uid, "f": f, "policy": pol, "k": k, "H": r["H"],
                               "rec_kv_bytes_frac_of_full": rec_bytes / (r["H"] * rb["rec_full_per_token"]),
                               "total_kv_bytes_frac_of_full": (rec_bytes + r["H"] * rb["plain_per_token"]) /
                               (r["H"] * (rb["rec_full_per_token"] + rb["plain_per_token"])),
                               "gain_vs_last_A": vA, "gain_vs_last_B": vB, "klgain_vs_last_A": kA, "klgain_vs_last_B": kB,
                               "gain_vs_random_B": vB - st.mean(x["V_B"] for x in rnd)})
        per_prompt.append({"dataset": ds, "uid": uid, "P": r["P"], "n": r["n"], "cut": r["cut"], "H": r["H"],
                           "full_correct": r["full_correct"], "noise_A": nz["A"], "noise_B": nz["B"],
                           "anchor_spread_A": max(a_vals) - min(a_vals),
                           "full_minus_last_gain_A": r["full_vs_last"]["A"], "full_minus_last_gain_B": r["full_vs_last"]["B"],
                           "n_cand": len(singles), "n_above_noise_A": sum(x["V_A"] > nz["A"] for x in singles),
                           "n_above_noise_both": sum(x["V_A"] > nz["A"] and x["V_B"] > nz["B"] for x in singles),
                           "best_single_V_A": max(x["V_A"] for x in singles), "wall_s": r["wall_s"]})

    for name, rows in (("stage3_oracle_candidates.csv", cand_rows), ("stage3_oracle_pareto.csv", pareto),
                       ("stage3_oracle_prompts.csv", per_prompt)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    md = ["# Huginn R32 Stage 3 — token-level KV restoration oracle (pilot)", "",
          f"Run `{run}`; {len(recs)} prompts ({sum(r['job']['dataset'] == 'gsm8k' for r in recs)} GSM8K, "
          f"{sum(r['job']['dataset'] == 'math500' for r in recs)} MATH-500); frozen FULL-generated prefix; "
          f"window A (selection) and held-out window B = {W} tokens each after the cut; loss = mean next-token NLL "
          "of the FULL-generated tokens (nats/token). Gains are LAST-anchor loss minus row loss in the SAME batch "
          "(positive = restoration helped). Decision rules: header of scripts/oracle/s3_analyze.py (fixed before "
          "results).", ""]
    md += ["## Scale of the effect being chased", ""]
    for ds in ("gsm8k", "math500"):
        g = [p for p in per_prompt if p["dataset"] == ds]
        if not g:
            continue
        md.append(f"- {ds}: NLL gain of FULL over LAST on window A mean {st.mean(p['full_minus_last_gain_A'] for p in g):+.4f}, "
                  f"B {st.mean(p['full_minus_last_gain_B'] for p in g):+.4f}; latent-noise scale A mean "
                  f"{st.mean(p['noise_A'] for p in g):.4f}, B {st.mean(p['noise_B'] for p in g):.4f}; LAST-anchor spread "
                  f"across batch shapes (A) mean {st.mean(p['anchor_spread_A'] for p in g):.4f}, max {max(p['anchor_spread_A'] for p in g):.4f}")
    md += ["", "## Single-token restoration values (48 candidates per prompt)", "",
           "| dataset | candidates | V_A > 0 | V_A > noise_A | V_A>noise_A & V_B>noise_B | prompts with ≥1 such | "
           "Spearman V_A~V_B | Spearman V_A~dist_to_terminal | Spearman V_A~consecutive | Spearman V_A~recency |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    verdict = {}
    for ds in ("gsm8k", "math500", "all"):
        c = [x for x in cand_rows if ds == "all" or x["dataset"] == ds]
        p = [x for x in per_prompt if ds == "all" or x["dataset"] == ds]
        if not c:
            continue
        md.append(f"| {ds} | {len(c)} | {sum(x['V_A'] > 0 for x in c)} | {sum(x['above_noise_A'] for x in c)} | "
                  f"{sum(x['above_noise_both'] for x in c)} | {sum(x['n_above_noise_both'] > 0 for x in p)}/{len(p)} | "
                  f"{spearman([x['V_A'] for x in c], [x['V_B'] for x in c]):+.3f} | "
                  f"{spearman([x['V_A'] for x in c], [x['dist_to_terminal'] for x in c]):+.3f} | "
                  f"{spearman([x['V_A'] for x in c], [x['consecutive'] for x in c]):+.3f} | "
                  f"{spearman([x['V_A'] for x in c], [-x['dist_from_window'] for x in c]):+.3f} |")
    allp = per_prompt
    frac_b = sum(p["n_above_noise_both"] > 0 for p in allp) / len(allp)
    verdict["b_prompts_with_reproducible_positive_value"] = frac_b
    md += ["", "Mean V by candidate source tag (all prompts): " + ", ".join(
        f"{t}: V_A {st.mean(x['V_A'] for x in cand_rows if t in x['tags']):+.5f} / V_B "
        f"{st.mean(x['V_B'] for x in cand_rows if t in x['tags']):+.5f}" for t in ("recent", "conv", "random")) + "; "
        f"prompt-token candidates V_A {st.mean([x['V_A'] for x in cand_rows if x['in_prompt']] or [float('nan')]):+.5f}, "
        f"continuation-token candidates V_A {st.mean([x['V_A'] for x in cand_rows if not x['in_prompt']] or [float('nan')]):+.5f}.", ""]

    md += ["## Equal-budget selection policies (mean over prompts, [95% prompt-bootstrap CI])", "",
           "| dataset | f | policy | restored k (mean) | recurrent KV / FULL | total KV / FULL | gain vs LAST, window A | "
           "gain vs LAST, held-out B | gain vs RANDOM, B | KL gain vs LAST, B |",
           "|---|---:|---|---:|---:|---:|---|---|---|---|"]
    c_hold = []
    for ds in ("gsm8k", "math500", "all"):
        for f in sorted({x["f"] for x in pareto}):
            for pol in POLICIES:
                g = [x for x in pareto if (ds == "all" or x["dataset"] == ds) and x["f"] == f and x["policy"] == pol]
                if not g:
                    continue
                cells = []
                for key in ("gain_vs_last_A", "gain_vs_last_B", "gain_vs_random_B", "klgain_vs_last_B"):
                    m, lo, hi = boot([x[key] for x in g], f"{ds}{f}{pol}{key}")
                    cells.append(f"{m:+.5f} [{lo:+.5f}, {hi:+.5f}]")
                    if ds == "all" and key in ("gain_vs_last_B", "gain_vs_random_B"):
                        verdict[f"{pol}@{f}:{key}_ci"] = [lo, hi]
                md.append(f"| {ds} | {f:.2f} | {pol} | {st.mean(x['k'] for x in g):.1f} | "
                          f"{st.mean(x['rec_kv_bytes_frac_of_full'] for x in g):.3f} | "
                          f"{st.mean(x['total_kv_bytes_frac_of_full'] for x in g):.3f} | " + " | ".join(cells) + " |")
    fs = sorted({x["f"] for x in pareto})
    c_ok = [f for f in fs if verdict[f"oracle_fill@{f}:gain_vs_last_B_ci"][0] > 0 and verdict[f"oracle_fill@{f}:gain_vs_random_B_ci"][0] > 0]
    simple_ok = [f for f in c_ok if any(verdict[f"{p}@{f}:gain_vs_random_B_ci"][0] > 0 for p in ("conv_dist", "conv_cons", "recent"))]
    parity = "FP32 gate: PASS" in Path(args.stage1_report).read_text() if Path(args.stage1_report).exists() else None
    verdict.update({"a_parity_pass": parity, "c_budgets_where_oracle_beats_last_and_random_on_B": c_ok,
                    "simple_policies_beating_random_at_those_budgets": simple_ok})
    if parity and frac_b >= 0.5 and c_ok:
        decision = "GO"
    elif c_ok and not simple_ok:
        decision = "CONDITIONAL"
    else:
        decision = "NO-GO"
    verdict["decision"] = decision
    md += ["", "## Go / No-Go (rules fixed before results)", "", f"**{decision}**", "", "```", json.dumps(verdict, indent=1), "```", "",
           "Notes: V_i and ORACLE sets are selected with the FULL-generated target tokens of window A (target leakage); "
           "window B is held out for the selection but is still a continuation of the same FULL prefix. Restored "
           "per-loop K,V of prompt tokens come from the parallel FULL prefill; for continuation tokens they are "
           "computed causally inside the mixed run — no claim that a deployable predictor can choose them online. "
           "Single-token gains are not additive; sets are re-evaluated as wholes. Latency / memory of this "
           "reference engine (broadcast layout) is not a serving measurement.", ""]
    (out / "stage3_oracle_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
