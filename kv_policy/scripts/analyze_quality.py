"""Aggregate Stage-2 raw per-item records into summary tables with paired statistics.

Outputs (in <run>/analysis/):
  items_gen.csv / items_nll.csv     -- one row per (model, dataset, policy, mode, uid)
  summary.csv / summary.md          -- per-cell accuracy, parse failures, NLL, generation stats
  paired_vs_full.csv                -- paired deltas vs FULL(decode) with bootstrap CIs, McNemar
  scorer_audit.json                 -- primary-vs-reference scorer disagreements, parse-failure samples
  completeness.json                 -- missing / duplicated items, failed jobs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BOOT = 10000
SEED = 20261022
ORDER = [("full", "decode"), ("last", "convert"), ("average", "convert"), ("last", "decode"), ("average", "decode"),
         ("last", "strict"), ("first", "strict"), ("average", "strict"), ("full", "strict"), ("full_altbatch", "decode")]
# strict cells are additionally compared against the FULL/strict numerical-path control
STRICT_BASE = ("full", "strict")


GRADE_FIELDS = ("gold", "pred", "pred_raw", "extract_method", "answer_source", "parse_ok", "correct",
                "correct_format_only", "failure_reason", "lm_eval_strict_match", "lm_eval_flexible_extract",
                "math_verify", "minerva_is_equiv", "normalized_string_equal")


def regrade(gen: list[dict]) -> tuple[str, dict]:
    """Re-score every stored generation with the current kvp.grading (raw files are not modified)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kvp.data import load_subset
    from kvp.grading import grade

    docs = {}
    for ds in ("gsm8k", "math500"):
        for d in load_subset(ds)[0]:
            docs[(ds, d["uid"])] = d
    for r in gen:
        for k in GRADE_FIELDS:
            r.pop(k, None)
        r.update(grade(r["dataset"], r["answer_text"], docs[(r["dataset"], r["uid"])], r["finish_reason"]))
    grading = Path(__file__).resolve().parents[1] / "kvp" / "grading.py"
    return hashlib.sha256(grading.read_bytes()).hexdigest(), docs


def load_raw(run: Path):
    gen, nll, metas = [], [], []
    for f in sorted((run / "raw").glob("*.jsonl")):
        lines = f.read_text().splitlines()
        metas.append(json.loads(lines[0])["job_meta"])
        for ln in lines[1:]:
            r = json.loads(ln)
            if r.get("batching", "main") == "alt":
                r["policy"] = "full_altbatch"
            (gen if r["task"] == "gen" else nll).append(r)
    return gen, nll, metas


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(values: np.ndarray, rng: np.random.Generator, stat=np.mean):
    idx = rng.integers(0, len(values), size=(BOOT, len(values)))
    draws = stat(values[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _cell(df, m, d, cell):
    p, mo = cell
    return df[(df.model == m) & (df.dataset == d) & (df.policy == p) & (df["mode"] == mo)].set_index("uid")


def paired_row(g, n, m, d, base_cell, cell, rng):
    base, cur = _cell(g, m, d, base_cell), _cell(g, m, d, cell)
    common = base.index.intersection(cur.index)
    if not len(common):
        return None
    a = base.loc[common].correct.astype(int).to_numpy()
    b = cur.loc[common].correct.astype(int).to_numpy()
    diff = b - a
    lo, hi = boot_ci(diff.astype(float), rng)
    only_base, only_pol = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    row = {"model": m, "dataset": d, "baseline": "/".join(base_cell), "policy": cell[0], "mode": cell[1],
           "n_paired": len(common), "acc_base": a.mean(), "acc_policy": b.mean(), "delta_acc": diff.mean(),
           "delta_acc_ci95_lo": lo, "delta_acc_ci95_hi": hi, "base_only_correct": only_base,
           "policy_only_correct": only_pol, "mcnemar_exact_p": mcnemar_exact(only_base, only_pol)}
    nbase, ncur = _cell(n, m, d, base_cell), _cell(n, m, d, cell)
    nc = nbase.index.intersection(ncur.index)
    if len(nc):
        sb, sc = nbase.loc[nc].sum_logprob.to_numpy(), ncur.loc[nc].sum_logprob.to_numpy()
        tt = nbase.loc[nc].target_tokens.to_numpy()
        idx = rng.integers(0, len(nc), size=(BOOT, len(nc)))
        draws = (sb[idx].sum(1) - sc[idx].sum(1)) / tt[idx].sum(1)
        per = ncur.loc[nc].nll_mean.to_numpy() - nbase.loc[nc].nll_mean.to_numpy()
        plo, phi = boot_ci(per, rng)
        row.update({"nll_base_token_weighted": float(-sb.sum() / tt.sum()),
                    "nll_policy_token_weighted": float(-sc.sum() / tt.sum()),
                    "delta_nll_token_weighted": float((sb.sum() - sc.sum()) / tt.sum()),
                    "delta_nll_tw_ci95_lo": float(np.percentile(draws, 2.5)),
                    "delta_nll_tw_ci95_hi": float(np.percentile(draws, 97.5)),
                    "delta_nll_per_problem_mean": float(per.mean()),
                    "delta_nll_pp_ci95_lo": plo, "delta_nll_pp_ci95_hi": phi,
                    "frac_problems_nll_worse": float((per > 0).mean())})
    return row


# ----------------------------------------------------------------------------- degeneration analysis
STOP_RULE = "\n\n\n"  # secondary, policy-independent stop rule applied post hoc to every cell
NLL_BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 100000)]


def degenerate(raw: str) -> bool:
    """Tail is a long run of newlines or a short loop (<= 3 distinct non-blank lines in the last 400 chars)."""
    tail = raw[-400:]
    if "\n" * 16 in tail:
        return True
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    return len(tail) >= 400 and len(lines) >= 8 and len(set(lines)) <= 3


def degeneration_tables(gen, nll, docs, rng):
    from kvp.grading import grade

    rows, per_item = [], {}
    for r in gen:
        text = r["answer_text"]
        cut = text.find(STOP_RULE)
        if cut >= 0:
            g = grade(r["dataset"], text[:cut], docs[(r["dataset"], r["uid"])], "stop_string")
        else:
            g = {"correct": r["correct"]}
        key = (r["model"], r["dataset"], r["policy"], r["mode"])
        per_item.setdefault(key, {})[r["uid"]] = (bool(r["correct"]), bool(g["correct"]),
                                                   degenerate(r["raw_generation"]), r["finish_reason"])
    for key, items in sorted(per_item.items()):
        main_c = np.array([v[0] for v in items.values()])
        stop_c = np.array([v[1] for v in items.values()])
        deg = np.array([v[2] for v in items.values()])
        length = np.array([v[3] == "length" for v in items.values()])
        row = {"model": key[0], "dataset": key[1], "policy": key[2], "mode": key[3], "n": len(items),
               "degenerate_rate": deg.mean(), "hit_max_tokens_rate": length.mean(),
               "acc_main": main_c.mean(), "acc_stop_rule": stop_c.mean(),
               "answered_then_degenerated": int((deg & stop_c).sum()),
               "degenerated_without_correct_answer": int((deg & ~stop_c).sum())}
        base = per_item.get((key[0], key[1], "full", "decode"))
        if base is not None and key[2:] != ("full", "decode"):
            common = sorted(set(base) & set(items))
            d = np.array([items[u][1] for u in common], float) - np.array([base[u][1] for u in common], float)
            lo, hi = boot_ci(d, rng)
            row.update({"delta_acc_stop_rule_vs_full": d.mean(), "ci95_lo": lo, "ci95_hi": hi})
        rows.append(row)
    deg_df = pd.DataFrame(rows)

    # teacher-forced NLL by target position bucket, paired vs FULL/decode on identical gold tokens
    lps = {}
    for r in nll:
        lps[(r["model"], r["dataset"], r["policy"], r["mode"], r["uid"])] = np.array(r["token_logprobs"])
    brows = []
    cells = sorted({k[:4] for k in lps})
    for cell in cells:
        if cell[2:] == ("full", "decode"):
            continue
        for lo_b, hi_b in NLL_BUCKETS:
            diffs, ntok = [], 0
            for k, v in lps.items():
                if k[:4] != cell:
                    continue
                b = lps.get((cell[0], cell[1], "full", "decode", k[4]))
                if b is None or len(v) <= lo_b:
                    continue
                seg = slice(lo_b, min(hi_b, len(v)))
                diffs.append((-(v[seg]).sum() + b[seg].sum(), seg.stop - seg.start))
                ntok += seg.stop - seg.start
            if not diffs:
                continue
            dn = np.array([x[0] for x in diffs])
            tn = np.array([x[1] for x in diffs])
            idx = rng.integers(0, len(dn), size=(BOOT, len(dn)))
            draws = dn[idx].sum(1) / tn[idx].sum(1)
            brows.append({"model": cell[0], "dataset": cell[1], "policy": cell[2], "mode": cell[3],
                          "positions": f"{lo_b}-{hi_b - 1}", "n_problems": len(dn), "n_tokens": ntok,
                          "delta_nll_per_token_vs_full": dn.sum() / tn.sum(),
                          "ci95_lo": float(np.percentile(draws, 2.5)), "ci95_hi": float(np.percentile(draws, 97.5))})
    return deg_df, pd.DataFrame(brows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    gen, nll, metas = load_raw(run)
    grader_sha, docs = regrade(gen)

    keep_gen = ["model", "dataset", "policy", "mode", "uid", "batch", "batch_pos", "prompt_tokens",
                "n_gen_tokens", "finish_reason", "correct", "correct_format_only", "parse_ok", "answer_source",
                "failure_reason", "extract_method",
                "pred", "gold", "lm_eval_strict_match", "lm_eval_flexible_extract", "math_verify",
                "minerva_is_equiv", "normalized_string_equal"]
    g = pd.DataFrame([{k: r.get(k) for k in keep_gen} for r in gen])
    n = pd.DataFrame([{k: r.get(k) for k in ("model", "dataset", "policy", "mode", "uid", "target_tokens",
                                              "sum_logprob", "nll_mean", "target_tokenization")} for r in nll])
    g.to_csv(out / "items_gen.csv", index=False)
    n.to_csv(out / "items_nll.csv", index=False)

    # ---------------- completeness
    comp = {"grader_sha256": grader_sha, "n_gen_records": len(g), "n_nll_records": len(n), "jobs_finished": len(metas),
            "failed_jobs": sorted(p.name for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []}
    for name, df in (("gen", g), ("nll", n)):
        cnt = df.groupby(["model", "dataset", "policy", "mode"]).uid.agg(["count", "nunique"]).reset_index()
        comp[f"{name}_cells"] = cnt.to_dict("records")
        comp[f"{name}_incomplete_cells"] = cnt[(cnt["count"] != 200) | (cnt["nunique"] != 200)].to_dict("records")
    (out / "completeness.json").write_text(json.dumps(comp, indent=1, default=str))

    # ---------------- per-cell summary
    rows = []
    for (m, d, p, mo), df in g.groupby(["model", "dataset", "policy", "mode"]):
        nd = n[(n.model == m) & (n.dataset == d) & (n.policy == p) & (n["mode"] == mo)]
        row = {"model": m, "dataset": d, "policy": p, "mode": mo, "n": len(df),
               "accuracy": df.correct.mean(), "accuracy_format_only": df.correct_format_only.mean(),
               "fallback_answer_rate": (df.answer_source == "fallback").mean(),
               "parse_fail_rate": 1 - df.parse_ok.mean(),
               "fail_truncated_no_answer": int((df.failure_reason == "truncated_no_answer").sum()),
               "fail_finished_no_answer": int((df.failure_reason == "finished_no_answer").sum()),
               "fail_unparseable_answer": int((df.failure_reason == "unparseable_answer").sum()),
               "wrong_but_parsed": int(((~df.correct.astype(bool)) & df.parse_ok.astype(bool)).sum()),
               "parse_failures": int((~df.parse_ok.astype(bool)).sum()),
               "mean_gen_tokens": df.n_gen_tokens.mean(),
               "hit_max_tokens_rate": (df.finish_reason == "length").mean(),
               "nll_token_weighted": float(-nd.sum_logprob.sum() / nd.target_tokens.sum()) if len(nd) else None,
               "nll_mean_per_problem": float(nd.nll_mean.mean()) if len(nd) else None,
               "nll_n": len(nd), "target_tokens": int(nd.target_tokens.sum()) if len(nd) else None}
        if d == "gsm8k":
            row["lm_eval_strict_match"] = df.lm_eval_strict_match.mean()
            row["lm_eval_flexible_extract"] = df.lm_eval_flexible_extract.mean()
        else:
            row["minerva_is_equiv"] = df.minerva_is_equiv.fillna(False).astype(bool).mean()
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ["_o"] = summ.apply(lambda r: ORDER.index((r.policy, r["mode"])), axis=1)
    summ = summ.sort_values(["model", "dataset", "_o"]).drop(columns="_o")
    summ.to_csv(out / "summary.csv", index=False)

    # ---------------- paired deltas vs baselines
    rng = np.random.default_rng(SEED)
    comparisons = [(("full", "decode"), c) for c in ORDER[1:]] + [
        (STRICT_BASE, ("last", "strict")), (STRICT_BASE, ("average", "strict"))]
    pair_rows = []
    for (m, d), _ in g.groupby(["model", "dataset"]):
        for base_cell, cell in comparisons:
            row = paired_row(g, n, m, d, base_cell, cell, rng)
            if row is not None:
                pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "paired_vs_full.csv", index=False)

    # generated-token divergence vs FULL(decode)
    txt = {(r["model"], r["dataset"], r["policy"], r["mode"], r["uid"]): r["gen_token_ids"] for r in gen}
    div = []
    for (m, d, p, mo, u), ids in txt.items():
        base = txt.get((m, d, "full", "decode", u))
        if (p, mo) == ("full", "decode") or base is None:
            continue
        k = next((i for i, (x, y) in enumerate(zip(base, ids)) if x != y), min(len(base), len(ids)))
        div.append({"model": m, "dataset": d, "policy": p, "mode": mo, "identical": base == ids, "first_div": k})
    dv = pd.DataFrame(div)
    if len(dv):
        agg = dv.groupby(["model", "dataset", "policy", "mode"]).agg(
            identical_rate=("identical", "mean"), q25_first_divergence=("first_div", lambda x: x.quantile(.25)),
            median_first_divergence=("first_div", "median"), q75_first_divergence=("first_div", lambda x: x.quantile(.75)),
            frac_diverged_within_16=("first_div", lambda x: float((x < 16).mean()))).reset_index()
        agg.to_csv(out / "generation_divergence_vs_full.csv", index=False)

    # ---------------- scorer audit
    audit = {"gsm8k_primary_vs_lm_eval_strict": [], "math_primary_vs_minerva": [], "parse_failure_samples": []}
    rnd = random.Random(SEED)
    for r in gen:
        if r["dataset"] == "gsm8k" and bool(r["correct"]) != bool(r["lm_eval_strict_match"]):
            audit["gsm8k_primary_vs_lm_eval_strict"].append(
                {k: r[k] for k in ("model", "policy", "mode", "uid", "pred", "pred_raw", "gold", "correct",
                                   "lm_eval_strict_match")} | {"tail": r["answer_text"][-200:]})
        if r["dataset"] == "math500" and r["parse_ok"] and bool(r["correct"]) != bool(r.get("minerva_is_equiv")):
            audit["math_primary_vs_minerva"].append(
                {k: r.get(k) for k in ("model", "policy", "mode", "uid", "pred", "gold", "correct",
                                       "math_verify", "minerva_is_equiv", "normalized_string_equal")})
    fb = [r for r in gen if r.get("answer_source") == "fallback"]
    audit["fallback_samples"] = [
        {k: r.get(k) for k in ("model", "dataset", "policy", "mode", "uid", "extract_method", "pred", "gold",
                               "correct", "finish_reason")} | {"last_line": r["answer_text"].strip().splitlines()[-1][-250:]}
        for r in rnd.sample(fb, min(40, len(fb)))]
    fails = [r for r in gen if not r["parse_ok"]]
    by_cell = defaultdict(list)
    for r in fails:
        by_cell[(r["model"], r["dataset"], r["policy"], r["mode"])].append(r)
    for cell, rs in sorted(by_cell.items()):
        for r in rnd.sample(rs, min(3, len(rs))):
            audit["parse_failure_samples"].append(
                {"cell": list(cell), "uid": r["uid"], "finish_reason": r["finish_reason"],
                 "failure_reason": r["failure_reason"], "n_gen_tokens": r["n_gen_tokens"],
                 "answer_tail": r["answer_text"][-300:]})
    audit["counts"] = {k: len(v) for k, v in audit.items() if isinstance(v, list)}
    audit["parse_failure_reasons"] = {f"{k[0]}|{k[1]}|{k[2]}-{k[3]}": dict(Counter(r["failure_reason"] for r in v))
                                      for k, v in by_cell.items()}
    (out / "scorer_audit.json").write_text(json.dumps(audit, indent=1, default=str))

    # ---------------- degeneration / error-accumulation diagnostics
    deg_df, pos_df = degeneration_tables(gen, nll, docs, rng)
    deg_df.to_csv(out / "degeneration.csv", index=False)
    pos_df.to_csv(out / "nll_by_position_vs_full.csv", index=False)

    # ---------------- markdown
    def fmt(x, nd=3):
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    md = [f"Grader: kvp/grading.py sha256 {grader_sha[:12]} (all generations re-scored at analysis time)\n",
          "## Per-cell summary\n", "| model | dataset | policy | mode | n | acc | acc (format only) | fallback | "
          "no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |",
          "|" + "---|" * 14]
    for _, r in summ.iterrows():
        ref = r.get("lm_eval_strict_match") if r.dataset == "gsm8k" else r.get("minerva_is_equiv")
        md.append(f"| {r.model} | {r.dataset} | {r.policy} | {r['mode']} | {r.n} | {fmt(r.accuracy)} | "
                  f"{fmt(r.accuracy_format_only)} | {fmt(r.fallback_answer_rate)} | "
                  f"{r.fail_truncated_no_answer} / {r.fail_finished_no_answer} | {fmt(ref)} | {fmt(r.nll_token_weighted, 4)} | "
                  f"{fmt(r.nll_mean_per_problem, 4)} | {fmt(r.mean_gen_tokens, 1)} | {fmt(r.hit_max_tokens_rate)} |")
    md += ["\n## Paired deltas vs baseline\n",
           "| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | "
           "ΔNLL tok-w [95% CI] |", "|" + "---|" * 9]
    for _, r in pairs.iterrows():
        dn = (f"{r.delta_nll_token_weighted:+.4f} [{r.delta_nll_tw_ci95_lo:+.4f}, {r.delta_nll_tw_ci95_hi:+.4f}]"
              if "delta_nll_token_weighted" in r and not pd.isna(r.get("delta_nll_token_weighted")) else "")
        md.append(f"| {r.model} | {r.dataset} | {r.baseline} | {r.policy} | {r['mode']} | {r.delta_acc:+.3f} "
                  f"[{r.delta_acc_ci95_lo:+.3f}, {r.delta_acc_ci95_hi:+.3f}] | {r.base_only_correct} / "
                  f"{r.policy_only_correct} | {r.mcnemar_exact_p:.3g} | {dn} |")
    md += ["\n## Degeneration and a policy-independent stop rule (secondary)\n",
           f"Stop rule: cut every generation at the first {STOP_RULE!r} and re-score (identical for all cells; "
           "greedy decoding makes this equal to stopping online). Main accuracy is unchanged.\n",
           "| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | "
           "Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |", "|" + "---|" * 11]
    for _, r in deg_df.iterrows():
        dl = "" if pd.isna(r.get("delta_acc_stop_rule_vs_full")) else \
            f"{r.delta_acc_stop_rule_vs_full:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]"
        md.append(f"| {r.model} | {r.dataset} | {r.policy} | {r['mode']} | {r.degenerate_rate:.3f} | "
                  f"{r.hit_max_tokens_rate:.3f} | {r.acc_main:.3f} | {r.acc_stop_rule:.3f} | {dl} | "
                  f"{r.answered_then_degenerated} | {r.degenerated_without_correct_answer} |")
    md += ["\n## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position\n",
           "| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |", "|" + "---|" * 7]
    for _, r in pos_df.iterrows():
        md.append(f"| {r.model} | {r.dataset} | {r.policy} | {r['mode']} | {r.positions} | {r.n_problems} | "
                  f"{r.delta_nll_per_token_vs_full:+.4f} [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps({k: v for k, v in comp.items() if "incomplete" in k or k.startswith("n_") or k == "failed_jobs"},
                     default=str)[:2000])


if __name__ == "__main__":
    main()
