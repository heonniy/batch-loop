"""Stage 1 report: official-protocol audit + native parity gate -> analysis/stage1_parity.md.

Gate criteria (fixed in code before any Stage-1 GPU result was inspected):
  FP32 parity PASSES iff, on every case,
    (1) engine-FP32 vs FP64 max |logit error| <= 2 x native-FP32 vs FP64 max |logit error| + 1e-4
        (the engine is no less accurate than the native model's own FP32 execution),
    (2) engine vs native FP32: no top-1 disagreement where the FP64 top-2 margin exceeds 1e-3,
    (3) engine vs native FP32: max KL <= 1e-5 nats.
  BF16 disagreement is quantified (first divergences, teacher-forced KL / ΔNLL / top-1 agreement,
  batch-shape noise) but has no pass threshold.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path

GATE_KL = 1e-5
GATE_MARGIN = 1e-3


def load_samples(path_glob: str) -> list[dict]:
    rows = []
    for f in glob.glob(path_glob, recursive=True):
        rows += [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
    return rows


def official_section(run: Path, base: Path, tok) -> tuple[list[str], dict]:
    md, manifest = ["## 1.1 Official Huginn lm-eval GSM8K (separate from the project runner)", ""], {}
    proj = {}
    for f in glob.glob(str(base / "raw" / "huginn-0125__gsm8k__gen__full-decode__b*.jsonl")):
        for line in Path(f).read_text().splitlines()[1:]:
            r = json.loads(line)
            proj[int(r["uid"].split("-")[-1])] = r
    prompts = {}
    for line in (base / "prompts" / "huginn-0125__gsm8k.jsonl").read_text().splitlines()[1:]:
        r = json.loads(line)
        prompts[int(r["uid"].split("-")[-1])] = r
    md += ["| variant | items | strict-match | flexible-extract | on frozen-200: strict | flexible | project FULL primary | project FULL lm-eval strict / flexible |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for var in ("A_b1_full",):
        res = glob.glob(str(run / "official" / var / "**" / "results_*.json"), recursive=True)
        smp = load_samples(str(run / "official" / var / "**" / "samples_gsm8k_cot_*.jsonl"))
        if not res or not smp:
            md.append(f"| {var} | (missing) | | | | | |")
            continue
        r = json.loads(Path(res[0]).read_text())
        n_tok = [len(tok.encode(per_s["resps"][0][0], add_special_tokens=False)) for per_s in smp
                 if per_s["filter"] == "strict-match"]
        manifest_cap = {"n": len(n_tok), "ge_256_tokens": sum(n >= 256 for n in n_tok)}
        manifest[var] = {"generation_length": manifest_cap, "results_file": res[0], "config": r.get("config"), "versions": r.get("versions"),
                         "n-shot": r.get("n-shot"), "git_hash": r.get("git_hash"),
                         "date": r.get("date"), "task_hashes": r.get("task_hashes"),
                         "model_args": r.get("config", {}).get("model_args"),
                         "batch_sizes": r.get("config", {}).get("batch_sizes"),
                         "random_seed": r.get("config", {}).get("random_seed"),
                         "torch_seed": r.get("config", {}).get("torch_seed")}
        res_t = r["results"]["gsm8k_cot"]
        by = {s["doc_id"]: s for s in smp}
        sub = [i for i in proj if i in by]
        strict = [by[i]["exact_match"] if "exact_match" in by[i] else None for i in sub]
        # lm-eval writes one sample row per filter; collect both filters per doc
        per = {}
        for s in smp:
            per.setdefault(s["doc_id"], {})[s["filter"]] = s
        sm = [per[i]["strict-match"]["exact_match"] for i in sub if "strict-match" in per.get(i, {})]
        fx = [per[i]["flexible-extract"]["exact_match"] for i in sub if "flexible-extract" in per.get(i, {})]
        pj = [proj[i]["correct"] for i in sub]
        pjs = [bool(proj[i]["lm_eval_strict_match"]) for i in sub]
        pjf = [bool(proj[i]["lm_eval_flexible_extract"]) for i in sub]
        md.append(f"| {var} | {len(per)} | {res_t.get('exact_match,strict-match', float('nan')):.4f} | "
                  f"{res_t.get('exact_match,flexible-extract', float('nan')):.4f} | "
                  f"{(sum(sm) / len(sm)) if sm else float('nan'):.3f} | {(sum(fx) / len(fx)) if fx else float('nan'):.3f} | "
                  f"{(sum(pj) / len(pj)) if pj else float('nan'):.3f} (n={len(pj)}) | "
                  f"{sum(pjs) / len(pjs):.3f} / {sum(pjf) / len(pjf):.3f} |")
        # prompt-token-ID identity on the frozen subset
        same_text = same_ids = 0
        diffs = []
        for i in sub:
            s = per[i].get("strict-match") or next(iter(per[i].values()))
            text = s["arguments"]["gen_args_0"]["arg_0"] if isinstance(s["arguments"], dict) else s["arguments"][0][0]
            p = prompts[i]
            if text == p["prompt"]:
                same_text += 1
            ids = tok.encode(text, add_special_tokens=False)
            if ids == p["prompt_ids"]:
                same_ids += 1
            elif len(diffs) < 3:
                diffs.append(i)
        manifest[var]["prompt_identity_on_subset"] = {"n": len(sub), "same_text": same_text, "same_token_ids": same_ids,
                                                      "first_mismatching_doc_ids": diffs}
        if sub and "strict-match" in per[sub[0]]:
            both = [(bool(per[i]["strict-match"]["exact_match"]), bool(proj[i]["lm_eval_strict_match"])) for i in sub]
            manifest[var]["paired_vs_project_full"] = {
                "official_only": sum(a and not b for a, b in both), "project_only": sum(b and not a for a, b in both),
                "both": sum(a and b for a, b in both), "neither": sum((not a) and (not b) for a, b in both)}
    md.append("")
    for var, m in manifest.items():
        pi = m.get("prompt_identity_on_subset", {})
        md.append(f"- {var}: batch_sizes={m.get('batch_sizes')}, seeds random/torch={m.get('random_seed')}/"
                  f"{m.get('torch_seed')}; prompt text identical {pi.get('same_text')}/{pi.get('n')}, "
                  f"prompt token IDs identical {pi.get('same_token_ids')}/{pi.get('n')}; paired strict-match, official vs "
                  f"project FULL (lm-eval strict on project outputs): {m.get('paired_vs_project_full')}; generation length: {m.get('generation_length')}")
    md += ["", "Deviations from the README command (scripts/oracle/s1_official_eval.sh): revision pinned "
           "(== HF main), transformers 4.53.3 (4.54.1 breaks the remote HuginnDynamicCache), batch_size=1 "
           "(README `auto` chose 64 and OOMed, fixed 16 OOMed; failed logs kept under official/logs/).", "",
           "Protocol differences vs the project runner that remain by construction: official cap = lm-eval "
           "default max_gen_toks 256 (project 512); official initial latent = model `randn` under lm-eval torch "
           "seed 1234, consumed sequentially per rank (project: per-(uid, position) seeded draw); native dynamic "
           "cache vs project engine; project primary GSM8K scorer adds a last-number fallback for untruncated "
           "outputs (official strict-match does not; official flexible-extract takes the last number even when "
           "truncated).", ""]
    return md, manifest


def fmt(x, f=".2e"):
    return "—" if x is None else format(x, f)


def parity_section(run: Path) -> tuple[list[str], dict]:
    raw = run / "parity" / "raw"
    md, verdict = [], {}
    fp_path = raw / "fp_parity.json"
    md += ["## 1.2 FP32 parity: engine FULL vs native HF (FP64 native no-cache = ground truth)", ""]
    if fp_path.exists():
        fp = json.loads(fp_path.read_text())
        md += ["| case | P+cont | native32 vs 64: max|Δ| | engine32 vs 64: max|Δ| | native-cache32 vs 64: max|Δ| | "
               "engine vs native32: max|Δ| | KL max | top-1 disagree (margin>1e-3) | first-token top-1 |",
               "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        ok = True
        for r in fp["fp_parity"]:
            n, e, c, en = (r["native_fp32_vs_fp64"], r["engine_fp32_vs_fp64"], r["native_cached_fp32_vs_fp64"],
                           r["engine_fp32_vs_native_fp32"])
            c1 = e["max_abs_logit_diff"] <= 2 * n["max_abs_logit_diff"] + 1e-4
            c2 = en["n_top1_disagree_margin_gt_1e-3"] == 0
            c3 = en["kl_max"] <= GATE_KL
            ok &= c1 and c2 and c3
            md.append(f"| {r['dataset']} {r['uid']} | {r['P']}+{r['n_cont']} | {fmt(n['max_abs_logit_diff'])} | "
                      f"{fmt(e['max_abs_logit_diff'])} | {fmt(c['max_abs_logit_diff'])} | "
                      f"{fmt(en['max_abs_logit_diff'])} | {fmt(en['kl_max'])} | {en['n_top1_disagree']} "
                      f"({en['n_top1_disagree_margin_gt_1e-3']}) | {en['first_token_top1_agree']} |")
        md += ["", "Engine inside a left-padded B4 batch, FP32 (teacher-forced gold logprobs):", ""]
        for r in fp["fp_parity"]:
            if "engine_b4_vs_b1_fp32_logprob_max_abs_diff" in r:
                md.append(f"- {r['uid']}: max|Δlogprob| B4 vs FP64 {fmt(r['engine_b4_fp32_logprob_max_abs_diff_vs_fp64'])}, "
                          f"B1 vs FP64 {fmt(r['engine_b1_fp32_logprob_max_abs_diff_vs_fp64'])}, "
                          f"B4 vs B1 {fmt(r['engine_b4_vs_b1_fp32_logprob_max_abs_diff'])}")
        md += ["", "Initial-latent sensitivity of the native model, FP32 (vs paired engine latents):", ""]
        for r in fp["latent_init_sensitivity_fp32"]:
            a, z = r["paired_vs_alt_seed"], r["paired_vs_model_randn_init"]
            md.append(f"- {r['case']}: independent seeded draw: max|Δlogit| {fmt(a['max_abs_logit_diff'])}, KL max "
                      f"{fmt(a['kl_max'])}, top-1 agree {a['top1_agreement']:.4f}; model randn init: max|Δlogit| "
                      f"{fmt(z['max_abs_logit_diff'])}, KL max {fmt(z['kl_max'])}, top-1 agree {z['top1_agreement']:.4f}")
        verdict["fp32_parity_pass"] = bool(ok)
        md += ["", f"**FP32 gate: {'PASS' if ok else 'FAIL'}** (criteria in the header of scripts/oracle/s1_report.py).", ""]
    else:
        md += ["(fp_parity job missing)", ""]
        verdict["fp32_parity_pass"] = None

    md += ["## 1.3 BF16 paired greedy generations + teacher-forced comparison", ""]
    greedy = [json.loads(p.read_text()) for p in sorted(raw.glob("greedy__*.json"))]
    rows = []
    for g in greedy:
        tf = g["teacher_forced_on_native_sequence"]
        n = len(tf["kl"])
        rows.append({
            "dataset": g["job"]["dataset"], "uid": g["job"]["uid"], "P": g["P"],
            "native_n": g["native"]["n_tokens"], "native_finish": g["native"]["finish"], "native_correct": g["native"]["correct"],
            "engine_b1_n": g["engine_b1"]["n_tokens"], "engine_b1_finish": g["engine_b1"]["finish"],
            "engine_b1_correct": g["engine_b1"]["correct"],
            "engine_batched_n": g["engine_batched_r32_run"]["n_tokens"],
            "engine_batched_correct": g["engine_batched_r32_run"]["correct"],
            "div_native_vs_engine_b1": g["first_divergence"]["native_vs_engine_b1"],
            "div_native_vs_engine_batched": g["first_divergence"]["native_vs_engine_batched"],
            "div_engine_b1_vs_batched": g["first_divergence"]["engine_b1_vs_engine_batched"],
            "first_token_top1_agree": g["first_token_logits_engine_vs_native"]["top1_agreement"] == 1.0,
            "tf_positions": n, "tf_top1_agreement": sum(tf["top1_agree"]) / n,
            "tf_first_top1_disagreement": tf["first_top1_disagreement"],
            "tf_kl_mean": sum(tf["kl"]) / n, "tf_kl_max": max(tf["kl"]),
            "tf_nll_native_mean": sum(tf["nll_native"]) / n, "tf_nll_engine_mean": sum(tf["nll_engine"]) / n,
            "tf_norm_err": tf["normalization_max_err"], "tf_nan": tf["any_nan"],
            "native_greedy_s": g["wall_s"]["native_greedy"]})
    if rows:
        out = run / "analysis"
        out.mkdir(exist_ok=True)
        with open(out / "stage1_greedy_pairs.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        md += ["| dataset | n | identical native vs engine B1 | identical engine B1 vs batched | acc native | acc engine B1 | "
               "acc engine batched | first-token top-1 agree | TF top-1 agree (mean) | TF KL mean | TF ΔNLL eng−nat (mean) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for ds in ("gsm8k", "math500"):
            g = [r for r in rows if r["dataset"] == ds]
            if not g:
                continue
            md.append(f"| {ds} | {len(g)} | {sum(r['div_native_vs_engine_b1'] is None for r in g)} | "
                      f"{sum(r['div_engine_b1_vs_batched'] is None for r in g)} | "
                      f"{sum(r['native_correct'] for r in g)}/{len(g)} | {sum(r['engine_b1_correct'] for r in g)}/{len(g)} | "
                      f"{sum(r['engine_batched_correct'] for r in g)}/{len(g)} | {sum(r['first_token_top1_agree'] for r in g)}/{len(g)} | "
                      f"{st.mean(r['tf_top1_agreement'] for r in g):.4f} | {st.mean(r['tf_kl_mean'] for r in g):.2e} | "
                      f"{st.mean(r['tf_nll_engine_mean'] - r['tf_nll_native_mean'] for r in g):+.2e} |")
        md += ["", "First greedy divergence index (native vs engine B1), per item: " + ", ".join(
            f"{r['uid'].split('/')[-1]}={r['div_native_vs_engine_b1']}" for r in rows), "",
            f"NaN in any teacher-forced distribution: {any(r['tf_nan'] for r in rows)}; "
            f"max |sum p − 1|: {max(r['tf_norm_err'] for r in rows):.1e}", ""]
        verdict["bf16_greedy_items"] = len(rows)
    batch = [json.loads(p.read_text()) for p in sorted(raw.glob("batch__*.json"))]
    if batch:
        rr = [r for b in batch for r in b["rows"]]
        md += ["### BF16 batch-shape noise control (engine FULL, B4 vs B1, R32 FULL continuation)", "",
               f"{len(rr)} items: max |Δlogprob| median {st.median(r['max_abs_logprob_diff'] for r in rr):.3f}, "
               f"max {max(r['max_abs_logprob_diff'] for r in rr):.3f}; mean |Δlogprob| mean "
               f"{st.mean(r['mean_abs_logprob_diff'] for r in rr):.2e}; |Δ mean NLL| max "
               f"{max(abs(r['mean_nll_b1'] - r['mean_nll_b4']) for r in rr):.2e}", ""]
    return md, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-run", default="results/20261023T060125Z_huginn_r32_h200x8")
    args = ap.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kvp.models import load  # noqa: F401
    from kvp.specs import MODELS
    from transformers import AutoTokenizer

    spec = MODELS["huginn-0125"]
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, trust_remote_code=True)
    run, base = Path(args.run_dir), Path(args.base_run)
    md = ["# Huginn R32 Stage 1 — native correctness / official evaluation audit", "",
          f"Run: `{run}`; base quality run: `{base}`. Plan: `kv_policy/HUGINN_R32_KV_ORACLE_PLAN.md`.", ""]
    om, manifest = official_section(run, base, tok)
    pm, verdict = parity_section(run)
    audit = run / "analysis" / "stage1_item_audit.md"
    md += om + pm + ["## 1.4 Item audit", "", f"See `analysis/stage1_item_audit.md` / `.csv`.", ""]
    if audit.exists():
        md += audit.read_text().splitlines()[4:13] + [""]
    md += ["## Incidents", "",
           "- A first FP32-parity pass used a GPU float64 forward as ground truth and showed |Δlogit| ≈ 10 for "
           "both the native and the engine FP32 paths. Root cause: CUDA float64 softmax on this torch build "
           "(2.6.0+cu124, H200) is wrong for some row lengths (S=837: rows ≥ 31 off by up to 0.2; "
           "`repro_cuda_fp64_softmax.txt`, `scripts/oracle/repro_cuda_fp64_softmax.py`). CUDA FP32/BF16 softmax and "
           "SDPA agree with CPU float64, so the BF16/FP32 experiment paths are unaffected. The FP64 reference is now "
           "computed on CPU; the invalid result is kept as `parity/superseded/fp_parity_gpu_fp64_INVALID.json`.",
           "- Official lm-eval needed transformers 4.53.3 and batch_size=1 (see 1.1).", "",
           "## Gate", "", json.dumps(verdict), ""]
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "stage1_parity.md").write_text("\n".join(md) + "\n")
    (run / "official_eval_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
