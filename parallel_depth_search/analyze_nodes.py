"""Per-node metric aggregation from REAL raw artifacts under <run>/results/<NODE>/ (CPU).

usage: python -m parallel_depth_search.analyze_nodes --run-dir RUN --node NODE --out metrics.json
Diagnostic nodes keep teacher-forced/oracle labels; only e2e_decoder records can carry
`full_e2e_measured`. Bootstrap CIs resample whole cohorts for throughput (rows of one cohort share
wall time) and prompts (paired) for accuracy.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

from parallel_depth_poc.summarize_gpu import diag_cell, mean, q

MODELS = ("huginn-3.5b", "ouro-1.4b", "ouro-2.6b")


# ---------------------------------------------------------------- G0 / G1
def g0_native(res):
    out = dict(models_verified=[], fp32_cells={}, varengine={}, bf16_batch_order={})
    fp32_ok = True
    for m in MODELS:
        e0 = res / f"e0_{m}.json"
        nat = res / f"native_{m}.json"
        if not e0.exists() or not nat.exists():
            fp32_ok = False
            continue
        cells = json.loads(e0.read_text())["cells"]
        ok_m = all(c["greedy_agree"] == 1.0 and c["max_abs_logit_diff"] <= 1e-2 for c in cells.values())
        out["fp32_cells"][m] = {k: dict(greedy_agree=c["greedy_agree"], max_abs_logit_diff=c["max_abs_logit_diff"],
                                        positions=c["positions"]) for k, c in cells.items()}
        n = json.loads(nat.read_text())
        ve = n["varengine_vs_depthengine_fp32"]
        out["varengine"][m] = ve
        out["bf16_batch_order"][m] = {D: {k: {kk: vv for kk, vv in v.items() if kk != "first_mismatch"}
                                          for k, v in c.items()} for D, c in n["bf16_batch_order"]["cells"].items()}
        ok_m = ok_m and ve["tokens_identical"] and ve["max_abs_logit_diff"] <= 1e-2
        fp32_ok &= ok_m
        if ok_m:
            out["models_verified"].append(m)
    out.update(native_parity_pass=fp32_ok and len(out["models_verified"]) == 3, fp32_native_parity=fp32_ok,
               bf16_batch_order_tested=all(m in out["bf16_batch_order"] for m in MODELS),
               sample_count=sum(c["positions"] for m in out["fp32_cells"].values() for c in m.values()),
               parity_tolerance="FP32 greedy agreement 1.0 and max|dlogit|<=1e-2 vs HF-native forward and PoC engine")
    return out


def g0_kv(res):
    cells, gates = {}, []
    for f in sorted(res.glob("kv_*.json")):
        r = json.loads(f.read_text())["kv"]
        for D, c in r["cells"].items():
            key = f"{f.stem[3:]}_{D}"
            cells[key] = dict(gates=c["gates"], **{m: {k: c[m][k] for k in ("seq_exact_rate", "tokens_identical",
                          "logits_bitwise", "max_abs_logit_diff", "bank_max_rel_diff", "rounds_mean")} for m in ("cand", "carry", "jexact")})
            gates.append(c["gates"])
    expected = 2 * len(MODELS)
    ok = len(list(res.glob("kv_*.json"))) == expected and gates and all(all(g.values()) for g in gates)
    allc = list(cells.values())
    return dict(exact_kv_commit_pass=bool(ok), candidate_cache_isolated=all(g["candidate_cache_isolated"] for g in gates),
                coda_cache_parity=all(g["coda_cache_parity"] for g in gates),
                per_depth_bank_parity=all(g["per_depth_bank_parity"] for g in gates),
                native_rollout_match=all(c["cand"]["tokens_identical"] for c in allc),
                carry_seq_exact_rate_min=min(c["carry"]["seq_exact_rate"] for c in allc),
                jexact_seq_exact_rate_min=min(c["jexact"]["seq_exact_rate"] for c in allc),
                cells=cells, sample_count=len(allc) * 8 * 24,
                note="cand = candidate Jacobi rows + candidate coda each token then exact serial commit; "
                     "carry = E5-style candidate KV kept (approximate); jexact = D-1 sweeps, no recompute")


def g1_width(res):
    out = {}
    for f in sorted(res.glob("e1_*.json")):
        d = json.loads(f.read_text())
        out[f.stem[3:]] = [{k: v for k, v in c.items() if k in ("L", "B", "status", "S_oracle", "S_oracle_host",
                                                                 "hbm_peak_alloc_gb", "error")}
                           | {"T_F_ms": {w: t["median_ms"] for w, t in c.get("T_F", {}).items()},
                              "serial_D_ms": c.get("serial_D", {}).get("median_ms"),
                              "coda_ms": c.get("coda", {}).get("median_ms")} for c in d["cells"]]
    n = sum(1 for v in out.values() for c in v if c.get("status") == "ok")
    return dict(cells=out, sample_count=n, oracle_label="per-core oracle upper bound; not a serving claim")


def g1_dynamics(res):
    out = {}
    for cfg in sorted(res.glob("*.config.json")):
        prefix = str(cfg)[: -len(".config.json")]
        if Path(prefix + ".summary.json").exists():
            c = diag_cell(prefix)
            c.pop("_tokens", None)
            out[Path(prefix).name] = c
    return dict(cells=out, sample_count=sum(c["n_rows"] for c in out.values()),
                label="teacher-forced oracle diagnostics (E2 predictors, E3 Jacobi/Parareal, E4 perturbation)")


# ---------------------------------------------------------------- gpu_diagnostic nodes
def diag_jobs(res):
    out = {}
    for f in sorted(res.glob("*.json")):
        if f.name.endswith(".meta.json"):
            continue
        d = json.loads(f.read_text())
        m = d["metadata"]
        recs = [r for r in d["records"] if "rounds" in r]
        if not recs:
            continue
        K = len(recs[0]["rounds"])
        curve = [dict(k=k + 1,
                      top1=mean([r["rounds"][k]["top1_agreement"] for r in recs]),
                      kl=mean([r["rounds"][k]["kl_mean"] for r in recs]),
                      rel=mean([r["rounds"][k]["hidden_rel_l2_mean"] for r in recs]),
                      rows=mean([r["rounds"][k]["core_rows_cumulative"] for r in recs]) / m["B"],
                      solver_ms=mean([sum(x["solver_wall_ms"] + x["coda_wall_ms"] for x in r["rounds"][:k + 1]) for r in recs]))
                 for k in range(K)]
        key = (f"{m['model']}|D{m['depth']}|{m['predictor']}|m{m['warmup']}|{m['corrector']}|"
               f"{m['allocation']}{m['width'] or ''}|a{m['scale']}")
        row = dict(model=m["model"], depth=m["depth"], B=m["B"], predictor=m["predictor"], warmup=m["warmup"],
                   corrector=m["corrector"], allocation=m["allocation"], width=m["width"], scale=m["scale"],
                   n_tokens=len(recs), curve=curve,
                   serial_ref_ms=mean([r["reference_serial_ms"] for r in recs]),
                   warmup_ms=mean([r["warmup_ms"] for r in recs]), predictor_ms=mean([r["predictor_ms"] for r in recs]),
                   commit_mismatch=d["exact_commit_mismatch_count"],
                   rows_per_serial_token=m["depth"])
        jv = [r["jvp_probe"] for r in recs if r.get("jvp_probe")]
        if jv:
            row["jvp"] = {k: [mean([x[k][i] for x in jv]) for i in range(3)]
                          for k in ("gain_q10_50_90", "alpha_opt_q10_50_90", "rel_err_identity_q",
                                    "rel_err_scaled_q", "rel_err_best_scalar_q", "cos_resp_delta_q")}
        out[key] = row
    return dict(cells=out, sample_count=sum(c["n_tokens"] * c["B"] for c in out.values()),
                label="teacher-forced diagnostic (exact AR advances; exact commit each token); not serving speed")


# ---------------------------------------------------------------- e2e
def load_e2e(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def e2e_summary(rows, baseline="AR", n_boot=2000, seed=0):
    rng = random.Random(seed)
    by = collections.defaultdict(dict)          # (file-cell, cohort) -> arm -> [rows]
    for r in rows:
        if r.get("oom"):
            continue
        key = (r["model"], r.get("depth_base", None), r["B"], r["task"], r["split"], r["cohort"], r.get("_cell", ""))
        by[key].setdefault(r["arm"], []).append(r)
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for (model, _, B, task, split, coh, cell), arms in by.items():
        for arm, rs in arms.items():
            cells[(model, B, task, split, cell)][arm].append(rs)
    out = {}
    for (model, B, task, split, cell), arms in cells.items():
        if baseline not in arms:
            continue
        base = {rs[0]["cohort"]: rs for rs in arms[baseline]}
        res = {}
        for arm, cohorts in arms.items():
            co = {rs[0]["cohort"]: rs for rs in cohorts}
            ids = sorted(set(co) & set(base))
            if not ids:
                continue
            def agg(sel):
                wt_c = sum(co[i][0]["wall_s"] for i in sel); wt_b = sum(base[i][0]["wall_s"] for i in sel)
                tk_c = sum(co[i][0]["cohort_tokens"] for i in sel); tk_b = sum(base[i][0]["cohort_tokens"] for i in sel)
                dc_c = sum(co[i][0]["decode_s"] for i in sel); dc_b = sum(base[i][0]["decode_s"] for i in sel)
                return (tk_c / wt_c) / (tk_b / wt_b), (tk_c / dc_c) / (tk_b / dc_b), wt_b / wt_c
            sp, sp_dec, req = agg(ids)
            boots = sorted(agg([rng.choice(ids) for _ in ids])[0] for _ in range(n_boot))
            pr_c = {r["prompt_idx"]: r for i in ids for r in co[i]}
            pr_b = {r["prompt_idx"]: r for i in ids for r in base[i]}
            pids = sorted(set(pr_c) & set(pr_b))
            dacc = [int(pr_c[p]["correct_strict"]) - int(pr_b[p]["correct_strict"]) for p in pids]
            accb = sorted(mean([dacc[rng.randrange(len(dacc))] for _ in dacc]) for _ in range(n_boot))
            flexk = "correct_flex" if "correct_flex" in pr_c[pids[0]] else "correct_boxed"
            dflex = [int(pr_c[p].get(flexk, 0)) - int(pr_b[p].get(flexk, 0)) for p in pids]
            flexb = sorted(mean([dflex[rng.randrange(len(dflex))] for _ in dflex]) for _ in range(n_boot))
            r0 = [co[i][0] for i in ids]
            res[arm] = dict(n_prompts=len(pids), n_cohorts=len(ids),
                            speedup=sp, speedup_ci=[boots[int(.025 * n_boot)], boots[int(.975 * n_boot)]],
                            decode_speedup=sp_dec, request_wall_speedup=req,
                            tok_per_s=sum(x["cohort_tokens"] for x in r0) / sum(x["wall_s"] for x in r0),
                            tpot_p50_ms=mean([x["tpot_p50_ms"] for x in r0]), tpot_p90_ms=mean([x["tpot_p90_ms"] for x in r0]),
                            ttft_s=mean([x["ttft_s"] for x in r0]),
                            rounds_mean=mean([x["rounds_mean"] for x in r0]),
                            core_rows_per_token=sum(x["core_rows"] for x in r0) / sum(x["cohort_tokens"] for x in r0),
                            hbm_peak_gb=max(x["hbm_peak_gb"] for x in r0),
                            acc_strict=mean([pr_c[p]["correct_strict"] for p in pids]),
                            acc_secondary=mean([pr_c[p].get(flexk, 0) for p in pids]), secondary_metric=flexk,
                            acc_delta=mean(dacc), acc_delta_ci=[accb[int(.025 * n_boot)], accb[int(.975 * n_boot)]],
                            acc2_delta=mean(dflex), acc2_delta_ci=[flexb[int(.025 * n_boot)], flexb[int(.975 * n_boot)]],
                            seq_exact_rate=mean([bool(pr_c[p].get("exact_to_AR")) for p in pids]),
                            mean_tokens=mean([pr_c[p]["n_tokens"] for p in pids]))
        out[f"{model}|B{B}|{task}|{split}|{cell}"] = res
    return out


def e2e_node(res, baseline="AR"):
    rows = []
    for f in sorted(res.glob("*.jsonl")):
        rs = load_e2e([f])
        meta = json.loads(Path(str(f)[:-len(".jsonl")] + ".meta.json").read_text())
        for r in rs:
            r["_cell"] = (f"D{meta['depth']}|init={meta['init']}|corr={meta['corrector']}|{meta.get('fewshot_used')}shot"
                          f"|{meta.get('dtype', 'bfloat16')}|seed{meta.get('seed')}")
            for k in ("model", "task", "split"):
                r.setdefault(k, meta[k])
            r["B_actual"] = r.get("B")
            r["B"] = meta["batch"]                 # nominal batch: a trailing partial cohort stays in its cell
        rows += rs
    s = e2e_summary(rows, baseline=baseline)
    ooms = sorted({f"{r['model']}|B{r['B']}|{r['arm']}" for r in rows if r.get("oom")})
    return dict(cells=s, oom=ooms, sample_count=len({(r["model"], r["prompt_idx"]) for r in rows if not r.get("oom")}),
                full_e2e_measured=True)


NODE_FN = {"G0_NATIVE": g0_native, "G0_KV": g0_kv, "G1_WIDTH": g1_width, "G1_DYNAMICS": g1_dynamics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default="AR", help="e2e reference arm (e.g. trunc3 for candidate-vs-truncation)")
    a = p.parse_args()
    res = Path(a.run_dir) / "results" / a.node
    if a.node in NODE_FN:
        m = NODE_FN[a.node](res)
    elif list(res.glob("*.jsonl")):
        m = e2e_node(res, a.baseline)
    else:
        m = diag_jobs(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, (dict, list))}))


if __name__ == "__main__":
    main()
