#!/usr/bin/env python3
"""LoopZip v5 Stage 1 (LOOPZIP_EVIDENCE_CONTROLLED_LOOP_SPECIFIC_PLAN.md §3): evidence-matched selection.

Arm 1B (GPU): for each frozen Phase-1 request (same prompts/targets/UIDs as --source-run), keep the
annotated evidence E fixed in every policy and fill the remaining K-|E| Context slots with
Mean/Last/Max/Recent+Mean attention or seeded Random over NON-evidence tokens. Infeasible
(|E| > K) cells are recorded, never re-budgeted. Measures future gold NLL, greedy EM and the
gold-independent fixed-prefix KL(p_FULL || p_policy) on the Full-KV greedy continuation.
Arm 1A (CPU, --analyze): naturally occurring Phase-1 policy pairs with IDENTICAL retained evidence sets.

OFFLINE ORACLE CONTROL: evidence annotation is used on purpose; never a deployable policy.
--prepare (CPU) / --execute (GPU ids) / --analyze (CPU) / internal --worker.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from kvp.loopzip_evidence_lock import LOCK_POLICIES  # noqa: E402
from kvp.specs import MODELS  # noqa: E402

PROTOCOL = "loopzip-v5-stage1-evidence-lock-v1"


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def _json(path: Path, value):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def prepare(run: Path, cfg: dict):
    from transformers import AutoTokenizer
    from kvp.loopzip_evidence_audit import evidence_spans, token_offsets
    from kvp.loopzip_policy import context_budget

    src = Path(cfg["source_run"])
    if not json.loads((src / "completeness.json").read_text())["complete"]:
        raise SystemExit("Source Phase-1 run must be complete")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run directory")
    run.mkdir(parents=True, exist_ok=True)
    src_jobs = json.loads((src / "jobs.json").read_text())
    per_cell = defaultdict(int)
    jobs, toks = [], {}
    for j in src_jobs:
        if j["model"] not in cfg["models"] or j["dataset"] not in cfg["datasets"]:
            continue
        cell = (j["model"], j["dataset"])
        if per_cell[cell] >= cfg["n_examples"]:
            continue
        per_cell[cell] += 1
        if j["model"] not in toks:
            ck = MODELS[j["model"]]
            toks[j["model"]] = AutoTokenizer.from_pretrained(ck.repo, revision=ck.revision, trust_remote_code=True)
        ev = evidence_spans(j, token_offsets(toks[j["model"]], j["prompt_text"], j["prompt_ids"]))
        feas = {str(b): len(ev["E"]) <= context_budget(j["content_positions"], b) for b in cfg["retention"]}
        jobs.append({**j, "evidence_positions": ev["E"], "evidence_code": ev.get("code"),
                     "evidence_id": ev.get("id"), "lock_feasible": feas, "source_job_key": j["key"]})
    (run / "jobs.json").write_text(json.dumps(jobs, indent=1))
    _json(run / "run_manifest.json", {
        "protocol": PROTOCOL, "code_sha": git_sha(), "config": cfg, "job_count": len(jobs),
        "source_run": str(src), "source_code_sha": json.loads((src / "run_manifest.json").read_text()).get("code_sha"),
        "model_revisions": {m: MODELS[m].revision for m in cfg["models"]},
        "oracle_control": True,
        "notes": ["Evidence-locked sets use ground-truth annotation: offline oracle control only",
                  "Recent+Attn pins floor(free/4) most recent NON-evidence tokens inside K",
                  "Infeasible |E|>K cells are recorded, never re-budgeted",
                  "Full-sized reference KV; no HBM/latency claims"]})
    return {"jobs": len(jobs), "sha": git_sha(),
            "feasible_cells": sum(v for j in jobs for v in j["lock_feasible"].values()),
            "total_cells": sum(len(j["lock_feasible"]) for j in jobs)}


def evaluate(bb, tok, job, cfg):
    from loopzip_failure_screen import _new_cached
    from kvp.loopzip_evidence_lock import fixed_prefix_kl, lock_select, teacher_forced_logprobs

    cache = _new_cached(bb, job, {"generation_tokens": cfg["generation_tokens"]})
    parity = cache.assert_clean_reference_parity()
    base = cache.score_gold(())
    tgt = job["target_ids"]
    lp_gold = teacher_forced_logprobs(cache, tgt)
    kl_path_diff = max(abs(float(-lp_gold[i, t]) - base["token_losses"][i]) for i, t in enumerate(tgt))
    if kl_path_diff > 1e-4:
        raise RuntimeError(f"Teacher-forced log-prob path disagrees with score_gold: {kl_path_diff}")
    parity["kl_path_vs_score_gold_max_abs"] = kl_path_diff
    g_full = cache.greedy(tok, job["gold_answer"], ())
    ref_tokens = g_full["token_ids"]
    lp_full = teacher_forced_logprobs(cache, ref_tokens) if len(ref_tokens) >= 2 else None
    att = cache.clean_attention["content_attention_by_source"]
    rows = []
    for b in cfg["retention"]:
        for pol in cfg["policies"]:
            for seed in (cfg["random_seeds"] if pol == "random" else (0,)):
                sel = lock_select(job["content_positions"], job["evidence_positions"], att, b, pol,
                                  uid=job["uid"], seed=seed)
                key = f"random_s{seed}" if pol == "random" else pol
                if not sel["feasible"]:
                    rows.append({"retention": b, "policy_key": key, **sel})
                    continue
                if sel["n_retained"] != job["budget_counts"][str(b)]:
                    raise RuntimeError("Budget differs from frozen Phase-1 budget")
                s = cache.score_gold(sel["evicted"])
                if abs(s["first_target_nll"] - base["first_target_nll"]) > 1e-4:
                    raise RuntimeError("First gold token changed")
                g = cache.greedy(tok, job["gold_answer"], sel["evicted"])
                kl = (fixed_prefix_kl(lp_full, teacher_forced_logprobs(cache, ref_tokens, sel["evicted"]))
                      if lp_full is not None else None)
                rows.append({"retention": b, "policy_key": key, **sel,
                             "nll_future": s["nll_future"], "delta_nll": s["nll_future"] - base["nll_future"],
                             "greedy_correct": g["exact_match"], "greedy_answer": g["extracted_answer"],
                             "fixed_prefix_kl": kl})
    return {"meta": {k: job[k] for k in ("key", "uid", "model", "dataset", "revision", "prompt_sha256",
                                          "source_job_key", "attributes")} | {"sha": git_sha()},
            "parity": parity, "n_evidence": len(job["evidence_positions"]),
            "full": {"nll_future": base["nll_future"], "greedy_correct": g_full["exact_match"],
                     "greedy_answer": g_full["extracted_answer"], "kl_reference_tokens": len(ref_tokens)},
            "rows": rows}


def worker(run: Path, idx: int):
    import torch
    from loopzip_failure_screen import _model_context

    m = json.loads((run / "run_manifest.json").read_text())
    if m["code_sha"] != git_sha():
        raise RuntimeError("Code SHA changed; use a NEW run directory")
    jobs = json.loads((run / "jobs.json").read_text())
    raw, claims, errors = (run / d for d in ("raw", "claims", "errors"))
    for d in (raw, claims, errors):
        d.mkdir(exist_ok=True)
    cur, tok, bb = None, None, None
    while True:
        pend = [j for j in jobs if not (raw / f"{j['key']}.json").exists() and not (claims / f"{j['key']}.lock").exists()]
        if not pend:
            return
        pend.sort(key=lambda j: (j["model"] != cur, j["model"], j["key"]))
        job = pend[0]
        try:
            fd = os.open(claims / f"{job['key']}.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"worker={idx} pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        try:
            if cur != job["model"]:
                tok = bb = None
                torch.cuda.empty_cache()
                tok, bb = _model_context(job["model"])
                cur = job["model"]
            _json(raw / f"{job['key']}.json", evaluate(bb, tok, job, m["config"]))
        except Exception:
            _json(errors / f"{job['key']}.json", {"key": job["key"], "traceback": traceback.format_exc()})
            raise


def _boot(vals, n=1000, seed=0):
    if not vals:
        return None
    rng = random.Random(seed)
    d = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return [d[int(.025 * n)], d[int(.975 * n) - 1]]


def _write(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols or ["empty"])
        w.writeheader()
        w.writerows(rows)


def analyze(run: Path):
    m = json.loads((run / "run_manifest.json").read_text())
    jobs = json.loads((run / "jobs.json").read_text())
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    res = {p.stem: json.loads(p.read_text()) for p in (run / "raw").glob("*.json")}
    missing = [j["key"] for j in jobs if j["key"] not in res]
    errs = sorted(p.name for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    bad_parity = [k for k, r in res.items() if r["parity"]["reference_max_abs_nll_diff"] > 1e-4]
    q, feas = [], []
    for k, r in res.items():
        mt = r["meta"]
        for x in r["rows"]:
            base = {"model": mt["model"], "dataset": mt["dataset"], "uid": mt["uid"], "budget": x["retention"],
                    "policy": x["policy_key"], "family": "random" if x["policy"] == "random" else x["policy"],
                    "n_evidence": x["n_evidence"], "k": x["n_retained"], "feasible": x["feasible"],
                    "full_correct": r["full"]["greedy_correct"]}
            if x["policy_key"] in ("mean_attn",):
                feas.append({**{k2: base[k2] for k2 in ("model", "dataset", "uid", "budget", "n_evidence", "k", "feasible")}})
            if x["feasible"]:
                q.append({**base, "delta_nll": x["delta_nll"], "greedy_correct": x["greedy_correct"],
                          "fixed_prefix_kl": x["fixed_prefix_kl"], "greedy_answer": x["greedy_answer"]})
    _write(out / "evidence_matched_policy_quality.csv", q)
    _write(out / "evidence_lock_feasibility.csv", feas)
    # per-request family means (random averaged over seeds), then paired effects vs mean_attn
    cell = defaultdict(lambda: defaultdict(list))
    for r in q:
        cell[(r["model"], r["dataset"], r["budget"], r["uid"])][r["family"]].append(r)
    summ = []
    groups = sorted({(a, b, c) for a, b, c, _ in cell})
    for g in groups:
        reqs = {u: fam for (a, b, c, u), fam in cell.items() if (a, b, c) == g}
        for fam in LOCK_POLICIES:
            vals = {u: f[fam] for u, f in reqs.items() if fam in f}
            if not vals:
                continue
            mean = lambda rs, k: sum(float(x[k]) for x in rs) / len(rs)  # noqa: E731
            dn = [mean(v, "delta_nll") for v in vals.values()]
            kl = [mean(v, "fixed_prefix_kl") for v in vals.values() if v[0]["fixed_prefix_kl"] is not None]
            fc = [mean(v, "greedy_correct") for v in vals.values() if v[0]["full_correct"]]
            paired = [mean(v, "delta_nll") - mean(reqs[u]["mean_attn"], "delta_nll") for u, v in vals.items()
                      if "mean_attn" in reqs[u]]
            summ.append({"model": g[0], "dataset": g[1], "budget": g[2], "policy": fam, "n_requests": len(vals),
                         "mean_delta_nll": sum(dn) / len(dn), "ci95": _boot(dn),
                         "mean_fixed_prefix_kl": sum(kl) / len(kl) if kl else None, "kl_ci95": _boot(kl),
                         "n_full_correct": len(fc), "acc_given_full_correct": sum(fc) / len(fc) if fc else None,
                         "paired_minus_mean_attn": sum(paired) / len(paired) if paired else None,
                         "paired_ci95": _boot(paired, seed=1)})
    # Arm 1A: identical retained evidence across Phase-1 attention policies
    src = Path(m["source_run"])
    ev_by_key = {j["source_job_key"]: set(j["evidence_positions"]) for j in jobs}
    pairs = []
    for p in (src / "raw").glob("*.json"):
        r = json.loads(p.read_text())
        key = r["meta"]["key"]
        if key not in ev_by_key:
            continue
        E = ev_by_key[key]
        by_b = defaultdict(list)
        for s in r["screening"]:
            if not s["policy_key"].startswith("random"):
                by_b[s["retention_requested"]].append(s)
        for b, ss in by_b.items():
            for i in range(len(ss)):
                for k2 in range(i + 1, len(ss)):
                    a, c = ss[i], ss[k2]
                    if set(a["retained"]) & E == set(c["retained"]) & E:
                        pairs.append({"model": r["meta"]["model"], "dataset": r["meta"]["dataset"],
                                      "uid": r["meta"]["uid"], "budget": b, "a": a["policy_key"],
                                      "b": c["policy_key"], "evidence_kept": len(set(a["retained"]) & E),
                                      "n_evidence": len(E), "abs_delta_nll_diff": abs(a["delta_nll"] - c["delta_nll"]),
                                      "correct_differs": a["greedy_correct"] != c["greedy_correct"]})
    _write(out / "arm1a_identical_evidence_pairs.csv", pairs)
    complete = not missing and not errs and not bad_parity
    rep = {"protocol": PROTOCOL, "complete": complete, "expected_jobs": len(jobs), "completed_jobs": len(res),
           "feasible_cells": sum(f["feasible"] for f in feas), "total_cells": len(feas),
           "summary": summ, "arm1a_pairs": len(pairs),
           "arm1a_pairs_correct_differs": sum(p["correct_differs"] for p in pairs),
           "arm1a_by_evidence_kept_all": sum(p["evidence_kept"] == p["n_evidence"] for p in pairs)}
    _json(out / "paired_effects.json", rep)
    _json(run / "completeness.json", {"complete": complete, "missing": missing, "errors": errs,
                                      "parity_failures": bad_parity})
    fmt = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    ci = lambda c: "" if not c else f" [{c[0]:.3f},{c[1]:.3f}]"  # noqa: E731
    md = ["# LoopZip v5 Stage 1 — evidence-locked policies (OFFLINE ORACLE CONTROL)", "",
          f"complete={complete}; jobs {len(res)}/{len(jobs)}; feasible request-budget cells "
          f"{rep['feasible_cells']}/{rep['total_cells']}; parity failures {len(bad_parity)}.", "",
          "Every policy keeps the same evidence positions E and exactly K Context tokens.", "",
          "| model | dataset | b | policy | n | ΔNLL [95% CI] | KL(Full‖policy) | acc (full-correct) | ΔNLL − mean_attn [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        md.append(f"| {s['model']} | {s['dataset']} | {s['budget']} | {s['policy']} | {s['n_requests']} | "
                  f"{fmt(s['mean_delta_nll'])}{ci(s['ci95'])} | {fmt(s['mean_fixed_prefix_kl'], 4)} | "
                  f"{fmt(s['acc_given_full_correct'], 2)} (n={s['n_full_correct']}) | "
                  f"{fmt(s['paired_minus_mean_attn'])}{ci(s['paired_ci95'])} |")
    md += ["", f"Arm 1A (Phase-1 attention policies, identical retained evidence set): {len(pairs)} pairs; "
           f"greedy correctness differs in {rep['arm1a_pairs_correct_differs']}; "
           f"pairs with full evidence kept: {rep['arm1a_by_evidence_kept_all']}.",
           "", "Annotation-informed oracle control; residual differences are not by themselves recurrence-specific."]
    (out / "EVIDENCE_MATCHED_FINDINGS.md").write_text("\n".join(md) + "\n")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    for n in ("prepare", "execute", "analyze"):
        g.add_argument("--" + n, action="store_true")
    g.add_argument("--worker", type=int)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--source-run", default="results/20261030T164908Z_loopzip_failure_pilot20")
    ap.add_argument("--models", default="ouro-2.6b,huginn-0125")
    ap.add_argument("--datasets", default="controlled_retrieval,hotpotqa_distractor")
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--retention", default="0.10,0.30,0.50")
    ap.add_argument("--policies", default=",".join(LOCK_POLICIES))
    ap.add_argument("--random-seeds", default="11,29,47")
    ap.add_argument("--generation-tokens", type=int, default=32)
    ap.add_argument("--gpu-ids", default="0")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    if a.prepare:
        cfg = {"source_run": str(Path(a.source_run).resolve()), "models": sp(a.models), "datasets": sp(a.datasets),
               "n_examples": a.n_examples, "retention": [float(x) for x in sp(a.retention)],
               "policies": sp(a.policies), "random_seeds": [int(x) for x in sp(a.random_seeds)],
               "generation_tokens": a.generation_tokens}
        if not set(cfg["policies"]) <= set(LOCK_POLICIES) or "mean_attn" not in cfg["policies"]:
            ap.error("policies must include mean_attn and be a subset of " + ",".join(LOCK_POLICIES))
        print(json.dumps(prepare(run, cfg), indent=2))
    elif a.worker is not None:
        worker(run, a.worker)
    elif a.analyze:
        r = analyze(run)
        print(json.dumps({k: r[k] for k in ("complete", "completed_jobs", "feasible_cells", "total_cells",
                                            "arm1a_pairs", "arm1a_pairs_correct_differs")}, indent=2))
    else:
        ids = sp(a.gpu_ids)
        (run / "logs").mkdir(exist_ok=True)
        ps = []
        for i, gpu in enumerate(ids):
            h = (run / "logs" / f"worker{i}.txt").open("a")
            ps.append((subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i),
                                         "--run-dir", str(run)], cwd=ROOT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                        stdout=h, stderr=subprocess.STDOUT), h))
        codes = [p.wait() for p, _ in ps]
        r = analyze(run)
        print(json.dumps({"worker_exit_codes": codes, "complete": r["complete"]}))
        if any(codes) or not r["complete"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
