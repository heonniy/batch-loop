#!/usr/bin/env python3
"""H200x8 quality-only KV knobs: controlled job builder, resume and worker launcher.

No GPU job is run unless --execute is explicitly set. Each worker receives a
single CUDA_VISIBLE_DEVICES ordinal. Never use this to claim physical HBM savings:
lastwin / bank...win policies only mask attention over the broadcast store.

Examples, from kv_policy/:
 python scripts/kv_knob_sweep.py --phase depth --task nll --sample pilot --run-dir results/knob_depth_nll_pilot
 python scripts/kv_knob_sweep.py --phase postlast --task nll --sample pilot --run-dir results/knob_postlast_nll_pilot
 python scripts/kv_knob_sweep.py --phase oneplane --task nll --sample pilot --run-dir results/knob_oneplane_nll_pilot
 python scripts/kv_knob_sweep.py --phase factored --task nll --sample pilot --run-dir results/knob_factored_nll_pilot
 python scripts/kv_knob_sweep.py --phase sub50 --task nll --sample pilot --run-dir results/knob_sub50_nll_pilot
 # Add --execute --gpu-ids 0,1,2,3,4,5,6,7 only on the authorized/idle H200 host.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KV_ROOT = HERE.parent
REPO = KV_ROOT.parent
# The launcher is executed as scripts/foo.py; kvp sits in the parent directory.
sys.path.insert(0, str(KV_ROOT))
SOURCE_BANKS = KV_ROOT / "results/20261027T152845Z_ouro_last_strict_protect/banks.json"

PHASES = {
    # Two focused 2026-10-29 hypotheses. Other legacy phases remain usable
    # but are NOT part of the focused study/decision plan.
    "ouro_token": {
        "models": ["ouro-1.4b", "ouro-2.6b"],
        "policies": [
            "full-strict", "last-strict", "bankR1all-strict",
            "bankP0-strict", "bankcal-strict",
            "bankTopB35-strict", "bankTopB40-strict",
            "bankPrefixB35-strict", "bankPrefixB40-strict",
            "tokL1hash0-strict", "tokL1hash20-strict",
            "tokL1hash40-strict", "tokL1hash60-strict",
            "tokL1hash80-strict", "tokL1hash100-strict",
            "tokL1prompt-strict", "tokL1decode-strict",
            "tokL1delta50-strict", "tokL1delta100-strict",
        ],
    },
    "huginn_hard": {
        "models": ["huginn-0125-r8", "huginn-0125-r12",
                   "huginn-0125-r16", "huginn-0125-r20",
                   "huginn-0125-r24", "huginn-0125",
                   "huginn-0125-r40", "huginn-0125-r48",
                   "huginn-0125-r64"],
        "policies": ["full-strict", "last-strict"],
    },
    # R4/R8/R16/R24/R32; note R4 may be poor even with Full and is not automatically GO.
    "depth": {
        "models": ["huginn-0125-r4", "huginn-0125-r8", "huginn-0125-r16",
                   "huginn-0125-r24", "huginn-0125"],
        "policies": ["full-strict", "last-strict", "average-strict"],
    },
    # This is the *post-LAST* knob: mask past tokens in every layer.
    # Sinks=first four real tokens; optionally keep the entire prompt.
    "postlast": {
        "models": ["huginn-0125", "huginn-0125-r16"],
        "policies": ["full-strict", "last-strict", "lastwin64-strict",
                     "lastwin128-strict", "lastwin256-strict",
                     "lastpromptwin64-strict", "lastpromptwin128-strict",
                     "lastpromptwin256-strict"],
    },
    # At the SAME one-plane footprint, does a first/last blend beat Last?
    "oneplane": {
        "models": ["ouro-1.4b", "ouro-2.6b"],
        "policies": ["full-strict", "last-strict", "first-strict",
                     "average-strict", "blend25-strict", "blend50-strict",
                     "blend75-strict", "bankP0-strict", "bankcal-strict",
                     "bankR1all-strict"],
    },
    # Ouro first-recurrence protection: keep near-Loop1all accuracy using <50%
    # of FULL recurrent KV. Exact budget counts fixed per model in sub50_budget.
    "sub50": {
        "models": ["ouro-1.4b", "ouro-2.6b"],
        "policies": ["full-strict", "last-strict", "bankP0-strict",
                     "bankcal-strict", "bankR1all-strict"] +
                    [f"bank{selection}B{pct}-strict"
                     for pct in (30, 35, 40, 45)
                     for selection in ("Top", "Prefix", "Rand")],
    },
    # Factorial interaction screening; bank+window is a SEPARABLE baseline,
    # NOT a truly per-token joint bank allocator.
    "factored": {
        "models": ["ouro-1.4b", "ouro-2.6b"],
        "policies": ["full-strict", "last-strict", "bankP0-strict",
                     "bankcal-strict", "lastwin64-strict", "lastwin128-strict",
                     "lastpromptwin128-strict", "bankP0win64-strict",
                     "bankP0win128-strict", "bankP0promptwin128-strict",
                     "bankcalwin128-strict"],
    },
}


def repo_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def job_name(job: dict) -> str:
    suffix = "" if job.get("batching", "main") == "main" else f"-{job['batching']}"
    style = "" if job.get("prompt_style", "default") == "default" else f"__style{job['prompt_style']}"
    return (f"{job['model']}__{job['dataset']}__{job['task']}__{job['policy']}-"
            f"{job['mode']}{suffix}{style}__b{job['batch']:02d}")


def write_nominal_logical_budgets(run: Path, spec: dict) -> None:
    """Gold-prefix terminal-length analytic ratios, NEVER GPU peak-memory readings.

    This is comparable within the same model/dataset/gold-token cohort.
    Free-generation lengths differ by policy and are not captured here.
    """
    sys.path.insert(0, str(KV_ROOT))
    from kvp.knob_budget import layout_for_model

    output = run / "analysis"
    output.mkdir(exist_ok=True)
    dest = output / "nominal_logical_kv_budget.csv"
    with dest.open("w", newline="") as f:
        fields = ["model", "dataset", "policy", "uid", "prompt_tokens",
                  "gold_tokens", "nominal_total_tokens", "ratio_to_full",
                  "reference_only", "definition"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model in spec["models"]:
            m = layout_for_model(model)
            for dataset in spec["datasets"]:
                suffix = "" if spec.get("prompt_style", "default") == "default" else f"__style{spec['prompt_style']}"
                prompt_file = run / "prompts" / f"{model}__{dataset}{suffix}.jsonl"
                for line in prompt_file.read_text().splitlines()[1:]:
                    item = json.loads(line)
                    P, N = len(item["prompt_ids"]), len(item["target_ids"])
                    T = P + N
                    for label in spec["policies"]:
                        policy, mode = label.rsplit("-", 1)
                        if mode != "strict":
                            raise ValueError("Analytic budget requires strict policy")
                        wm = re.fullmatch(r"(last|bankP0|bankR1all|bankcal)(prompt)?win([1-9][0-9]*)", policy)
                        if policy == "full":
                            ratio = 1.0
                            budget_definition = "logical exact FULL"
                        elif policy.startswith("tokL1"):
                            # Hash policy is a causal Bernoulli rate (expected
                            # only); prompt/decode uses gold lengths (not free
                            # generation); delta selection requires a model run.
                            mhash = re.fullmatch(r"tokL1hash(0|20|40|60|80|100)(?:s[12])?", policy)
                            if mhash:
                                ratio = m.last_over_full() * (1.0 + int(mhash.group(1)) / 100)
                                budget_definition = "expected hash protection; actual per-job in engine_stats"
                            elif policy == "tokL1prompt":
                                ratio = m.last_over_full() * (1.0 + P / T)
                                budget_definition = "gold-length prompt-only estimate"
                            elif policy == "tokL1decode":
                                ratio = m.last_over_full() * (1.0 + N / T)
                                budget_definition = "gold-length decode-only estimate"
                            else:
                                ratio = None
                                budget_definition = "dynamic trajectory-selected fraction: requires runtime"
                        else:
                            budget_definition = "logical active KV at end of gold prefix"
                            n_extra = 0
                            if policy.startswith("bank"):
                                bank_name = re.sub(r"(?:prompt)?win[1-9][0-9]*$", "", policy)
                                banks = json.loads((run / "banks.json").read_text())
                                if model not in banks or bank_name not in banks[model]:
                                    raise ValueError(f"Missing bank allocation: {model}/{bank_name}")
                                n_extra = len(banks[model][bank_name])
                            ratio = m.protected_over_full(n_extra)
                            if wm:
                                # The window prunes every attention layer, not
                                # just the recurrent KV. Bank protection lives
                                # only for surviving historical tokens.
                                surviving = m.postlast_window_over_full(
                                    T, int(wm.group(3)), 4, P, bool(wm.group(2)))
                                ratio *= surviving / m.last_over_full()
                        w.writerow({"model": model, "dataset": dataset, "policy": policy,
                                    "uid": item["uid"], "prompt_tokens": P,
                                    "gold_tokens": N, "nominal_total_tokens": T,
                                    "ratio_to_full": ratio, "reference_only": True,
                                    "definition": budget_definition + "; not measured HBM"})


def report(run: Path) -> dict:
    jobs_path = run / "jobs.json"
    if not jobs_path.exists():
        return {"status": "not_prepared"}
    jobs = json.loads(jobs_path.read_text())
    missing = []
    for job in jobs:
        if not (run / "raw" / f"{job_name(job)}.jsonl").exists():
            missing.append(job_name(job))
    errors = sorted(p.name for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    out = {"status": "complete" if not missing and not errors else "incomplete",
           "expected_jobs": len(jobs), "completed_jobs": len(jobs) - len(missing),
           "missing_jobs": missing, "worker_errors": errors}
    (run / "completeness.json").write_text(json.dumps(out, indent=2))
    # Works with NLL-only and generation-only runs. No rewriting raw records.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    # Sum truly processed, row/length-adjusted token events. The reference
    # still allocates FULL-sized tensors; these are LOGICAL KV amounts.
    token_budget: dict[tuple, dict] = defaultdict(lambda: {"real": 0, "extra": 0, "jobs": 0})
    for job in jobs:
        file = run / "raw" / f"{job_name(job)}.jsonl"
        if not file.exists():
            continue
        lines = file.read_text().splitlines()
        if not lines:
            continue
        head = json.loads(lines[0]).get("job_meta", {})
        logical = head.get("engine_stats", {}).get("logical_retained_kv", {})
        if logical and job["policy"].startswith("tokL1"):
            entry = token_budget[(job["model"], job["dataset"], job["task"], job["policy"])]
            entry["real"] += int(logical["n_processed_real_token_positions"])
            entry["extra"] += int(logical["n_protected_token_positions"])
            entry["jobs"] += 1
        for line in lines[1:]:
            row = json.loads(line)
            groups[(row["model"], row["dataset"], row["task"],
                    row["policy"], row["mode"])].append(row)
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    budget_fields = ["model", "dataset", "task", "policy", "n_jobs",
        "processed_real_tokens", "protected_first_recurrence_tokens",
        "protected_token_fraction", "logical_full_recurrent_kv_percent",
        "additional_memory_saved_vs_loop1all_percent", "reference_full_allocated"]
    with (analysis / "token_loop_logical_memory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=budget_fields)
        writer.writeheader()
        for (model, ds, task, policy), b in sorted(token_budget.items()):
            if not b["real"]:
                continue
            actual = (b["real"] + b["extra"]) / (4 * b["real"])
            writer.writerow(dict(zip(budget_fields, [
                model, ds, task, policy, b["jobs"], b["real"], b["extra"],
                b["extra"] / b["real"], actual * 100.0,
                (1 - actual / .5) * 100.0, True
            ])))
    with (analysis / "screen_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "task", "policy", "mode",
                 "n_items", "accuracy", "token_weighted_nll", "mean_output_tokens",
                 "length_cap_fraction", "parse_failure_fraction"])
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            task = key[2]
            if task == "gen":
                n = len(rows)
                acc = sum(int(bool(x["correct"])) for x in rows) / n
                length = sum(x["finish_reason"] == "length" for x in rows) / n
                parse = sum(not x.get("parse_ok", True) for x in rows) / n
                avg_len = sum(x["n_gen_tokens"] for x in rows) / n
                nll = ""
            else:
                total_toks = sum(x["target_tokens"] for x in rows)
                nll = -sum(x["sum_logprob"] for x in rows) / total_toks if total_toks else ""
                acc, length, parse, avg_len = "", "", "", ""
            writer.writerow(dict(zip(writer.fieldnames, [*key, len(rows), acc, nll,
                                                         avg_len, length, parse])))
    # Paired controls: every policy is compared to FULL/strict on identical
    # frozen UIDs. This is a descriptive screening CI, not a multiple-testing
    # corrected paper-level claim.
    index = {k: {r["uid"]: r for r in rows} for k, rows in groups.items()}
    pair_rows = []
    rng = random.Random(20261029)
    for (model, dataset, task, policy, mode), current in sorted(index.items()):
        if (policy, mode) == ("full", "strict"):
            continue
        baseline = index.get((model, dataset, task, "full", "strict"))
        if baseline is None:
            continue
        common = sorted(set(baseline) & set(current))
        if not common:
            continue
        if task == "gen":
            metric = "delta_accuracy"
            delta = [int(bool(current[u]["correct"])) -
                     int(bool(baseline[u]["correct"])) for u in common]
        else:
            metric = "delta_nll_per_problem"
            delta = []
            for u in common:
                assert baseline[u]["target_tokens"] == current[u]["target_tokens"], "target mismatch"
                delta.append(float(current[u]["nll_mean"] - baseline[u]["nll_mean"]))
        avg = sum(delta) / len(delta)
        boot = []
        for _ in range(2000):
            boot.append(sum(delta[rng.randrange(len(delta))] for _ in delta) / len(delta))
        boot.sort()
        pair_rows.append({"model": model, "dataset": dataset, "task": task,
                          "policy": policy, "mode": mode, "n_paired": len(delta),
                          "metric": metric, "delta": avg,
                          "ci95_lo": boot[49], "ci95_hi": boot[1949]})
    with (analysis / "paired_vs_full_strict.csv").open("w", newline="") as f:
        fields = ["model", "dataset", "task", "policy", "mode", "n_paired",
                  "metric", "delta", "ci95_lo", "ci95_hi"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    # Dedicated scientific question: how much less than 50%-FULL KV is
    # sufficient to match the best fixed ALL-recurrence-1 protected baseline?
    spec_file = run / "knob_manifest.json"
    if spec_file.exists() and json.loads(spec_file.read_text())["spec"]["phase"] in ("sub50", "ouro_token"):
        from kvp.knob_budget import layout_for_model
        all_rows = []
        registry = json.loads((run / "banks.json").read_text())
        for (model, ds, task, pol, mode), candidate in sorted(index.items()):
            if pol in {"full", "bankR1all"}:
                continue
            reference = index.get((model, ds, task, "bankR1all", "strict"))
            if not reference:
                continue
            shared = sorted(set(candidate) & set(reference))
            if not shared:
                continue
            if task == "gen":
                diffs = [float(bool(candidate[u]["correct"])) -
                         float(bool(reference[u]["correct"])) for u in shared]
                metric = "delta_accuracy_vs_50"
            else:
                diffs = [float(candidate[u]["nll_mean"]) -
                         float(reference[u]["nll_mean"]) for u in shared]
                metric = "delta_mean_problem_nll_vs_50"
            if pol.startswith("tokL1"):
                budget = token_budget.get((model, ds, task, pol))
                if not budget or not budget["real"]:
                    continue
                pct = 25.0 * (1 + budget["extra"] / budget["real"])
                budget_type = "observed logical; actual processed rows"
            else:
                if pol == "last" or pol in {"average", "first"} or pol.startswith("blend"):
                    k = 0
                elif pol.startswith("bank"):
                    k = len(registry[model][pol])
                else:
                    continue
                layout = layout_for_model(model)
                pct = 100.0 * layout.protected_over_full(k)
                budget_type = "static logical"
            avg = sum(diffs) / len(diffs)
            boot = []
            for _ in range(2000):
                boot.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
            boot.sort()
            all_rows.append({
                "model": model, "dataset": ds, "task": task,
                "policy": pol, "metric": metric,
                "n_paired": len(shared), "nominal_full_kv_percent": pct,
                "reference_50_full_kv_percent": 50.0,
                "extra_saving_vs_loop1all_percent": (1 - pct / 50.0) * 100.0,
                "budget_type": budget_type,
                "mean_delta_vs_loop1all": avg,
                "ci95_lo": boot[49], "ci95_hi": boot[1949],
                "within_3pp_point_estimate": (task == "gen" and avg >= -0.03)
            })
        fields = ["model", "dataset", "task", "policy", "metric", "n_paired",
                  "nominal_full_kv_percent", "reference_50_full_kv_percent",
                  "extra_saving_vs_loop1all_percent", "budget_type",
                  "mean_delta_vs_loop1all", "ci95_lo", "ci95_hi",
                  "within_3pp_point_estimate"]
        with (analysis / "sub50_paired_vs_loop1all.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--task", choices=["nll", "gen", "both"], default="nll")
    ap.add_argument("--sample", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--prompt-style", default="default",
                    choices=["default", "reason", "verify", "distract512"])
    ap.add_argument("--models", default="", help="optional subset/override of models for this phase")
    ap.add_argument("--policies", default="",
                    help="optional comma-separated subset of this phase's policies")
    ap.add_argument("--datasets", default="gsm8k,math500")
    ap.add_argument("--batch-size-gsm8k", type=int, default=8)
    ap.add_argument("--batch-size-math500", type=int, default=4)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--execute", action="store_true", help="actually launch CUDA jobs (opt-in)")
    ap.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7",
                    help="must list only explicitly authorized idle H200 ordinals")
    args = ap.parse_args()
    chosen = PHASES[args.phase]
    models = [m for m in (args.models.split(",") if args.models else chosen["models"]) if m]
    valid_models = set(chosen["models"])
    if not models or not set(models) <= valid_models:
        ap.error(f"models must be a subset of {sorted(valid_models)}")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()] if args.policies else list(chosen["policies"])
    if len(policies) != len(set(policies)) or not set(policies) <= set(chosen["policies"]):
        ap.error("Requested policies must be a unique subset of selected phase")
    if args.prompt_style != "default" and args.phase != "huginn_hard":
        ap.error("Prompt stress styles are valid only for huginn_hard")
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets or not set(datasets) <= {"gsm8k", "math500"}:
        ap.error("datasets must be a nonempty subset of gsm8k,math500")
    if args.batch_size_gsm8k < 1 or args.batch_size_math500 < 1:
        ap.error("batch sizes must be positive")
    batch_sizes = {"gsm8k": args.batch_size_gsm8k, "math500": args.batch_size_math500}
    tasks = ["nll", "gen"] if args.task == "both" else [args.task]
    spec = {"phase": args.phase, "sample": args.sample, "models": models,
            "policies": policies, "tasks": tasks, "datasets": datasets,
            "batch_sizes": batch_sizes, "prompt_style": args.prompt_style}
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    plan_file = run / "knob_manifest.json"
    if plan_file.exists():
        old = json.loads(plan_file.read_text())
        if old["spec"] != spec:
            ap.error("Existing run contains a different matrix; choose a new --run-dir")
        if old["source_sha"] != repo_sha():
            ap.error("Git commit changed after preparing run; preserve evidence, use a new run-dir")
    elif (run / "jobs.json").exists():
        ap.error("Untracked jobs.json exists: refusing to mix runs")
    else:
        cmd = [sys.executable, str(HERE / "make_quality_jobs.py"),
               "--run-dir", str(run), "--models", ",".join(models),
               "--policies", ",".join(policies),
               "--batching", args.sample, "--prompt-style", args.prompt_style,
               "--tasks", ",".join(tasks),
               "--datasets", ",".join(datasets),
               "--batch-size-gsm8k", str(args.batch_size_gsm8k),
               "--batch-size-math500", str(args.batch_size_math500)]
        subprocess.run(cmd, cwd=KV_ROOT, check=True)
        if any(p.startswith("bank") for p in policies):
            if not SOURCE_BANKS.is_file():
                raise FileNotFoundError(f"Expected frozen bank mapping: {SOURCE_BANKS}")
            if args.phase in ("sub50", "ouro_token"):
                from kvp.sub50_budget import build_registry, assert_disjoint_from_eval
                registry, provenance = build_registry(KV_ROOT, SOURCE_BANKS)
                assert_disjoint_from_eval(provenance, run, datasets)
                (run / "banks.json").write_text(json.dumps(registry, indent=2))
                (run / "sub50_provenance.json").write_text(json.dumps(provenance, indent=2))
            else:
                shutil.copyfile(SOURCE_BANKS, run / "banks.json")
        manifest = {"source_sha": repo_sha(), "spec": spec,
                    "source_banks_sha256": hashlib.sha256((run / "banks.json").read_bytes()).hexdigest()
                    if (run / "banks.json").exists() else None,
                    "note": "All policies use broadcast reference KV. sub50 is layer-wise first-loop protection; postlast/factored include logical eviction masks. No physical compact memory claim."}
        plan_file.write_text(json.dumps(manifest, indent=2))
        write_nominal_logical_budgets(run, spec)
    if args.execute:
        ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
        if not ids or len(ids) != len(set(ids)) or any(not x.isdigit() for x in ids):
            ap.error("--gpu-ids must contain unique nonnegative physical device ordinals")
        active = []
        try:
            for worker, gpu in enumerate(ids):
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                path = run / "logs" / f"worker{worker}_stdout.log"
                log = path.open("a")
                cmd = [sys.executable, str(HERE / "quality_worker.py"),
                       "--run-dir", str(run), "--worker", str(worker)]
                proc = subprocess.Popen(cmd, cwd=KV_ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                active.append((proc, log))
            codes = [proc.wait() for proc, _ in active]
            if any(code != 0 for code in codes):
                raise RuntimeError(f"Worker process exit codes: {codes}")
        finally:
            for proc, log in active:
                if proc.poll() is None:
                    proc.wait()
                log.close()
    result = report(run)
    print(json.dumps({k: v for k, v in result.items() if k != "missing_jobs"}, indent=2))
    print(f"Analysis: {run / 'analysis' / 'screen_summary.csv'}")
    if args.execute and result["status"] != "complete":
        raise SystemExit("Incomplete jobs/errors: inspect logs and claims before retrying")


if __name__ == "__main__":
    main()
