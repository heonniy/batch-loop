#!/usr/bin/env python3
"""Phase 1 & 2 failure-first LoopZip screening for fixed-depth FULL prefill.

--prepare/--analyze/--prepare-swap/--analyze-swap are CPU-only phases.
--execute/--execute-swap explicitly opt in to model/GPU workers. The same
cached clean prefix is reused across all 10/30/50% policy branches. This is
NOT physical KV compaction and must not be used as measured serving TPOT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.insert(0,str(ROOT))

from kvp.loopzip_policy import (
    BUDGETS, ATTENTION_POLICIES, ALL_POLICIES, RANDOM_SEEDS,
    context_budget, selection_manifest, swap_neighbors, classify_failure)
from kvp.specs import MODELS

MODELS_ALLOWED=("ouro-2.6b","huginn-0125","ouro-1.4b")
DATASETS_ALLOWED=("controlled_retrieval","hotpotqa_distractor")
PROTOCOL="loopzip-failure-first-fixed-v1"


def digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def git_sha() -> str:
    return subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,
                          capture_output=True,text=True,check=True).stdout.strip()


def _json(path: Path, value):
    temp=path.with_name("."+path.name+".tmp")
    temp.write_text(json.dumps(value,indent=2))
    os.replace(temp,path)


def _read(path: Path):
    return json.loads(path.read_text())


def prepare(run: Path,cfg: dict):
    from transformers import AutoTokenizer
    from kvp.role_data import build_evidence_prompt
    from kvp.role_evidence_data import load_evidence_workload

    if run.exists() and any(run.iterdir()):
        raise FileExistsError("A frozen run directory must start empty")
    run.mkdir(parents=True,exist_ok=True)
    jobs=[];dataset_meta={};uids={}
    for ds in cfg["datasets"]:
        docs,meta=load_evidence_workload(
            ds,cfg["n_examples"],cfg["seed"],
            controlled_facts=cfg["controlled_facts"],
            hotpot_context_mode=cfg["hotpot_context_mode"],
            hotpot_distractor_sentences=cfg["hotpot_distractor_sentences"])
        if len(docs)!=cfg["n_examples"]:
            raise RuntimeError("Not enough valid frozen dataset requests")
        dataset_meta[ds]=meta
        uids[ds]=[doc["uid"] for doc in docs]
        for mk in cfg["models"]:
            checkpoint=MODELS[mk]
            tok=AutoTokenizer.from_pretrained(checkpoint.repo,
                  revision=checkpoint.revision,trust_remote_code=True)
            for doc in docs:
                prompt=build_evidence_prompt(tok,checkpoint.family,doc)
                if len(prompt.token_ids)>cfg["max_prompt_tokens"]:
                    raise RuntimeError(
                        f"{mk}/{ds}/{doc['uid']}: prompt {len(prompt.token_ids)} "
                        f"exceeds {cfg['max_prompt_tokens']} without any truncation")
                # Same v3 short-answer target with no legacy math CoT reuse.
                text="Answer: "+doc["answer"].strip()+"\n"
                target=tok.encode(text,add_special_tokens=False)
                if len(target)<2:
                    raise RuntimeError("Gold continuation too short for future-only NLL")
                target=target[:cfg["max_target_tokens"]]
                if len(target)<2:
                    raise RuntimeError("Gold token truncation discarded the future")
                n_context=len(prompt.content_positions)
                if n_context<12:
                    raise RuntimeError("Context too short for controlled 10% retention")
                uid=doc["uid"]
                key=digest(f"{PROTOCOL}/{mk}/{ds}/{uid}")[:24]
                jobs.append({
                    "key":key,"model":mk,"revision":checkpoint.revision,
                    "dataset":ds,"uid":uid,
                    "prompt_text":prompt.text,
                    "prompt_sha256":digest(prompt.text),
                    "prompt_ids":list(prompt.token_ids),
                    "prompt_roles":list(prompt.roles),
                    "content_positions":list(prompt.content_positions),
                    "instruction_positions":list(prompt.instruction_positions),
                    "distractor_positions":list(prompt.distractor_positions),
                    "relevant_positions":list(prompt.relevant_positions),
                    "irrelevant_positions":list(prompt.irrelevant_positions),
                    "question_positions":list(prompt.question_positions),
                    "context_spans":[{**s,"positions":list(s["positions"])}
                                     for s in prompt.context_spans],
                    "target_ids":list(target),
                    "target_truncated":len(tok.encode(text,add_special_tokens=False))>len(target),
                    "gold_answer":doc["answer"].strip(),
                    "attributes":doc.get("attributes",{}),
                    "budget_counts":{str(b):context_budget(prompt.content_positions,b)
                                      for b in cfg["retention"]},
                    "protocol":PROTOCOL})
    expected=len(cfg["models"])*len(cfg["datasets"])*cfg["n_examples"]
    if len(jobs)!=expected or len(set(x["key"] for x in jobs))!=expected:
        raise RuntimeError("Incomplete/duplicate frozen request jobs")
    _json(run/"jobs.json",jobs)
    _json(run/"run_manifest.json",{
        "protocol":PROTOCOL,"code_sha":git_sha(),
        "config":cfg,"job_count":len(jobs),
        "dataset_sources":dataset_meta,"uids":uids,
        "model_revisions":{k:MODELS[k].revision for k in cfg["models"]},
        "statement":"Research Reference: full-size KV with future-only read masks, "
                    "not physical HBM compression or official H2O/SnapKV evaluation."})
    return {"jobs":len(jobs),"protocol":PROTOCOL,"sha":git_sha()}


def read_run(run: Path):
    manifest=_read(run/"run_manifest.json")
    jobs=_read(run/"jobs.json")
    if manifest["protocol"]!=PROTOCOL or manifest["code_sha"]!=git_sha():
        raise RuntimeError("Code/protocol SHA changed; use a NEW run directory")
    if len(jobs)!=manifest["job_count"]:
        raise RuntimeError("Run manifest job-count mismatch")
    if any(MODELS[j["model"]].revision!=j["revision"] for j in jobs):
        raise RuntimeError("Pinned model revision mismatch")
    if len({j["key"] for j in jobs})!=len(jobs):
        raise RuntimeError("Repeated job key")
    return manifest,jobs


def make_prompt(job: dict):
    from kvp.role_data import RolePrompt
    return RolePrompt(
        job["prompt_text"],tuple(job["prompt_ids"]),
        tuple(job["prompt_roles"]),tuple(job["content_positions"]),
        tuple(job["instruction_positions"]),tuple(job["distractor_positions"]),
        tuple(job["relevant_positions"]),tuple(job["irrelevant_positions"]),
        tuple(job["question_positions"]),
        tuple({**s,"positions":tuple(s["positions"])}
              for s in job["context_spans"]))


def spec_for(mk: str,config: dict):
    from kvp.depth_eviction_probe import DepthProbeSpec
    if mk.startswith("ouro"):
        return DepthProbeSpec("ouro",4,1.0,min_depth=2)
    return DepthProbeSpec("huginn",32,0.0,min_depth=2,cosine_dtype="float32")


def _model_context(mk: str):
    from kvp.models import load, make_runner
    pinned,tok,model=load(mk)
    return tok,make_runner(pinned,tok,model).bb


def _new_cached(bb,job,cfg):
    from kvp.loopzip_cached_eval import CachedFullPrefill
    return CachedFullPrefill(bb,spec_for(job["model"],cfg),make_prompt(job),
                            job["target_ids"],job["uid"],
                            max_generation_tokens=cfg["generation_tokens"])


def evaluate_screen(bb,tok,job: dict,cfg: dict):
    cache=_new_cached(bb,job,cfg)
    parity=cache.assert_clean_reference_parity()
    baseline=cache.score_gold(())
    baseline_greedy=(cache.greedy(tok,job["gold_answer"],())
                     if cfg["free_generate"] else None)
    attention=cache.clean_attention["content_attention_by_source"]
    selects=selection_manifest(
        job["content_positions"],attention,job["uid"],
        cfg["retention"],cfg["policies"],cfg["random_seeds"])
    records=[]
    for policy in selects:
        if len(policy["retained"])!=int(job["budget_counts"][str(policy["retention_requested"])]):
            raise RuntimeError("Policy does not meet identical budget")
        row=cache.score_gold(policy["evicted"])
        if abs(row["first_target_nll"]-baseline["first_target_nll"])>1e-4:
            raise RuntimeError("First gold token changed despite clean full prefill")
        g=(cache.greedy(tok,job["gold_answer"],policy["evicted"])
           if cfg["free_generate"] else None)
        if g and baseline_greedy and (
                not g["token_ids"] or g["token_ids"][0]!=baseline_greedy["token_ids"][0]):
            raise RuntimeError("Post-prefill eviction changed first generated token")
        records.append({
            "retention":policy["retention_requested"],
            "retention_requested":policy["retention_requested"],
            "policy":policy["policy"],"policy_key":policy["policy_key"],
            "seed":policy["seed"],"n_context":policy["n_context"],
            "n_retained":policy["n_retained"],
            "actual_retention":policy["actual_retention"],
            "retained":policy["retained"],"evicted":policy["evicted"],
            "attention_scores":policy["scores"],
            "nll_future":row["nll_future"],
            "delta_nll":row["nll_future"]-baseline["nll_future"],
            "first_target_nll":row["first_target_nll"],
            "greedy_correct":g["exact_match"] if g else None,
            "greedy_answer":g["extracted_answer"] if g else None,
            "greedy_generated_tokens":g["generated_tokens"] if g else None,
        })
    result={
        "meta":{"uid":job["uid"],"dataset":job["dataset"],
                "model":job["model"],"revision":job["revision"],
                "key":job["key"],"sha":git_sha(),
                "prompt_sha256":job["prompt_sha256"],
                "n_prompt":len(job["prompt_ids"]),
                "n_context":len(job["content_positions"]),
                "n_target":len(job["target_ids"]),
                "target_truncated":job["target_truncated"],
                "attributes":job["attributes"]},
        "parity":parity,
        "full":{"nll_future":baseline["nll_future"],
                "nll_all":baseline["nll_all"],
                "first_target_nll":baseline["first_target_nll"],
                "greedy_correct":baseline_greedy["exact_match"] if baseline_greedy else None,
                "greedy_answer":baseline_greedy["extracted_answer"] if baseline_greedy else None,
                "greedy_generated_tokens":baseline_greedy["generated_tokens"]
                    if baseline_greedy else None},
        "attention_by_source":attention,
        "screening":records}
    return result


def worker(run: Path, phase: str, worker_index: int):
    import torch
    cfg,jobs=read_run(run)
    if phase=="swap":
        tasks=_read(run/"swap_jobs.json")
        tasks_by_key={t["key"]:t for t in tasks}
        jobs=[j for j in jobs if j["key"] in tasks_by_key]
        raw=run/"swap_raw"
        claims=run/"swap_claims"
        errors=run/"swap_errors"
    else:
        tasks_by_key={}
        raw=run/"raw"
        claims=run/"claims"
        errors=run/"errors"
    for folder in (raw,claims,errors):
        folder.mkdir(exist_ok=True)
    current_model=None
    tok=bb=None
    while True:
        pending=[j for j in jobs if
                 not (raw/(j["key"]+".json")).exists() and
                 not (claims/(j["key"]+".lock")).exists()]
        if not pending:
            return
        pending.sort(key=lambda j:(j["model"]!=current_model,j["model"],j["key"]))
        job=pending[0]
        lock=claims/(job["key"]+".lock")
        try:
            fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd,"w") as file:
            file.write(f"worker={worker_index} pid={os.getpid()} gpu="
                       f"{os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        try:
            if current_model!=job["model"]:
                bb=tok=None
                torch.cuda.empty_cache()
                tok,bb=_model_context(job["model"])
                current_model=job["model"]
            outcome=(evaluate_screen(bb,tok,job,cfg["config"]) if phase=="screen"
                     else evaluate_swaps(bb,tok,job,cfg["config"],
                                         tasks_by_key[job["key"]],_read(
                                             run/"raw"/(job["key"]+".json"))))
            _json(raw/(job["key"]+".json"),outcome)
        except Exception:
            _json(errors/(job["key"]+".json"),{
                "key":job["key"],"phase":phase,"uid":job["uid"],
                "model":job["model"],"traceback":traceback.format_exc()})
            raise


def _percentile(values, q: float):
    if not values:
        return None
    a=sorted(values)
    pos=(len(a)-1)*q
    low=math.floor(pos);high=math.ceil(pos)
    return a[low]*(high-pos)+a[high]*(pos-low) if low!=high else a[low]


def _bootstrap_mean_ci(vals, seed: int = 20261031):
    import random
    if not vals:
        return None
    rng=random.Random(seed)
    boot=[sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals)
          for i in range(1000)]
    return [_percentile(boot,.025),_percentile(boot,.975)]


def _write_csv(path,rows,columns):
    with path.open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({col:row.get(col) for col in columns})


def analyze_screen(run: Path):
    cfg,jobs=read_run(run)
    raw=run/"raw"
    incomplete=[];bad=[];records=[];full=[];failure=[]
    n_each=len(cfg["config"]["retention"])*(
        sum(len(cfg["config"]["random_seeds"]) if p=="random" else 1
            for p in cfg["config"]["policies"]))
    for j in jobs:
        path=raw/(j["key"]+".json")
        if not path.exists():
            incomplete.append(j["key"])
            continue
        obj=_read(path)
        if (obj["meta"]["key"]!=j["key"] or obj["meta"]["sha"]!=cfg["code_sha"] or
            obj["meta"]["prompt_sha256"]!=j["prompt_sha256"]):
            bad.append(j["key"]+":provenance")
            continue
        if obj["parity"]["reference_max_abs_nll_diff"]>1e-4:
            bad.append(j["key"]+":parity")
        if len(obj["screening"])!=n_each:
            bad.append(j["key"]+":count")
        f=obj["full"]
        full.append({"model":j["model"],"dataset":j["dataset"],
                     "uid":j["uid"],"full_nll":f["nll_future"],
                     "full_correct":f["greedy_correct"],
                     "full_answer":f["greedy_answer"]})
        groups=defaultdict(dict)
        for sel in obj["screening"]:
            label=(float(sel["retention"]),sel["policy_key"])
            if label in groups["seen"]:
                bad.append(j["key"]+":duplicate")
            groups["seen"][label]=True
            b=float(sel["retention"])
            K=context_budget(j["content_positions"],b)
            if len(sel["retained"])!=K or len(sel["evicted"])!=len(j["content_positions"])-K:
                bad.append(j["key"]+":budget")
            if set(sel["retained"])|set(sel["evicted"])!=set(j["content_positions"]):
                bad.append(j["key"]+":partition")
            if abs(f["first_target_nll"]-sel["first_target_nll"])>1e-4:
                bad.append(j["key"]+":first_token")
            records.append({
                "model":j["model"],"dataset":j["dataset"],"uid":j["uid"],
                "budget":b,"policy":sel["policy_key"],
                "n_context":len(j["content_positions"]),
                "n_retained":K,"real_retention":K/len(j["content_positions"]),
                "full_nll":f["nll_future"],"policy_nll":sel["nll_future"],
                "delta_nll":sel["delta_nll"],
                "full_correct":f["greedy_correct"],
                "policy_correct":sel["greedy_correct"],
                "correct_to_wrong":(
                    bool(f["greedy_correct"] and not sel["greedy_correct"])
                    if f["greedy_correct"] is not None and
                       sel["greedy_correct"] is not None else None),
                "target_truncated":j["target_truncated"],
                "n_prompt":len(j["prompt_ids"])})
        for b in cfg["config"]["retention"]:
            per={x["policy_key"]:x for x in obj["screening"]
                 if float(x["retention"])==float(b)}
            cls=classify_failure(f["greedy_correct"],per)
            attention=[per[p] for p in ATTENTION_POLICIES if p in per]
            failure.append({
                "key":j["key"],"model":j["model"],"dataset":j["dataset"],
                "uid":j["uid"],"budget":b,"category":cls,
                "full_correct":f["greedy_correct"],
                "mean_delta_nll":per["mean_attn"]["delta_nll"],
                "max_attn_delta_nll":max(x["delta_nll"] for x in attention),
                "best_attn_delta_nll":min(x["delta_nll"] for x in attention),
                "best_attn_policy":min(attention,key=lambda x:x["delta_nll"])["policy"],
                "all_attention_wrong":(all(x["greedy_correct"] is False for x in attention)
                                       if f["greedy_correct"] is not None else None),
                "n_context":len(j["content_positions"])})
    errors=(list((run/"errors").glob("*.json")) if (run/"errors").exists() else [])
    complete=not (incomplete or bad or errors)
    out=run/"analysis";out.mkdir(exist_ok=True)
    _write_csv(out/"screening_quality.csv",records,[
        "model","dataset","uid","budget","policy","n_context","n_retained",
        "real_retention","full_nll","policy_nll","delta_nll",
        "full_correct","policy_correct","correct_to_wrong",
        "target_truncated","n_prompt"])
    _write_csv(out/"baseline_qa.csv",full,[
        "model","dataset","uid","full_nll","full_correct","full_answer"])
    _write_csv(out/"failure_cases.csv",failure,[
        "key","model","dataset","uid","budget","category","full_correct",
        "mean_delta_nll","max_attn_delta_nll","best_attn_delta_nll",
        "best_attn_policy","all_attention_wrong","n_context"])
    groups=defaultdict(list)
    for r in records:
        groups[(r["model"],r["dataset"],r["budget"],r["policy"])].append(r)
    summary=[]
    for (mk,ds,b,p),group in sorted(groups.items()):
        values=[x["delta_nll"] for x in group]
        correct=[x for x in group if x["full_correct"] is True]
        accuracy=[x for x in group if x["policy_correct"] is not None]
        summary.append({
            "model":mk,"dataset":ds,"budget":b,"policy":p,
            "n_requests":len(group),
            "mean_signed_delta_nll":sum(values)/len(values),
            "median_signed_delta_nll":_percentile(values,.5),
            "p90_signed_delta_nll":_percentile(values,.9),
            "request_bootstrap_ci95":_bootstrap_mean_ci(values),
            "n_full_correct":len(correct),
            "full_correct_to_wrong_rate":(
                sum(x["policy_correct"] is False for x in correct)/len(correct)
                if correct and all(x["policy_correct"] is not None for x in correct)
                else None),
            "policy_greedy_accuracy":(
                sum(x["policy_correct"] is True for x in accuracy)/len(accuracy)
                if accuracy else None)})
    failures_by_class=defaultdict(int)
    for f in failure:
        failures_by_class[f["category"]]+=1
    report={"complete":complete,"expected_jobs":len(jobs),
            "completed_jobs":len(jobs)-len(incomplete),
            "retention_ratios":cfg["config"]["retention"],
            "policy_groups":summary,"failure_counts":dict(failures_by_class),
            "notes":[
                "Joint removal of all evicted Context tokens at exactly the same K.",
                "Attention only from CLEAN Instruction/Question prefill, not future answer.",
                "HotpotQA excerpt is not official full-context evaluation.",
                "All KV remains allocated; no measured memory or serving latency savings.",
                "Accuracy fields null if --free-generate was disabled.",
                "No recurrent-specific causal claim can be made from failure screening alone."]}
    _json(out/"SUMMARY.json",report)
    _json(run/"completeness.json",{
        "complete":complete,"expected_jobs":len(jobs),
        "missing_jobs":incomplete,"invalid_jobs":bad,
        "worker_errors":[p.name for p in errors]})
    return report


def prepare_swaps(run: Path,max_failures_per_cell: int = 12,
                  max_controls_per_cell: int = 3,min_delta_nll: float = 0.05):
    cfg,jobs=read_run(run)
    if (run/"swap_jobs.json").exists():
        raise FileExistsError("Refusing to overwrite predeclared swap cohort")
    if not _read(run/"completeness.json")["complete"]:
        raise RuntimeError("Cannot prepare swaps from incomplete screening")
    bad=run/"analysis"/"failure_cases.csv"
    if not bad.exists():
        raise RuntimeError("Run --analyze before preparing swaps")
    with bad.open(newline="") as file:
        rows=list(csv.DictReader(file))
    cells=defaultdict(list)
    for r in rows:
        cells[(r["model"],r["dataset"],r["budget"])].append(r)
    per_job=defaultdict(list)
    for cell,rows in sorted(cells.items()):
        # Deliberately favor cases where even the best tested attention
        # aggregation suffers. This selection is OFFLINE and uses gold NLL.
        failed=[r for r in rows if
                (r["full_correct"]=="True" and
                 r["category"]=="all_attention_wrong") or
                (r["full_correct"]=="" and
                 float(r["best_attn_delta_nll"])>=min_delta_nll)]
        failed.sort(key=lambda r:(-float(r["best_attn_delta_nll"]),r["uid"]))
        selected=failed[:max_failures_per_cell]
        controls=[r for r in rows if r not in failed and
                  r["full_correct"]=="True" and
                  r["category"]=="no_attention_failure"]
        controls.sort(key=lambda r:(r["uid"]))
        selected+=controls[:max_controls_per_cell]
        for row in selected:
            per_job[row["key"]].append({
                "budget":float(row["budget"]),
                "category":row["category"],
                "anchor_policy":row["best_attn_policy"],
                "role":"control" if row in controls else "failure"})
    prepared=[{"key":j["key"],"uid":j["uid"],"model":j["model"],
               "dataset":j["dataset"],"swap_targets":per_job[j["key"]]}
              for j in jobs if j["key"] in per_job]
    _json(run/"swap_jobs.json",prepared)
    _json(run/"swap_selection.json",{
        "source_sha":cfg["code_sha"],"n_swap_requests":len(prepared),
        "max_failures_per_cell":max_failures_per_cell,
        "max_controls_per_cell":max_controls_per_cell,
        "min_delta_nll":min_delta_nll,
        "selection":"Full-correct + all-attention-wrong OR NLL-only best-attn loss; "
                    "with no-failure full-correct controls",
        "planned_policy":"best of four attention policies by offline NLL; never deployed"})
    return {"swap_jobs":len(prepared),"swap_budget_targets":
            sum(len(j["swap_targets"]) for j in prepared)}


def evaluate_swaps(bb,tok,job: dict,cfg: dict,task: dict,screen: dict):
    cache=_new_cached(bb,job,cfg)
    cache.assert_clean_reference_parity()
    baseline=cache.score_gold(())
    if abs(baseline["nll_future"]-screen["full"]["nll_future"])>1e-4:
        raise RuntimeError("Swap run changed full reference NLL")
    by={(x["retention"],x["policy"]):x for x in screen["screening"]}
    results=[]
    for target in task["swap_targets"]:
        key=(target["budget"],target["anchor_policy"])
        if key not in by:
            raise RuntimeError("Missing matching anchor policy")
        original=by[key]
        neighbors=swap_neighbors(
            original,job["content_positions"],
            cache.clean_attention["content_attention_by_source"],
            uid=job["uid"],max_retained=cfg["swap_max_retained"],
            max_evicted=cfg["swap_max_evicted"])
        scored=[]
        for candidate in neighbors:
            row=cache.score_gold(candidate["evicted"])
            if abs(row["first_target_nll"]-baseline["first_target_nll"])>1e-4:
                raise RuntimeError("Swap changed first target NLL")
            scored.append({
                **candidate,"policy":target["anchor_policy"],
                "budget":target["budget"],
                "category":target["category"],"role":target["role"],
                "n_retained":len(candidate["retained"]),
                "nll_future":row["nll_future"],
                "delta_nll":row["nll_future"]-baseline["nll_future"],
                "improvement_over_original_nll":
                    original["nll_future"]-row["nll_future"],
            })
        actual_original=scored[0]["nll_future"]
        if abs(actual_original-original["nll_future"])>1e-4:
            raise RuntimeError("Original attention policy NLL did not replay in swap phase")
        best=min(scored,key=lambda s:(s["nll_future"],s["swap_id"]))
        if cfg["swap_greedy"] and best["swap_id"]!="original":
            best["greedy"]=cache.greedy(tok,job["gold_answer"],best["evicted"])
        else:
            best["greedy"]=None
        results.append({
            "budget":target["budget"],"role":target["role"],
            "category":target["category"],"anchor_policy":target["anchor_policy"],
            "anchor_nll":actual_original,"best_nll":best["nll_future"],
            "best_swap_id":best["swap_id"],
            "local_oracle_gain":actual_original-best["nll_future"],
            "best_greedy":best["greedy"],
            "n_swaps_evaluated":len(scored)-1,
            "swaps":scored})
    return {"meta":{"key":job["key"],"uid":job["uid"],
                    "model":job["model"],"dataset":job["dataset"],
                    "sha":git_sha()},
            "swap_targets":results}


def analyze_swaps(run: Path):
    cfg,jobs=read_run(run)
    cohort=_read(run/"swap_jobs.json")
    path_by_key={j["key"]:j for j in jobs}
    out=run/"analysis";out.mkdir(exist_ok=True)
    rows=[];missing=[];bad=[]
    for task in cohort:
        key=task["key"]
        path=run/"swap_raw"/(key+".json")
        if not path.exists():
            missing.append(key)
            continue
        result=_read(path)
        if result["meta"]["key"]!=key or result["meta"]["sha"]!=cfg["code_sha"]:
            bad.append(key+":provenance")
            continue
        if len(result["swap_targets"])!=len(task["swap_targets"]):
            bad.append(key+":count")
        j=path_by_key[key]
        for target in result["swap_targets"]:
            K=context_budget(j["content_positions"],target["budget"])
            for sw in target["swaps"]:
                if len(sw["retained"])!=K or len(sw["evicted"])!=len(j["content_positions"])-K:
                    bad.append(key+":budget")
                if set(sw["retained"])|set(sw["evicted"])!=set(j["content_positions"]):
                    bad.append(key+":partition")
                rows.append({
                    "uid":j["uid"],"model":j["model"],"dataset":j["dataset"],
                    "budget":target["budget"],"category":target["category"],
                    "anchor_policy":target["anchor_policy"],"study_role":target["role"],
                    "swap_id":sw["swap_id"],"drop":sw["drop"],"restore":sw["restore"],
                    "swap_type":sw["type"],"n_retained":K,
                    "original_nll":target["anchor_nll"],
                    "swapped_nll":sw["nll_future"],
                    "delta_future_nll":sw["delta_nll"],
                    "gain":sw["improvement_over_original_nll"],
                    "best_swap_id":target["best_swap_id"],
                    "best_gain":target["local_oracle_gain"]})
    _write_csv(out/"swap_candidates.csv",rows,[
        "uid","model","dataset","budget","category","anchor_policy",
        "study_role","swap_id","drop","restore","swap_type","n_retained",
        "original_nll","swapped_nll","delta_future_nll","gain",
        "best_swap_id","best_gain"])
    groups=defaultdict(dict)
    for r in rows:
        g=(r["model"],r["dataset"],r["budget"],r["anchor_policy"],r["uid"])
        groups[g]["role"]=r["study_role"]
        groups[g]["best_gain"]=r["best_gain"]
        groups[g]["n_neighbors"]=groups[g].get("n_neighbors",0)+(r["swap_id"]!="original")
    by_cell=defaultdict(list)
    for (mk,ds,b,p,uid),r in groups.items():
        by_cell[(mk,ds,b,r["role"])].append(r["best_gain"])
    summary=[{
        "model":mk,"dataset":ds,"budget":b,"study_role":role,
        "n_requests":len(vals),
        "n_positive_rescue_requests":sum(v>1e-8 for v in vals),
        "mean_local_swap_nll_gain":sum(vals)/len(vals),
        "request_bootstrap_ci95":_bootstrap_mean_ci(vals)}
        for (mk,ds,b,role),vals in sorted(by_cell.items())]
    error_files=(list((run/"swap_errors").glob("*.json"))
                 if (run/"swap_errors").exists() else [])
    complete=not (missing or bad or error_files)
    report={"complete":complete,"expected_swap_jobs":len(cohort),
            "completed_swap_jobs":len(cohort)-len(missing),
            "groups":summary,
            "interpretation":"A positive local-swap gain demonstrates a tested better "
                             "equal-budget set; NOT a global oracle, robust policy or "
                             "proof that recurrence caused attention failure."}
    _json(out/"SWAP_SUMMARY.json",report)
    _json(run/"swap_completeness.json",{
        "complete":complete,"missing":missing,"invalid":bad,
        "worker_errors":[p.name for p in error_files]})
    return report


def execute(run: Path,phase: str,gpus: str):
    read_run(run)
    if phase=="swap" and not (run/"swap_jobs.json").exists():
        raise RuntimeError("Prepare a frozen swap cohort first")
    ids=[x.strip() for x in gpus.split(",") if x.strip()]
    if not ids or len(ids)!=len(set(ids)) or any(not s.isdigit() for s in ids):
        raise ValueError("Unique nonnegative CUDA GPU ordinals required")
    logs=run/"logs";logs.mkdir(exist_ok=True)
    handles=[];processes=[]
    try:
        for i,gpu in enumerate(ids):
            handle=(logs/f"{phase}_worker{i}.txt").open("a")
            proc=subprocess.Popen([
                sys.executable,str(Path(__file__).resolve()),
                "--worker",phase,"--worker-index",str(i),
                "--run-dir",str(run)],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu})
            handles.append(handle);processes.append(proc)
        codes=[p.wait() for p in processes]
        if any(code!=0 for code in codes):
            raise RuntimeError(f"Workers failed {codes}; check logs/errors/locks")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.wait()
        for f in handles:
            f.close()
    result=analyze_screen(run) if phase=="screen" else analyze_swaps(run)
    if not result["complete"]:
        raise RuntimeError("Incomplete experiment; inspect completeness.json")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    phase=ap.add_mutually_exclusive_group(required=True)
    for name in ("prepare","execute","analyze","prepare-swap","execute-swap",
                 "analyze-swap"):
        phase.add_argument("--"+name,action="store_true")
    phase.add_argument("--worker",choices=("screen","swap"))
    ap.add_argument("--worker-index",type=int,default=0)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--models",default="ouro-2.6b,huginn-0125")
    ap.add_argument("--datasets",default="controlled_retrieval,hotpotqa_distractor")
    ap.add_argument("--n-examples",type=int,default=20)
    ap.add_argument("--retention",default="0.10,0.30,0.50")
    ap.add_argument("--policies",default=",".join(ALL_POLICIES))
    ap.add_argument("--random-seeds",default="11,29,47")
    ap.add_argument("--controlled-facts",type=int,default=16)
    ap.add_argument("--hotpot-context-mode",
                    choices=("evidence_excerpt","full_context"),
                    default="evidence_excerpt")
    ap.add_argument("--hotpot-distractor-sentences",type=int,default=8)
    ap.add_argument("--max-prompt-tokens",type=int,default=1536)
    ap.add_argument("--max-target-tokens",type=int,default=64)
    ap.add_argument("--generation-tokens",type=int,default=32)
    ap.add_argument("--free-generate",action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--max-failures-per-cell",type=int,default=12)
    ap.add_argument("--max-controls-per-cell",type=int,default=3)
    ap.add_argument("--min-delta-nll",type=float,default=.05)
    ap.add_argument("--swap-max-retained",type=int,default=6)
    ap.add_argument("--swap-max-evicted",type=int,default=6)
    ap.add_argument("--swap-greedy",action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--gpu-ids",default="0,1,2,3,4,5,6,7")
    a=ap.parse_args()
    run=Path(a.run_dir).resolve()
    if a.prepare:
        models=tuple(x.strip() for x in a.models.split(",") if x.strip())
        datasets=tuple(x.strip() for x in a.datasets.split(",") if x.strip())
        policies=tuple(x.strip() for x in a.policies.split(",") if x.strip())
        budgets=tuple(float(x) for x in a.retention.split(",") if x.strip())
        seeds=tuple(int(x) for x in a.random_seeds.split(",") if x.strip())
        if (not models or len(set(models))!=len(models) or
            not set(models)<=set(MODELS_ALLOWED)):
            ap.error("Invalid/duplicate model set")
        if (not datasets or len(set(datasets))!=len(datasets) or
            not set(datasets)<=set(DATASETS_ALLOWED)):
            ap.error("Invalid/duplicate dataset set")
        if (not policies or len(set(policies))!=len(policies) or
            not set(policies)<=set(ALL_POLICIES) or
            "mean_attn" not in policies or
            not set(ATTENTION_POLICIES)<=set(policies)):
            ap.error("Require all four attention families for fair screening")
        if budgets!=BUDGETS:
            ap.error("Frozen phase-1 budgets are exactly 0.10,0.30,0.50")
        if not seeds or len(set(seeds))!=len(seeds):
            ap.error("Random seeds must be unique")
        if min(a.n_examples,a.max_prompt_tokens,a.max_target_tokens,
               a.generation_tokens,a.controlled_facts,
               a.hotpot_distractor_sentences,a.swap_max_retained,
               a.swap_max_evicted)<1:
            ap.error("Positive numerical bounds required")
        if a.max_target_tokens<2 or a.generation_tokens<2:
            ap.error("Need at least two target/generation tokens")
        cfg={"models":models,"datasets":datasets,"n_examples":a.n_examples,
             "seed":20261031,"retention":budgets,"policies":policies,
             "random_seeds":seeds,"controlled_facts":a.controlled_facts,
             "hotpot_context_mode":a.hotpot_context_mode,
             "hotpot_distractor_sentences":a.hotpot_distractor_sentences,
             "max_prompt_tokens":a.max_prompt_tokens,
             "max_target_tokens":a.max_target_tokens,
             "generation_tokens":a.generation_tokens,
             "free_generate":a.free_generate,
             "swap_max_retained":a.swap_max_retained,
             "swap_max_evicted":a.swap_max_evicted,
             "swap_greedy":a.swap_greedy}
        result=prepare(run,cfg)
    elif a.worker is not None:
        worker(run,a.worker,a.worker_index)
        return
    elif a.execute:
        execute(run,"screen",a.gpu_ids)
        result={"status":"screen complete"}
    elif a.execute_swap:
        execute(run,"swap",a.gpu_ids)
        result={"status":"swap complete"}
    elif a.analyze:
        result=analyze_screen(run)
    elif a.prepare_swap:
        result=prepare_swaps(run,a.max_failures_per_cell,
                             a.max_controls_per_cell,a.min_delta_nll)
    else:
        result=analyze_swaps(run)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
