#!/usr/bin/env python3
"""H200x8 Content->Instruction recurrent-sensitivity study.

One command per phase: --prepare (tokenizers/CPU), --execute (explicit GPU
opt-in), --analyze (CPU), or internal --worker. Fixed full-depth PREFILL is
mandatory; DECODE can be fixed or native dynamic with recorded-replay
counterfactuals. All returned savings are logical/scientific diagnostics,
not physically compacted KV cache memory.

B1 causal experiment; NO old 96-token truncation, NO old fewshot parser.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO=ROOT.parent
sys.path.insert(0,str(ROOT))

from kvp.role_data import RolePrompt
from kvp.role_span_oracle import role_evidence_oracle
from kvp.depth_eviction_probe import DepthProbeSpec
from kvp.specs import MODELS

MODEL_KEYS=("huginn-0125","ouro-1.4b","ouro-2.6b")
DATASET_KEYS=("controlled_retrieval","hotpotqa_distractor")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_sha() -> str:
    return subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,
        text=True,capture_output=True,check=True).stdout.strip()


def spec_for(key: str, cfg: dict) -> DepthProbeSpec:
    if key.startswith("ouro"):
        return DepthProbeSpec("ouro",4,cfg["ouro_q"],cfg["min_depth"])
    if key=="huginn-0125":
        return DepthProbeSpec("huginn",cfg["huginn_max_depth"],
                              cfg["huginn_cosine"],cfg["min_depth"],
                              cfg["huginn_cosine_dtype"])
    raise ValueError(key)


def prepare(run: Path, cfg: dict):
    """Freeze requests and candidates under the specified MAIN or ABLATION protocol.

    MAIN candidate sampling receives ONLY Context token positions; annotation
    roles and answer text are used solely for offline ablation/diagnostics.
    --prepare is CPU-only, downloading pinned dataset/tokenizers if needed.
    """
    from transformers import AutoTokenizer
    from kvp.role_data import build_evidence_prompt, choose_evidence_candidates
    from kvp.role_evidence_data import load_evidence_workload
    from kvp.role_main_candidates import choose_main_candidates

    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite any existing run directory")
    run.mkdir(parents=True, exist_ok=True)
    jobs = []
    subset_ids, dataset_sources = {}, {}
    for dataset in cfg["datasets"]:
        docs, meta = load_evidence_workload(
            dataset, cfg["n_examples"], cfg["seed"],
            controlled_facts=cfg["controlled_facts"],
            hotpot_context_mode=cfg["hotpot_context_mode"],
            hotpot_distractor_sentences=cfg["hotpot_distractor_sentences"])
        dataset_sources[dataset] = meta
        if len(docs) != cfg["n_examples"]:
            raise RuntimeError("Dataset returned an incomplete frozen sample")
        subset_ids[dataset] = [d["uid"] for d in docs]
        for model_key in cfg["models"]:
            checkpoint = MODELS[model_key]
            tok = AutoTokenizer.from_pretrained(
                checkpoint.repo, revision=checkpoint.revision,
                trust_remote_code=True)
            for doc in docs:
                prompt = build_evidence_prompt(tok, checkpoint.family, doc)
                if len(prompt.token_ids) > cfg["max_prompt_tokens"]:
                    raise RuntimeError(
                        f'Prompt too long for {model_key}/{dataset}/{doc["uid"]}: '
                        f'{len(prompt.token_ids)} > {cfg["max_prompt_tokens"]}; '
                        "Never silently trim or drop Supporting Facts")
                # Two-stage decoding target: "Answer" is first, then the
                # gold entity/code is predicted by future generated queries.
                # Avoid attributing an immutable first gold token to eviction.
                target_text = "Answer: " + doc["answer"].strip() + "\n"
                target_all = tok.encode(target_text, add_special_tokens=False)
                if len(target_all) < 2:
                    raise RuntimeError("Gold answer sequence too short")
                target_ids = target_all[:cfg["max_target_tokens"]]
                if len(target_ids) < 2:
                    raise RuntimeError("max_target_tokens must be >=2")
                salt = int(sha(f'{cfg["seed"]}:{dataset}:{doc["uid"]}'.encode())[:10], 16)
                if cfg["study"] == "main":
                    # The only selection input is the set of aligned Context
                    # token positions. No ground-truth label, gold answer,
                    # attention score, or masked future NLL is consulted.
                    candidates = choose_main_candidates(
                        prompt.content_positions, salt,
                        n_candidates=cfg["main_candidates"],
                        window_tokens=cfg["main_window_tokens"],
                        position_bins=cfg["main_position_bins"])
                    if any(c["role"] != "context" or c["answer_anchor_used"]
                           for c in candidates):
                        raise RuntimeError("MAIN candidates leaked oracle labels")
                elif cfg["study"] == "ablation":
                    candidates = choose_evidence_candidates(
                        prompt, salt, n_relevant=cfg["n_relevant"],
                        n_irrelevant=cfg["n_irrelevant"],
                        window_tokens=cfg["window_tokens"])
                    if len({c["role"] for c in candidates}) != 2:
                        raise RuntimeError("Missing Relevant/Irrelevant pair")
                else:
                    raise RuntimeError("Unknown study mode")
                for mode in cfg["decode_modes"]:
                    key = sha(f'v3/{cfg["study"]}/{model_key}/{dataset}/{doc["uid"]}/{mode}'.encode())[:24]
                    jobs.append({
                        "key": key, "model": model_key,
                        "revision": checkpoint.revision, "dataset": dataset,
                        "uid": doc["uid"], "decode_mode": mode,
                        "study": cfg["study"],
                        "prompt_text": prompt.text,
                        "prompt_ids": list(prompt.token_ids),
                        "prompt_roles": list(prompt.roles),
                        "content_positions": list(prompt.content_positions),
                        "instruction_positions": list(prompt.instruction_positions),
                        "distractor_positions": list(prompt.distractor_positions),
                        "relevant_positions": list(prompt.relevant_positions),
                        "irrelevant_positions": list(prompt.irrelevant_positions),
                        "question_positions": list(prompt.question_positions),
                        "context_spans": [
                            {**entry, "positions": list(entry["positions"])}
                            for entry in prompt.context_spans],
                        "target_ids": target_ids, "gold_answer": doc["answer"],
                        "target_total_length": len(target_all),
                        "target_truncated": len(target_all) > len(target_ids),
                        "kv_candidates": candidates,
                        "candidate_protocol": (
                            "all-context-position-stratified-label-blind-v3"
                            if cfg["study"] == "main"
                            else "annotated-evidence-distractor-v3"),
                        "source_attributes": doc.get("attributes", {}),
                        "prompt_sha256": sha(prompt.text.encode()),
                        "protocol": "context-instruction-question-main-ablation-v3",
                        "question_uid": doc["uid"]})
    expected = (len(cfg["models"])*len(cfg["datasets"])*
                cfg["n_examples"]*len(cfg["decode_modes"]))
    if len(jobs) != expected or len({j["key"] for j in jobs}) != expected:
        raise RuntimeError("Missing or duplicated job identity")
    (run/"jobs.json").write_text(json.dumps(jobs, indent=1))
    (run/"run_manifest.json").write_text(json.dumps({
        "source_sha": git_sha(), "config": cfg, "job_count": len(jobs),
        "subsets": subset_ids, "dataset_sources": dataset_sources,
        "model_revisions": {k: MODELS[k].revision for k in cfg["models"]},
        "protocol": "context-instruction-question-main-ablation-v3",
        "study": cfg["study"],
        "limitations": [
            "Full-sized R-plane reference KV; no physical HBM reclamation",
            "Teacher-forced NLL always; optional greedy Answer Exact Match requires --free-generate",
            "MAIN samples across all Context positions without using gold/support labels",
            "ABLATION samples ground-truth supporting/distractor windows only",
            "Gold labels stored as metadata are not input to online predictors",
            "HotpotQA evidence_excerpt is NOT the full-document HotpotQA benchmark",
            "First gold token predicted from unmodified final prompt query; excluded from future utility"
        ]}, indent=2))
    return {"jobs": len(jobs), "examples_per_model_dataset": cfg["n_examples"],
            "decode_modes": cfg["decode_modes"], "source_sha": git_sha(),
            "protocol": "context-instruction-question-main-ablation-v3",
            "study": cfg["study"]}



def manifest(run: Path):
    m=json.loads((run/"run_manifest.json").read_text())
    if m["source_sha"] != git_sha():
        raise RuntimeError("Source code changed; use frozen checkout or NEW run directory")
    jobs=json.loads((run/"jobs.json").read_text())
    if m["job_count"]!=len(jobs):
        raise RuntimeError("Job count mismatch")
    for job in jobs:
        if m["model_revisions"].get(job["model"])!=job["revision"]:
            raise RuntimeError("Checkpoint identity mismatch")
    return m,jobs


def worker(run: Path, index: int):
    import torch
    from kvp.models import load, make_runner

    m,jobs=manifest(run)
    raw=run/"raw";raw.mkdir(exist_ok=True)
    claims=run/"claims";claims.mkdir(exist_ok=True)
    errors=run/"errors";errors.mkdir(exist_ok=True)
    context=None; current_model=None
    while True:
        pending=[j for j in jobs if not (raw/(j["key"]+".json")).exists()
                 and not (claims/(j["key"]+".lock")).exists()]
        if not pending:
            return
        pending.sort(key=lambda j:(j["model"]!=current_model,j["model"],j["key"]))
        job=pending[0]
        lock=claims/(job["key"]+".lock")
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd,"w") as log:
            log.write(f'worker={index} pid={os.getpid()} gpu={os.environ.get("CUDA_VISIBLE_DEVICES")}\n')
        try:
            if current_model!=job["model"]:
                context=None
                torch.cuda.empty_cache()
                pinned,tok,model=load(job["model"])
                context=make_runner(pinned,tok,model)
                current_model=job["model"]
            p=RolePrompt(
                job["prompt_text"], tuple(job["prompt_ids"]),
                tuple(job["prompt_roles"]), tuple(job["content_positions"]),
                tuple(job["instruction_positions"]), tuple(job["distractor_positions"]),
                tuple(job["relevant_positions"]), tuple(job["irrelevant_positions"]),
                tuple(job["question_positions"]),
                tuple({**e, "positions": tuple(e["positions"])}
                      for e in job["context_spans"]))
            result=role_evidence_oracle(
                context.bb, job["uid"], job["revision"],
                spec_for(job["model"],m["config"]), p,
                job["target_ids"], job["kv_candidates"],
                decode_mode=job["decode_mode"],
                run_one_loop=m["config"]["one_loop"],
                live_exit=m["config"]["live_exit"],
                question_control=m["config"]["n_instruction"] > 0,
                study=m["config"]["study"],
                tokenizer=tok if m["config"]["free_generate"] else None,
                gold_answer=job["gold_answer"],
                max_generation_tokens=m["config"]["generation_tokens"])
            result["meta"]={
                "key":job["key"],"model":job["model"],
                "dataset":job["dataset"],"uid":job["uid"],
                "study":job["study"],
                "source_sha":m["source_sha"],"revision":job["revision"],
                "decode_mode":job["decode_mode"],
                "prompt_sha256":job["prompt_sha256"],
                "prompt_tokens":len(p.token_ids),
                "future_tokens":len(job["target_ids"]),
                "gold_truncated":job["target_truncated"],
                "gold_answer":job["gold_answer"],
                "source_attributes":job["source_attributes"],
                "gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
                "reference_full_allocated":True}
            temp=raw/("."+job["key"]+".tmp")
            temp.write_text(json.dumps(result))
            os.replace(temp,raw/(job["key"]+".json"))
        except Exception:
            (errors/(job["key"]+".json")).write_text(json.dumps({
                "key":job["key"],"uid":job["uid"],
                "model":job["model"],"traceback":traceback.format_exc()},indent=2))
            raise


def _rank(values):
    order=sorted(range(len(values)),key=lambda i:values[i])
    ranks=[0.0]*len(values)
    i=0
    while i<len(order):
        j=i
        while j+1<len(order) and values[order[j+1]]==values[order[i]]:
            j+=1
        for k in range(i,j+1):
            ranks[order[k]]=(i+j)/2
        i=j+1
    return ranks


def _corr(a,b):
    if len(a)<3:
        return None
    ra,rb=_rank(a),_rank(b)
    ma=sum(ra)/len(ra);mb=sum(rb)/len(rb)
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    den=(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5
    return num/den if den>1e-10 else None


def heldout_importance_test(rows: list[dict]) -> dict | None:
    """Frozen request-level split; no token from test request enters fitting.

    ONLINE = age + instruction-to-context attention + equal-width budget +
             source's own clean recurrent change and Question-loop alignment.
    ORACLE = ONLINE + prefill counterfactual sensitivity (expensive).
    Negative R2 is a legitimate held-out failure, not clamped away.
    """
    import numpy as np
    uids=sorted(set(row["uid"] for row in rows))
    if len(uids)<20:
        return None
    shuffled=sorted(uids,key=lambda u:sha(f"role-holdout-v1/{u}".encode()))
    cutoff=int(len(shuffled)*.7)
    train_uids=set(shuffled[:cutoff])
    train=[r for r in rows if r["uid"] in train_uids]
    test=[r for r in rows if r["uid"] not in train_uids]
    if len(train)<20 or len(test)<10:
        return None
    ytrain=np.array([r["postprefill_eviction_delta_future_nll"] for r in train],dtype=float)
    ytest=np.array([r["postprefill_eviction_delta_future_nll"] for r in test],dtype=float)
    models={
        "attention_age":("age_at_prefill_end","attention_mean","attention_max","n_masked_tokens"),
        "attention_age_plus_prefill_state":(
            "age_at_prefill_end","attention_mean","attention_max","n_masked_tokens",
            "source_mean_recurrent_change","attention_recurrent_alignment"),
        "ORACLE_attention_plus_masked_instruction":(
            "age_at_prefill_end","attention_mean","attention_max","n_masked_tokens",
            "prefill_persistent_final_error"),
    }
    scores={}
    denom=float(((ytest-ytest.mean())**2).sum())
    for name,features in models.items():
        x=np.array([[r[f] for f in features] for r in train],dtype=float)
        z=np.array([[r[f] for f in features] for r in test],dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            scores[name]=None
            continue
        means=x.mean(axis=0)
        std=x.std(axis=0)
        std[std<1e-8]=1.
        a=np.column_stack((np.ones(len(x)),(x-means)/std))
        b=np.column_stack((np.ones(len(z)),(z-means)/std))
        coef=np.linalg.lstsq(a,ytrain,rcond=None)[0]
        pred=b@coef
        scores[name]=float(1-((ytest-pred)**2).sum()/denom) if denom>1e-12 else None
    base=scores.get("attention_age")
    online=scores.get("attention_age_plus_prefill_state")
    oracle=scores.get("ORACLE_attention_plus_masked_instruction")
    return {"train_requests":len(train_uids),
            "test_requests":len(uids)-len(train_uids),
            "train_span_interventions":len(train),
            "test_span_interventions":len(test),
            "heldout_r2":scores,
            "online_incremental_r2":online-base
                if online is not None and base is not None else None,
            "oracle_incremental_r2":oracle-base
                if oracle is not None and base is not None else None,
            "split":"sha256(role-holdout-v1/uid), 70/30, frozen; negative R2 retained"}


def _request_paired_effect(rows: list[dict], seed: int = 20261030) -> dict:
    """Paired Relevant - Irrelevant signed future-NLL damage, bootstrapped by UID."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["uid"]][row["source_role"]].append(
            row["postprefill_eviction_delta_future_nll"])
    diffs = []
    for uid, roles in sorted(grouped.items()):
        a, b = roles.get("relevant_content"), roles.get("irrelevant_content")
        if a and b:
            diffs.append(sum(a)/len(a)-sum(b)/len(b))
    if not diffs:
        return {"n_paired_requests": 0, "mean_relevant_minus_irrelevant": None,
                "request_bootstrap_ci95": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(1000):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        draws.append(sum(sample)/len(sample))
    draws.sort()
    return {"n_paired_requests": len(diffs),
            "mean_relevant_minus_irrelevant": sum(diffs)/len(diffs),
            "fraction_relevant_greater": sum(x > 0 for x in diffs)/len(diffs),
            "request_bootstrap_ci95": [draws[25], draws[975]],
            "bootstrap": "1000 paired request-resamples; descriptive CI"}


def _attention_matched_pairs(rows: list[dict], relative_tolerance: float = .20) -> dict:
    """Closest Relevant/Irrelevant source attention within same request.

    Controls actual attention magnitude, masked KV-token count (enforced by
    oracle), and request identity; source age is still a possible confound.
    The 20% threshold is frozen before reading measured effects.
    """
    per_uid = defaultdict(list)
    for row in rows:
        per_uid[row["uid"]].append(row)
    accepted = []
    for uid, candidates in sorted(per_uid.items()):
        evidence = [c for c in candidates if c["source_role"] == "relevant_content"]
        irrelevant = [c for c in candidates if c["source_role"] == "irrelevant_content"]
        pairs = [(abs(a["attention_mean"]-b["attention_mean"]) /
                  max(a["attention_mean"], b["attention_mean"], 1e-12), a, b)
                 for a in evidence for b in irrelevant
                 if a["n_masked_tokens"] == b["n_masked_tokens"]]
        if not pairs:
            continue
        ratio, a, b = min(pairs, key=lambda x: (x[0], x[1]["candidate_id"], x[2]["candidate_id"]))
        if ratio <= relative_tolerance:
            accepted.append({
                "uid": uid, "attention_relative_gap": ratio,
                "age_difference": a["age_at_prefill_end"]-b["age_at_prefill_end"],
                "sensitivity_difference":
                    a["prefill_persistent_final_error"]-b["prefill_persistent_final_error"],
                "future_nll_difference":
                    a["postprefill_eviction_delta_future_nll"]-
                    b["postprefill_eviction_delta_future_nll"]})
    return {
        "relative_attention_tolerance": relative_tolerance,
        "matched_requests": len(accepted),
        "mean_attention_relative_gap": (sum(x["attention_relative_gap"] for x in accepted)
                                        /len(accepted)) if accepted else None,
        "spearman_recurrent_diff_vs_future_loss_diff": _corr(
            [x["sensitivity_difference"] for x in accepted],
            [x["future_nll_difference"] for x in accepted]),
        "notes": "Matched within request/width/attention; age difference is reported, not experimentally held equal."}


def _main_attention_matched_pairs(rows: list[dict],
                                  relative_tolerance: float = .20,
                                  max_relative_age_gap: float = .20) -> dict:
    """Unlabeled, within-request attention/recency matched pairs for MAIN.

    This analysis does NOT use evidence/distractor annotations. Select at most
    one pair per request to avoid treating correlated token windows as
    independent requests. All thresholds are fixed in the experiment plan.
    A nonmatch is a result, not a reason to widen limits after inspecting NLL.
    """
    per_uid = defaultdict(list)
    for row in rows:
        if row["source_role"] != "context":
            raise ValueError("MAIN attention matching received labeled candidate")
        per_uid[row["uid"]].append(row)
    matched = []
    for uid, candidates in sorted(per_uid.items()):
        available = []
        for i, a in enumerate(candidates):
            for b in candidates[i+1:]:
                if a["n_masked_tokens"] != b["n_masked_tokens"]:
                    continue
                attention_gap = abs(a["attention_mean"]-b["attention_mean"]) / max(
                    a["attention_mean"],b["attention_mean"],1e-12)
                max_age = max(a["age_at_prefill_end"],b["age_at_prefill_end"],1)
                age_gap = abs(a["age_at_prefill_end"]-b["age_at_prefill_end"]) / max_age
                if attention_gap <= relative_tolerance and age_gap <= max_relative_age_gap:
                    available.append((attention_gap + age_gap, attention_gap,
                                      age_gap, a, b))
        if not available:
            continue
        _, attention_gap, age_gap, a, b = min(
            available,key=lambda v:(v[0],v[3]["candidate_id"],v[4]["candidate_id"]))
        matched.append({
            "uid":uid,"attention_relative_gap":attention_gap,
            "relative_age_gap":age_gap,
            "recurrent_error_difference":a["prefill_persistent_final_error"]-
                                         b["prefill_persistent_final_error"],
            "future_nll_difference":a["postprefill_eviction_delta_future_nll"]-
                                    b["postprefill_eviction_delta_future_nll"]})
    return {
        "n_requests_with_pair":len(matched),
        "n_requests_total":len(per_uid),
        "max_attention_gap":relative_tolerance,
        "max_relative_age_gap":max_relative_age_gap,
        "mean_attention_gap":(sum(x["attention_relative_gap"] for x in matched)
                              /len(matched)) if matched else None,
        "mean_relative_age_gap":(sum(x["relative_age_gap"] for x in matched)
                                 /len(matched)) if matched else None,
        "spearman_recurrent_difference_vs_future_nll_difference":_corr(
            [x["recurrent_error_difference"] for x in matched],
            [x["future_nll_difference"] for x in matched]),
        "interpretation":"Matched on clean attention/age/width within request; "
                         "residual unobserved confounding remains."}


def analyze(run: Path):
    m,jobs=manifest(run)
    if m.get("protocol")!="context-instruction-question-main-ablation-v3":
        raise RuntimeError("Refusing to mix v1/v2 and v3 protocols")
    study=m["study"]
    if study not in ("main","ablation") or study != m["config"]["study"]:
        raise RuntimeError("Manifest study mismatch")
    rows=[]; controls=[]; masses=[]; missing=[]; replay_errors=[]
    candidate_count_mismatches=[]
    for job in jobs:
        if job["study"]!=study or job["protocol"]!=m["protocol"]:
            raise RuntimeError("Mixing experiment arm or protocol within one run")
        path=run/"raw"/(job["key"]+".json")
        if not path.exists():
            missing.append(job["key"])
            continue
        result=json.loads(path.read_text())
        if (result["meta"]["source_sha"] != m["source_sha"] or
            result["meta"]["uid"] != job["uid"] or
            result["meta"]["study"] != study):
            raise RuntimeError("Result provenance/study mismatch")
        b=result["baseline"]
        if b["replay_max_abs_token_nll_delta"] > 1e-4:
            replay_errors.append(job["key"])
        candidate_rows=result["candidate_rows"]
        if len(candidate_rows)!=len(job["kv_candidates"]):
            candidate_count_mismatches.append(job["key"])
        for r in candidate_rows:
            if r["study"] != study:
                raise RuntimeError("Candidate row has wrong experimental arm")
            if (study=="main" and r["source_role"]!="context" or
                study=="ablation" and r["source_role"]
                not in ("relevant_content","irrelevant_content")):
                raise RuntimeError("Ground-truth label selection leaked across arms")
            rows.append({"model":job["model"],"dataset":job["dataset"],
                         "decode_mode":job["decode_mode"],
                         "clean_greedy_exact_match":b.get("greedy_exact_match"),**r})
        for r in result["role_control_rows"]:
            controls.append({"model":job["model"],"dataset":job["dataset"],
                             "decode_mode":job["decode_mode"],**r})
        for r in b["attention_role_mass_by_loop"]:
            mass={**r}
            if study=="main":
                # Labels may exist in dataset metadata but must not appear
                # in primary MAIN measurement or candidate/predictor inputs.
                mass["relevant_mass"]=None
                mass["irrelevant_mass"]=None
            masses.append({"model":job["model"],"dataset":job["dataset"],
                           "decode_mode":job["decode_mode"],"uid":job["uid"],**mass})
    out=run/"analysis";out.mkdir(exist_ok=True)

    def write(name,data,columns):
        with (out/name).open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader()
            for row in data:
                wr.writerow({col:row.get(col) for col in columns})

    output_name="main_context_sensitivity.csv" if study=="main" else "ablation_evidence_sensitivity.csv"
    write(output_name,rows,[
        "study","model","dataset","decode_mode","uid","candidate_id",
        "source_role","selection_policy","source","source_token_position",
        "source_token_positions","n_masked_tokens","full_sentence_token_count",
        "age_at_prefill_end","answer_anchor_used","attention_by_loop",
        "attention_mean","attention_max","attention_recurrent_alignment",
        "source_mean_recurrent_change","prefill_instruction_error_by_loop",
        "prefill_persistent_final_error","prefill_persistent_recovery_ratio",
        "prefill_one_loop_error_by_loop","prefill_one_loop_final_error",
        "postprefill_eviction_delta_future_nll",
        "postprefill_eviction_delta_all_nll",
        "future_decode_recurrent_error_mean",
        "future_decode_recurrent_error_by_token",
        "clean_greedy_exact_match","postprefill_greedy_exact_match",
        "postprefill_greedy_answer_flip","live_exit_flip_fraction",
        "live_exit_signed_depth_shift","live_exit_delta_future_nll"])
    write("question_controls.csv",controls,[
        "model","dataset","decode_mode","uid","source_token_position",
        "source_role","postprefill_eviction_delta_future_nll"])
    write("instruction_attention_by_loop.csv",masses,[
        "model","dataset","decode_mode","uid","loop",
        "n_query_layer_observations","content_mass","relevant_mass",
        "irrelevant_mass","instruction_previous_mass","distractor_mass",
        "self_mass","content_mass_per_token"])
    groups=defaultdict(list)
    for row in rows:
        groups[(row["model"],row["dataset"],row["decode_mode"])].append(row)
    summaries=[]
    for (model,ds,mode),group in sorted(groups.items()):
        roles=(("context",) if study=="main"
               else ("relevant_content","irrelevant_content"))
        per_role={}
        for role in roles:
            candidates=[r for r in group if r["source_role"]==role]
            if not candidates:
                per_role[role]={"n_span_interventions":0}
                continue
            y=[r["postprefill_eviction_delta_future_nll"] for r in candidates]
            att=[r["attention_mean"] for r in candidates]
            mech=[r["prefill_persistent_final_error"] for r in candidates]
            valid=[r for r in candidates if r["source_mean_recurrent_change"] is not None]
            gen=[r for r in candidates
                 if r["postprefill_greedy_exact_match"] is not None]
            clean_correct=[r for r in gen if r["clean_greedy_exact_match"]]
            per_role[role]={
                "n_span_interventions":len(candidates),
                "n_requests":len({r["uid"] for r in candidates}),
                "n_generation_interventions":len(gen),
                "greedy_clean_accuracy":sum(r["clean_greedy_exact_match"] for r in gen)/len(gen)
                    if gen else None,
                "greedy_evicted_accuracy":sum(r["postprefill_greedy_exact_match"] for r in gen)/len(gen)
                    if gen else None,
                "greedy_correct_to_incorrect_rate":sum(
                    not r["postprefill_greedy_exact_match"] for r in clean_correct
                    )/len(clean_correct) if clean_correct else None,
                "mean_future_nll_delta":sum(y)/len(y),
                "positive_future_nll_damage_fraction":sum(x>0 for x in y)/len(y),
                "mean_future_decode_state_error":sum(
                    r["future_decode_recurrent_error_mean"] for r in candidates)/len(candidates),
                "spearman_attention_vs_future_damage":_corr(att,y),
                "spearman_prefill_oracle_vs_future_damage":_corr(mech,y),
                "spearman_online_attention_recurrence_alignment_vs_future_damage":_corr(
                    [r["attention_recurrent_alignment"] for r in candidates],y),
                "spearman_online_source_change_vs_future_damage":_corr(
                    [r["source_mean_recurrent_change"] for r in valid],
                    [r["postprefill_eviction_delta_future_nll"] for r in valid])
                }
        summary={
            "model":model,"dataset":ds,"decode_mode":mode,"study":study,
            "n_requests":len({r["uid"] for r in group}),
            "n_interventions":len(group),
            "per_role":per_role,
            "request_heldout_prediction":heldout_importance_test(group)}
        if study=="main":
            summary["attention_age_matched_main_pairs"]=_main_attention_matched_pairs(group)
        else:
            summary["paired_evidence_damage"]=_request_paired_effect(group)
            summary["attention_matched_evidence_pairs"]=_attention_matched_pairs(group)
        summaries.append(summary)
    error_files=([p.name for p in (run/"errors").glob("*.json")]
                 if (run/"errors").exists() else [])
    failures=bool(missing or replay_errors or error_files or candidate_count_mismatches)
    report={
        "study":study,"expected_jobs":len(jobs),
        "completed_jobs":len(jobs)-len(missing),"missing":missing,
        "parity_failures":replay_errors,
        "candidate_count_mismatches":candidate_count_mismatches,
        "groups":summaries,
        "limitations":[
            "MAIN candidates are position-stratified sampled Context windows, not an exhaustive all-token scan.",
            "MAIN uses neither supporting-fact annotations nor gold answer anchoring in candidate selection.",
            "ABLATION is ground-truth-labeled and cannot be evaluated as a deployable policy.",
            "Mechanism A is a counterfactual oracle and distinct from post-prefill intervention B.",
            "HotpotQA evidence_excerpt is NOT the full official HotpotQA benchmark.",
            "Fixed replay controls depth; live exit allows changed exit/quality.",
            "First target NLL is unaffected by post-prefill read mask and excluded.",
            "Greedy Exact Match only exists if --free-generate was enabled.",
            "All results are FULL-sized KV attention masks; no physical HBM or serving speedups.",
            "Single 70/30 heldout R2 and pairwise correlations are exploratory; repeat splits before a paper claim."]}
    (out/"SUMMARY.json").write_text(json.dumps(report,indent=2))
    (run/"completeness.json").write_text(json.dumps({
        "complete":not failures,"study":study,
        "missing_jobs":missing,"replay_failures":replay_errors,
        "candidate_count_mismatches":candidate_count_mismatches,
        "error_files":error_files},indent=2))
    return report



def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true")
    mode.add_argument("--execute",action="store_true")
    mode.add_argument("--analyze",action="store_true")
    mode.add_argument("--worker",type=int)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--models",default=",".join(MODEL_KEYS))
    ap.add_argument("--datasets",default=",".join(DATASET_KEYS))
    ap.add_argument("--decode-modes",default="fixed,dynamic")
    ap.add_argument("--study",choices=("main","ablation"),default="main")
    ap.add_argument("--main-candidates",type=int,default=8)
    ap.add_argument("--main-window-tokens",type=int,default=1)
    ap.add_argument("--main-position-bins",type=int,default=4)
    ap.add_argument("--n-examples",type=int,default=8)
    ap.add_argument("--n-relevant",type=int,default=2)
    ap.add_argument("--n-irrelevant",type=int,default=2)
    ap.add_argument("--window-tokens",type=int,default=12)
    ap.add_argument("--n-instruction",type=int,choices=(0,1),default=1)
    ap.add_argument("--controlled-facts",type=int,default=16)
    ap.add_argument("--hotpot-context-mode",
                    choices=("evidence_excerpt","full_context"),
                    default="evidence_excerpt")
    ap.add_argument("--hotpot-distractor-sentences",type=int,default=8)
    ap.add_argument("--max-prompt-tokens",type=int,default=1536)
    ap.add_argument("--max-target-tokens",type=int,default=64)
    ap.add_argument("--free-generate",action=argparse.BooleanOptionalAction,default=False)
    ap.add_argument("--generation-tokens",type=int,default=32)
    ap.add_argument("--seed",type=int,default=20261030)
    ap.add_argument("--ouro-q",type=float,default=.7)
    ap.add_argument("--huginn-cosine",type=float,default=1e-3)
    ap.add_argument("--huginn-max-depth",type=int,default=32)
    ap.add_argument("--huginn-cosine-dtype",choices=("float32","model"),default="float32")
    ap.add_argument("--min-depth",type=int,default=2)
    ap.add_argument("--one-loop",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--live-exit",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--gpu-ids",default="0,1,2,3,4,5,6,7")
    a=ap.parse_args()
    run=Path(a.run_dir).resolve()
    if a.prepare:
        models=[x.strip() for x in a.models.split(",") if x.strip()]
        datasets=[x.strip() for x in a.datasets.split(",") if x.strip()]
        modes=[x.strip() for x in a.decode_modes.split(",") if x.strip()]
        if (not models or len(models)!=len(set(models)) or
            not set(models)<=set(MODEL_KEYS)):
            ap.error("Invalid model set")
        if (not datasets or len(datasets)!=len(set(datasets)) or
            not set(datasets)<=set(DATASET_KEYS)):
            ap.error("Invalid dataset set")
        if (not modes or len(modes)!=len(set(modes)) or
            not set(modes)<={"fixed","dynamic"}):
            ap.error("Invalid decode modes")
        if min(a.n_examples,a.n_relevant,a.n_irrelevant,a.window_tokens,
               a.main_candidates,a.main_window_tokens,a.main_position_bins,
               a.max_prompt_tokens,a.max_target_tokens)<1:
            ap.error("Positive bounds required")
        if a.generation_tokens<2:
            ap.error("Generation requires >=2 tokens")
        if a.max_target_tokens<2:
            ap.error("Need at least two future gold tokens")
        if a.controlled_facts<4 or a.hotpot_distractor_sentences<1:
            ap.error("Need >=4 retrieval facts and >=1 HotpotQA distractor")
        cfg={"models":models,"datasets":datasets,"decode_modes":modes,
             "study":a.study,"main_candidates":a.main_candidates,
             "main_window_tokens":a.main_window_tokens,
             "main_position_bins":a.main_position_bins,
             "n_examples":a.n_examples,"n_relevant":a.n_relevant,
             "n_irrelevant":a.n_irrelevant,"window_tokens":a.window_tokens,
             "n_instruction":a.n_instruction,
             "controlled_facts":a.controlled_facts,
             "hotpot_context_mode":a.hotpot_context_mode,
             "hotpot_distractor_sentences":a.hotpot_distractor_sentences,
             "max_prompt_tokens":a.max_prompt_tokens,
             "max_target_tokens":a.max_target_tokens,"seed":a.seed,
             "free_generate":a.free_generate,
             "generation_tokens":a.generation_tokens,
             "ouro_q":a.ouro_q,"huginn_cosine":a.huginn_cosine,
             "huginn_max_depth":a.huginn_max_depth,
             "huginn_cosine_dtype":a.huginn_cosine_dtype,
             "min_depth":a.min_depth,"one_loop":a.one_loop,
             "live_exit":a.live_exit}
        for mk in models:
            spec_for(mk,cfg)
        print(json.dumps(prepare(run,cfg),indent=2))
    elif a.worker is not None:
        worker(run,a.worker)
    elif a.analyze:
        output=analyze(run)
        print(json.dumps({"expected_jobs":output["expected_jobs"],
            "completed_jobs":output["completed_jobs"],
            "missing":len(output["missing"]),
            "parity_failures":len(output["parity_failures"]),
            "groups":output["groups"]},indent=2))
    else:
        manifest(run)
        ids=[x.strip() for x in a.gpu_ids.split(",") if x.strip()]
        if not ids or len(ids)!=len(set(ids)) or any(not x.isdigit() for x in ids):
            ap.error("GPU ordinals must be unique nonnegative")
        (run/"logs").mkdir(exist_ok=True)
        ps=[]
        try:
            for i,gpu in enumerate(ids):
                handle=(run/"logs"/f"worker{i}.txt").open("a")
                proc=subprocess.Popen([
                    sys.executable,str(Path(__file__).resolve()),
                    "--worker",str(i),"--run-dir",str(run)],
                    cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu},
                    stdout=handle,stderr=subprocess.STDOUT)
                ps.append((proc,handle))
            status=[proc.wait() for proc,_ in ps]
            if any(code!=0 for code in status):
                raise RuntimeError(f"GPU worker failed {status}; inspect errors/claims")
        finally:
            for proc,handle in ps:
                if proc.poll() is None: proc.wait()
                handle.close()
        result=analyze(run)
        if result["missing"] or result["parity_failures"]:
            raise RuntimeError("Incomplete study or record/replay parity failure")
        print("All role-sensitivity jobs complete; see analysis/SUMMARY.json")


if __name__=="__main__":
    main()
