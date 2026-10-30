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
    """Freeze sampled questions, ground-truth evidence tags and matched KV windows.

    --prepare is CPU-only but downloads the PINNED HotpotQA validation Parquet
    and model tokenizers as needed. No GPU model weights are loaded.
    """
    from transformers import AutoTokenizer
    from kvp.role_data import build_evidence_prompt, choose_evidence_candidates
    from kvp.role_evidence_data import load_evidence_workload

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
                candidates = choose_evidence_candidates(
                    prompt, salt, n_relevant=cfg["n_relevant"],
                    n_irrelevant=cfg["n_irrelevant"],
                    window_tokens=cfg["window_tokens"])
                if len({c["role"] for c in candidates}) != 2:
                    raise RuntimeError("Missing Relevant/Irrelevant paired source")
                for mode in cfg["decode_modes"]:
                    key = sha(f'{model_key}/{dataset}/{doc["uid"]}/{mode}'.encode())[:24]
                    jobs.append({
                        "key": key, "model": model_key,
                        "revision": checkpoint.revision, "dataset": dataset,
                        "uid": doc["uid"], "decode_mode": mode,
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
                        "evidence_candidates": candidates,
                        "source_attributes": doc.get("attributes", {}),
                        "prompt_sha256": sha(prompt.text.encode()),
                        "protocol": "evidence-context-instruction-question-v2",
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
        "protocol": "evidence-context-instruction-question-v2",
        "limitations": [
            "Full-sized R-plane reference KV; no physical HBM reclamation",
            "Teacher-forced NLL, NOT free-generation Answer Exact Match",
            "Gold annotation used to choose diagnostic candidates, never online features",
            "HotpotQA evidence_excerpt is NOT the full-document HotpotQA benchmark",
            "First gold token predicted from unmodified final prompt query; excluded from future utility"
        ]}, indent=2))
    return {"jobs": len(jobs), "examples_per_model_dataset": cfg["n_examples"],
            "decode_modes": cfg["decode_modes"], "source_sha": git_sha(),
            "protocol": "evidence-context-instruction-question-v2"}



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
                job["target_ids"], job["evidence_candidates"],
                decode_mode=job["decode_mode"],
                run_one_loop=m["config"]["one_loop"],
                live_exit=m["config"]["live_exit"],
                question_control=m["config"]["n_instruction"] > 0)
            result["meta"]={
                "key":job["key"],"model":job["model"],
                "dataset":job["dataset"],"uid":job["uid"],
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

    ONLINE = age + observed Instruction-to-Content Attention +
             source's own (single clean prefill) recurrent change.
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
            "source_mean_recurrent_change"),
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
    return {"train_requests":len(train_uids),
            "test_requests":len(uids)-len(train_uids),
            "train_tokens":len(train),"test_tokens":len(test),
            "heldout_r2":scores,
            "split":"sha256(role-holdout-v1/uid), 70/30, frozen"}


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


def analyze(run: Path):
    m, jobs = manifest(run)
    if m.get("protocol") != "evidence-context-instruction-question-v2":
        raise RuntimeError("Refusing to mix v1 and v2 role protocols")
    rows, controls, masses, missing, replay_errors = [], [], [], [], []
    for job in jobs:
        path = run/"raw"/(job["key"]+".json")
        if not path.exists():
            missing.append(job["key"])
            continue
        result = json.loads(path.read_text())
        if (result["meta"]["source_sha"] != m["source_sha"] or
            result["meta"]["uid"] != job["uid"]):
            raise RuntimeError("Source/job provenance mismatch")
        b = result["baseline"]
        if b["replay_max_abs_token_nll_delta"] > 1e-4:
            replay_errors.append(job["key"])
        for r in result["candidate_rows"]:
            rows.append({"model": job["model"], "dataset": job["dataset"],
                         "decode_mode": job["decode_mode"], **r})
        for r in result["role_control_rows"]:
            controls.append({"model": job["model"], "dataset": job["dataset"],
                             "decode_mode": job["decode_mode"], **r})
        for r in b["attention_role_mass_by_loop"]:
            masses.append({"model": job["model"], "dataset": job["dataset"],
                           "decode_mode": job["decode_mode"], "uid": job["uid"], **r})
    out = run/"analysis"
    out.mkdir(exist_ok=True)

    def write(name, data, columns):
        with (out/name).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=columns)
            wr.writeheader()
            for row in data:
                wr.writerow({key: row.get(key) for key in columns})

    write("evidence_sensitivity.csv", rows, [
        "model","dataset","decode_mode","uid","candidate_id","source_role",
        "source","source_token_position","source_token_positions",
        "n_masked_tokens","full_sentence_token_count","age_at_prefill_end",
        "answer_anchor_used","attention_by_loop","attention_mean","attention_max",
        "source_mean_recurrent_change","prefill_instruction_error_by_loop",
        "prefill_persistent_final_error","prefill_persistent_recovery_ratio",
        "prefill_one_loop_error_by_loop","prefill_one_loop_final_error",
        "postprefill_eviction_delta_future_nll",
        "postprefill_eviction_delta_all_nll",
        "future_decode_recurrent_error_mean",
        "future_decode_recurrent_error_by_token",
        "live_exit_flip_fraction","live_exit_signed_depth_shift",
        "live_exit_delta_future_nll"])
    write("question_controls.csv", controls, [
        "model","dataset","decode_mode","uid","source_token_position",
        "source_role","postprefill_eviction_delta_future_nll"])
    write("instruction_attention_by_loop.csv", masses, [
        "model","dataset","decode_mode","uid","loop",
        "n_query_layer_observations","content_mass","relevant_mass",
        "irrelevant_mass","instruction_previous_mass","distractor_mass",
        "self_mass","content_mass_per_token"])

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"],row["dataset"],row["decode_mode"])].append(row)
    summaries = []
    for (model, dataset, mode), cohort in sorted(groups.items()):
        per_role = {}
        for role in ("relevant_content", "irrelevant_content"):
            g = [r for r in cohort if r["source_role"] == role]
            if not g:
                per_role[role] = {"n": 0}
                continue
            y = [r["postprefill_eviction_delta_future_nll"] for r in g]
            att = [r["attention_mean"] for r in g]
            mech = [r["prefill_persistent_final_error"] for r in g]
            valid_source = [r for r in g if r["source_mean_recurrent_change"] is not None]
            per_role[role] = {
                "n_span_interventions": len(g),
                "n_requests": len({r["uid"] for r in g}),
                "mean_future_nll_delta": sum(y)/len(y),
                "positive_future_nll_damage_fraction": sum(v > 0 for v in y)/len(y),
                "mean_replay_decode_state_error": sum(
                    r["future_decode_recurrent_error_mean"] for r in g)/len(g),
                "spearman_attention_vs_future_loss": _corr(att, y),
                "spearman_prefill_sensitivity_vs_future_loss": _corr(mech, y),
                "spearman_source_change_vs_future_loss": _corr(
                    [r["source_mean_recurrent_change"] for r in valid_source],
                    [r["postprefill_eviction_delta_future_nll"] for r in valid_source]),
            }
        summaries.append({
            "model": model, "dataset": dataset, "decode_mode": mode,
            "n_requests": len({r["uid"] for r in cohort}),
            "n_equal_width_span_interventions": len(cohort),
            "per_role": per_role,
            "paired_evidence_damage": _request_paired_effect(cohort),
            "request_heldout_prediction": heldout_importance_test(cohort)})
    result = {"expected_jobs": len(jobs),
              "completed_jobs": len(jobs)-len(missing),
              "missing": missing, "parity_failures": replay_errors,
              "groups": summaries,
              "limitations": [
                  "HotpotQA evidence_excerpt is NOT the original 10-paragraph benchmark.",
                  "Supporting-fact ground truth selects offline oracle probe windows; no evidence-label online feature.",
                  "All candidates per request use exactly matched masked token counts; ages and attention differ.",
                  "Paired signed future NLL can be <=0; no assumption that evidence eviction always hurts.",
                  "The first target token is predicted before post-prefill eviction and excluded.",
                  "Recurrent prefill counterfactual is an oracle, not a free online predictor.",
                  "No free-generation exact-match accuracy and no actual compact HBM/latency savings.",
                  "Recorded replay and live exit are separate dynamic comparisons."]}
    (out/"SUMMARY.json").write_text(json.dumps(result, indent=2))
    error_files = ([p.name for p in (run/"errors").glob("*.json")]
                   if (run/"errors").exists() else [])
    (run/"completeness.json").write_text(json.dumps({
        "complete": not missing and not replay_errors and not error_files,
        "missing_jobs": missing, "replay_failures": replay_errors,
        "error_files": error_files}, indent=2))
    return result



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
               a.max_prompt_tokens,a.max_target_tokens)<1:
            ap.error("Positive bounds required")
        if a.max_target_tokens<2:
            ap.error("Need at least two future gold tokens")
        if a.controlled_facts<4 or a.hotpot_distractor_sentences<1:
            ap.error("Need >=4 retrieval facts and >=1 HotpotQA distractor")
        cfg={"models":models,"datasets":datasets,"decode_modes":modes,
             "n_examples":a.n_examples,"n_relevant":a.n_relevant,
             "n_irrelevant":a.n_irrelevant,"window_tokens":a.window_tokens,
             "n_instruction":a.n_instruction,
             "controlled_facts":a.controlled_facts,
             "hotpot_context_mode":a.hotpot_context_mode,
             "hotpot_distractor_sentences":a.hotpot_distractor_sentences,
             "max_prompt_tokens":a.max_prompt_tokens,
             "max_target_tokens":a.max_target_tokens,"seed":a.seed,
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
