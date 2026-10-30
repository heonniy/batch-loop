#!/usr/bin/env python3
"""H200x8 B1 dynamic-depth KV-eviction experiment for Ouro AND Huginn.

Prepare CPU-only: --prepare. Execute pinned jobs only: --execute --gpu-ids ...
Score receipts without GPUs: --analyze. Never run CUDA at import/prepare time.
Experiments are per-token single-source NLL oracles, NOT multi-token eviction
serving results, and do NOT measure physically reclaimed HBM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KV_ROOT = HERE.parent
REPO = KV_ROOT.parent
sys.path.insert(0, str(KV_ROOT))

from kvp.depth_eviction_probe import DepthProbeSpec, choose_sources, single_source_oracle
from kvp.specs import MODELS

MODELS_OK = ("ouro-1.4b", "ouro-2.6b", "huginn-0125")
DATASETS_OK = ("gsm8k", "math500")


def sha_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def source_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          check=True, capture_output=True, text=True).stdout.strip()


def make_spec(model_key: str, cfg: dict) -> DepthProbeSpec:
    if model_key.startswith("ouro"):
        return DepthProbeSpec("ouro", 4, float(cfg["ouro_q"]), int(cfg["min_depth"]))
    if model_key == "huginn-0125":
        return DepthProbeSpec("huginn", int(cfg["huginn_max_depth"]),
                              float(cfg["huginn_cosine"]), int(cfg["min_depth"]))
    raise ValueError(model_key)


def prepare(run: Path, cfg: dict) -> dict:
    from transformers import AutoTokenizer
    from kvp.jobs import prepare_items
    import torch

    run.mkdir(parents=True, exist_ok=True)
    if any((run/p).exists() for p in ("jobs.json", "run_manifest.json", "raw")):
        raise FileExistsError("Existing run data: choose a NEW directory")
    jobs = []
    for mk in cfg["models"]:
        model = MODELS[mk]
        tok = AutoTokenizer.from_pretrained(model.repo, revision=model.revision,
                                            trust_remote_code=True)
        for ds in cfg["datasets"]:
            items, meta = prepare_items(mk, ds, tok,
                "pilot" if cfg["subset"]=="pilot" else "main", 1)
            for it in sorted(items, key=lambda x:x["uid"])[:cfg["n_examples"]]:
                original = it["prompt_ids"]
                prompt = original[-cfg["max_prompt_tokens"]:]
                target = it["target_ids"][:cfg["max_target_tokens"]]
                if len(prompt) < 6 or len(target) < 2:
                    # An eviction after completed prompt cannot affect the
                    # first gold token prediction; need >=2 future tokens.
                    continue
                uid = it["uid"]
                seed = int(sha_bytes(f'{cfg["seed"]}/{mk}/{ds}/{uid}'.encode())[:8], 16)
                jobs.append({
                    "key": sha_bytes(f"{mk}/{ds}/{uid}".encode())[:24],
                    "model": mk, "revision": model.revision, "dataset": ds, "uid": uid,
                    "prompt_ids": prompt, "target_ids": target,
                    "source_positions": choose_sources(len(prompt), cfg["n_candidates"], seed),
                    "full_prompt_token_count": len(original),
                    "context_truncated": len(original)>len(prompt),
                    "prompt_sha256": sha_bytes(it["prompt"].encode()),
                    "target_sha256": sha_bytes(it["target"].encode()),
                    "protocol": meta["protocol"]})
    if not jobs or len({j["key"] for j in jobs})!=len(jobs):
        raise RuntimeError("No unique jobs")
    counts={(mk,ds):sum(j["model"]==mk and j["dataset"]==ds for j in jobs)
            for mk in cfg["models"] for ds in cfg["datasets"]}
    if any(n!=cfg["n_examples"] for n in counts.values()):
        raise RuntimeError(f"Requested EXACT {cfg['n_examples']} examples per model/dataset, got {counts}")
    if {j["model"] for j in jobs}!=set(cfg["models"]):
        raise RuntimeError("At least one requested model lacks usable examples")
    (run/"jobs.json").write_text(json.dumps(jobs, indent=1))
    manifest={"source_sha":source_sha(), "config":cfg,
              "model_revisions":{k:MODELS[k].revision for k in cfg["models"]},
              "job_count":len(jobs), "python":sys.version.split()[0],
              "torch":torch.__version__,
              "warning":("B1 teacher-forced gold NLL on potentially LEFT-TRUNCATED "
                         "prompts. Not benchmark accuracy or measured HBM.")}
    (run/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def read_manifest(run: Path) -> dict:
    m=json.loads((run/"run_manifest.json").read_text())
    if m["source_sha"]!=source_sha():
        raise RuntimeError("Source SHA changed; use frozen checkout or NEW run dir")
    jobs=json.loads((run/"jobs.json").read_text())
    if len(jobs)!=m["job_count"]:
        raise RuntimeError("Job manifest mismatch")
    for j in jobs:
        if m["model_revisions"].get(j["model"])!=j["revision"]:
            raise RuntimeError("Checkpoint revision mismatch")
    return m


def worker(run: Path, index: int):
    import torch
    from kvp.models import load, make_runner
    m=read_manifest(run)
    jobs=json.loads((run/"jobs.json").read_text())
    raw=run/"raw";raw.mkdir(exist_ok=True)
    claims=run/"claims";claims.mkdir(exist_ok=True)
    errors=run/"errors";errors.mkdir(exist_ok=True)
    ctx=None; loaded=None
    while True:
        pending=[j for j in jobs if not (raw/(j["key"]+".json")).exists()
                 and not (claims/(j["key"]+".lock")).exists()]
        if not pending:
            break
        pending.sort(key=lambda j:(j["model"]!=loaded,j["model"],j["key"]))
        job=pending[0]
        try:
            fd=os.open(claims/(job["key"]+".lock"),
                       os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd,"w") as f:
            f.write(f'worker={index} pid={os.getpid()} gpu={os.environ.get("CUDA_VISIBLE_DEVICES")}\n')
        try:
            if loaded!=job["model"]:
                ctx=None
                torch.cuda.empty_cache()
                sp,tok,model=load(job["model"])
                ctx=make_runner(sp,tok,model)
                loaded=job["model"]
            spec=make_spec(job["model"],m["config"])
            ids=job["prompt_ids"]+job["target_ids"]
            result=single_source_oracle(ctx.bb,ids,job["uid"],spec,
                job["revision"],prompt_len=len(job["prompt_ids"]),
                candidates=job["source_positions"])
            result["meta"]={"model":job["model"],"dataset":job["dataset"],
                "uid":job["uid"],"key":job["key"],
                "prompt_sha256":job["prompt_sha256"],"revision":job["revision"],
                "source_sha":m["source_sha"],
                "gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
                "prompt_tokens":len(job["prompt_ids"]),
                "target_tokens":len(job["target_ids"]),
                "context_truncated":job["context_truncated"],
                "exit_rule":spec.rule().__dict__}
            temp=raw/("."+job["key"]+".tmp")
            temp.write_text(json.dumps(result))
            os.replace(temp,raw/(job["key"]+".json"))
        except Exception:
            (errors/(job["key"]+".json")).write_text(json.dumps({
                "key":job["key"],"model":job["model"],"uid":job["uid"],
                "traceback":traceback.format_exc()},indent=2))
            raise  # stale claim preserved for manual diagnosis


def ranks(seq):
    order=sorted(range(len(seq)),key=lambda i:seq[i])
    result=[0.0]*len(seq)
    i=0
    while i<len(order):
        j=i
        while j+1<len(order) and seq[order[j+1]]==seq[order[i]]:
            j+=1
        for k in range(i,j+1):
            result[order[k]]=(i+j)/2
        i=j+1
    return result


def pearson(x,y):
    if len(x)<3:
        return None
    a=sum(x)/len(x); b=sum(y)/len(y)
    num=sum((p-a)*(q-b) for p,q in zip(x,y))
    den=(sum((p-a)**2 for p in x)*sum((q-b)**2 for q in y))**.5
    return num/den if den>1e-12 else None


def incremental_r2(rows):
    """In-sample residual ΔR2 after age and prompt-observed attention. Diagnostic only."""
    import numpy as np
    if len(rows)<8:
        return None
    y=np.array([r["delta_nll"] for r in rows],dtype=float)
    X=np.array([[1.0,float(r["age_at_eval"]),
                 math.log1p(max(0.0,float(r["attention_pre_per_opportunity"])))]
                for r in rows],dtype=float)
    depth=np.array([float(r["exit_depth"]) for r in rows])[:,None]
    if len(set(depth.ravel()))<2:
        return None
    r0=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    full=np.concatenate((X,depth),axis=1)
    r1=y-full@np.linalg.lstsq(full,y,rcond=None)[0]
    den=float(((y-y.mean())**2).sum())
    return float((r0@r0-r1@r1)/den) if den>1e-12 else None


def analyze(run: Path):
    m=read_manifest(run)
    jobs=json.loads((run/"jobs.json").read_text())
    missing=[]; rows=[]; traces=[]
    for j in jobs:
        p=run/"raw"/(j["key"]+".json")
        if not p.exists():
            missing.append(j["key"])
            continue
        data=json.loads(p.read_text())
        if data["meta"]["uid"]!=j["uid"] or data["meta"]["source_sha"]!=m["source_sha"]:
            raise RuntimeError("Raw provenance mismatch")
        t=data["trace"]
        if data["reference"]["replay_max_abs_loss_delta"]>1e-4:
            raise RuntimeError("Baseline record/replay parity failed")
        ds=t["depths"]
        traces.append({"model":j["model"],"dataset":j["dataset"],"uid":j["uid"],
            "mean_depth":sum(ds)/len(ds),
            "fraction_max_depth":sum(x==t["max_depth"] for x in ds)/len(ds)})
        rows.extend([{**r,"model":j["model"],"dataset":j["dataset"]}
                     for r in data["candidate_rows"]])
    out=run/"analysis";out.mkdir(exist_ok=True)
    cols=["model","dataset","uid","source_position","source_token_id",
          "exit_depth","attention_pre","attention_pre_per_opportunity",
          "attention_future_ORACLE","delta_nll","baseline_nll",
          "evicted_nll","age_at_eval","replay_depths_identical"]
    with (out/"token_oracle.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in rows:
        groups[(r["model"],r["dataset"])].append(r)
    summaries=[]
    for (mk,ds),group in sorted(groups.items()):
        depth=[float(r["exit_depth"]) for r in group]
        importance=[float(r["delta_nll"]) for r in group]
        attn=[float(r["attention_pre_per_opportunity"]) for r in group]
        summaries.append({"model":mk,"dataset":ds,
            "n_source_tokens":len(group),
            "n_requests":len({r["uid"] for r in group}),
            "spearman_depth_vs_oracle_utility":pearson(ranks(depth),ranks(importance)),
            "spearman_attention_vs_oracle_utility":pearson(ranks(attn),ranks(importance)),
            "in_sample_incremental_r2_depth_over_age_attention":incremental_r2(group),
            "mean_depth":sum(depth)/len(depth),
            "mean_eviction_delta_nll":sum(importance)/len(importance),
            "positive_loss_fraction":sum(u>0 for u in importance)/len(importance)})
    (out/"depth_utility_summary.json").write_text(json.dumps({
        "complete":not missing,"missing_jobs":missing,
        "groups":summaries,"traces":traces,
        "caveats":["B1 gold suffix, prompt may be truncated; not task accuracy",
          "Single-token oracle, not simultaneous eviction serving policy",
          "Attention_pre excludes future and self attention but is observed during prompt",
          "Raw exit depths across model families are not comparable",
          "In-sample incremental R2 is not held-out predictive performance",
          "Reference GPU cache remains FULL-sized; no measured memory saving"]},indent=2))
    return {"expected":len(jobs),"complete":len(jobs)-len(missing),
            "oracle_rows":len(rows),"missing_jobs":len(missing)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    actions=ap.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare",action="store_true")
    actions.add_argument("--execute",action="store_true")
    actions.add_argument("--analyze",action="store_true")
    actions.add_argument("--worker",type=int)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--models",default=",".join(MODELS_OK))
    ap.add_argument("--datasets",default=",".join(DATASETS_OK))
    ap.add_argument("--subset",choices=("pilot","main"),default="main")
    ap.add_argument("--n-examples",type=int,default=100)
    ap.add_argument("--max-prompt-tokens",type=int,default=96)
    ap.add_argument("--max-target-tokens",type=int,default=16)
    ap.add_argument("--n-candidates",type=int,default=4)
    ap.add_argument("--seed",type=int,default=20261030)
    ap.add_argument("--ouro-q",type=float,default=0.7)
    ap.add_argument("--huginn-cosine",type=float,default=1e-3)
    ap.add_argument("--huginn-max-depth",type=int,default=32)
    ap.add_argument("--min-depth",type=int,default=2)
    ap.add_argument("--gpu-ids",default="0,1,2,3,4,5,6,7")
    a=ap.parse_args()
    run=Path(a.run_dir).resolve()
    if a.prepare:
        models=[x.strip() for x in a.models.split(",") if x.strip()]
        datasets=[x.strip() for x in a.datasets.split(",") if x.strip()]
        if not models or len(models)!=len(set(models)) or not set(models)<=set(MODELS_OK):
            ap.error("Invalid models")
        if not datasets or len(datasets)!=len(set(datasets)) or not set(datasets)<=set(DATASETS_OK):
            ap.error("Invalid datasets")
        if min(a.n_examples,a.max_prompt_tokens,a.max_target_tokens,a.n_candidates)<1:
            ap.error("Positive budgets required")
        cfg={"models":models,"datasets":datasets,"subset":a.subset,
             "n_examples":a.n_examples,"max_prompt_tokens":a.max_prompt_tokens,
             "max_target_tokens":a.max_target_tokens,"n_candidates":a.n_candidates,
             "seed":a.seed,"ouro_q":a.ouro_q,"huginn_cosine":a.huginn_cosine,
             "huginn_max_depth":a.huginn_max_depth,"min_depth":a.min_depth}
        for model in models:
            make_spec(model,cfg)
        print(json.dumps(prepare(run,cfg),indent=2))
    elif a.worker is not None:
        worker(run,a.worker)
    elif a.analyze:
        print(json.dumps(analyze(run),indent=2))
    else:
        read_manifest(run)
        ids=[x.strip() for x in a.gpu_ids.split(",") if x.strip()]
        if not ids or len(ids)!=len(set(ids)) or any(not i.isdigit() for i in ids):
            ap.error("GPU ordinals must be unique nonnegative integers")
        (run/"logs").mkdir(exist_ok=True)
        procs=[]
        try:
            for rank,gpu in enumerate(ids):
                log=(run/"logs"/f"worker{rank}.txt").open("a")
                proc=subprocess.Popen([
                    sys.executable,str(Path(__file__).resolve()),
                    "--run-dir",str(run),"--worker",str(rank)],
                    cwd=KV_ROOT, env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu},
                    stdout=log,stderr=subprocess.STDOUT)
                procs.append((proc,log))
            exit_codes=[p.wait() for p,_ in procs]
            if any(code!=0 for code in exit_codes):
                raise RuntimeError(f"Worker exit codes {exit_codes}; inspect errors/claims/logs")
        finally:
            for proc,log in procs:
                if proc.poll() is None:
                    proc.wait()
                log.close()
        print(json.dumps(analyze(run),indent=2))


if __name__=="__main__":
    main()
