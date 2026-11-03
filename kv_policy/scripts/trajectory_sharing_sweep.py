#!/usr/bin/env python3
"""V10: Full recurrent KV vs Ouro FIRST+LAST / Huginn LAST under Context eviction.

Runs on existing FULL-depth frozen v5 prompts; reads only AFTER prefill change.
No active GPU work is launched by --prepare or --analyze.

  cd kv_policy
  python scripts/trajectory_sharing_sweep.py --prepare --model huginn-0125 --dataset controlled_retrieval --run-dir results/v10_huginn_cr
  CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_sharing_sweep.py --execute --model huginn-0125 --dataset controlled_retrieval --run-dir results/v10_huginn_cr
  python scripts/trajectory_sharing_sweep.py --analyze --model huginn-0125 --dataset controlled_retrieval --run-dir results/v10_huginn_cr

The two policy-specific CLEAN fixed-depth trajectories calibrate the SAME
native exit thresholds; thresholds are frozen before eviction outcome analysis.
A post-prefill mask must never change the first generated token.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ffn_gate_causal_sweep import (load_backbone, load_v5_jobs,
                                   source_windows, gold_score)
from kvp.depth_eviction_probe import DepthProbeSpec
from kvp.trajectory_sharing_probe import TrajectorySharingRunner
from kvp.trajectory_sharing_policy import (
    choose_clean_thresholds, compare_exit_stats, make_read_map,
    paired_trajectory, policy_planes, same_depth_nll_interaction,
    sharing_name,
)

PROTOCOL="full-vs-model-sharing-trajectory-exit-v10-20261103"
MODELS=("huginn-0125","ouro-2.6b","ouro-1.4b")
DATASETS=("controlled_retrieval","hotpotqa_distractor")


def _write(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name("."+path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=float))
    os.replace(tmp,path)


def _sha():
    return subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,
                          capture_output=True,text=True,check=True).stdout.strip()


def _docs(model,dataset,n_cal,n_eval):
    docs=load_v5_jobs(model,dataset)
    if n_cal<2 or n_eval<1 or n_cal+n_eval>len(docs):
        raise ValueError("Need >=2 disjoint calibration UIDs + >=1 evaluation UID")
    return docs[:n_cal],docs[n_cal:n_cal+n_eval]


def _runner(bb,doc,max_new):
    from loopzip_failure_screen import make_prompt
    prompt=make_prompt(doc)
    return TrajectorySharingRunner(bb,prompt,doc["uid"],
                                   horizon=max(len(doc["target_ids"])+2,max_new+2))


def _maps(r):
    sh=sharing_name(r.bb.family)
    return {"full":make_read_map(r.bb.family,r.R,r.T,"full"),
            sh:make_read_map(r.bb.family,r.R,r.T,sh)}


def _spec(bb,q):
    return DepthProbeSpec(bb.family,bb.R,float(q),min_depth=2,
                          cosine_dtype="float32")


def _compare_logits(a,b,*,tol=1e-4):
    if a["depths"]!=b["depths"]:
        raise RuntimeError("Recorded replay used a different depth schedule")
    if len(a["logprobs"])!=len(b["logprobs"]):
        raise RuntimeError("Mismatched replay logit count")
    drift=max(float((x-y).abs().max()) for x,y in
              zip(a["logprobs"],b["logprobs"]))
    if drift>tol:
        raise RuntimeError(f"FULL live-vs-replay logits parity failed: {drift}")
    return drift


def _public_arm(arm):
    return {"depths":list(arm["depths"]),
            "total_depth":int(arm["total_depth"]),
            "nll_future":float(arm["nll_future"]),
            "future_target_nll":[float(x) for x in arm["token_losses"][1:]],
            "first_target_nll":float(arm["first_target_nll"])}


def _assert_first(full,other,tol=1e-4):
    if abs(full["first_target_nll"]-other["first_target_nll"])>tol:
        raise RuntimeError("A post-prefill intervention changed the first target")


def prepare(a):
    run=Path(a.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run")
    cal,ev=_docs(a.model,a.dataset,a.n_cal,a.n_eval)
    run.mkdir(parents=True,exist_ok=True)
    config={"protocol":PROTOCOL,"model":a.model,"dataset":a.dataset,
            "family":"huginn" if a.model.startswith("huginn") else "ouro",
            "calibration_uids":[x["uid"] for x in cal],
            "evaluation_uids":[x["uid"] for x in ev],
            "n_cal":a.n_cal,"n_eval":a.n_eval,
            "source_width":a.width,"greedy_tokens":a.greedy_tokens,
            "shared_policy":"last" if a.model.startswith("huginn") else "first_last",
            "ouro_firstlast_semantics":"first_for_r1_last_for_r2_plus",
            "code_sha":_sha(),
            "policy_groups":["full","model_specific_shared"],
            "notes":[
                "FULL-depth untouched prefill, causal post-prefill Context eviction only",
                "Huginn all historical recurrent query reads -> last source plane",
                "Ouro query recurrence 1 reads source plane 1, recurrences >=2 read last",
                "Current token always reads its own plane r; no lookahead into current token",
                "Same source positions and frozen target token IDs in all arms",
                "Source selection is position-hash blind; evidence/distractor oracle strata separate",
                "Full-sized KV reference + read maps, not HBM or latency evidence",
                "Dynamic threshold calibrated from both CLEAN KV policies with no eviction labels",
                "Huginn FP32-cosine over BF16 model hidden states is diagnostic only",
            ]}
    _write(run/"manifest.json",config)
    print(json.dumps({"run":str(run),"model":a.model,"dataset":a.dataset,
                      "calibration":len(cal),"evaluation":len(ev),
                      "policies":["full",config["shared_policy"]]},indent=2))


def calibrate(a):
    run=Path(a.run_dir).resolve()
    m=json.loads((run/"manifest.json").read_text())
    if (m["model"],m["dataset"])!=(a.model,a.dataset):
        raise RuntimeError("Frozen model/dataset mismatch")
    if "thresholds" in m:
        raise RuntimeError("Thresholds already frozen; use a fresh run")
    cal,ev=_docs(a.model,a.dataset,m["n_cal"],m["n_eval"])
    if [x["uid"] for x in cal]!=m["calibration_uids"]:
        raise RuntimeError("Calibration dataset mutated")
    bb,tok,eos=load_backbone(a.model)
    obs=[]
    policy_counts={}
    for doc in cal:
        r=_runner(bb,doc,m["greedy_tokens"])
        r.reference(eos,frozen_ids=list(doc["target_ids"]))
        for p,readmap in _maps(r).items():
            arm=r.trajectory_arm(plane_map=readmap)
            if p=="full":
                r.check_baseline_reference(arm)
            obs+=arm["exit_stats"]
            policy_counts[p]=policy_counts.get(p,0)+len(arm["exit_stats"])
    m["thresholds"]=choose_clean_thresholds(obs,bb.family)
    m["calibration_observations_by_policy"]=policy_counts
    m["code_sha_at_calibration"]=_sha()
    _write(run/"manifest.json",m)
    print(json.dumps({"selected":m["thresholds"]["selected"],
                      "n_observations":policy_counts},indent=2))


def _trace_shift(base,ev):
    return paired_trajectory(base["states"],ev["states"])


def _policy_clean_trajectory(full,shared):
    return paired_trajectory(full["states"],shared["states"])


def evaluate_doc(bb,tok,eos,doc,m):
    r=_runner(bb,doc,m["greedy_tokens"])
    r.reference(eos,frozen_ids=list(doc["target_ids"]))
    policy_maps=_maps(r)
    policies=tuple(policy_maps)
    thresholds=[float(q["threshold"]) for q in m["thresholds"]["selected"]]
    frozen_targets=list(doc["target_ids"])
    sources=source_windows(r.prompt,doc["uid"],m["source_width"])

    clean={}
    fixed_greedy={}
    clean_dyn={}
    # Native Coda/LM-head is evaluated at these FORCED query depths;
    # NLL damage always compares Full/Shared at the same depth.
    depth_probe_values=[8,16,24,32] if bb.family=="huginn" else [2,3,4]
    clean_depth_probe={}
    for policy,pmap in policy_maps.items():
        clean[policy]=r.trajectory_arm(plane_map=pmap)
        if policy=="full":
            r.check_baseline_reference(clean[policy])
        clean_repeat=r.trajectory_arm(plane_map=pmap)
        floor=_compare_logits(clean[policy],clean_repeat,tol=1e-4)
        fixed_greedy[policy]=r.greedy((),m["greedy_tokens"],eos,plane_map=pmap)
        clean_dyn[policy]={}
        for q in thresholds:
            spec=_spec(bb,q)
            live=r.trajectory_arm(plane_map=pmap,spec=spec,dynamic=True)
            replay=r.trajectory_arm(plane_map=pmap,spec=spec,
                                    forced_depths=live["depths"])
            parity=_compare_logits(live,replay)
            gg=r.greedy_dynamic(sources=(),spec=spec,
                                max_new=m["greedy_tokens"],eos_ids=eos,
                                plane_map=pmap)
            if gg["token_ids"][0]!=fixed_greedy[policy][0]:
                raise RuntimeError("Clean live exit changed prefill first token")
            clean_dyn[policy][q]={"live":live,"replay_max_gap":parity,
                                 "greedy":gg,"grade":gold_score(tok,doc["gold_answer"],gg["token_ids"])}
        clean[policy]["repeat_full_logits_floor"]=floor
        clean_depth_probe[policy]={}
        for d in depth_probe_values:
            arm=(clean[policy] if d==bb.R else
                 r.trajectory_arm(plane_map=pmap,forced_depths=[d]*len(r.ref_ids)))
            clean_depth_probe[policy][d]=arm

    out={"uid":doc["uid"],"model":m["model"],"dataset":m["dataset"],
         "full_prompt_tokens":r.P,"n_gold_targets":len(frozen_targets),
         "n_candidate_windows":len(sources),
         "shared_policy":m["shared_policy"],
         "clean_policy_comparison":{
            "shared_minus_full_nll":clean[policies[1]]["nll_future"]-clean["full"]["nll_future"],
            "full_vs_shared_trajectory":_policy_clean_trajectory(clean["full"],clean[policies[1]]),
            "full_vs_shared_KL":r.full_vs_evicted_kl(clean["full"],clean[policies[1]]),
         },
         "clean":{p:{"fixed":_public_arm(clean[p]),
                     "fixed_greedy_ids":fixed_greedy[p],
                     "fixed_grade":gold_score(tok,doc["gold_answer"],fixed_greedy[p]),
                     "repeat_logprob_floor":clean[p]["repeat_full_logits_floor"],
                     "live":{str(q):{
                         "arm":_public_arm(clean_dyn[p][q]["live"]),
                         "replay_logit_max_gap":clean_dyn[p][q]["replay_max_gap"],
                         "greedy_ids":clean_dyn[p][q]["greedy"]["token_ids"],
                         "greedy_depths":clean_dyn[p][q]["greedy"]["depths"],
                         "greedy_grade":clean_dyn[p][q]["grade"]}
                         for q in thresholds}} for p in policies},
         "candidates":[]}

    for candidate in sources:
        src=candidate["positions"]
        row={"source_positions":src,"stratum":candidate["stratum"],"by_policy":{}}
        evfixed={}
        evgreedy={}
        for policy,pmap in policy_maps.items():
            base=clean[policy]
            ev=r.trajectory_arm(sources=src,plane_map=pmap)
            _assert_first(base,ev)
            evfixed[policy]=ev
            evgreedy[policy]=r.greedy(src,m["greedy_tokens"],eos,plane_map=pmap)
            if evgreedy[policy][0]!=fixed_greedy[policy][0]:
                raise RuntimeError("Eviction changed prefill-first greedy token")
            kval=r.full_vs_evicted_kl(base,ev)
            row["by_policy"][policy]={
                "fixed":_public_arm(ev),
                "eviction_fixed_penalty":ev["nll_future"]-base["nll_future"],
                "fixed_kl":kval,
                "recurrent_trajectory":_trace_shift(base,ev),
                "depth_probe":{},
                "evicted_fixed_greedy_ids":evgreedy[policy],
                "evicted_fixed_grade":gold_score(tok,doc["gold_answer"],evgreedy[policy]),
                "clean_fixed_grade":out["clean"][policy]["fixed_grade"],
                "dynamic":{},
            }
            for d in depth_probe_values:
                pe=(ev if d==bb.R else
                    r.trajectory_arm(sources=src,plane_map=pmap,
                                     forced_depths=[d]*len(r.ref_ids)))
                pc=clean_depth_probe[policy][d]
                _assert_first(pc,pe)
                st=_trace_shift(pc,pe)
                row["by_policy"][policy]["depth_probe"][str(d)]={
                    "eviction_nll_penalty":pe["nll_future"]-pc["nll_future"],
                    "eviction_kl":r.full_vs_evicted_kl(pc,pe)["mean_scored_kl"],
                    "mean_final_trajectory_error":statistics.mean(
                        st["per_query_final_relative_error"]),
                    "per_query_final_trajectory_error":st["per_query_final_relative_error"]}
            for q in thresholds:
                spec=_spec(bb,q)
                live=r.trajectory_arm(sources=src,plane_map=pmap,spec=spec,dynamic=True)
                replay=r.trajectory_arm(sources=src,plane_map=pmap,spec=spec,
                                        forced_depths=clean_dyn[policy][q]["live"]["depths"])
                fl=clean_dyn[policy][q]["live"]
                _assert_first(base,live)
                _assert_first(base,replay)
                gd=r.greedy_dynamic(sources=src,spec=spec,max_new=m["greedy_tokens"],
                                    eos_ids=eos,plane_map=pmap)
                if gd["token_ids"][0]!=fixed_greedy[policy][0]:
                    raise RuntimeError("Dynamic eviction changed first predicted token")
                depthdiff=[b-a for a,b in zip(fl["depths"],live["depths"])]
                would=compare_exit_stats(base["exit_stats"],ev["exit_stats"],bb.family,q)
                row["by_policy"][policy]["dynamic"][str(q)]={
                    "evicted_live":_public_arm(live),
                    "evicted_replay":_public_arm(replay),
                    "full_live_depths":fl["depths"],
                    "depth_delta_by_query":depthdiff,
                    "depth_changed_fraction":sum(d!=0 for d in depthdiff)/len(depthdiff),
                    "sum_depth_delta":sum(depthdiff),
                    "would_exit_from_fixed":would,
                    "live_nll_eviction_penalty":live["nll_future"]-fl["nll_future"],
                    "replay_nll_eviction_penalty":replay["nll_future"]-fl["nll_future"],
                    "live_minus_replay_nll_descriptive":live["nll_future"]-replay["nll_future"],
                    "fixed_vs_live_eviction_interaction":(
                        live["nll_future"]-fl["nll_future"]
                        -(ev["nll_future"]-base["nll_future"])),
                    "greedy_token_ids":gd["token_ids"],
                    "greedy_depths":gd["depths"],
                    "greedy_grade":gold_score(tok,doc["gold_answer"],gd["token_ids"]),
                    "clean_live_grade":clean_dyn[policy][q]["grade"],
                    "answer_changed_vs_clean_live":(
                        gd["token_ids"]!=clean_dyn[policy][q]["greedy"]["token_ids"]),
                    "correctness_changed_vs_clean_live":(
                        gold_score(tok,doc["gold_answer"],gd["token_ids"])["exact_match"]
                        !=clean_dyn[policy][q]["grade"]["exact_match"]),
                }
        inter=same_depth_nll_interaction(
            clean["full"]["nll_future"],evfixed["full"]["nll_future"],
            clean[policies[1]]["nll_future"],evfixed[policies[1]]["nll_future"])
        row["fixed_eviction_sharing_interaction"]=inter
        row["depth_probe_interaction"]={str(d):{
            "full_penalty":row["by_policy"]["full"]["depth_probe"][str(d)]["eviction_nll_penalty"],
            "shared_penalty":row["by_policy"][policies[1]]["depth_probe"][str(d)]["eviction_nll_penalty"],
            "shared_minus_full_penalty":(
                row["by_policy"][policies[1]]["depth_probe"][str(d)]["eviction_nll_penalty"]-
                row["by_policy"]["full"]["depth_probe"][str(d)]["eviction_nll_penalty"])}
            for d in depth_probe_values}
        row["eviction_trajectory_sharing_interaction"]={
            "mean_relative_error_shared_minus_full_by_loop":(
                np.array(row["by_policy"][policies[1]]["recurrent_trajectory"]["mean_relative_error_by_loop"])
                -np.array(row["by_policy"]["full"]["recurrent_trajectory"]["mean_relative_error_by_loop"])).tolist()
        }
        out["candidates"].append(row)
    return out


def execute(a):
    run=Path(a.run_dir).resolve()
    m=json.loads((run/"manifest.json").read_text())
    if m["model"]!=a.model or m["dataset"]!=a.dataset:
        raise RuntimeError("Frozen model/dataset mismatch")
    if "thresholds" not in m:
        raise RuntimeError("Calibrate on separate clean prompts before execution")
    if m.get("execution_code_sha") and m["execution_code_sha"]!=_sha():
        raise RuntimeError("Code changed during the run; freeze a new run directory")
    if "execution_code_sha" not in m:
        m["execution_code_sha"]=_sha()
        _write(run/"manifest.json",m)
    _,jobs=_docs(a.model,a.dataset,m["n_cal"],m["n_eval"])
    if [d["uid"] for d in jobs]!=m["evaluation_uids"]:
        raise RuntimeError("Frozen evaluation UID mismatch")
    bb,tok,eos=load_backbone(a.model)
    for doc in jobs:
        output=run/"raw"/(doc["uid"].replace("/","_")+".json")
        if output.exists():
            print("completed earlier",doc["uid"],flush=True)
            continue
        try:
            _write(output,evaluate_doc(bb,tok,eos,doc,m))
            print("done",doc["uid"],flush=True)
        except Exception:
            _write(run/"errors"/(doc["uid"].replace("/","_")+".json"),
                   {"uid":doc["uid"],"traceback":traceback.format_exc()})
            raise


def analyze(a):
    run=Path(a.run_dir).resolve()
    m=json.loads((run/"manifest.json").read_text())
    rows=[]
    for uid in m["evaluation_uids"]:
        p=run/"raw"/(uid.replace("/","_")+".json")
        if not p.exists():continue
        d=json.loads(p.read_text())
        sh=m["shared_policy"]
        for c in d["candidates"]:
            dd=c["fixed_eviction_sharing_interaction"]
            row={"uid":uid,"model":m["model"],"dataset":m["dataset"],
                 "stratum":c["stratum"],"source_start":c["source_positions"][0],
                 "n_evicted":len(c["source_positions"]),
                 "full_eviction_penalty":dd["full_eviction_penalty"],
                 "shared_eviction_penalty":dd["shared_eviction_penalty"],
                 "interaction":dd["sharing_minus_full_eviction_penalty"],
                 "clean_sharing_shift":dd["sharing_baseline_nll_shift"]}
            for d,rr in c["depth_probe_interaction"].items():
                row["depth"+d+"_interaction"]=rr["shared_minus_full_penalty"]
                row["depth"+d+"_full_penalty"]=rr["full_penalty"]
                row["depth"+d+"_shared_penalty"]=rr["shared_penalty"]
            for policy in ("full",sh):
                arm=c["by_policy"][policy]
                row[policy+"_fixed_kl"]=arm["fixed_kl"]["mean_scored_kl"]
                row[policy+"_mean_trajectory_error"]=statistics.mean(
                    arm["recurrent_trajectory"]["mean_relative_error_by_loop"][1:])
                row[policy+"_fixed_correct"]=float(arm["evicted_fixed_grade"]["exact_match"])
                for j,q in enumerate(m["thresholds"]["selected"]):
                    x=arm["dynamic"][str(q["threshold"])]
                    row[f"{policy}_exit_changed_fraction_{j}"]=x["depth_changed_fraction"]
                    row[f"{policy}_would_exit_changed_fraction_{j}"]=x["would_exit_from_fixed"]["fraction_changed"]
                    row[f"{policy}_live_nll_penalty_{j}"]=x["live_nll_eviction_penalty"]
                    row[f"{policy}_live_replay_gap_{j}"]=x["live_minus_replay_nll_descriptive"]
                    row[f"{policy}_correctness_flip_{j}"]=float(x["correctness_changed_vs_clean_live"])
                    row[f"{policy}_live_correct_{j}"]=float(x["greedy_grade"]["exact_match"])
            rows.append(row)
    missing=[x for x in m["evaluation_uids"] if
             not (run/"raw"/(x.replace("/","_")+".json")).exists()]
    errors=sorted(p.name for p in (run/"errors").glob("*.json")) if (run/"errors").exists() else []
    summary={"model":m["model"],"dataset":m["dataset"],"shared_policy":m["shared_policy"],
             "completed_prompts":len(m["evaluation_uids"])-len(missing),
             "expected_prompts":len(m["evaluation_uids"]),"n_source_windows":len(rows),
             "missing":missing,"errors":errors,"complete":not missing and not errors,
             "status":"EXPLORATORY: no physical HBM savings or loop-specific causal claim"}
    _write(run/"analysis"/"summary.json",summary)
    if rows:
        path=run/"analysis"/"per_window.csv"
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    report=[
        f"# V10 Full vs {m['shared_policy']} recurrent KV: Context eviction × trajectories × exit",
        "",
        f"Model {m['model']}; {m['dataset']}; {summary['completed_prompts']}/{summary['expected_prompts']} prompts, {len(rows)} source windows.",
        "All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.",
        "The first generated token is predicted before post-prefill eviction and excluded from damage.",
        "",
        "| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def mean(rows,key):
        x=[float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        return statistics.mean(x) if x else float("nan")
    for stratum in sorted({x["stratum"] for x in rows}):
        grp=[x for x in rows if x["stratum"]==stratum]
        report.append("| "+stratum+" | "+str(len(grp))+
                      " | "+f"{mean(grp,'full_eviction_penalty'):.4g}"+
                      " | "+f"{mean(grp,'shared_eviction_penalty'):.4g}"+
                      " | "+f"{mean(grp,'interaction'):.4g}"+
                      " | "+f"{mean(grp,'full_mean_trajectory_error'):.4g}"+
                      " | "+f"{mean(grp,m['shared_policy']+'_mean_trajectory_error'):.4g} |")
    report+=["","## Fixed-depth recurrent damage curve (equal depth within each policy)",
             "| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |",
             "|---|---:|---:|---:|"]
    for d in ([8,16,24,32] if m["family"]=="huginn" else [2,3,4]):
        report.append(f"| {d} | {mean(rows,'depth'+str(d)+'_full_penalty'):.4g} | "
                      f"{mean(rows,'depth'+str(d)+'_shared_penalty'):.4g} | "
                      f"{mean(rows,'depth'+str(d)+'_interaction'):.4g} |")
    report+=["","## Native exit conditions (clean-calibrated, same threshold for both policies)"]
    for j,th in enumerate(m["thresholds"]["selected"]):
        report+=["",f"Threshold {th['threshold']}; CLEAN pooled early fraction {th['early_fraction']:.3f}; saturated={th['saturated']}",
                 "| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Accuracy flips vs clean live |",
                 "|---|---:|---:|---:|---:|---:|"]
        for pol in ("full",m["shared_policy"]):
            report.append(f"| {pol} | "+
                          " | ".join(f"{mean(rows,pol+'_'+k+'_'+str(j)):.4g}" for k in
                          ("exit_changed_fraction","would_exit_changed_fraction","live_nll_penalty",
                           "live_replay_gap","correctness_flip"))+" |")
    report+=["","Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.",
             "Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.",
             "Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.",
             "This reference uses full-size KV + read mapping (no actual HBM/TPOT savings)."]
    (run/"analysis"/"TRAJECTORY_SHARING_FINDINGS.md").write_text("\n".join(report)+"\n")
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    action=ap.add_mutually_exclusive_group(required=True)
    for x in ("prepare","calibrate","execute","analyze"):
        action.add_argument("--"+x,action="store_true")
    ap.add_argument("--model",required=True,choices=MODELS)
    ap.add_argument("--dataset",required=True,choices=DATASETS)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--n-cal",type=int,default=3)
    ap.add_argument("--n-eval",type=int,default=5)
    ap.add_argument("--width",type=int,default=4)
    ap.add_argument("--greedy-tokens",type=int,default=24)
    a=ap.parse_args()
    if a.prepare:prepare(a)
    elif a.calibrate:calibrate(a)
    elif a.execute:execute(a)
    else:analyze(a)


if __name__=="__main__":
    main()
