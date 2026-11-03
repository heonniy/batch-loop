"""v9 causal FFN-gating / dynamic-depth pilot (one GPU process per model+dataset).

Requires the v5 frozen retrieval jobs and the pinned Huginn/Ouro checkpoints.
This is an EXECUTABLE pilot, not a GPU result. Use fresh --run-dir for each
model+dataset. No GPU task starts until --calibrate or --execute is called.

  python scripts/ffn_gate_causal_sweep.py --calibrate --model ouro-2.6b \
      --dataset controlled_retrieval --run-dir results/v9_ouro_cr
  python scripts/ffn_gate_causal_sweep.py --execute --model ouro-2.6b \
      --dataset controlled_retrieval --run-dir results/v9_ouro_cr
  python scripts/ffn_gate_causal_sweep.py --analyze --model ouro-2.6b \
      --dataset controlled_retrieval --run-dir results/v9_ouro_cr

Control: pinned FULL prefill -> LAST sharing applied to every historic
recurrent KV bank on generated-query reads; ordinary layers unchanged.
Only CONTEXT positions are evicted. Gate intervention changes exactly
the designated CORE FFN projections, not attention, up branch or KV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kvp.depth_eviction_probe import DepthProbeSpec
from kvp.ffn_gate_causal import GateCausalRunner, nll_recovery, paired_logit_kl, gate_trace_comparison

MODELS = ("huginn-0125", "ouro-2.6b", "ouro-1.4b")
DATASETS = ("controlled_retrieval", "hotpotqa_distractor")
V5 = ROOT / "results" / "20261030T164908Z_loopzip_failure_pilot20"
PROTOCOL = "ffn-gate-causal-v9-20261103"
Q_GRID = (0.25, 0.40, 0.55, 0.70, 0.85, 0.95)
COS_GRID = (0.00010, 0.00025, 0.00050, 0.0010, 0.0020, 0.0040, 0.0080, 0.0160)


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=float))
    os.replace(tmp, path)


def git_sha():
    return subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,check=True,
                          capture_output=True,text=True).stdout.strip()


def load_v5_jobs(model, dataset):
    jobs = json.loads((V5 / "jobs.json").read_text())
    source_model = "ouro-2.6b" if model == "ouro-1.4b" else model
    rows = sorted((j for j in jobs if j["model"] == source_model and
                   j["dataset"] == dataset), key=lambda j: j["uid"])
    if not rows:
        raise RuntimeError(f"No frozen v5 units for {model}/{dataset}")
    return rows


def last_plane_map(r):
    """Every historical recurrent read uses that source's LAST plane."""
    return np.full((r.R, r.T), r.R-1, dtype=np.int64)


def load_backbone(model):
    from kvp.models import eos_ids, load, make_runner
    spec, tok, mdl = load(model)
    return make_runner(spec, tok, mdl).bb, tok, eos_ids(spec, tok, mdl)


def make_runner(bb, doc, horizon):
    from loopzip_failure_screen import make_prompt
    p = make_prompt(doc)
    # Ouro-1.4B uses identical tokenisation to Ouro-2.6B (v8b validated).
    runner = GateCausalRunner(bb, p, doc["uid"], horizon=horizon)
    return runner, p


def _depths(exit_stats, family, threshold, min_depth=2):
    depths = []
    for ex in exit_stats:
        v = ex["cumulative"] if family == "ouro" else ex["distance_fp32"]
        R = len(v)
        if family == "ouro":
            d = next((r for r in range(min_depth,R+1)
                      if r==R or v[r-1]>=threshold),R)
        else:
            d = next((r for r in range(min_depth,R+1)
                      if r==R or v[r-1]<threshold),R)
        depths.append(d)
    return depths


def choose_thresholds(clean_exits: list[dict], family: str,
                      targets=(0.25,0.60)) -> dict:
    """Calibrate only on FULL-KV clean generated-query gate observations.

    Explicitly report saturated gates. Do NOT adjust after inspecting any
    eviction or correctness outcome. FP32 Huginn cosine is diagnostic arithmetic.
    """
    if not clean_exits:
        raise ValueError("No clean exit observations")
    grid = Q_GRID if family == "ouro" else COS_GRID
    R = len(clean_exits[0]["cumulative"] if family=="ouro" else clean_exits[0]["distance_fp32"])
    rows = []
    for threshold in grid:
        ds = _depths(clean_exits, family, threshold)
        hist = dict(sorted(Counter(ds).items()))
        rows.append({"threshold": threshold,
                     "early_fraction": sum(x<R for x in ds)/len(ds),
                     "mean_depth": statistics.mean(ds),
                     "distinct_depths": len(hist), "depth_hist": hist})
    chosen = []
    for goal in targets:
        eligible = [x for x in rows if x["threshold"] not in [z["threshold"] for z in chosen]]
        # Give a modest preference to non-degenerate histograms; even a
        # saturated model is reported rather than silently replaced.
        cand = min(eligible,key=lambda x:(abs(x["early_fraction"]-goal)
                       + (0.25 if x["distinct_depths"]<2 else 0.0),
                       x["threshold"]))
        chosen.append(cand | {"target_early_fraction":goal})
    return {"family":family, "dtype_exit":"q-exit" if family=="ouro" else "fp32_cosine_diagnostic",
            "n_clean_queries":len(clean_exits), "grid":rows, "selected":chosen}


def source_windows(prompt, uid, width: int, n_main: int = 4) -> list[dict]:
    """Freeze main source positions using ROLE-BLIND position/hash alone.

    Additional evidence/distractor windows are explicitly annotated ORACLE
    diagnostics, never model-selection features or learned attention scores.
    """
    pos=tuple(sorted(int(p) for p in prompt.content_positions))
    if width<1 or len(pos)<width:
        raise ValueError("Source window not feasible")
    valid = [tuple(range(p,p+width)) for p in pos if set(range(p,p+width)) <= set(pos)]
    if not valid:
        raise ValueError("No contiguous source windows")
    by_bin={k:[] for k in range(n_main)}
    for j,w in enumerate(valid):
        k=min(n_main-1, n_main*j//len(valid))
        h=hashlib.sha256(f"v9/{uid}/{w[0]}/{width}".encode()).hexdigest()
        by_bin[k].append((h,w))
    selected=[]
    used=set()
    for k in range(n_main):
        for _,w in sorted(by_bin[k]):
            if not used.intersection(w):
                selected.append({"positions":list(w),"stratum":"main_position_hash",
                                 "selection_only_from_clean":True})
                used.update(w)
                break
    # ORACLE DIAGNOSTIC STRATA: role annotations are used for sampling only.
    # At analysis time report them separately from the label-blind main arm.
    for role, label in (("relevant_content","oracle_evidence"),
                        ("irrelevant_content","oracle_distractor")):
        cand = [w for w in valid if all(prompt.roles[i]==role for i in w) and
                not used.intersection(w)]
        if cand:
            chosen = sorted(cand,key=lambda w:hashlib.sha256(
                f"v9oracle/{uid}/{label}/{w[0]}".encode()).hexdigest())[0]
            selected.append({"positions":list(chosen), "stratum":label,
                             "selection_only_from_clean":False})
            used.update(chosen)
    return selected


def gold_score(tok, gold, ids):
    from kvp.role_generation import score_free_answer
    return score_free_answer(tok, gold, {"token_ids": list(ids), "decoded_tokens":len(ids)})


def make_spec(bb, threshold):
    return DepthProbeSpec(bb.family,bb.R,float(threshold),min_depth=2,
                          cosine_dtype="float32")


def arm_summary(full, evict, arm):
    kl=paired_logit_kl(full["logprobs"],arm["logprobs"])
    return {"future_nll":float(arm["nll_future"]),
            "delta_nll_vs_clean":float(arm["nll_future"]-full["nll_future"]),
            "relative_nll_recovery":nll_recovery(
                full["nll_future"],evict["nll_future"],arm["nll_future"]),
            "kl_to_clean":kl["mean_future_kl"],
            "depths":arm["depths"],"total_depth":arm["total_depth"],
            "patches":arm["n_patched_components"],
            "mean_relative_patch": (statistics.mean(arm["patch_stats"].get("relative_patch_norm",[]))
                         if arm["patch_stats"].get("relative_patch_norm") else None)}


def _same_logits(a,b,tol=1e-5):
    if len(a["logprobs"])!=len(b["logprobs"]):
        raise AssertionError("different length")
    gap=max(float((x-y).abs().max()) for x,y in zip(a["logprobs"],b["logprobs"]))
    if gap>tol:
        raise AssertionError(f"Clean projection restoration should be parity: {gap}")
    return gap


def get_disjoint_jobs(model,dataset,n_cal,n_eval):
    docs=load_v5_jobs(model,dataset)
    if n_cal<1 or n_eval<1 or n_cal+n_eval>len(docs):
        raise ValueError("Calibration+evaluation count exceeds pinned cohort")
    return docs[:n_cal],docs[n_cal:n_cal+n_eval]


def calibrate(a):
    run=Path(a.run_dir)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Calibration run must have a NEW, empty directory")
    run.mkdir(parents=True,exist_ok=True)
    cal,ev=get_disjoint_jobs(a.model,a.dataset,a.n_cal,a.n_eval)
    bb,tok,eos=load_backbone(a.model)
    observations=[]
    for doc in cal:
        r,p=make_runner(bb,doc,max(len(doc["target_ids"])+2,a.greedy_tokens+2))
        r.reference(eos,frozen_ids=list(doc["target_ids"]))
        observations+=r.clean_last_exit_observations(last_plane_map(r))
    selected=choose_thresholds(observations,bb.family)
    manifest={"protocol":PROTOCOL,"model":a.model,"dataset":a.dataset,
              "code_sha_at_calibration":git_sha(),
              "calibration_uids":[x["uid"] for x in cal],
              "evaluation_uids":[x["uid"] for x in ev],
              "width":a.width,"greedy_tokens":a.greedy_tokens,
              "n_cal":a.n_cal,"n_eval":a.n_eval,
              "sharing":"last-bank-for-all-prompt-and-generated-histories",
              "thresholds":selected,
              "note":"calibrated on CLEAN full-depth LAST-sharing reference; dynamic evaluated under SAME shared last-bank reads; no eviction outcomes used"}
    dump(run/"manifest.json",manifest)
    print(json.dumps(selected,indent=2))


def evaluate_doc(bb,tok,eos,doc,manifest):
    r,p=make_runner(bb,doc,max(len(doc["target_ids"])+2,manifest["greedy_tokens"]+2))
    r.reference(eos,frozen_ids=list(doc["target_ids"]))
    last=last_plane_map(r)
    width=manifest["width"]
    candidates=source_windows(p,doc["uid"],width)
    # Fixed-depth FULL control with LAST sharing. Every counterfactual shares it.
    clean=r.run_gate_arm(sources=(),spec=r.spec,plane_map=last,mode="capture")
    clean_parity=r.run_gate_arm(sources=(),spec=r.spec,plane_map=last,mode="restore_gate",
                               reference=clean["ffn_trace"])
    _same_logits(clean,clean_parity)
    ref_greedy=r.greedy((),manifest["greedy_tokens"],eos,plane_map=last)
    gold=doc["gold_answer"]
    full_acc=gold_score(tok,gold,ref_greedy)
    full_dyn={}
    for q in manifest["thresholds"]["selected"]:
        spec=make_spec(bb,q["threshold"])
        dyn=r.run_gate_arm(sources=(),spec=spec,dynamic=True,plane_map=last,mode="capture")
        replay=r.run_gate_arm(sources=(),spec=spec,forced_depths=dyn["depths"],plane_map=last)
        _same_logits(dyn,replay)
        greedy=r.greedy_dynamic(sources=(),spec=spec,max_new=manifest["greedy_tokens"],
                                 eos_ids=eos,plane_map=last)
        full_dyn[str(q["threshold"])]={"live":dyn,"replay":replay,
            "greedy":greedy,"grade":gold_score(tok,gold,greedy["token_ids"])}
    out={"uid":doc["uid"],"model":manifest["model"],"dataset":manifest["dataset"],
         "n_prompt":r.P,"n_targets":len(r.ref_ids),"n_sources":len(candidates),
         "width":width,"full_fixed_grade":full_acc,
         "full_fixed_nll":clean["nll_future"],"clean_gate_restore_parity":True,
         "full_dynamic":[{"threshold":float(q),"depths":v["live"]["depths"],
                           "sum_depth":v["live"]["total_depth"],"nll_future":v["live"]["nll_future"],
                           "grade":v["grade"],"greedy_ids":v["greedy"]["token_ids"]}
                         for q,v in full_dyn.items()],"candidates":[]}
    for cand in candidates:
        src=cand["positions"]
        ev=r.run_gate_arm(sources=src,spec=r.spec,plane_map=last)
        restore={}
        trace_diff=gate_trace_comparison(clean["ffn_trace"],ev["ffn_trace"],bb.family)
        for mode in ("restore_gate","restore_up","shift_gate","restore_ffn"):
            cur=r.run_gate_arm(sources=src,spec=r.spec,plane_map=last,
                               mode=mode,reference=clean["ffn_trace"])
            restore[mode]=arm_summary(clean,ev,cur)
        # Patch restricted to the middle/late recurrence, not just final Loop.
        loop_groups = ([(1,8),(9,16),(17,24),(25,32)] if bb.family=="huginn"
                       else [(1,2),(3,4)])
        for lo,hi in loop_groups:
            cur=r.run_gate_arm(sources=src,spec=r.spec,plane_map=last,mode="restore_gate",
                               selected_loops=range(lo,hi+1),reference=clean["ffn_trace"])
            restore[f"gate_loops_{lo}_{hi}"]=arm_summary(clean,ev,cur)
        ev_fixed_greedy=r.greedy(src,manifest["greedy_tokens"],eos,plane_map=last)
        causal={"source":src,"stratum":cand["stratum"],"evict_fixed":arm_summary(clean,ev,ev),
                "restorations":restore,
                "evicted_fixed_grade":gold_score(tok,gold,ev_fixed_greedy),
                "gate_trace":trace_diff,
                "dynamic":[]}
        for q in manifest["thresholds"]["selected"]:
            k=str(q["threshold"])
            spec=make_spec(bb,q["threshold"])
            fl=full_dyn[k]["live"]
            el=r.run_gate_arm(sources=src,spec=spec,dynamic=True,plane_map=last)
            er=r.run_gate_arm(sources=src,spec=spec,forced_depths=fl["depths"],
                               plane_map=last)
            eg=r.run_gate_arm(sources=src,spec=spec,forced_depths=fl["depths"],
                               plane_map=last,mode="restore_gate",
                               reference=fl["ffn_trace"])
            # Diagnostic ORACLE: restoring gates from the clean FULL-FIXED
            # path at every depth, then letting the native exit gate re-decide.
            # This is not a deployable intervention or pure mediation estimate.
            egl=r.run_gate_arm(sources=src,spec=spec,dynamic=True,plane_map=last,
                                mode="restore_gate",reference=clean["ffn_trace"])
            gd=r.greedy_dynamic(sources=src,spec=spec,max_new=manifest["greedy_tokens"],
                                 eos_ids=eos,plane_map=last)
            if (gd["token_ids"][0]!=ref_greedy[0] or
                full_dyn[k]["greedy"]["token_ids"][0]!=ref_greedy[0]):
                raise RuntimeError("Post-prefill eviction or exit changed the already predicted first token")
            # full_live and evicted_live may have different generated KV and
            # stop depths; recorded replay holds the depth schedule fixed.
            diff=[b-a for a,b in zip(fl["depths"],el["depths"])]
            causal["dynamic"].append({
                "threshold":float(q["threshold"]),
                "full_depths":fl["depths"],"evicted_depths":el["depths"],
                "depth_deltas":diff,"n_depth_changed":sum(x!=0 for x in diff),
                "sum_depth_delta":sum(diff),
                "evicted_live_nll":el["nll_future"],
                "evicted_replay_nll":er["nll_future"],
                "gate_restored_replay_nll":eg["nll_future"],
                "gate_restored_replay_nll_recovery":er["nll_future"]-eg["nll_future"],
                "gate_restored_live_ORACLE_nll":egl["nll_future"],
                "gate_restored_live_ORACLE_depths":egl["depths"],
                "gate_restored_live_ORACLE_delta_depth_vs_evict":sum(a!=b for a,b in zip(egl["depths"],el["depths"])),
                "evicted_live_kl_to_clean":paired_logit_kl(fl["logprobs"],el["logprobs"])["mean_future_kl"],
                "full_live_grade":full_dyn[k]["grade"],
                "evicted_live_grade":gold_score(tok,gold,gd["token_ids"]),
                "evicted_live_greedy_ids":gd["token_ids"],
                "full_live_greedy_ids":full_dyn[k]["greedy"]["token_ids"],
                "evicted_fixed_correct":causal["evicted_fixed_grade"]["exact_match"],
                "live_vs_fixed_correctness_changed":gold_score(tok,gold,gd["token_ids"])["exact_match"] != causal["evicted_fixed_grade"]["exact_match"],
                "full_live_first_token":full_dyn[k]["greedy"]["token_ids"][0],
                "evicted_live_first_token":gd["token_ids"][0],
                "live_answer_changed":full_dyn[k]["grade"]["extracted_answer"] != gold_score(tok,gold,gd["token_ids"])["extracted_answer"],
                "full_live_total_compute":full_dyn[k]["greedy"]["total_depth"],
                "evicted_live_total_compute":gd["total_depth"],
            })
        out["candidates"].append(causal)
    return out


def execute(a):
    run=Path(a.run_dir)
    manifest=json.loads((run/"manifest.json").read_text())
    if manifest["model"]!=a.model or manifest["dataset"]!=a.dataset:
        raise RuntimeError("Model/dataset mismatch with frozen manifest")
    current = git_sha()
    # Require same CODE for all GPU receipts. An analysis/document-only
    # commit may change HEAD; record both and refuse silent mixing.
    pinned=manifest.get("execution_code_sha")
    if pinned is not None and pinned!=current:
        raise RuntimeError(f"Execution SHA drift {pinned} -> {current}: use a new run")
    if pinned is None:
        manifest["execution_code_sha"]=current
        dump(run/"manifest.json",manifest)
    cal,docs=get_disjoint_jobs(a.model,a.dataset,manifest["n_cal"],manifest["n_eval"])
    if [j["uid"] for j in docs]!=manifest["evaluation_uids"]:
        raise RuntimeError("Evaluation cohort changed since clean calibration")
    bb,tok,eos=load_backbone(a.model)
    for doc in docs:
        dest=run/"raw"/(doc["uid"].replace("/","_")+".json")
        if dest.exists():
            print("skipping complete",doc["uid"])
            continue
        try:
            output=evaluate_doc(bb,tok,eos,doc,manifest)
            dump(dest,output)
            print("complete",doc["uid"],flush=True)
        except Exception as exc:
            dump(run/"errors"/(doc["uid"].replace("/","_")+".json"),
                 {"uid":doc["uid"],"error":repr(exc)})
            raise


def analyze(a):
    manifest=json.loads((Path(a.run_dir)/"manifest.json").read_text())
    raw=Path(a.run_dir)/"raw"
    rows=[]
    for uid in manifest["evaluation_uids"]:
        p=raw/(uid.replace("/","_")+".json")
        if not p.exists():
            continue
        d=json.loads(p.read_text())
        for c in d["candidates"]:
            rec={"uid":uid,"model":manifest["model"],"dataset":manifest["dataset"],
                 "stratum":c["stratum"],"source":c["source"],
                 "eviction_penalty":c["evict_fixed"]["relative_nll_recovery"]["eviction_penalty"],
                 "full_fixed_correct":d["full_fixed_grade"]["exact_match"],
                 "evict_fixed_correct":c["evicted_fixed_grade"]["exact_match"],
                 "gate_trace_peak_relative":max((x.get("gate_max_relative_change",0.0)
                    for x in c.get("gate_trace",{}).get("by_loop",[])),default=0.0)}
            for name,x in c["restorations"].items():
                rec["recovery_"+name]=x["relative_nll_recovery"]["nll_recovered"]
                rec["fraction_"+name]=x["relative_nll_recovery"]["fraction_recovered"]
            for j,x in enumerate(c["dynamic"]):
                rec[f"depth_changed_{j}"]=x["n_depth_changed"]
                rec[f"nll_live_{j}"]=x["evicted_live_nll"]
                rec[f"nll_replay_{j}"]=x["evicted_replay_nll"]
                rec[f"nll_gate_replay_{j}"]=x["gate_restored_replay_nll"]
                rec[f"n_gate_restored_live_exit_changes_{j}"]=x["gate_restored_live_ORACLE_delta_depth_vs_evict"]
                rec[f"answer_changed_{j}"]=x["live_answer_changed"]
                rec[f"full_live_correct_{j}"]=x["full_live_grade"]["exact_match"]
                rec[f"evict_live_correct_{j}"]=x["evicted_live_grade"]["exact_match"]
                rec[f"evict_live_vs_fixed_answer_flip_{j}"]=x["live_vs_fixed_correctness_changed"]
            rows.append(rec)
    done=len(set(x["uid"] for x in rows))
    errors=list((Path(a.run_dir)/"errors").glob("*.json")) if (Path(a.run_dir)/"errors").exists() else []
    summary={"expected_units":len(manifest["evaluation_uids"]),"completed_units":done,
             "errors":[p.name for p in errors],
             "complete":done==len(manifest["evaluation_uids"]) and not errors,
             "n_candidate_windows":len(rows),
             "by_stratum":dict(Counter(r["stratum"] for r in rows)),
             "status":"NO PAPER CLAIM UNTIL GATES AND HELD-OUT TESTS PASS"}
    dump(Path(a.run_dir)/"analysis"/"summary.json",summary)
    if rows:
        import csv
        cols=list(dict.fromkeys(k for r in rows for k in r))
        with (Path(a.run_dir)/"analysis"/"per_window.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    for x in ("calibrate","execute","analyze"):
        g.add_argument("--"+x,action="store_true")
    ap.add_argument("--model",required=True,choices=MODELS)
    ap.add_argument("--dataset",required=True,choices=DATASETS)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--n-cal",type=int,default=3)
    ap.add_argument("--n-eval",type=int,default=5)
    ap.add_argument("--width",type=int,default=4)
    ap.add_argument("--greedy-tokens",type=int,default=24)
    a=ap.parse_args()
    if a.calibrate:
        calibrate(a)
    elif a.execute:
        execute(a)
    else:
        analyze(a)


if __name__=="__main__":
    main()
