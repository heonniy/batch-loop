"""CPU-only v2 evidence-input / manifest / request-heldout tests."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class CharTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":self.encode(text),
                "offset_mapping":[(i,i+1) for i in range(len(text))]}

    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return ("<system>"+messages[0]["content"]+"</system><user>"+
                messages[1]["content"]+"</user><assistant>")


def load_runner():
    path=Path(__file__).resolve().parents[1]/"scripts/role_sensitivity_sweep.py"
    spec=importlib.util.spec_from_file_location("role_runner_test_v2",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def example(uid):
    return {
        "uid":uid, "dataset":"fixture", "answer":"VX-40592",
        "question":"Which access code belongs to record A?",
        "spans":[
            {"id":"e1","role":"relevant_content","text":
                "[A] Record A has access code VX-40592.","source":"A/0"},
            {"id":"e2","role":"relevant_content","text":
                "[B] Record B points to record A.","source":"B/0"},
            {"id":"d1","role":"irrelevant_content","text":
                "[C] Record C has access code VX-29202.","source":"C/0"},
            {"id":"d2","role":"irrelevant_content","text":
                "[D] Record D has access code VX-18764.","source":"D/0"}],
        "attributes":{"target_position":1}}


def test_prepare_freezes_ground_truth_roles_and_matched_kv(tmp_path,monkeypatch):
    runner=load_runner()
    tok=CharTokenizer()
    monkeypatch.setitem(sys.modules,"transformers",types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args,**kw:tok)))
    def fake_load(ds,n,seed,**opts):
        assert ds in ("controlled_retrieval","hotpotqa_distractor")
        assert n==2
        return [example(ds+"-a"),example(ds+"-b")],{
            "repo":"fixture","revision":"pinned-fixture",
            "sha256":"fixture-hash","context_mode":opts["hotpot_context_mode"]}
    monkeypatch.setitem(sys.modules,"kvp.role_evidence_data",
                        types.SimpleNamespace(load_evidence_workload=fake_load))
    monkeypatch.setattr(runner,"git_sha",lambda:"v2-fixture-code")
    cfg={"models":["huginn-0125","ouro-1.4b","ouro-2.6b"],
         "datasets":["controlled_retrieval","hotpotqa_distractor"],
         "decode_modes":["fixed","dynamic"],
         "n_examples":2,"n_relevant":2,"n_irrelevant":2,
         "window_tokens":12,"n_instruction":1,
         "controlled_facts":16,"hotpot_context_mode":"evidence_excerpt",
         "hotpot_distractor_sentences":8,
         "max_prompt_tokens":1536,"max_target_tokens":64,
         "seed":20261030,"ouro_q":.7,"huginn_cosine":.001,
         "huginn_max_depth":32,"huginn_cosine_dtype":"float32",
         "min_depth":2,"one_loop":True,"live_exit":True}
    run=tmp_path/"run"
    result=runner.prepare(run,cfg)
    assert result["jobs"]==24
    manifest,jobs=runner.manifest(run)
    assert manifest["dataset_sources"]["hotpotqa_distractor"]["revision"]=="pinned-fixture"
    assert len({(j["model"],j["dataset"],j["uid"],j["decode_mode"]) for j in jobs})==24
    for j in jobs:
        assert max(j["content_positions"])<min(j["instruction_positions"])
        assert max(j["instruction_positions"])==max(j["question_positions"])
        assert len(j["prompt_ids"])==len(j["prompt_roles"])
        assert len(j["evidence_candidates"])==4
        assert {c["role"] for c in j["evidence_candidates"]} == {
            "relevant_content","irrelevant_content"}
        assert len({len(c["positions"]) for c in j["evidence_candidates"]})==1
        assert all(all(j["prompt_roles"][p]==c["role"] for p in c["positions"])
                   for c in j["evidence_candidates"])
        assert j["target_truncated"] is False
        assert j["gold_answer"]=="VX-40592"
        assert j["protocol"]=="evidence-context-instruction-question-v2"


def test_request_level_holdout_is_not_per_span_leakage():
    r=load_runner()
    rows=[]
    for u in range(30):
        for i in range(2):
            rows.append({
                "uid":f"question-{u}","n_masked_tokens":12,
                "age_at_prefill_end":i+3,
                "attention_mean":i+.01*u,
                "attention_max":i+.02*u,
                "source_mean_recurrent_change":.1+.001*u,
                "attention_recurrent_alignment":.04+.003*u,
                "prefill_persistent_final_error":.2+.002*u,
                "postprefill_eviction_delta_future_nll":.03*u+.01*i})
    output=r.heldout_importance_test(rows)
    assert output["train_requests"]==21
    assert output["test_requests"]==9
    assert output["train_tokens"]==42
    assert output["test_tokens"]==18
    assert set(output["heldout_r2"])=={
        "attention_age","attention_age_plus_prefill_state",
        "ORACLE_attention_plus_masked_instruction"}


def test_request_paired_effect_is_signed_and_cluster_bootstrapped():
    r=load_runner()
    rows=[]
    for i in range(30):
        rows.extend([
            {"uid":str(i),"source_role":"relevant_content",
             "postprefill_eviction_delta_future_nll":0.15},
            {"uid":str(i),"source_role":"irrelevant_content",
             "postprefill_eviction_delta_future_nll":0.05}])
    stats=r._request_paired_effect(rows)
    assert stats["n_paired_requests"]==30
    assert abs(stats["mean_relevant_minus_irrelevant"]-.1)<1e-8
    assert stats["fraction_relevant_greater"]==1
    assert stats["request_bootstrap_ci95"][0]>.09
