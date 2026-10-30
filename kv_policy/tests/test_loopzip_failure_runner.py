"""CPU fixture for v4 run manifest, screening completeness and swap selection."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from kvp.loopzip_policy import selection_manifest


class CharTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(x) for x in text]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":self.encode(text),
                "offset_mapping":[(i,i+1) for i in range(len(text))]}
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return "<system>"+messages[0]["content"]+"</system><user>"+messages[1]["content"]+"</user><assistant>"


def runner():
    spec=importlib.util.spec_from_file_location(
        "failure_runner_test",
        Path(__file__).resolve().parents[1]/"scripts/loopzip_failure_screen.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def doc(uid):
    return {
        "uid":uid,"answer":"VX-40592","question":"Which code belongs to record A?",
        "spans":[
            {"id":"a","role":"relevant_content",
             "text":"[A] Record A has access code VX-40592.","source":"A/0"},
            {"id":"b","role":"irrelevant_content",
             "text":"[B] Record B has access code VX-38310.","source":"B/0"},
            {"id":"c","role":"irrelevant_content",
             "text":"[C] Record C has access code VX-94831.","source":"C/0"}],
        "attributes":{"target_position":0}}


def fixture(tmp_path,monkeypatch):
    r=runner()
    t=CharTokenizer()
    monkeypatch.setitem(sys.modules,"transformers",types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:t)))
    monkeypatch.setitem(sys.modules,"kvp.role_evidence_data",
                        types.SimpleNamespace(load_evidence_workload=lambda name,n,seed,**kw:
                        ([doc(f"{name}-a"),doc(f"{name}-b")],
                         {"repo":"fixture","revision":"pinned","sha256":"abcd123"})))
    monkeypatch.setattr(r,"git_sha",lambda:"test-revision-sha")
    cfg={"models":("ouro-2.6b",),
         "datasets":("controlled_retrieval",),"n_examples":2,
         "seed":20261031,"retention":(.1,.3,.5),
         "policies":("mean_attn","last_attn","max_attn","recent_attn","random"),
         "random_seeds":(11,29,47),
         "controlled_facts":16,"hotpot_context_mode":"evidence_excerpt",
         "hotpot_distractor_sentences":8,
         "max_prompt_tokens":1536,"max_target_tokens":64,
         "generation_tokens":8,"free_generate":True,
         "swap_max_retained":6,"swap_max_evicted":6,"swap_greedy":False}
    run=tmp_path/"run"
    assert r.prepare(run,cfg)["jobs"]==2
    return r,run,cfg


def test_prepare_manifest_and_screen_cpu_analysis(tmp_path,monkeypatch):
    r,run,cfg=fixture(tmp_path,monkeypatch)
    manifest,jobs=r.read_run(run)
    assert manifest["code_sha"]=="test-revision-sha"
    assert len(jobs)==2
    (run/"raw").mkdir()
    for number,j in enumerate(jobs):
        attention={str(p):[.01+p/1000,.02+p/1000,.03+p/1000,.04+p/1000]
                   for p in j["content_positions"]}
        policies=selection_manifest(j["content_positions"],attention,j["uid"],
                                    cfg["retention"],cfg["policies"],cfg["random_seeds"])
        assert len(policies)==21
        is_failure=(number==0)
        rows=[]
        for sel in policies:
            is_attn=sel["policy"]!="random"
            # Request 0 is full-correct, every attention policy fails.
            correct=False if is_failure and is_attn else True
            loss=(.8 if is_failure and is_attn else .3)
            rows.append({**sel,
                "nll_future":loss,"delta_nll":loss-.2,
                "first_target_nll":.7,"greedy_correct":correct,
                "greedy_answer":"Wrong" if not correct else "VX-40592",
                "greedy_generated_tokens":4})
        r._json(run/"raw"/(j["key"]+".json"),{
            "meta":{"key":j["key"],"sha":"test-revision-sha",
                    "prompt_sha256":j["prompt_sha256"]},
            "parity":{"reference_max_abs_nll_diff":0.,
                      "repeated_cache_max_abs_nll_diff":0.},
            "full":{"nll_future":.2,"nll_all":.2,
                    "first_target_nll":.7,"greedy_correct":True,
                    "greedy_answer":"VX-40592"},
            "attention_by_source":attention,
            "screening":rows})
    report=r.analyze_screen(run)
    assert report["complete"]
    assert report["expected_jobs"]==2
    assert report["completed_jobs"]==2
    assert (run/"analysis"/"screening_quality.csv").exists()
    assert (run/"analysis"/"failure_cases.csv").exists()
    assert report["failure_counts"]["all_attention_wrong"]==3
    assert report["failure_counts"]["no_attention_failure"]==3
    swap=r.prepare_swaps(run,max_failures_per_cell=1,max_controls_per_cell=1)
    assert swap["swap_jobs"]==2
    tasks=r._read(run/"swap_jobs.json")
    assert all(len(t["swap_targets"])==3 for t in tasks)
    assert any(x["role"]=="failure" for t in tasks
               for x in t["swap_targets"])
    assert any(x["role"]=="control" for t in tasks
               for x in t["swap_targets"])
    with pytest.raises(FileExistsError):
        r.prepare_swaps(run)


def test_missing_screening_jobs_prevent_swap_preparation(tmp_path,monkeypatch):
    r,run,cfg=fixture(tmp_path,monkeypatch)
    result=r.analyze_screen(run)
    assert not result["complete"]
    assert len(r._read(run/"completeness.json")["missing_jobs"])==2
    with pytest.raises(RuntimeError,match="incomplete"):
        r.prepare_swaps(run)
