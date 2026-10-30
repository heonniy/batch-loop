"""CPU-only fake-backbone tests of one-prefill replay and future-only masks.

The fake recurrence exercises the REAL RoleProbeKV read/write/mask storage
without depending on large Huginn/Ouro checkpoints. Separate HF-backed model
tests below are mandatory before the H200 main run.
"""
from __future__ import annotations

import pytest
import torch

from kvp.role_data import RolePrompt
from kvp.depth_eviction_probe import DepthProbeSpec
from kvp.loopzip_cached_eval import CachedFullPrefill
from kvp.role_sensitivity import role_forward


class FakeTokenizer:
    eos_token_id=None
    def decode(self, ids, skip_special_tokens=True):
        return "Answer: Paris" if ids else ""


class FakeBB:
    family="ouro"
    R=2
    n_rec=1
    n_plain=0
    H=D=1

    def __init__(self):
        self.model=torch.nn.Embedding(600,1)
        with torch.no_grad():
            self.model.weight.copy_(torch.arange(600,dtype=torch.float32).reshape(-1,1)/100)

    def make_aux(self,*args,**kwargs):
        return None


def fake_one(bb, ids, pos, kv, col, mask, forced, spec):
    assert forced==bb.R
    val=bb.model(ids).reshape(1,1,1,1)
    q=torch.ones_like(val)
    k=torch.zeros_like(val)
    total=0.0
    for r in range(bb.R):
        attention=kv.attend(("rec",0,r),q,k,val,col,col+1,mask)
        total=total+attention.reshape(()).float()
        kv.actual_calls+=1
    x=total/2
    logits=torch.stack((x,-x,2*x,torch.tensor(.1),torch.tensor(-.3)))
    return logits.view(1,1,5),bb.R


def prompt():
    return RolePrompt(
        text="fake",token_ids=(10,100,200,30,40,50,60),
        roles=("other","relevant_content","irrelevant_content","other",
               "instruction","question","other"),
        content_positions=(1,2),instruction_positions=(4,5),
        distractor_positions=(),relevant_positions=(1,),
        irrelevant_positions=(2,),question_positions=(5,),
        context_spans=(
            {"id":"a","role":"relevant_content","positions":(1,),"source":"a"},
            {"id":"b","role":"irrelevant_content","positions":(2,),"source":"b"}))


def test_cached_prompt_matches_full_restart_with_joint_eviction(monkeypatch):
    import kvp.loopzip_cached_eval as impl
    import kvp.role_sensitivity as role
    monkeypatch.setattr(impl,"_ouro_one",fake_one)
    monkeypatch.setattr(role,"_ouro_one",fake_one)
    bb=FakeBB()
    spec=DepthProbeSpec("ouro",2,1.0,min_depth=2)
    p=prompt()
    target=(1,3,2,4)
    cache=CachedFullPrefill(bb,spec,p,target,"same",
                            max_generation_tokens=5)
    diff=cache.assert_clean_reference_parity(tolerance=1e-7)
    assert diff["reference_max_abs_nll_diff"]<1e-7

    clean=cache.score_gold(())
    for evicted in ((1,),(2,),(1,2),(1,),()):
        got=cache.score_gold(evicted)
        fresh=role_forward(bb,"same",spec,p,target,decode_mode="fixed",
                           role_mode="postprefill" if evicted else "none",
                           source=evicted if evicted else None)
        assert got["token_losses"]==pytest.approx(fresh["token_losses"],abs=1e-7)
        assert got["first_target_nll"]==pytest.approx(clean["first_target_nll"])
    assert cache.score_gold((1,2))["nll_future"]!=pytest.approx(clean["nll_future"])

    tok=FakeTokenizer()
    g1=cache.greedy(tok,"Paris",(),max_generation_tokens=5)
    g2=cache.greedy(tok,"Paris",(1,2),max_generation_tokens=5)
    assert len(g1["token_ids"])==5
    assert g1["token_ids"][0]==g2["token_ids"][0]
    assert cache.score_gold(())["token_losses"]==pytest.approx(clean["token_losses"])


def test_cached_prompt_prohibits_question_eviction(monkeypatch):
    import kvp.loopzip_cached_eval as impl
    monkeypatch.setattr(impl,"_ouro_one",fake_one)
    cache=CachedFullPrefill(FakeBB(),DepthProbeSpec("ouro",2,1.0),
                            prompt(),(1,2,3),"case",max_generation_tokens=5)
    with pytest.raises(ValueError,match="Only Context"):
        cache.score_gold((5,))
    with pytest.raises(ValueError,match="duplicates"):
        cache.score_gold((1,1))



def test_phase1_to_phase2_end_to_end_uses_same_budget_and_replay(monkeypatch):
    import importlib.util
    from pathlib import Path
    import kvp.loopzip_cached_eval as impl
    import kvp.role_sensitivity as role
    monkeypatch.setattr(impl,"_ouro_one",fake_one)
    monkeypatch.setattr(role,"_ouro_one",fake_one)
    path=Path(__file__).resolve().parents[1]/"scripts/loopzip_failure_screen.py"
    script_spec=importlib.util.spec_from_file_location("swap_runner_fake",path)
    r=importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(r)
    monkeypatch.setattr(r,"git_sha",lambda:"fake-source-sha")
    monkeypatch.setattr(
        r,"spec_for",lambda key,cfg:DepthProbeSpec("ouro",2,1.0,min_depth=2))
    p=prompt()
    jb={
        "model":"ouro-2.6b","dataset":"controlled_retrieval","uid":"u",
        "key":"case-001","revision":"fixture","protocol":"fixture",
        "prompt_text":p.text,"prompt_ids":list(p.token_ids),
        "prompt_roles":list(p.roles),
        "content_positions":list(p.content_positions),
        "instruction_positions":list(p.instruction_positions),
        "distractor_positions":list(p.distractor_positions),
        "relevant_positions":list(p.relevant_positions),
        "irrelevant_positions":list(p.irrelevant_positions),
        "question_positions":list(p.question_positions),
        "context_spans":[{**e,"positions":list(e["positions"])}
                         for e in p.context_spans],
        "prompt_sha256":"fixed-prompt-sha",
        "target_ids":[1,3,2,4],"target_truncated":False,
        "gold_answer":"Paris","attributes":{},
        "budget_counts":{"0.1":1,"0.3":1,"0.5":1}}
    cfg={"retention":(.1,.3,.5),
         "policies":("mean_attn","last_attn","max_attn","recent_attn","random"),
         "random_seeds":(11,29,47),
         "generation_tokens":5,"free_generate":True,
         "swap_max_retained":6,"swap_max_evicted":6,
         "swap_greedy":True}
    bb=FakeBB()
    tk=FakeTokenizer()
    screen=r.evaluate_screen(bb,tk,jb,cfg)
    assert screen["parity"]["reference_max_abs_nll_diff"]<1e-7
    assert len(screen["screening"])==21
    assert all(len(s["retained"])==1 for s in screen["screening"])
    assert all(s["first_target_nll"]==pytest.approx(
               screen["full"]["first_target_nll"]) for s in screen["screening"])
    target={"key":jb["key"],"swap_targets":[{
        "budget":.3,"category":"unit_test","role":"failure",
        "anchor_policy":"mean_attn"}]}
    swap=r.evaluate_swaps(bb,tk,jb,cfg,target,screen)
    assert len(swap["swap_targets"])==1
    result=swap["swap_targets"][0]
    assert result["best_nll"]<=result["anchor_nll"]+1e-7
    assert result["local_oracle_gain"]>=-1e-7
    assert result["n_swaps_evaluated"]==1
    assert all(len(v["retained"])==1 for v in result["swaps"])
