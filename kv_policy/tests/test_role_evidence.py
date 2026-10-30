"""CPU-only evidence annotations and causal multi-KV masking tests."""
from __future__ import annotations

import math

import pytest
import torch

from kvp.role_data import RolePrompt, build_evidence_prompt, choose_evidence_candidates
from kvp.role_evidence_data import (
    controlled_retrieval, _hotpot_sentences, _excerpt)
from kvp.role_sensitivity import RoleProbeKV
from kvp.role_span_oracle import role_evidence_oracle
from kvp.depth_eviction_probe import DepthProbeSpec


class CharTokenizer:
    def encode(self,s,add_special_tokens=False):
        return [ord(c) for c in s]

    def __call__(self,s,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":self.encode(s),
                "offset_mapping":[(i,i+1) for i in range(len(s))]}

    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return ("<system>"+messages[0]["content"]+"</system>"+
                "<user>"+messages[1]["content"]+"</user><assistant>")


def test_controlled_retrieval_is_deterministic_and_position_balanced():
    a,ma=controlled_retrieval(9,20261030,16)
    b,mb=controlled_retrieval(9,20261030,16)
    assert a==b and ma==mb
    assert [d["attributes"]["target_position"] for d in a[:3]] == [0,8,15]
    for doc in a:
        relevant=[s for s in doc["spans"] if s["role"]=="relevant_content"]
        irrelevant=[s for s in doc["spans"] if s["role"]=="irrelevant_content"]
        assert len(relevant)==1 and len(irrelevant)==15
        assert doc["answer"] in relevant[0]["text"]
        assert all(doc["answer"] not in s["text"] for s in irrelevant)
        assert doc["question"].endswith("?")


def test_hotpot_supporting_fact_provenance_and_excerpt():
    row={"context":{
          "title":["Bridge A","Bridge B","Noise C","Noise D"],
          "sentences":[["A supports the link.","Another fact."],
                       ["B concludes the answer."],
                       ["C unrelated.","C more irrelevant."],
                       ["D unrelated."]]},
         "supporting_facts":{"title":["Bridge A","Bridge B"],"sent_id":[0,0]}}
    spans=_hotpot_sentences(row)
    assert sum(s["role"]=="relevant_content" for s in spans)==2
    excerpt=_excerpt(spans,"example",20261030,2)
    assert sum(s["role"]=="relevant_content" for s in excerpt)==2
    assert sum(s["role"]=="irrelevant_content" for s in excerpt)==2
    assert all("source" not in s or "text" in s for s in excerpt)
    bad={**row,"supporting_facts":{"title":["Bridge A","Bridge Z"],"sent_id":[0,1]}}
    with pytest.raises(ValueError,match="missing"):
        _hotpot_sentences(bad)


@pytest.mark.parametrize("family",["ouro","huginn"])
def test_token_roles_and_evidence_candidate_window_budget(family):
    docs,_=controlled_retrieval(1,50,8)
    p=build_evidence_prompt(CharTokenizer(),family,docs[0])
    assert min(p.question_positions)>max(p.content_positions)
    assert max(p.question_positions)==max(p.instruction_positions)
    assert len(p.relevant_positions)>0 and len(p.irrelevant_positions)>0
    assert {p.roles[x] for x in p.content_positions}=={
        "relevant_content","irrelevant_content"}
    c=choose_evidence_candidates(p,7,n_relevant=2,n_irrelevant=3,
                                  window_tokens=11)
    assert len(c)==4  # one evidence sentence; 3 irrelevant sentences
    assert len({len(v["positions"]) for v in c})==1
    assert all(all(p.roles[x]==v["role"] for x in v["positions"]) for v in c)
    assert p.context_spans[0]["id"].startswith("fact-")


def _tiny_prompt():
    return RolePrompt(
        "tiny",tuple(range(7)),
        ("other","relevant_content","irrelevant_content","other",
         "instruction","question","other"),
        (1,2),(4,5),(),(1,),(2,),(5,),
        ({"id":"e1","role":"relevant_content","positions":(1,),"source":"A/0"},
         {"id":"d1","role":"irrelevant_content","positions":(2,),"source":"C/0"}))


def test_multi_key_mask_is_only_applied_in_selected_phase():
    class Tiny:
        R=2
        n_rec=1
        n_plain=0
        H=D=1
    prompt=_tiny_prompt()
    kv=RoleProbeKV(Tiny(),9,"cpu",torch.float32,prompt=prompt,
                   mode="postprefill",source=(1,2))
    assert not kv._should_mask(("rec",0,0),6)
    assert kv._should_mask(("rec",0,0),7)
    kv2=RoleProbeKV(Tiny(),9,"cpu",torch.float32,prompt=prompt,
                    mode="instruction_one_loop",source=(1,2))
    assert not kv2._should_mask(("rec",0,0),4)
    assert kv2._should_mask(("rec",0,0),5)
    assert not kv2._should_mask(("rec",0,1),5)


def test_tiny_model_paired_span_oracle_has_clean_replay_and_generated_state():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).with_name("test_policy_semantics.py")
    spec=importlib.util.spec_from_file_location("evidence_tiny_fixture",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model,run=mod._tiny("ouro",2)
    p=_tiny_prompt()
    candidates=[
        {"id":"e1","role":"relevant_content","positions":[1]},
        {"id":"d1","role":"irrelevant_content","positions":[2]}]
    result=role_evidence_oracle(
        run.bb,"u","fixture-revision",
        DepthProbeSpec("ouro",2,1.0,min_depth=2),
        p,mod.CONT,candidates,decode_mode="fixed",live_exit=False)
    assert result["baseline"]["replay_max_abs_token_nll_delta"]<1e-8
    assert len(result["candidate_rows"])==2
    assert {r["source_role"] for r in result["candidate_rows"]}=={
        "relevant_content","irrelevant_content"}
    for row in result["candidate_rows"]:
        assert row["n_masked_tokens"]==1
        assert len(row["attention_by_loop"])==2
        assert row["future_decode_recurrent_error_mean"]>=0
        assert math.isfinite(row["postprefill_eviction_delta_future_nll"])
    assert result["role_control_rows"][0]["source_role"]=="question"



def test_greedy_generation_first_token_is_untouched_by_future_eviction():
    from kvp.role_generation import greedy_answer, _normalize_answer, _extract_answer
    import importlib.util
    from pathlib import Path
    fixture=Path(__file__).with_name("test_policy_semantics.py")
    module_spec=importlib.util.spec_from_file_location("generation_tiny_fixture",fixture)
    mod=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(mod)
    _,run=mod._tiny("ouro",2)
    prompt=_tiny_prompt()
    spec=DepthProbeSpec("ouro",2,1.0,min_depth=2)
    clean=greedy_answer(run.bb,"u",spec,prompt,4,decode_mode="fixed")
    alt=greedy_answer(run.bb,"u",spec,prompt,4,decode_mode="fixed",source=(1,2))
    assert clean["token_ids"][0]==alt["token_ids"][0]
    assert clean["decoded_tokens"]==4
    assert len(clean["decode_depths"])==3
    assert _extract_answer("Answer: Paris.\nMore")=="Paris."
    assert _normalize_answer("The United States.")==_normalize_answer("united states")
