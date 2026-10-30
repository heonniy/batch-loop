"""Fast CPU-only checks for label-blind MAIN Context candidate universe."""
from __future__ import annotations

import pytest

from kvp.role_main_candidates import (
    choose_main_candidates, main_candidate_universe)
from kvp.role_data import build_evidence_prompt, choose_evidence_candidates
from kvp.role_evidence_data import controlled_retrieval


class CharTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(x) for x in text]

    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":self.encode(text),
                "offset_mapping":[(i,i+1) for i in range(len(text))]}

    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return ("<system>"+messages[0]["content"]+"</system>"+
                "<user>"+messages[1]["content"]+"</user><assistant>")


def test_main_uses_only_aligned_positions_never_answer_or_labels():
    docs,_=controlled_retrieval(1,20261031,16)
    original=docs[0]
    tok=CharTokenizer()
    a=build_evidence_prompt(tok,"ouro",original)
    # Relabel every Context sentence and falsify the gold answer; actual
    # prompt tokens/positions remain the same. Main selection MUST remain
    # bit-for-bit identical despite changed annotation and answer.
    relabeled={**original,
               "answer":"SOME_UNRELATED_ANSWER",
               "spans":[{**s,"role":("irrelevant_content" if s["role"]=="relevant_content"
                                    else "relevant_content")}
                        for s in original["spans"]]}
    b=build_evidence_prompt(tok,"ouro",relabeled)
    assert a.content_positions==b.content_positions
    first=choose_main_candidates(a.content_positions,seed=11,
                                 n_candidates=8,window_tokens=1)
    second=choose_main_candidates(b.content_positions,seed=11,
                                  n_candidates=8,window_tokens=1)
    assert first==second
    assert {c["role"] for c in first}=={"context"}
    assert {c["selection_policy"] for c in first}=={"positional-stratified-hash-v3"}
    assert all(c["answer_anchor_used"] is False for c in first)
    assert len({p for c in first for p in c["positions"]})==8
    assert all(p in a.content_positions for c in first for p in c["positions"])


def test_main_multi_token_windows_are_contiguous_equal_budget_and_nonoverlap():
    positions=tuple(range(10,32))+tuple(range(35,61))
    pool=main_candidate_universe(positions,window_tokens=4)
    assert pool and all(w[-1]-w[0]==3 for w in pool)
    assert all(set(w)<=set(positions) for w in pool)
    chosen=choose_main_candidates(positions,seed=1,n_candidates=6,
                                  window_tokens=4,position_bins=3)
    flat=[p for c in chosen for p in c["positions"]]
    assert len(flat)==24 and len(set(flat))==24
    assert all(len(c["positions"])==4 for c in chosen)
    assert all(max(c["positions"])-min(c["positions"])==3 for c in chosen)
    assert chosen==choose_main_candidates(positions,seed=1,n_candidates=6,
                                          window_tokens=4,position_bins=3)


def test_main_fails_closed_on_impossible_budget_and_wrong_pool():
    with pytest.raises(ValueError,match="nonoverlapping"):
        choose_main_candidates(range(10,20),seed=0,n_candidates=8,
                               window_tokens=3)
    with pytest.raises(ValueError,match="tokenizer-aligned"):
        main_candidate_universe([1,3,5],window_tokens=2)
    with pytest.raises(ValueError,match="Empty"):
        main_candidate_universe([],window_tokens=1)


def test_ablation_is_explicitly_ground_truth_conditioned():
    doc=controlled_retrieval(1,7,8)[0][0]
    prompt=build_evidence_prompt(CharTokenizer(),"huginn",doc)
    sampled=choose_evidence_candidates(prompt,seed=7,n_relevant=1,
                                       n_irrelevant=1,window_tokens=8)
    assert {s["role"] for s in sampled}=={
        "relevant_content","irrelevant_content"}
    assert len({len(s["positions"]) for s in sampled})==1
    assert all(all(prompt.roles[p]==s["role"] for p in s["positions"])
               for s in sampled)


def test_main_oracle_does_not_require_relevant_distractor_pair():
    from kvp.role_span_oracle import role_evidence_oracle
    import importlib.util
    from pathlib import Path
    from kvp.role_data import RolePrompt
    from kvp.depth_eviction_probe import DepthProbeSpec
    p=Path(__file__).with_name("test_policy_semantics.py")
    sp=importlib.util.spec_from_file_location("main_tiny_fixture",p)
    module=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    _,run=module._tiny("ouro",2)
    prompt=RolePrompt("tiny",tuple(module.PROMPT),
                      ("other","relevant_content","irrelevant_content",
                       "other","instruction","question","other"),
                      (1,2),(4,5),(),(1,),(2,),(5,),
                      ({"id":"e","role":"relevant_content",
                        "positions":(1,),"source":"A"},))
    candidate=[{"id":"main-1","role":"context","positions":[1],
                "answer_anchor_used":False}]
    result=role_evidence_oracle(
        run.bb,"u","test",
        DepthProbeSpec("ouro",2,1.0,min_depth=2),
        prompt,module.CONT,candidate,decode_mode="fixed",
        study="main",run_one_loop=True,live_exit=False)
    assert result["baseline"]["study"]=="main"
    assert result["candidate_rows"][0]["source_role"]=="context"
    assert len(result["candidate_rows"][0]["attention_by_loop"])==2
