"""CPU-only LoopZip budget, label-blind selection and Phase-2 swap tests."""
from __future__ import annotations

from collections import Counter

import pytest

from kvp.loopzip_policy import (
    BUDGETS, ATTENTION_POLICIES, RANDOM_SEEDS,
    context_budget, choose_retained, selection_manifest,
    swap_neighbors, classify_failure)


POSITIONS = list(range(10, 110))
ATTENTION = {
    str(p): [.001 + (p % 17)/1000,
             .002 + ((p * 3) % 19)/1000,
             .003 + ((p * 5) % 23)/1000,
             .004 + ((p * 7) % 29)/1000]
    for p in POSITIONS}


@pytest.mark.parametrize("budget,expected",[(.10,10),(.30,30),(.50,50),(1,100)])
def test_same_exact_context_budget_for_every_policy(budget,expected):
    assert context_budget(POSITIONS,budget)==expected
    for policy in ("mean_attn","last_attn","max_attn","recent_attn","random"):
        result=choose_retained(POSITIONS,ATTENTION,budget,policy,uid="case1")
        assert result["n_retained"]==expected
        assert len(result["retained"])==expected
        assert len(result["evicted"])==100-expected
        assert not (set(result["evicted"]) & set(result["retained"]))
        assert sorted(result["evicted"]+result["retained"])==POSITIONS
        assert result["actual_retention"]==pytest.approx(budget)


def test_recent_context_counts_toward_exact_budget():
    x=choose_retained(POSITIONS,ATTENTION,.10,"recent_attn",uid="case1")
    assert 109 in x["retained"] and 108 in x["retained"]
    assert len(x["retained"])==10
    with pytest.raises(ValueError,match="Retention"):
        context_budget(POSITIONS,0)
    with pytest.raises(ValueError,match="Context"):
        context_budget([1,1,2],.5)


def test_all_clean_attention_policy_profiles_differ_and_are_deterministic():
    selected=selection_manifest(POSITIONS,ATTENTION,"case1")
    assert len(selected)==3*(4+len(RANDOM_SEEDS))
    assert {(s["retention_requested"],s["policy_key"]) for s in selected}.__len__()==len(selected)
    selected2=selection_manifest(POSITIONS,ATTENTION,"case1")
    assert selected==selected2
    # Corrupt/missing clean attention must fail instead of falling back to
    # answer-time attention, gold labels, or a silent random ranking.
    with pytest.raises(ValueError,match="Missing clean"):
        choose_retained(POSITIONS,{},.3,"mean_attn",uid="case1")
    with pytest.raises(ValueError,match="Invalid attention"):
        bad={**ATTENTION,"10":[.1,float("nan"),.2,.3]}
        choose_retained(POSITIONS,bad,.3,"mean_attn",uid="case1")


def test_no_label_or_gold_inputs_for_policy_and_swap():
    # No dataset labels are ever passed into selection; same positions +
    # same clean profiles produce exactly same selections across all datasets.
    a=choose_retained(POSITIONS,ATTENTION,.30,"mean_attn",uid="same")
    b=choose_retained(POSITIONS,ATTENTION,.30,"mean_attn",uid="same")
    assert a==b
    assert a["scores"] is not None
    assert "answer" not in str(a).lower()
    swaps=swap_neighbors(a,POSITIONS,ATTENTION,uid="same")
    assert swaps[0]["swap_id"]=="original"
    assert len(swaps)<=37
    assert len({tuple(x["retained"]) for x in swaps})==len(swaps)
    for x in swaps:
        assert len(x["retained"])==a["n_retained"]
        assert set(x["retained"])|set(x["evicted"])==set(POSITIONS)
        assert not (set(x["retained"]) & set(x["evicted"]))
    assert any(x["type"]=="boundary" for x in swaps)
    assert any(x["type"]=="tail_or_diverse" for x in swaps)
    assert swaps==swap_neighbors(a,POSITIONS,ATTENTION,uid="same")


def test_short_budget_swap_is_capped_and_never_invents_extra_retained():
    positions=list(range(10))
    attn={str(i):[.01*i,.02*i] for i in positions}
    base=choose_retained(positions,attn,.1,"mean_attn",uid="u")
    neighbors=swap_neighbors(base,positions,attn,uid="u")
    assert len(neighbors)<=7
    assert all(len(x["retained"])==1 for x in neighbors)


def test_failure_classification_requires_clean_qa_accuracy():
    vals={p:{"greedy_correct":False} for p in ATTENTION_POLICIES}
    assert classify_failure(True,vals)=="all_attention_wrong"
    assert classify_failure(False,vals)=="full_wrong"
    assert classify_failure(None,vals)=="nll_only_unclassified"
    vals["last_attn"]["greedy_correct"]=True
    assert classify_failure(True,vals)=="mean_wrong_other_attention_correct"
    vals["mean_attn"]["greedy_correct"]=True
    assert classify_failure(True,vals)=="mixed_attention_failure"
    for p in ATTENTION_POLICIES:
        vals[p]["greedy_correct"]=True
    assert classify_failure(True,vals)=="no_attention_failure"
