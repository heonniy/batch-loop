"""CPU-only measured-results audit tests; no invented empirical claims."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


def observations():
    path=Path(__file__).resolve().parents[1]/"scripts/loopzip_observations.py"
    spec=importlib.util.spec_from_file_location("loopzip_obs_test",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def fake_run(tmp_path):
    run=tmp_path/"run"; (run/"analysis").mkdir(parents=True)
    (run/"run_manifest.json").write_text(json.dumps({
        "protocol":"loopzip-failure-first-fixed-v1","code_sha":"test"}))
    (run/"completeness.json").write_text(json.dumps({"complete":True}))
    cases=[]
    quality=[]
    for b in (.1,.3,.5):
        cases.append({
            "key":"a","model":"ouro-2.6b","dataset":"controlled_retrieval",
            "uid":"one","budget":str(b),"category":"all_attention_wrong",
            "full_correct":True,"best_attn_delta_nll":.2,
            "mean_delta_nll":.5})
        for policy in ("mean_attn","last_attn","max_attn","recent_attn"):
            quality.append({
                "model":"ouro-2.6b","dataset":"controlled_retrieval",
                "uid":"one","budget":str(b),"policy":policy,"delta_nll":.2})
    _csv(run/"analysis"/"failure_cases.csv",cases)
    _csv(run/"analysis"/"screening_quality.csv",quality)
    return run


def test_phase1_only_reports_unrecovered_failure(tmp_path):
    m=observations()
    run=fake_run(tmp_path)
    report=m.render(run)
    assert report["counts"]["all_attention_failure_cases"]==3
    assert not report["phase2_available"]
    txt=(run/"analysis"/"OBSERVATIONS.md").read_text()
    assert "not tested" in txt
    assert "does **NOT** prove" in txt
    with pytest.raises(RuntimeError,match="Missing Phase-2"):
        m.render(run,require_swaps=True)


def test_phase2_positive_nll_and_accuracy_rescue_are_separate(tmp_path):
    m=observations()
    run=fake_run(tmp_path)
    (run/"swap_raw").mkdir()
    (run/"swap_completeness.json").write_text(json.dumps({"complete":True}))
    (run/"swap_raw"/"a.json").write_text(json.dumps({
        "meta":{"model":"ouro-2.6b","dataset":"controlled_retrieval","uid":"one"},
        "swap_targets":[{
            "budget":.3,"local_oracle_gain":.15,
            "best_greedy":{"exact_match":True}}]}))
    r=m.render(run,require_swaps=True)
    mid=next(x for x in r["groups"] if x["budget"]==.3)
    assert mid["n_positive_same_budget_swap_gain"]==1
    assert mid["n_confirmed_accuracy_rescues"]==1
    assert next(x for x in r["groups"] if x["budget"]==.1)["n_positive_same_budget_swap_gain"]==0
    assert (run/"analysis"/"OBSERVATIONS.json").exists()


def test_incomplete_run_is_not_renderable(tmp_path):
    m=observations()
    run=fake_run(tmp_path)
    (run/"completeness.json").write_text(json.dumps({"complete":False}))
    with pytest.raises(RuntimeError,match="incomplete"):
        m.render(run)
