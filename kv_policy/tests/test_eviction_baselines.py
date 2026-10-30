"""CPU-only baseline scoring, controlled same-budget comparisons."""
from kvp.eviction_baselines import POLICIES, score, evaluate_singleton_oracle


def test_attention_uses_only_pre_eviction_observations():
    x={"uid":"u","source_position":8,"exit_depth":4,"attention_pre":.1,
       "attention_pre_per_opportunity":.02,"attention_future_ORACLE":999,
       "age_at_eval":20,"delta_nll":.2,"model":"ouro-1.4b","dataset":"gsm8k"}
    y={**x,"attention_future_ORACLE":-999}
    for p in POLICIES:
        assert score(x,p)==score(y,p)


def test_rankings_are_fixed_budget_and_depth_sensitive():
    rows=[{"uid":"u","model":"ouro-1.4b","dataset":"gsm8k",
           "source_position":i+4,"exit_depth":[4,1,3,2][i],
           "attention_pre":[.1,.9,.2,.6][i],
           "attention_pre_per_opportunity":[.02,.5,.04,.3][i],
           "age_at_eval":4-i,"delta_nll":[.5,.1,.2,.3][i]}
          for i in range(4)]
    out=evaluate_singleton_oracle(rows,.5)
    assert len(out)==len(POLICIES)
    assert {x["n_retained"] for x in out}=={2}
    by={r["policy"]:r for r in out}
    assert by["depth"]["evicted_source_positions"]=="7;5"
    assert by["attention_h2o_proxy"]["evicted_source_positions"]=="6;4"
    assert by["depth"]["sum_individual_delta_nll_ORACLE_DIAGNOSTIC"]==.4
