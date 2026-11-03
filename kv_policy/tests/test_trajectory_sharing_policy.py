"""V10 CPU contracts: distinct recurrence policies, causal pair statistics and exits.

From kv_policy/: python -m pytest -q tests/test_trajectory_sharing_policy.py
No checkpoint or CUDA needed; GPU smoke must additionally validate true KV reads.
"""
import numpy as np
import pytest

from kvp.trajectory_sharing_policy import (
    sharing_name, make_read_map, policy_planes, exit_at,
    choose_clean_thresholds, compare_exit_stats, paired_trajectory,
    same_depth_nll_interaction,
)


def test_model_specific_maps_are_exact():
    assert make_read_map("huginn",32,9,"full") is None
    assert make_read_map("ouro",4,9,"full") is None
    h=make_read_map("huginn",32,9,"last")
    o=make_read_map("ouro",4,9,"first_last")
    assert h.shape==(32,9)
    assert np.all(h==31)
    assert o.shape==(4,9)
    assert np.all(o[0]==0)
    assert np.all(o[1:]==3)
    assert policy_planes("huginn",32,"last")==1
    assert policy_planes("ouro",4,"first_last")==2
    assert policy_planes("ouro",4,"full")==4


def test_ouro_first_last_differs_from_full_and_nearest():
    o=make_read_map("ouro",4,5,"first_last")
    near=make_read_map("ouro",4,5,"first_last",ouro_nearest=True)
    assert np.all(o[1]==3)
    assert np.all(near[1]==0)
    assert np.all(near[2]==3)
    with pytest.raises(ValueError):
        make_read_map("ouro",4,5,"last")
    with pytest.raises(ValueError):
        make_read_map("huginn",32,5,"first_last")
    with pytest.raises(ValueError):
        make_read_map("huginn",1,5,"last")


def test_exit_depths_use_native_rules_not_hidden_norm():
    assert exit_at({"cumulative":[0.1,0.5,0.8,1.0]},
                   "ouro",0.7)==3
    assert exit_at({"cumulative":[0.1,0.75,0.9,1.0]},
                   "ouro",0.7)==2
    assert exit_at({"distance_fp32":[0.1,0.003,0.0005,0.0001]},
                   "huginn",0.001)==3
    assert exit_at({"distance_fp32":[0.1,0.003,0.002,0.0005]},
                   "huginn",0.001)==4
    with pytest.raises(ValueError):
        exit_at({"cumulative":[0.1,0.5,0.8,1.0]},"ouro",1.2)


def test_clean_calibration_covers_huginn_small_distance_region():
    # Simulate v9 situation where even 1e-4 early exits almost everyone.
    clean=[{"distance_fp32":[0.1,1e-5*(1+i/10),1e-6*(1+i/10),1e-7]}
           for i in range(20)]
    tuned=choose_clean_thresholds(clean,"huginn")
    assert tuned["family"]=="huginn"
    assert tuned["calibration_from"]=="full_and_shared_clean_fixed_depth"
    assert len(tuned["selected"])==2
    assert any(x["threshold"]<1e-4 for x in tuned["selected"])
    assert len({x["threshold"] for x in tuned["selected"]})==2
    assert all(0<=x["early_fraction"]<=1 for x in tuned["selected"])


def test_ouro_calibration_is_clean_only():
    clean=[{"cumulative":[0.1,0.45+0.02*i,0.85,1.0]}
           for i in range(10)]
    tuned=choose_clean_thresholds(clean,"ouro")
    assert len(tuned["selected"])==2
    assert all(0<x["threshold"]<1 for x in tuned["selected"])


def test_would_exit_is_separate_from_actual_live_depth():
    a=[{"distance_fp32":[0.2,0.003,0.0003,0.0001]}]
    b=[{"distance_fp32":[0.2,0.003,0.002,0.0001]}]
    r=compare_exit_stats(a,b,"huginn",0.001)
    assert r["depth_delta"]==[1]
    assert r["fraction_changed"]==1
    assert r["mean_abs_exit_stat_shift"]>0
    with pytest.raises(ValueError):
        compare_exit_stats(a,[],"huginn",0.001)


def test_trajectory_error_same_query_same_loop_and_update():
    a=np.array([[1.,0.],[2.,0.],[3.,0.]],dtype=np.float32)
    b=np.array([[1.,0.],[2.,1.],[3.,2.]],dtype=np.float32)
    out=paired_trajectory([a],[b])
    assert out["n_queries"]==1
    assert out["R"]==2
    assert out["mean_relative_error_by_loop"][0]==pytest.approx(0)
    assert out["mean_relative_error_by_loop"][1]==pytest.approx(0.5)
    assert out["per_query_final_relative_error"][0]==pytest.approx(2/3)
    assert out["mean_clean_update_by_loop"][0]==pytest.approx(1)
    with pytest.raises(ValueError):
        paired_trajectory([a],[b[:2]])


def test_sharing_eviction_difference_in_differences():
    x=same_depth_nll_interaction(1.0,1.3,1.1,1.2)
    assert x["full_eviction_penalty"]==pytest.approx(0.3)
    assert x["shared_eviction_penalty"]==pytest.approx(0.1)
    assert x["sharing_minus_full_eviction_penalty"]==pytest.approx(-0.2)
    assert x["sharing_baseline_nll_shift"]==pytest.approx(0.1)
