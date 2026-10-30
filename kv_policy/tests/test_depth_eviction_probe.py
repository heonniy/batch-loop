"""CPU-only tests for native-depth KV cache and causal single-source masking."""
import pytest
import torch

from kvp.depth_eviction_probe import DepthOracleKV, DepthProbeSpec, choose_sources


class TinyBackbone:
    R = 4
    n_rec = 2
    n_plain = 1
    H = 1
    D = 1
    family = "ouro"


def store(evicted=None):
    return DepthOracleKV(TinyBackbone(), 3, "cpu", torch.float32,
                         evicted=evicted, evaluation_start=2)


def test_missing_higher_loop_bank_falls_back_to_last_executed_bank():
    kv = store()
    q = torch.ones(1,1,1,1)
    one = torch.ones(1,1,1,1,dtype=torch.bool)
    kv.attend(("rec",0,0),q,q,torch.full_like(q,17),0,1,one)
    kv.attend(("rec",1,0),q,q,torch.full_like(q,23),0,1,one)
    assert kv.rv[0][3,0,0,0,0].item() == 0
    kv.finish(0,1)
    assert kv.depths == [1]
    assert [kv.rv[0][r,0,0,0,0].item() for r in range(4)] == [17]*4
    assert [kv.rv[1][r,0,0,0,0].item() for r in range(4)] == [23]*4
    # Query executes loop4 although source token only executed loop1.
    qmask = torch.ones(1,1,1,2,dtype=torch.bool)
    out = kv.attend(("rec",0,3),q,q,torch.full_like(q,100),1,2,qmask)
    assert torch.isfinite(out).all()


def test_evict_all_layer_history_but_never_current_token():
    no, ev = store(), store(evicted=0)
    q = torch.ones(1,1,1,1)
    for kv in (no,ev):
        kv.attend(("plain",0,0),q,q,torch.full_like(q,100),0,1,
                  torch.ones(1,1,1,1,dtype=torch.bool))
        kv.finish(0,1)
    m = torch.ones(1,1,1,2,dtype=torch.bool)
    a=no.attend(("plain",0,0),q,q,q,1,2,m)
    b=ev.attend(("plain",0,0),q,q,q,1,2,m)
    assert a.item() > b.item()+10
    assert b.item()==pytest.approx(1.0)


def test_source_selection_and_gate_sanity():
    assert choose_sources(20,4,3) == choose_sources(20,4,3)
    assert len(choose_sources(20,4,3))==4
    assert all(i>=4 for i in choose_sources(20,4,3))
    with pytest.raises(ValueError): choose_sources(4,2,0)
    assert DepthProbeSpec("ouro",4,.7,2).rule().criterion=="q-exit"
    assert DepthProbeSpec("huginn",32,1e-3,2).rule().criterion=="cosine"
    with pytest.raises(ValueError): DepthProbeSpec("ouro",4,1.7)
    with pytest.raises(ValueError): DepthProbeSpec("huginn",1,.001)


def test_attention_accounting_excludes_self_and_future_queries():
    bb=TinyBackbone()
    kv=DepthOracleKV(bb,3,"cpu",torch.float32,
                     evicted=None,evaluation_start=1,collect_attention=True)
    q=torch.ones(1,1,1,1)
    kv.attend(("plain",0,0),q,q,q,0,1,torch.ones(1,1,1,1,dtype=torch.bool))
    assert kv.attention_pre.sum().item()==0
    kv.finish(0,1)
    kv.attend(("plain",0,0),q,q,q,1,2,torch.ones(1,1,1,2,dtype=torch.bool))
    assert kv.attention_pre.sum().item()==0
    assert kv.attention_future[0].item()>0
