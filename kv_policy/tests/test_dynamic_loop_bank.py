"""CPU-only loop-bank read semantics tests for dynamic-depth source tokens."""
import pytest
import torch
from kvp.depth_eviction_probe import DepthOracleKV

class BB:
    R=4
    n_rec=1
    n_plain=0
    H=1
    D=1

def setup(policy, restore_token=None, restore_loop=None):
    kv=DepthOracleKV(BB(),5,"cpu",torch.float32,evaluation_start=2,
        history_policy=policy,restore_token=restore_token,restore_loop=restore_loop)
    q=torch.zeros(1,1,1,1)
    for c,d in ((0,3),(1,2)):
        for r in range(d):
            kv.attend(("rec",0,r),q,q,torch.full_like(q,10*(c+1)+r),
                      c,c+1,torch.ones(1,1,1,c+1,dtype=torch.bool))
        kv.finish(c,d)
    return kv

def future_value(policy,query_r,**kwargs):
    kv=setup(policy,**kwargs)
    q=torch.zeros(1,1,1,1)
    mask=torch.ones(1,1,1,3,dtype=torch.bool)
    # Query k is constant, so uniform attention across 2 historic + self.
    return float(kv.attend(("rec",0,query_r),q,q,q,2,3,mask).item())

def test_last_vs_firstlast_vs_restore_loop2():
    assert future_value("last",1)==pytest.approx((12+21)/3)
    assert future_value("firstlast",0)==pytest.approx((10+20)/3)
    assert future_value("firstlast",1)==pytest.approx((12+21)/3)
    assert future_value("restore",1,restore_token=0,restore_loop=2)==pytest.approx((11+21)/3)
    assert future_value("restore",0,restore_token=0,restore_loop=1)==pytest.approx((10+21)/3)

def test_restore_unexecuted_bank_rejected():
    kv=setup("restore",restore_token=1,restore_loop=3)
    q=torch.zeros(1,1,1,1)
    with pytest.raises(ValueError,match="unexecuted"):
        kv.attend(("rec",0,2),q,q,q,2,3,torch.ones(1,1,1,3,dtype=torch.bool))

def test_prefill_not_altered_by_history_policies():
    for policy in ("last","firstlast","restore"):
        kv=DepthOracleKV(BB(),5,"cpu",torch.float32,evaluation_start=2,
            history_policy=policy,restore_token=0,restore_loop=1)
        q=torch.zeros(1,1,1,1)
        kv.attend(("rec",0,0),q,q,torch.full_like(q,7),0,1,
                  torch.ones(1,1,1,1,dtype=torch.bool))
        assert kv.rv[0][0,0,0,0,0].item()==7
