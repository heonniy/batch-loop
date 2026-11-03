"""CPU-only FFN causal hook contracts; no checkpoint or GPU needed.

cd kv_policy && python -m pytest -q tests/test_ffn_gate_causal.py
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kvp.ffn_gate_causal import (
    RecurrentFFNHooks, gate_trace_comparison, patch_projection,
    nll_recovery, paired_logit_kl,
)


class DummyOuroMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DummyHuginnMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 16, bias=False)
        self.proj = nn.Linear(8, 4, bias=False)
    def forward(self, x):
        g, u = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(g) * u)


def fake_bb(family, R=3, L=2):
    mlps = [DummyOuroMLP() if family=="ouro" else DummyHuginnMLP()
            for _ in range(L)]
    layers = nn.ModuleList([SimpleLayer(m) for m in mlps])
    if family=="ouro":
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    else:
        model = SimpleNamespace(transformer=SimpleNamespace(core_block=layers))
    return SimpleNamespace(family=family,R=R,n_rec=L,model=model)


class SimpleLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp=mlp


def simulate(bb, controller, x):
    out = []
    with controller.install():
        controller.begin_column(100)
        for _ in range(bb.R):
            for layer in (bb.model.model.layers if bb.family=="ouro"
                          else bb.model.transformer.core_block):
                out.append(layer.mlp(x))
        controller.end_column(bb.R)
    return torch.stack(out).detach()


@pytest.mark.parametrize("family",["huginn","ouro"])
def test_capture_restore_identity_and_restricted_loop(family):
    torch.manual_seed(14)
    bb=fake_bb(family)
    x=torch.randn(1,1,4)
    y=simulate(bb,RecurrentFFNHooks(bb,mode="capture"),x)
    clean=RecurrentFFNHooks(bb,mode="capture")
    assert torch.equal(y, simulate(bb,clean,x))
    restored=RecurrentFFNHooks(bb,mode="restore_gate",reference=clean.trace)
    y_same=simulate(bb,restored,x)
    assert torch.equal(y_same,y)
    assert restored.patched==bb.R*bb.n_rec
    x1=x+0.3
    raw=simulate(bb,RecurrentFFNHooks(bb,mode="capture"),x1)
    altered=RecurrentFFNHooks(bb,mode="restore_gate",reference=clean.trace,
                              selected_loops=[2],selected_layers=[1])
    patched=simulate(bb,altered,x1)
    assert altered.patched==1
    assert not torch.equal(patched,raw)
    assert torch.equal(patched[0],raw[0])
    assert torch.equal(patched[1],raw[1])
    assert torch.equal(patched[2],raw[2])
    assert torch.equal(patched[4],raw[4])
    assert torch.equal(patched[5],raw[5])


def test_huginn_fc_restores_only_gate_half():
    out=torch.tensor([[[1.,2.,3.,4.]]])
    clean=torch.tensor([[[6.,7.,8.,9.]]])
    gate=patch_projection(out,clean,family="huginn",component="fc",mode="restore_gate")
    up=patch_projection(out,clean,family="huginn",component="fc",mode="restore_up")
    shift=patch_projection(out,clean,family="huginn",component="fc",mode="shift_gate")
    assert gate.tolist()==[[[6.,7.,3.,4.]]]
    assert up.tolist()==[[[1.,2.,8.,9.]]]
    assert torch.allclose((shift[...,:2]-out[...,:2]).norm(),
                          (clean[...,:2]-out[...,:2]).norm())
    assert torch.equal(shift[...,2:],out[...,2:])


def test_ouro_projection_action_guard():
    x=torch.ones(1,1,8)
    z=torch.full_like(x,2)
    assert torch.equal(patch_projection(x,z,family="ouro",component="gate",
                                        mode="restore_gate"),z)
    with pytest.raises(ValueError):
        patch_projection(x,z,family="ouro",component="up",mode="restore_gate")
    with pytest.raises(ValueError):
        patch_projection(x,z[:,:,:3],family="ouro",component="gate",mode="restore_gate")


@pytest.mark.parametrize("family",["huginn","ouro"])
def test_gate_trace_statistics_compare_full_vs_evicted(family):
    torch.manual_seed(18)
    bb=fake_bb(family)
    clean=RecurrentFFNHooks(bb,mode="capture")
    ev=RecurrentFFNHooks(bb,mode="capture")
    simulate(bb,clean,torch.ones(1,1,4))
    simulate(bb,ev,torch.ones(1,1,4)*1.05)
    stats=gate_trace_comparison(clean.trace,ev.trace,family)
    assert len(stats["by_loop"])==bb.R
    assert stats["total_positions"]==bb.R*bb.n_rec*2
    assert all(x["gate_mean_relative_change"]>0 for x in stats["by_loop"])
    assert all(x["ffn_mean_relative_change"]>0 for x in stats["by_loop"])


def test_selected_loop_call_count_and_missing_patch_fail_closed():
    bb=fake_bb("ouro",R=2,L=1)
    x=torch.randn(1,1,4)
    bad=RecurrentFFNHooks(bb,mode="restore_gate",reference={})
    with pytest.raises(KeyError):
        simulate(bb,bad,x)
    c=RecurrentFFNHooks(bb,mode="capture")
    simulate(bb,c,x)
    no=RecurrentFFNHooks(bb,mode="restore_gate",reference=c.trace,selected_loops=[1])
    simulate(bb,no,x)
    assert no.patched==1


def test_recovery_and_paired_kl():
    full=-F.log_softmax(torch.tensor([1.0,0.0]),-1)[0]
    a=F.log_softmax(torch.tensor([1.0,0.0]),-1)
    b=F.log_softmax(torch.tensor([0.0,1.0]),-1)
    z=paired_logit_kl([a,a],[a,b],exclude_last_unscored=False)
    assert z["kl_by_query"][0]==pytest.approx(0.0,abs=1e-6)
    assert z["kl_by_query"][1]>0
    rec=nll_recovery(1.0,2.0,1.4)
    assert rec["eviction_penalty"]==1
    assert rec["fraction_recovered"]==pytest.approx(0.6)
    assert nll_recovery(2,1,1.5)["fraction_recovered"] is None


def test_hook_context_removes_handles_on_failure():
    bb=fake_bb("huginn",R=2,L=1)
    c=RecurrentFFNHooks(bb,mode="capture")
    with pytest.raises(RuntimeError):
        with c.install():
            c.begin_column(3)
            raise RuntimeError("deliberate")
    assert c.handles==[]
