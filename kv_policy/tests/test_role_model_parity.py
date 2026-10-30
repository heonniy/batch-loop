"""Model-backed tiny Huginn/Ouro reference parity (HF checkpoint CONFIG access).

Run before H200 execution. No large checkpoint weights are required:
the existing test_policy_semantics fixture instantiates tiny random models.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from kvp.role_data import RolePrompt
from kvp.role_sensitivity import role_forward, role_intervention_oracle
from kvp.depth_eviction_probe import DepthProbeSpec

fixture_file=Path(__file__).with_name("test_policy_semantics.py")
spec=importlib.util.spec_from_file_location("role_tiny_fixture",fixture_file)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_tiny,PROMPT,CONT=mod._tiny,mod.PROMPT,mod.CONT


def prompt():
    roles=("other","content","content","other",
           "instruction","instruction","other")
    return RolePrompt("tiny",tuple(PROMPT),roles,(1,2),(4,5),())


@pytest.mark.parametrize("family",["huginn","ouro"])
def test_full_prefill_fixed_decode_matches_existing_full_strict_nll(family):
    model,run=_tiny(family,2)
    exit_spec=DepthProbeSpec(family,2,
                            1.0 if family=="ouro" else 0.0,min_depth=2)
    base=run.teacher_forced_logits(PROMPT,CONT,"u","full","strict")
    got=role_forward(run.bb,"u",exit_spec,prompt(),CONT,
                     decode_mode="fixed")
    reference=[-torch.log_softmax(base[len(PROMPT)-1+i].float(),-1)[tok].item()
               for i,tok in enumerate(CONT)]
    assert max(abs(a-b) for a,b in zip(reference,got["token_losses"]))<1e-8
    assert got["depths"]==[2]*(len(PROMPT)+len(CONT)-1)


def test_paired_content_intervention_is_causal_and_has_parity():
    model,run=_tiny("ouro",2)
    exit_spec=DepthProbeSpec("ouro",2,1.0,min_depth=2)
    r=role_intervention_oracle(run.bb,"u","test-revision",exit_spec,
                               prompt(),CONT,[1],
                               decode_mode="fixed",
                               run_one_loop=True,
                               live_exit=False,
                               role_controls=[4])
    assert r["baseline"]["replay_max_abs_token_nll_delta"]<1e-8
    assert len(r["candidate_rows"])==1
    row=r["candidate_rows"][0]
    assert len(row["prefill_instruction_error_by_loop"])==2
    assert len(row["attention_by_loop"])==2
    assert math.isfinite(row["postprefill_eviction_delta_future_nll"])
    assert r["role_control_rows"][0]["source_role"]=="instruction"
    assert r["trace"]["reference_policy"]=="full-depth-prefill-fixed-decode"


def test_dynamic_decode_only_changes_target_exit_paths():
    model,run=_tiny("ouro",4)
    exit_spec=DepthProbeSpec("ouro",4,.5,min_depth=2)
    p=prompt()
    r=role_forward(run.bb,"u",exit_spec,p,CONT,decode_mode="dynamic")
    assert r["depths"][:len(PROMPT)]==[4]*len(PROMPT)
    assert all(2<=d<=4 for d in r["depths"][len(PROMPT):])
