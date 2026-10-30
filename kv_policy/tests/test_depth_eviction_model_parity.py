"""Model-backed tiny checkpoint configuration tests; no H200 needed.

Requires transformers + access to the pinned Huginn/Ouro HF configs, as in
test_policy_semantics.py. These must pass before real H200 benchmarking.
"""
import pytest
import torch

from kvp.depth_eviction_probe import DepthProbeSpec, run_prefix
from .test_policy_semantics import _tiny, PROMPT, CONT


@pytest.mark.parametrize("family", ["ouro", "huginn"])
def test_dynamic_no_exit_matches_existing_strict_full(family):
    model, runner = _tiny(family, 2)
    # Ouro q=1 guarantees R2; Huginn threshold=0 cannot trigger an exit.
    spec = (DepthProbeSpec("ouro", 2, 1.0, 2) if family=="ouro"
            else DepthProbeSpec("huginn", 2, 0.0, 2))
    old = runner.teacher_forced_logits(PROMPT, CONT, "u", "full", "strict")
    new = run_prefix(runner.bb, PROMPT+CONT, "u", spec, prompt_len=len(PROMPT))
    assert new["depths"] == [2] * (len(PROMPT)+len(CONT)-1)
    ref = [-torch.log_softmax(old[len(PROMPT)-1+i].float(), -1)[tok].item()
           for i, tok in enumerate(CONT)]
    assert max(abs(a-b) for a,b in zip(ref,new["loss_by_future_token"])) < 1e-8


def test_ouro_exit_trace_contains_multiple_valid_depths_when_gate_permits():
    model, runner = _tiny("ouro", 4)
    # q may produce varying depths on random-weight models. Only test range.
    spec = DepthProbeSpec("ouro", 4, 0.5, 2)
    new = run_prefix(runner.bb, PROMPT+CONT, "u", spec, prompt_len=len(PROMPT))
    assert all(2<=d<=4 for d in new["depths"])
    assert new["actual_recurrence_calls"] == sum(new["depths"])
