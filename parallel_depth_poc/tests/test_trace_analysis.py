"""Trace analyzer unit tests. No GPU required."""

import pytest

torch = pytest.importorskip("torch")

from parallel_depth_poc.trace_analysis import evaluate_states


def test_linear_trajectory_predictor_and_causal_previous():
    # Same trajectory slope, different token initial states.
    # State shape [T=3,D+1=5,B=2,H=3].
    init = torch.arange(18.0).reshape(3, 2, 3) / 10.0
    slope = torch.tensor([0.2, 0.1, -0.05]).view(1, 1, 3)
    states = torch.stack([init + r * slope for r in range(5)], dim=1)
    metrics = evaluate_states(states)
    assert metrics["shape"] == {
        "token_steps": 3, "depth": 4, "batch": 2, "hidden": 3,
    }
    lin = metrics["predictors"]["linear"]["per_depth"]
    prev = metrics["predictors"]["previous_token"]["per_depth"]
    assert len(lin) == len(prev) == 3  # depths 2,3,4
    assert max(x["relative_l2"]["max"] for x in lin) < 1e-5
    assert max(x["relative_l2"]["max"] for x in prev) < 1e-5
    assert metrics["predictors"]["previous_token"]["evaluated_token_steps"] == 2


def test_single_step_has_no_previous_token_predictor():
    states = torch.randn(4, 2, 5)
    result = evaluate_states(states)
    assert "previous_token" not in result["predictors"]


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        evaluate_states(torch.zeros(2, 2))
    with pytest.raises(TypeError):
        evaluate_states(torch.zeros(3, 2, 5, dtype=torch.int64))
