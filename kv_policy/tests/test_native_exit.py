"""CPU-only checks for official exit-rule selection and trace safety."""
import pytest

from kvp.native_exit import ExitRule, select_depth, recorded_trace, replay_depth


def test_ouro_q_exit():
    r = ExitRule("ouro", "q-exit", 0.8)
    assert select_depth(r, [0.2, 0.51, 0.86, 1.0], 4) == 3
    assert select_depth(ExitRule("ouro", "q-exit", 1.0), [0.2, 0.51, 1, 1], 4) == 4


def test_huginn_official_cosine_auto_threshold():
    r = ExitRule("huginn", "cosine", 1e-3, min_depth=2)
    assert select_depth(r, [0.5, 0.1, .03, .0008, .0001], 5) == 4


def test_huginn_argmax_stability():
    r = ExitRule("huginn", "argmax-stability", 5)
    assert select_depth(r, [0, 0, 1, 2, 3, 4, 5], 7) == 7


def test_trace_identity_and_exhaustion():
    rule = ExitRule("ouro", "q-exit", .8)
    tr = recorded_trace("rev", rule, "u", [1, 2, 3], [2, 3, 4], 4, "full-strict")
    assert replay_depth(tr, 1, "rev", "u") == 3
    with pytest.raises(IndexError):
        replay_depth(tr, 3, "rev", "u")
    with pytest.raises(ValueError):
        replay_depth(tr, 0, "other-rev", "u")


def test_invalid_rule_and_values():
    with pytest.raises(ValueError):
        ExitRule("huginn", "q-exit", .5)
    with pytest.raises(ValueError):
        select_depth(ExitRule("huginn", "cosine", .001), [float("nan")], 1)
