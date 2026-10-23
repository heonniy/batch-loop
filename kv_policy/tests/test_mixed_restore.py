"""Restoration-oracle cache (kvp.oracle.MixedKV) vs the independent dictionary-cache reference.

FULL = restore-all, LAST = restore-none, arbitrary restore sets, per-row bases and initial latents
in one batch; tiny random Huginn/Ouro models, CPU float64."""

import random

import pytest
import torch

from kvp.oracle import convergence_scores, run_rows
from kvp.reference import reference_logits
from tests.test_policy_semantics import CONT, PROMPT, _noise, _tiny


@pytest.fixture(scope="module", params=["huginn", "ouro"])
def tiny(request):
    return request.param, *_tiny(request.param, 3)


def _rows_logits(run, rows):
    out = []
    run_rows(run, PROMPT, CONT, rows, lambda g, lg: out.append(lg.clone()))
    return torch.stack(out, 1)  # [B, n, V] predicting CONT[g]


def _ref(model, family, run, policy, keep=frozenset(), uid="u"):
    noise = run.bb.example_noise(uid, 64) if family == "huginn" else None
    lg = reference_logits(model, family, run.bb.R, PROMPT, CONT, policy, "convert", noise, frozenset(keep))
    P = len(PROMPT)
    return lg[P - 1: P - 1 + len(CONT)]


def test_restore_all_none_and_sets_match_reference(tiny):
    family, model, run = tiny
    T = len(PROMPT) + len(CONT)
    rng = random.Random(0)
    sets = [frozenset(rng.sample(range(T - 1), k)) for k in (1, 3, 6)] + [frozenset({0}), frozenset({len(PROMPT)})]
    rows = [{"base": "last", "keep": "all", "uid": "u"}, {"base": "last", "keep": None, "uid": "u"},
            {"base": "average", "keep": None, "uid": "u"}]
    for s in sets:
        rows.append({"base": "last", "keep": [i in s for i in range(T)], "uid": "u"})
    rows.append({"base": "average", "keep": [i in sets[1] for i in range(T)], "uid": "u"})
    got = _rows_logits(run, rows)
    refs = [_ref(model, family, run, "full"), _ref(model, family, run, "last"), _ref(model, family, run, "average")]
    refs += [_ref(model, family, run, "last", s) for s in sets] + [_ref(model, family, run, "average", sets[1])]
    for b, ref in enumerate(refs):
        assert torch.allclose(got[b], ref, atol=1e-9, rtol=1e-9), (b, (got[b] - ref).abs().max())
    # restoring a set actually changes the LAST logits, and restoring everything gives FULL
    assert (got[3] - got[1]).abs().max() > 1e-8
    allkeep = _rows_logits(run, [{"base": "last", "keep": [True] * T, "uid": "u"}])[0]
    assert torch.allclose(allkeep, refs[0], atol=1e-9)


def test_rows_with_different_latents_are_independent(tiny):
    family, model, run = tiny
    if family != "huginn":
        pytest.skip("initial latent is Huginn-only")
    got = _rows_logits(run, [{"base": "last", "keep": None, "uid": "u"}, {"base": "last", "keep": None, "uid": "w"}])
    assert torch.allclose(got[0], _ref(model, family, run, "last", uid="u"), atol=1e-9)
    assert torch.allclose(got[1], _ref(model, family, run, "last", uid="w"), atol=1e-9)


def test_no_future_leak_from_restore_mask(tiny):
    """Restoring a column at or after the predicted position cannot change that prediction."""
    family, model, run = tiny
    T = len(PROMPT) + len(CONT)
    base = _rows_logits(run, [{"base": "last", "keep": None, "uid": "u"}])[0]
    c = len(PROMPT) + 2  # column of CONT[2]; distributions predicting CONT[0..2] are computed before it is history
    got = _rows_logits(run, [{"base": "last", "keep": [i == c for i in range(T)], "uid": "u"}])[0]
    assert torch.equal(got[:3], base[:3])
    assert (got[4:] - base[4:]).abs().max() > 1e-10


def test_convergence_scores_shape(tiny):
    family, model, run = tiny
    kv = run_rows(run, PROMPT, CONT, [{"base": "last", "keep": "all", "uid": "u"}], lambda g, lg: None, keep_kv=True)
    s = convergence_scores(kv, 0, len(PROMPT))
    assert s["dist_to_terminal"].shape == (len(PROMPT),) and (s["dist_to_terminal"] > 0).all()
