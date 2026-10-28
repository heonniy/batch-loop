"""Selective bank protection (kvp.banks): none == LAST-CONVERT, all == FULL, vs dict reference per bank."""

import pytest
import torch

from kvp.banks import REGISTRY
from tests.test_policy_semantics import CONT, PROMPT, _tiny


@pytest.fixture(scope="module", params=["ouro", "huginn"])
def tiny(request):
    model, run = _tiny(request.param, 3)
    run.model_key = f"tiny-{request.param}"
    return request.param, model, run


def test_bank_extremes(tiny):
    family, model, run = tiny
    R, n = run.bb.R, run.bb.n_rec
    REGISTRY[(run.model_key, "banknone")] = frozenset()
    REGISTRY[(run.model_key, "bankall")] = frozenset((l, r) for l in range(n) for r in range(R - 1))
    last = run.teacher_forced_logits(PROMPT, CONT, "u", "last", "convert")
    full = run.teacher_forced_logits(PROMPT, CONT, "u", "full", "decode")
    assert torch.allclose(run.teacher_forced_logits(PROMPT, CONT, "u", "banknone", "convert"), last, atol=1e-12)
    assert torch.allclose(run.teacher_forced_logits(PROMPT, CONT, "u", "bankall", "convert"), full, atol=1e-12)


def test_single_bank_changes_only_that_read(tiny):
    family, model, run = tiny
    REGISTRY[(run.model_key, "bankone")] = frozenset({(0, 0)})
    last = run.teacher_forced_logits(PROMPT, CONT, "u", "last", "convert")
    one = run.teacher_forced_logits(PROMPT, CONT, "u", "bankone", "convert")
    P = len(PROMPT)
    assert torch.allclose(one[:P], last[:P])          # prefill identical (FULL)
    assert (one[P:] - last[P:]).abs().max() > 1e-8    # decoding reads layer-0 step-0 history differently
    ids, pos, valid, kv, aux, P_, T = run._setup([PROMPT], 2, "bankone", "convert", ["u"])
    with torch.inference_mode():
        run._prefill(ids, pos, valid, kv, aux, P_, "convert")
    R = run.bb.R
    assert not torch.equal(kv.rk[0][0, :, :, :P_], kv.rk[0][R - 1, :, :, :P_])   # protected plane kept
    assert torch.equal(kv.rk[0][1, :, :, :P_], kv.rk[0][R - 1, :, :, :P_])       # unprotected converted


@pytest.mark.parametrize("cells", [((0, 0),), ((1, 0),), ((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 0), (0, 1))])
def test_bank_policy_matches_dict_reference(tiny, cells):
    """BankKV (broadcast store + per-bank convert/finalize) vs the independent dictionary reference in
    which a protected (layer, step) bank serves per-step history and every other bank the terminal step."""
    from kvp.reference import reference_logits

    family, model, run = tiny
    name = "bankref" + "".join(f"{l}{r}" for l, r in cells)
    REGISTRY[(run.model_key, name)] = frozenset(cells)
    noise = run.bb.example_noise("u", 64) if family == "huginn" else None
    ref = reference_logits(model, family, run.bb.R, PROMPT, CONT, "last", "convert", noise, banks=frozenset(cells))
    eng = run.teacher_forced_logits(PROMPT, CONT, "u", name, "convert")
    assert torch.allclose(eng, ref, atol=1e-9, rtol=1e-9), (eng - ref).abs().max()


@pytest.mark.parametrize("cells", [(), ((0, 0),), ((1, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1))])
def test_bank_strict_matches_dict_reference(tiny, cells):
    """LAST/strict + protected banks vs the dictionary reference; no banks == LAST/strict."""
    from kvp.reference import reference_logits

    family, model, run = tiny
    name = "bankst" + "".join(f"{l}{r}" for l, r in cells)
    REGISTRY[(run.model_key, name)] = frozenset(cells)
    noise = run.bb.example_noise("u", 64) if family == "huginn" else None
    ref = reference_logits(model, family, run.bb.R, PROMPT, CONT, "last", "strict", noise, banks=frozenset(cells))
    eng = run.teacher_forced_logits(PROMPT, CONT, "u", name, "strict")
    assert torch.allclose(eng, ref, atol=1e-9, rtol=1e-9), (eng - ref).abs().max()
    if not cells:
        assert torch.allclose(eng, run.teacher_forced_logits(PROMPT, CONT, "u", "last", "strict"), atol=1e-12)
