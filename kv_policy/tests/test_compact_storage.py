"""Compact (policy-sized) LAST/AVERAGE storage must reproduce the broadcast-layout semantics."""

import pytest
import torch

from tests.test_policy_semantics import CONT, PROMPT, _tiny


@pytest.mark.parametrize("family", ["ouro", "huginn"])
@pytest.mark.parametrize("policy", ["last", "average"])
@pytest.mark.parametrize("mode", ["decode", "convert", "strict"])
def test_compact_equals_broadcast(family, policy, mode):
    model, run = _tiny(family, 3)
    ref = run.teacher_forced_logits(PROMPT, CONT, "u", policy, mode)
    run.store = "compact"
    got = run.teacher_forced_logits(PROMPT, CONT, "u", policy, mode)
    assert torch.allclose(got, ref, atol=1e-9, rtol=1e-9), (got - ref).abs().max()
    # batched + left padding through the compact path
    res = run.score([PROMPT, PROMPT[:4]], [CONT, CONT], ["u", "v"], policy, mode)
    lp = torch.log_softmax(ref, -1)
    want = [lp[len(PROMPT) - 1 + i, t].item() for i, t in enumerate(CONT)]
    assert max(abs(a - b) for a, b in zip(want, res.target_logprobs[0])) < 1e-9


@pytest.mark.parametrize("family", ["ouro", "huginn"])
def test_compact_bytes(family):
    model, run = _tiny(family, 3)
    ids, pos, valid, full_kv, aux, P, T = run._setup([PROMPT], 10, "full", "decode", ["u"])
    run.store = "compact"
    _, _, _, dec_kv, _, _, _ = run._setup([PROMPT], 10, "last", "decode", ["u"])
    _, _, _, str_kv, _, _, _ = run._setup([PROMPT], 10, "last", "strict", ["u"])
    R, bb = run.bb.R, run.bb
    per_tok_plane = 2 * bb.H * bb.D * 8  # K+V, float64
    rec = bb.n_rec * per_tok_plane
    plain = bb.n_plain * per_tok_plane * T
    assert full_kv.bytes_allocated() == rec * R * T + plain
    assert dec_kv.bytes_allocated() == rec * (R * P + (T - P)) + plain
    assert str_kv.bytes_allocated() == rec * T + plain
