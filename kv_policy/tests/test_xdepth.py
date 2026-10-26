"""kvp.xdepth single-token cross-depth functions vs the engine (tiny Ouro / Huginn, CPU float64)."""

import pytest
import torch

from kvp.oracle import MixedKV
from kvp.xdepth import make_fn, run_all
from tests.test_policy_semantics import CONT, PROMPT, _tiny


@pytest.fixture(scope="module", params=["ouro", "huginn"])
def tiny(request):
    model, run = _tiny(request.param, 3)
    for p in model.parameters():
        p.requires_grad_(False)
    return request.param, model, run


def _site(run, g=3):
    bb = run.bb
    P, T = len(PROMPT), len(PROMPT) + len(CONT)
    kv = MixedKV(["last"], torch.ones(1, T, dtype=torch.bool), bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D, run.dtype, run.device)
    pos, valid = torch.arange(T)[None], torch.ones(1, T, dtype=torch.bool)
    aux = bb.make_aux(["u"], [0], T)
    with torch.inference_mode():
        run._prefill_parallel(torch.tensor([PROMPT]), pos, valid, kv, aux, P)
        for k in range(1, g):
            run._step(torch.tensor([CONT[k - 1]]), P + k - 1, pos, valid, kv, aux)
    c = P + g - 1
    x0 = aux[0, c] if bb.family == "huginn" else None
    fn = make_fn(bb, kv, CONT[g - 1], c, x0)
    with torch.inference_mode():
        eng = run._step(torch.tensor([CONT[g - 1]]), c, pos, valid, kv, aux)[0, 0]
    return fn, eng


def test_functional_matches_engine(tiny):
    family, model, run = tiny
    fn, eng = _site(run)
    hs = run_all(fn, fn.init())
    assert torch.allclose(fn.readout(hs[-1])[0], eng, atol=1e-9)


def test_inject_exact_delta_equals_terminal_read(tiny):
    family, model, run = tiny
    fn, _ = _site(run)
    cap = {}
    hs = run_all(fn, fn.init(), cap=cap)
    l, s = 1, 1
    x_in, a, out = cap[(l, s)]
    a2, out2 = fn.block_io(l, x_in, s, s)
    assert torch.allclose(out2, out, atol=1e-12)
    _, out_t = fn.block_io(l, x_in, s, fn.R)
    via_inject = run_all(fn, fn.init(), inject={(l, s): out_t - out})[-1]
    via_read = run_all(fn, fn.init(), read_terminal=frozenset({(l, s)}))[-1]
    assert torch.allclose(via_inject, via_read, atol=1e-10)
    assert (via_read - hs[-1]).abs().max() > 1e-8
    # terminal step reading terminal history is a no-op; rows are independent
    same = run_all(fn, fn.init(), read_terminal=frozenset({(l, fn.R)}))[-1]
    assert torch.equal(same, hs[-1])
    z = fn.init().expand(3, -1, -1).contiguous()
    d = torch.zeros(3, 1, z.shape[-1], dtype=z.dtype)
    d[1] = 1e-3
    rows = run_all(fn, z, inject={(l, s): d})[-1]
    assert torch.allclose(rows[0], hs[-1][0]) and torch.allclose(rows[2], hs[-1][0]) and not torch.allclose(rows[1], rows[0])
