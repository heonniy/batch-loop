"""Functional single-token recurrence (kvp.stability) vs the engine; tiny Huginn, CPU float64."""

import pytest
import torch

from kvp.oracle import MixedKV
from kvp.stability import loop_fn, make_ctx, readout, trajectory
from tests.test_policy_semantics import CONT, PROMPT, _tiny


@pytest.fixture(scope="module")
def tiny():
    model, run = _tiny("huginn", 4)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, run


def _full_pass_ctx(run, upto_g):
    """FULL B=1 pass; returns (ctx for the token fed at step upto_g, engine logits of that step)."""
    bb = run.bb
    P = len(PROMPT)
    T = P + len(CONT)
    kv = MixedKV(["last"], torch.ones(1, T, dtype=torch.bool), bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D,
                 run.dtype, run.device)
    ids = torch.tensor([PROMPT])
    pos = torch.arange(T)[None]
    valid = torch.ones(1, T, dtype=torch.bool)
    aux = bb.make_aux(["u"], [0], T)
    with torch.inference_mode():
        run._prefill_parallel(ids, pos, valid, kv, aux, P)
        for g in range(1, upto_g):
            run._step(torch.tensor([CONT[g - 1]]), P + g - 1, pos, valid, kv, aux)
    c = P + upto_g - 1
    tok = CONT[upto_g - 1]
    ctx = make_ctx(bb, kv, tok, c, aux[0, c])
    with torch.inference_mode():
        eng = run._step(torch.tensor([tok]), c, pos, valid, kv, aux)[0, 0]
    return ctx, eng, kv


def test_functional_forward_matches_engine(tiny):
    model, run = tiny
    for g in (1, 3):
        ctx, eng, _ = _full_pass_ctx(run, g)
        zs = trajectory(ctx, ctx.x0, 0)
        assert torch.allclose(readout(ctx, zs[-1])[0], eng, atol=1e-9)


def test_rows_are_independent_and_history_frozen(tiny):
    model, run = tiny
    ctx, _, kv = _full_pass_ctx(run, 3)
    snap = [k.clone() for k in kv.rk]
    z = trajectory(ctx, ctx.x0, 0, 2)[-1]
    zb = torch.cat([z, z + 1e-3 * torch.randn_like(z), z], 0)
    out = trajectory(ctx, zb, 2)
    base = trajectory(ctx, z, 2)
    assert torch.equal(out[-1][0], base[-1][0]) and torch.equal(out[-1][2], base[-1][0])
    assert all(torch.equal(a, b) for a, b in zip(snap, kv.rk))


def test_jvp_matches_finite_difference(tiny):
    model, run = tiny
    ctx, _, _ = _full_pass_ctx(run, 3)
    z = trajectory(ctx, ctx.x0, 0, 1)[-1]
    v = torch.randn_like(z)
    v = v / v.norm()
    _, jv = torch.func.jvp(lambda x: loop_fn(ctx, x, 2), (z,), (v,))
    # Huginn's RoPE helper computes in float32 even for a float64 model, so a central difference
    # with tiny eta is dominated by float32 rounding; eta = 1e-3 |z| keeps both errors ~1e-5.
    eta = 1e-3 * z.norm()
    fd = (loop_fn(ctx, z + eta * v, 2) - loop_fn(ctx, z - eta * v, 2)) / (2 * eta)
    assert (jv - fd).norm() / jv.norm() < 1e-3
    # VJP consistency: <u, J v> == <J^T u, v>
    u = torch.randn_like(z)
    _, vjp = torch.func.vjp(lambda x: loop_fn(ctx, x, 2), z)
    (jtu,) = vjp(u)
    assert abs((u * jv).sum() - (jtu * v).sum()) < 1e-6 * abs((u * jv).sum())  # float32 RoPE inside


def test_terminal_swap_is_noop_at_last_loop(tiny):
    model, run = tiny
    ctx, _, _ = _full_pass_ctx(run, 3)
    R = run.bb.R
    z = trajectory(ctx, ctx.x0, 0, R - 1)[-1]
    assert torch.equal(loop_fn(ctx, z, R), loop_fn(ctx, z, R, terminal_layers=(0, 1)))
    z1 = ctx.x0
    assert not torch.equal(loop_fn(ctx, z1, 1), loop_fn(ctx, z1, 1, terminal_layers=(0,)))


def test_sublayer_maps_compose_to_loop(tiny):
    from kvp.stability import sublayer_maps

    model, run = tiny
    ctx, _, _ = _full_pass_ctx(run, 3)
    z = trajectory(ctx, ctx.x0, 0, 1)[-1]
    x = z
    for _, f in sublayer_maps(ctx, 2):
        x = f(x)
    assert torch.equal(x, loop_fn(ctx, z, 2))
