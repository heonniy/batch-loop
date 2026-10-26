"""Attention-mechanism instrumentation (kvp.mechanism) on tiny random Huginn models, CPU float64."""

import pytest
import torch

from kvp.engine import Runner
from kvp.mechanism import B_METRICS, C_METRICS, TraceHuginn, Tracer, intervention_pass, kv_geometry
from kvp.oracle import run_rows
from tests.test_policy_semantics import CONT, PROMPT, _tiny

ROWS = [{"base": "last", "keep": "all", "uid": "u"}, {"base": "last", "keep": None, "uid": "u"}]


@pytest.fixture(scope="module")
def tiny():
    model, run = _tiny("huginn", 4)
    trun = Runner(TraceHuginn(run.bb), pad_id=1)
    return model, run, trun


def _logits(runner, rows, tracer=None):
    out = []
    if tracer is not None:
        runner.bb.tracer = tracer

    def on_step(g, lg):
        out.append(lg.clone())
        if tracer is not None and g > 0:
            tracer.end_step()

    kv = run_rows(runner, PROMPT, CONT, rows, on_step, keep_kv=True)
    runner.bb.tracer = None
    return torch.stack(out, 1), kv


def test_tracing_does_not_change_outputs(tiny):
    model, run, trun = tiny
    ref, _ = _logits(run, ROWS)
    plain, _ = _logits(trun, ROWS)
    traced, _ = _logits(trun, ROWS, Tracer(trun.bb))
    assert torch.equal(ref, plain) and torch.equal(ref, traced)


def test_fixed_q_controls(tiny):
    model, run, trun = tiny
    tr = Tracer(trun.bb)
    _logits(trun, ROWS, tr)
    b = torch.stack(tr.b_steps)  # [steps, R, n_core, metrics]
    assert len(tr.b_steps) == len(CONT) - 1 and not torch.isnan(b).any()
    i = {m: k for k, m in enumerate(B_METRICS)}
    assert b[..., i["D_precision"]].max() < 1e-5  # engine (fp64) vs direct fp32 attention
    assert b[..., i["D_layout"]].max() < 1e-5
    last = b[:, -1]  # loop R: past K^(R) is the terminal itself -> swap is exactly a no-op
    for m in ("D_swap", "D_konly", "D_vonly", "tv_mean"):
        assert last[..., i[m]].abs().max() == 0
    assert b[:, 0, :, i["D_swap"]].max() > 1e-6  # early loop: terminal history differs
    # key-only + value-only swaps both feed the full swap
    assert b[..., i["D_konly"]].max() > 0 and b[..., i["D_vonly"]].max() > 0


def test_errors_zero_for_identical_rows_and_prelude(tiny):
    model, run, trun = tiny
    tr = Tracer(trun.bb)
    _logits(trun, [ROWS[0], dict(ROWS[0])], tr)
    c = torch.stack(tr.c_steps)
    i = {m: k for k, m in enumerate(C_METRICS)}
    for m in ("err_in", "rel_da", "err_mid", "rel_dm", "err_out"):
        assert c[..., i[m]].abs().max() == 0
    tr = Tracer(trun.bb)
    _logits(trun, ROWS, tr)
    assert all(p["prelude0"] == 0 for p in tr.plain_steps)  # prelude reads token-only history
    c = torch.stack(tr.c_steps)
    assert c[1:, ..., i["err_in"]].max() > 1e-8  # FULL vs LAST differ once history exists
    assert c[0, 0, 0, i["err_in"]] == 0  # loop 1, layer 0 input of the first decode step: same latent+prelude


def test_interventions(tiny):
    model, run, trun = tiny
    R, nj = trun.bb.R, trun.bb.n_core
    grid = [(j, s) for j in range(nj) for s in range(R)]
    recs = intervention_pass(trun, PROMPT, CONT, "u", {2, 4}, grid)
    assert len(recs) == 2 * len(grid)
    for r in recs:
        if r["loop"] == R:  # terminal loop: intervention reads the very same history
            assert r["imm_attn"] == 0 and r["kl_ref_int"] == 0 and r["final_lnf"] == 0
    assert max(r["imm_attn"] for r in recs if r["loop"] == 1) > 1e-6
    # the pass leaves no trace: its real FULL steps equal an untouched FULL run
    full, _ = _logits(run, ROWS[:1])
    lp = torch.log_softmax(full[0].double(), -1)
    for r in recs:
        assert abs(r["target_logprob_ref"] - lp[r["g"], CONT[r["g"]]].item()) < 1e-10


def test_kv_geometry(tiny):
    model, run, trun = tiny
    _, kv = _logits(run, ROWS[:1])
    geo = kv_geometry(kv, 0, len(PROMPT) + len(CONT) - 1)
    assert geo["dK"].shape == (trun.bb.n_core, trun.bb.R, len(PROMPT) + len(CONT) - 1, trun.bb.H)
    assert geo["dK"][:, -1].abs().max() == 0 and geo["dV"][:, -1].abs().max() == 0
    assert torch.allclose(geo["cosK"][:, -1], torch.ones_like(geo["cosK"][:, -1]))
    assert geo["dK"][:, 0].min() > 0
