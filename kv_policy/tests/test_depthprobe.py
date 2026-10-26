"""Depth probes (kvp.depthprobe): identical outputs, native last-depth readout, ACT pdf, geometry."""

import pytest
import torch

from kvp.depthprobe import ProbeHuginn, ProbeOuro, ouro_exit_pdf, pairwise_kv_geometry
from kvp.engine import Runner
from kvp.oracle import MixedKV, run_rows
from tests.test_policy_semantics import CONT, PROMPT, _tiny


@pytest.fixture(scope="module", params=["ouro", "huginn"])
def tiny(request):
    model, run = _tiny(request.param, 3)
    probe = (ProbeOuro if request.param == "ouro" else ProbeHuginn)(run.bb)
    return request.param, model, run, Runner(probe, pad_id=run.pad_id)


def _logits(runner):
    out = []
    run_rows(runner, PROMPT, CONT, [{"base": "last", "keep": "all", "uid": "u"}], lambda g, lg: out.append(lg.clone()))
    return torch.stack(out)


def test_probe_backbone_is_identical(tiny):
    family, model, run, prun = tiny
    prun.bb.record = True
    assert torch.equal(_logits(run), _logits(prun))
    prun.bb.record = False


def test_last_depth_probe_equals_model_output(tiny):
    family, model, run, prun = tiny
    bb = prun.bb
    P, T = len(PROMPT), len(PROMPT) + len(CONT)
    kv = MixedKV(["last"], torch.ones(1, T, dtype=torch.bool), bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D, run.dtype, run.device)
    pos, valid = torch.arange(T)[None], torch.ones(1, T, dtype=torch.bool)
    aux = bb.make_aux(["u"], [0], T)
    bb.record = True
    with torch.inference_mode():
        lg = prun._prefill_parallel(torch.tensor([PROMPT]), pos, valid, kv, aux, P)
        if family == "ouro":
            dl, gates = bb.depth_logits([s[0] for s in bb.states])
            assert torch.allclose(dl[-1], lg[0], atol=1e-10)
            pdf = ouro_exit_pdf(gates)
            assert torch.allclose(pdf.sum(0), torch.ones_like(pdf[0]))
        out = prun._step(torch.tensor([CONT[0]]), P, pos, valid, kv, aux)[0, 0]
        if family == "huginn":
            dl = bb.depth_logits_decode(bb.states, kv, P)
            assert torch.allclose(dl[-1], out, atol=1e-9)
            assert (dl[0] - dl[-1]).abs().max() > 1e-6
    bb.record = False
    geo = pairwise_kv_geometry(kv, 0, P + 1, list(range(bb.R)), {"prompt": slice(0, P)})
    d = geo["K_dist|prompt"]
    assert torch.allclose(torch.diagonal(d, dim1=1, dim2=2), torch.zeros(d.shape[0], bb.R, dtype=d.dtype))
