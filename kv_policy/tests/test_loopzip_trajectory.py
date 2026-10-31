"""CPU fake-backbone tests for LoopZip v6 persistent eviction and trajectory recording."""
from __future__ import annotations

import math

import pytest
import torch

from kvp.loopzip_persistent_eviction import PersistentEvictionRunner, eligible_sources
from kvp.loopzip_trajectory_recorder import logit_metrics, paired_metrics, trajectory_metrics
from kvp.role_data import RolePrompt

V, DM = 7, 3


class FakeBB:
    family = "ouro"
    R = 3
    n_rec = 1
    n_plain = 1
    H = 1
    D = DM

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.model = torch.nn.Module()
        self.model.emb = torch.nn.Parameter(torch.randn(50, DM, generator=g))
        self.model.out = torch.nn.Parameter(torch.randn(DM, V, generator=g))
        self.model.lm_head = torch.nn.Linear(DM, V, bias=False)
        with torch.no_grad():
            self.model.lm_head.weight.copy_(self.model.out.T)

    def make_aux(self, *a, **k):
        return None


def fake_one(bb, ids, pos, kv, col, mask, forced, spec):
    h = bb.model.emb[ids.reshape(-1)].reshape(1, 1, DM)
    kv.observe_initial(col, h)
    e = kv.attend(("plain", 0, 0), h.view(1, 1, 1, DM), h.view(1, 1, 1, DM), h.view(1, 1, 1, DM),
                  col, col + 1, mask).reshape(1, 1, DM)
    h = h + 0.5 * e
    for r in range(bb.R):
        q = torch.tanh(h).view(1, 1, 1, DM)
        o = kv.attend(("rec", 0, r), q, h.view(1, 1, 1, DM), torch.sin(h).view(1, 1, 1, DM), col, col + 1, mask)
        h = h + o.reshape(1, 1, DM) + 0.1 * torch.cos(h * (r + 1))
        kv.observe_state(col, r + 1, h)
        kv.actual_calls += 1
    return (h.reshape(1, DM) @ bb.model.out).view(1, 1, V), bb.R


def _prompt():
    roles = ("other", "content", "content", "content", "other", "instruction", "question", "other")
    return RolePrompt("fake", (1, 2, 3, 4, 5, 6, 7, 8), roles, (1, 2, 3), (5, 6), (), question_positions=(6,))


@pytest.fixture
def runner(monkeypatch):
    import kvp.loopzip_persistent_eviction as impl
    import kvp.role_sensitivity as role
    monkeypatch.setattr(impl, "_ouro_one", fake_one)
    monkeypatch.setattr(role, "_ouro_one", fake_one)
    r = PersistentEvictionRunner(FakeBB(), _prompt(), "u", horizon=6)
    r.reference(eos_ids=set(), frozen_ids=None)
    return r


def test_metrics_shapes_and_definitions():
    H = torch.tensor([[1.0, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1]])
    m = trajectory_metrics(H)
    assert math.isnan(m["U"][0]) and m["U"][1] == pytest.approx(1.0)
    assert m["F"][-1] == 0 and math.isnan(m["C"][0]) and math.isnan(m["C"][-1])
    assert m["C"][1] == pytest.approx(0.0)
    p = paired_metrics(H, H)
    assert float(p["E"].abs().max()) == 0 and p["dircos"][1] == pytest.approx(1.0)
    lp = torch.log_softmax(torch.tensor([1.0, 2.0, 0.5]), -1)
    lm = logit_metrics(lp, torch.tensor([1.0, 2.0, 0.5]), 1)
    assert lm["kl"] == pytest.approx(0, abs=1e-7) and not lm["top1_changed"] and lm["nll_shift"] == pytest.approx(0)


def test_eligibility_rule():
    assert eligible_sources(_prompt()) == [1, 2, 3, 5, 6]


def test_reference_and_validation_gate(runner):
    assert len(runner.ref_ids) == 6 and runner.ref_H.shape == (6, 4, DM)
    rep = runner.validate(1, 3, t_probe=3, tolerance=1e-6)
    assert rep["pass"], rep
    assert rep["nomask_max_E"] == 0 and rep["independent_nll_max_abs"] < 1e-6


def test_accumulated_differs_from_clean_after_t1(runner):
    acc = runner.accumulated((2,))
    cl = runner.clean_probes((2,), (1, 2, 4, 6, 99))
    assert set(cl) == {1, 2, 4, 6}
    assert torch.allclose(acc["E"][0], cl[1]["E"])
    assert not torch.allclose(acc["E"][3], cl[4]["E"])  # altered generated KV accumulates
    assert acc["token"][-1]["nll_shift"] is None and acc["token"][0]["nll_shift"] is not None
    # clean branch untouched afterwards
    again = runner.accumulated(())
    assert float(again["E"].abs().max()) == 0


def test_mask_never_covers_current_or_prompt(runner):
    with pytest.raises(RuntimeError, match="own KV"):
        runner.accumulated((runner.P,))
    before = runner._kv_digest(0, runner.P)
    runner.accumulated((1, 2, 3))
    assert runner._kv_digest(0, runner.P) == before


def test_frozen_reference_reproduces_greedy(monkeypatch, runner):
    ids = list(runner.ref_ids)
    E0 = runner.ref_H.clone()
    rep = runner.reference(eos_ids=set(), frozen_ids=ids)
    assert rep["frozen_argmax_agreement"] == 1.0 and torch.equal(runner.ref_H, E0)


def test_source_features(runner):
    f = runner.source_features()
    assert f["recv_mean"].shape == (3, runner.P)
    tot = f["last_query"][:, :runner.P - 1].sum(-1)  # excludes the final token's self-attention
    assert bool(((tot > 0) & (tot < 1)).all()) and float(f["last_query"][:, runner.P - 1:].abs().max()) == 0


def test_control_probe_semantics(runner):
    R = runner.R
    cl = runner.clean_probes((2,), (3,))[3]
    allm = runner.control_probe(3, (2,))
    assert torch.equal(allm["E"], cl["E"]) and allm["token"]["kl"] == cl["token"]["kl"]
    # one-loop masks: applied once per recurrent layer of that loop (+ plain if requested)
    one = runner.control_probe(3, (2,), loops=(2,), plain=False)
    assert float(one["E"][:2].abs().max()) == 0 and float(one["E"][2]) > 0  # untouched before loop 2
    rec_only = runner.control_probe(3, (2,), loops=None, plain=False)
    assert not torch.equal(rec_only["E"], allm["E"])
    # zero injection == FULL; nonzero injection only changes loops >= r
    z = runner.control_probe(3, inject=(2, torch.zeros(DM)))
    assert float(z["E"].abs().max()) == 0 and z["token"]["kl"] == 0
    v = runner.control_probe(3, inject=(2, torch.ones(DM) * 0.3))
    assert float(v["E"][:2].abs().max()) == 0 and float(v["E"][2]) > 0
    assert float(v["delta"][2].norm()) == pytest.approx(0.3 * DM ** 0.5, rel=1e-5)
    # history restored: accumulated FULL still identical
    assert float(runner.accumulated(())["E"].abs().max()) == 0
    with pytest.raises(ValueError):
        runner.control_probe(99, (2,))


def test_readout_only_native_for_ouro(runner):
    runner.readout = True
    runner.reference(eos_ids=set(), frozen_ids=list(runner.ref_ids))
    assert len(runner.ref_readout) == len(runner.ref_ids)
    ro = runner.ref_readout[0]
    assert ro["kl_to_final"].shape == (runner.R,) and float(ro["kl_to_final"][-1]) == pytest.approx(0, abs=1e-6)
    assert float(ro["top1_agree"][-1]) == 1.0
