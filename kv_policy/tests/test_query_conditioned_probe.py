"""CPU fake-backbone tests for the v7 query-conditioned runner (no checkpoints, no GPU)."""
from __future__ import annotations

import pytest
import torch

from kvp.depth_eviction_probe import DepthProbeSpec
from kvp.query_conditioned_probe import QueryConditionedRunner, anchor_columns
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
        self.model.model = torch.nn.Module()
        self.model.model.early_exit_gate = torch.nn.Linear(DM, 1)
        with torch.no_grad():
            self.model.lm_head.weight.copy_(self.model.out.T)
            self.model.model.early_exit_gate.weight.copy_(torch.tensor([[2.0, -1.0, 0.5]]))
            self.model.model.early_exit_gate.bias.fill_(0.3)

    def make_aux(self, *a, **k):
        return None


def fake_one(bb, ids, pos, kv, col, mask, forced, spec):
    h = bb.model.emb[ids.reshape(-1)].reshape(1, 1, DM)
    kv.observe_initial(col, h)
    e = kv.attend(("plain", 0, 0), h.view(1, 1, 1, DM), h.view(1, 1, 1, DM), h.view(1, 1, 1, DM),
                  col, col + 1, mask).reshape(1, 1, DM)
    h = h + 0.5 * e
    remaining, cumul = 1.0, 0.0
    for r in range(bb.R):
        q = torch.tanh(h).view(1, 1, 1, DM)
        o = kv.attend(("rec", 0, r), q, h.view(1, 1, 1, DM), torch.sin(h).view(1, 1, 1, DM), col, col + 1, mask)
        h = h + o.reshape(1, 1, DM) + 0.1 * torch.cos(h * (r + 1))
        kv.observe_state(col, r + 1, h)
        lam = torch.sigmoid(bb.model.model.early_exit_gate(h)).float().item()
        if r < bb.R - 1:
            cumul += lam * remaining
            remaining *= (1.0 - lam)
        else:
            cumul = 1.0
        kv.actual_calls += 1
        depth = r + 1
        if forced is not None:
            if depth == forced:
                break
        elif depth >= spec.min_depth and (depth == bb.R or (spec.threshold < 1 and cumul >= spec.threshold)):
            break
    return (h.reshape(1, DM) @ bb.model.out).view(1, 1, V), depth


def _prompt():
    roles = ("other", "content", "content", "content", "other", "instruction", "question", "other")
    return RolePrompt("fake", (1, 2, 3, 4, 5, 6, 0, 2), roles, (1, 2, 3), (5, 6), (), question_positions=(6,))


@pytest.fixture
def runner(monkeypatch):
    import kvp.loopzip_persistent_eviction as impl
    import kvp.query_conditioned_probe as qc
    import kvp.role_sensitivity as role
    for mod in (impl, qc, role):
        monkeypatch.setattr(mod, "_ouro_one", fake_one)
    r = QueryConditionedRunner(FakeBB(), _prompt(), "u", horizon=6)
    r.reference(eos_ids=set(), frozen_ids=None)
    return r


def test_anchor_rule():
    a = anchor_columns(_prompt())
    assert a == {"question_last": 6, "prompt_last": 7}


def test_census_recorded_for_anchors_and_generated_only(runner):
    assert set(runner.local) == {6, 7} | set(range(runner.P, runner.P + len(runner.ref_ids)))
    c = runner.census(6)
    assert c["attn"].shape == (3, 6) and c["delta"].shape == (3, 6)
    # head-mean attention over all earlier keys + self sums to one per loop (self excluded here)
    assert torch.all(c["attn"].sum(1) <= 1.0 + 1e-6)
    assert torch.all(c["delta"] >= 0)
    # A key with zero attention has zero closed-form local perturbation.
    w = c["attn"][0]
    assert bool(((w > 0) | (c["delta"][0] == 0)).all())


def test_query_probe_rolls_back_prefix_and_is_exact(runner):
    live = DepthProbeSpec("ouro", 3, 0.5, min_depth=2)
    rep = runner.validate_v7(1, 3, live_spec=live, tolerance=1e-6)
    assert rep["pass"], rep
    assert rep["probe_nomask_max_E"] == 0 and rep["probe_prompt_kv_drift"] == 0


def test_one_loop_mask_only_affects_later_loops(runner):
    out = runner.query_probe(6, (2,), loops=(2,), plain=False)
    assert float(out["E"][0]) == 0 and float(out["E"][1]) == 0 and float(out["E"][2]) > 0
    allm = runner.query_probe(6, (2,))
    assert float(allm["E"][1]) > 0
    assert len(out["exit"]["lambda"]) == 3 and out["exit"]["cumulative"][-1] == 1.0


def test_probe_rejects_future_sources(runner):
    with pytest.raises(ValueError, match="precede"):
        runner.query_probe(6, (7,))


def test_generated_column_probe_matches_clean_probe(runner):
    cp = runner.clean_probes((2,), (3,))[3]
    qp = runner.query_probe(runner.P + 2, (2,))
    assert torch.allclose(cp["E"], qp["E"]) and cp["token"]["kl"] == pytest.approx(qp["token"]["kl"])
    again = runner.accumulated(())
    assert float(again["E"].abs().max()) == 0


def test_dynamic_decode_changes_depth_and_restores(runner):
    live = DepthProbeSpec("ouro", 3, 0.5, min_depth=2)
    full = runner.decode((), live, dynamic=True)
    assert len(full["depths"]) == len(runner.ref_ids) and min(full["depths"]) >= 2
    ev = runner.decode((2,), live, dynamic=True)
    rp = runner.decode((2,), live, forced_depths=full["depths"])
    assert rp["depths"] == full["depths"]
    assert ev["first_target_nll"] == rp["first_target_nll"] == full["first_target_nll"]
    with pytest.raises(ValueError):
        runner.decode((), live, forced_depths=[3], dynamic=True)
    clean = runner.accumulated(())
    assert float(clean["E"].abs().max()) == 0


def test_greedy_reproduces_reference_without_mask(runner):
    ids = runner.greedy((), 6, set())
    assert ids == list(runner.ref_ids)
    runner.greedy((1, 2), 6, set())
    assert float(runner.accumulated(())["E"].abs().max()) == 0
