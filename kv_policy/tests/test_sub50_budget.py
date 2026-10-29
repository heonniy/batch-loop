"""CPU-only tests for Ouro sub-50% first-recurrence bank-budget ladders."""
import copy
import json

import pytest

from kvp.sub50_budget import BUDGETS, bank_count, construct, assert_disjoint_from_eval


def fake_calibration(model="ouro-1.4b", layers=24):
    return {"model": model, "R": 4, "L": layers,
            "items": [{"uid": "cal-0"}, {"uid": "cal-1"}],
            "single_bank": [
                {"layer": l, "step": 0, "gain": float(layers-l)} for l in range(layers)
            ] + [{"layer": l, "step": 1, "gain": 1} for l in range(layers)]}


@pytest.mark.parametrize("model,layers", [("ouro-1.4b", 24), ("ouro-2.6b", 48)])
def test_bank_budgets_never_exceed_ceiling(model, layers):
    reg, meta = construct(fake_calibration(model, layers), model)
    for pct in BUDGETS:
        k = bank_count(layers, pct)
        assert 0 < k < layers
        for typ in ("Top", "Prefix", "Rand"):
            name = f"bank{typ}B{pct}"
            cells = reg[name]
            assert len(cells) == k
            assert all(step == 0 for _, step in cells)
            assert len({layer for layer, _ in cells}) == k
            expected = (layers + k) / (4 * layers)
            assert meta["policies"][name]["full_fraction"] == pytest.approx(expected)
            assert expected <= pct / 100 + 1e-12
    # 25% LAST < all selected budgets < 50% Loop1-all.
    assert bank_count(layers, 25) == 0
    assert bank_count(layers, 50) == layers


def test_top_ranking_uses_scores_and_random_is_deterministic():
    a = fake_calibration()
    a["single_bank"][0]["gain"] = -999
    a["single_bank"][13]["gain"] = 999
    ra, _ = construct(a, "ouro-1.4b")
    rb, _ = construct(copy.deepcopy(a), "ouro-1.4b")
    assert ra == rb
    assert [13, 0] not in ra["bankTopB30"]
    assert [13, 0] in ra["bankTopB30"]
    assert [0, 0] in ra["bankPrefixB30"]


def test_reject_missing_layer_and_bad_depth():
    c = fake_calibration()
    c["single_bank"][0]["step"] = 2
    with pytest.raises(ValueError):
        construct(c, "ouro-1.4b")
    with pytest.raises(ValueError):
        bank_count(24, 20)


def test_calibration_eval_overlap_guard(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    p = d / "ouro-1.4b__gsm8k.jsonl"
    p.write_text("{}\n" + json.dumps({"uid": "evaluation-0"}) + "\n")
    provenance = {"by_model": {"ouro-1.4b": {"calibration_ids": ["cal-0"]}}}
    assert_disjoint_from_eval(provenance, tmp_path, ["gsm8k"])
    p.write_text("{}\n" + json.dumps({"uid": "cal-0"}) + "\n")
    with pytest.raises(ValueError, match="contamination"):
        assert_disjoint_from_eval(provenance, tmp_path, ["gsm8k"])
