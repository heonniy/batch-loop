"""LoopZip v5 Stage 0/1: exact evidence offsets, evidence lock budgets, fixed-prefix KL."""
from __future__ import annotations

import pytest
import torch

from kvp.loopzip_evidence_audit import AlignmentError, coverage_row, evidence_spans
from kvp.loopzip_evidence_lock import fixed_prefix_kl, lock_select


def _cr_job():
    text = "Context:\nRecord R-7 has access code VX-12.\nRecord R-9 has access code VX-34.\nQ"
    offsets = [(i, i + 1) for i in range(len(text))]
    a = text.index("Record R-7")
    b = text.index("\nRecord R-9")
    c = text.index("Record R-9")
    d = text.rindex("\nQ")
    rel, irr = list(range(a, b)), list(range(c, d))
    job = {"prompt_text": text, "dataset": "controlled_retrieval", "gold_answer": "VX-12",
           "relevant_positions": rel, "content_positions": rel + irr,
           "context_spans": [{"id": "f1", "role": "relevant_content", "positions": rel},
                             {"id": "f2", "role": "irrelevant_content", "positions": irr}]}
    return job, offsets, text


def test_cr_key_value_subspans_exact():
    job, off, text = _cr_job()
    ev = evidence_spans(job, off)
    assert "".join(text[i] for i in ev["id"]) == "R-7"
    assert "".join(text[i] for i in ev["code"]) == "VX-12"
    assert not ev["ambiguous"]
    full = coverage_row(job["content_positions"], ev, job["content_positions"])
    assert full["complete_E"] and full["code_complete"] and full["coverage_E"] == 1
    miss = coverage_row([p for p in job["content_positions"] if p != ev["code"][0]], ev, job["content_positions"])
    assert not miss["complete_E"] and not miss["code_complete"] and miss["evidence_missed"] == 1


def test_cr_gold_mismatch_fails_closed():
    job, off, _ = _cr_job()
    job["gold_answer"] = "VX-99"
    with pytest.raises(AlignmentError):
        evidence_spans(job, off)


def test_lock_select_same_evidence_exact_k_and_infeasible():
    ctx = list(range(10, 50))
    E = [20, 21, 22, 23]
    att = {p: [((p * 7) % 13) / 13, ((p * 3) % 11) / 11] for p in ctx}
    for b in (0.1, 0.3, 0.5):
        sets = []
        for pol in ("mean_attn", "last_attn", "max_attn", "recent_attn", "random"):
            s = lock_select(ctx, E, att, b, pol, uid="u", seed=11)
            assert s["feasible"] and len(s["retained"]) == s["n_retained"] == max(1, int(b * 40))
            assert set(E) <= set(s["retained"])
            assert set(s["retained"]) | set(s["evicted"]) == set(ctx)
            sets.append(set(s["retained"]) & set(E))
        assert all(x == set(E) for x in sets)
    big = list(range(10, 20))
    assert lock_select(ctx, big, att, 0.1, "mean_attn", uid="u")["feasible"] is False


def test_fixed_prefix_kl_zero_and_positive():
    lp = torch.log_softmax(torch.randn(5, 7), -1)
    assert fixed_prefix_kl(lp, lp) == pytest.approx(0.0, abs=1e-7)
    other = lp.clone()
    other[2:] = torch.log_softmax(torch.randn(3, 7), -1)
    assert fixed_prefix_kl(lp, other) > 0
