"""Per-row bank cache (kvp.rowbanks) equals the single-policy engine for every row."""

import pytest
import torch

from kvp.banks import REGISTRY
from kvp.rowbanks import run_bank_rows
from tests.test_policy_semantics import CONT, PROMPT, _tiny


@pytest.fixture(scope="module", params=["ouro", "huginn"])
def tiny(request):
    model, run = _tiny(request.param, 3)
    run.model_key = f"tinyrow-{request.param}"
    return request.param, model, run


def test_rows_match_single_policy_runs(tiny):
    family, model, run = tiny
    L, R = run.bb.n_rec, run.bb.R
    allb = frozenset((l, r) for l in range(L) for r in range(R - 1))
    sets = {"full": allb, "last": frozenset(), "p0": frozenset({(0, 0)}), "swap0": allb - {(0, 0)}}
    out = []
    run_bank_rows(run, PROMPT, CONT, list(sets.values()), "u", lambda g, lg: out.append(lg.clone()))
    got = torch.stack(out, 1)  # [B, n, V]
    P = len(PROMPT)
    refs = {"full": run.teacher_forced_logits(PROMPT, CONT, "u", "full", "decode"),
            "last": run.teacher_forced_logits(PROMPT, CONT, "u", "last", "convert")}
    for k in ("p0", "swap0"):
        REGISTRY[(run.model_key, f"bankrow{k}")] = sets[k]
        refs[k] = run.teacher_forced_logits(PROMPT, CONT, "u", f"bankrow{k}", "convert")
    for b, k in enumerate(sets):
        ref = refs[k][P - 1: P - 1 + len(CONT)]
        assert torch.allclose(got[b], ref, atol=1e-9), (k, (got[b] - ref).abs().max())
