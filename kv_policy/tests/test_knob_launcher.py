"""CPU-only launcher contract tests; no HF downloads, network, or CUDA."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kv_knob_sweep.py"
spec = importlib.util.spec_from_file_location("kv_knob_sweep_module", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_phase_matrix_and_names():
    assert {"depth", "postlast", "oneplane", "factored"} <= set(runner.PHASES)
    assert "huginn-0125-r24" in runner.PHASES["depth"]["models"]
    assert "lastwin128-strict" in runner.PHASES["postlast"]["policies"]
    assert "blend50-strict" in runner.PHASES["oneplane"]["policies"]
    assert "bankP0win128-strict" in runner.PHASES["factored"]["policies"]
    assert all(policy.count("-") == 1 for phase in runner.PHASES.values()
               for policy in phase["policies"])
    job = {"model": "huginn-0125-r8", "dataset": "gsm8k", "policy": "last",
           "mode": "strict", "task": "nll", "batching": "pilot", "batch": 1}
    assert runner.job_name(job) == "huginn-0125-r8__gsm8k__nll__last-strict-pilot__b01"


def test_partial_nll_only_summary_and_paired_ci(tmp_path):
    jobs = []
    for pol, lp in (("full", -2.0), ("last", -3.0)):
        job = {"model": "huginn-0125", "dataset": "gsm8k",
               "policy": pol, "mode": "strict", "task": "nll", "batch": 0}
        jobs.append(job)
        path = tmp_path / "raw" / f"{runner.job_name(job)}.jsonl"
        path.parent.mkdir(exist_ok=True)
        record = {"uid": "u1", "model": "huginn-0125", "dataset": "gsm8k",
                  "task": "nll", "policy": pol, "mode": "strict",
                  "sum_logprob": lp, "target_tokens": 2, "nll_mean": -lp / 2}
        path.write_text(json.dumps({"job_meta": job}) + "\n" + json.dumps(record) + "\n")
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    out = runner.report(tmp_path)
    assert out["status"] == "complete"
    with (tmp_path / "analysis/screen_summary.csv").open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 2
    with (tmp_path / "analysis/paired_vs_full_strict.csv").open() as f:
        paired = list(csv.DictReader(f))
    assert len(paired) == 1
    assert paired[0]["metric"] == "delta_nll_per_problem"
    assert float(paired[0]["delta"]) == 0.5
    assert float(paired[0]["ci95_lo"]) == 0.5
