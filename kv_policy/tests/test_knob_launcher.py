"""CPU-only launcher contract tests; no HF downloads, network, or CUDA."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kv_knob_sweep.py"
spec = importlib.util.spec_from_file_location("kv_knob_sweep_module", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_phase_matrix_and_names():
    assert {"depth", "postlast", "oneplane", "factored", "sub50", "ouro_token", "huginn_hard"} <= set(runner.PHASES)
    assert "huginn-0125-r24" in runner.PHASES["depth"]["models"]
    assert "lastwin128-strict" in runner.PHASES["postlast"]["policies"]
    assert "blend50-strict" in runner.PHASES["oneplane"]["policies"]
    assert "bankP0win128-strict" in runner.PHASES["factored"]["policies"]
    assert "bankTopB30-strict" in runner.PHASES["sub50"]["policies"]
    assert "bankPrefixB45-strict" in runner.PHASES["sub50"]["policies"]
    assert "tokL1hash40-strict" in runner.PHASES["ouro_token"]["policies"]
    assert "tokL1prompt-strict" in runner.PHASES["ouro_token"]["policies"]
    assert "huginn-0125-r48" in runner.PHASES["huginn_hard"]["models"]
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



def test_sub50_analysis_compares_against_loop1all(tmp_path):
    model, ds = "ouro-1.4b", "gsm8k"
    jobs = []
    for policy, correct in (("bankR1all", True), ("bankTopB30", False)):
        job = {"model": model, "dataset": ds, "task": "gen", "policy": policy,
               "mode": "strict", "batch": 0}
        jobs.append(job)
        path = tmp_path / "raw" / f"{runner.job_name(job)}.jsonl"
        path.parent.mkdir(exist_ok=True)
        rec = {"uid": "x", "model": model, "dataset": ds, "task": "gen",
               "policy": policy, "mode": "strict", "correct": correct,
               "finish_reason": "eos", "parse_ok": True, "n_gen_tokens": 5}
        path.write_text(json.dumps({"job_meta": job}) + "\n" + json.dumps(rec) + "\n")
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "knob_manifest.json").write_text(json.dumps({"spec": {"phase": "sub50"}}))
    (tmp_path / "banks.json").write_text(json.dumps({model: {
        "bankR1all": [[i, 0] for i in range(24)],
        "bankTopB30": [[i, 0] for i in range(4)]}}))
    assert runner.report(tmp_path)["status"] == "complete"
    with (tmp_path / "analysis/sub50_paired_vs_loop1all.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["policy"] == "bankTopB30"
    assert rows[0]["metric"] == "delta_accuracy_vs_50"
    assert float(rows[0]["mean_delta_vs_loop1all"]) == -1.0
    assert float(rows[0]["nominal_full_kv_percent"]) == pytest.approx(100 * 28 / 96)



def test_job_names_include_huginn_prompt_stress_style():
    j = {"model": "huginn-0125-r16", "dataset": "math500", "task": "gen",
         "policy": "last", "mode": "strict", "batching": "pilot",
         "prompt_style": "reason", "batch": 3}
    assert runner.job_name(j) == (
        "huginn-0125-r16__math500__gen__last-strict-pilot__stylereason__b03")
