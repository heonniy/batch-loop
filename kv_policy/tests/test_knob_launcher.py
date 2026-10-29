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



def test_ouro_token_actual_memory_and_quality_vs_loop1all(tmp_path):
    """A 40% protected-token rate costs 35% Full, saves 30% vs Loop1-all."""
    jobs = []
    for policy, correct in (("bankR1all", True), ("tokL1hash40", True)):
        j = {"model": "ouro-1.4b", "dataset": "gsm8k", "task": "gen",
             "policy": policy, "mode": "strict", "batch": 0}
        jobs.append(j)
        path = tmp_path / "raw" / f"{runner.job_name(j)}.jsonl"
        path.parent.mkdir(exist_ok=True)
        logical = {"n_processed_real_token_positions": 10,
                   "n_protected_token_positions": 4}
        meta = {"job_meta": {**j, "engine_stats": {
            "logical_retained_kv": logical if policy.startswith("tokL1") else {}}}}
        rec = {"uid": "a", "model": j["model"], "dataset": j["dataset"],
               "task": "gen", "policy": policy, "mode": "strict",
               "correct": correct, "finish_reason": "eos",
               "parse_ok": True, "n_gen_tokens": 10}
        path.write_text(json.dumps(meta) + "\n" + json.dumps(rec) + "\n")
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "knob_manifest.json").write_text(json.dumps({
        "spec": {"phase": "ouro_token", "prompt_style": "default"}}))
    (tmp_path / "banks.json").write_text(json.dumps({
        "ouro-1.4b": {"bankR1all": [[i, 0] for i in range(24)]}}))
    assert runner.report(tmp_path)["status"] == "complete"
    with (tmp_path / "analysis/token_loop_logical_memory.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["logical_full_recurrent_kv_percent"]) == pytest.approx(35.0)
    assert float(rows[0]["additional_memory_saved_vs_loop1all_percent"]) == pytest.approx(30.0)
    with (tmp_path / "analysis/sub50_paired_vs_loop1all.csv").open() as f:
        paired = list(csv.DictReader(f))
    assert len(paired) == 1
    assert paired[0]["policy"] == "tokL1hash40"
    assert paired[0]["budget_type"].startswith("observed")
    assert float(paired[0]["mean_delta_vs_loop1all"]) == 0.0


def test_huginn_stress_strata_on_matched_full_prefix(tmp_path):
    jobs = []
    for policy, success in (("full", True), ("last", False)):
        j = {"model": "huginn-0125-r16", "dataset": "math500", "task": "gen",
             "policy": policy, "mode": "strict", "batch": 0,
             "prompt_style": "distract512"}
        jobs.append(j)
        path = tmp_path / "raw" / f"{runner.job_name(j)}.jsonl"
        path.parent.mkdir(exist_ok=True)
        row = {"uid": "u", "model": j["model"], "dataset": j["dataset"],
               "task": "gen", "policy": policy, "mode": "strict",
               "correct": success, "finish_reason": "eos", "parse_ok": True,
               "n_gen_tokens": 300, "prompt_tokens": 900}
        path.write_text(json.dumps({"job_meta": j}) + "\n"
                        + json.dumps(row) + "\n")
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "knob_manifest.json").write_text(json.dumps({
        "spec": {"phase": "huginn_hard", "prompt_style": "distract512"}}))
    assert runner.report(tmp_path)["status"] == "complete"
    with (tmp_path / "analysis/huginn_hard_length_strata.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert any(x["axis"] == "prompt_length" and
               x["bucket"] == "513-1024" for x in rows)
    assert all(float(x["delta_last_minus_full"]) == -1.0 for x in rows)
