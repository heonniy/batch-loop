"""CPU-only controlled Huginn stress-prompt tests.

These are system-instruction / irrelevant-context interventions on the
SAME frozen math questions. They are not an official 'thinking mode' switch.
"""
import pytest

from kvp import jobs


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


def test_huginn_reason_vs_default_preserves_target(monkeypatch):
    source = [{"uid": "item1", "question": "Compute the result.", "answer": "steps #### 42"}]
    monkeypatch.setattr(jobs, "load_subset", lambda dataset: (source, {"source": "fixture"}))
    monkeypatch.setattr(jobs, "render", lambda d, p, tok:
                        f"SYSTEM={p.system_instruction}\nQUESTION={d['question']}")
    monkeypatch.setattr(jobs, "gold_target", lambda d, p: " 42")
    tok = FakeTokenizer()
    base, meta0 = jobs.prepare_items(
        "huginn-0125", "gsm8k", tok, "main", 1, "default")
    reason, meta1 = jobs.prepare_items(
        "huginn-0125", "gsm8k", tok, "main", 1, "reason")
    assert base[0]["uid"] == reason[0]["uid"]
    assert base[0]["target_ids"] == reason[0]["target_ids"]
    assert base[0]["prompt"] != reason[0]["prompt"]
    assert "step by step" in reason[0]["prompt"]
    assert meta0["prompt_style"] == "default"
    assert meta1["prompt_style"] == "reason"


def test_distractor_stress_changes_context_not_solution(monkeypatch):
    source = [{"uid": "item1", "question": "Compute the result.", "answer": "steps #### 42"}]
    monkeypatch.setattr(jobs, "load_subset", lambda dataset: (source, {"source": "fixture"}))
    monkeypatch.setattr(jobs, "render", lambda d, p, tok: d["question"])
    monkeypatch.setattr(jobs, "gold_target", lambda d, p: " 42")
    base, _ = jobs.prepare_items(
        "huginn-0125", "gsm8k", FakeTokenizer(), "main", 1, "default")
    long, _ = jobs.prepare_items(
        "huginn-0125", "gsm8k", FakeTokenizer(), "main", 1, "distract512")
    assert "Actual problem:\nCompute the result." in long[0]["prompt"]
    assert "Unrelated background" in long[0]["prompt"]
    assert len(long[0]["prompt_ids"]) >= len(base[0]["prompt_ids"]) + 512
    assert base[0]["target_ids"] == long[0]["target_ids"]
    # Original source doc must remain immutable.
    assert source[0]["question"] == "Compute the result."


def test_unsupported_stress_is_rejected_before_model_execution(monkeypatch):
    monkeypatch.setattr(jobs, "load_subset", lambda ds: (
        [{"uid": "item1", "question": "Q", "answer": "A #### 1"}], {}))
    with pytest.raises(ValueError, match="Huginn-only"):
        jobs.prepare_items("ouro-1.4b", "gsm8k", FakeTokenizer(),
                           "main", 1, "reason")


def test_job_names_isolate_prompt_variants():
    j = {"model": "huginn-0125-r16", "dataset": "gsm8k", "task": "gen",
         "policy": "last", "mode": "strict", "batch": 1, "batching": "pilot"}
    assert jobs.job_name(j).endswith("last-strict-pilot__b01")
    assert "__stylereason__b01" in jobs.job_name({**j, "prompt_style": "reason"})
