"""No-GPU input contract tests for trace collector."""
import json

import pytest
pytest.importorskip("torch")
from parallel_depth_poc.collect_states import _load_records


def test_full_teacher_forced_rows(tmp_path):
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join([
        json.dumps({"prompt": "A", "decode_token_ids": [1, 2]}),
        json.dumps({"prompt": "B", "decode_token_ids": [4, 5]}),
    ]) + "\n")
    assert len(_load_records(f, 2)) == 2


def test_rejects_mixed_teacher_forcing(tmp_path):
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join([
        json.dumps({"prompt": "A", "decode_token_ids": [1, 2]}),
        json.dumps({"prompt": "B"}),
    ]) + "\n")
    with pytest.raises(ValueError):
        _load_records(f, 2)


def test_rejects_missing_rows(tmp_path):
    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps({"prompt": "A"}) + "\n")
    with pytest.raises(ValueError):
        _load_records(f, 2)
