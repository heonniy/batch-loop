"""CPU-only RIC and fair-cohort tests."""

from __future__ import annotations

import tempfile
import gzip
import unittest
from pathlib import Path

from phase02_runtime.oracle_ric import characterize, load_depths, sha256_file


class OracleRicTests(unittest.TestCase):
    def test_known_grouping_and_no_reorder_control(self):
        matrix = [[2], [8], [2], [8]]
        summary, rows = characterize(matrix, 2, 8)
        self.assertEqual((summary["fifo_ric"], summary["oracle_ric"]), (16, 10))
        self.assertEqual(summary["oracle_ric_reduction_pct"], 37.5)
        self.assertEqual(summary["total_token_loop_work"], 20)
        self.assertEqual(len(rows), 1)
        control, _ = characterize(matrix, 4, 8)
        self.assertEqual(control["fifo_ric"], control["oracle_ric"])

    def test_trace_must_be_complete_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"request_id":0,"token_idx":0,"depth":2}\n')
            with self.assertRaisesRegex(ValueError, "incomplete trace"):
                load_depths(path, 2, 1, 4)
            path.write_text(path.read_text() + '{"request_id":0,"token_idx":0,"depth":2}\n')
            with self.assertRaisesRegex(ValueError, "duplicate depth"):
                load_depths(path, 1, 1, 4)

    def test_gzip_trace_preserves_raw_hash_and_depths(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "trace.jsonl"
            compressed = Path(directory) / "trace.jsonl.gz"
            raw.write_text('{"request_id":0,"token_idx":0,"depth":2}\n', encoding="utf-8")
            with raw.open("rb") as source, gzip.open(compressed, "wb") as target:
                target.write(source.read())
            self.assertEqual(sha256_file(raw), sha256_file(compressed))
            self.assertEqual(load_depths(compressed, 1, 1, 4), [[2]])


if __name__ == "__main__":
    unittest.main()
