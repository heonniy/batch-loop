"""CPU-only audit tests for measured physical-run summaries."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from phase02_runtime.summarize_oracle_physical import TIMING_COLUMNS, summarize_run


class PhysicalSummaryTests(unittest.TestCase):
    def test_measured_medians_and_equal_work(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest = {
                "stage": "m2", "repeats": 2, "warmups": 1, "source_sha": "source",
                "gpu": {"physical_gpu": 7, "gpu_query": "7, gpu-uuid, H100, 80 GiB"},
                "model": "huginn", "model_revision": "model", "kv_backend": "shared",
                "exit_source": "trace", "trace": {"sha256": "trace"},
                "prompt_source": "alpaca", "prompt_sha256": "prompts",
                "threshold": 0.35, "num_rows": 4, "batch_size": 2, "num_tokens": 1,
            }
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows = []
            for strategy, ric, cores, walls in (
                ("fifo", 5, (10, 12), (20, 22)),
                ("oracle", 3, (7, 9), (15, 17)),
            ):
                for repetition in range(2):
                    row = {
                        "strategy": strategy, "repetition": repetition,
                        "generated_tokens": 4, "num_intervals": 2,
                        "total_token_loop_work": 9, "same_work_checksum": "same",
                        "actual_kv_writes": 9, "last_exited_fill_operations": 0,
                        "shared_kv_overwrites": 9,
                        "recurrent_invocation_count": ric,
                    }
                    row.update({column: 1.0 for column in TIMING_COLUMNS})
                    row["recurrent_pass_gpu_ms_including_kv"] = cores[repetition]
                    row["total_decode_wall_ms"] = walls[repetition]
                    rows.append(row)
            with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = summarize_run(run_dir)
            self.assertEqual((result["fifo_ric"], result["oracle_ric"]), (5, 3))
            self.assertEqual(result["fifo_median_recurrent_pass_gpu_ms_including_kv"], 11)
            self.assertEqual(result["oracle_median_recurrent_pass_gpu_ms_including_kv"], 8)
            self.assertEqual(result["fifo_median_total_decode_wall_ms"], 21)
            self.assertEqual(result["oracle_median_total_decode_wall_ms"], 16)

            rows[-1]["same_work_checksum"] = "different"
            with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "same_work_checksum"):
                summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
