"""CPU-only regression tests for saved-receipt rank analysis."""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from phase04_analysis.depth_rank_stability import (
    INPUT_ROOT,
    cluster_bootstrap_ci,
    pair_counts,
    pair_summary,
    rank_metrics,
    validate_receipt,
)


def row(request: int, reference: int, live: int) -> dict:
    return {"request_id": str(request), "reference_depth": str(reference), "live_depth": str(live)}


class RankStabilityTests(unittest.TestCase):
    def test_ties_and_constant_depth_pool(self):
        tied = [row(0, 1, 1), row(1, 1, 1), row(2, 2, 2)]
        metric = rank_metrics(tied)
        self.assertEqual(metric["status"], "defined")
        self.assertAlmostEqual(metric["kendall_tau_b"], 1.0)
        constant = rank_metrics([row(0, 1, 2), row(1, 1, 3)])
        self.assertEqual(constant["status"], "NA_constant")
        self.assertIsNone(constant["kendall_tau_b"])
        self.assertIsNone(constant["spearman_rho"])

    def test_all_pair_outcomes_and_empty_gap_bins(self):
        pool = [row(0, 1, 1), row(1, 2, 2), row(2, 3, 1), row(3, 4, 3)]
        counts = pair_counts(pool)
        self.assertGreater(sum(counts["1"].values()), 0)
        self.assertGreater(sum(counts["2"].values()), 0)
        self.assertGreater(sum(counts["3"].values()), 0)
        combined = {kind: sum(gap[kind] for gap in counts.values()) for kind in ("preserved", "collapsed_to_tie", "inverted")}
        self.assertGreater(combined["preserved"], 0)
        self.assertGreater(combined["collapsed_to_tie"], 0)
        self.assertGreater(combined["inverted"], 0)
        empty = pair_summary("ouro", 2, "fifo", "condition", ">=4", counts[">=4"])
        self.assertEqual(empty["comparable_pairs"], 0)
        self.assertIsNone(empty["preserved_rate"])

    def test_missing_request_in_bootstrap_pool_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing request"):
            cluster_bootstrap_ci([[row(0, 1, 1)]], [0, 1], reps=1, seed=1)

    def test_real_receipt_integrity_and_duplicate_rejection(self):
        rows, summaries, _, _ = validate_receipt("ouro", INPUT_ROOT)
        self.assertEqual(len(rows), 2304)
        self.assertEqual(len(summaries), 9)
        with tempfile.TemporaryDirectory() as temp:
            target_root = Path(temp)
            folder = target_root / "ouro_fresh_validation"
            folder.mkdir()
            for name in ("per_token_validation.csv", "batch_shape_validation.csv", "margin_bin_summary.csv", "validation_manifest.json"):
                shutil.copy(INPUT_ROOT / "ouro_fresh_validation" / name, folder / name)
            csv_path = folder / "per_token_validation.csv"
            with csv_path.open(newline="") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                copied = list(reader)
            copied[1]["request_id"] = copied[0]["request_id"]
            with csv_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(copied)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_receipt("ouro", target_root)
            copied.pop()
            with csv_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(copied)
            with self.assertRaisesRegex(ValueError, "row counts"):
                validate_receipt("ouro", target_root)


if __name__ == "__main__":
    unittest.main()
