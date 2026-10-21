"""CPU tests for Phase-07 metrics and grouping utilities."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import kendall_tau_b_within_step, lockstep_ric, prob_metrics, recovery  # noqa: E402
from e4_utility import evaluate  # noqa: E402


class Phase07Tests(unittest.TestCase):
    def test_ric_and_recovery(self):
        d = np.array([[1, 4, 1, 4]])
        self.assertEqual(lockstep_ric(d, None, 2), 8)
        self.assertEqual(lockstep_ric(d, d.astype(float), 2), 5)
        self.assertAlmostEqual(recovery(8, 5, 5), 1.0)
        self.assertTrue(np.isnan(recovery(5, 5, 5)))

    def test_kendall_ties(self):
        d = np.array([[1, 2, 3, 3]])
        self.assertAlmostEqual(kendall_tau_b_within_step(d, d.astype(float)), 1.0)
        self.assertAlmostEqual(kendall_tau_b_within_step(d, -d.astype(float)), -1.0)

    def test_prob_metrics(self):
        p = np.array([[0.9, 0.1], [0.2, 0.8]])
        m = prob_metrics(p, np.array([1, 2]))
        self.assertAlmostEqual(m["acc"], 1.0)
        self.assertGreater(m["nll"], 0)

    def test_e4_slot_utilization_and_expected_max(self):
        d = np.array([[1, 4, 1, 4]])
        probs = np.zeros((1, 4, 4))
        probs[0, [0, 2], 0] = 1
        probs[0, [1, 3], 3] = 1
        r = evaluate(d, d.astype(float), probs, 4, 2)
        self.assertEqual(r["RIC"], 5)
        self.assertAlmostEqual(r["slot_utilization"], 10 / 10)
        self.assertAlmostEqual(r["expected_max_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
