"""CPU-only tests for WP3 selection metric and physical plan (no GPU)."""
import unittest
from collections import Counter

import numpy as np

from phase05.wp3_predictor import EVAL_B, PHYSICAL_POLICIES, REPEATS, lockstep_recovery, physical_plan


class WP3Tests(unittest.TestCase):
    def test_lockstep_recovery_bounds(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(1, 5, size=(3, 512))
        self.assertTrue(all(v == 1.0 for v in lockstep_recovery(labels, labels.astype(float)).values()))
        fifo = lockstep_recovery(labels, np.zeros(labels.shape))
        self.assertTrue(all(abs(v) < 1e-12 for v in fifo.values()))
        worst = lockstep_recovery(labels, -labels.astype(float))  # descending sort packs equally well
        self.assertTrue(all(v == 1.0 for v in worst.values()))

    def test_recovery_uses_cohort_blocks(self):
        labels = np.array([[1] * 256 + [4] * 256])
        # a global sort cannot mix blocks; each 256 block is constant, so FIFO == oracle -> undefined
        self.assertTrue(all(np.isnan(v) for v in lockstep_recovery(labels, labels.astype(float)).values()))

    def test_plan_balanced(self):
        plan = physical_plan()
        scored = Counter((c["cohort"], c["B"], c["policy"]) for c in plan if c["kind"] == "scored")
        self.assertEqual(set(scored.values()), {REPEATS})
        self.assertEqual(len(scored), 2 * len(EVAL_B) * len(PHYSICAL_POLICIES))


if __name__ == "__main__":
    unittest.main()
