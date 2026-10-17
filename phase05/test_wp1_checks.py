"""CPU-only tests for the WP1 preregistered gate checks (no torch import)."""
import unittest

from phase05.wp1_confirmation import (N, cohort_errors, fresh_prompts, orders,
                                      ordering_flags, recompute_depth, row_identity_errors)


class WP1CheckTests(unittest.TestCase):
    def test_cohort_is_fresh_and_orders_are_permutations(self):
        prompts = fresh_prompts()
        self.assertEqual(len(set(prompts)), N)
        self.assertFalse(any("Validation request" in p for p in prompts))
        for name, order in orders().items():
            self.assertEqual(sorted(order), list(range(N)), name)
        self.assertEqual(orders()["fifo"], orders()["fifo_repeat"])
        self.assertNotEqual(orders()["permutation_seed_41"], orders()["permutation_seed_53"])

    def test_comparator_threshold_and_cap(self):
        self.assertEqual(recompute_depth("ouro", [0.1, 0.25, 0.9, 1.0], 0.2, 4), 2)
        self.assertEqual(recompute_depth("ouro", [0.2], 0.2, 4), 1)  # >= is a pass
        self.assertEqual(recompute_depth("ouro", [0.0, 0.1, 0.1, 0.1], 0.2, 4), 4)  # cap
        self.assertEqual(recompute_depth("huginn", [0.5, 0.12, 0.11], 0.12, 16), 3)  # strict <
        self.assertEqual(recompute_depth("huginn", [0.5, 0.4], 0.12, 16), -1)  # truncated

    def test_cohort_widths_and_refill_detection(self):
        members = [0, 1, 2]
        ok = cohort_errors(members, [1, 3, 2], [3, 2, 1], [[0, 1, 2], [1, 2], [1]],
                           [[0, 1, 2], [1, 2], [1]], 3)
        self.assertEqual(ok, [])
        self.assertTrue(cohort_errors(members, [1, 3, 2], [3, 2, 2], [[0, 1, 2], [1, 2], [1, 2]],
                                      [[0, 1, 2], [1, 2], [1, 2]], 3))
        refill = cohort_errors(members, [1, 3, 2], [3, 2, 1], [[0, 1, 2], [1, 7], [1]],
                               [[0, 1, 2], [1, 7], [1]], 3)
        self.assertTrue(any("refill" in e for e in refill))
        self.assertTrue(cohort_errors([0, 0, 1], [1, 1, 1], [3], [[0, 0, 1]], [[0, 0, 1]], 3))

    def test_row_identity(self):
        full = [(t, r) for t in range(2) for r in range(3)]
        self.assertEqual(row_identity_errors(full, 2, 3), [])
        self.assertTrue(row_identity_errors(full[:-1], 2, 3))
        self.assertTrue(row_identity_errors(full + [(0, 0)], 2, 3))

    def test_ordering_flags(self):
        ref = {(0, 0): 1, (0, 1): 2, (0, 2): 4}
        self.assertEqual(ordering_flags(ref, {(0, 0): 2, (0, 1): 2, (0, 2): 4})["status"],
                         "ordering stable at observed scope")
        flipped = ordering_flags(ref, {(0, 0): 3, (0, 1): 2, (0, 2): 2})
        self.assertEqual(flipped["status"], "ordering materially unstable")
        self.assertEqual(flipped["max_abs_delta"], 2)
        self.assertTrue(flipped["gap_ge2_inversions"])


if __name__ == "__main__":
    unittest.main()
