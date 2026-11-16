"""CPU-only tests for WP2 split, policy ordering and run plan (no torch import)."""
import unittest
from collections import Counter

from phase05.wp2_headroom import BATCHES, POLICIES, REPEATS, policy_order, run_plan, split_prompts


class WP2PlanTests(unittest.TestCase):
    def test_splits_are_disjoint_and_exclude_prior_prefix(self):
        import random
        texts = [f"p{i}" for i in range(9000)] + ["p1", "p2"]  # duplicates collapse
        prior = texts.copy()
        random.Random(1).shuffle(prior)
        used = set(prior[:4096])
        parts = {s: split_prompts(texts, s) for s in ("wp2_headroom", "wp3_train", "wp3_validation", "wp3_test")}
        seen = set()
        for name, part in parts.items():
            self.assertFalse(set(part) & used, name)
            self.assertFalse(set(part) & seen, name)
            seen |= set(part)
        self.assertEqual(len(parts["wp2_headroom"]), 256)
        self.assertEqual(parts["wp2_headroom"], split_prompts(list(texts), "wp2_headroom"))

    def test_policies_are_causal_permutations(self):
        n = 6
        b1 = [[3, 1, 2, 1, 4, 2], [1, 1, 1, 4, 4, 2]]
        own = [[2, 2, 1, 4, 1, 3]]
        for policy in POLICIES:
            for t in (0, 1):
                order = policy_order(policy, t, n, own[:t], b1, 0)
                self.assertEqual(sorted(order), list(range(n)), (policy, t))
        self.assertEqual(policy_order("oracle_b1", 0, n, [], b1, 0), [1, 3, 2, 5, 0, 4])
        self.assertEqual(policy_order("history_last", 0, n, [], b1, 0), list(range(n)))
        self.assertEqual(policy_order("history_last", 1, n, own, b1, 0), [2, 4, 0, 1, 5, 3])
        self.assertEqual(policy_order("random", 1, n, own, b1, 7), policy_order("random", 1, n, own, b1, 7))
        with self.assertRaises(IndexError):  # history policies cannot read the current step
            policy_order("history_last", 2, n, own, b1, 0)

    def test_plan_is_balanced(self):
        plan = run_plan()
        scored = Counter((c["B"], c["policy"]) for c in plan if c["kind"] == "scored")
        self.assertEqual(set(scored.values()), {REPEATS})
        self.assertEqual(len(scored), len(BATCHES) * len(POLICIES))
        self.assertEqual(sum(c["kind"] == "control_no_composition" for c in plan), REPEATS)
        self.assertEqual(plan[0]["kind"], "warmup")


if __name__ == "__main__":
    unittest.main()
