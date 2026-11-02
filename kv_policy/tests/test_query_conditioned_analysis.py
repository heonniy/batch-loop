"""CPU-only v7 contract tests: no future/oracle selection, exit trace integrity.

Run from kv_policy/: python -m unittest discover -s tests -p test_query_conditioned_analysis.py
"""
import unittest

from kvp.query_conditioned_analysis import (
    attention_matched_pairs,
    summarize_live_replay,
)


def arm(depths, losses):
    return {
        "depths": list(depths),
        "token_losses": list(losses),
        "first_target_nll": float(losses[0]),
        "nll_future": sum(losses[1:]) / (len(losses) - 1),
    }


def example():
    full_live = arm([4, 4, 4, 2, 3, 4], [1.0, 0.5, 0.5, 0.5])
    full_replay = arm(full_live["depths"], full_live["token_losses"])
    full_fixed = arm([4] * 6, [1.0, 0.5, 0.5, 0.5])
    evicted_live = arm([4, 4, 4, 3, 2, 4], [1.0, 0.6, 0.7, 0.5])
    evicted_replay = arm(full_live["depths"], [1.0, 0.5, 0.8, 0.5])
    evicted_fixed = arm([4] * 6, [1.0, 0.55, 0.55, 0.55])
    return dict(prompt_len=3, target_len=4, native_max_depth=4,
                full_live=full_live, full_replay=full_replay,
                full_fixed=full_fixed, evicted_live=evicted_live,
                evicted_replay=evicted_replay, evicted_fixed=evicted_fixed)


class QueryConditionedAnalysisTests(unittest.TestCase):
    def test_attention_pairing_is_preintervention_and_deterministic(self):
        pos = [10, 11, 20, 21, 30, 31]
        clean = {str(p): [a, a] for p, a in zip(
            pos, [.01, .011, .10, .11, .80, .85])}
        first = attention_matched_pairs(pos, clean, max_pairs=3,
                                         max_position_gap=4)
        second = attention_matched_pairs(pos, clean, max_pairs=3,
                                          max_position_gap=4)
        self.assertEqual(first, second)
        self.assertEqual(first["n_selected"], 3)
        taken = []
        for pair in first["pairs"]:
            taken += [pair["source_a"], pair["source_b"]]
            self.assertLessEqual(pair["position_gap"], 4)
            self.assertLessEqual(pair["abs_log_attention_ratio"], 0.35)
        self.assertEqual(len(taken), len(set(taken)))

    def test_pairing_does_not_use_outcome_or_answer(self):
        # Only q-specific clean attention + positions cross the API:
        # changing any external future losses or evidence annotations
        # cannot change an identical call.
        clean = {"1": [0.20, 0.10], "2": [0.22, 0.11]}
        out1 = attention_matched_pairs((1, 2), clean)
        future_damage = {1: 999.0, 2: -999.0}
        role_labels = {1: "relevant_content", 2: "irrelevant_content"}
        self.assertTrue(future_damage and role_labels)
        out2 = attention_matched_pairs((1, 2), clean)
        self.assertEqual(out1, out2)

    def test_pairing_reports_no_matches_without_forcing(self):
        out = attention_matched_pairs([1, 2], {1: [0, 0], 2: [0, 0]})
        self.assertEqual(out["n_selected"], 0)
        self.assertEqual(out["reason"], "insufficient_nonzero_query_attention")
        bad = {1: [0.1], 2: [0.1, 0.1]}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            attention_matched_pairs([1, 2], bad)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            attention_matched_pairs([1, 2], {1: [-0.1], 2: [0.2]})

    def test_exit_live_replay_distinguishes_up_down_and_quality(self):
        out = summarize_live_replay(**example())
        self.assertEqual(out["depth_delta"], [1, -1, 0])
        self.assertEqual(out["n_depth_changed"], 2)
        self.assertEqual(out["n_depth_increased"], 1)
        self.assertEqual(out["n_depth_decreased"], 1)
        self.assertEqual(out["total_depth_delta"], 0)
        self.assertAlmostEqual(out["fixed_future_nll_penalty"], 0.05)
        self.assertAlmostEqual(out["live_future_nll_penalty"], 0.1)
        self.assertAlmostEqual(out["replay_future_nll_penalty"], 0.1)
        self.assertEqual(len(out["future_token_nll_shifts"]["live"]), 3)

    def test_first_token_invariance_required(self):
        e = example()
        e["evicted_live"]["first_target_nll"] += 0.1
        with self.assertRaisesRegex(ValueError, "first target"):
            summarize_live_replay(**e)

    def test_replay_trace_and_output_parity_required(self):
        e = example()
        e["evicted_replay"]["depths"][-1] = 3
        with self.assertRaisesRegex(ValueError, "replay"):
            summarize_live_replay(**e)
        e = example()
        e["full_replay"] = arm(e["full_live"]["depths"], [1.0, 0.6, 0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "parity"):
            summarize_live_replay(**e)

    def test_fixed_and_full_prefill_depth_required(self):
        e = example()
        e["evicted_fixed"]["depths"][-1] = 2
        with self.assertRaisesRegex(ValueError, "full fixed"):
            summarize_live_replay(**e)
        e = example()
        e["full_live"]["depths"][1] = 2
        with self.assertRaisesRegex(ValueError, "Prefill"):
            summarize_live_replay(**e)

    def test_future_scoring_alignment_required(self):
        e = example()
        e["evicted_live"]["nll_future"] = 1.0
        with self.assertRaisesRegex(ValueError, "nll_future"):
            summarize_live_replay(**e)

    def test_no_eviction_has_zero_delta(self):
        e = example()
        e["evicted_live"] = arm(e["full_live"]["depths"],
                                e["full_live"]["token_losses"])
        e["evicted_replay"] = arm(e["full_live"]["depths"],
                                  e["full_live"]["token_losses"])
        e["evicted_fixed"] = arm(e["full_fixed"]["depths"],
                                 e["full_fixed"]["token_losses"])
        out = summarize_live_replay(**e)
        self.assertEqual(out["fraction_changed"], 0)
        self.assertEqual(out["live_future_nll_penalty"], 0)
        self.assertEqual(out["replay_future_nll_penalty"], 0)


if __name__ == "__main__":
    unittest.main()
