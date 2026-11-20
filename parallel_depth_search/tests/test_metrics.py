"""All arrays below are synthetic unit fixtures, NOT experimental evidence."""
import unittest
from parallel_depth_search.paired_metrics import align, aggregate, paired_bootstrap
from parallel_depth_search.rank_candidates import assess, family, shortlist
from parallel_depth_search.control import program


class PairedTest(unittest.TestCase):
    def test_pairing_and_speedups(self):
        rows = []
        for i in range(8):
            tokens = [1, i, 2, 3, 4]
            rows.append(dict(prompt_idx=i, arm="AR", tokens=tokens,
                             n_tokens=5, wall_s=2.0, correct_strict=i % 2 == 0))
            rows.append(dict(prompt_idx=i, arm="test",
                             tokens=tokens if i < 4 else [3, i, 2, 3, 4],
                             n_tokens=5, wall_s=1.0, correct_strict=i % 2 == 0))
        pairs = align(rows, "AR", "test")
        self.assertEqual(len(pairs), 8)
        agg = aggregate(pairs)
        self.assertAlmostEqual(agg["speedup"], 2.0)
        self.assertAlmostEqual(agg["requests_per_sec_speedup"], 2.0)
        self.assertAlmostEqual(agg["sequence_exact_rate"], 0.5)
        ci = paired_bootstrap(pairs, reps=200, seed=4)
        self.assertAlmostEqual(ci["speedup_ci_low"], 2.0)
        self.assertAlmostEqual(ci["accuracy_delta_ci_low"], 0.0)
        self.assertTrue(ci["unverified_kv_fidelity"])

    def test_unknown_sequences_are_not_exact(self):
        rows = [dict(prompt_idx=0, arm=arm, n_tokens=1, wall_s=1.0,
                     correct_strict=True) for arm in ("AR","test")]
        with self.assertRaisesRegex(ValueError, "token IDs"):
            aggregate(align(rows, "AR", "test"))

    def test_duplicate_pair_rejected(self):
        rows = [dict(prompt_idx=0, arm="AR")] * 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align(rows, "AR", "test")


class RankingTest(unittest.TestCase):
    def receipt(self, *, tier="quality_noninferior", kv=True, split="dev"):
        return {
            "trial_id": "UNIT_FIXTURE_ONLY", "node_id": "J1_PARETO",
            "state": "executed", "model": "SYNTHETIC",
            "method": "newton", "data_split": split,
            "hardware": {"gpu_name": "SIMULATED_H200"},
            "metrics": {
                "depth": 32, "batch": 16, "prompt_len": 256,
                "dtype": "bf16", "kv_policy": "FULL",
                "sample_count": 300, "speedup_ci_low": 1.2,
                "accuracy_delta_ci_low": -0.01, "full_e2e_measured": True,
                "fidelity_tier": tier, "unverified_kv_fidelity": kv,
            }
        }

    def test_approximate_quality_matched_can_be_ranked_but_not_called_exact(self):
        a = assess(self.receipt(), program())
        self.assertTrue(a["rankable"])
        self.assertEqual(a["tier"], "measured_full_e2e")
        b = assess(self.receipt(tier="exact"), program())
        self.assertFalse(b["rankable"])

    def test_no_adaptive_ranking_on_final_test(self):
        a = assess(self.receipt(split="test"), program())
        self.assertFalse(a["rankable"])

    def test_grouped_shortlist_preserves_family_diversity(self):
        one = self.receipt()
        two = self.receipt()
        two["trial_id"] = "UNIT_FIXTURE_2"
        two["method"] = "anderson"
        two["metrics"] = dict(two["metrics"], speedup_ci_low=1.15)
        out = shortlist([{"receipt": one}, {"receipt": two}])
        val = list(out["grouped_candidate_shortlists"].values())[0]
        self.assertEqual({x["family"] for x in val}, {"newton", "anderson"})
        self.assertTrue(out["no_test_split_used"])


if __name__ == "__main__":
    unittest.main()
