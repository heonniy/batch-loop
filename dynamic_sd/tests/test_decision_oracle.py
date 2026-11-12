"""CPU tests: strict target equivalence and noncausal oracle labeling."""
import unittest

from dynamic_sd.decision_oracle import action_name, analyze, collect_receipts


def sample_file(cfg_override=None, n=75):
    cfg = dict(model="ouro-1.4b", dataset="gsm8k", exit="live", tau=0.4, eps=None,
               dtype="bfloat16", sampling="greedy", seed=100, T_total=4,
               max_new_tokens=16, max_prompt_tokens=1024, kv_s=0,
               T_draft=1, K=4)
    cfg.update(cfg_override or {})
    samples = []
    for i in range(n):
        target = [1, i + 5, 99]
        samples.append({
            "idx": i, "prefix_len": 35,
            "tokens_ar": target, "wall_ar": 1.0,
            "tokens_wfd": target if i % 2 == 0 else [1, 4, 99],
            "wall_wfd": 0.7,
        })
    return {"meta": {"args": cfg}, "per_sample": samples}


class OracleTest(unittest.TestCase):
    def test_action_names(self):
        self.assertEqual(action_name("ar", {}), "ar")
        self.assertEqual(action_name("wfd", dict(T_draft=2)), "wfd:d2")
        self.assertEqual(action_name("dtv", dict(T_draft=4, K=8)), "dtv:d4:K8")

    def test_collect_equivalence_and_unavailable_actions(self):
        table = collect_receipts([("run-a", sample_file(n=2))])
        self.assertEqual(len(table["groups"]), 2)
        eligible = table["groups"][0]["actions"]
        self.assertTrue(any(x["action"] == "wfd:d1" and x["target_equivalent"] for x in eligible))
        ineligible = table["groups"][1]["actions"]
        self.assertTrue(any(x["action"] == "wfd:d1" and not x["target_equivalent"] for x in ineligible))

    def test_reference_mismatch_blocks_prefix(self):
        a = sample_file(n=3)
        b = sample_file(n=3)
        b["per_sample"][1]["tokens_ar"] = [1, 888, 99]
        table = collect_receipts([("run-a", a), ("run-b", b)])
        self.assertEqual(len(table["groups"]), 2)
        self.assertTrue(any("changed" in issue["reason"] for issue in table["issues"]))

    def test_noncausal_heldout(self):
        table = collect_receipts([("run-a", sample_file(n=100))])
        r = analyze(table)["reports"][0]
        self.assertEqual(r["status"], "hindsight_diagnostic_only")
        self.assertEqual(r["train_selected_fixed_action"], "ar")
        self.assertGreaterEqual(r["oracle_over_fixed_factor"], 1.)
        self.assertIn("Hindsight", r["caution"])

    def test_invalid_config_and_small_split(self):
        with self.assertRaises(ValueError):
            collect_receipts([("wrong-kv", sample_file(dict(kv_s=1)))])
        with self.assertRaises(ValueError):
            collect_receipts([("replay", sample_file(dict(exit="replay")))])
        table = collect_receipts([("one", sample_file(n=1))])
        self.assertEqual(analyze(table)["reports"][0]["status"], "insufficient_split")


if __name__ == "__main__":
    unittest.main()
