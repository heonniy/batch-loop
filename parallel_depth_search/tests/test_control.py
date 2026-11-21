"""Stdlib-only tests. Synthetic fixtures are NEVER scientific measurements."""
import json
import tempfile
import unittest
from pathlib import Path

from parallel_depth_search import control, make_jobs, run_jobs


class TreeTest(unittest.TestCase):
    def test_preregistration_is_valid(self):
        p = control.program()
        self.assertEqual(len(p["nodes"]), 23)
        self.assertEqual(sum(n["planned_gpu_hours"] for n in p["nodes"]), 384)
        self.assertTrue(control.validate_tree(p))

    def test_cycle_rejected(self):
        p = json.loads(control.TREE.read_text())
        p["nodes"][0]["requires_all"] = ["F1_FULL_ROLLOUT"]
        with self.assertRaises(ValueError):
            control.validate_tree(p)

    def test_initial_status_has_no_gpu_claim(self):
        with tempfile.TemporaryDirectory() as d:
            m = control.init_run(d)
            self.assertEqual(m["status"], "PLAN_ONLY_UNTIL_REAL_RECEIPTS")
            s = control.propose(d)
            self.assertEqual(s["run_gpu_hours"], 0)
            self.assertIn("G0_NATIVE", {x["id"] for x in s["eligible_next"]})
            self.assertIn("G0_DATA", {x["id"] for x in s["eligible_next"]})
            self.assertIn("J1_PARETO", {x["id"] for x in s["blocked"]})

    def test_receipt_fails_without_primary_paper_check(self):
        p = control.program()
        r = dict(node_id="G0_NATIVE", state="executed", trial_id="FAKE_TEST_FIXTURE",
                 source_git_sha="TEST_ONLY", command="not run", data_split="pilot",
                 model="SYNTHETIC", seed=0, method="mock", literature_checks=[],
                 metrics={"sample_count": 16, "native_parity_pass": True},
                 hardware={"gpu_name": "SIMULATED_NOT_A_REAL_DEVICE"},
                 artifacts={"sha256": "synthetic"}, gpu_hours=1.0)
        with self.assertRaisesRegex(ValueError, "inspected references"):
            control.validate_receipt(r, p)

    def test_test_split_forbidden_before_final(self):
        p = control.program()
        r = dict(node_id="C1_JACOBI", state="not_run", trial_id="FAKE_TEST_FIXTURE",
                 source_git_sha="TEST_ONLY", command="not run", data_split="test",
                 model="SYNTHETIC", seed=0, method="mock", literature_checks=[],
                 metrics={}, hardware={}, artifacts={}, reason="test fixture")
        with self.assertRaisesRegex(ValueError, "held-out test"):
            control.validate_receipt(r, p)

    def test_append_only_not_run_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            control.init_run(d)
            r = dict(node_id="G0_NATIVE", state="not_run", trial_id="no-gpu-test",
                     source_git_sha="TEST_ONLY", command="no command executed",
                     data_split="pilot", model="SIMULATED", seed=0,
                     method="no run", literature_checks=[], metrics={},
                     hardware={}, artifacts={}, gpu_hours=0,
                     reason="no allocated GPU; synthetic unit fixture")
            p = Path(d) / "fixture.json"
            p.write_text(json.dumps(r))
            result = control.ingest(d, p)
            self.assertEqual(result["classification"], "not_run")
            with self.assertRaisesRegex(ValueError, "already ingested"):
                control.ingest(d, p)
            self.assertEqual(len(control.read_ledger(d)), 1)

    def test_generate_foundation_jobs_does_not_start_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            g = make_jobs.generate(d, max_jobs=20)
            self.assertEqual(g["status"], "PLANNED_ONLY_NO_GPU_JOB_LAUNCHED")
            self.assertTrue(g["jobs"])
            self.assertIn("G0_NATIVE", {x["node_id"] for x in g["jobs"]})
            self.assertIn("G0_DATA", {x["node_id"] for x in g["manual_nodes"]})

    def test_executor_refuses_implicit_gpu(self):
        with self.assertRaisesRegex(RuntimeError, "explicitly gated"):
            run_jobs.run_manifest("not-found.json", "unused", [0], 12, allowed=False)


if __name__ == "__main__":
    unittest.main()
