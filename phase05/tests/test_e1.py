"""CPU-only checks for phase05/scripts/e1_regime_reanalysis.py.

  CUDA_VISIBLE_DEVICES="" python phase05/tests/test_e1.py -v
"""

import csv
import importlib.util
import itertools
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("e1", ROOT / "phase05/scripts/e1_regime_reanalysis.py")
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)
OUT = ROOT / "phase05/results/20261019_e1"


def brute_min_ric(depths, b):
    """Minimum sum of group maxima over every partition into groups of size b."""
    best = math.inf
    items = list(range(len(depths)))

    def rec(remaining, acc):
        nonlocal best
        if not remaining:
            best = min(best, acc)
            return
        first, rest = remaining[0], remaining[1:]
        for others in itertools.combinations(rest, b - 1):
            group = (first,) + others
            rec([i for i in rest if i not in others], acc + max(depths[i] for i in group))

    rec(items, 0)
    return best


class Arithmetic(unittest.TestCase):
    def test_ric_work_slot_hand_example(self):
        # one token boundary, N=4, B=2, depths [1,4,2,3]
        m = np.array([[1, 4, 2, 3]])
        fifo, orc = e1.fifo_oracle_metrics(m, 2)
        self.assertEqual(fifo["ric"], 4 + 3)          # {1,4}->4, {2,3}->3
        self.assertEqual(orc["ric"], 2 + 4)           # {1,2}->2, {3,4}->4
        self.assertEqual(fifo["work"], 10)
        self.assertAlmostEqual(fifo["slot_util"], 10 / (2 * 4 + 2 * 3))
        self.assertAlmostEqual(orc["slot_util"], 10 / (2 * 2 + 2 * 4))
        self.assertEqual(fifo["maxdepth_dist"], '{"3":1,"4":1}')

    def test_multi_token_sums_per_boundary(self):
        m = np.array([[1, 1, 4, 4], [4, 1, 4, 1]])  # [T=2, N=4]
        fifo, orc = e1.fifo_oracle_metrics(m, 2)
        self.assertEqual(fifo["ric"], (1 + 4) + (4 + 4))
        self.assertEqual(orc["ric"], (1 + 4) + (1 + 4))
        self.assertEqual(fifo["work"], orc["work"])

    def test_depth_sort_is_optimal_against_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            for n, b in ((6, 2), (6, 3), (8, 4), (8, 2)):
                d = rng.integers(1, 17, size=n)
                orc = e1.schedule_metrics(d[None, :], lambda x: e1.oracle_groups(x, b))
                self.assertEqual(orc["ric"], brute_min_ric(list(d), b))

    def test_variable_size_slot_utilization(self):
        m = np.array([[1, 2, 3, 4, 4]])
        met = e1.schedule_metrics(m, lambda d: [[0, 1, 2], [3, 4]])
        self.assertEqual(met["ric"], 3 + 4)
        self.assertAlmostEqual(met["slot_util"], 14 / (3 * 3 + 2 * 4))

    def test_each_request_exactly_once(self):
        m = np.array([[1, 2, 3, 4]])
        with self.assertRaises(AssertionError):
            e1.schedule_metrics(m, lambda d: [[0, 1], [1, 2]])
        with self.assertRaises(ValueError):
            e1.fifo_groups(6, 4)

    def test_reduction_vs_speedup(self):
        self.assertAlmostEqual(e1.reduction(100.0, 80.0), 0.2)
        self.assertAlmostEqual(e1.speedup(100.0, 80.0), 1.25)
        for a, b in ((100.0, 80.0), (10.0, 13.0), (7.0, 7.0)):
            self.assertAlmostEqual(e1.speedup(a, b), 1 / (1 - e1.reduction(a, b)))
        self.assertTrue(math.isnan(e1.reduction(100.0, float("nan"))))
        self.assertTrue(math.isnan(e1.speedup(float("nan"), 3.0)))
        self.assertTrue(math.isnan(e1.reduction(0.0, 0.0)))

    def test_missing_is_nan_not_zero(self):
        r = e1.base_row(cell_id="x", evidence_class="TRACE_ONLY", phase="t", runtime_kind="trace-sim", model="ouro",
                        N=8, B=4, ric_baseline=10, ric_policy=8)
        self.assertTrue(math.isnan(r["core_reduction"]))
        self.assertTrue(math.isnan(r["wall_ms_policy_median"]))
        self.assertAlmostEqual(r["ric_reduction"], 0.2)

    def test_depth_stats_hist_matches_values(self):
        vals = [1] * 3 + [2] * 5 + [4] * 2
        a, b = e1.depth_stats(vals), e1.depth_stats_from_hist({1: 3, 2: 5, 4: 2})
        for k in a:
            self.assertAlmostEqual(a[k], b[k])
        self.assertAlmostEqual(a["depth_cv"], np.std(vals) / np.mean(vals))

    def test_checksum_matches_phase02_implementation(self):
        sys.path.insert(0, str(ROOT))
        from phase02_runtime.oracle_ric import _cohort_hash
        m = np.array([[1, 2, 3], [4, 1, 2]])  # [T, N]
        self.assertEqual(e1.cohort_checksum(m), _cohort_hash(m.T.tolist()))

    def test_evidence_guard(self):
        with self.assertRaises(ValueError):
            e1.base_row(cell_id="x", evidence_class="PHYSICAL", phase="t", runtime_kind="", model="ouro", N=1, B=1)
        bad = e1.base_row(cell_id="x", evidence_class="LATENCY_MODEL", phase="t", runtime_kind="replay (not integrated runtime)",
                          model="ouro", N=8, B=4, core_ms_baseline_median=1.0, core_ms_policy_median=0.9)
        with self.assertRaises(AssertionError):
            e1.validate_cells([bad])


@unittest.skipUnless((OUT / "e1_regime_cells.csv").exists(), "full E1 outputs not generated")
class Outputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(OUT / "e1_regime_cells.csv", newline="") as f:
            cls.cells = list(csv.DictReader(f))
        with open(OUT / "e1_excluded_rows.csv", newline="") as f:
            cls.excluded = list(csv.DictReader(f))

    def test_evidence_classes_and_provenance(self):
        for r in self.cells:
            self.assertIn(r["evidence_class"], e1.EVIDENCE)
            self.assertTrue(r["source_files"] and r["source_rows"], r["cell_id"])

    def test_no_model_or_trace_row_labeled_physical(self):
        for r in self.cells:
            if r["evidence_class"] in ("TRACE_ONLY", "LATENCY_MODEL"):
                for c in ("core_ms_baseline_median", "core_ms_policy_median", "wall_ms_baseline_median",
                          "wall_ms_policy_median", "core_reduction", "wall_reduction"):
                    self.assertEqual(r[c], "", f"{r['cell_id']} {c}")
            if "replay" in r["runtime_kind"] or "trace-sim" in r["runtime_kind"]:
                self.assertFalse(r["evidence_class"].startswith("PHYSICAL"), r["cell_id"])
            if r["evidence_class"].startswith("PHYSICAL"):
                self.assertEqual(r["model_time_ms_policy"], "", r["cell_id"])

    def test_ric_reduction_and_slot_util_recompute(self):
        for r in self.cells:
            if r["ric_baseline"] and r["ric_policy"]:
                fb, fp = float(r["ric_baseline"]), float(r["ric_policy"])
                self.assertAlmostEqual(float(r["ric_reduction"]), (fb - fp) / fb, places=12)
                self.assertAlmostEqual(float(r["ric_ratio_baseline_over_policy"]), fb / fp, places=12)
                # all microbatches are full (N divisible by B): util = work / (B * RIC)
                if r["token_loop_work"] and r["slot_util_baseline"] and int(r["N"]) % int(r["B"]) == 0:
                    self.assertAlmostEqual(float(r["slot_util_baseline"]),
                                           float(r["token_loop_work"]) / (int(r["B"]) * fb), places=9)
            if r["is_n_eq_b_control"] == "True" and r["ric_policy"] and r["ric_baseline"]:
                self.assertEqual(float(r["ric_reduction"]), 0.0)

    def test_reconciled(self):
        for r in self.cells:
            self.assertFalse(r["reconciliation"].startswith("MISMATCH"), r["cell_id"])

    def test_native_and_oom_excluded(self):
        self.assertFalse(any("ouro_native_e2e.csv" in r["source_files"] for r in self.cells))
        oom = [r for r in self.excluded if r["status"] == "OOM_PARTIAL"]
        self.assertEqual(len(oom), 20)
        native = [r for r in self.excluded if r["evidence_kind"] == "NATIVE_WHOLE_MODEL_POSTHOC_EXIT_BASELINE"]
        self.assertEqual(len(native), 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
