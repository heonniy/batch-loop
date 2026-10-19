"""CPU-only tests for E3 cost arithmetic and groupings."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e3_latency_oracle import (depth_oracle, fifo, group_cost, latency_oracle, metrics,  # noqa: E402
                               plan_groups)


class Affine:
    def __init__(self, a, c):
        self.a, self.c = a, c

    def __call__(self, w, L):
        return self.a + self.c * w


class Step:
    """Cost jumps above width 4 (e.g. a tile boundary)."""

    def __call__(self, w, L):
        return 1.0 if w <= 4 else 3.0


class E3Tests(unittest.TestCase):
    def test_group_cost_sums_widths_per_recurrence(self):
        # depths 1,2,3 -> widths 3,2,1
        self.assertAlmostEqual(group_cost([1, 2, 3], 0, Affine(0, 1)), 6.0)
        self.assertAlmostEqual(group_cost([1, 2, 3], 0, Affine(1, 0)), 3.0)  # = RIC when cost is flat

    def test_metrics_ric_work_slots(self):
        d = [[1, 4, 1, 4]]
        m = metrics([fifo(d[0], 2)], d, [0], None)
        self.assertEqual((m["RIC"], m["work"]), (8, 10))
        self.assertAlmostEqual(m["slot_utilization"], 10 / 16)
        m = metrics([depth_oracle(d[0], 2)], d, [0], None)
        self.assertEqual((m["RIC"], m["work"]), (5, 10))

    def test_latency_oracle_never_worse_and_equal_under_affine(self):
        d = [3, 1, 2, 4, 1, 1, 2, 3, 4, 2, 1, 3]
        for model in (Affine(1, 0.1), Affine(0.2, 1), Step()):
            for B in (2, 3, 4, 6):
                g, _ = latency_oracle(d, B, L=0, model=model)
                lat = sum(group_cost([d[i] for i in x], 0, model) for x in g)
                dep = sum(group_cost([d[i] for i in x], 0, model) for x in depth_oracle(d, B))
                self.assertLessEqual(lat, dep + 1e-9)
                if isinstance(model, Affine):
                    self.assertAlmostEqual(lat, dep)  # affine cost: depth sorting is already optimal

    def test_plan_covers_each_request_once(self):
        depth = [[1, 2, 3, 4, 1, 2, 3, 4], [4, 3, 2, 1, 4, 3, 2, 1]]
        for p in ("fifo", "depth_oracle", "latency_oracle", "random"):
            plan, _ = plan_groups(p, depth, 4, [10, 11], Affine(1, 0.1))
            self.assertEqual(len(plan), 2)


if __name__ == "__main__":
    unittest.main()
