"""CPU-only regression tests for strict no-refill frozen cohort scheduling."""
import csv
import tempfile
import unittest
from pathlib import Path

from phase05.frozen_cohort_sim import Cost, load_trace, simulate


class FrozenCohortTests(unittest.TestCase):
    def test_oracle_packing_reduces_ric_on_frozen_work(self):
        traces = {0: [1], 1: [4], 2: [1], 3: [4]}
        fifo = simulate(traces, 2, "fifo")
        oracle = simulate(traces, 2, "oracle_depth")
        self.assertEqual(fifo["recurrent_invocations_RIC"], 8)
        self.assertEqual(oracle["recurrent_invocations_RIC"], 5)
        self.assertEqual(fifo["token_loop_work"], oracle["token_loop_work"])
        self.assertEqual(oracle["idle_slots_within_cohorts"], 0)
        self.assertEqual(fifo["evidence_kind"], "CPU_FROZEN_TRACE_ABSTRACT_COST_NOT_LIVE")

    def test_shrink_only_and_request_dependency(self):
        traces = {0: [1, 2, 1], 1: [4, 1], 2: [2, 3]}
        for policy in ("fifo", "static_first", "history_last",
                       "history_mean", "history_long_first", "oracle_depth"):
            result = simulate(traces, 2, policy)
            seen = {rid: -1 for rid in traces}
            for cohort in result["events"]:
                sizes = cohort["survivor_widths"]
                self.assertEqual(sizes, sorted(sizes, reverse=True))
                self.assertEqual(sizes[0], len(cohort["requests"]))
                self.assertLessEqual(len(cohort["requests"]), 2)
                for rid in cohort["requests"]:
                    self.assertEqual(cohort["token_idx"][rid], seen[rid] + 1)
                    seen[rid] += 1
            self.assertEqual(seen, {0: 2, 1: 1, 2: 1})
            self.assertEqual(result["token_loop_work"], 14)
            self.assertEqual(result["tokens"], 7)

    def test_cost_is_measured_in_abstract_units(self):
        traces = {0: [1], 1: [2]}
        cost = Cost(core_fixed=1, core_per_token=.5,
                    prelude_fixed=2, prelude_per_token=.5,
                    coda_fixed=3, coda_per_token=1)
        result = simulate(traces, 2, "fifo", cost)
        # Prelude 3 + cores (2+1.5) + Coda 5 = 11.5
        self.assertEqual(result["simulated_time_units"], 11.5)

    def test_reader_explicit_condition_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "trace.csv"
            columns = ["model", "B", "order", "request_id",
                       "token_idx", "reference_depth", "live_depth"]
            with file.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for b in (2, 4):
                    for rid, dep in ((0, 1), (1, 3)):
                        writer.writerow(dict(model="ouro", B=b, order="fifo",
                                             request_id=rid, token_idx=0,
                                             reference_depth=dep, live_depth=dep+1))
            with self.assertRaises(ValueError):
                load_trace(file)
            self.assertEqual(load_trace(file, model="ouro",
                                        source_B=2, order="fifo"),
                             {0: [1], 1: [3]})
            with file.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writerow(dict(model="ouro", B=2, order="fifo",
                                     request_id=0, token_idx=0,
                                     reference_depth=1, live_depth=2))
            with self.assertRaises(ValueError):
                load_trace(file, model="ouro", source_B=2, order="fifo")

    def test_reject_invalid(self):
        for bad in ({}, {0: []}, {0: [0]}, {0: [-1]}):
            with self.assertRaises(ValueError):
                simulate(bad, 2, "fifo")
        with self.assertRaises(ValueError):
            simulate({0: [1]}, 0, "fifo")
        with self.assertRaises(ValueError):
            simulate({0: [1]}, 2, "fifo", Cost(core_fixed=-1))


if __name__ == "__main__":
    unittest.main()
