"""CPU-only contracts for bounded divergence receipts."""

from __future__ import annotations

import unittest

import torch

from phase04_resolution.batch_shape_matrix import _selected
from phase04_resolution.counterexamples import _mismatch_map
from phase04_resolution.trace_prefill import _diff


class ResolutionTests(unittest.TestCase):
    def test_fixed_fillers_change_only_target_position_and_order(self):
        fillers = [1, 2, 3, 4, 5, 6, 7]
        first = _selected(22, fillers, 4, 0, False)
        last = _selected(22, fillers, 4, 3, True)
        self.assertEqual(first, [22, 1, 2, 3])
        self.assertEqual(last, [3, 2, 1, 22])
        self.assertEqual(set(first), set(last))

    def test_mismatch_map_keeps_depth_and_token_fields(self):
        rows = [{"token_idx": 1, "request_id": 7, "field": "depth"},
                {"token_idx": 1, "request_id": 7, "field": "next_token"}]
        selected = _mismatch_map({"decision_mismatches": rows})
        self.assertEqual(len(selected), 1)
        self.assertEqual([row["field"] for row in selected[(1, 7)]], ["depth", "next_token"])

    def test_target_valid_positions_ignore_padding(self):
        left = torch.tensor([[99.0], [1.0], [2.0]])
        right = torch.tensor([[0.0], [1.0], [2.0]])
        result = _diff(left, right, torch.tensor([False, True, True]))
        self.assertTrue(result["exact_equal"])
        self.assertEqual(result["max_abs"], 0.0)

    def test_target_valid_positions_detect_first_difference(self):
        left = torch.tensor([[99.0], [1.0], [2.0]])
        right = torch.tensor([[99.0], [1.0], [3.0]])
        result = _diff(left, right, torch.tensor([False, True, True]))
        self.assertFalse(result["exact_equal"])
        self.assertEqual(result["max_abs"], 1.0)


if __name__ == "__main__":
    unittest.main()
