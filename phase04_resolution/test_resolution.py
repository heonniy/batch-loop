"""CPU-only contracts for bounded divergence receipts."""

from __future__ import annotations

import unittest

import torch

from phase04_resolution.counterexamples import _mismatch_map
from phase04_resolution.trace_prefill import _diff


class ResolutionTests(unittest.TestCase):
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
