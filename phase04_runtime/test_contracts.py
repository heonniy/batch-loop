"""CPU regressions for the Phase-4 policy, identity and split gates."""

from __future__ import annotations

import unittest

import torch

from phase02_runtime.kv import OuroLastExitedKV
from phase04_runtime.data_contract import audit_splits, effective_token_hashes
from phase04_runtime.identity import HuginnStateIdentity, state_sha256
from phase04_runtime.policy import huginn_decision, ouro_decision


class Phase04ContractTests(unittest.TestCase):
    def test_huginn_rowwise_strict_tie_and_cap(self):
        before = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]])
        after = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]], [[1.0, 0.0]]])
        first = huginn_decision(before, after, 1, 2, 0.5)
        self.assertEqual(first.mask.tolist(), [True, False, False])
        self.assertEqual(first.score.tolist(), [0.0, 0.5, 1.0])
        cap = huginn_decision(before, after, 2, 2, 0.5)
        self.assertEqual(cap.mask.tolist(), [True, True, True])
        with self.assertRaises(ValueError):
            huginn_decision(before, torch.full_like(after, float("nan")), 1, 2, 0.5)

    def test_ouro_gate_cumulative_once_reset_per_token_and_permutation(self):
        gate = torch.tensor([0.0, 1.0, -1.0])
        remaining = torch.ones(3)
        first, after_first = ouro_decision(gate, remaining, 1, 4, 0.6)
        self.assertEqual(first.mask.tolist(), [False, True, False])
        second, after_second = ouro_decision(gate, after_first, 2, 4, 0.6)
        self.assertEqual(second.mask.tolist(), [True, True, False])
        self.assertTrue(torch.allclose(after_second, (1 - torch.sigmoid(gate)) ** 2))
        order = torch.tensor([2, 0, 1])
        permuted, _ = ouro_decision(gate[order], remaining[order], 1, 4, 0.6)
        self.assertEqual(permuted.mask.tolist(), first.mask[order].tolist())
        restarted, _ = ouro_decision(gate, torch.ones(3), 1, 4, 0.6)
        self.assertEqual(restarted.mask.tolist(), first.mask.tolist())
        cap, _ = ouro_decision(gate, remaining, 4, 4, 0.6)
        self.assertEqual(cap.mask.tolist(), [True, True, True])

    def test_request_keyed_h0_is_batch_and_padding_invariant(self):
        identity = HuginnStateIdentity(seed=9, standard_deviation=0.01, embedding_scale=4.0)
        ids = torch.tensor([17, 23, 5])
        positions = torch.tensor([[7], [11], [3]])
        joint = identity.decode(ids, positions, 32, torch.float32, torch.device("cpu"))
        order = torch.tensor([2, 0, 1])
        shuffled = identity.decode(ids[order], positions[order], 32,
                                   torch.float32, torch.device("cpu"))
        self.assertTrue(torch.equal(joint[order], shuffled))
        self.assertNotEqual(state_sha256(joint[0]), state_sha256(joint[1]))
        valid = torch.tensor([[False, False, True, True], [True, True, True, True]])
        padded = identity.prefill(ids[:2], valid, 32, torch.float32, torch.device("cpu"))
        solo = identity.prefill(ids[:1], torch.tensor([[True, True]]), 32,
                                torch.float32, torch.device("cpu"))
        self.assertTrue(torch.equal(padded[0, 2:], solo[0]))

    def test_effective_token_hash_catches_truncated_duplicate_and_family(self):
        tokenized = torch.tensor([[0, 2, 3], [8, 2, 3], [0, 4, 5]])
        mask = torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        hashes = effective_token_hashes(tokenized, mask)
        self.assertEqual(hashes[0], hashes[1])
        with self.assertRaisesRegex(ValueError, "effective model input"):
            audit_splits(hashes, ["a", "b", "c"], {"train": [0], "validation": [1], "test": [2]})
        with self.assertRaisesRegex(ValueError, "family crosses"):
            audit_splits(["a", "b", "c"], ["same", "same", "c"],
                         {"train": [0], "validation": [1], "test": [2]})

    def test_kv_benchmark_has_no_events_or_unbounded_receipts(self):
        backend = OuroLastExitedKV(1, torch.device("cpu"), 1, 2, instrumentation="benchmark")
        backend.begin_prefill(torch.ones((1, 1), dtype=torch.bool))
        for depth in (1, 2):
            value = torch.ones((1, 1, 1, 1))
            backend.update(value, value, depth - 1)
        backend.finish_prefill()
        backend.begin_token()
        backend.set_active(torch.tensor([0]))
        backend.update(torch.ones((1, 1, 1, 1)), torch.ones((1, 1, 1, 1)), 0)
        backend.on_exit(torch.tensor([0]), 1)
        backend.commit_token()
        self.assertEqual(backend.pop_update_gpu_ms(), 0.0)
        self.assertEqual(backend.fill_receipts, [])
        self.assertEqual(backend.fill_operations, 1)


if __name__ == "__main__":
    unittest.main()
