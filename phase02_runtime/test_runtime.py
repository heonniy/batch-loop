"""CPU-only invariant checks before any Phase-2 GPU execution."""

from __future__ import annotations

import unittest

import torch

from phase02_runtime.common import run_interval
from phase02_runtime.kv import HuginnSharedKV, OuroLastExitedKV


class _FakeBackend:
    def __init__(self):
        self.rows = None
        self.active_history = []

    def set_active(self, rows):
        self.rows = rows
        self.active_history.append(rows.tolist())

    def pop_update_gpu_ms(self):
        return 0.0

    def on_exit(self, rows, depth):
        del rows, depth


class _FakeAdapter:
    rmax = 4

    def __init__(self):
        self.backend = _FakeBackend()

    def recurrent_step(self, state, embedded, recurrence):
        del embedded, recurrence
        return state + 1

    def live_exit(self, before, after, rows, recurrence, threshold):
        del before, after, rows, recurrence, threshold
        raise AssertionError("forced smoke must not call live gate")


class SurvivorTests(unittest.TestCase):
    def test_forced_compaction_and_scatter(self):
        adapter = _FakeAdapter()
        state = torch.zeros((4, 1, 2))
        result = run_interval(
            adapter, state, state, torch.tensor([3, 1, 0, 2]),
            torch.tensor([10, 11, 12, 13]), exit_source="trace",
            forced_depths=torch.tensor([1, 2, 4, 2]),
        )
        self.assertEqual([x["active_batch_size"] for x in result.events], [4, 3, 1, 1])
        self.assertEqual(adapter.backend.active_history, [[3, 1, 0, 2], [1, 0, 2], [0], [0]])
        self.assertEqual(result.final_state[:, 0, 0].tolist(), [1, 2, 4, 2])
        self.assertEqual(result.exit_depths.tolist(), [1, 2, 4, 2])

    def test_ouro_last_exited_fills_only_own_rows(self):
        backend = OuroLastExitedKV(4, torch.device("cpu"), num_layers=2, rmax=4)
        backend.begin_prefill(torch.ones((4, 2), dtype=torch.bool))
        for depth in range(1, 5):
            for layer in range(2):
                values = torch.full((4, 1, 2, 1), float(depth * 10 + layer))
                backend.update(values, values, (depth - 1) * 2 + layer)
        backend.finish_prefill()
        backend.begin_token()
        groups = [([0, 1, 2, 3], [0]), ([1, 2, 3], [1, 3]), ([2], []), ([2], [2])]
        for depth, (active, exits) in enumerate(groups, 1):
            backend.set_active(torch.tensor(active))
            for layer in range(2):
                values = torch.tensor(active, dtype=torch.float32).view(-1, 1, 1, 1) + depth * 10 + layer
                backend.update(values, values, (depth - 1) * 2 + layer)
            backend.on_exit(torch.tensor(exits, dtype=torch.long), depth)
        backend.commit_token()
        for depth in range(1, 5):
            for layer in range(2):
                k, v = backend.history[(depth, layer)]
                self.assertTrue(torch.equal(k, v))
                expected = {
                    1: [10, 11, 12, 13], 2: [10, 21, 22, 23],
                    3: [10, 21, 32, 23], 4: [10, 21, 42, 23],
                }[depth]
                self.assertEqual(k[:, 0, -1, 0].tolist(), [x + layer for x in expected])
        self.assertEqual(backend.fill_operations, (3 + 2 + 2) * 2)

    def test_huginn_shared_slots_have_no_depth_axis(self):
        backend = HuginnSharedKV(4, torch.device("cpu"))
        backend.begin_prefill(torch.ones((4, 2), dtype=torch.bool))
        for idx in (0, 1, 2, 3, 4, 5, -1, -2):
            values = torch.ones((4, 1, 2, 1))
            backend.update(values, values, idx)
        backend.finish_prefill()
        backend.begin_token()
        backend.set_active(torch.arange(4))
        for idx in (0, 1):
            values = torch.ones((4, 1, 1, 1))
            backend.update(values, values, idx)
        for depth in range(1, 5):
            rows = torch.tensor([0, 1, 2, 3] if depth == 1 else [1, 2, 3] if depth == 2 else [2])
            backend.set_active(rows)
            for layer in range(4):
                values = torch.full((len(rows), 1, 1, 1), float(depth))
                backend.update(values, values, 2 + (depth - 1) * 4 + layer)
            backend.on_exit(rows[:1], depth)
        backend.set_active(torch.arange(4))
        for idx in (-1, -2):
            values = torch.ones((4, 1, 1, 1))
            backend.update(values, values, idx)
        backend.commit_token()
        self.assertEqual(len(backend.history), 8)
        self.assertEqual(backend.history[("core", 0)][0][:, 0, -1, 0].tolist(), [1, 2, 4, 2])


if __name__ == "__main__":
    unittest.main()
