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
    def test_profile_event_drain_waits_for_completion(self):
        backend = OuroLastExitedKV(1, torch.device("cpu"), num_layers=1)

        class EndEvent:
            ready = False

            def synchronize(self):
                self.ready = True

        class StartEvent:
            def elapsed_time(self, end):
                if not end.ready:
                    raise RuntimeError("device not ready")
                return 1.25

        end = EndEvent()
        backend._update_events.append((StartEvent(), end))
        self.assertEqual(backend.pop_update_gpu_ms(), 1.25)
        self.assertTrue(end.ready)
        self.assertEqual(backend.pop_update_gpu_ms(), 0.0)

    def test_forced_compaction_and_scatter(self):
        adapter = _FakeAdapter()
        state = torch.zeros((4, 1, 2))
        entered = []
        result = run_interval(
            adapter, state, state, torch.tensor([3, 1, 0, 2]),
            torch.tensor([10, 11, 12, 13]), exit_source="trace",
            forced_depths=torch.tensor([1, 2, 4, 2]),
            debug_pre_hook=lambda depth, requests, before, _: entered.append(
                (depth, requests.tolist(), before.shape[0])),
        )
        self.assertEqual([x["active_batch_size"] for x in result.events], [4, 3, 1, 1])
        self.assertEqual(adapter.backend.active_history, [[3, 1, 0, 2], [1, 0, 2], [0], [0]])
        self.assertEqual(result.final_state[:, 0, 0].tolist(), [1, 2, 4, 2])
        self.assertEqual(result.exit_depths.tolist(), [1, 2, 4, 2])
        self.assertEqual(entered, [(1, [10, 11, 12, 13], 4),
                                   (2, [11, 12, 13], 3), (3, [12], 1), (4, [12], 1)])

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
        self.assertEqual(backend.history[("core", 0)][0][:, 0, backend.history_length - 1, 0].tolist(), [1, 2, 4, 2])

    def test_token_boundary_grows_history_without_changing_stable_rows(self):
        ouro = OuroLastExitedKV(2, torch.device("cpu"), num_layers=1, rmax=2)
        huginn = HuginnSharedKV(2, torch.device("cpu"))
        valid = torch.tensor([[False, True], [True, True]])
        for backend in (ouro, huginn):
            backend.begin_prefill(valid)
            slots = (0, 1) if backend is ouro else (0, 1, 2, 3, 4, 5, -1, -2)
            for slot in slots:
                tensor = torch.ones((2, 1, 2, 1))
                backend.update(tensor, tensor, slot)
            backend.finish_prefill()
            reserved_ptrs = None
            for token_idx in (1, 2):
                backend.begin_token()
                if backend is huginn and token_idx == 2:
                    self.assertEqual(
                        reserved_ptrs,
                        {slot: values[0].data_ptr() for slot, values in backend.history.items()},
                    )
                backend.set_active(torch.tensor([1, 0]))
                for slot in slots:
                    tensor = torch.tensor([token_idx + 10, token_idx + 20], dtype=torch.float32).view(2, 1, 1, 1)
                    visible_k, _ = backend.update(tensor, tensor, slot)
                    self.assertEqual(visible_k.shape[2], 2 + token_idx)
                backend.commit_token()
                if backend is huginn and token_idx == 1:
                    reserved_ptrs = {slot: values[0].data_ptr() for slot, values in backend.history.items()}
            self.assertEqual(backend.key_valid.shape, (2, 4))
            for key, (k, v) in backend.history.items():
                self.assertGreaterEqual(k.shape[2], backend.key_valid.shape[1], key)
                self.assertTrue(torch.equal(k, v), key)
                logical_last = backend.history_length - 1 if backend is huginn else -1
                self.assertEqual(k[:, 0, logical_last, 0].tolist(), [22.0, 12.0], key)


if __name__ == "__main__":
    unittest.main()
