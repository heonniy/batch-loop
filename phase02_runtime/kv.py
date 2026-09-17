"""Stable logical-row KV backends for physically compacted recurrent compute.

Attention sees gathered active rows; persistent cache rows are never renumbered
when the active tensor shrinks. Growth of history occurs at token boundaries,
not as a side effect of survivor compaction.
"""

from __future__ import annotations

import torch


KV = tuple[torch.Tensor, torch.Tensor]


class _StableRowKV:
    def __init__(self, num_rows: int, device: torch.device,
                 *, instrumentation: str = "profile"):
        if instrumentation not in {"profile", "benchmark"}:
            raise ValueError("KV instrumentation must be profile or benchmark")
        self.num_rows = num_rows
        self.device = device
        self.instrumentation = instrumentation
        self.active_rows = torch.arange(num_rows, device=device, dtype=torch.long)
        self.key_valid: torch.Tensor | None = None
        self.phase = "new"
        self._update_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._update_gpu_ms_accum = 0.0
        self.actual_writes = 0
        self.fill_operations = 0
        self.overwrites = 0

    def set_active(self, rows: torch.Tensor) -> None:
        if rows.dtype != torch.long or rows.ndim != 1:
            raise ValueError("active KV rows must be a one-dimensional LongTensor")
        if self.instrumentation == "profile" and rows.numel() and (
                int(rows.min()) < 0 or int(rows.max()) >= self.num_rows):
            raise ValueError("active KV row outside persistent cache")
        self.active_rows = rows

    def _event_start(self):
        if self.device.type != "cuda" or self.instrumentation != "profile":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _event_end(self, start):
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._update_events.append((start, end))
            if len(self._update_events) >= 256:
                self._drain_update_events()

    def _drain_update_events(self) -> None:
        # Profile events are asynchronous. Their durations are only available
        # after completion; benchmark mode never creates these events.
        for _, end in self._update_events:
            end.synchronize()
        self._update_gpu_ms_accum += sum(start.elapsed_time(end) for start, end in self._update_events)
        self._update_events.clear()

    def pop_update_gpu_ms(self) -> float:
        self._drain_update_events()
        total = self._update_gpu_ms_accum
        self._update_gpu_ms_accum = 0.0
        return total

    def begin_prefill(self, prompt_valid: torch.Tensor) -> None:
        if prompt_valid.ndim != 2 or prompt_valid.shape[0] != self.num_rows:
            raise ValueError("prompt-valid mask must have stable global rows")
        self.phase = "prefill"
        self.key_valid = prompt_valid.bool().to(self.device)
        self.set_active(torch.arange(self.num_rows, device=self.device))

    def begin_token(self) -> None:
        if self.phase != "ready" or self.key_valid is None:
            raise RuntimeError("finish prefill/previous token before next token")
        self.phase = "token"
        self.key_valid = torch.cat(
            [self.key_valid, torch.ones((self.num_rows, 1), dtype=torch.bool, device=self.device)], dim=1
        )

    def attention_bias(self, q_len: int, k_len: int, dtype: torch.dtype) -> torch.Tensor:
        """Additive causal/key-padding mask for the active rows, including pad self-loops."""
        if self.key_valid is None or self.key_valid.shape[1] != k_len:
            raise AssertionError("key-mask/cache length mismatch")
        valid = self.key_valid.index_select(0, self.active_rows)
        keys = torch.arange(k_len, device=self.device)
        queries = torch.arange(k_len - q_len, k_len, device=self.device)
        causal = keys.unsqueeze(0) <= queries.unsqueeze(1)
        allowed = valid[:, None, :] & causal[None, :, :]
        query_valid = valid.index_select(1, queries)
        # Padding queries have no semantic output, but need one allowed key to
        # avoid all-masked softmax/NaN. Real queries cannot attend to pad keys.
        for q in range(q_len):
            allowed[:, q, queries[q]] |= ~query_valid[:, q]
        bias = torch.zeros(allowed.shape, dtype=dtype, device=self.device)
        bias.masked_fill_(~allowed, torch.finfo(dtype).min)
        return bias.unsqueeze(1)


class OuroLastExitedKV(_StableRowKV):
    """Depth-indexed KV; skipped slots inherit the last actual recurrence KV."""

    def __init__(self, num_rows: int, device: torch.device, num_layers: int, rmax: int = 4,
                 *, instrumentation: str = "profile"):
        super().__init__(num_rows, device, instrumentation=instrumentation)
        self.num_layers = num_layers
        self.rmax = rmax
        self.history: dict[tuple[int, int], KV] = {}
        self.current: dict[tuple[int, int], KV] = {}
        self.valid_current: dict[tuple[int, int], torch.Tensor] = {}
        self.fill_receipts: list[dict] = []
        self.fill_receipts_truncated = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               flat_layer_idx: int, cache_kwargs: dict | None = None) -> KV:
        del cache_kwargs
        depth, layer = divmod(int(flat_layer_idx), self.num_layers)
        depth += 1
        if not (1 <= depth <= self.rmax):
            raise ValueError("Ouro recurrence slot outside Rmax")
        slot = (depth, layer)
        if key_states.shape[0] != self.active_rows.numel():
            raise AssertionError("actual recurrent tensor width != active KV rows")
        start = self._event_start()
        if self.phase == "prefill":
            self.history[slot] = (key_states, value_states)
            output = (key_states, value_states)
        elif self.phase == "token":
            if slot not in self.history:
                raise AssertionError(f"missing Ouro prefill cache slot {slot}")
            if slot not in self.current:
                shape = (self.num_rows, *key_states.shape[1:])
                self.current[slot] = (
                    torch.empty(shape, dtype=key_states.dtype, device=self.device),
                    torch.empty(shape, dtype=value_states.dtype, device=self.device),
                )
                self.valid_current[slot] = torch.zeros(self.num_rows, dtype=torch.bool, device=self.device)
            current_k, current_v = self.current[slot]
            current_k.index_copy_(0, self.active_rows, key_states)
            current_v.index_copy_(0, self.active_rows, value_states)
            self.valid_current[slot].index_fill_(0, self.active_rows, True)
            history_k, history_v = self.history[slot]
            output = (
                torch.cat([history_k.index_select(0, self.active_rows), key_states], dim=2),
                torch.cat([history_v.index_select(0, self.active_rows), value_states], dim=2),
            )
        else:
            raise RuntimeError("Ouro KV update outside prefill/token")
        self.actual_writes += int(key_states.shape[0])
        self._event_end(start)
        return output

    def finish_prefill(self) -> None:
        if len(self.history) != self.rmax * self.num_layers:
            raise AssertionError("Ouro prefill did not populate all depth/layer slots")
        self.pop_update_gpu_ms()  # Prefill events cannot be attributed to decode.
        self.phase = "ready"

    def begin_token(self) -> None:
        super().begin_token()
        self.current.clear()
        self.valid_current.clear()

    def on_exit(self, rows: torch.Tensor, depth: int) -> None:
        if self.phase != "token" or rows.numel() == 0:
            return
        for next_depth in range(depth + 1, self.rmax + 1):
            for layer in range(self.num_layers):
                source = self.current[(depth, layer)]
                target_slot = (next_depth, layer)
                if target_slot not in self.current:
                    self.current[target_slot] = (torch.empty_like(source[0]), torch.empty_like(source[1]))
                    self.valid_current[target_slot] = torch.zeros(
                        self.num_rows, dtype=torch.bool, device=self.device
                    )
                target = self.current[target_slot]
                target[0].index_copy_(0, rows, source[0].index_select(0, rows))
                target[1].index_copy_(0, rows, source[1].index_select(0, rows))
                self.valid_current[target_slot].index_fill_(0, rows, True)
                self.fill_operations += int(rows.numel())
                if self.instrumentation == "profile":
                    if len(self.fill_receipts) < 4096:
                        self.fill_receipts.append({
                            "rows": rows.detach().cpu().tolist(), "from_depth": depth,
                            "to_depth": next_depth, "physical_layer": layer,
                        })
                    else:
                        self.fill_receipts_truncated += 1

    def commit_token(self) -> None:
        if self.phase != "token":
            raise RuntimeError("no current Ouro token to commit")
        for slot in self.history:
            if slot not in self.current or not bool(self.valid_current[slot].all()):
                raise AssertionError(f"incomplete last-exited Ouro KV slot {slot}")
            old_k, old_v = self.history[slot]
            new_k, new_v = self.current[slot]
            self.history[slot] = (torch.cat([old_k, new_k], dim=2), torch.cat([old_v, new_v], dim=2))
        self.current.clear()
        self.valid_current.clear()
        self.phase = "ready"


class HuginnSharedKV(_StableRowKV):
    """One recurrent slot per *physical* layer and logical request."""

    def __init__(self, num_rows: int, device: torch.device, n_prelude: int = 2,
                 n_recurrent: int = 4, n_coda: int = 2,
                 *, instrumentation: str = "profile"):
        super().__init__(num_rows, device, instrumentation=instrumentation)
        self.n_prelude = n_prelude
        self.n_recurrent = n_recurrent
        self.n_coda = n_coda
        self.history: dict[tuple[str, int], KV] = {}
        self.current: dict[tuple[str, int], KV] = {}
        self.valid_current: dict[tuple[str, int], torch.Tensor] = {}
        self.history_length = 0
        self.history_capacity = 0

    def _slot(self, block_idx: int | torch.Tensor) -> tuple[str, int]:
        idx = int(block_idx)
        if idx < 0:
            if -idx > self.n_coda:
                raise ValueError("coda block outside configured range")
            return ("coda", -idx - 1)
        if idx < self.n_prelude:
            return ("prelude", idx)
        return ("core", (idx - self.n_prelude) % self.n_recurrent)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               block_idx: int | torch.Tensor) -> KV:
        slot = self._slot(block_idx)
        if key_states.shape[0] != self.active_rows.numel():
            raise AssertionError("actual Huginn tensor width != active KV rows")
        start = self._event_start()
        if slot not in self.current:
            shape = (self.num_rows, *key_states.shape[1:])
            self.current[slot] = (
                torch.empty(shape, dtype=key_states.dtype, device=self.device),
                torch.empty(shape, dtype=value_states.dtype, device=self.device),
            )
            self.valid_current[slot] = torch.zeros(self.num_rows, dtype=torch.bool, device=self.device)
        else:
            self.overwrites += int(self.active_rows.numel())
        current_k, current_v = self.current[slot]
        current_k.index_copy_(0, self.active_rows, key_states)
        current_v.index_copy_(0, self.active_rows, value_states)
        self.valid_current[slot].index_fill_(0, self.active_rows, True)
        if self.phase == "prefill":
            output = (key_states, value_states)
        elif self.phase == "token":
            history_k, history_v = self.history[slot]
            output = (
                torch.cat([history_k[:, :, :self.history_length].index_select(0, self.active_rows), key_states], dim=2),
                torch.cat([history_v[:, :, :self.history_length].index_select(0, self.active_rows), value_states], dim=2),
            )
        else:
            raise RuntimeError("Huginn KV update outside prefill/token")
        self.actual_writes += int(key_states.shape[0])
        self._event_end(start)
        return output

    def finish_prefill(self) -> None:
        expected = self.n_prelude + self.n_recurrent + self.n_coda
        if len(self.current) != expected:
            raise AssertionError("Huginn prefill did not populate all physical layers")
        if any(not bool(mask.all()) for mask in self.valid_current.values()):
            raise AssertionError("incomplete Huginn prefill KV")
        self.history = self.current
        lengths = {k.shape[2] for k, _ in self.history.values()}
        if len(lengths) != 1:
            raise AssertionError("Huginn prefill layer lengths differ")
        self.history_length = self.history_capacity = lengths.pop()
        self.pop_update_gpu_ms()  # Exclude prefill/Prelude from later decode attribution.
        self.current = {}
        self.valid_current = {}
        self.phase = "ready"

    def _reserve_history(self, required_length: int) -> None:
        if required_length <= self.history_capacity:
            return
        new_capacity = max(required_length, self.history_capacity * 2)
        for slot, (old_k, old_v) in tuple(self.history.items()):
            shape = (self.num_rows, old_k.shape[1], new_capacity, old_k.shape[3])
            new_k = torch.empty(shape, dtype=old_k.dtype, device=self.device)
            new_v = torch.empty(shape, dtype=old_v.dtype, device=self.device)
            new_k[:, :, :self.history_length].copy_(old_k[:, :, :self.history_length])
            new_v[:, :, :self.history_length].copy_(old_v[:, :, :self.history_length])
            self.history[slot] = (new_k, new_v)
        self.history_capacity = new_capacity

    def begin_token(self) -> None:
        super().begin_token()
        self._reserve_history(self.history_length + 1)
        self.current.clear()
        self.valid_current.clear()

    def on_exit(self, rows: torch.Tensor, depth: int) -> None:
        del rows, depth
        # Last actual recurrent write stays in the same physical-layer slot.

    def commit_token(self) -> None:
        if self.phase != "token":
            raise RuntimeError("no current Huginn token to commit")
        for slot in self.history:
            if slot not in self.current or not bool(self.valid_current[slot].all()):
                raise AssertionError(f"incomplete shared Huginn KV slot {slot}")
            history_k, history_v = self.history[slot]
            new_k, new_v = self.current[slot]
            history_k[:, :, self.history_length:self.history_length + 1].copy_(new_k)
            history_v[:, :, self.history_length:self.history_length + 1].copy_(new_v)
        self.history_length += 1
        self.current.clear()
        self.valid_current.clear()
        self.phase = "ready"
