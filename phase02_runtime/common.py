"""Stable-row survivor compaction shared by the Ouro and Huginn adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import torch


T = TypeVar("T")


def timed(device: torch.device, fn: Callable[[], T],
          *, instrumentation: str = "profile") -> tuple[T, float, float]:
    """Return value, GPU milliseconds (profile only), and CPU dispatch wall ms."""
    if instrumentation not in {"profile", "benchmark"}:
        raise ValueError("timing instrumentation must be profile or benchmark")
    if instrumentation == "benchmark":
        start = time.perf_counter()
        value = fn()
        return value, 0.0, (time.perf_counter() - start) * 1000
    if device.type != "cuda":
        start = time.perf_counter()
        value = fn()
        elapsed = (time.perf_counter() - start) * 1000
        return value, 0.0, elapsed
    torch.cuda.synchronize(device)
    start_wall = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end), (time.perf_counter() - start_wall) * 1000


class SurvivorAdapter(Protocol):
    rmax: int
    backend: object

    def recurrent_step(self, state: torch.Tensor, embedded: torch.Tensor, recurrence: int) -> torch.Tensor: ...

    def live_exit(self, before: torch.Tensor, after: torch.Tensor, rows: torch.Tensor,
                  recurrence: int, threshold: float) -> torch.Tensor: ...


@dataclass
class IntervalResult:
    final_state: torch.Tensor
    exit_depths: torch.Tensor
    events: list[dict]
    core_gpu_ms: float
    gate_gpu_ms: float
    compaction_gpu_ms: float
    kv_gpu_ms: float
    kv_exit_gpu_ms: float
    wall_ms: float


@torch.inference_mode()
def run_interval(
    adapter: SurvivorAdapter,
    embedded: torch.Tensor,
    initial_state: torch.Tensor,
    global_rows: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    exit_source: str,
    forced_depths: torch.Tensor | None = None,
    threshold: float = 0.2,
    debug_ids: bool = True,
    trajectory_sink: dict[int, list[float]] | None = None,
    trajectory_kind: str | None = None,
    instrumentation: str = "profile",
    debug_hook: Callable[[int, torch.Tensor, torch.Tensor, SurvivorAdapter], None] | None = None,
    debug_pre_hook: Callable[[int, torch.Tensor, torch.Tensor, SurvivorAdapter], None] | None = None,
) -> IntervalResult:
    """Execute one recurrent token interval with a genuinely shrinking tensor.

    ``global_rows`` are persistent cache row IDs; local tensor positions are
    allowed to change after every step. The returned state is in original local
    order, independently of exit order.
    """
    if exit_source not in {"trace", "live"}:
        raise ValueError(f"unsupported exit source: {exit_source}")
    if instrumentation not in {"profile", "benchmark"}:
        raise ValueError("instrumentation must be profile or benchmark")
    if trajectory_sink is not None and trajectory_kind not in {"huginn_convergence", "ouro_gate"}:
        raise ValueError("trajectory sink requires a model-specific causal kind")
    if embedded.shape != initial_state.shape or embedded.ndim != 3:
        raise ValueError("embedded and initial_state must have equal [B,S,H] shape")
    n = initial_state.shape[0]
    if n < 1 or global_rows.shape != (n,) or request_ids.shape != (n,):
        raise ValueError("stable row and request ID vectors must match B")
    if instrumentation == "profile" and (
            torch.unique(global_rows).numel() != n or torch.unique(request_ids).numel() != n):
        raise ValueError("duplicate persistent row or request ID")
    if exit_source == "trace":
        if forced_depths is None or forced_depths.shape != (n,):
            raise ValueError("trace mode requires one depth per local request")
        if not bool(((forced_depths >= 1) & (forced_depths <= adapter.rmax)).all()):
            raise ValueError("forced depth outside 1..Rmax")

    device = initial_state.device
    local = torch.arange(n, device=device)
    active_rows = global_rows
    active_requests = request_ids
    active_state = initial_state
    active_embedded = embedded
    final = torch.empty_like(initial_state)
    final_depth = torch.zeros(n, dtype=torch.long, device=device)
    seen = torch.zeros(n, dtype=torch.bool, device=device)
    events: list[dict] = []
    total_core = total_gate = total_compaction = total_kv = total_exit_kv = 0.0
    start_interval = time.perf_counter()

    for recurrence in range(1, adapter.rmax + 1):
        if local.numel() == 0:
            break
        if instrumentation == "profile" and bool(seen.index_select(0, local).any()):
            raise AssertionError("exited request reached a later recurrent step")
        assert active_state.shape[0] == local.numel()
        adapter.backend.set_active(active_rows)
        before = active_state
        if debug_pre_hook is not None:
            debug_pre_hook(recurrence, active_requests, before, adapter)
        after, core_ms, core_wall_ms = timed(
            device, lambda: adapter.recurrent_step(before, active_embedded, recurrence),
            instrumentation=instrumentation,
        )
        if after.shape != before.shape:
            raise AssertionError("recurrent step changed non-batch tensor shape")
        if debug_hook is not None:
            debug_hook(recurrence, active_requests, after, adapter)
        if trajectory_sink is not None:
            if trajectory_kind == "huginn_convergence":
                values = ((after.float() - before.float()).norm(dim=-1) /
                          after.float().norm(dim=-1).clamp_min(1e-12)).mean(dim=-1)
            else:
                values = torch.sigmoid(adapter.model.model.early_exit_gate(after).squeeze(-1).squeeze(-1)).float()
            for request_id, value in zip(active_requests.detach().cpu().tolist(),
                                         values.detach().cpu().tolist()):
                trajectory_sink.setdefault(int(request_id), []).append(float(value))
        kv_ms = adapter.backend.pop_update_gpu_ms()
        if exit_source == "trace":
            exit_mask = forced_depths.index_select(0, local) == recurrence
            gate_ms = gate_wall_ms = 0.0
        else:
            exit_mask, gate_ms, gate_wall_ms = timed(
                device, lambda: adapter.live_exit(before, after, active_rows, recurrence, threshold),
                instrumentation=instrumentation,
            )
        if recurrence == adapter.rmax:
            exit_mask = torch.ones_like(exit_mask, dtype=torch.bool)
        if exit_mask.shape != (local.numel(),) or exit_mask.dtype != torch.bool:
            raise AssertionError("exit mask must be one bool per active row")
        exiting_rows = active_rows[exit_mask]
        _, exit_kv_ms, exit_kv_wall_ms = timed(
            device, lambda: adapter.backend.on_exit(exiting_rows, recurrence),
            instrumentation=instrumentation,
        )

        def scatter_and_compact():
            nonlocal final, final_depth, seen
            exiting_local = local[exit_mask]
            final.index_copy_(0, exiting_local, after[exit_mask])
            final_depth.index_fill_(0, exiting_local, recurrence)
            seen.index_fill_(0, exiting_local, True)
            keep = torch.nonzero(~exit_mask, as_tuple=False).squeeze(-1)
            return (
                local.index_select(0, keep),
                active_rows.index_select(0, keep),
                active_requests.index_select(0, keep),
                after.index_select(0, keep),
                active_embedded.index_select(0, keep),
            )

        nxt, compaction_ms, compaction_wall_ms = timed(
            device, scatter_and_compact, instrumentation=instrumentation)
        event = {
            "recurrence": recurrence,
            "active_batch_size": int(local.numel()),
            "active_request_ids": active_requests.detach().cpu().tolist() if debug_ids else None,
            "active_kv_rows": active_rows.detach().cpu().tolist() if debug_ids else None,
            "exited_request_ids": active_requests[exit_mask].detach().cpu().tolist() if debug_ids else None,
            "recurrent_pass_gpu_ms_including_kv": core_ms,
            "recurrent_pass_wall_ms": core_wall_ms,
            "kv_update_gpu_ms_within_pass": kv_ms,
            "gate_gpu_ms": gate_ms,
            "gate_wall_ms": gate_wall_ms,
            "exit_kv_gpu_ms": exit_kv_ms,
            "exit_kv_wall_ms": exit_kv_wall_ms,
            "compaction_gpu_ms": compaction_ms,
            "compaction_wall_ms": compaction_wall_ms,
        }
        if debug_ids and exit_source == "live" and getattr(adapter, "last_exit_decision", None) is not None:
            decision = adapter.last_exit_decision
            event["exit_score"] = decision.score.detach().cpu().tolist()
            event["exit_margin"] = decision.margin.detach().cpu().tolist()
        events.append(event)
        total_core += core_ms
        total_gate += gate_ms
        total_kv += kv_ms
        total_exit_kv += exit_kv_ms
        total_compaction += compaction_ms
        local, active_rows, active_requests, active_state, active_embedded = nxt

    if not bool(seen.all()) or bool((final_depth == 0).any()):
        raise AssertionError("every request must exit exactly once")
    if exit_source == "trace" and not torch.equal(final_depth, forced_depths):
        raise AssertionError("forced exit depth mismatch")
    widths = [row["active_batch_size"] for row in events]
    if widths != sorted(widths, reverse=True):
        raise AssertionError("survivor batch grew inside recurrent interval")
    if instrumentation == "benchmark" and device.type == "cuda":
        torch.cuda.synchronize(device)
    return IntervalResult(
        final_state=final, exit_depths=final_depth, events=events,
        core_gpu_ms=total_core, gate_gpu_ms=total_gate,
        compaction_gpu_ms=total_compaction, kv_gpu_ms=total_kv,
        kv_exit_gpu_ms=total_exit_kv,
        wall_ms=(time.perf_counter() - start_interval) * 1000,
    )
