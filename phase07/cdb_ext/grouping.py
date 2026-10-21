"""[Copied from Phase-06 phase06/cdb_ext/grouping.py @ local commit 3e2779e (branch research/phase06-optimized-cdb-20261019, not pushed); extended with the `predicted` policy for Phase 07.]

Cohort-selection extension point for the upstream CDB no-refill wave.

Derived from ``ContinuousDepthBatchingEngine._run_no_refill_decode_wave`` of
kschwethelm/continuous-depth-batching @ abdfae0adbdd12be8c6123ccb34eee98a38afb39
(MIT; portions Apache-2.0, see the upstream THIRD_PARTY_NOTICES.md). The upstream
source tree is not modified: :func:`install` replaces the method at runtime.

Change versus upstream: after the wave's single Prelude launch, the wave's items (one
token per resident decoding request) are partitioned by a *selector* into cohorts of at
most ``width``; each cohort then runs its own shrink-only recurrent loop (no refill, no
merging across cohorts), and every exited token of the wave goes to the single
wave-sized Coda exactly as upstream. With ``width >= len(items)`` and the identity
selector this is the upstream wave. Kernels, attention, KV policy and CUDA graphs are
untouched; the Core launch width is the only thing a policy changes.

Policies (all evidence is PHYSICAL_FORCED_DEPTH: depths come from a frozen replay):
  fifo          stable request index
  depth_oracle  sort by the item's *current-token* replay exit step, then request index
  random        seeded shuffle per wave (equal-work control)
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

import torch

POLICIES = ("fifo", "depth_oracle", "random", "predicted")


def _rid(item) -> int:
    m = re.findall(r"\d+", str(item.state.request_id))
    return int(m[-1]) if m else 0


def select(policy: str, items: list, width: int, seed: int) -> list[list]:
    if policy == "fifo":
        order = sorted(items, key=_rid)
    elif policy == "depth_oracle":
        order = sorted(items, key=lambda it: (it.synthetic_exit_depth, _rid(it)))
    elif policy == "predicted":  # ascending predictor score of the CURRENT token (cursor-1), then request id
        order = sorted(items, key=lambda it: (STATE.scores[_rid(it)][it.state._synthetic_exit_cursor - 1], _rid(it)))
    elif policy == "random":
        order = sorted(items, key=_rid)
        random.Random(seed).shuffle(order)
    else:
        raise ValueError(policy)
    return [order[i:i + width] for i in range(0, len(order), width)]


@dataclass
class GroupingState:
    policy: str = "fifo"
    width: int = 10**9
    seed: int = 0
    wave: int = 0
    first_wave_t: float | None = None
    waves: list = field(default_factory=list)       # per wave: list of cohorts [(rid, cursor, exit_step)]
    widths: list = field(default_factory=list)      # per recurrent launch: (wave, cohort, step, width)
    events: dict = field(default_factory=lambda: {"prelude": [], "recurrent": [], "coda": []})
    scores: object = None  # [N][T] predictor scores for the `predicted` policy (precomputed, frozen)

    def reset(self, policy: str, width: int, seed: int, scores=None) -> None:
        self.__init__(policy=policy, width=width, seed=seed, scores=scores)

    def gpu_ms(self) -> dict:
        torch.cuda.synchronize()
        return {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.events.items()}


STATE = GroupingState()


def _timed_stage(runner, name: str, fn):
    stream = runner.inputs_and_outputs.compute_stream or torch.cuda.current_stream()

    def wrapper(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream.record_event(start)
        out = fn(*args, **kwargs)
        stream.record_event(end)
        STATE.events[name].append((start, end))
        return out
    return wrapper


def instrument_runner(engine) -> None:
    """CUDA events on the compute stream around the Prelude, Core and Coda stage launches."""
    r = engine.runner
    if getattr(r, "_phase06_instrumented", False):
        return
    r.compute_prelude_batch = _timed_stage(r, "prelude", r.compute_prelude_batch)
    r.compute_recurrent_batch = _timed_stage(r, "recurrent", r.compute_recurrent_batch)
    r.stage_coda_batch = _timed_stage(r, "coda", r.stage_coda_batch)
    r._phase06_instrumented = True


def install() -> None:
    from looped_cdb.benchmarks import nvtx
    from looped_cdb.continuous_depth_batching import continuous_api as api
    from looped_cdb.continuous_depth_batching.requests import RequestStatus

    CarriedPreludeEntries = api.CarriedPreludeEntries

    def _run_no_refill_decode_wave(self, recurrent_kwargs, boundary) -> bool:
        if STATE.first_wave_t is None:
            STATE.first_wave_t = time.perf_counter()
        carried, carried_coda = boundary.take_decode_carry()
        carried = self._drop_cancelled_carried(carried)
        carried_items = [item for entries in carried for item in entries.items]
        in_flight_ids = {item.state.request_id for item in carried_coda.items} if carried_coda is not None else set()
        decode_cohort = [state for state in self._no_refill_decode_cohort() if state.request_id not in in_flight_ids]
        decode_cohort = decode_cohort[: max(0, self.cdb_config.max_num_seqs - len(carried_items))]
        new_items = [item for state in decode_cohort if (item := self._stage_decode_token(state)) is not None]
        items = carried_items + new_items
        if not items:
            if carried_coda is not None:
                self._consume_no_refill_coda(carried_coda)
                return True
            return False
        self._record_prelude("wave_cohort", len(items), gathered=len(carried_items))
        with nvtx.range("cdb.prelude.wave_cohort"):
            self.runner.compute_prelude_batch(
                items, device_groups=[(entries.result, entries.rows) for entries in carried]
            )
        self._trace_cohort(len(items))
        self._trace("prelude", len(items), site="wave_cohort")
        max_steps = self.cdb_config.max_recurrent_steps
        assert max_steps is not None

        delayed_synthetic = self._uses_delayed_synthetic_exit()
        delayed_gate = self._uses_delayed_gate_consumption(recurrent_kwargs)
        delayed = delayed_synthetic or delayed_gate
        stage_signal = self._uses_delayed_gate_readout(recurrent_kwargs)

        # ---- phase06 extension point: cohort selection before any recurrent launch ----
        cohorts = select(STATE.policy, items, STATE.width, STATE.seed * 100003 + STATE.wave)
        STATE.waves.append([[(_rid(it), it.state._synthetic_exit_cursor, it.synthetic_exit_depth) for it in c]
                            for c in cohorts])
        wave_index = STATE.wave
        STATE.wave += 1

        exited_wave = []
        for cohort_index, cohort in enumerate(cohorts):
            survivors = cohort
            for step in range(max_steps):
                for item in survivors:
                    item.recurrent_step = step
                STATE.widths.append((wave_index, cohort_index, step, len(survivors)))
                exit_signals = self._launch_no_refill_recurrent(survivors, recurrent_kwargs)
                self.last_stats.recurrent_step_histogram[step] += len(survivors)
                pending_signal_results = (
                    self.runner.stage_delayed_exit_signals(survivors, exit_signals) if stage_signal else None
                )
                if carried_coda is not None:
                    self._consume_no_refill_coda(carried_coda)
                    carried_coda = None
                forced = step == max_steps - 1
                exited, continuing, cancelled = [], [], []
                for batch_index, item in enumerate(survivors):
                    if item.state.status == RequestStatus.FINISHED:
                        cancelled.append(item)
                        continue
                    if delayed_synthetic:
                        should_exit = item.apply_exit_gate and self._consume_delayed_synthetic_exit(item)
                    elif delayed_gate:
                        should_exit = item.apply_exit_gate and self._consume_delayed_gate(item)
                    else:
                        should_exit = self._should_exit(item, step, exit_signals, batch_index)
                    if forced or should_exit:
                        self._record_exit(item, step)
                        exited.append(item)
                    else:
                        if delayed_synthetic:
                            item.pending_synthetic_exit_step = step
                            item.pending_gate = (
                                pending_signal_results[batch_index] if pending_signal_results is not None else None
                            )
                        elif delayed_gate:
                            assert pending_signal_results is not None
                            item.pending_gate = pending_signal_results[batch_index]
                        continuing.append(item)
                if cancelled:
                    self.runner.release_hidden_slots(cancelled)
                    self.last_stats.cancelled_depth_items += len(cancelled)
                self._flush_exit_kv_copies()
                if self.schedule_trace is not None:
                    self._trace("recurrent", len(survivors), depths={step: len(survivors)}, exited=len(exited))
                exited_wave.extend(exited)
                survivors = continuing
                if not survivors:
                    break
        self._trace_cohort(0)

        if delayed and exited_wave:
            with nvtx.range("cdb.coda_stage"):
                pending = self.runner.stage_coda_batch(exited_wave, allow_async=True)
            self._trace("coda", len(exited_wave))
            eager_items, eager_rows = self._select_eager_reentries(pending)
            if eager_items:
                boundary.carried.append(CarriedPreludeEntries(result=pending, items=eager_items, rows=eager_rows))
            boundary.coda = pending
            return True
        self._coda_and_sample_no_refill(exited_wave)
        return True

    api.ContinuousDepthBatchingEngine._run_no_refill_decode_wave = _run_no_refill_decode_wave
    api.ContinuousDepthBatchingEngine._phase06_patched = True
