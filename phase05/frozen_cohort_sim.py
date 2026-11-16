"""CPU-only frozen-depth cohort simulator for offline no-refill experiments.

This is NOT live generation and not a BF16/live Oracle. The supplied token
depths remain fixed across policies, and all stage times are an abstract
serial model. No torch, CUDA, or model code is imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

POLICIES = ("fifo", "static_first", "history_last", "history_mean",
            "history_long_first", "oracle_depth")


@dataclass(frozen=True)
class Cost:
    core_fixed: float = 1.0
    core_per_token: float = 0.0
    prelude_fixed: float = 0.0
    prelude_per_token: float = 0.0
    coda_fixed: float = 0.0
    coda_per_token: float = 0.0

    def validate(self) -> None:
        if any(getattr(self, field) < 0 for field in self.__dataclass_fields__):
            raise ValueError("All stage costs must be nonnegative")


def load_trace(path: Path, *, model: str | None = None,
               source_B: int | None = None,
               order: str | None = None) -> dict[int, list[int]]:
    """Read simple depth CSV, or explicitly select ONE condition from Phase-04R.

    Never interpret batch-specific 'live_depth' as a transferable Oracle.
    Phase-04R comparisons use its B1 'reference_depth' only.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        if not {"request_id", "token_idx"}.issubset(columns):
            raise ValueError("Missing request_id/token_idx")
        depth_key = "reference_depth" if "reference_depth" in columns else "depth"
        if depth_key not in columns:
            raise ValueError("Requires reference_depth or depth, not live_depth")
        if "B" in columns and (source_B is None or order is None):
            raise ValueError("Select a single source B and order for repeated receipts")
        by_request: dict[int, dict[int, int]] = defaultdict(dict)
        n = 0
        for row in reader:
            if model is not None and row.get("model") != model:
                continue
            if source_B is not None and int(row.get("B", -1)) != source_B:
                continue
            if order is not None and row.get("order") != order:
                continue
            rid, t, depth = (int(row["request_id"]), int(row["token_idx"]),
                              int(row[depth_key]))
            if rid < 0 or t < 0 or depth < 1:
                raise ValueError("IDs, token indices and depth must be valid")
            if t in by_request[rid]:
                raise ValueError(f"Duplicate request/token: {rid}/{t}")
            by_request[rid][t] = depth
            n += 1
    if not n:
        raise ValueError("No trace rows matched the requested selector")
    traces = {}
    for rid, positions in by_request.items():
        indices = sorted(positions)
        if indices != list(range(len(indices))):
            raise ValueError(f"Noncontiguous token indices for request {rid}")
        traces[rid] = [positions[t] for t in indices]
    return dict(sorted(traces.items()))


def simulate(traces: dict[int, list[int]], batch_size: int, policy: str,
             cost: Cost = Cost()) -> dict:
    """Sequential stage execution, strict shrink-only recurrent microbatches.

    Only one token per request can be ready at a time. A request's next token
    becomes ready after the current cohort's Coda, never before. Sorting
    happens at each cohort boundary. No in-core refill/inter-cohort merge.
    """
    if batch_size < 1 or policy not in POLICIES:
        raise ValueError("Bad batch_size or policy")
    cost.validate()
    if not traces or any(not seq or any(d < 1 for d in seq) for seq in traces.values()):
        raise ValueError("Empty/invalid depth trace")
    ready = {rid: (0, age) for age, rid in enumerate(sorted(traces))}
    arrival_clock = len(ready)
    histories: dict[int, list[int]] = {rid: [] for rid in traces}
    fixed_first: dict[int, int] = {}
    events = []
    time = 0.0
    ric = 0
    work = 0
    idle_slots = 0
    while ready:
        def key(rid: int):
            t, age = ready[rid]
            hist = histories[rid]
            if policy == "fifo":
                score = 0.0
            elif policy == "oracle_depth":
                score = float(traces[rid][t])  # FUTURE information; diagnostic only
            elif policy == "static_first":
                score = float(fixed_first.get(rid, 0))
            elif policy == "history_mean":
                score = sum(hist) / len(hist) if hist else 0.0
            else:
                score = float(hist[-1]) if hist else 0.0
            if policy == "history_long_first":
                score = -score
            return (score, age, rid)

        selected = sorted(ready, key=key)[:batch_size]
        tokens = {rid: ready[rid][0] for rid in selected}
        depths = {rid: traces[rid][tokens[rid]] for rid in selected}
        for rid in selected:
            del ready[rid]
        max_depth = max(depths.values())
        widths = [sum(depth >= r for depth in depths.values())
                  for r in range(1, max_depth + 1)]
        if widths[0] != len(selected) or widths != sorted(widths, reverse=True):
            raise AssertionError("No-refill invariant violated")
        interval_work = sum(depths.values())
        ric += max_depth
        work += interval_work
        idle_slots += len(selected) * max_depth - interval_work
        duration = (
            cost.prelude_fixed + cost.prelude_per_token * len(selected)
            + sum(cost.core_fixed + cost.core_per_token * width for width in widths)
            + cost.coda_fixed + cost.coda_per_token * len(selected)
        )
        time += duration
        events.append({
            "cohort": len(events), "requests": selected,
            "token_idx": tokens, "depth": depths, "survivor_widths": widths,
            "core_calls": max_depth, "core_work": interval_work,
            "duration_units": duration, "cumulative_time_units": time,
        })
        for rid in selected:
            histories[rid].append(depths[rid])
            if rid not in fixed_first:
                fixed_first[rid] = depths[rid]
            next_idx = tokens[rid] + 1
            if next_idx < len(traces[rid]):
                ready[rid] = (next_idx, arrival_clock)
                arrival_clock += 1
    total_tokens = sum(len(v) for v in traces.values())
    assert work == sum(map(sum, traces.values()))
    assert total_tokens == sum(len(e["requests"]) for e in events)
    return {
        "evidence_kind": "CPU_FROZEN_TRACE_ABSTRACT_COST_NOT_LIVE",
        "policy": policy, "batch_size": batch_size,
        "requests": len(traces), "tokens": total_tokens, "cohorts": len(events),
        "recurrent_invocations_RIC": ric, "token_loop_work": work,
        "idle_slots_within_cohorts": idle_slots,
        "slot_utilization": work / (work + idle_slots),
        "simulated_time_units": time, "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--source-B", type=int)
    parser.add_argument("--order")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--core-fixed", type=float, default=1.0)
    parser.add_argument("--core-per-token", type=float, default=0.0)
    parser.add_argument("--prelude-fixed", type=float, default=0.0)
    parser.add_argument("--prelude-per-token", type=float, default=0.0)
    parser.add_argument("--coda-fixed", type=float, default=0.0)
    parser.add_argument("--coda-per-token", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = load_trace(args.input, model=args.model, source_B=args.source_B, order=args.order)
    cost = Cost(args.core_fixed, args.core_per_token, args.prelude_fixed,
                args.prelude_per_token, args.coda_fixed, args.coda_per_token)
    outcomes = [simulate(traces, args.batch_size, policy, cost) for policy in args.policies]
    provenance = {
        "input_path": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_B": args.source_B, "order": args.order, "model": args.model,
        "note": "Frozen-work structural study only. No measured GPU times or live counterfactual.",
        "cost": vars(cost),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"provenance": provenance, "outcomes": outcomes},
                                      indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
