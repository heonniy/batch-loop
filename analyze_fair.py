#!/usr/bin/env python3
"""Fair analysis for offline no-refill looped-LM batching.

All strategies see EXACTLY the same token/request subset for each (token_idx, batch_size).
Only the ordering inside that shared subset changes. This removes the remainder/drop bias
that affected the first PoC.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from poc import add_causal_predictors, order_token_group, read_traces


def simulate_fair(df: pd.DataFrame, batch_size: int, strategy: str, seed: int):
    rng = random.Random(seed + batch_size * 1009 + sum(ord(c) for c in strategy))
    total_work = total_capacity = total_idle = num_batches = 0
    depth_stds = []
    active_by_loop = defaultdict(list)
    num_selected_tokens = 0
    selected_work_checksum = 0

    for token_idx, g in df.groupby("token_idx", sort=True):
        # IMPORTANT: choose the common subset BEFORE strategy-specific ordering.
        base = g.sort_values("request_id")
        n_full = len(base) // batch_size
        if n_full == 0:
            continue
        base = base.iloc[: n_full * batch_size].copy()
        num_selected_tokens += len(base)
        selected_work_checksum += int(base["depth"].sum())

        ordered = order_token_group(base, strategy, rng)
        assert set(ordered.index) == set(base.index)
        assert int(ordered["depth"].sum()) == int(base["depth"].sum())

        for bi in range(n_full):
            mb = ordered.iloc[bi * batch_size : (bi + 1) * batch_size]
            depths = mb["depth"].to_numpy(dtype=int)
            max_d = int(depths.max())
            work = int(depths.sum())
            capacity = int(batch_size * max_d)
            idle = capacity - work
            total_work += work
            total_capacity += capacity
            total_idle += idle
            num_batches += 1
            depth_stds.append(float(np.std(depths)))
            for r in range(1, max_d + 1):
                active_by_loop[r].append(float(np.mean(depths >= r)))

    metrics = {
        "batch_size": batch_size,
        "strategy": strategy,
        "slot_utilization": (total_work / total_capacity) if total_capacity else np.nan,
        "idle_slot_ratio": (total_idle / total_capacity) if total_capacity else np.nan,
        "num_microbatches": num_batches,
        "mean_depth_std": float(np.mean(depth_stds)) if depth_stds else np.nan,
        "total_token_loop_work": total_work,
        "total_capacity_slots": total_capacity,
        "num_selected_tokens": num_selected_tokens,
        "selected_work_checksum": selected_work_checksum,
    }
    curve = pd.DataFrame([
        {
            "batch_size": batch_size,
            "strategy": strategy,
            "loop_step": r,
            "mean_active_fraction": float(np.mean(vals)),
            "num_microbatch_samples": len(vals),
        }
        for r, vals in sorted(active_by_loop.items())
    ])
    return metrics, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--batch-sizes", default="16,32,64,128,256,512")
    ap.add_argument("--strategies", default="fifo,random,prev,recent,seqmean,oracle")
    ap.add_argument("--recent-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    df = read_traces(args.traces)
    batches = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_curves = []
    validation = []

    for (source_file, model), raw in df.groupby(["source_file", "model"], dropna=False):
        run = Path(str(source_file)).stem
        d = add_causal_predictors(raw, args.recent_k)
        for b in batches:
            per_strategy = []
            for strategy in strategies:
                m, c = simulate_fair(d, b, strategy, args.seed)
                m.update({"run": run, "model": model, "source_file": source_file})
                per_strategy.append(m)
                if not c.empty:
                    c.insert(0, "run", run)
                    c["model"] = model
                    all_curves.append(c)
            # Fairness invariant: every strategy must see identical work/subset.
            work = {m["selected_work_checksum"] for m in per_strategy}
            ntok = {m["num_selected_tokens"] for m in per_strategy}
            fair = (len(work) == 1 and len(ntok) == 1)
            validation.append({"run": run, "model": model, "batch_size": b, "fair": fair,
                               "selected_work_checksum": next(iter(work)) if len(work) == 1 else None,
                               "num_selected_tokens": next(iter(ntok)) if len(ntok) == 1 else None})
            if not fair:
                raise AssertionError(f"Fair-subset invariant failed for {run}, B={b}")
            all_rows.extend(per_strategy)

    summary = pd.DataFrame(all_rows)
    fifo = summary[summary.strategy == "fifo"][["run", "batch_size", "slot_utilization"]].rename(columns={"slot_utilization":"fifo_utilization"})
    summary = summary.merge(fifo, on=["run", "batch_size"], how="left")
    summary["gain_over_fifo_pp"] = (summary.slot_utilization - summary.fifo_utilization) * 100.0

    summary.to_csv(out / "summary_fair.csv", index=False)
    pd.DataFrame(validation).to_csv(out / "fairness_validation.csv", index=False)
    if all_curves:
        pd.concat(all_curves, ignore_index=True).to_csv(out / "valley_curves_fair.csv", index=False)

    # Compact oracle-vs-FIFO table for go/no-go.
    compact = summary[summary.strategy.isin(["fifo", "oracle"])].copy()
    compact.to_csv(out / "oracle_vs_fifo.csv", index=False)
    print(compact[["run", "batch_size", "strategy", "slot_utilization", "idle_slot_ratio", "gain_over_fifo_pp"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
