"""Trace-only FIFO/Oracle recurrent-invocation characterization.

This module never loads a model or touches CUDA. RIC and slot utilization are
logical diagnostics, not physical latency or speedup measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_depths(path: Path, n: int, num_tokens: int, rmax: int) -> list[list[int]]:
    """Require a complete rectangular request-prefix/token-prefix cohort."""
    matrix = [[0] * num_tokens for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            request_id, token_idx, depth = (int(row[key]) for key in ("request_id", "token_idx", "depth"))
            if request_id < 0 or token_idx < 0:
                raise ValueError(f"negative request/token at {path}:{line_number}")
            if request_id >= n or token_idx >= num_tokens:
                continue
            key = (request_id, token_idx)
            if key in seen:
                raise ValueError(f"duplicate depth at {path}:{line_number}: {key}")
            if not 1 <= depth <= rmax:
                raise ValueError(f"depth {depth} outside 1..{rmax} at {path}:{line_number}")
            seen.add(key)
            matrix[request_id][token_idx] = depth
    if len(seen) != n * num_tokens:
        missing = next((i, t) for i in range(n) for t in range(num_tokens) if (i, t) not in seen)
        raise ValueError(f"incomplete trace {path}: {len(seen)}/{n * num_tokens}, first missing {missing}")
    return matrix


def _percentile(values: list[float] | list[int], q: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if upper == lower:
        return float(ordered[lower])
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def _cohort_hash(matrix: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for request_id, depths in enumerate(matrix):
        for token_idx, depth in enumerate(depths):
            digest.update(f"{request_id},{token_idx},{depth}\n".encode())
    return digest.hexdigest()


def _microbatches(depths: list[int], batch_size: int, strategy: str) -> list[list[int]]:
    order = list(range(len(depths)))
    if strategy == "oracle":
        order.sort(key=lambda request_id: (depths[request_id], request_id))
    elif strategy != "fifo":
        raise ValueError(strategy)
    if len(order) % batch_size:
        raise ValueError("N must be divisible by B")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    assert sorted(request_id for batch in batches for request_id in batch) == list(range(len(depths)))
    return batches


def characterize(matrix: list[list[int]], batch_size: int, rmax: int) -> tuple[dict, list[dict]]:
    n = len(matrix)
    if n < 1 or not matrix or not matrix[0] or n % batch_size:
        raise ValueError("nonempty rectangular N and divisible B required")
    num_tokens = len(matrix[0])
    if any(len(row) != num_tokens or any(not 1 <= d <= rmax for d in row) for row in matrix):
        raise ValueError("invalid trace matrix")
    flattened = [depth for row in matrix for depth in row]
    histogram = Counter(flattened)
    work = sum(flattened)
    stats: dict[str, dict] = {
        strategy: {"ric": 0, "maxima": [], "full": 0, "early": 0}
        for strategy in ("fifo", "oracle")
    }
    boundary_rows: list[dict] = []
    for token_idx in range(num_tokens):
        depths = [matrix[request_id][token_idx] for request_id in range(n)]
        per_strategy = {}
        for strategy in ("fifo", "oracle"):
            batches = _microbatches(depths, batch_size, strategy)
            maxima = [max(depths[request_id] for request_id in batch) for batch in batches]
            if sum(depths[request_id] for batch in batches for request_id in batch) != sum(depths):
                raise AssertionError("FIFO/Oracle selected work differs")
            ric = sum(maxima)
            slot_utilization = sum(depths) / (batch_size * ric)
            stats[strategy]["ric"] += ric
            stats[strategy]["maxima"].extend(maxima)
            stats[strategy]["full"] += sum(m == rmax for m in maxima)
            stats[strategy]["early"] += sum(m < rmax for m in maxima)
            per_strategy[strategy] = (ric, slot_utilization, len(maxima))
        fifo_ric, fifo_util, batch_count = per_strategy["fifo"]
        oracle_ric, oracle_util, _ = per_strategy["oracle"]
        boundary_rows.append({
            "token_idx": token_idx, "num_microbatches": batch_count,
            "token_work": sum(depths), "fifo_ric": fifo_ric, "oracle_ric": oracle_ric,
            "oracle_ric_reduction_pct": 100 * (fifo_ric - oracle_ric) / fifo_ric,
            "fifo_slot_utilization": fifo_util, "oracle_slot_utilization": oracle_util,
        })
    fifo_ric = stats["fifo"]["ric"]
    oracle_ric = stats["oracle"]["ric"]
    total_batches = n // batch_size * num_tokens
    result = {
        "N": n, "B": batch_size, "num_tokens": num_tokens,
        "num_microbatches_per_token": n // batch_size,
        "num_microbatches": total_batches,
        "same_work_checksum": _cohort_hash(matrix), "total_token_loop_work": work,
        "mean_depth": statistics.mean(flattened),
        "depth_std": statistics.pstdev(flattened),
        "depth_p10": _percentile(flattened, 0.1),
        "depth_p50": _percentile(flattened, 0.5),
        "depth_p90": _percentile(flattened, 0.9),
        "depth_histogram": json.dumps(dict(sorted(histogram.items())), separators=(",", ":")),
        "fifo_ric": fifo_ric, "oracle_ric": oracle_ric,
        "oracle_ric_reduction_pct": 100 * (fifo_ric - oracle_ric) / fifo_ric,
        "fifo_slot_utilization": work / (batch_size * fifo_ric),
        "oracle_slot_utilization": work / (batch_size * oracle_ric),
        "fifo_mean_microbatch_max_depth": statistics.mean(stats["fifo"]["maxima"]),
        "oracle_mean_microbatch_max_depth": statistics.mean(stats["oracle"]["maxima"]),
        "fifo_num_full_rmax_microbatches": stats["fifo"]["full"],
        "oracle_num_full_rmax_microbatches": stats["oracle"]["full"],
        "fifo_num_early_microbatches": stats["fifo"]["early"],
        "oracle_num_early_microbatches": stats["oracle"]["early"],
        "fifo_fraction_early_microbatches": stats["fifo"]["early"] / total_batches,
        "oracle_fraction_early_microbatches": stats["oracle"]["early"] / total_batches,
        "fifo_max_depth_distribution": json.dumps(dict(sorted(Counter(stats["fifo"]["maxima"]).items())), separators=(",", ":")),
        "oracle_max_depth_distribution": json.dumps(dict(sorted(Counter(stats["oracle"]["maxima"]).items())), separators=(",", ":")),
        "token_ric_reduction_p10": _percentile([r["oracle_ric_reduction_pct"] for r in boundary_rows], 0.1),
        "token_ric_reduction_p50": _percentile([r["oracle_ric_reduction_pct"] for r in boundary_rows], 0.5),
        "token_ric_reduction_p90": _percentile([r["oracle_ric_reduction_pct"] for r in boundary_rows], 0.9),
    }
    assert fifo_ric >= oracle_ric, "sorted grouping should minimize sum of group maxima"
    assert stats["fifo"]["full"] + stats["fifo"]["early"] == total_batches
    assert stats["oracle"]["full"] + stats["oracle"]["early"] == total_batches
    return result, boundary_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True,
                        help="model:source:threshold:path; may be repeated")
    parser.add_argument("--n", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--b", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.num_tokens < 1 or any(x < 1 for x in args.n + args.b):
        parser.error("N, B, and token count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"execution_kind": "trace-sim", "source_sha": source_sha,
                "num_tokens": args.num_tokens, "N": args.n, "B": args.b,
                "trace_sources": [], "metric_scope": "logical RIC/headroom only; not physical speedup"}
    ranked = []
    boundaries = []
    depth_distributions = []
    for spec in args.trace:
        try:
            model, source, threshold, path_text = spec.split(":", 3)
        except ValueError as exc:
            raise ValueError(f"invalid --trace spec: {spec}") from exc
        if model not in {"ouro", "huginn"}:
            raise ValueError(model)
        path = Path(path_text).resolve()
        rmax = 4 if model == "ouro" else 16
        trace_hash = sha256_file(path)
        matrix_max = load_depths(path, max(args.n), args.num_tokens, rmax)
        manifest["trace_sources"].append({
            "model": model, "source": source, "threshold": threshold,
            "path": str(path), "sha256": trace_hash, "Rmax": rmax,
        })
        for n in args.n:
            matrix = matrix_max[:n]
            for b in args.b:
                if n % b:
                    continue
                summary, token_rows = characterize(matrix, b, rmax)
                shared = {"model": model, "source": source, "threshold": threshold,
                          "trace_sha256": trace_hash}
                ranked.append({**shared, **summary})
                boundaries.extend({**shared, "N": n, "B": b, **row} for row in token_rows)
                if b == min(args.b):
                    depth_distributions.append({
                        **shared, "N": n, "num_tokens": args.num_tokens,
                        "mean_depth": summary["mean_depth"], "depth_std": summary["depth_std"],
                        "p10": summary["depth_p10"], "p50": summary["depth_p50"],
                        "p90": summary["depth_p90"], "histogram": summary["depth_histogram"],
                        "fraction_by_depth": json.dumps({
                            depth: count / (n * args.num_tokens)
                            for depth, count in sorted(Counter(d for row in matrix for d in row).items())
                        }, separators=(",", ":")),
                    })
    ranked.sort(key=lambda row: (-row["oracle_ric_reduction_pct"], row["model"], row["source"], row["threshold"], row["N"], row["B"]))
    _write_csv(args.output_dir / "oracle_ric_sweep.csv", ranked)
    _write_csv(args.output_dir / "token_boundary_ric.csv", boundaries)
    _write_csv(args.output_dir / "depth_distributions.csv", depth_distributions)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for row in ranked[:15]:
        print(f"{row['model']:6s} {row['source']:8s} threshold={row['threshold']:>4s} "
              f"N={row['N']:3d} B={row['B']:2d} FIFO_RIC={row['fifo_ric']:5d} "
              f"Oracle_RIC={row['oracle_ric']:5d} reduction={row['oracle_ric_reduction_pct']:5.1f}%")


if __name__ == "__main__":
    main()
