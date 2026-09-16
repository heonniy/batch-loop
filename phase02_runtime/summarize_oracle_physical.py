"""Audit and summarize selected physical FIFO/Oracle M2 run directories.

Warmups are excluded by the benchmark before it writes ``summary.csv``.  The
reported timing columns are medians of *measured* repetitions, never trace-only
estimates.  Each strategy must have the same frozen depth work and prompt set.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


TIMING_COLUMNS = (
    "recurrent_pass_gpu_ms_including_kv",
    "prelude_gpu_ms",
    "coda_gpu_ms",
    "compaction_gpu_ms",
    "kv_update_gpu_ms_within_stages",
    "exit_kv_gpu_ms",
    "total_decode_wall_ms",
    "generated_tokens_per_decode_wall_s",
    "peak_allocated_gib",
    "peak_reserved_gib",
)


def summarize_run(directory: Path) -> dict:
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if manifest["stage"] != "m2":
        raise ValueError(f"not an M2 run: {directory}")
    if len(rows) != 2 * manifest["repeats"]:
        raise ValueError(f"unexpected measured repeat count: {directory}")
    groups = {strategy: [row for row in rows if row["strategy"] == strategy]
              for strategy in ("fifo", "oracle")}
    if any(len(group) != manifest["repeats"] for group in groups.values()):
        raise ValueError(f"missing FIFO or Oracle measured repeats: {directory}")
    for strategy, group in groups.items():
        if sorted(int(row["repetition"]) for row in group) != list(range(manifest["repeats"])):
            raise ValueError(f"nonconsecutive {strategy} repeats: {directory}")
    invariant_fields = ("generated_tokens", "num_intervals", "total_token_loop_work",
                        "same_work_checksum", "actual_kv_writes", "last_exited_fill_operations",
                        "shared_kv_overwrites")
    for field in invariant_fields:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"FIFO/Oracle work differs in {field}: {directory}")
    ric = {}
    for strategy, group in groups.items():
        values = {int(row["recurrent_invocation_count"]) for row in group}
        if len(values) != 1:
            raise ValueError(f"unstable {strategy} RIC: {directory}")
        ric[strategy] = values.pop()
    if ric["oracle"] > ric["fifo"]:
        raise ValueError(f"Oracle RIC exceeds FIFO: {directory}")
    result = {
        "run_dir": str(directory),
        "source_sha": manifest["source_sha"],
        "physical_gpu": manifest["gpu"]["physical_gpu"],
        "gpu_uuid": manifest["gpu"]["gpu_query"].split(", ")[1],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "kv_backend": manifest["kv_backend"],
        "exit_source": manifest["exit_source"],
        "trace_sha256": manifest["trace"]["sha256"],
        "prompt_source": manifest["prompt_source"],
        "prompt_sha256": manifest["prompt_sha256"],
        "threshold": manifest["threshold"],
        "N": manifest["num_rows"],
        "B": manifest["batch_size"],
        "num_tokens": manifest["num_tokens"],
        "warmups_per_strategy": manifest["warmups"],
        "measured_repeats_per_strategy": manifest["repeats"],
        "generated_tokens": int(rows[0]["generated_tokens"]),
        "total_token_loop_work": int(rows[0]["total_token_loop_work"]),
        "same_work_checksum": rows[0]["same_work_checksum"],
        "fifo_ric": ric["fifo"],
        "oracle_ric": ric["oracle"],
        "oracle_ric_reduction_pct": 100 * (ric["fifo"] - ric["oracle"]) / ric["fifo"],
    }
    for column in TIMING_COLUMNS:
        for strategy, group in groups.items():
            result[f"{strategy}_median_{column}"] = statistics.median(float(row[column]) for row in group)
    for column in ("recurrent_pass_gpu_ms_including_kv", "total_decode_wall_ms"):
        fifo = result[f"fifo_median_{column}"]
        oracle = result[f"oracle_median_{column}"]
        result[f"oracle_{column}_reduction_pct"] = 100 * (fifo - oracle) / fifo
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize_run(path) for path in args.run_dir]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model']} t={row['threshold']} N={row['N']} B={row['B']} "
              f"RIC {row['fifo_ric']}->{row['oracle_ric']} "
              f"({row['oracle_ric_reduction_pct']:.1f}%), "
              f"core {row['oracle_recurrent_pass_gpu_ms_including_kv_reduction_pct']:.1f}%, "
              f"wall {row['oracle_total_decode_wall_ms_reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()
