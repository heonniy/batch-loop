"""Audit and summarize a completed two-model Phase-3 predictor packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


CELLS = (("huginn", "huginn_threshold_0p12_n640_t32"),
         ("ouro", "ouro_q0p2_n640_t32"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def summarize(root: Path) -> None:
    combined_tournament: list[dict] = []
    physical_matrix: list[dict] = []
    model_reviews = []
    artifacts: dict[str, str] = {}
    for model, feature_name in CELLS:
        feature_root = root / feature_name
        tournament_root = root / f"{model}_tournament"
        physical_root = root / f"{model}_physical"
        feature_manifest = json.loads((feature_root / "feature_manifest.json").read_text())
        physical_manifest = json.loads((physical_root / "run_manifest.json").read_text())
        if feature_manifest["status"] != "COMPLETE" or physical_manifest["status"] != "COMPLETE":
            raise RuntimeError(f"{model} collection or physical run incomplete")
        if physical_manifest["feature_manifest_sha256"] != _sha256(feature_root / "feature_manifest.json"):
            raise RuntimeError(f"{model} physical feature binding mismatch")
        tournament = _csv(tournament_root / "predictor_tournament.csv")
        test = {row["candidate"]: row for row in tournament if row["split"] == "test"}
        validation = {row["candidate"]: row for row in tournament if row["split"] == "validation"}
        if len(test) != 10 or len(validation) != 10:
            raise RuntimeError(f"{model} all-ten validation/test tournament missing")
        for row in tournament:
            combined_tournament.append({"model": model, "regime": feature_name, **row})
        physical = _csv(physical_root / "summary.csv")
        measured = [row for row in physical if row["run_kind"] == "measured"]
        warmups = [row for row in physical if row["run_kind"] == "warmup"]
        strategies = physical_manifest["strategies"]
        for strategy in strategies:
            if sum(row["strategy"] == strategy for row in measured) < 5:
                raise RuntimeError(f"{model}/{strategy} fewer than five measured repetitions")
            if sum(row["strategy"] == strategy for row in warmups) < 2:
                raise RuntimeError(f"{model}/{strategy} fewer than two warmups")
        if len({row["same_work_checksum"] for row in physical}) != 1:
            raise RuntimeError(f"{model} strategies did not execute identical work")
        baseline = "STAGED_FIFO" if model == "huginn" else "FIFO"
        b_rows = [row for row in measured if row["strategy"] == baseline]
        o_rows = [row for row in measured if row["strategy"] == "ORACLE"]
        fifo_time, oracle_time = _median(b_rows, "decode_wall_s"), _median(o_rows, "decode_wall_s")
        fifo_ric, oracle_ric = int(b_rows[0]["recurrent_invocations"]), int(o_rows[0]["recurrent_invocations"])
        if fifo_ric <= oracle_ric:
            raise RuntimeError(f"{model} held-out physical cohort has no Oracle RIC headroom")
        predictors = []
        for strategy in strategies:
            rows = [row for row in measured if row["strategy"] == strategy]
            ric_values = {int(row["recurrent_invocations"]) for row in rows}
            if len(ric_values) != 1:
                raise RuntimeError(f"{model}/{strategy} RIC changed across repeats")
            elapsed = _median(rows, "decode_wall_s")
            ric = next(iter(ric_values))
            matrix_row = {
                "model": model, "regime": feature_name, "strategy": strategy,
                "num_rows": int(rows[0]["num_rows"]), "batch_size": int(rows[0]["batch_size"]),
                "num_tokens": int(rows[0]["num_tokens"]), "measured_repeats": len(rows),
                "warmups": sum(row["strategy"] == strategy for row in warmups),
                "recurrent_invocations": ric, "ric_recovery":
                (fifo_ric - ric) / (fifo_ric - oracle_ric),
                "frozen_trace_test_ric": (int(test[strategy]["predictor_ric"])
                                          if strategy in test else None),
                "median_decode_wall_s": elapsed,
                "wall_recovery": ((fifo_time - elapsed) / (fifo_time - oracle_time)
                                  if fifo_time != oracle_time else float("nan")),
                "median_recurrent_gpu_s": _median(rows, "recurrent_gpu_s"),
                "median_staging_wall_s": _median(rows, "staging_wall_s"),
                "median_predictor_wall_s": _median(rows, "predictor_wall_s"),
                "median_sort_wall_s": _median(rows, "sort_wall_s"),
                "median_prefill_token_match_fraction": _median(rows, "prefill_token_match_fraction"),
                "median_generated_next_token_match_fraction": _median(rows, "generated_next_token_match_fraction"),
                "peak_reserved_gib_max": max(float(row["peak_reserved_gib"]) for row in rows),
                "same_work_checksum": rows[0]["same_work_checksum"],
                "run_directory": str(physical_root.relative_to(root)),
                "gpu_uuid": physical_manifest["gpu"]["uuid"],
            }
            physical_matrix.append(matrix_row)
            if strategy not in {baseline, "ORACLE"}:
                predictors.append(matrix_row)
        winner = min(predictors, key=lambda row: row["median_decode_wall_s"])
        trace_winner = max(test.values(), key=lambda row: float(row["oracle_ric_recovery"]))
        model_reviews.append({"model": model, "feature_name": feature_name,
                              "test_trace_winner": trace_winner["candidate"],
                              "test_trace_winner_ric_recovery": float(trace_winner["oracle_ric_recovery"]),
                              "physical_winner": winner["strategy"],
                              "physical_winner_ric_recovery": winner["ric_recovery"],
                              "physical_winner_wall_recovery": winner["wall_recovery"],
                              "physical_winner_predictor_wall_s": winner["median_predictor_wall_s"],
                              "fifo_wall_s": fifo_time, "oracle_wall_s": oracle_time,
                              "fifo_ric": fifo_ric, "oracle_ric": oracle_ric,
                              "gpu_uuid": physical_manifest["gpu"]["uuid"],
                              "same_work_checksum": rows[0]["same_work_checksum"]})
        for path in (feature_root / "feature_manifest.json", tournament_root / "predictor_tournament.csv",
                     physical_root / "run_manifest.json", physical_root / "summary.csv",
                     physical_root / "recurrence_events.jsonl.gz"):
            artifacts[str(path.relative_to(root))] = _sha256(path)
    with (root / "predictor_tournament.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined_tournament[0]))
        writer.writeheader()
        writer.writerows(combined_tournament)
    with (root / "physical_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(physical_matrix[0]))
        writer.writeheader()
        writer.writerows(physical_matrix)
    artifacts["predictor_tournament.csv"] = _sha256(root / "predictor_tournament.csv")
    artifacts["physical_matrix.csv"] = _sha256(root / "physical_matrix.csv")
    lines = [
        "# Phase 03 — Practical predictor tournament review",
        "",
        "This packet covers two request-disjoint held-out N256/B32, 32-token cohorts. Both lanes used",
        "frozen B=1 live depth labels and physically shrinking survivor batches. The user corrected the",
        "Phase-3 GPU assignment to GPU6=Ouro and GPU7=Huginn, overriding the main-branch GPU8 plan",
        "for this execution; no other GPU was used for model work.",
        "",
        "| Model | Best test RIC method | Physical wall winner | Winner RIC recovery | Winner wall recovery | Predictor overhead / decode |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in model_reviews:
        overhead = item["physical_winner_predictor_wall_s"] / item["fifo_wall_s"]
        lines.append(f"| {item['model']} | {item['test_trace_winner']} | {item['physical_winner']} | "
                     f"{item['physical_winner_ric_recovery']:.1%} | "
                     f"{item['physical_winner_wall_recovery']:.1%} | {overhead:.1%} |")
    lines += [
        "", "## Evidence and limits", "",
        "The complete C0–C9 validation and test rows are in `predictor_tournament.csv`. All top-three",
        "methods selected by test RIC were physically measured with two warmups and five measured",
        "repeats each. `physical_matrix.csv` gives medians; raw `summary.csv` and recurrence-event",
        "receipts remain in each model's physical directory. The wall timer includes predictor feature",
        "extraction, CPU head inference, sort, and (for Huginn) the common Prelude staging path.",
        "",
        "The work checksum is identical across strategies in each model. Physical labels and input token",
        "IDs are trace-forced to hold recurrent work and token trajectories fixed; this is not live generation",
        "or an output-quality comparison. The generated-next-token match fraction in the physical matrix",
        "quantifies output drift from the B=1 feature collector, despite frozen input tokens.",
        "Dataset revisions remain UNKNOWN (cached Hugging Face datasets); prompt content hashes and",
        "request split IDs are frozen in the feature manifests. The Phase-2 Oracle ceiling used a different",
        "first-N cohort, so these held-out FIFO/Oracle numbers are the appropriate denominator.",
        "The primary collectors started from an uncommitted worktree: their launch Git HEAD predates the",
        "collector implementation commit. Post-run source-file hashes and that provenance limitation are",
        "recorded in each feature manifest rather than mislabeling the launch HEAD as the source commit.",
        "",
        "## Model-specific physical results", "",
    ]
    for item in model_reviews:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append(f"Held-out STAGED_FIFO/FIFO→Oracle: RIC {item['fifo_ric']}→{item['oracle_ric']}; "
                     f"median decode wall {item['fifo_wall_s']:.3f}→{item['oracle_wall_s']:.3f} s. "
                     f"Best measured predictor: {item['physical_winner']}.")
        lines.append("")
    lines += ["## Decision", "",
              "The best measured candidates and negative competitors are preserved. Interpret positive",
              "wall recovery as recovery of this trace-forced no-refill Oracle opportunity, not as native",
              "quality or production throughput. The next finite action is secondary-threshold/q",
              "generalization on fresh self-consistent features, then a live-exit quality/trajectory audit.", ""]
    (root / "RESULT_REVIEW.md").write_text("\n".join(lines))
    review = {
        "result_commit": "PENDING_REVIEW_COMMIT",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "plan_or_priority": "phase03/PREDICTOR_TOURNAMENT_PLAN.md and user GPU6/7 correction",
        "execution_kind": ["exit-trace-collection", "saved-data-analysis", "real-no-refill-runtime"],
        "hardware": {item["model"]: item["gpu_uuid"] for item in model_reviews},
        "model_revisions": {model: json.loads((root / name / "feature_manifest.json").read_text())["model_revision"]
                            for model, name in CELLS},
        "dataset_revision": "UNKNOWN", "regimes": [item["feature_name"] for item in model_reviews],
        "measurement_status": "VALID_FOR_FROZEN_TRACE_FORCED_WORK",
        "implementation_status": "CHECKED_IN_SCOPE_FOR_SURVIVOR_RUNTIME",
        "simulator_status": "CHECKED_IN_SCOPE_SAME_WORK",
        "comparison_status": "COMPARABLE_WITHIN_EACH_HELD_OUT_COHORT",
        "hypothesis_status": "MEASURED_WITH_PRESERVED_NEGATIVES",
        "model_reviews": model_reviews, "artifacts_sha256": artifacts,
        "known_missing_fields": ["dataset revisions", "native/full-depth quality", "secondary-threshold generalization"],
        "next_action": "collect secondary q/threshold cohorts, evaluate frozen winner transfer and quality/trajectory drift",
    }
    (root / "review_manifest.json").write_text(json.dumps(review, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.result_root)


if __name__ == "__main__":
    main()
