"""Audit and summarize a completed two-model Phase-3 predictor packet."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


CELLS = (("huginn", "huginn_threshold_0p12_n640_t32"),
         ("ouro", "ouro_q0p2_n640_t32"))
SECONDARY = (("huginn", "huginn_threshold_0p26_n640_t32", "huginn_threshold_0p26_tournament",
              "huginn_threshold_0p26_transfer", "huginn_threshold_0p26_physical"),
             ("ouro", "ouro_q0p4_n640_t32", "ouro_q0p4_tournament",
              "ouro_q0p4_transfer", "ouro_q0p4_physical"))
CONTROLS = (("huginn", "huginn_n_eq_b_control"), ("ouro", "ouro_n_eq_b_control"))


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


def _event_diagnostics(path: Path, model: str, regime: str, rmax: int) -> tuple[list[dict], list[dict], dict]:
    maxima: dict[tuple, int] = {}
    widths: dict[tuple, list[int]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            key = (event["strategy"], event["repetition"], event["token_idx"], event["microbatch_id"])
            depth = int(event["recurrence"])
            maxima[key] = max(maxima.get(key, 0), depth)
            widths[(event["strategy"], depth)].append(int(event["active_batch_size"]))
    hist = Counter((key[0], depth) for key, depth in maxima.items())
    totals = Counter(key[0] for key in maxima)
    distribution = [{"model": model, "regime": regime, "strategy": strategy,
                     "max_depth": depth, "microbatches": hist[(strategy, depth)],
                     "fraction": hist[(strategy, depth)] / total}
                    for strategy, total in sorted(totals.items()) for depth in range(1, rmax + 1)]
    width_curves = [{"model": model, "regime": regime, "strategy": strategy,
                     "recurrence": depth, "invocations": len(values),
                     "mean_active_width": statistics.mean(values),
                     "min_active_width": min(values), "max_active_width": max(values)}
                    for (strategy, depth), values in sorted(widths.items())]
    shallow = {strategy: 1 - hist[(strategy, rmax)] / total for strategy, total in totals.items()}
    return distribution, width_curves, shallow


def summarize(root: Path) -> None:
    combined_tournament: list[dict] = []
    physical_matrix: list[dict] = []
    secondary_matrix: list[dict] = []
    control_matrix: list[dict] = []
    max_depth_rows: list[dict] = []
    active_width_rows: list[dict] = []
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
        max_depth, active_width, shallow_fraction = _event_diagnostics(
            physical_root / "recurrence_events.jsonl.gz", model, feature_name, feature_manifest["rmax"])
        max_depth_rows.extend(max_depth)
        active_width_rows.extend(active_width)
        baseline = "STAGED_FIFO" if model == "huginn" else "FIFO"
        b_rows = [row for row in measured if row["strategy"] == baseline]
        o_rows = [row for row in measured if row["strategy"] == "ORACLE"]
        baseline_by_repetition = {int(row["repetition"]): float(row["decode_wall_s"]) for row in b_rows}
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
            paired_gains = [baseline_by_repetition[int(row["repetition"])] - float(row["decode_wall_s"])
                            for row in rows]
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
                "median_paired_wall_gain_s": statistics.median(paired_gains),
                "min_paired_wall_gain_s": min(paired_gains),
                "max_paired_wall_gain_s": max(paired_gains),
                "wall_recovery": ((fifo_time - elapsed) / (fifo_time - oracle_time)
                                  if fifo_time != oracle_time else float("nan")),
                "median_recurrent_gpu_s": _median(rows, "recurrent_gpu_s"),
                "median_staging_wall_s": _median(rows, "staging_wall_s"),
                "median_predictor_wall_s": _median(rows, "predictor_wall_s"),
                "median_sort_wall_s": _median(rows, "sort_wall_s"),
                "median_prefill_token_match_fraction": _median(rows, "prefill_token_match_fraction"),
                "median_generated_next_token_match_fraction": _median(rows, "generated_next_token_match_fraction"),
                "peak_reserved_gib_max": max(float(row["peak_reserved_gib"]) for row in rows),
                "fraction_microbatches_before_rmax": shallow_fraction[strategy],
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
                              "same_work_checksum": rows[0]["same_work_checksum"],
                              "test_ric_recovery_by_candidate": {
                                  candidate: float(test[candidate]["oracle_ric_recovery"])
                                  for candidate in sorted(test)}})
        for path in (feature_root / "feature_manifest.json", tournament_root / "predictor_tournament.csv",
                     physical_root / "run_manifest.json", physical_root / "summary.csv",
                     physical_root / "recurrence_events.jsonl.gz"):
            artifacts[str(path.relative_to(root))] = _sha256(path)
    by_model = {item["model"]: item for item in model_reviews}
    for model, dirname in CONTROLS:
        control_root = root / dirname
        if not (control_root / "run_manifest.json").exists():
            continue
        manifest = json.loads((control_root / "run_manifest.json").read_text())
        if manifest["status"] != "COMPLETE":
            raise RuntimeError(f"{model} N=B control incomplete")
        measured = [row for row in _csv(control_root / "summary.csv") if row["run_kind"] == "measured"]
        fifo = "STAGED_FIFO" if model == "huginn" else "FIFO"
        first = [row for row in measured if row["strategy"] == fifo]
        oracle = [row for row in measured if row["strategy"] == "ORACLE"]
        if len(first) < 5 or len(oracle) < 5:
            raise RuntimeError(f"{model} N=B control repetition count low")
        if len({int(row["recurrent_invocations"]) for row in measured}) != 1:
            raise RuntimeError(f"{model} N=B control must have zero RIC headroom")
        if len({row["same_work_checksum"] for row in measured}) != 1:
            raise RuntimeError(f"{model} N=B control work mismatch")
        control_matrix.append({
            "model": model, "N": int(first[0]["num_rows"]), "B": int(first[0]["batch_size"]),
            "num_tokens": int(first[0]["num_tokens"]),
            "fifo_ric": int(first[0]["recurrent_invocations"]),
            "oracle_ric": int(oracle[0]["recurrent_invocations"]),
            "fifo_median_wall_s": _median(first, "decode_wall_s"),
            "oracle_median_wall_s": _median(oracle, "decode_wall_s"),
            "same_work_checksum": first[0]["same_work_checksum"],
            "gpu_uuid": manifest["gpu"]["uuid"],
        })
        for path in (control_root / "run_manifest.json", control_root / "summary.csv",
                     control_root / "recurrence_events.jsonl.gz"):
            artifacts[str(path.relative_to(root))] = _sha256(path)
    for model, feature_name, tournament_name, transfer_name, physical_name in SECONDARY:
        feature_root = root / feature_name
        tournament_root = root / tournament_name
        transfer_root = root / transfer_name
        physical_root = root / physical_name
        if not all(path.exists() for path in (feature_root / "feature_manifest.json",
                                               tournament_root / "predictor_tournament.csv",
                                               transfer_root / "transfer_tournament.csv",
                                               physical_root / "run_manifest.json")):
            continue
        feature_manifest = json.loads((feature_root / "feature_manifest.json").read_text())
        physical_manifest = json.loads((physical_root / "run_manifest.json").read_text())
        if feature_manifest["status"] != "COMPLETE" or physical_manifest["status"] != "COMPLETE":
            raise RuntimeError(f"{model} secondary collection or physical run incomplete")
        if _sha256(feature_root / "feature_manifest.json") != physical_manifest["feature_manifest_sha256"]:
            raise RuntimeError(f"{model} secondary feature binding mismatch")
        frozen_winner = by_model[model]["physical_winner"]
        baseline = "STAGED_FIFO" if model == "huginn" else "FIFO"
        if physical_manifest["strategies"] != [baseline, frozen_winner, "ORACLE"]:
            raise RuntimeError(f"{model} secondary physical did not confirm the frozen primary winner")
        transfer_rows = _csv(transfer_root / "transfer_tournament.csv")
        transfer_test = {row["candidate"]: row for row in transfer_rows if row["split"] == "test"}
        if len(transfer_test) != 10:
            raise RuntimeError(f"{model} secondary all-ten frozen transfer table missing")
        tournament_rows = _csv(tournament_root / "predictor_tournament.csv")
        tournament_test = {row["candidate"]: row for row in tournament_rows if row["split"] == "test"}
        if len(tournament_test) != 10:
            raise RuntimeError(f"{model} secondary all-ten refit tournament table missing")
        physical = _csv(physical_root / "summary.csv")
        measured = [row for row in physical if row["run_kind"] == "measured"]
        warmups = [row for row in physical if row["run_kind"] == "warmup"]
        if len({row["same_work_checksum"] for row in physical}) != 1:
            raise RuntimeError(f"{model} secondary work mismatch")
        groups = {strategy: [row for row in measured if row["strategy"] == strategy]
                  for strategy in (baseline, frozen_winner, "ORACLE")}
        for strategy, rows in groups.items():
            if len(rows) < 5 or sum(row["strategy"] == strategy for row in warmups) < 2:
                raise RuntimeError(f"{model}/{strategy} secondary repetitions insufficient")
        t_fifo = _median(groups[baseline], "decode_wall_s")
        t_pred = _median(groups[frozen_winner], "decode_wall_s")
        t_oracle = _median(groups["ORACLE"], "decode_wall_s")
        r_fifo = int(groups[baseline][0]["recurrent_invocations"])
        r_pred = int(groups[frozen_winner][0]["recurrent_invocations"])
        r_oracle = int(groups["ORACLE"][0]["recurrent_invocations"])
        if r_fifo <= r_oracle:
            raise RuntimeError(f"{model} secondary cohort has no Oracle RIC headroom")
        baseline_by_repetition = {int(item["repetition"]): float(item["decode_wall_s"])
                                  for item in groups[baseline]}
        paired_gains = [baseline_by_repetition[int(item["repetition"])] - float(item["decode_wall_s"])
                        for item in groups[frozen_winner]]
        row = {
            "model": model, "regime": feature_name, "frozen_primary_winner": frozen_winner,
            "transfer_test_ric_recovery": float(transfer_test[frozen_winner]["oracle_ric_recovery"]),
            "secondary_refit_test_ric_recovery": float(tournament_test[frozen_winner]["oracle_ric_recovery"]),
            "fifo_ric": r_fifo, "predictor_ric": r_pred, "oracle_ric": r_oracle,
            "physical_ric_recovery": (r_fifo - r_pred) / (r_fifo - r_oracle),
            "fifo_median_wall_s": t_fifo, "predictor_median_wall_s": t_pred,
            "oracle_median_wall_s": t_oracle,
            "physical_wall_recovery": (t_fifo - t_pred) / (t_fifo - t_oracle),
            "median_paired_wall_gain_s": statistics.median(paired_gains),
            "min_paired_wall_gain_s": min(paired_gains),
            "max_paired_wall_gain_s": max(paired_gains),
            "predictor_overhead_wall_s": _median(groups[frozen_winner], "predictor_wall_s"),
            "sort_overhead_wall_s": _median(groups[frozen_winner], "sort_wall_s"),
            "generated_next_token_match_fraction": _median(groups[frozen_winner],
                                                             "generated_next_token_match_fraction"),
            "gpu_uuid": physical_manifest["gpu"]["uuid"],
            "same_work_checksum": groups[frozen_winner][0]["same_work_checksum"],
        }
        secondary_matrix.append(row)
        by_model[model]["secondary"] = row
        for path in (feature_root / "feature_manifest.json", tournament_root / "predictor_tournament.csv",
                     transfer_root / "transfer_tournament.csv",
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
    with (root / "microbatch_max_depth_distribution.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(max_depth_rows[0]))
        writer.writeheader()
        writer.writerows(max_depth_rows)
    with (root / "active_width_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(active_width_rows[0]))
        writer.writeheader()
        writer.writerows(active_width_rows)
    artifacts["microbatch_max_depth_distribution.csv"] = _sha256(root / "microbatch_max_depth_distribution.csv")
    artifacts["active_width_curves.csv"] = _sha256(root / "active_width_curves.csv")
    if secondary_matrix:
        with (root / "secondary_matrix.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(secondary_matrix[0]))
            writer.writeheader()
            writer.writerows(secondary_matrix)
        artifacts["secondary_matrix.csv"] = _sha256(root / "secondary_matrix.csv")
    if control_matrix:
        with (root / "control_matrix.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(control_matrix[0]))
            writer.writeheader()
            writer.writerows(control_matrix)
        artifacts["control_matrix.csv"] = _sha256(root / "control_matrix.csv")
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
        recovery = item["test_ric_recovery_by_candidate"]
        lines.append(f"Trace ablation (Oracle RIC recovery): previous depth C0 {recovery['C0']:.1%}; "
                     f"history C1 {recovery['C1']:.1%}; trajectory C2 {recovery['C2']:.1%}; "
                     f"uncertainty C3 {recovery['C3']:.1%}; embedding C4 {recovery['C4']:.1%}; "
                     f"representation C5 {recovery['C5']:.1%}; fused regression C6 {recovery['C6']:.1%}; "
                     f"ordinal C7 {recovery['C7']:.1%}; pairwise C8 {recovery['C8']:.1%}; "
                     f"RIC-aware listwise C9 {recovery['C9']:.1%}.")
        if item["physical_winner_wall_recovery"] <= 0:
            lines.append("The best measured physical predictor did not beat staged FIFO/FIFO after overhead.")
        elif item["physical_winner_wall_recovery"] < 0.25:
            lines.append("The best measured physical predictor recovered less than 25% of Oracle wall headroom.")
        lines.append("")
    if secondary_matrix:
        lines += ["## Secondary-regime transfer", "",
                  "The primary physical winner and head weights were frozen before the secondary",
                  "threshold/q cohort was scored. The full C0–C9 frozen transfer and independently",
                  "refit secondary tables are retained; refitting is diagnostic, not the transfer claim.",
                  "Secondary physical runs compare that same frozen winner with FIFO and Oracle using",
                  "new self-consistent features/labels and the same 2+5 timing protocol.", "",
                  "| Model / regime | Frozen winner | Transfer test RIC recovery | Refit test RIC recovery | Physical RIC recovery | Physical wall recovery |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for item in secondary_matrix:
            lines.append(f"| {item['model']} / {item['regime']} | {item['frozen_primary_winner']} | "
                         f"{item['transfer_test_ric_recovery']:.1%} | "
                         f"{item['secondary_refit_test_ric_recovery']:.1%} | "
                         f"{item['physical_ric_recovery']:.1%} | "
                         f"{item['physical_wall_recovery']:.1%} |")
        lines.append("")
    if control_matrix:
        lines += ["## N=B controls", "",
                  "With one complete microbatch, FIFO and Oracle execute the same recurrent invocation",
                  "count. Their small wall differences are timing noise/control overhead, not scheduler gain.", "",
                  "| Model | N=B | FIFO/Oracle RIC | FIFO wall (s) | Oracle wall (s) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for item in control_matrix:
            lines.append(f"| {item['model']} | {item['N']} | {item['fifo_ric']} / {item['oracle_ric']} | "
                         f"{item['fifo_median_wall_s']:.3f} | {item['oracle_median_wall_s']:.3f} |")
        lines.append("")
    lines += ["## Decision", "",
              "Pairwise C8 versus Huber regression C6, and listwise C9 versus C8, are shown explicitly",
              "above. On Huginn, the Prelude-only C5 result should be weighed against the staging and",
              "predictor overhead in the physical matrix; representation predictiveness alone is not a",
              "physical speed claim. The best measured candidates and negative competitors are preserved.",
              "Interpret positive",
              "wall recovery as recovery of this trace-forced no-refill Oracle opportunity, not as native",
              "quality or production throughput. The next finite action is a live-exit",
              "quality/trajectory audit once the secondary-threshold/q transfer is complete.", ""]
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
        "known_missing_fields": ["dataset revisions", "native/full-depth quality"] +
                                ([] if len(secondary_matrix) == 2 else ["secondary-threshold generalization"]),
        "next_action": "audit native live-exit trajectory/quality before any final system claim",
    }
    (root / "review_manifest.json").write_text(json.dumps(review, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.result_root)


if __name__ == "__main__":
    main()
