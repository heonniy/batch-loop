"""Analyze BF16 live-depth ordering using committed Phase-04R receipts only.

No model imports, CUDA calls, or counterfactual live-Oracle calculations occur here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import kendalltau, pearsonr, spearmanr


REPO = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO / "phase04/results/20260917T060824Z_50910e6_bf16_resolution"
MODELS = ("ouro", "huginn")
REQUIRED_COLUMNS = {
    "model", "B", "order", "token_idx", "request_id", "reference_depth",
    "live_depth", "depth_flip", "reference_next_token", "live_next_token",
    "next_token_flip", "reference_min_abs_margin", "margin_bin",
}
SUMMARY_COLUMNS = {
    "model", "B", "order", "positions", "depth_flips", "next_token_flips",
    "reference_B1_work", "actual_live_work", "actual_live_RIC",
    "frozen_B1_depth_grouping_RIC", "reference_margin_lt_0.005",
}
GAPS = ("1", "2", "3", ">=4")
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_REPS = 500


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required CSV columns in {path}")
        return list(reader)


def integer(row: dict, field: str) -> int:
    return int(row[field])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_key(row: dict) -> tuple[int, str]:
    return integer(row, "B"), row["order"]


def identity_key(row: dict) -> tuple[int, int]:
    return integer(row, "token_idx"), integer(row, "request_id")


def full_key(row: dict) -> tuple[str, int, str, int, int]:
    return row["model"], *condition_key(row), *identity_key(row)


def ric_for_order(rows: list[dict], order: list[int], width: int, depth_field: str) -> int:
    by_step = defaultdict(dict)
    for row in rows:
        by_step[integer(row, "token_idx")][integer(row, "request_id")] = integer(row, depth_field)
    total = 0
    for by_request in by_step.values():
        ordered = [by_request[request] for request in order]
        total += sum(max(ordered[start:start + width]) for start in range(0, len(order), width))
    return total


def validate_receipt(model: str, root: Path) -> tuple[list[dict], list[dict], dict, dict]:
    folder = root / f"{model}_fresh_validation"
    paths = {name: folder / name for name in (
        "per_token_validation.csv", "batch_shape_validation.csv", "margin_bin_summary.csv", "validation_manifest.json"
    )}
    manifest = json.loads(paths["validation_manifest.json"].read_text())
    if manifest.get("model") != model or manifest.get("status") != "FRESH_VALIDATION_RECORDED":
        raise ValueError(f"Invalid {model} receipt manifest")
    rows = read_csv(paths["per_token_validation.csv"], REQUIRED_COLUMNS)
    summaries = read_csv(paths["batch_shape_validation.csv"], SUMMARY_COLUMNS)
    widths = manifest["B_values"]
    orders = manifest["orderings"]
    request_ids = orders["fifo"]
    n_requests = len(request_ids)
    n_steps = int(manifest["decode_positions"])
    if widths != [2, 4, 8] or set(orders) != {"fifo", "permutation_seed_17", "permutation_seed_29"}:
        raise ValueError(f"Unexpected B/order coverage in {model} manifest")
    if n_requests != 32 or n_steps != 8 or request_ids != list(range(32)):
        raise ValueError(f"Unexpected {model} cohort shape")
    if len(manifest["effective_input_hashes"]) != n_requests or len(set(manifest["effective_input_hashes"])) != n_requests:
        raise ValueError(f"Invalid {model} input hashes")
    if any(sorted(order) != request_ids for order in orders.values()):
        raise ValueError(f"Invalid {model} order permutation")
    expected_conditions = {(width, name) for width in widths for name in orders}
    expected_ids = {(step, request) for step in range(n_steps) for request in request_ids}
    expected_count = len(expected_conditions) * len(expected_ids)
    if len(rows) != expected_count or len(summaries) != len(expected_conditions):
        raise ValueError(f"Unexpected {model} row counts: {len(rows)} / {len(summaries)}")
    if len({full_key(row) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate {model} model/B/order/token/request identity")
    by_condition = defaultdict(list)
    references = defaultdict(set)
    for row in rows:
        if row["model"] != model:
            raise ValueError(f"Mismatched model in {model} receipt")
        condition = condition_key(row)
        if condition not in expected_conditions or identity_key(row) not in expected_ids:
            raise ValueError(f"Unexpected condition or request-token in {model} receipt")
        reference = integer(row, "reference_depth")
        live = integer(row, "live_depth")
        if not 1 <= reference <= (4 if model == "ouro" else 16) or not 1 <= live <= (4 if model == "ouro" else 16):
            raise ValueError(f"Out-of-range depth in {model} receipt")
        if integer(row, "depth_flip") != int(reference != live):
            raise ValueError(f"Depth flag mismatch in {model} receipt")
        if integer(row, "next_token_flip") != int(row["reference_next_token"] != row["live_next_token"]):
            raise ValueError(f"Next-token flag mismatch in {model} receipt")
        margin = float(row["reference_min_abs_margin"])
        if not math.isfinite(margin) or margin < 0:
            raise ValueError(f"Invalid margin in {model} receipt")
        references[identity_key(row)].add((reference, row["reference_next_token"], row["margin_bin"], margin))
        by_condition[condition].append(row)
    if len(references) != len(expected_ids) or any(len(values) != 1 for values in references.values()):
        raise ValueError(f"Canonical B1 reference differs by condition for {model}")
    if set(by_condition) != expected_conditions or any({identity_key(row) for row in part} != expected_ids for part in by_condition.values()):
        raise ValueError(f"Incomplete request-token intersections for {model}")
    summary_by_condition = {}
    for row in summaries:
        key = condition_key(row)
        if row["model"] != model or key not in expected_conditions or key in summary_by_condition:
            raise ValueError(f"Invalid/duplicate condition summary for {model}")
        summary_by_condition[key] = row
    if set(summary_by_condition) != expected_conditions:
        raise ValueError(f"Missing condition summary for {model}")
    for (width, name), part in by_condition.items():
        summary = summary_by_condition[width, name]
        checks = {
            "positions": len(part),
            "depth_flips": sum(integer(row, "depth_flip") for row in part),
            "next_token_flips": sum(integer(row, "next_token_flip") for row in part),
            "reference_margin_lt_0.005": sum(float(row["reference_min_abs_margin"]) < 0.005 for row in part),
            "reference_B1_work": sum(integer(row, "reference_depth") for row in part),
            "actual_live_work": sum(integer(row, "live_depth") for row in part),
            "actual_live_RIC": ric_for_order(part, orders[name], width, "live_depth"),
            "frozen_B1_depth_grouping_RIC": ric_for_order(part, orders[name], width, "reference_depth"),
        }
        if any(integer(summary, field) != value for field, value in checks.items()):
            raise ValueError(f"Saved summary differs from raw {model} rows: B{width}/{name}: {checks}")
        if not math.isclose(float(summary["depth_flip_rate"]), checks["depth_flips"] / len(part), abs_tol=1e-12):
            raise ValueError(f"Saved depth-flip rate differs for {model}/B{width}/{name}")
        if not math.isclose(float(summary["next_token_flip_rate"]), checks["next_token_flips"] / len(part), abs_tol=1e-12):
            raise ValueError(f"Saved next-token-flip rate differs for {model}/B{width}/{name}")
        if checks["reference_B1_work"] != int(manifest["reference_B1_total_work"]):
            raise ValueError(f"Manifest work differs for {model}")
    return rows, summaries, manifest, {name: {"path": str(path.relative_to(REPO)), "sha256": sha256(path)} for name, path in paths.items()}


def rank_metrics(rows: list[dict]) -> dict:
    ref = [integer(row, "reference_depth") for row in rows]
    live = [integer(row, "live_depth") for row in rows]
    defined = len(set(ref)) > 1 and len(set(live)) > 1
    tau = float(kendalltau(ref, live).statistic) if defined else None
    rho = float(spearmanr(ref, live).statistic) if defined else None
    return {
        "kendall_tau_b": tau, "spearman_rho": rho,
        "status": "defined" if defined else "NA_constant",
        "reference_levels": len(set(ref)), "live_levels": len(set(live)),
        "positions": len(rows), "exact_depth_matches": sum(a == b for a, b in zip(ref, live)),
        "exact_depth_agreement": sum(a == b for a, b in zip(ref, live)) / len(rows),
    }


def pair_counts(rows: list[dict]) -> dict[str, Counter]:
    counts = {gap: Counter() for gap in GAPS}
    for left_idx, left in enumerate(rows):
        for right in rows[left_idx + 1:]:
            a, b = integer(left, "reference_depth"), integer(right, "reference_depth")
            if a == b:
                continue
            shallow, deep = (left, right) if a < b else (right, left)
            gap = abs(a - b)
            bucket = str(gap) if gap < 4 else ">=4"
            live_shallow = integer(shallow, "live_depth")
            live_deep = integer(deep, "live_depth")
            kind = "preserved" if live_shallow < live_deep else ("collapsed_to_tie" if live_shallow == live_deep else "inverted")
            counts[bucket][kind] += 1
    return counts


def percentile_summary(values: list[float]) -> dict:
    if not values:
        return {"median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {name: float(value) for name, value in zip(
        ("median", "q1", "q3", "min", "max"),
        (np.median(values), np.quantile(values, .25), np.quantile(values, .75), min(values), max(values)),
    )}


def cluster_bootstrap_ci(pools: list[list[dict]], request_ids: list[int], *, reps: int, seed: int) -> tuple[float | None, float | None]:
    """Resample request IDs jointly across every token/condition pool."""
    rng = random.Random(seed)
    indexed = [{integer(row, "request_id"): row for row in pool} for pool in pools]
    if any(set(pool) != set(request_ids) for pool in indexed):
        raise ValueError("Missing request in bootstrap pool")
    medians = []
    for _ in range(reps):
        sampled = rng.choices(request_ids, k=len(request_ids))
        values = []
        for pool in indexed:
            reference = [integer(pool[request], "reference_depth") for request in sampled]
            live = [integer(pool[request], "live_depth") for request in sampled]
            if len(set(reference)) > 1 and len(set(live)) > 1:
                values.append(float(kendalltau(reference, live).statistic))
        defined = [value for value in values if value is not None]
        if defined:
            medians.append(float(np.median(defined)))
    if not medians:
        return None, None
    return float(np.quantile(medians, .025)), float(np.quantile(medians, .975))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def delta_summary(rows: list[dict], scope: str, model: str, width: int | None, order: str | None, margin_bin: str) -> dict:
    deltas = [integer(row, "live_depth") - integer(row, "reference_depth") for row in rows]
    flipped = [abs(delta) for delta in deltas if delta]
    count = len(rows)
    return {
        "scope": scope, "model": model, "B": width, "order": order, "margin_bin": margin_bin,
        "positions": count, "exact_matches": count - len(flipped), "exact_match_rate": (count - len(flipped)) / count if count else None,
        "depth_flips": len(flipped), "flip_rate": len(flipped) / count if count else None,
        "flip_abs_mean": float(np.mean(flipped)) if flipped else None,
        "flip_abs_median": float(np.median(flipped)) if flipped else None,
        "flip_abs_max": max(flipped) if flipped else None,
        "abs_1": sum(value == 1 for value in flipped), "abs_2": sum(value == 2 for value in flipped),
        "abs_ge_3": sum(value >= 3 for value in flipped),
        "abs_1_fraction_of_flips": sum(value == 1 for value in flipped) / len(flipped) if flipped else None,
        "abs_2_fraction_of_flips": sum(value == 2 for value in flipped) / len(flipped) if flipped else None,
        "abs_ge_3_fraction_of_flips": sum(value >= 3 for value in flipped) / len(flipped) if flipped else None,
        "shallower": sum(value < 0 for value in deltas), "deeper": sum(value > 0 for value in deltas),
    }


def pair_summary(model: str, width: int | None, order: str | None, scope: str, gap: str, counts: Counter) -> dict:
    n = sum(counts.values())
    return {
        "scope": scope, "model": model, "B": width, "order": order, "canonical_depth_gap": gap,
        "comparable_pairs": n, "preserved": counts["preserved"], "collapsed_to_tie": counts["collapsed_to_tie"],
        "inverted": counts["inverted"],
        "preserved_rate": counts["preserved"] / n if n else None,
        "tie_rate": counts["collapsed_to_tie"] / n if n else None,
        "inversion_rate": counts["inverted"] / n if n else None,
        "tie_aware_concordance": (counts["preserved"] + .5 * counts["collapsed_to_tie"]) / n if n else None,
    }


def summarize_rank(rows: list[dict], ci: tuple[float | None, float | None]) -> dict:
    taus = [row["kendall_tau_b"] for row in rows if row["kendall_tau_b"] is not None]
    rhos = [row["spearman_rho"] for row in rows if row["spearman_rho"] is not None]
    tau = percentile_summary(taus)
    rho = percentile_summary(rhos)
    return {
        "token_pools": len(rows), "tau_defined_pools": len(taus),
        "tau_defined_fraction": len(taus) / len(rows),
        **{f"tau_{key}": value for key, value in tau.items()},
        "tau_median_bootstrap_ci_low": ci[0], "tau_median_bootstrap_ci_high": ci[1],
        **{f"spearman_{key}": value for key, value in rho.items()},
        "exact_depth_agreement_mean_by_pool": float(np.mean([row["exact_depth_agreement"] for row in rows])),
    }


def plot_delta(model: str, hist: list[dict], output: Path) -> None:
    chosen = [row for row in hist if row["model"] == model and row["scope"] == "model" and row["histogram_kind"] == "signed"]
    chosen.sort(key=lambda row: int(row["delta"]))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([int(row["delta"]) for row in chosen], [int(row["count"]) for row in chosen], color="#32658f")
    ax.set(xlabel="Live depth - canonical B1 depth", ylabel="Observed positions across 9 conditions", title=f"{model.capitalize()} BF16 depth delta")
    ax.set_yscale("symlog")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_pairs(model: str, pairs: list[dict], output: Path) -> None:
    chosen = [row for row in pairs if row["model"] == model and row["scope"] == "model"]
    by_gap = {row["canonical_depth_gap"]: row for row in chosen}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bottoms = np.zeros(len(GAPS))
    for field, label, color in (("preserved_rate", "Preserved", "#31826a"), ("tie_rate", "Collapsed to tie", "#e7b454"), ("inversion_rate", "Inverted", "#c04b52")):
        values = np.array([float(by_gap[gap][field] or 0) for gap in GAPS])
        ax.bar(GAPS, values, bottom=bottoms, label=label, color=color)
        bottoms += values
    for index, gap in enumerate(GAPS):
        n = int(by_gap[gap]["comparable_pairs"])
        ax.text(index, min(bottoms[index] + .02, 1.02), f"n={n}" if n else "NA", ha="center", fontsize=8)
    ax.set(xlabel="Canonical B1 depth gap", ylabel="Pair fraction", ylim=(0, 1.12), title=f"{model.capitalize()} within-step order preservation")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def analyze(root: Path, output: Path, *, bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    hist, steps, pairs, positions, rank_summary_rows, delta_summaries = [], [], [], [], [], []
    input_manifest, model_facts = {}, {}
    for model in MODELS:
        rows, summaries, manifest, hashes = validate_receipt(model, root)
        input_manifest[model] = {"files": hashes, "rows": len(rows), "condition_rows": len(summaries), "source_sha": manifest["source_sha"], "model_revision": manifest["model_revision"]}
        groups = defaultdict(list)
        for row in rows:
            groups[condition_key(row)].append(row)
        model_pool_rows, model_pools = [], []
        all_pair_counts = {gap: Counter() for gap in GAPS}
        by_position = defaultdict(dict)
        for condition in sorted(groups):
            width, order = condition
            part = groups[condition]
            margin_bins = sorted({row["margin_bin"] for row in part})
            delta_summaries.append(delta_summary(part, "condition", model, width, order, "ALL"))
            for margin_bin in margin_bins:
                delta_summaries.append(delta_summary([row for row in part if row["margin_bin"] == margin_bin], "margin_bin", model, width, order, margin_bin))
            for kind in ("signed", "absolute"):
                counts = Counter((integer(row, "live_depth") - integer(row, "reference_depth")) if kind == "signed" else abs(integer(row, "live_depth") - integer(row, "reference_depth")) for row in part)
                for delta, count in sorted(counts.items()):
                    hist.append({"scope": "condition", "model": model, "B": width, "order": order, "histogram_kind": kind, "delta": delta, "count": count, "fraction": count / len(part)})
            condition_counts = {gap: Counter() for gap in GAPS}
            condition_pools = []
            condition_step_rows = []
            for step in range(int(manifest["decode_positions"])):
                pool = sorted((row for row in part if integer(row, "token_idx") == step), key=lambda row: integer(row, "request_id"))
                metric = rank_metrics(pool)
                step_row = {"model": model, "B": width, "order": order, "token_idx": step, **metric}
                steps.append(step_row)
                condition_step_rows.append(step_row)
                model_pool_rows.append(step_row)
                condition_pools.append(pool)
                model_pools.append(pool)
                for gap, counts in pair_counts(pool).items():
                    condition_counts[gap].update(counts)
                    all_pair_counts[gap].update(counts)
            for gap, counts in condition_counts.items():
                pairs.append(pair_summary(model, width, order, "condition", gap, counts))
            ci = cluster_bootstrap_ci(condition_pools, manifest["orderings"]["fifo"], reps=bootstrap_reps, seed=BOOTSTRAP_SEED)
            saved = next(row for row in summaries if condition_key(row) == condition)
            canonical_work = integer(saved, "reference_B1_work")
            live_work = integer(saved, "actual_live_work")
            rank_summary_rows.append({"scope": "condition", "model": model, "B": width, "order": order,
                                      **summarize_rank(condition_step_rows, ci),
                                      "positions": len(part), "depth_flips": sum(integer(row, "depth_flip") for row in part),
                                      "depth_flip_rate": sum(integer(row, "depth_flip") for row in part) / len(part),
                                      "canonical_B1_work": canonical_work, "actual_live_work": live_work,
                                      "live_work_delta": live_work - canonical_work,
                                      "live_work_pct_delta": 100 * (live_work - canonical_work) / canonical_work,
                                      "actual_live_RIC": integer(saved, "actual_live_RIC"),
                                      "frozen_B1_depth_grouping_RIC": integer(saved, "frozen_B1_depth_grouping_RIC")})
            for row in part:
                by_position[identity_key(row)][condition] = integer(row, "live_depth")
        delta_summaries.append(delta_summary(rows, "model", model, None, None, "ALL"))
        for margin_bin in sorted({row["margin_bin"] for row in rows}):
            delta_summaries.append(delta_summary([row for row in rows if row["margin_bin"] == margin_bin], "model_margin_bin", model, None, None, margin_bin))
        for kind in ("signed", "absolute"):
            counts = Counter((integer(row, "live_depth") - integer(row, "reference_depth")) if kind == "signed" else abs(integer(row, "live_depth") - integer(row, "reference_depth")) for row in rows)
            for delta, count in sorted(counts.items()):
                hist.append({"scope": "model", "model": model, "B": None, "order": None, "histogram_kind": kind, "delta": delta, "count": count, "fraction": count / len(rows)})
        for gap, counts in all_pair_counts.items():
            pairs.append(pair_summary(model, None, None, "model", gap, counts))
        for key in sorted(by_position):
            depths = by_position[key]
            if len(depths) != len(groups):
                raise ValueError(f"Incomplete live-depth condition intersection: {model}/{key}")
            values = list(depths.values())
            reference = next(integer(row, "reference_depth") for row in rows if identity_key(row) == key)
            position = {"model": model, "token_idx": key[0], "request_id": key[1], "canonical_B1_depth": reference,
                        "min_live_depth": min(values), "max_live_depth": max(values), "live_depth_range": max(values) - min(values)}
            for width in manifest["B_values"]:
                values_b = [depths[width, name] for name in manifest["orderings"]]
                position[f"same_B{width}_order_range"] = max(values_b) - min(values_b)
            for name in manifest["orderings"]:
                values_o = [depths[width, name] for width in manifest["B_values"]]
                position[f"cross_B_{name}_range"] = max(values_o) - min(values_o)
            positions.append(position)
        model_ci = cluster_bootstrap_ci(model_pools, manifest["orderings"]["fifo"], reps=bootstrap_reps, seed=BOOTSTRAP_SEED)
        condition_summaries = [row for row in rank_summary_rows if row["model"] == model and row["scope"] == "condition"]
        flips = [row["depth_flips"] for row in condition_summaries]
        work_deltas = [row["live_work_delta"] for row in condition_summaries]
        correlation = float(pearsonr(flips, work_deltas).statistic) if len(set(flips)) > 1 and len(set(work_deltas)) > 1 else None
        model_positions = [row for row in positions if row["model"] == model]
        ranges = Counter(row["live_depth_range"] for row in model_positions)
        rank_summary_rows.append({"scope": "model", "model": model, "B": None, "order": None,
                                  **summarize_rank(model_pool_rows, model_ci),
                                  "positions": len(rows), "depth_flips": sum(integer(row, "depth_flip") for row in rows),
                                  "depth_flip_rate": sum(integer(row, "depth_flip") for row in rows) / len(rows),
                                  "unique_positions": len(by_position), "unique_depth_flipped_positions": sum(any(integer(row, "live_depth") != integer(row, "reference_depth") for row in rows if identity_key(row) == key) for key in by_position),
                                  "range_0": ranges[0], "range_1": ranges[1], "range_2": ranges[2],
                                  "range_ge_3": sum(count for value, count in ranges.items() if value >= 3),
                                  "max_live_depth_range": max(ranges),
                                  "flip_count_vs_work_delta_pearson_r": correlation,
                                  "correlation_condition_count": len(condition_summaries)})
        model_facts[model] = {"receipt_rows": len(rows), "conditions": len(groups), "token_pools": len(model_pools), "unique_positions": len(by_position)}
    specs = {
        "depth_delta_histogram.csv": (hist, ["scope", "model", "B", "order", "histogram_kind", "delta", "count", "fraction"]),
        "depth_delta_summary.csv": (delta_summaries, list(delta_summaries[0])),
        "rank_stability_by_step.csv": (steps, list(steps[0])),
        "pairwise_gap_stability.csv": (pairs, list(pairs[0])),
        "position_depth_range.csv": (positions, list(positions[0])),
        "rank_stability_summary.csv": (rank_summary_rows, list(rank_summary_rows[0]) + [key for key in rank_summary_rows[-1] if key not in rank_summary_rows[0]]),
    }
    for name, (data, fields) in specs.items():
        write_csv(output / name, data, fields)
    for model in MODELS:
        plot_delta(model, hist, output / f"{model}_depth_delta_histogram.png")
        plot_pairs(model, pairs, output / f"{model}_pairwise_gap_preservation.png")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    context_paths = (
        "AGENTS.md", "phase04/DEPTH_RANK_STABILITY_CPU_PLAN.md",
        "phase04/DEPTH_RANK_STABILITY_AGENT_TASK.md",
        "phase04/results/20260917T060824Z_50910e6_bf16_resolution/CAUSE_REVIEW.md",
        "phase04/results/20260917T060824Z_50910e6_bf16_resolution/PROPOSED_B2_CONTRACT.md",
        "phase01/EXPERIMENT_REVIEW_PROTOCOL.md",
    )
    manifest = {
        "status": "CPU_SAVED_DATA_ANALYSIS_COMPLETE", "execution_kind": "saved-data-analysis",
        "source_sha": source_sha, "source_file_sha256": sha256(Path(__file__)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "exact_command": shlex.join(sys.orig_argv), "gpu_execution": False,
        "bootstrap": {"unit": "request ID jointly resampled across token pools and repeated B/order conditions", "seed": BOOTSTRAP_SEED, "repetitions": bootstrap_reps, "interval": "2.5%-97.5% percentile of median per-pool Kendall tau-b"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "inputs": input_manifest, "integrity": model_facts,
        "context_files": {path: sha256(REPO / path) for path in context_paths},
        "outputs": {name: {"rows": len(data), "sha256": sha256(output / name)} for name, (data, _) in specs.items()},
    }
    manifest["figures"] = {path.name: sha256(path) for path in sorted(output.glob("*.png"))}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    manifest = analyze(args.input_root, args.output_dir, bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({"status": manifest["status"], "source_sha": manifest["source_sha"], "integrity": manifest["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
