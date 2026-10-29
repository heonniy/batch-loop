"""Build a frozen <50%-FULL recurrent-KV protection ladder for Ouro.

All masks are STATIC layer x first-recurrence banks, applied to EVERY token.
This is a cheap test of the accuracy/memory Pareto curve, not a token-adaptive
method or true allocated-HBM benchmark.

The historical calibration uses LAST-CONVERT on disjoint prompts. Its ranking
is a transferable PRIOR and must not be described as strict-mode calibration.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

BUDGETS = (30, 35, 40, 45)
BASELINE_PERCENT = 25
SOURCES = {
    "ouro-1.4b": "results/20261026T080106Z_huginn_ouro_full_last_convert/e3/calibration_ouro-1.4b.json",
    "ouro-2.6b": "results/20261026T080106Z_huginn_ouro_full_last_convert/e3/calibration_ouro-2.6b.json",
}
RANDOM_SEED = 20261029


def bank_count(layer_count: int, budget_percent: int) -> int:
    """Max number of extra L0..L{L-1} recurrence-1 banks within budget.

    Ouro R4 FULL = 4L, LAST = L, each bank adds one KV plane for all
    tokens. Entire first recurrence protected = 2L / 4L = 50%.
    """
    if layer_count < 1 or not (BASELINE_PERCENT <= budget_percent <= 50):
        raise ValueError("Ouro R4 budget must be in [25,50]")
    return (budget_percent - BASELINE_PERCENT) * 4 * layer_count // 100


def construct(calibration: dict, model: str, base_registry: dict | None = None) -> tuple[dict, dict]:
    """Return registry and provenance. No dataset eval records used for selection."""
    layer_count = int(calibration["L"])
    if int(calibration["R"]) != 4 or layer_count not in (24, 48):
        raise ValueError("Unexpected Ouro geometry / R")
    if calibration["model"] != model or model not in SOURCES:
        raise ValueError("Calibration/model mismatch")
    candidates = [row for row in calibration["single_bank"] if int(row["step"]) == 0]
    if len(candidates) != layer_count:
        raise ValueError("Need a separate recurrence-1 sensitivity for each physical layer")
    seen = [int(row["layer"]) for row in candidates]
    if set(seen) != set(range(layer_count)):
        raise ValueError("Missing/duplicated first-recurrence calibration layers")
    ordered = sorted(candidates, key=lambda row: (-float(row["gain"]), int(row["layer"])))
    layer_order = [int(row["layer"]) for row in ordered]
    regs = {} if base_registry is None else dict(base_registry)
    meta = {
        "model": model, "R": 4, "layers": layer_count,
        "calibration_source_policy": "LAST-CONVERT (NOT strict)",
        "calibration_n_items": len(calibration.get("items", [])),
        "calibration_ids": [x["uid"] for x in calibration.get("items", [])],
        "ranked_layers": layer_order,
        "base_seed": RANDOM_SEED,
        "policies": {},
    }
    for pct in BUDGETS:
        k = bank_count(layer_count, pct)
        rng = random.Random(f"{RANDOM_SEED}:{model}:{pct}")
        selections = {
            f"bankTopB{pct}": layer_order[:k],
            f"bankPrefixB{pct}": list(range(k)),
            f"bankRandB{pct}": rng.sample(list(range(layer_count)), k),
        }
        for policy, layers in selections.items():
            assert len(layers) == len(set(layers)) == k
            regs[policy] = [[l, 0] for l in sorted(layers)]
            meta["policies"][policy] = {
                "n_banks": k, "layers": sorted(layers),
                "full_fraction": (layer_count + k) / (4 * layer_count),
                "last_relative_fraction": (layer_count + k) / layer_count,
                "requested_full_percent_ceiling": pct,
                "selection": policy[4:policy.index("B" + str(pct))],
            }
            if 100 * meta["policies"][policy]["full_fraction"] > pct + 1e-9:
                raise AssertionError("Budget exceeded")
    return regs, meta


def build_registry(kv_root: Path, base_json: Path) -> tuple[dict, dict]:
    original = json.loads(base_json.read_text())
    registry = {k: dict(v) for k, v in original.items()}
    provenance = {"reference_registry_sha256": hashlib.sha256(base_json.read_bytes()).hexdigest(),
                  "rank_source_sha256": {}, "by_model": {}}
    for model, rel in SOURCES.items():
        src = kv_root / rel
        body = src.read_bytes()
        calibration = json.loads(body)
        registry[model], meta = construct(calibration, model, registry.get(model))
        provenance["by_model"][model] = meta
        provenance["rank_source_sha256"][model] = hashlib.sha256(body).hexdigest()
    return registry, provenance


def assert_disjoint_from_eval(provenance: dict, run: Path, datasets: list[str]) -> None:
    """Calibration examples cannot overlap eval UID cohort."""
    for model, info in provenance["by_model"].items():
        cal = set(info["calibration_ids"])
        for dataset in datasets:
            path = run / "prompts" / f"{model}__{dataset}.jsonl"
            if not path.exists():
                continue
            eval_ids = {json.loads(line)["uid"] for line in path.read_text().splitlines()[1:]}
            overlap = cal & eval_ids
            if overlap:
                raise ValueError(f"Calibration contamination {model}/{dataset}: {sorted(overlap)[:4]}")
