"""Offline trajectory-predictability analysis for recorded fixed-depth states.

Input is an explicitly produced .pt torch file loaded with weights_only=True.
Required key: states [T, D+1, B, H] (or [D+1, B, H] for T=1).
This script does NOT run a model or measure physical speed. It compares
training-free guesses to the *recorded reference* only after generation.

Each T index must correspond to the same request cohort at one decode step.
Do not mix prompt tokens/teacher-forced positions indiscriminately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .solver import normalized_error


def _shape(states: torch.Tensor) -> torch.Tensor:
    if states.ndim == 3:
        states = states.unsqueeze(0)
    if states.ndim != 4 or states.size(1) < 2:
        raise ValueError("states must be [T,D+1,B,H], D>=1")
    if not torch.is_floating_point(states):
        raise TypeError("states must have floating point type")
    return states.float()


def _summary(x: torch.Tensor) -> dict[str, float | int]:
    x = x.flatten().float()
    return {
        "count": x.numel(),
        "mean": float(x.mean()),
        "p50": float(torch.quantile(x, 0.5)),
        "p95": float(torch.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def evaluate_states(states: torch.Tensor) -> dict[str, Any]:
    """No future state is used to construct a deployable candidate.

    Oracle is the exact reference used only for error calculation.
    Previous-token predictor is available for t>=1 in a fixed ordered cohort,
    because it uses states recorded at t-1; this still assumes that previous
    token states were physically calculated (their cost must be counted).
    """
    states = _shape(states)
    t, d1, b, h = states.shape
    d = d1 - 1
    h0, h1 = states[:, 0], states[:, 1]
    delta = h1 - h0

    candidates: dict[str, torch.Tensor] = {
        "constant": torch.stack(
            [h0] + [h1 for _ in range(d)], dim=1
        ),
        "linear": torch.stack(
            [h0] + [h1 + (r - 1) * delta for r in range(1, d + 1)], dim=1
        ),
    }
    if t >= 2:
        # t-1 is a causal, already observed decode step, never future t+1.
        # Only evaluate positions t>=1 and report reduced sample count.
        previous = states[:-1]
        current = states[1:]
        prev_shift = current[:, 0] - previous[:, 0]
        candidates["previous_token"] = torch.stack(
            [current[:, 0]] +
            [previous[:, r] + prev_shift for r in range(1, d + 1)],
            dim=1,
        )

    results: dict[str, Any] = {
        "kind": "offline_hidden_prediction_only",
        "shape": {"token_steps": t, "depth": d, "batch": b, "hidden": h},
        "predictors": {},
        "caveat": (
            "Recorded reference states are used only for after-the-fact scoring. "
            "No logits, KV correctness, runtime speed or accuracy is established."
        ),
    }

    for name, candidate in candidates.items():
        ref = states
        if name == "previous_token":
            ref = states[1:]
        records = []
        for r in range(2, d + 1):
            relative = normalized_error(candidate[:, r], ref[:, r])
            a = candidate[:, r].reshape(-1, h)
            bvec = ref[:, r].reshape(-1, h)
            cosine = torch.nn.functional.cosine_similarity(a, bvec, dim=-1, eps=1e-8)
            records.append({
                "depth": r,
                "relative_l2": _summary(relative),
                "cosine": _summary(cosine),
            })
        results["predictors"][name] = {
            "evaluated_token_steps": int(ref.size(0)),
            "per_depth": records,
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = torch.load(args.trace, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "states" not in payload:
        raise ValueError("expected .pt dictionary with states tensor")
    report = evaluate_states(payload["states"])
    report["trace_filename"] = args.trace.name
    report["source_metadata"] = payload.get("metadata", {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "analyzed", "output": str(args.output), "shape": report["shape"]}))


if __name__ == "__main__":
    main()
