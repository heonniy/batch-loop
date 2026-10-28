"""Official model-family exit criteria and immutable recorded-replay trace format.

This module deliberately does *not* claim to implement variable-depth attention.
A separate runtime must actually skip recurrences and handle missing historical KV.
Ouro's Q-exit and Huginn's official convergence evaluators are different mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import hashlib
import json
import math


@dataclass(frozen=True)
class ExitRule:
    family: str
    criterion: str
    threshold: float
    min_depth: int = 1

    def __post_init__(self):
        accepted = {"ouro": {"q-exit"}, "huginn": {"cosine", "argmax-stability", "kl"}}
        if self.family not in accepted or self.criterion not in accepted[self.family]:
            raise ValueError("Unsupported official evaluator")
        if not (0 < self.min_depth):
            raise ValueError("min_depth must be >=1")
        if self.criterion == "q-exit" and not (0 < self.threshold <= 1):
            raise ValueError("Q-exit threshold must be in (0,1]")
        if self.criterion != "q-exit" and self.threshold < 0:
            raise ValueError("Convergence threshold cannot be negative")


def select_depth(rule: ExitRule, observations: Sequence[float], max_depth: int) -> int:
    """Select first eligible depth, or max_depth. observations indexed by loop 1..R.

    ouro q-exit: observations = official ACT *cumulative probabilities*, not a
      freshly invented gate. With q=1 always run to R (even if floating-point
      sums round to 1 early).
    huginn cosine: observations = 1 - mean cosine(latents_r, latents_{r-1})
      as in official CosineExitEvaluator (default threshold 1e-3).
    huginn KL: observations = official log-probability KL evaluator values.
    huginn argmax-stability: observations = consecutive unchanged argmax count,
      with the first recurrence count 0 (default threshold 5).
    """
    if len(observations) != max_depth:
        raise ValueError("Must supply one observation per executed recurrence")
    if rule.min_depth > max_depth:
        raise ValueError("min_depth > max_depth")
    if rule.criterion == "q-exit" and rule.threshold == 1:
        return max_depth
    for i, value in enumerate(observations, start=1):
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite gate statistic")
        if i < rule.min_depth:
            continue
        stop = (value >= rule.threshold if rule.criterion in ("q-exit", "argmax-stability")
                else value < rule.threshold)
        if stop:
            return i
    return max_depth


def recorded_trace(model_revision: str, rule: ExitRule, uid: str, token_ids: Sequence[int],
                   depths: Sequence[int], max_depth: int, policy: str) -> dict:
    """Freeze exit depths by *token position*; reject ambiguous non-matching traces."""
    if len(token_ids) != len(depths):
        raise ValueError("Each traced token must have one depth")
    if any(not 1 <= d <= max_depth for d in depths):
        raise ValueError("Exit depth out of range")
    serial = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return {"model_revision": model_revision, "family": rule.family,
            "exit_criterion": rule.criterion, "exit_threshold": rule.threshold,
            "min_depth": rule.min_depth, "max_depth": max_depth,
            "reference_policy": policy, "uid": uid,
            "token_ids_sha256": hashlib.sha256(serial).hexdigest(),
            "token_count": len(token_ids), "depths": list(depths)}


def replay_depth(trace: dict, token_position: int, expected_revision: str,
                 expected_uid: str) -> int:
    """No silent extension to an unseen generated token; fail closed."""
    if trace["model_revision"] != expected_revision or trace["uid"] != expected_uid:
        raise ValueError("Trace identity/revision mismatch")
    if not (0 <= token_position < len(trace["depths"])):
        raise IndexError("Exit trace exhausted: define replay continuation semantics")
    return int(trace["depths"][token_position])
