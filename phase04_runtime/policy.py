"""Pure, shared early-exit arithmetic for collection, reference and compaction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExitDecision:
    mask: torch.Tensor
    margin: torch.Tensor
    score: torch.Tensor
    reason: str


def _check(depth: int, rmax: int, threshold: float) -> None:
    if not 1 <= depth <= rmax or not 0.0 < threshold < 1.0:
        raise ValueError("depth/Rmax or exit threshold outside supported range")


def huginn_decision(before: torch.Tensor, after: torch.Tensor, depth: int,
                    rmax: int, threshold: float, *, validate_finite: bool = True) -> ExitDecision:
    _check(depth, rmax, threshold)
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("Huginn states must have matching [B,S,H] shapes")
    before_f, after_f = before.float(), after.float()
    score = ((after_f - before_f).norm(dim=-1) /
             after_f.norm(dim=-1).clamp_min(1e-12)).mean(dim=-1)
    if validate_finite and not bool(torch.isfinite(score).all()):
        raise ValueError("nonfinite Huginn convergence score")
    threshold_pass = score < threshold
    mask = threshold_pass if depth < rmax else torch.ones_like(threshold_pass)
    return ExitDecision(mask=mask, margin=threshold - score, score=score,
                        reason="threshold_or_rmax_cap" if depth == rmax else "threshold")


def ouro_decision(gate_logit: torch.Tensor, remaining: torch.Tensor, depth: int,
                  rmax: int, threshold: float, *, validate_finite: bool = True
                  ) -> tuple[ExitDecision, torch.Tensor]:
    _check(depth, rmax, threshold)
    logits = gate_logit.float().reshape(-1)
    previous = remaining.float().reshape(-1)
    if logits.shape != previous.shape:
        raise ValueError("Ouro gate and remaining probability width mismatch")
    if validate_finite and not bool((torch.isfinite(logits) & torch.isfinite(previous)).all()):
        raise ValueError("nonfinite Ouro gate or remaining probability")
    hazard = torch.sigmoid(logits)
    next_remaining = previous * (1.0 - hazard)
    cumulative = 1.0 - next_remaining
    threshold_pass = cumulative >= threshold
    mask = threshold_pass if depth < rmax else torch.ones_like(threshold_pass)
    return (ExitDecision(mask=mask, margin=cumulative - threshold, score=cumulative,
                         reason="threshold_or_rmax_cap" if depth == rmax else "threshold"),
            next_remaining)
