"""Predictor and correction kernels for physically batched depth-indexed step_many.

These solvers do NOT own KV; the caller must supply isolation and explicit
commit semantics. Candidate outputs at one token are diagnostic only until
the engine proves cache correctness. Exact AR future states must never be
passed to a predictor, except its authorized warmup prefix.
"""
from __future__ import annotations

from collections import deque

import torch

PREDICTORS = ("constant", "linear", "geometric", "secant", "prev_shift")
CORRECTORS = ("jacobi", "damped", "newton_identity", "newton_scaled", "anderson")
ALLOCATIONS = ("full", "suffix", "prefix", "alternating", "residual_topk", "random")


def predict(prefix, depth, method, prev=None, clip=1.5):
    """Exact observed prefix [m+1,B,H] -> predicted [D+1,B,H].

    m≥1 and m<=D, no oracle future states. Delta models use only available
    prefix and previous-token trajectory (which must be committed or labeled).
    """
    if prefix.ndim != 3 or not 1 <= prefix.shape[0] - 1 <= depth:
        raise ValueError("prefix must be [m+1,B,H], 1<=m<=D")
    if method not in PREDICTORS:
        raise ValueError("unknown predictor " + method)
    m = prefix.shape[0] - 1
    last = prefix[m]
    delta = prefix[m] - prefix[m - 1]
    if method == "geometric" and m < 2:
        raise ValueError("geometric prediction needs >=2 true recurrent transitions")
    if method == "prev_shift" and (prev is None or prev.shape != (depth + 1, *last.shape)):
        raise ValueError("prev_shift requires previous token trajectory [D+1,B,H]")
    if method == "geometric":
        v1 = (prefix[m - 1] - prefix[m - 2]).float()
        v2 = delta.float()
        alpha = (v1 * v2).sum(-1) / v1.square().sum(-1).clamp_min(1e-6)
        alpha = alpha.clamp(-clip, clip).to(last.dtype)[:, None]
    else:
        alpha = None
    states = list(prefix.unbind(0))
    for r in range(m + 1, depth + 1):
        j = r - m
        if method == "constant":
            state = last
        elif method == "linear":
            state = last + j * delta
        elif method == "secant":
            # Last observed secant slope, not the long h0-to-h1 slope.
            state = last + j * delta
        elif method == "geometric":
            step = torch.zeros_like(last)
            power = alpha.clone()
            for _ in range(j):
                step = step + power * delta
                power = power * alpha
            state = last + step
        elif method == "prev_shift":
            state = prev[r] + last - prev[m]
        states.append(state.clone())
    return torch.stack(states)


def select_depths(depth, anchor, policy="full", width=0, iteration=0, activity=None):
    """One shared selected depth set across the cohort; fixed per-depth KV bank.

    Non-recomputed rows are provisional. Enforce the last depth so coda has an
    updated candidate, but never interpret this as cache correctness.
    """
    if not 1 <= anchor <= depth or policy not in ALLOCATIONS:
        raise ValueError("bad anchor or policy")
    eligible = list(range(anchor + 1, depth + 1))
    if not eligible:
        return []
    w = max(1, min(int(width or len(eligible)), len(eligible)))
    if policy == "full":
        chosen = eligible
    elif policy == "suffix":
        chosen = eligible[-w:]
    elif policy == "prefix":
        chosen = eligible[:w]
    elif policy == "alternating":
        chosen = eligible[(iteration % 2)::2][:w]
    elif policy == "random":
        # equal-row random control (seeded by iteration; same width as the compared policy)
        import random as _random
        chosen = sorted(_random.Random(1000003 * depth + iteration).sample(eligible, w))
    else:
        if activity is None or len(activity) != depth + 1:
            raise ValueError("residual_topk requires activity [D+1]")
        chosen = sorted(eligible, key=lambda r: (-float(activity[r]), r))[:w]
    if depth not in chosen:
        chosen = (chosen[:-1] if len(chosen) >= w else chosen) + [depth]
    return sorted(set(chosen))


class CorrectionStepper:
    """Keeps history for one predictor/solver/token; no training and no KV commit.

    Corrector Newton uses cheap *triangular* predecessor updates:
        z_new[r] = F_r(z_old[r-1]) + a*(z_new[r-1]-z_old[r-1]).
    If r is not active, provisional z_r is retained. The low-dimensional
    update is computed in float32 and cast back only at the end. Identity or
    scalar A is **not a true Jacobian** unless independently validated.
    """
    def __init__(self, depth, anchor, method, step_many, *,
                 omega=0.75, scale=1.0, history=4, damping=1.0, ridge=1e-3):
        if method not in CORRECTORS:
            raise ValueError("unknown corrector")
        if depth < anchor or anchor < 1 or history < 1 or ridge <= 0:
            raise ValueError("invalid numerical configuration")
        self.depth, self.anchor, self.method = depth, anchor, method
        self.step_many = step_many
        self.omega, self.scale, self.damping, self.ridge = omega, scale, damping, ridge
        self.history = deque(maxlen=history)
        self.eval_rows = 0
        self.calls = 0
        self.iteration = 0

    def _anderson(self, old, candidate):
        # Solve batched constrained least squares, distinct coefficients per
        # request row. Residual vectors cover candidate depths only.
        residual = candidate - old
        self.history.append((candidate.detach().float(), residual.detach().float()))
        if len(self.history) < 2:
            return candidate
        candidates = torch.stack([x[0] for x in self.history], dim=0)
        residuals = torch.stack([x[1] for x in self.history], dim=0)
        m, r, b, h = candidates.shape
        mat = residuals.permute(2, 0, 1, 3).reshape(b, m, r * h)
        gram = torch.matmul(mat, mat.transpose(-1, -2))
        eye = torch.eye(m, device=gram.device, dtype=gram.dtype).expand(b, m, m)
        reg = (gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-9)
               * self.ridge + 1e-7)[:, None, None]
        gram = gram + reg * eye
        ones = torch.ones((b, m, 1), device=gram.device, dtype=gram.dtype)
        try:
            unnorm = torch.linalg.solve(gram, ones)
            alpha = unnorm / unnorm.sum(1, keepdim=True).clamp_min(1e-6)
        except RuntimeError:
            return candidate
        # Reject wildly extrapolated coefficients; fail closed to single step.
        if not torch.isfinite(alpha).all() or alpha.abs().max() > 6:
            return candidate
        w = alpha.squeeze(-1).T[:, None, :, None]
        merged = (candidates * w).sum(0).to(candidate.dtype)
        return ((1 - self.damping) * candidate.float()
                + self.damping * merged.float()).to(candidate.dtype)

    def step(self, z, *, policy="full", width=0, activity=None):
        if z.ndim != 3 or len(z) != self.depth + 1:
            raise ValueError("state shape must be [D+1,B,H]")
        ds = select_depths(self.depth, self.anchor, policy, width, self.iteration, activity)
        self.iteration += 1
        if not ds:
            return z.clone(), torch.zeros(len(z), device=z.device)
        ids = torch.tensor(ds, dtype=torch.long, device=z.device)
        outs = self.step_many(torch.stack([z[d - 1] for d in ds]), ids)
        if tuple(outs.shape) != (len(ds), *tuple(z.shape[1:])):
            raise ValueError("step_many emitted wrong shape")
        self.calls += 1
        self.eval_rows += len(ds) * z.shape[1]
        provisional = z.clone()
        if self.method in ("jacobi", "anderson"):
            for j, d in enumerate(ds):
                provisional[d] = outs[j]
            if self.method == "anderson":
                # History extrapolates whole candidate trajectory but only
                # active F evaluations were made: still an approximate arm.
                provisional = self._anderson(z, provisional)
        elif self.method == "damped":
            for j, d in enumerate(ds):
                provisional[d] = (z[d].float() * (1 - self.omega)
                                  + outs[j].float() * self.omega).to(z.dtype)
        else:
            mapping = {d: j for j, d in enumerate(ds)}
            for d in range(self.anchor + 1, self.depth + 1):
                if d not in mapping:
                    continue
                delta = provisional[d - 1].float() - z[d - 1].float()
                a = 1.0 if self.method == "newton_identity" else self.scale
                provisional[d] = (outs[mapping[d]].float() + a * delta).to(z.dtype)
        # Keep truly executed anchor prefix independent of history.
        provisional[:self.anchor + 1] = z[:self.anchor + 1]
        changes = ((provisional.float() - z.float()).norm(dim=-1)
                   / provisional.float().norm(dim=-1).clamp_min(1e-8))
        return provisional, changes.amax(-1)
