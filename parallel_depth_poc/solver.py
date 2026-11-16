"""Reference numerical depth-parallel solvers; NOT a model-specific GPU backend.

All states have shape [B, H]. A batched recurrent operator receives:
    states: [R, B, H]
    depth_ids: [R]  (1-based depth indices)
and returns [R, B, H].

For a real Huginn/Ouro adapter, the operator MUST use the correct prefix,
attention mask, cache-bank and recurrent-depth semantics independently for each
row. These reference solvers do not create or commit any KV caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

Tensor = torch.Tensor
StepMany = Callable[[Tensor, Tensor], Tensor]
CoarseMany = Callable[[Tensor, Tensor], Tensor]


@dataclass
class SolverResult:
    """Final states and per-iteration *diagnostic* update magnitudes.

    `max_update` measures changes between iterations, NOT the true recurrent
    residual and NOT next-token quality. It is unsafe as a sole acceptance gate.
    """

    trajectory: Tensor  # [D+1,B,H] for Jacobi; [num_blocks+1,B,H] for Parareal
    max_update: list[float]
    fine_calls_per_round: list[int]
    coarse_calls_per_round: list[int]


def _check_state(h0: Tensor) -> None:
    if h0.ndim != 2:
        raise ValueError(f"expected initial hidden [B,H], got {tuple(h0.shape)}")
    if not torch.is_floating_point(h0):
        raise TypeError("hidden states must be floating point")


def _depth_ids(depths: list[int], h0: Tensor) -> Tensor:
    return torch.tensor(depths, device=h0.device, dtype=torch.long)


def _call(fn: StepMany, states: Tensor, ids: Tensor) -> Tensor:
    if states.ndim != 3 or ids.ndim != 1 or states.shape[0] != len(ids):
        raise ValueError("expected [R,B,H] states and [R] IDs")
    out = fn(states, ids)
    if not isinstance(out, torch.Tensor) or out.shape != states.shape:
        raise ValueError("batched operator must preserve [R,B,H] shape")
    return out


def serial_trajectory(h0: Tensor, depth: int, step_many: StepMany) -> Tensor:
    """Exact serial composition of the supplied depth-indexed operator.

    'Exact' here means *for this adapter and its cache conditions*, not
    necessarily bitwise equal to a model's author-provided generation path.
    """
    _check_state(h0)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    states = [h0]
    for r in range(1, depth + 1):
        nxt = _call(step_many, states[-1].unsqueeze(0), _depth_ids([r], h0))[0]
        states.append(nxt)
    return torch.stack(states)


def init_depth_states(
    h0: Tensor, depth: int, step_many: StepMany, scheme: str = "linear"
) -> Tensor:
    """Training-free starting trajectory, with the first step truly evaluated."""
    _check_state(h0)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    h1 = _call(step_many, h0.unsqueeze(0), _depth_ids([1], h0))[0]
    states = [h0, h1]
    delta = h1 - h0
    for r in range(2, depth + 1):
        if scheme == "constant":
            states.append(h1.clone())
        elif scheme == "linear":
            states.append(h1 + (r - 1) * delta)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
    return torch.stack(states)


def jacobi_depth(
    h0: Tensor,
    depth: int,
    step_many: StepMany,
    *,
    rounds: int,
    init: str = "linear",
    initial_states: Optional[Tensor] = None,
) -> SolverResult:
    """Synchronous Jacobi recurrence on all depths.

    z_0^(k)=h0; z_r^(k+1)=F_r(z_(r-1)^(k)).
    All D recurrent calls within a round are supplied to ONE batch API call.
    Runtime claims still require a real GPU adapter that physically batches it.

    Initial states may be supplied for an oracle/controlled diagnostic but MUST
    not contain reference future states for a deployable inference comparison.
    """
    _check_state(h0)
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if initial_states is None:
        z = init_depth_states(h0, depth, step_many, scheme=init)
    else:
        if initial_states.shape != (depth + 1, *h0.shape):
            raise ValueError("initial_states must be [D+1,B,H]")
        z = initial_states.clone()
        z[0] = h0
    ds = torch.arange(1, depth + 1, device=h0.device, dtype=torch.long)
    updates = []
    for _ in range(rounds):
        nxt = _call(step_many, z[:-1], ds)
        new_z = torch.cat([h0.unsqueeze(0), nxt], dim=0)
        updates.append(float((new_z - z).abs().amax().item()))
        z = new_z
    return SolverResult(
        trajectory=z,
        max_update=updates,
        fine_calls_per_round=[depth] * rounds,
        coarse_calls_per_round=[0] * rounds,
    )


def identity_coarse(states: Tensor, block_ids: Tensor) -> Tensor:
    """Control G=identity; cheap but often a weak approximation."""
    del block_ids
    return states


def parareal_blocks(
    h0: Tensor,
    depth: int,
    block_size: int,
    step_many: StepMany,
    *,
    rounds: int,
    coarse_many: CoarseMany = identity_coarse,
) -> SolverResult:
    """Reference block Parareal with uniform block length.

    Fine solves inside a block are serial; DIFFERENT blocks are vectorized in
    one step_many call per local depth. Coarse corrections propagate serially
    across block boundaries. Not yet a Huginn/Ouro adapter.

    q_(j+1)^(k+1) = G_j(q_j^(k+1))
                      + Fblock_j(q_j^(k)) - G_j(q_j^(k))

    With k >= num_blocks, this matches the serial reference to numerical
    tolerance (assuming deterministic operators).
    """
    _check_state(h0)
    if depth < 1 or block_size < 1 or depth % block_size:
        raise ValueError("depth must be a positive multiple of block_size")
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    num_blocks = depth // block_size
    ids = _depth_ids(list(range(num_blocks)), h0)

    def coarse(states: Tensor, block_ids: Tensor) -> Tensor:
        return _call(coarse_many, states, block_ids)

    # Serial cheap coarse initialization.
    boundaries = [h0]
    for j in range(num_blocks):
        q = coarse(boundaries[-1].unsqueeze(0), ids[j:j+1])[0]
        boundaries.append(q)
    q = torch.stack(boundaries)
    updates: list[float] = []

    for _ in range(rounds):
        # Block j starts with previous round's q_j. Across blocks the fine
        # operations run batched; within blocks we advance recurrent depth.
        work = q[:-1]
        for local in range(block_size):
            d = _depth_ids(
                [j * block_size + local + 1 for j in range(num_blocks)], h0
            )
            work = _call(step_many, work, d)
        fine_old = work
        coarse_old = coarse(q[:-1], ids)

        newer = [h0]
        for j in range(num_blocks):
            fresh = coarse(newer[-1].unsqueeze(0), ids[j:j+1])[0]
            newer.append(fresh + fine_old[j] - coarse_old[j])
        new_q = torch.stack(newer)
        updates.append(float((new_q - q).abs().amax().item()))
        q = new_q

    return SolverResult(
        trajectory=q,
        max_update=updates,
        fine_calls_per_round=[depth] * rounds,
        coarse_calls_per_round=[num_blocks * 2] * rounds,
    )


def normalized_error(candidate: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-row relative L2 error, along last dimension."""
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference shapes must match")
    numer = torch.linalg.vector_norm((candidate - reference).float(), dim=-1)
    denom = torch.linalg.vector_norm(reference.float(), dim=-1).clamp_min(eps)
    return numer / denom
