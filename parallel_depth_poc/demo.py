"""Deterministic CPU toy demonstration, NOT measured model speedups."""

import json
import torch

from .solver import (
    jacobi_depth,
    normalized_error,
    parareal_blocks,
    serial_trajectory,
)


def toy_step(states: torch.Tensor, depth_ids: torch.Tensor) -> torch.Tensor:
    # Depth-conditioned non-stationary affine recurrence. The [R,B,H] API
    # performs all same-round evaluations together.
    d = depth_ids.to(states.dtype)[:, None, None]
    return 0.83 * states + 0.04 * d + 0.07


def main() -> None:
    h0 = torch.tensor([[0.4, -0.1], [0.2, 0.6]], dtype=torch.float64)
    depth = 8
    ref = serial_trajectory(h0, depth, toy_step)
    rows = []
    for rounds in [0, 1, 2, 4, 8]:
        jacobi = jacobi_depth(h0, depth, toy_step, rounds=rounds)
        rows.append({
            "solver": "jacobi",
            "rounds": rounds,
            "final_relative_l2_max": float(
                normalized_error(jacobi.trajectory[-1], ref[-1]).max()
            ),
            "fine_evaluations_not_latency": sum(jacobi.fine_calls_per_round),
        })
    for rounds in [0, 1, 2, 4]:
        para = parareal_blocks(h0, depth, 2, toy_step, rounds=rounds)
        rows.append({
            "solver": "parareal_identity_coarse",
            "rounds": rounds,
            "final_relative_l2_max": float(
                normalized_error(para.trajectory[-1], ref[-1]).max()
            ),
            "fine_evaluations_not_latency": sum(para.fine_calls_per_round),
        })
    print(json.dumps({
        "status": "synthetic_cpu_only",
        "model": "toy_affine_depth_operator",
        "warning": "No Huginn/Ouro weights, KV, GPU timings or quality claims",
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
