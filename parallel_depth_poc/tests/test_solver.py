"""CPU-only mathematical tests. No pretrained weights, GPU or network needed."""

import pytest

torch = pytest.importorskip("torch")

from parallel_depth_poc.solver import (
    init_depth_states,
    jacobi_depth,
    normalized_error,
    parareal_blocks,
    serial_trajectory,
)


def depth_affine(states, depths):
    """A deterministic nonstationary F_r with true [R,B,H] vectorization."""
    return 0.78 * states + depths[:, None, None].to(states.dtype) * 0.035 + 0.06


def test_exact_serial_matches_manual():
    x = torch.tensor([[0.2, -0.1], [0.3, 0.8]], dtype=torch.float64)
    y = serial_trajectory(x, 6, depth_affine)
    expected = [x]
    for r in range(1, 7):
        expected.append(0.78 * expected[-1] + r * 0.035 + 0.06)
    torch.testing.assert_close(y, torch.stack(expected), atol=1e-12, rtol=0)


def test_jacobi_exact_after_at_most_depth_rounds():
    x = torch.tensor([[0.3, -0.2], [0.0, 0.4]], dtype=torch.float64)
    ref = serial_trajectory(x, 7, depth_affine)
    init = init_depth_states(x, 7, depth_affine, scheme="linear")
    assert normalized_error(init[-1], ref[-1]).amax() > 0.001
    got = jacobi_depth(x, 7, depth_affine, rounds=7)
    torch.testing.assert_close(got.trajectory, ref, atol=1e-12, rtol=0)
    assert len(got.max_update) == 7
    assert got.fine_calls_per_round == [7] * 7


def test_jacobi_batch_dispatches_all_depths_together():
    widths = []

    def observed(states, ids):
        widths.append(len(ids))
        return depth_affine(states, ids)

    x = torch.ones((2, 3), dtype=torch.float32)
    jacobi_depth(x, 5, observed, rounds=2)
    # Initial h1 is a single-row call; subsequent rounds each dispatch 5.
    assert widths == [1, 5, 5]


def test_parareal_identity_coarse_recovers_reference():
    x = torch.tensor([[0.2, -0.1], [0.4, 0.9]], dtype=torch.float64)
    ref = serial_trajectory(x, 6, depth_affine)
    result = parareal_blocks(x, 6, 2, depth_affine, rounds=3)
    torch.testing.assert_close(result.trajectory[-1], ref[-1], atol=1e-11, rtol=0)
    assert result.fine_calls_per_round == [6, 6, 6]


def test_parareal_fine_evaluations_are_depth_batched():
    widths = []

    def observed(states, ids):
        widths.append(len(ids))
        return depth_affine(states, ids)

    parareal_blocks(torch.ones(1, 2), 8, 2, observed, rounds=2)
    # Four blocks, two fine calls per correction round.
    assert widths == [4] * 4


def test_invalid_shapes_and_depths_fail():
    x = torch.ones(2, 3)
    with pytest.raises(ValueError):
        jacobi_depth(x, 4, depth_affine, rounds=-1)
    with pytest.raises(ValueError):
        parareal_blocks(x, 5, 2, depth_affine, rounds=1)
    with pytest.raises(ValueError):
        init_depth_states(x, 4, depth_affine, scheme="magic")
    with pytest.raises(ValueError):
        normalized_error(x, torch.ones(3, 3))
