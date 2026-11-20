"""Synthetic affine numerical tests; NOT pretrained model results."""
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from parallel_depth_search.numerics import (
        CorrectionStepper, predict, select_depths
    )


@unittest.skipUnless(torch is not None, "PyTorch not installed")
class NumericalTest(unittest.TestCase):
    def _states(self, D=5):
        h0 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        states = [h0]
        for r in range(1, D + 1):
            states.append(states[-1] + 1)
        return torch.stack(states)

    @staticmethod
    def step_many(x, ids):
        return x + 1

    def test_constant_from_h1(self):
        ref = self._states(5)
        z = predict(ref[:2], 5, "constant")
        self.assertEqual(z.shape, ref.shape)
        self.assertTrue(torch.equal(z[1], ref[1]))
        self.assertTrue(torch.equal(z[5], ref[1]))

    def test_geometric_increment_predictor(self):
        a = torch.tensor([[[0.0]], [[1.0]], [[1.5]]], dtype=torch.float64)
        z = predict(a, 4, "geometric")
        self.assertAlmostEqual(float(z[3, 0, 0]), 1.75, places=8)
        self.assertAlmostEqual(float(z[4, 0, 0]), 1.875, places=8)

    def test_jacobi_recovers_after_sufficient_rounds(self):
        ref = self._states(6)
        z = predict(ref[:2], 6, "constant")
        solver = CorrectionStepper(6, 1, "jacobi", self.step_many)
        for _ in range(5):
            z, change = solver.step(z)
        torch.testing.assert_close(z, ref, rtol=0, atol=0)
        self.assertEqual(solver.calls, 5)
        self.assertEqual(solver.eval_rows, 5 * 5 * 1)

    def test_identity_newton_exact_for_translation(self):
        ref = self._states(6)
        z = predict(ref[:2], 6, "constant")
        solver = CorrectionStepper(6, 1, "newton_identity", self.step_many)
        z, change = solver.step(z)
        torch.testing.assert_close(z, ref, rtol=0, atol=0)

    def test_anderson_preserves_anchor_and_finite(self):
        ref = self._states(6)
        z = predict(ref[:2], 6, "constant")
        solver = CorrectionStepper(6, 1, "anderson", self.step_many, history=3)
        for _ in range(3):
            z, _ = solver.step(z)
        self.assertTrue(torch.isfinite(z).all())
        torch.testing.assert_close(z[:2], ref[:2], rtol=0, atol=0)

    def test_allocation_keeps_final_depth(self):
        self.assertEqual(select_depths(8, 1, "suffix", 3), [6, 7, 8])
        self.assertEqual(select_depths(8, 1, "prefix", 3), [2, 3, 8])
        self.assertEqual(select_depths(8, 1, "full"), list(range(2, 9)))
        with self.assertRaises(ValueError):
            select_depths(8, 1, "residual_topk", 2)

    def test_prediction_must_not_access_future_exact_state(self):
        ref = self._states(4)
        z = predict(ref[:2], 4, "linear")
        self.assertFalse(torch.equal(z[4], ref[4]) and not torch.equal(z[1], ref[1]))


if __name__ == "__main__":
    unittest.main()
