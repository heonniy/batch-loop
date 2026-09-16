"""Small CPU-only checks for causal RIC evaluation and all loss families."""

import unittest

import numpy as np

from phase03_predictor.tournament import _metrics, _prepare, _ric, _scores, _train_one


class TournamentTests(unittest.TestCase):
    def test_ric_reordering(self):
        depths = np.array([[1, 4, 1, 4]], dtype=np.int16)
        fifo, predictor, oracle = _ric(depths, np.array([[1, 4, 1, 4]], dtype=np.float32), 2)
        self.assertEqual((fifo, predictor, oracle), (8, 5, 5))

    def test_all_trainable_loss_families(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(3, 64, 8)).astype(np.float32)
        labels = np.where(raw[..., 0] > 0, 7, 2).astype(np.int16)
        x, _, _ = _prepare(raw, list(range(32)))
        for candidate in ("C1", "C7", "C8", "C9"):
            with self.subTest(candidate=candidate):
                head, config = _train_one(candidate, x, labels, list(range(32)),
                                          list(range(32, 64)), 8, 0, 1e-3, 2, 1)
                scores, elapsed = _scores(head, x[:, 32:64], candidate)
                result = _metrics(candidate, labels[:, 32:64], scores, 8, 1, elapsed)
                self.assertTrue(np.isfinite(result["oracle_ric_recovery"]))
                self.assertGreaterEqual(config["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
