"""Synthetic end-to-end CPU test of the all-ten tournament CLI."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phase03_predictor.collect_huginn import SCALARS
from phase03_predictor.tournament import main
from phase03_predictor.transfer import evaluate


class FullTournamentTests(unittest.TestCase):
    def test_all_ten_saved_data_routes(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature = root / "features"
            feature.mkdir()
            labels = rng.integers(1, 5, size=(3, 96), dtype=np.int16)
            scalar = rng.normal(size=(3, 96, len(SCALARS))).astype(np.float32)
            scalar[0, :, SCALARS.index("previous_depth")] = 0
            scalar[1:, :, SCALARS.index("previous_depth")] = labels[:-1]
            scalar[:, :, SCALARS.index("previous_trajectory_length")] = scalar[
                :, :, SCALARS.index("previous_depth")]
            np.save(feature / "scalars.npy", scalar)
            np.save(feature / "embedding.npy", rng.normal(size=(3, 96, 8)).astype(np.float16))
            np.save(feature / "prelude.npy", rng.normal(size=(3, 96, 8)).astype(np.float16))
            np.save(feature / "depth_labels.npy", labels)
            np.save(feature / "projection.npy", rng.normal(size=(8, 8)).astype(np.float32))
            hashes = {}
            for name in ("scalars.npy", "embedding.npy", "prelude.npy", "depth_labels.npy"):
                hashes[name] = hashlib.sha256((feature / name).read_bytes()).hexdigest()
            manifest = {"status": "COMPLETE", "model": "ouro-1.4b", "rmax": 4,
                        "scalar_names": SCALARS, "artifact_sha256": hashes,
                        "source_sha": "synthetic-test", "request_split": {
                            "train": list(range(32)), "validation": list(range(32, 64)),
                            "test": list(range(64, 96))}}
            (feature / "feature_manifest.json").write_text(json.dumps(manifest))
            output = root / "output"
            with patch.object(sys, "argv", ["tournament", "--feature-root", str(feature),
                                            "--output-dir", str(output), "--batch-size", "8",
                                            "--epochs", "1"]):
                main()
            lines = (output / "predictor_tournament.csv").read_text().splitlines()
            self.assertEqual(len(lines), 21)
            self.assertTrue((output / "C9_checkpoint.pt").exists())
            transfer = root / "transfer"
            evaluate(feature, output, transfer, 8)
            self.assertEqual(len((transfer / "transfer_tournament.csv").read_text().splitlines()), 21)


if __name__ == "__main__":
    unittest.main()
