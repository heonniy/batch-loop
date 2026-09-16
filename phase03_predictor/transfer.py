"""Evaluate frozen primary C0-C9 heads on a secondary threshold/q cohort.

No fitting or model forward occurs here. The primary feature normalization,
head weights, and candidate selection remain frozen. Secondary validation/test
labels are used only to score RIC and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from phase03_predictor.tournament import CANDIDATES, Head, _feature_sets, _metrics, _scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(secondary: Path, primary: Path, output: Path, batch_size: int) -> None:
    manifest = json.loads((secondary / "feature_manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("secondary feature collection incomplete")
    for filename, expected in manifest["artifact_sha256"].items():
        if _sha256(secondary / filename) != expected:
            raise RuntimeError(f"secondary feature hash mismatch: {filename}")
    primary_run = json.loads((primary / "run_manifest.json").read_text())
    primary_feature_root = Path(primary_run["feature_root"])
    if _sha256(primary_feature_root / "projection.npy") != _sha256(secondary / "projection.npy"):
        raise RuntimeError("secondary projection differs from frozen primary basis")
    primary_feature = json.loads((primary_feature_root / "feature_manifest.json").read_text())
    if tuple(primary_feature["scalar_names"]) != tuple(manifest["scalar_names"]):
        raise RuntimeError("secondary scalar feature schema differs from primary")
    labels = np.asarray(np.load(secondary / "depth_labels.npy", mmap_mode="r"), dtype=np.int16)
    scalar = np.load(secondary / "scalars.npy", mmap_mode="r")
    if not (np.all(scalar[0, :, 2] == 0) and np.array_equal(scalar[1:, :, 2], labels[:-1])):
        raise RuntimeError("secondary previous-depth causal audit failed")
    sets = _feature_sets(secondary, manifest)
    split = manifest["request_split"]
    if any(len(ids) % batch_size for ids in split.values()):
        raise RuntimeError("secondary split rows must be divisible by B")
    rows = []
    for candidate in CANDIDATES:
        if candidate == "C0":
            scores = np.asarray(scalar[:, :, 2], dtype=np.float32)
            elapsed = 0.0
            params = 0
        else:
            checkpoint = torch.load(primary / f"{candidate}_checkpoint.pt",
                                    map_location="cpu", weights_only=False)
            x = sets[candidate]
            mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
            if x.shape[-1] != len(mean):
                raise RuntimeError(f"{candidate} primary/secondary feature schema mismatch")
            x = np.ascontiguousarray((x - mean) / std, dtype=np.float32)
            head = Head(checkpoint["input_dim"], checkpoint["output_dim"],
                        checkpoint["config"]["width"])
            head.load_state_dict(checkpoint["state_dict"])
            scores, elapsed = _scores(head, x, candidate)
            params = sum(parameter.numel() for parameter in head.parameters())
        for name in ("validation", "test"):
            ids = split[name]
            rows.append({"split": name, "training_regime": str(primary),
                         "evaluation_regime": str(secondary),
                         **_metrics(candidate, labels[:, ids], scores[:, ids],
                                    batch_size, params, elapsed * len(ids) / labels.shape[1])})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "transfer_tournament.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_manifest.json").write_text(json.dumps({
        "execution_kind": "saved-data-analysis", "no_model_forward": True, "no_secondary_fitting": True,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "secondary_feature_manifest_sha256": _sha256(secondary / "feature_manifest.json"),
        "primary_tournament_sha256": _sha256(primary / "predictor_tournament.csv"),
        "batch_size": batch_size, "candidate_count": len(CANDIDATES), "rows": len(rows),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondary-feature-root", type=Path, required=True)
    parser.add_argument("--primary-tournament-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8)
    evaluate(args.secondary_feature_root, args.primary_tournament_root,
             args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
