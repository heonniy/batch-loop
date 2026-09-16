"""Add a transparent post-run source provenance receipt to frozen feature data.

The primary collectors were launched from a dirty worktree before their
implementation commit. Their launch Git HEAD is not the implementation SHA.
This command preserves that fact and binds the now-committed unchanged source
files by hash, without falsely claiming a cryptographic in-memory launch hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


IMPLEMENTATION_COMMIT = "305817f08fed7e281572a8d4de7df347cb6a1c45"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(feature_root: Path, model: str) -> None:
    path = feature_root / "feature_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["status"] != "COMPLETE" or not manifest["model"].startswith(model):
        raise RuntimeError("feature collection incomplete or wrong model")
    for filename, expected in manifest["artifact_sha256"].items():
        if _sha256((feature_root / filename).read_bytes()) != expected:
            raise RuntimeError(f"feature artifact changed: {filename}")
    files = [f"phase03_predictor/collect_{model}.py",
             f"phase02_runtime/{model}_adapter.py"]
    hashes = {}
    for filename in files:
        committed = subprocess.check_output(["git", "show", f"{IMPLEMENTATION_COMMIT}:{filename}"])
        current = Path(filename).read_bytes()
        if current != committed:
            raise RuntimeError(f"collector/adapter changed since implementation commit: {filename}")
        hashes[filename] = _sha256(current)
    launch_head = manifest["source_sha"]
    manifest["source_provenance"] = {
        "launch_git_head": launch_head,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "committed_collector_and_adapter_sha256": hashes,
        "launch_from_uncommitted_worktree": launch_head != IMPLEMENTATION_COMMIT,
        "code_hash_observed_at_launch": False,
        "limitation": ("Post-run source-file hashes match the implementation commit, but the collector "
                       "did not record an in-memory code hash at launch. Its original source_sha is "
                       "the Git HEAD at launch, not a claim that HEAD contains the new collector."),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model", choices=("huginn", "ouro"), required=True)
    args = parser.parse_args()
    audit(args.feature_root, args.model)


if __name__ == "__main__":
    main()
