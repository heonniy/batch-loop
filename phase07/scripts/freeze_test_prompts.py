"""Freeze the Phase-07 test-split prompt texts (both datasets) with hashes, for the upstream-venv scripts.
Run with the batch-loop venv: /home/hwlee/batch-loop/.venv/bin/python phase07/scripts/freeze_test_prompts.py"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import ROOT, SPLITS, dataset_prompts  # noqa: E402

out = {}
for ds in ("alpaca", "sharegpt"):
    lo, hi = SPLITS["test"]
    p = dataset_prompts(ds)[lo:hi]
    out[ds] = {"prompts": p, "sha256": [hashlib.sha256(x.encode()).hexdigest() for x in p], "split": "test", "range": [lo, hi]}
path = ROOT / "phase07/inputs/test_prompts.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, ensure_ascii=False) + "\n")
print(path, hashlib.sha256(path.read_bytes()).hexdigest())
