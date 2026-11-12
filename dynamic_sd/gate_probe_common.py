"""CPU-only checks and paired aggregation for E3 dynamic-exit gate diagnostics.

No PyTorch dependency: unit tests can run without a CUDA device. Never treat
shadow-gate differences as a native live-GPU-gate cost or SD speedup.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

MODES = ("replay", "shadow_gpu", "shadow_readback", "live")


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as inp:
        return json.load(inp)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as inp:
        for buf in iter(lambda: inp.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def validate_trace(row, max_positions=None, max_depth=None):
    if not isinstance(row.get("idx"), int) or not isinstance(row.get("prefix_len"), int):
        raise ValueError("AR trace needs integer idx and prefix_len")
    if row["prefix_len"] <= 0:
        raise ValueError("Invalid prefix_len")
    toks, ds = row.get("tokens_ar"), row.get("depths_ar")
    if not isinstance(toks, list) or len(toks) < 2 or not all(isinstance(x, int) for x in toks):
        raise ValueError("Missing committed AR token continuation")
    if not isinstance(ds, list) or len(ds) != len(toks)-1:
        raise ValueError("Depths must align with all tokens_ar[:-1]")
    if any(not isinstance(d, int) or d < 1 or max_depth is not None and d > max_depth for d in ds):
        raise ValueError("Depth out of range")
    n = len(ds) if max_positions is None else min(len(ds), max_positions)
    if n < 1:
        raise ValueError("Empty decode segment")
    return n


def summarize(trials):
    """Only A/B/C with identical forced traces AND predictions are paired."""
    groups = defaultdict(dict)
    for row in trials:
        if row["mode"] not in MODES:
            raise ValueError("Unrecognized probe mode")
        key = (row["idx"], row["repeat"])
        if row["mode"] in groups[key]:
            raise ValueError("Duplicate mode for sample and repetition")
        groups[key][row["mode"]] = row
    paired, rejected = [], []
    for (idx, rep), modes in sorted(groups.items()):
        if not all(k in modes for k in MODES[:3]):
            continue
        a, b, c = (modes[k] for k in MODES[:3])
        must_match = ("input_digest", "predicted_digest", "depths", "decode_positions")
        if any(a.get(k) != b.get(k) or b.get(k) != c.get(k) for k in must_match):
            rejected.append({"idx": idx, "repeat": rep, "reason": "A/B/C path diverged"})
            continue
        if any(not x.get("depths_match_reference", False)
               or not x.get("prefill_first_match_reference", False)
               or x.get("prediction_match_fraction_to_reference") != 1.0
               for x in (a, b, c)):
            rejected.append({"idx": idx, "repeat": rep,
                             "reason": "source AR depth/prefill/continuation mismatch"})
            continue
        n = a["decode_positions"]
        if n <= 0:
            rejected.append({"idx": idx, "repeat": rep, "reason": "zero decode positions"})
            continue
        paired.append({
            "idx": idx, "repeat": rep,
            "delta_B_minus_A_wall_ms_per_position": (b["decode_wall_ms"] - a["decode_wall_ms"]) / n,
            "delta_C_minus_B_wall_ms_per_position": (c["decode_wall_ms"] - b["decode_wall_ms"]) / n,
            "B_gate_cuda_event_ms_per_position": b.get("gate_event_ms", 0.) / n,
            "C_cpu_readback_wait_ms_per_position": c.get("readback_wait_ms", 0.) / n,
        })
    keys = ("delta_B_minus_A_wall_ms_per_position",
            "delta_C_minus_B_wall_ms_per_position",
            "B_gate_cuda_event_ms_per_position",
            "C_cpu_readback_wait_ms_per_position")
    return {"valid_triplets": len(paired), "excluded_triplets": rejected,
            "paired_metrics": {key: ({
                "median": statistics.median([r[key] for r in paired]),
                "mean": statistics.mean([r[key] for r in paired]),
            } if paired else None) for key in keys},
            "interpretation": "B−A/C−B are shadow implementation costs with forced AR trace; "
                              "native live exit and speculative branch gate costs remain unisolated."}
