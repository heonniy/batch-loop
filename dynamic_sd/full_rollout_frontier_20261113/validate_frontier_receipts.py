#!/usr/bin/env python3
"""Validate full-rollout frontier receipts (run_frontier.py JSONL + .meta.json). Stdlib only.

A CELL = (lane, model, exit, dataset) = all chunk files sharing everything but offset/limit/spec-ids.
Rejected (cell status `failed`, reasons listed):
  * mixed model / dtype / seed / T_total / d / exit rule / K grid / max_new / backend inside a cell
  * a Dory model labelled native live exit (no verified per-token gate)
  * dtype vs backend mismatch (FP32 must be SDPA-fallback, BF16 FA varlen)
  * per (prompt, repeat): arms with different prompt_hash, or an incomplete arm grid
  * duplicated (prompt, repeat, arm)
  * eos/stop matcher call count != committed tokens (stop logic not observed per token)
  * uninstrumented stage-time fields present as numbers (must be null = missing, never fabricated)
  * non-positive wall / tokens
  * parity lane (FP32): any SD arm not token-identical to AR -> `parity_failed` (kept as data)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

STAGE = ("draft_ms", "verifier_ms", "coda_ms", "exit_gate_ms", "host_sync_ms")
LOCK = ("model", "dtype", "seed", "T_total", "d", "exit", "tau", "eps", "Ks", "wfd", "max_new_tokens",
        "max_prompt_tokens", "stops", "dataset", "num_fewshot", "repeats")


def cell_key(path):
    name = Path(path).name.replace(".jsonl", "")
    lane = Path(path).parent.name
    return f"{lane}/" + re.sub(r"_(o\d+n\d+(r\d+)?|i\d+n\d+|n\d+)$", "", name)


def load_cell_files(paths):
    files = []
    for p in paths:
        mp = p + ".meta.json"
        meta = json.load(open(mp)) if Path(mp).exists() else None
        rows = [json.loads(l) for l in open(p) if l.strip()]
        files.append((p, meta, rows))
    return files


def validate_cell(files):
    reasons, warnings = [], []
    metas = [m for _, m, _ in files]
    if any(m is None for m in metas):
        reasons.append("missing .meta.json (run incomplete)")
        metas = [m for m in metas if m]
    if not metas:
        return {"status": "failed", "reasons": reasons, "rows": []}
    a0 = metas[0]["args"]
    for m in metas[1:]:
        for k in LOCK:
            if m["args"].get(k) != a0.get(k):
                reasons.append(f"mixed {k}: {a0.get(k)} vs {m['args'].get(k)}")
        for k in ("wf_commit", "gpu", "torch", "backend"):
            if m.get(k) != metas[0].get(k):
                reasons.append(f"mixed runtime {k}")
    if "dory" in a0["model"].lower() and a0["exit"] == "live":
        reasons.append("Dory labelled native live exit (no verified per-token gate)")
    backend = metas[0].get("backend", "")
    if (a0["dtype"] == "float32") != ("SDPA" in backend):
        reasons.append(f"dtype/backend mismatch: {a0['dtype']} vs {backend}")
    arms = metas[0]["arms"]
    rows = [r for _, _, rs in files for r in rs]
    groups = defaultdict(dict)
    for r in rows:
        key = (r["prompt_id"], r["repeat"])
        if r["arm"] in groups[key]:
            reasons.append(f"duplicate {key} {r['arm']}")
        groups[key][r["arm"]] = r
        if r["eos_compare_calls"] != r["tokens_committed"]:
            reasons.append(f"stop matcher calls != tokens at {key} {r['arm']}")
        for f in STAGE:
            if r.get(f, None) is not None:
                reasons.append(f"stage field {f} present but not instrumented at {key}")
        if not (r["measured_wall_ms"] > 0 and r["tokens_committed"] >= 1):
            reasons.append(f"non-positive wall/tokens at {key} {r['arm']}")
    for key, g in groups.items():
        if sorted(g) != sorted(arms):
            reasons.append(f"incomplete arm grid at {key}: missing {sorted(set(arms) - set(g))}")
        if len({r["prompt_hash"] for r in g.values()}) != 1:
            reasons.append(f"different prompt_hash across arms at {key}")
    parity = None
    if a0["dtype"] == "float32":
        bad = [(k, arm) for k, g in groups.items() for arm, r in g.items() if arm != "AR" and not r["exact_match_to_AR"]]
        parity = {"n_rollouts": sum(len(g) - 1 for g in groups.values()), "inexact": len(bad), "examples": bad[:5]}
    status = "failed" if reasons else ("parity_failed" if parity and parity["inexact"] else "passed")
    # config/runtime-level reasons first, then per-row ones (capped)
    uniq = list(dict.fromkeys(reasons))
    top = [x for x in uniq if x.startswith(("mixed", "Dory", "dtype/backend", "missing .meta"))]
    return {"status": status, "reasons": top + [x for x in uniq if x not in top][:200], "n_reasons": len(uniq),
            "n_prompts": len({k[0] for k in groups}), "repeats": sorted({k[1] for k in groups}),
            "arms": arms, "args": a0, "rule": metas[0].get("target_exit_rule"), "parity": parity,
            "wf_commit": metas[0].get("wf_commit"), "source_commit": metas[0].get("source_commit"),
            "warnings": warnings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="results_<date>/raw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = defaultdict(list)
    for p in sorted(glob.glob(f"{a.raw}/*/*.jsonl")):
        cells[cell_key(p)].append(p)
    res = {c: validate_cell(load_cell_files(ps)) | {"files": ps} for c, ps in sorted(cells.items())}
    json.dump(res, open(a.out, "w"), indent=1)
    for c, r in res.items():
        print(f"{r['status']:13s} {c}  n={r.get('n_prompts')} {('; '.join(r['reasons'][:2])) if r['reasons'] else ''}")


if __name__ == "__main__":
    main()
