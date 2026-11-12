#!/usr/bin/env python3
"""Shared loader / validator for the committed same-prefix one-round counterfactual
receipts (dynamic_sd/followup_20261112/results_20261112/cf/*.jsonl.gz).

Read-only: files are streamed from .gz, never unpacked or modified.
Stdlib only (runs on CPU without torch).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REQUIRED = ("prefix_id", "prefix_hash", "target_hash", "exit_policy", "kv_policy", "batch_regime",
            "backend", "precision", "action", "wall_ms", "committed_tokens", "target_exact",
            "measured", "repeat", "prompt_idx", "k", "pos", "ref_next_depths", "info")
CONTRACT = ("prefix_hash", "target_hash", "exit_policy", "kv_policy", "batch_regime", "backend",
            "precision", "prompt_idx", "k", "pos")


class ReceiptError(ValueError):
    pass


def parse_action(a: str):
    """'AR' -> ('AR', None, None); 'DtV:d4:K8' -> ('DtV', 4, 8)."""
    if a == "AR":
        return ("AR", None, None)
    kind, d, K = a.split(":")
    if kind != "DtV" or not d.startswith("d") or not K.startswith("K"):
        raise ReceiptError(f"unknown action {a}")
    return ("DtV", int(d[1:]), int(K[1:]))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_check(path, manifest):
    """Return (sha256, listed_sha256 or None). Raises if listed and different."""
    got = sha256_file(path)
    want = None
    if manifest and Path(manifest).exists():
        name = Path(path).name
        for line in open(manifest):
            parts = line.split()
            if len(parts) == 2 and parts[1].endswith("/" + name):
                want = parts[0]
        if want is not None and want != got:
            raise ReceiptError(f"sha256 mismatch for {path}: manifest {want} file {got}")
    return got, want


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_meta(path):
    p = str(path)
    for cand in (p.replace(".jsonl.gz", ".jsonl.meta.json.gz"), p + ".meta.json", p + ".meta.json.gz"):
        if Path(cand).exists():
            opener = gzip.open if cand.endswith(".gz") else open
            with opener(cand, "rt") as f:
                return json.load(f), cand
    return {}, None


def validate(rows, expected_repeats=None):
    """Validate a cell. Returns dict with
       prefixes: {prefix_id: {"meta": first_row, "actions": {action: [rows sorted by repeat]}}} (strict-exact only)
       excluded: list of prefix_ids dropped because any trial was not target-exact
       actions: sorted action grid, repeats: sorted repeat ids, audit: counts.
    Raises ReceiptError on contract / completeness / measurement / geometry violations."""
    if not rows:
        raise ReceiptError("empty cell")
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        miss = [k for k in REQUIRED if k not in r]
        if miss:
            raise ReceiptError(f"row {i}: missing fields {miss}")
        if r["measured"] is not True:
            raise ReceiptError(f"row {i}: not physically measured")
        if not isinstance(r["wall_ms"], (int, float)) or not r["wall_ms"] > 0:
            raise ReceiptError(f"row {i}: zero/invalid wall_ms {r['wall_ms']}")
        if not isinstance(r["committed_tokens"], int) or r["committed_tokens"] < 1:
            raise ReceiptError(f"row {i}: invalid committed_tokens")
        if r["kv_policy"] != "FULL":
            raise ReceiptError(f"row {i}: kv_policy {r['kv_policy']} (FULL required)")
        if str(r["prefix_id"]).split(":")[0] != str(r["prompt_idx"]):
            raise ReceiptError(f"row {i}: prefix_id {r['prefix_id']} not of prompt {r['prompt_idx']}")
        parse_action(r["action"])
        groups[r["prefix_id"]].append(r)
    grid = sorted({r["action"] for r in rows})
    reps = sorted({r["repeat"] for r in rows})
    if expected_repeats is not None and len(reps) != expected_repeats:
        raise ReceiptError(f"expected {expected_repeats} repeats, found {reps}")
    prefixes, excluded = {}, []
    by_hash = {}
    for pid, trials in groups.items():
        first = trials[0]
        for t in trials:
            for k in CONTRACT:
                if t[k] != first[k]:
                    raise ReceiptError(f"changed prefix/target contract at {pid}: {k}")
        if by_hash.setdefault(first["prefix_hash"], pid) != pid:
            raise ReceiptError(f"prefix_hash reused by two prefix_ids: {pid}, {by_hash[first['prefix_hash']]}")
        acts = defaultdict(list)
        for t in trials:
            acts[t["action"]].append(t)
        if sorted(acts) != grid:
            raise ReceiptError(f"missing action grid at {pid}: {sorted(set(grid) - set(acts))}")
        for a, ts in acts.items():
            if sorted(t["repeat"] for t in ts) != reps:
                raise ReceiptError(f"missing/duplicate repeats at {pid} {a}")
            ts.sort(key=lambda t: t["repeat"])
            g0 = (ts[0]["committed_tokens"], json.dumps(ts[0]["info"], sort_keys=True), ts[0]["target_exact"])
            for t in ts[1:]:
                if (t["committed_tokens"], json.dumps(t["info"], sort_keys=True), t["target_exact"]) != g0:
                    raise ReceiptError(f"repeat geometry differs at {pid} {a}")
        if not all(t["target_exact"] for t in trials):
            excluded.append(pid)
            continue
        prefixes[pid] = {"meta": first, "actions": dict(acts)}
    audit = {"rows": len(rows), "prefixes_total": len(groups), "prefixes_kept": len(prefixes),
             "prefixes_excluded_inexact": len(excluded),
             "inexact_trials": sum(1 for r in rows if not r["target_exact"]),
             "prompts": len({r["prompt_idx"] for r in rows}), "actions": len(grid), "repeats": reps}
    return {"prefixes": prefixes, "excluded": excluded, "actions": grid, "repeats": reps,
            "audit": audit, "groups": groups}


def selection_bias(cell):
    """Compare excluded (inexact) vs kept prefixes on observable covariates."""
    def summ(pids, src):
        if not pids:
            return None
        D = [src[p][0]["ref_next_depths"][0] for p in pids]
        pos = [src[p][0]["pos"] - src[p][0]["k"] for p in pids]   # prompt length
        k = [src[p][0]["k"] for p in pids]
        return {"n": len(pids), "mean_ref_D": sum(D) / len(D), "mean_k": sum(k) / len(k),
                "mean_prompt_len": sum(pos) / len(pos)}
    g = cell["groups"]
    return {"kept": summ(list(cell["prefixes"]), g), "excluded": summ(cell["excluded"], g)}


def prompt_split(prompts, n_folds=2):
    """Deterministic prompt-disjoint folds: sorted prompt ids, fold = rank % n_folds."""
    ps = sorted(set(prompts))
    return [[p for i, p in enumerate(ps) if i % n_folds == f] for f in range(n_folds)]


def assert_disjoint(calib, evaluation):
    leak = set(calib) & set(evaluation)
    if leak:
        raise ReceiptError(f"prompt leakage between calibration and evaluation: {sorted(leak)[:5]}")


def tok_ms(trials):
    return sum(t["committed_tokens"] for t in trials), sum(t["wall_ms"] for t in trials)


def bootstrap_ratio(per_prompt_pairs, n=2000, seed=0):
    """per_prompt_pairs: {prompt: (num_tok, num_ms, den_tok, den_ms)} -> (point, lo, hi) of
    (Σnum_tok/Σnum_ms)/(Σden_tok/Σden_ms), prompt-cluster bootstrap."""
    keys = sorted(per_prompt_pairs)
    def ratio(ks):
        a = [sum(per_prompt_pairs[k][i] for k in ks) for i in range(4)]
        return (a[0] / a[1]) / (a[2] / a[3])
    point = ratio(keys)
    rng = random.Random(seed)
    bs = sorted(ratio([rng.choice(keys) for _ in keys]) for _ in range(n))
    return point, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]
