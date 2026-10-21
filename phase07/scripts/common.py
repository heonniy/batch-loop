"""Phase-07 shared CPU helpers: dataset loading, token classes, metrics, RIC utilities."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("whitespace", "punctuation", "digit", "code", "other")
CODE_CHARS = set("{}[]()<>=_;/\\|`$#@*&^~")


def load(features_root: Path, lane: str, split: str, arrays=()) -> dict:
    d = features_root / lane / split
    meta = np.load(d / "meta.npz")
    out = {k: meta[k] for k in meta.files}
    out["manifest"] = json.loads((d / "manifest.json").read_text())
    for a in arrays:
        p = d / f"{a}.npy"
        out[a] = np.load(p, mmap_mode="r") if p.exists() else None
    return out


@lru_cache(maxsize=4)
def tokenizer(family: str):
    from transformers import AutoTokenizer
    hf = Path.home() / ".cache/huggingface/hub"
    snap = (hf / "models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c" if family == "huginn"
            else hf / "models--ByteDance--Ouro-1.4B/snapshots/37bc7e8ff85a90d813dc096383405f521b6a67b0")
    return AutoTokenizer.from_pretrained(snap, trust_remote_code=True)


def token_class(family: str, ids: np.ndarray) -> np.ndarray:
    """Surface class of the CURRENT input token: whitespace / punctuation / digit / code / other."""
    tok = tokenizer(family)
    cache = {}
    out = np.empty(ids.shape, dtype=np.int8)
    for idx, t in np.ndenumerate(ids):
        t = int(t)
        if t not in cache:
            s = tok.decode([t])
            core = s.strip()
            if not core:
                c = 0
            elif any(ch in CODE_CHARS for ch in core):
                c = 3
            elif all(ch.isdigit() for ch in core):
                c = 2
            elif all(not ch.isalnum() for ch in core):
                c = 1
            else:
                c = 4
            cache[t] = c
        out[idx] = cache[t]
    return out


def ric(depth_t: np.ndarray, order: np.ndarray, B: int) -> int:
    d = depth_t[order]
    return int(sum(d[i:i + B].max() for i in range(0, len(d), B)))


def lockstep_ric(depth: np.ndarray, scores: np.ndarray | None, B: int) -> int:
    """depth/scores [T, N]; scores None = FIFO; sort ascending by score then request id."""
    total = 0
    ids = np.arange(depth.shape[1])
    for t in range(depth.shape[0]):
        order = ids if scores is None else np.lexsort((ids, scores[t]))
        total += ric(depth[t], order, B)
    return total


def recovery(fifo: int, pred: int, oracle: int) -> float:
    return (fifo - pred) / (fifo - oracle) if fifo != oracle else float("nan")


def kendall_tau_b_within_step(depth: np.ndarray, score: np.ndarray) -> float:
    """Mean over decode steps of Kendall tau-b between true depth and score across requests (ties handled)."""
    vals = []
    iu = None
    for d, s in zip(depth, score):
        if iu is None:
            iu = np.triu_indices(len(d), 1)
        dd = np.sign(d[:, None] - d[None, :])[iu]
        ss = np.sign(s[:, None] - s[None, :])[iu]
        den = math.sqrt(float(np.count_nonzero(dd)) * float(np.count_nonzero(ss)))
        if den:
            vals.append(float((dd * ss).sum() / den))
    return float(np.mean(vals)) if vals else float("nan")


def prob_metrics(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """probs [M, K] over depth classes 1..K, y [M] in 1..K."""
    eps = 1e-12
    p_true = probs[np.arange(len(y)), y - 1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y - 1] = 1
    conf = probs.max(1)
    pred = probs.argmax(1) + 1
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return {"nll": float(-np.log(p_true + eps).mean()), "brier": float(((probs - onehot) ** 2).sum(1).mean()),
            "ece": float(ece), "acc": float((pred == y).mean())}


def point_metrics(pred_depth: np.ndarray, y: np.ndarray) -> dict:
    return {"acc_round": float((np.rint(pred_depth) == y).mean()), "mae": float(np.abs(pred_depth - y).mean())}
