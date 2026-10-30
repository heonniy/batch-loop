"""Protocol-frozen KV candidate sampling for the MAIN characterization study.

No evidence labels, Hotpot supporting-fact annotations, answer strings,
attention scores, or counterfactual model outputs are accepted as inputs.
All fully aligned Context tokens are eligible, irrespective of relevance.
The main selector operates on positions ONLY and chooses uniform-position
coverage through stratified, deterministic hash sampling.
"""
from __future__ import annotations

import hashlib
from typing import Sequence


def _sha_key(seed: int, bucket: int, start: int) -> str:
    return hashlib.sha256(
        f"role-main-v3/{seed}/{bucket}/{start}".encode("utf-8")
    ).hexdigest()


def main_candidate_universe(content_positions: Sequence[int], window_tokens: int = 1
                            ) -> list[tuple[int, ...]]:
    """All consecutive, same-width Context windows; no annotation dependency.

    A candidate cannot include separators, chat delimiters, Instruction, or
    Question tokens. Each selected set is a KV *read mask* rather than text
    deletion or a physical cache compaction.
    """
    if window_tokens < 1:
        raise ValueError("window_tokens must be >=1")
    positions = tuple(sorted(set(int(p) for p in content_positions)))
    if any(p < 0 for p in positions):
        raise ValueError("Negative Context token position")
    if not positions:
        raise ValueError("Empty Context pool")
    windows: list[tuple[int, ...]] = []
    for start in range(len(positions)):
        end = start + window_tokens
        if end > len(positions):
            break
        trial = positions[start:end]
        if trial[-1] - trial[0] == window_tokens - 1:
            windows.append(trial)
    if not windows:
        raise ValueError("No tokenizer-aligned Context window has the requested width")
    return windows


def choose_main_candidates(content_positions: Sequence[int], seed: int, *,
                           n_candidates: int = 8, window_tokens: int = 1,
                           position_bins: int = 4) -> list[dict]:
    """Sample from the COMPLETE Context pool without ground-truth selection.

    Eligible windows are divided by their center's *position within the
    complete Context*, not answer relevance. Hash order, cycling through
    strata, avoids dependence on Python random state and preserves coverage.
    Candidate windows may not overlap; an impossible requested budget fails.
    """
    if n_candidates < 1 or position_bins < 1:
        raise ValueError("Positive candidate count and strata required")
    universe = main_candidate_universe(content_positions, window_tokens)
    lo = min(content_positions)
    hi = max(content_positions)
    pools: list[list[tuple[int, ...]]] = [[] for _ in range(position_bins)]
    for window in universe:
        mid = (window[0] + window[-1]) / 2
        bucket = min(position_bins - 1,
                     int(position_bins * (mid - lo) / max(1, hi - lo + 1)))
        pools[bucket].append(window)
    for bucket, candidates in enumerate(pools):
        candidates.sort(key=lambda w: (_sha_key(seed, bucket, w[0]), w[0]))
    selected: list[tuple[int, ...]] = []
    occupied: set[int] = set()
    # Prefer a candidate from every occupied position stratum in each pass.
    while len(selected) < n_candidates:
        changed = False
        for candidates in pools:
            while candidates:
                window = candidates.pop(0)
                if occupied.isdisjoint(window):
                    selected.append(window)
                    occupied.update(window)
                    changed = True
                    break
            if len(selected) >= n_candidates:
                break
        if not changed:
            break
    if len(selected) != n_candidates:
        raise ValueError(
            f"Requested {n_candidates} nonoverlapping Context windows of "
            f"{window_tokens} tokens but selected only {len(selected)}; "
            "reduce --main-candidates or --window-tokens in a NEW run")
    return [
        {"id": f"main-pos-{window[0]:06d}-{window[-1]:06d}",
         "role": "context",
         "positions": list(window),
         "source": "all-context-unlabeled",
         "window_tokens": window_tokens,
         "answer_anchor_used": False,
         "selection_policy": "positional-stratified-hash-v3"}
        for window in sorted(selected, key=lambda w: w[0])
    ]
