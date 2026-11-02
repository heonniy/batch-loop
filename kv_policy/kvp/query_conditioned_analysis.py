"""CPU-only v7 analysis helpers (optional tools, not a fixed agent implementation).

- Pair earlier Context KV by the *same later query's CLEAN attention*.
- Validate and summarize paired native-LIVE / FULL-depth REPLAY / FIXED
  exit traces for the SAME frozen continuation.

No gold answer, supporting-fact labels, future NLL or eviction outcomes enter
candidate selection. No PyTorch, GPU or model dependencies are required.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence


def attention_matched_pairs(
    context_positions: Sequence[int],
    clean_query_attention: Mapping[int | str, Sequence[float]],
    *,
    max_pairs: int = 4,
    max_position_gap: int = 16,
    max_log_attention_ratio: float = 0.35,
) -> dict:
    """Deterministic, non-overlapping within-query, within-prompt candidate pairs.

    clean_query_attention[p] must be the q->p profile by recurrent loop for
    the SAME preregistered later prompt Query q. Do NOT feed source's own
    attention, mean over generated future queries, or masked future outcomes.

    Selection samples low/middle/high clean-attention strata in round-robin
    order; it may return zero pairs, which must be reported rather than
    lowering thresholds after looking at quality labels.
    """
    p = tuple(sorted(int(x) for x in context_positions))
    if not p or len(set(p)) != len(p):
        raise ValueError("Require nonempty unique Context token positions")
    if max_pairs < 1 or max_position_gap < 1:
        raise ValueError("Invalid pair count / position gap")
    if not 0 <= max_log_attention_ratio <= 5:
        raise ValueError("Invalid log-attention matching tolerance")
    scores = {}
    depth = None
    for i in p:
        prof = clean_query_attention.get(str(i), clean_query_attention.get(i))
        if prof is None:
            raise ValueError(f"Missing CLEAN query attention for position {i}")
        vals = [float(x) for x in prof]
        if not vals or any(not math.isfinite(x) or x < 0 for x in vals):
            raise ValueError(f"Non-finite/negative query attention for {i}")
        if depth is None:
            depth = len(vals)
        elif depth != len(vals):
            raise ValueError("Source profiles have inconsistent loop depths")
        scores[i] = sum(vals) / len(vals)

    active = sorted(v for v in scores.values() if v > 1e-12)
    if len(active) < 2:
        return {"pairs": [], "n_eligible": len(p), "n_candidates": 0,
                "n_selected": 0, "depth": depth,
                "reason": "insufficient_nonzero_query_attention"}

    def band(value: float) -> int:
        return min(2, 3 * sum(x < value for x in active) // len(active))

    possible = defaultdict(list)
    for idx, a in enumerate(p):
        for b in p[idx + 1:]:
            if b - a > max_position_gap:
                break
            if scores[a] <= 1e-12 or scores[b] <= 1e-12:
                continue
            lr = abs(math.log(scores[a] / scores[b]))
            if lr <= max_log_attention_ratio:
                k = band((scores[a] + scores[b]) / 2)
                possible[k].append((lr, b-a, a, b))
    for pairs in possible.values():
        pairs.sort()
    chosen, used = [], set()
    while len(chosen) < max_pairs:
        progress = False
        for k in (0, 1, 2):
            for lr, gap, a, b in possible[k]:
                if a not in used and b not in used:
                    chosen.append({
                        "source_a": a, "source_b": b, "attention_a": scores[a],
                        "attention_b": scores[b], "attention_band": k,
                        "position_gap": gap, "abs_log_attention_ratio": lr})
                    used.update((a, b))
                    progress = True
                    break
            if len(chosen) == max_pairs:
                break
        if not progress:
            break
    return {
        "pairs": chosen, "n_eligible": len(p),
        "n_candidates": sum(map(len, possible.values())),
        "n_selected": len(chosen), "depth": depth,
        "reason": "ok" if chosen else "no_attention_matched_pairs",
    }


def summarize_live_replay(
    *,
    prompt_len: int,
    target_len: int,
    native_max_depth: int,
    full_live: dict,
    full_replay: dict,
    full_fixed: dict,
    evicted_live: dict,
    evicted_replay: dict,
    evicted_fixed: dict,
    parity_tol: float = 1e-4,
) -> dict:
    """Validate common-input/clean-trace parity and report exit/quality contrasts.

    All arms must be scored on the SAME externally frozen token IDs.
    Live-minus-replay is descriptive and must not be called a direct
    causal mediation estimate. nll_future excludes the first target token
    predicted from the unchanged last prompt position.
    """
    if prompt_len < 1 or target_len < 2 or native_max_depth < 2:
        raise ValueError("Invalid prompt/target/depth lengths")
    if parity_tol < 0:
        raise ValueError("Invalid replay tolerance")
    arms = {"full_live": full_live, "full_replay": full_replay,
            "full_fixed": full_fixed, "evicted_live": evicted_live,
            "evicted_replay": evicted_replay, "evicted_fixed": evicted_fixed}
    expected = prompt_len + target_len - 1
    for name, arm in arms.items():
        d = arm["depths"]
        losses = arm["token_losses"]
        if len(d) != expected or len(losses) != target_len:
            raise ValueError(f"Unaligned token/exit trace for {name}")
        if any(int(r) != r or not 1 <= r <= native_max_depth for r in d):
            raise ValueError(f"Invalid depth in {name}")
        if d[:prompt_len] != [native_max_depth] * prompt_len:
            raise ValueError(f"Prefill is not full depth in {name}")
        if any(not math.isfinite(float(v)) for v in losses):
            raise ValueError(f"Nonfinite loss in {name}")
        if not math.isfinite(float(arm["nll_future"])):
            raise ValueError(f"Nonfinite future NLL in {name}")
        future_mean = sum(float(x) for x in losses[1:]) / (target_len - 1)
        if abs(future_mean - float(arm["nll_future"])) > parity_tol:
            raise ValueError(f"nll_future excludes wrong token(s) in {name}")
        if abs(float(arm["first_target_nll"]) - float(losses[0])) > parity_tol:
            raise ValueError(f"Misaligned first target in {name}")

    for name in ("full_fixed", "evicted_fixed"):
        if any(d != native_max_depth for d in arms[name]["depths"]):
            raise ValueError(f"{name} did not execute full fixed depth")
    clean_depths = full_live["depths"]
    if full_replay["depths"] != clean_depths or evicted_replay["depths"] != clean_depths:
        raise ValueError("Recorded replay differs from FULL-LIVE depth trace")

    first_losses = [float(x["first_target_nll"]) for x in arms.values()]
    first_drift = max(first_losses) - min(first_losses)
    if first_drift > parity_tol:
        raise ValueError("First predicted target changed after post-prefill eviction")
    clean_replay_gap = max(abs(a-b) for a, b in
                           zip(full_live["token_losses"], full_replay["token_losses"]))
    if clean_replay_gap > parity_tol:
        raise ValueError("FULL-LIVE vs FULL-REPLAY output parity failed")

    clean = list(clean_depths[prompt_len:])
    changed = list(evicted_live["depths"][prompt_len:])
    delta = [new-old for old, new in zip(clean, changed)]
    fixed_penalty = evicted_fixed["nll_future"] - full_fixed["nll_future"]
    replay_penalty = evicted_replay["nll_future"] - full_live["nll_future"]
    live_penalty = evicted_live["nll_future"] - full_live["nll_future"]
    return {
        "n_future_query_tokens": target_len - 1,
        "clean_live_depths": clean, "evicted_live_depths": changed,
        "depth_delta": delta,
        "n_depth_changed": sum(x != 0 for x in delta),
        "n_depth_increased": sum(x > 0 for x in delta),
        "n_depth_decreased": sum(x < 0 for x in delta),
        "fraction_changed": sum(x != 0 for x in delta) / len(delta),
        "total_depth_full_live": sum(clean),
        "total_depth_evicted_live": sum(changed),
        "total_depth_delta": sum(delta),
        "mean_depth_delta": sum(delta) / len(delta),
        "fixed_future_nll_penalty": float(fixed_penalty),
        "replay_future_nll_penalty": float(replay_penalty),
        "live_future_nll_penalty": float(live_penalty),
        "live_minus_replay_nll_penalty_descriptive": float(live_penalty - replay_penalty),
        "full_live_replay_max_abs_nll_diff": clean_replay_gap,
        "first_target_max_nll_diff": first_drift,
        "future_token_nll_shifts": {
            mode: [float(b-a) for a, b in zip(ref["token_losses"][1:],
                                                alt["token_losses"][1:])]
            for mode, ref, alt in (
                ("fixed", full_fixed, evicted_fixed),
                ("replay", full_live, evicted_replay),
                ("live", full_live, evicted_live),
            )
        },
    }
