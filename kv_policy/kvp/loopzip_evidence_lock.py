"""LoopZip v5 Stage 1B: evidence-locked retention sets (OFFLINE ORACLE CONTROL).

Every policy keeps the SAME annotated evidence positions E and exactly K Context tokens; only the
K-|E| non-evidence slots differ (Mean/Last/Max/Recent+Mean attention over clean prefill, seeded
Random). Uses ground-truth evidence annotation by design: never a deployable policy.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F

from .loopzip_policy import _scores, context_budget, digest

LOCK_POLICIES = ("mean_attn", "last_attn", "max_attn", "recent_attn", "random")


def lock_select(context: Sequence[int], evidence: Sequence[int], attention_by_position,
                retention: float, policy: str, *, uid: str, seed: int = 11) -> dict:
    ctx = tuple(sorted(int(p) for p in context))
    E = tuple(sorted(int(p) for p in evidence))
    K = context_budget(ctx, retention)
    if not set(E) <= set(ctx) or not E:
        raise ValueError("Evidence must be a nonempty subset of Context")
    base = {"policy": policy, "seed": seed if policy == "random" else None,
            "retention_requested": float(retention), "n_context": len(ctx), "n_retained": K,
            "n_evidence": len(E)}
    if len(E) > K:
        return {**base, "feasible": False}
    rest = [p for p in ctx if p not in set(E)]
    free = K - len(E)
    if policy == "random":
        ranking = sorted(rest, key=lambda p: (digest(f"loopzip-v5-lock/{uid}/{seed}/{p}"), p))
        chosen = ranking[:free]
    else:
        sc = _scores(ctx, attention_by_position, policy)
        ranking = sorted(rest, key=lambda p: (-sc[p], p))
        if policy == "recent_attn":
            n_recent = min(free, free // 4)  # pinned recent NON-evidence slots, inside K
            recent = sorted(rest, reverse=True)[:n_recent]
            chosen = recent + [p for p in ranking if p not in set(recent)][:free - n_recent]
        else:
            chosen = ranking[:free]
    kept = tuple(sorted(set(E) | set(chosen)))
    if len(kept) != K or not set(E) <= set(kept):
        raise RuntimeError("Evidence-locked set violates exact budget or lock")
    return {**base, "feasible": True, "retained": list(kept),
            "evicted": [p for p in ctx if p not in set(kept)],
            "nonevidence_retained": sorted(set(chosen))}


@torch.no_grad()
def teacher_forced_logprobs(cache, tokens: Sequence[int], evicted: Sequence[int] = ()) -> torch.Tensor:
    """fp32 log-probs predicting tokens[0..n-1] after the clean prompt (CachedFullPrefill)."""
    cache._reset(evicted)
    out = [F.log_softmax(cache._last_prompt_logits[0, 0].float(), -1)]
    for i in range(1, len(tokens)):
        logits = cache._step(cache.P + i - 1, int(tokens[i - 1]))
        out.append(F.log_softmax(logits[0, 0].float(), -1))
    return torch.stack(out)


def fixed_prefix_kl(lp_full: torch.Tensor, lp_pol: torch.Tensor) -> float:
    """Mean KL(p_full || p_policy) over steps 2..n (step 1 is pre-mask and identical)."""
    if lp_full.shape != lp_pol.shape or lp_full.shape[0] < 2:
        raise ValueError("Need >=2 aligned steps")
    kl = (lp_full.exp() * (lp_full - lp_pol)).sum(-1)
    return float(kl[1:].mean())
