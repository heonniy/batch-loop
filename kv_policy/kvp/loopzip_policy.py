"""Frozen, annotation-blind LoopZip Phase-1 retention and Phase-2 swap selection.

CPU-only; importable without torch/model weights. All source positions are
aligned Context token IDs, and every retained position keeps *all* recurrent
loop banks/physical layers. There are no hidden extra pinned Context tokens.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Mapping, Sequence

BUDGETS = (0.10, 0.30, 0.50)
ATTENTION_POLICIES = ("mean_attn", "last_attn", "max_attn", "recent_attn")
ALL_POLICIES = ATTENTION_POLICIES + ("random",)
RANDOM_SEEDS = (11, 29, 47)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def context_budget(context_positions: Sequence[int], retention: float) -> int:
    """The same floor-rounded number of Context positions for every policy."""
    positions = tuple(context_positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Context positions must be nonempty and unique")
    if not math.isfinite(retention) or not (0 < retention <= 1):
        raise ValueError("Retention must lie in (0, 1]")
    return max(1, math.floor(retention * len(positions)))


def _scores(context_positions: Sequence[int],
            attention_by_position: Mapping[int | str, Sequence[float]],
            policy: str) -> dict[int, float]:
    """Normalized mean head/layer/query mass per recurrent loop from CLEAN prompt.

    The caller must provide only Instruction/Question -> prior Context Attention
    recorded during the FULL prompt prefill, never future/generated attention.
    """
    if policy not in ATTENTION_POLICIES:
        raise ValueError("Scores require an attention-based policy")
    scores = {}
    n_loops = None
    for p in context_positions:
        profile = attention_by_position.get(str(p),
                   attention_by_position.get(p))
        if profile is None:
            raise ValueError(f"Missing clean prefill attention for Context position {p}")
        values = tuple(float(x) for x in profile)
        if not values or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError(f"Invalid attention profile for {p}")
        if n_loops is None:
            n_loops = len(values)
        if n_loops != len(values):
            raise ValueError("Inconsistent recurrent attention profile depths")
        if policy in ("mean_attn", "recent_attn"):
            score = sum(values)/len(values)
        elif policy == "last_attn":
            score = values[-1]
        else:
            score = max(values)
        scores[int(p)] = score
    return scores


def choose_retained(context_positions: Sequence[int],
                    attention_by_position: Mapping[int | str, Sequence[float]],
                    retention: float, policy: str, *,
                    uid: str, random_seed: int = 11) -> dict:
    """Return exactly K retained Context token IDs and their common complement.

    No supporting-fact label, question answer, future NLL, or generated-token
    attention is an input. The scoring rule is frozen across all budgets.
    """
    positions = tuple(sorted(int(x) for x in context_positions))
    if not uid or len(set(positions)) != len(positions):
        raise ValueError("Missing UID or repeated Context token")
    if policy not in ALL_POLICIES:
        raise ValueError(f"Unknown retention policy {policy}")
    K = context_budget(positions, retention)
    if policy == "random":
        ranking = sorted(positions,
                         key=lambda p: (digest(f"loopzip-v1/{uid}/{random_seed}/{p}"),p))
        source_scores = None
    else:
        source_scores = _scores(positions,attention_by_position,policy)
        ranking = sorted(positions,key=lambda p:(-source_scores[p],p))
    if policy == "recent_attn":
        # Pinned recent Context tokens are INSIDE the K budget.
        recent_count = min(K, K//4)
        recent = sorted(positions,reverse=True)[:recent_count]
        chosen = recent + [p for p in ranking if p not in recent][:K-recent_count]
    else:
        chosen = ranking[:K]
    selected = tuple(sorted(chosen))
    if len(selected) != K or len(set(selected)) != K:
        raise RuntimeError("Policy exceeded or undershot Context KV budget")
    complement = tuple(p for p in positions if p not in selected)
    if len(selected)+len(complement)!=len(positions):
        raise RuntimeError("Selected + evicted Context sets do not partition the pool")
    return {
        "policy":policy, "seed": random_seed if policy=="random" else None,
        "retention_requested":float(retention),
        "n_context":len(positions),"n_retained":K,
        "actual_retention":K/len(positions),
        "retained":list(selected),"evicted":list(complement),
        "scores": {str(p):source_scores[p] for p in positions}
                  if source_scores is not None else None,
    }


def selection_manifest(context_positions: Sequence[int],
                       attention_by_position: Mapping[int | str, Sequence[float]],
                       uid: str, budgets: Sequence[float] = BUDGETS,
                       policies: Sequence[str] = ALL_POLICIES,
                       random_seeds: Sequence[int] = RANDOM_SEEDS) -> list[dict]:
    """All policies and budgets paired within each request."""
    selections=[]
    for b in budgets:
        for policy in policies:
            seeds = random_seeds if policy=="random" else (0,)
            for seed in seeds:
                selection=choose_retained(context_positions,attention_by_position,
                                          b,policy,uid=uid,random_seed=seed)
                selection["policy_key"]=(f"random_s{seed}" if policy=="random"
                                         else policy)
                selections.append(selection)
    if len({(r["retention_requested"],r["policy_key"]) for r in selections})!=len(selections):
        raise RuntimeError("Duplicate policy/budget label")
    return selections


def swap_neighbors(original: Mapping, context_positions: Sequence[int],
                   attention_by_position: Mapping[int | str, Sequence[float]], *,
                   uid: str, max_retained: int = 6, max_evicted: int = 6,
                   seed: int = 20261031) -> list[dict]:
    """Fixed-size local exchange neighborhood with boundary and tail probes.

    The unmodified original is always included as id=original. Return no gold
    outcome: the *offline evaluator* later scores each neighbor with NLL.
    Candidate design includes:
    - low-score retained positions nearest the top-K boundary;
    - high-score evicted positions near the boundary;
    - low-score evicted tail and deterministic positional diversity.
    Hence the search is not limited to a trivially wrong cutoff.
    """
    if max_retained<1 or max_evicted<1:
        raise ValueError("Swap probe counts must be positive")
    positions=tuple(sorted(set(int(p) for p in context_positions)))
    base=tuple(sorted(int(p) for p in original["retained"]))
    evicted=tuple(sorted(int(p) for p in original["evicted"]))
    if set(base)&set(evicted) or set(base)|set(evicted)!=set(positions):
        raise ValueError("Original retained/evicted sets are not a Context partition")
    if len(base)!=int(original["n_retained"]):
        raise ValueError("Inconsistent original budget")
    # Use Mean/Last/Max ranking of the *source policy*; fallback Mean for
    # random or recent for consistent boundary probes.
    key=original["policy"]
    scores=_scores(positions,attention_by_position,
                   key if key in ATTENTION_POLICIES else "mean_attn")
    remove=sorted(base,key=lambda p:(scores[p],p))[:max_retained]
    high=sorted(evicted,key=lambda p:(-scores[p],p))
    low=sorted(evicted,key=lambda p:(scores[p],p))
    candidate_add=high[:max(1,max_evicted//2)]
    if max_evicted>1:
        for p in low[:max(1,max_evicted//3)]:
            if p not in candidate_add and len(candidate_add)<max_evicted:
                candidate_add.append(p)
    rest=[p for p in evicted if p not in candidate_add]
    rng=random.Random(digest(f"swap/{seed}/{uid}/{original['retention_requested']}"))
    rng.shuffle(rest)
    for p in rest:
        if len(candidate_add)>=max_evicted:
            break
        candidate_add.append(p)
    outputs=[{"swap_id":"original", "retained":list(base),
              "evicted":list(evicted),"drop":None,"restore":None,
              "type":"original"}]
    for a in remove:
        for e in candidate_add:
            new_set=tuple(sorted((set(base)-{a})|{e}))
            if len(new_set)!=len(base):
                raise RuntimeError("Swap violated exact-K budget")
            outputs.append({
                "swap_id":f"remove-{a}-restore-{e}",
                "retained":list(new_set),
                "evicted":[p for p in positions if p not in new_set],
                "drop":a,"restore":e,
                "type":("boundary" if e in high[:max(1,max_evicted//2)]
                        else "tail_or_diverse")})
    if len({tuple(v["retained"]) for v in outputs})!=len(outputs):
        raise RuntimeError("Duplicate swap neighbor")
    return outputs


def classify_failure(full_correct: bool | None, quality_by_policy: Mapping[str, Mapping]) -> str:
    """Classify only when free-generation correctness is available."""
    if full_correct is None:
        return "nll_only_unclassified"
    if not full_correct:
        return "full_wrong"
    attn = [quality_by_policy[p].get("greedy_correct")
            for p in ATTENTION_POLICIES if p in quality_by_policy]
    if not attn or any(v is None for v in attn):
        return "accuracy_missing"
    if all(not v for v in attn):
        return "all_attention_wrong"
    if quality_by_policy["mean_attn"].get("greedy_correct") is False and any(attn):
        return "mean_wrong_other_attention_correct"
    return "no_attention_failure" if all(attn) else "mixed_attention_failure"
