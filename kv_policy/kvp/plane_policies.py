"""v8 loop-plane (per-recurrence KV bank) retention policies for looped transformers — CPU only.

A prompt token i of a recurrent-depth model owns R KV planes (one per recurrence r). A *plane set* S_i ⊆ {0..R-1}
is what is kept; a later query at recurrence r reads plane ``map[r, i] = nearest kept plane`` (ties -> the later
plane) or, in the 'latest' semantics, the latest kept plane <= r (else the first kept one). The budget is counted in
planes over the Context tokens only: tokens evicted outright contribute 0, kept tokens |S_i|.

All builders are deterministic and label-free: they take clean prefill quantities (settle depth of the token's own
trajectory, attention received per loop from later prompt queries) or a token score, never an outcome.
"""
from __future__ import annotations

import numpy as np

PLANE_POLICIES = ("full", "last", "first_last", "even", "settle", "attn")
MAP_SEMANTICS = ("nearest", "latest")


def plane_map_from_sets(sets: dict[int, np.ndarray], R: int, P: int, semantics: str = "nearest") -> np.ndarray:
    """(R, P) int map; prompt columns not in `sets` keep the identity (all planes)."""
    if semantics not in MAP_SEMANTICS:
        raise ValueError(semantics)
    m = np.tile(np.arange(R, dtype=np.int64)[:, None], (1, P))
    for i, S in sets.items():
        S = np.unique(np.asarray(S, dtype=np.int64))
        if S.size == 0 or S.min() < 0 or S.max() >= R:
            raise ValueError(f"Invalid plane set for token {i}: {S}")
        for r in range(R):
            if semantics == "nearest":
                d = np.abs(S - r)
                cand = S[d == d.min()]
                m[r, i] = int(cand.max())          # tie -> later plane
            else:
                le = S[S <= r]
                m[r, i] = int(le.max()) if le.size else int(S.min())
    return m


def _even(k: int, lo: int, hi: int) -> np.ndarray:
    """k distinct planes evenly spread over [lo, hi], always including hi (the terminal plane)."""
    k = max(1, min(k, hi - lo + 1))
    if k == 1:
        return np.array([hi])
    return np.unique(np.round(np.linspace(lo, hi, k)).astype(np.int64))


def plane_sets(policy: str, kept: list[int], R: int, k: int, *, settle: np.ndarray | None = None,
               recv: np.ndarray | None = None, plane_budget: int | None = None) -> dict[int, np.ndarray]:
    """Per-kept-token plane sets under policy with k planes per token (uniform policies) or a global plane budget
    (settle: adaptive allocation proportional to the token's settle depth).

    settle[i]: first loop index r (0-based plane index) at which the token's own clean trajectory is within the
               settle tolerance of its terminal state (v6 F metric). recv[r, i]: attention received at loop r.
    """
    if policy not in PLANE_POLICIES:
        raise ValueError(policy)
    k = int(max(1, min(k, R)))
    out = {}
    if policy == "full":
        return {i: np.arange(R) for i in kept}
    if policy == "last":
        return {i: np.arange(R - k, R) for i in kept}
    if policy == "first_last":
        return {i: np.unique(np.r_[0, np.arange(R - (k - 1), R)]) if k > 1 else np.array([R - 1]) for i in kept}
    if policy == "even":
        return {i: _even(k, 0, R - 1) for i in kept}
    if policy == "attn":
        if recv is None:
            raise ValueError("attn policy needs per-loop received attention")
        for i in kept:
            order = np.argsort(-recv[:, i], kind="stable")[:k]
            out[i] = np.unique(np.r_[order, R - 1]) if k > 1 else np.array([int(order[0])])
            if len(out[i]) > k:          # keep exactly k: drop the weakest non-terminal plane
                weakest = min((p for p in out[i] if p != R - 1), key=lambda p: recv[p, i])
                out[i] = np.array([p for p in out[i] if p != weakest])
        return out
    # settle: adaptive — planes evenly over [0, settle_i] with n_i ∝ (settle_i + 1) under the global budget
    if settle is None:
        raise ValueError("settle policy needs settle depths")
    budget = int(plane_budget if plane_budget is not None else k * len(kept))
    need = np.array([settle[i] + 1 for i in kept], dtype=float)      # planes needed to describe the trajectory
    n_i = np.maximum(1, np.floor(budget * need / need.sum())).astype(int)
    n_i = np.minimum(n_i, R)
    # distribute any remainder to the tokens with the largest unmet need
    rem = budget - int(n_i.sum())
    if rem > 0:
        order = np.argsort(-(need - n_i))
        for j in order[:rem]:
            if n_i[j] < R:
                n_i[j] += 1
    for i, n in zip(kept, n_i):
        hi = int(settle[i])
        S = _even(int(n), 0, hi) if hi > 0 else np.array([0])
        out[i] = S
    return out


def budget_planes(sets: dict[int, np.ndarray]) -> int:
    return int(sum(len(np.unique(S)) for S in sets.values()))


def two_axis_arms(content: list[int], R: int, budget: float, token_scores: dict[str, np.ndarray],
                  settle: np.ndarray, recv: np.ndarray, *, mixed_scores=("random_11", "def4_rR", "fut_oracle_rR"),
                  mixed_planes=("last", "even", "settle", "attn")) -> list[dict]:
    """Enumerate equal-plane-budget arms for one prompt: token-only (planes full), plane-only (all tokens), mixed
    (token fraction sqrt(b) x plane fraction sqrt(b)). Returns dicts with evicted tokens, plane sets, counted budget."""
    total = len(content) * R
    target = int(round(budget * total))
    arms = []

    def keep_by_score(name, frac):
        sc = token_scores[name]
        n_keep = max(1, int(round(frac * len(content))))
        order = sorted(range(len(content)), key=lambda j: (-float(sc[content[j]]), int(content[j])))
        kept = sorted(int(content[j]) for j in order[:n_keep])
        evicted = sorted(set(int(c) for c in content) - set(kept))
        return kept, evicted

    for name in token_scores:
        kept, ev = keep_by_score(name, budget)
        sets = plane_sets("full", kept, R, R)
        arms.append({"axis": "token", "token_score": name, "plane_policy": "full", "token_frac": budget,
                     "planes_per_token": R, "evicted": ev, "sets": sets, "planes": budget_planes(sets), "target": target})
    k = max(1, int(round(budget * R)))
    for pol in ("last", "first_last", "even", "settle", "attn"):
        kept = [int(c) for c in content]
        sets = plane_sets(pol, kept, R, k, settle=settle, recv=recv, plane_budget=target if pol == "settle" else None)
        arms.append({"axis": "plane", "token_score": "none", "plane_policy": pol, "token_frac": 1.0,
                     "planes_per_token": k, "evicted": [], "sets": sets, "planes": budget_planes(sets), "target": target})
    tf = float(np.sqrt(budget))
    km = max(1, int(round(np.sqrt(budget) * R)))
    for name in mixed_scores:
        if name not in token_scores:
            continue
        kept, ev = keep_by_score(name, tf)
        for pol in mixed_planes:
            sets = plane_sets(pol, kept, R, km, settle=settle, recv=recv,
                              plane_budget=min(target, len(kept) * R) if pol == "settle" else None)
            arms.append({"axis": "mixed", "token_score": name, "plane_policy": pol, "token_frac": tf,
                         "planes_per_token": km, "evicted": ev, "sets": sets, "planes": budget_planes(sets), "target": target})
    return arms
