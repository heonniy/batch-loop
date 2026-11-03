"""V10: explicit full-vs-architecture-specific shared recurrent KV policies.

All maps change ONLY reads of historical recurrent KV during decoding after
untouched full-depth prefill. There is no current-token / future-loop leakage.
FULL            r -> r                       (Huginn/Ouro control)
HUGINN-LAST     r -> R-1                     (Huginn primary sharing)
OURO-FIRSTLAST  r=0 -> 0; r>=1 -> R-1         (Ouro primary sharing)

This 'first loop plus last loop' deliberately protects the source's first
recurrent bank for the consuming query's FIRST recurrence and uses the final
bank for subsequent recurrences. It is NOT a prefill-time prompt modification.
A nearest-first/last map is only an explicitly labeled optional ablation.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

Q_GRID = (0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.98,0.995)
HUG_IN_GRID = (1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,
               3e-5,1e-4,3e-4,1e-3,3e-3,1e-2)


def sharing_name(family: str) -> str:
    if family == "huginn":
        return "last"
    if family == "ouro":
        return "first_last"
    raise ValueError("Unsupported recurrent model family: "+family)


def make_read_map(family: str, R: int, T: int, policy: str,
                  *, ouro_nearest: bool = False) -> np.ndarray | None:
    """Returns (R,T) source-plane indices or None for exact FULL identity.

    The CURRENT token is still read in its own loop's plane by the KV hook.
    r=0 means first recurrence; all planes are computed before sharing begins.
    """
    if R < 2 or T < 2:
        raise ValueError("Requires at least two recurrences and token columns")
    if policy == "full":
        return None
    if policy != sharing_name(family):
        raise ValueError(f"Invalid policy {policy} for family {family}")
    arr = np.full((R,T),R-1,dtype=np.int64)
    if family == "ouro":
        arr[0,:]=0
        if ouro_nearest:
            for r in range(R):
                arr[r,:] = 0 if r < (R-1)/2 else R-1
    return arr


def policy_planes(family: str, R: int, policy: str) -> int:
    if policy=="full":
        return R
    if policy!=sharing_name(family):
        raise ValueError(policy)
    return 1 if family=="huginn" else 2


def _values(row: Mapping, family: str) -> Sequence[float]:
    key="cumulative" if family=="ouro" else "distance_fp32"
    if key not in row:
        raise ValueError("Missing model-native exit statistic: "+key)
    return row[key]


def exit_at(stats: Mapping, family: str, threshold: float, min_depth: int=2) -> int:
    """First recurrent exit from a complete fixed-depth clean/evict state trace."""
    v=[float(x) for x in _values(stats,family)]
    R=len(v)
    if R < 2 or min_depth<1 or min_depth>R:
        raise ValueError("Invalid recurrence trace/min depth")
    if any(not math.isfinite(x) for x in v):
        raise ValueError("Invalid nonfinite exit statistic")
    if family=="ouro" and not (0<threshold<=1):
        raise ValueError("Ouro Q-exit threshold must be (0,1]")
    if family=="huginn" and not (0<threshold<1):
        raise ValueError("Huginn cosine threshold must be (0,1)")
    for r in range(min_depth,R+1):
        if r==R or (v[r-1]>=threshold if family=="ouro" else v[r-1]<threshold):
            return r
    return R


def choose_clean_thresholds(
    stats: Sequence[Mapping], family: str, targets: Sequence[float]=(0.25,0.60),
    min_depth: int=2,
) -> dict:
    """Choose threshold on pooled Full+Shared CLEAN reference traces only.

    Both KV policies MUST use exactly these thresholds in the paired test.
    Fine-grained Huginn candidates come from the clean gate-statistics
    themselves, avoiding the earlier v9 grid's early-exit saturation.
    All thresholds/quantiles are frozen before any eviction run.
    """
    if len(stats)<2 or any(not 0<=float(x)<=1 for x in targets):
        raise ValueError("Insufficient clean observations or invalid targets")
    R=len(_values(stats[0],family))
    if any(len(_values(row,family))!=R for row in stats):
        raise ValueError("Inconsistent calibration recurrent depth")
    if family=="ouro":
        grid=Q_GRID
    elif family=="huginn":
        dist=sorted(float(x) for row in stats for x in
                    _values(row,family)[min_depth-1:-1]
                    if math.isfinite(float(x)) and float(x)>0)
        # Sampling CLEAN gate statistics, not an observed eviction outcome.
        extra=np.quantile(dist,[0.02,0.05,0.10,0.20,0.35,0.50,0.65,
                                0.80,0.90,0.95,0.98]).tolist() if dist else []
        grid=tuple(sorted(set(float(x) for x in (*HUG_IN_GRID,*extra)
                              if 0<float(x)<1)))
    else:
        raise ValueError(family)
    candidates=[]
    for threshold in grid:
        ds=[exit_at(row,family,float(threshold),min_depth) for row in stats]
        hist=dict(sorted(Counter(ds).items()))
        candidates.append({"threshold":float(threshold),
                           "mean_depth":sum(ds)/len(ds),
                           "early_fraction":sum(d<R for d in ds)/len(ds),
                           "distinct_depths":len(hist),
                           "hist":{str(k):v for k,v in hist.items()}})
    chosen=[]
    for target in targets:
        available=[x for x in candidates if x["threshold"] not in
                   {v["threshold"] for v in chosen}]
        if not available:
            raise ValueError("Not enough distinct thresholds for targets")
        row=min(available,key=lambda x:(abs(x["early_fraction"]-target)
                         +(0.10 if x["distinct_depths"]<2 else 0),
                         abs(x["mean_depth"]-R*0.75),x["threshold"]))
        chosen.append({**row,"target_early_fraction":float(target),
                       "saturated":row["distinct_depths"]<2 or
                         row["early_fraction"]<0.01 or
                         row["early_fraction"]>0.99})
    return {"family":family,"calibration_from":"full_and_shared_clean_fixed_depth",
            "n_clean_query_observations":len(stats),
            "min_depth":min_depth,
            "cosine_arithmetic":"fp32_on_bf16_states" if family=="huginn" else None,
            "selected":chosen,"candidates":candidates}


def compare_exit_stats(clean: Sequence[Mapping], evicted: Sequence[Mapping],
                       family: str, threshold: float) -> dict:
    """Offline would-exit diagnostics from FULL-depth traces (NOT live depths)."""
    if len(clean)!=len(evicted):
        raise ValueError("Unpaired exit traces")
    cd,ed=[],[]
    max_abs_stats=[]
    for a,b in zip(clean,evicted):
        aa=np.asarray(_values(a,family),dtype=float)
        bb=np.asarray(_values(b,family),dtype=float)
        if aa.shape!=bb.shape:
            raise ValueError("Unequal depth statistics")
        cd.append(exit_at(a,family,threshold))
        ed.append(exit_at(b,family,threshold))
        max_abs_stats.append(float(np.max(np.abs(bb-aa))))
    delta=[b-a for a,b in zip(cd,ed)]
    return {"full_would_exit_depths":cd,"evict_would_exit_depths":ed,
            "depth_delta":delta,"fraction_changed":sum(x!=0 for x in delta)/len(delta)
            if delta else None,
            "mean_abs_exit_stat_shift":sum(max_abs_stats)/len(max_abs_stats)
            if max_abs_stats else None}


def paired_trajectory(clean: Sequence[np.ndarray], evicted: Sequence[np.ndarray],
                      *, norm_floor: float=1e-8) -> dict:
    """Full-depth trajectory error for identical frozen generated queries.

    Input entries are h[0:R+1, dim] after the same model observer checkpoint.
    Reports both absolute and relative differences and within-run update sizes.
    A large relative error at BF16 numerical floor is not mechanism evidence.
    """
    if len(clean)!=len(evicted) or not clean:
        raise ValueError("Unpaired clean/evicted query trajectories")
    n=len(clean)
    E,delta,U0,U1=[],[],[],[]
    for a,b in zip(clean,evicted):
        a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        if a.shape!=b.shape or a.ndim!=2:
            raise ValueError("Hidden state shape mismatch")
        da=np.linalg.norm(a[1:]-a[:-1],axis=-1)
        db=np.linalg.norm(b[1:]-b[:-1],axis=-1)
        norm=np.linalg.norm(a,axis=-1)
        absolute=np.linalg.norm(b-a,axis=-1)
        E.append(absolute/np.maximum(norm,norm_floor))
        delta.append(absolute)
        U0.append(da/np.maximum(norm[:-1],norm_floor))
        U1.append(db/np.maximum(np.linalg.norm(b[:-1],axis=-1),norm_floor))
    return {"n_queries":n,"R":len(E[0])-1,
            "mean_relative_error_by_loop":np.mean(E,axis=0).tolist(),
            "mean_absolute_error_by_loop":np.mean(delta,axis=0).tolist(),
            "mean_clean_update_by_loop":np.mean(U0,axis=0).tolist(),
            "mean_evict_update_by_loop":np.mean(U1,axis=0).tolist(),
            "per_query_final_relative_error":[float(v[-1]) for v in E],
            "per_query_relative_error": [v.tolist() for v in E]}


def same_depth_nll_interaction(full_keep: float, full_evict: float,
                               shared_keep: float, shared_evict: float) -> dict:
    """Difference-in-differences: positive => sharing suffers more from eviction."""
    a=float(full_evict)-float(full_keep)
    b=float(shared_evict)-float(shared_keep)
    return {"full_eviction_penalty":a,"shared_eviction_penalty":b,
            "sharing_minus_full_eviction_penalty":b-a,
            "sharing_baseline_nll_shift":float(shared_keep)-float(full_keep)}
