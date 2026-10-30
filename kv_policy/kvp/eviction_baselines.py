"""Fair pre-eviction selection policies over the SAME candidate token set.

These are ranking baselines applied to the single-source eviction oracle,
not implementations of real simultaneous multi-token cache eviction.
Only attention observed before eviction is allowed; future attention is
strictly excluded (an oracle-only diagnostic in the raw data).
H2O proxy uses accumulated past attention; StreamingLLM proxy uses
initial sink tokens + recency. These are not exact upstream code ports.
"""
from __future__ import annotations

import hashlib
import random


def score(row: dict, policy: str, *, seed: int = 20261030) -> float:
    """Higher score means RETAIN; all policies use identical token budget."""
    d = float(row["exit_depth"])
    a = float(row["attention_pre"])
    age = float(row["age_at_eval"])
    p = int(row["source_position"])
    if policy == "depth":
        return d
    if policy == "reverse_depth":
        return -d
    if policy == "attention_h2o_proxy":
        return a
    if policy == "attention_normalized":
        return float(row["attention_pre_per_opportunity"])
    if policy == "recency":
        return -age
    if policy == "streaming_sink_recent_proxy":
        # First four source positions are always pinned, then recent ones.
        return 1e9 if p < 4 else -age
    if policy == "random":
        raw=f'{seed}/{row["uid"]}/{p}'.encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")/2**64
    raise ValueError("Unknown offline ranking")


POLICIES=("depth","reverse_depth","attention_h2o_proxy",
          "attention_normalized","recency","streaming_sink_recent_proxy",
          "random")


def evaluate_singleton_oracle(rows: list[dict], retain_fraction: float = .5) -> list[dict]:
    """Score diagnostic: sum of U for tokens a policy WOULD evict individually.

    NOT the NLL of jointly removing these tokens, which is nonadditive!
    This is for prioritizing later online implementation, not benchmark wins.
    """
    if not (0 < retain_fraction < 1):
        raise ValueError("retain_fraction must be in (0,1)")
    by_request={}
    for row in rows:
        by_request.setdefault((row["model"],row["dataset"],row["uid"]),[]).append(row)
    out=[]
    for (model,dataset,uid),group in sorted(by_request.items()):
        n=len(group)
        if n<2: continue
        retain=max(1,min(n-1,round(n*retain_fraction)))
        for policy in POLICIES:
            ranked=sorted(group,key=lambda r:(-score(r,policy),int(r["source_position"])))
            evicted=ranked[retain:]
            out.append({"model":model,"dataset":dataset,"uid":uid,
                        "policy":policy,"n_candidates":n,"n_retained":retain,
                        "n_evicted":len(evicted),
                        "sum_individual_delta_nll_ORACLE_DIAGNOSTIC":
                            sum(float(r["delta_nll"]) for r in evicted),
                        "evicted_source_positions":";".join(
                            str(int(r["source_position"])) for r in evicted)})
    return out
