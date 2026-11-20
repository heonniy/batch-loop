"""Conservative, split-aware Pareto shortlisting from actual evidence receipts.

No LLM claims are inferred from this script: it sorts compatible measured
candidates, preserves uncertainty and diverse method families, and asks the
coding agent to review primary paper assumptions before promoting anything.
"""
from __future__ import annotations

import argparse
import json
import math

from .control import program, read_ledger


def context_key(r):
    m = r.get("metrics", {})
    hw = r.get("hardware", {})
    return (
        r.get("model"), m.get("depth"), m.get("batch"),
        m.get("prompt_len"), m.get("dtype"), m.get("kv_policy"),
        r.get("data_split"), hw.get("gpu_name")
    )


def family(r):
    kind = str(r.get("method", "")).lower()
    if "anderson" in kind or "taa" in kind:
        return "anderson"
    if "newton" in kind or "identity" in kind or "jvp" in kind or "low_rank" in kind:
        return "newton"
    if "parareal" in kind or "coarse" in kind:
        return "parareal"
    if "jacobi" in kind or "picard" in kind:
        return "jacobi"
    if "select" in kind or "window" in kind or "sparse" in kind:
        return "allocation"
    return "other"


def assess(r, cfg):
    if r["state"] != "executed":
        return {"rankable": False, "reason": r["state"]}
    m = r["metrics"]
    n = int(m.get("sample_count", 0))
    if r["data_split"] in ("test", "hardware"):
        return {"rankable": False, "reason": "held-out test or hardware only, not adaptive selection"}
    if n < cfg["screen_min_samples"]:
        return {"rankable": False, "reason": "insufficient sample count"}
    if any(x is None for x in context_key(r)):
        return {"rankable": False, "reason": "missing matched benchmark context"}
    if m.get("full_e2e_measured"):
        if r["data_split"] != "dev" or n < cfg["dev_min_samples"]:
            return {"rankable": False, "reason": "screen only: require dev>=200 for E2E promotion"}
        lo, acc = m.get("speedup_ci_low"), m.get("accuracy_delta_ci_low")
        if lo is None or acc is None:
            return {"rankable": False, "reason": "missing paired speed/accuracy CI"}
        tier = m.get("fidelity_tier")
        if tier not in ("exact", "quality_noninferior"):
            return {"rankable": False, "reason": "not matched-quality claim"}
        if tier == "exact" and (
            m.get("unverified_kv_fidelity") is not False
            or m.get("sequence_exact_rate", 0) < 1.0
        ):
            return {"rankable": False, "reason": "exact claim lacks sequence + KV verification"}
        if acc < -cfg["noninferiority_margin_accuracy"]:
            return {"rankable": False, "reason": "fails preregistered quality noninferiority"}
        if lo <= cfg["minimum_speedup_lower_ci"]:
            return {"rankable": False, "reason": "no confident complete-model speedup"}
        return {"rankable": True, "tier": "measured_full_e2e",
                "value": math.log(max(lo, 1e-8)), "reason": "paired CI, online E2E and quality passed"}
    # Diagnostic-only evidence cannot establish a deployed speedup. It is
    # useful to decide which family deserves the next expensive experiment.
    if not all(k in m for k in ("mean_kl", "mean_solver_ms", "mean_top1")):
        return {"rankable": False, "reason": "diagnostic lacks comparable KL, wall and top1"}
    if m["mean_solver_ms"] <= 0:
        return {"rankable": False, "reason": "invalid solver wall"}
    if m["mean_top1"] < 0.90:
        return {"rankable": False, "reason": "poor next-token stability; investigate, not global NO-GO"}
    score = (m["mean_top1"] - m["mean_kl"]) / m["mean_solver_ms"]
    return {"rankable": True, "tier": "diagnostic_only",
            "value": score, "reason": "exploratory score; no E2E, KV or quality claim"}


def shortlist(ledger_rows, cfg=None, keep=3):
    cfg = cfg or program()
    groups, discarded = {}, []
    for e in ledger_rows:
        r = e["receipt"]
        a = assess(r, cfg)
        row = {"trial_id": r["trial_id"], "node_id": r["node_id"],
               "method": r["method"], "family": family(r),
               "context": list(context_key(r)), "assessment": a}
        if not a["rankable"]:
            discarded.append(row)
            continue
        groups.setdefault(context_key(r), []).append(row)
    candidates = {}
    for key, items in groups.items():
        items.sort(key=lambda x: -x["assessment"]["value"])
        champions, diverse = [], set()
        for x in items:
            if len(champions) >= keep:
                break
            if x["family"] not in diverse:
                champions.append(x)
                diverse.add(x["family"])
        for x in items:
            if len(champions) >= keep:
                break
            if x not in champions:
                champions.append(x)
        candidates[str(key)] = champions
    return {
        "status": "RECOMMENDATION_ONLY_REQUIRES_METHOD_PAPER_REVIEW",
        "grouped_candidate_shortlists": candidates,
        "excluded": discarded,
        "not_global_no_go": True,
        "reserve_for_alternative_families": 0.20,
        "no_test_split_used": True
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--keep", type=int, default=3)
    a = p.parse_args()
    result = shortlist(read_ledger(a.run_dir), keep=a.keep)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
