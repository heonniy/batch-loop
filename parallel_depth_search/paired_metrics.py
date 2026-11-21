"""Paired bootstrap for real generated-output comparisons; no CUDA needed.

Input schema is the E5 JSONL from parallel_depth_poc/e5_decoder.py. Use ONLY
for genuine complete runs in the same model, split, batch, prompt/stop protocol
and generator; reporting matched quality requires separate KV checks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def percentile(values, p):
    s = sorted(values)
    if not s:
        raise ValueError("empty sample")
    i = (len(s) - 1) * p
    lo = int(math.floor(i)); hi = int(math.ceil(i))
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def read_rows(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("empty data")
    return rows


def align(rows, baseline, candidate):
    grouped = {}
    for item in rows:
        k = (item["prompt_idx"], item["arm"])
        if k in grouped:
            raise ValueError(f"duplicate prompt,arm {k}")
        grouped[k] = item
    ids = sorted({i for i, arm in grouped if arm == baseline} &
                 {i for i, arm in grouped if arm == candidate})
    if not ids:
        raise ValueError("no paired prompts")
    return [(grouped[(i, baseline)], grouped[(i, candidate)]) for i in ids]


def aggregate(pairs):
    n = len(pairs)
    ba = [a for a, _ in pairs]
    ca = [b for _, b in pairs]
    for row in ba + ca:
        if row["wall_s"] <= 0 or row["n_tokens"] <= 0:
            raise ValueError("nonpositive wall or token count")
    nt_b = sum(x["n_tokens"] for x in ba)
    nt_c = sum(x["n_tokens"] for x in ca)
    wt_b = sum(x["wall_s"] for x in ba)
    wt_c = sum(x["wall_s"] for x in ca)
    speedup = (nt_c / wt_c) / (nt_b / wt_b)
    req_speedup = wt_b / wt_c
    acc_delta = (sum(bool(x["correct_strict"]) for x in ca) -
                 sum(bool(x["correct_strict"]) for x in ba)) / n
    if any("tokens" not in a or "tokens" not in b for a, b in pairs):
        raise ValueError("cannot compute sequence exactness without generated token IDs")
    seq_exact = sum(a["tokens"] == b["tokens"] for a, b in pairs) / n
    return {
        "sample_count": n, "speedup": speedup,
        "requests_per_sec_speedup": req_speedup,
        "accuracy_delta": acc_delta, "sequence_exact_rate": seq_exact,
        "baseline_accuracy": sum(bool(x["correct_strict"]) for x in ba) / n,
        "candidate_accuracy": sum(bool(x["correct_strict"]) for x in ca) / n,
        "baseline_tokens": nt_b, "candidate_tokens": nt_c,
        "output_length_ratio": nt_c / nt_b,
        "baseline_wall_s": wt_b, "candidate_wall_s": wt_c,
        "all_outputs_same_length": all(a["n_tokens"] == b["n_tokens"] for a, b in pairs)
    }


def paired_bootstrap(pairs, reps=2000, seed=20261121):
    if reps < 200 or len(pairs) < 2:
        raise ValueError("need >=200 bootstrap reps and >=2 independent prompts")
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        estimates.append(aggregate(sample))
    central = aggregate(pairs)
    for name in ("speedup", "requests_per_sec_speedup", "accuracy_delta", "sequence_exact_rate"):
        central[name + "_ci_low"] = percentile([x[name] for x in estimates], 0.025)
        central[name + "_ci_high"] = percentile([x[name] for x in estimates], 0.975)
    central.update({
        "bootstrap_reps": reps, "bootstrap_seed": seed,
        "paired_prompt_unit": True,
        "full_e2e_measured": True,
        "unverified_kv_fidelity": True,  # never infer cache parity from tokens
        "type": "FREE_RUNNING_MODEL_SOURCE_REPORTED",
        "warning": "Speedup in tokens/s can reflect changed output length; inspect requests/s and completion quality. "
                   "Do not claim matched-B quality or exact cache without independent checks."
    })
    return central


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", default="AR")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261121)
    a = p.parse_args()
    pairs = align(read_rows(a.input), a.baseline, a.candidate)
    result = paired_bootstrap(pairs, reps=a.reps, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": a.out, "n": len(pairs),
                      "speedup": result["speedup"],
                      "speedup_ci_low": result["speedup_ci_low"],
                      "accuracy_delta_ci_low": result["accuracy_delta_ci_low"]}))


if __name__ == "__main__":
    main()
