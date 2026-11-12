#!/usr/bin/env python3
"""Offline one-decision oracle screen on ACTUALLY measured paired prefixes.

Not an achievable rollout, not a cache-equivalence proof, and not a GPU
benchmark. Refuses to compare non-identical prefix/target contracts.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


FIELDS = ("prefix_id", "prefix_hash", "target_hash", "exit_policy",
          "kv_policy", "batch_regime", "backend", "precision",
          "action", "wall_ms", "committed_tokens", "target_exact",
          "measured", "repeat")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for lineno, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            obj = json.loads(raw)
            missing = set(FIELDS) - obj.keys()
            if missing:
                raise ValueError(f"line {lineno}: missing {sorted(missing)}")
            yield obj


def analyze(rows):
    groups = defaultdict(list)
    contract_keys = ("prefix_hash", "target_hash", "exit_policy",
                     "kv_policy", "batch_regime", "backend", "precision")
    for row in rows:
        if row["measured"] is not True:
            raise ValueError("oracle requires physically measured action trials")
        if row["target_exact"] is not True:
            raise ValueError("oracle requires strict target-exact action trials; report failures separately")
        if not isinstance(row["wall_ms"], (int,float)) or row["wall_ms"] <= 0:
            raise ValueError("invalid wall_ms")
        if not isinstance(row["committed_tokens"], int) or row["committed_tokens"] < 1:
            raise ValueError("invalid committed_tokens")
        groups[row["prefix_id"]].append(row)
    if len(groups) < 1:
        raise ValueError("no counterfactual records")
    action_set = None
    prefix_metrics = {}
    fixed_total = defaultdict(lambda: [0.0,0.0])
    oracle_time = 0.0
    oracle_tokens = 0.0
    for prefix, trials in sorted(groups.items()):
        first = trials[0]
        for t in trials:
            if any(t[k] != first[k] for k in contract_keys):
                raise ValueError(f"nonidentical prefix/target/runtime contract: {prefix}")
        per_action = defaultdict(list)
        for t in trials:
            per_action[t["action"]].append(t)
        actions = set(per_action)
        if action_set is None:
            action_set = actions
        if action_set != actions:
            raise ValueError("incomplete action grid: each prefix needs same action set")
        repeat_sets = {k: {x["repeat"] for x in v} for k,v in per_action.items()}
        if len({tuple(sorted(v)) for v in repeat_sets.values()}) != 1:
            raise ValueError(f"inconsistent repeats at {prefix}")
        # Aggregate the actions' executed time across repeats; committed tokens
        # can differ across valid actions, so report tokens and time together.
        scores = {}
        for action, entries in per_action.items():
            tok = sum(x["committed_tokens"] for x in entries)
            ms = sum(x["wall_ms"] for x in entries)
            scores[action] = {"tokens":tok, "wall_ms":ms, "tokens_per_ms":tok/ms}
            fixed_total[action][0] += tok
            fixed_total[action][1] += ms
        best = max(sorted(scores), key=lambda a:scores[a]["tokens_per_ms"])
        prefix_metrics[prefix] = {"oracle_action":best, "action_scores":scores}
        oracle_tokens += scores[best]["tokens"]
        oracle_time += scores[best]["wall_ms"]
    static = {a:{"tokens":v[0],"wall_ms":v[1],"tokens_per_ms":v[0]/v[1]}
              for a,v in sorted(fixed_total.items())}
    best_static = max(sorted(static),key=lambda a:static[a]["tokens_per_ms"])
    oracle_ratio = (oracle_tokens/oracle_time)/static[best_static]["tokens_per_ms"]
    return {
        "n_prefixes":len(groups),"actions":sorted(action_set),
        "best_static_action":best_static,"static":static,
        "one_decision_optimistic_oracle_tokens_per_ms":oracle_tokens/oracle_time,
        "optimistic_ratio_vs_best_static":oracle_ratio,
        "by_prefix":prefix_metrics,
        "IMPORTANT":"This is a per-prefix optimistic counterfactual screen. It does not compose into a multi-token rollout and excludes controller cost; no achieved speedup."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    data=analyze(list(read_jsonl(args.input)))
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,indent=1)+"\n",encoding="utf-8")
    print(json.dumps({"best_static_action":data["best_static_action"],
                      "optimistic_ratio_vs_best_static":data["optimistic_ratio_vs_best_static"],
                      "n_prefixes":data["n_prefixes"]},indent=1))


if __name__=="__main__":
    main()
