#!/usr/bin/env python3
"""CPU-only full-request hindsight oracle over actual Dynamic SD receipts.

1) collect: read existing dsd_run.py JSON/JSON.gz files for alternative d/K.
2) analyze: tune one global fixed action on train, compare to per-prompt
   oracle over physically measured, output-exact actions on held-out prompts.

This is NOT a token-level oracle, a deployable controller, or a GPU experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dynamic_sd.gate_probe_common import load_json


CONFIG_KEYS = ("model", "dataset", "exit", "tau", "eps", "dtype",
               "sampling", "seed", "T_total", "max_new_tokens",
               "max_prompt_tokens", "kv_s")


def digest(seq):
    return hashlib.sha256(json.dumps(seq, separators=(",", ":")).encode()).hexdigest()


def action_name(arm, cfg):
    if arm == "ar":
        return "ar"
    if arm == "wfd":
        return "wfd:d{}".format(cfg["T_draft"])
    if arm == "dtv":
        return "dtv:d{}:K{}".format(cfg["T_draft"], cfg["K"])
    raise ValueError("Unexpected arm " + str(arm))


def collect_receipts(sources):
    """Group by exact source configuration AND prompt, reject conflicting AR."""
    by_prompt = defaultdict(list)
    canonical = {}
    issues = []
    for source, payload in sources:
        cfg = payload.get("meta", {}).get("args", {})
        missing = [k for k in CONFIG_KEYS + ("T_draft", "K") if k not in cfg]
        if missing:
            raise ValueError(f"{source}: missing source arguments {missing}")
        if cfg["exit"] not in ("fixed", "live") or cfg["sampling"] != "greedy":
            raise ValueError(f"{source}: require native fixed/live greedy outputs")
        if cfg["kv_s"] not in (0, None):
            raise ValueError(f"{source}: FULL KV required; shared not equivalent")
        regime = {k: cfg[k] for k in CONFIG_KEYS}
        regime["kv_s"] = 0
        rkey = json.dumps(regime, sort_keys=True)
        for record in payload.get("per_sample", []):
            tgt = record.get("tokens_ar")
            if not isinstance(tgt, list) or not tgt:
                issues.append({"file": source, "idx": record.get("idx"),
                               "reason": "missing matched AR reference"})
                continue
            if "idx" not in record or "prefix_len" not in record:
                raise ValueError(f"{source}: missing sample identity")
            key = (rkey, record["idx"], record["prefix_len"])
            target_hash = digest(tgt)
            if key in canonical:
                if canonical[key] != target_hash:
                    canonical[key] = None
                    issues.append({"file": source, "idx": record["idx"],
                                   "reason": "AR target sequence changed across action runs"})
            else:
                canonical[key] = target_hash
            for arm in ("ar", "wfd", "dtv"):
                output, wall = record.get("tokens_" + arm), record.get("wall_" + arm)
                if not isinstance(output, list) or not isinstance(wall, (int, float)) or wall <= 0:
                    continue
                by_prompt[key].append({
                    "action": action_name(arm, cfg),
                    "elapsed_ms": float(wall) * 1000.,
                    "committed_tokens": len(output),
                    "target_digest": target_hash,
                    "target_equivalent": output == tgt,
                    "receipt": source,
                })
    rows = []
    for key, actions in sorted(by_prompt.items()):
        if canonical[key] is None:
            continue
        regime, idx, prefix_len = key
        rows.append({"regime": json.loads(regime), "idx": idx,
                     "prefix_len": prefix_len, "target_digest": canonical[key],
                     "actions": actions})
    return {"version": 1, "scope": "actual full-request receipts; original wall includes prefill",
            "groups": rows, "issues": issues}


def analyze(table, test_mod=5):
    """Held-out hindsight: never call this a physical per-token policy speedup."""
    if test_mod < 2:
        raise ValueError("test_mod must be >=2")
    regimes = defaultdict(list)
    for record in table["groups"]:
        regimes[json.dumps(record["regime"], sort_keys=True)].append(record)
    reports = []
    for regime, groups in sorted(regimes.items()):
        parsed = []
        for row in groups:
            eligible = defaultdict(list)
            for action in row["actions"]:
                if (action["target_equivalent"] and action["target_digest"] == row["target_digest"]
                        and action["committed_tokens"] > 0 and action["elapsed_ms"] > 0):
                    eligible[action["action"]].append(action["elapsed_ms"])
            if "ar" not in eligible:
                continue
            parsed.append((row, {key: statistics.median(values)
                                 for key, values in eligible.items()}))
        train, test = [], []
        for row, actions in parsed:
            h = int(hashlib.sha256(str(row["idx"]).encode()).hexdigest(), 16)
            (test if h % test_mod == 0 else train).append((row, actions))
        if not train or not test:
            reports.append({"regime": json.loads(regime), "status": "insufficient_split",
                            "n": len(parsed), "train": len(train), "test": len(test)})
            continue
        # Avoid comparing a fixed configuration that is *missing* or mismatched on
        # any sample. AR must always be a candidate.
        common = set.intersection(*(set(actions) for _, actions in parsed))
        if "ar" not in common:
            raise AssertionError("AR must be present on every eligible prefix")
        train_cost = {action: sum(options[action] for _, options in train)
                      for action in common}
        fixed = min(sorted(common), key=lambda k: train_cost[k])
        fixed_cost = sum(options[fixed] for _, options in test)
        oracle_cost = sum(min(options.values()) for _, options in test)
        ar_cost = sum(options["ar"] for _, options in test)
        coverage = {action: sum(action in options for _, options in parsed)
                    for action in sorted(set().union(*(set(options) for _, options in parsed)))}
        reports.append({
            "regime": json.loads(regime), "status": "hindsight_diagnostic_only",
            "n": len(parsed), "train": len(train), "test": len(test),
            "train_selected_fixed_action": fixed, "fixed_actions_with_full_coverage": sorted(common),
            "test_fixed_elapsed_ms": fixed_cost,
            "test_oracle_elapsed_ms": oracle_cost,
            "test_ar_elapsed_ms": ar_cost,
            "oracle_over_fixed_factor": fixed_cost / oracle_cost,
            "oracle_over_ar_factor": ar_cost / oracle_cost,
            "exact_output_eligible_action_coverage": coverage,
            "caution": "Hindsight picks only actions known to match AR and selects after timing. "
                       "This optimistic full-request upper bound is not a deployable per-token "
                       "controller or a physical speedup from changing d/K online."
        })
    return {"version": 1, "scope": "CPU-only analyses of source-reported whole-request wall time; "
            "prefill included, no true token-level common-prefix replay",
            "reports": reports, "issues": table.get("issues", [])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--table", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--test-mod", type=int, default=5)
    opts = p.parse_args()
    if opts.task == "collect":
        result = collect_receipts([(name, load_json(name)) for name in opts.inputs])
    else:
        result = analyze(load_json(opts.table), opts.test_mod)
    dest = Path(opts.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[oracle] wrote {dest}; {len(result.get('groups', result.get('reports', [])))} entries")


if __name__ == "__main__":
    main()
