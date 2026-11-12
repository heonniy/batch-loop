#!/usr/bin/env python3
"""Check paired off/gpu/cpu replay invariants before reporting latency deltas."""
import argparse
import json
from pathlib import Path
import statistics


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def geometry(row, arm):
    # Width histogram keys may be strings after serialization.
    return (row.get(f"tokens_{arm}"), row.get(f"rcalls_{arm}"),
            row.get(f"rows_{arm}"), row.get(f"width_{arm}"),
            row.get("depths_ar") if arm == "ar" else row.get(f"ev_{arm}", {}).get("depth"))


def compare(reports):
    if set(reports) != {"off", "gpu", "cpu"}:
        raise ValueError("require off, gpu, cpu reports")
    reference = reports["off"]
    args = reference["meta"]["args"]
    if args.get("exit") != "replay":
        raise ValueError("reports must use replay")
    arms = args.get("arms", "").split(",")
    locked = ("model", "dataset", "offset", "limit", "dtype", "T_total",
              "T_draft", "K", "max_new_tokens", "max_prompt_tokens",
              "seed", "sampling", "kv_s", "exit", "replay_from", "arms")
    for mode, r in reports.items():
        if r.get("shadow_probe", {}).get("mode") != mode:
            raise ValueError(f"incorrect probe tag: {mode}")
        for k in locked:
            if r["meta"]["args"].get(k) != args.get(k):
                raise ValueError(f"configuration mismatch for {mode}: {k}")
        for field in ("wf_commit", "torch", "gpu"):
            if r["meta"].get(field) != reference["meta"].get(field):
                raise ValueError(f"different runtime identity: {field}")
        if len(r["per_sample"]) != len(reference["per_sample"]):
            raise ValueError("different prompt sample count")
        for a, b in zip(reference["per_sample"], r["per_sample"]):
            if a["idx"] != b["idx"]:
                raise ValueError("non-identical prompt ordering")
            for arm in arms:
                if geometry(a, arm) != geometry(b, arm):
                    raise ValueError(f"NOT MATCHED: sample {a['idx']} arm {arm} mode {mode}")
    if not reference["per_sample"]:
        raise ValueError("empty sample set")
    by_arm = {}
    for arm in arms:
        values = {}
        for mode, report in reports.items():
            values[mode] = [float(s[f"wall_{arm}"]) * 1000 for s in report["per_sample"]]
        # Paired differences in prefill-inclusive end-to-end time;
        # neither a GPU-only measurement nor an isolated gate critical path.
        delta = {
            "gpu_minus_off_ms": [a-b for a,b in zip(values["gpu"],values["off"])],
            "cpu_minus_gpu_ms": [a-b for a,b in zip(values["cpu"],values["gpu"])],
            "cpu_minus_off_ms": [a-b for a,b in zip(values["cpu"],values["off"])],
        }
        by_arm[arm] = {key: {"mean": statistics.mean(v), "median": statistics.median(v),
                            "n": len(v)} for key, v in delta.items()}
    return {"matched_geometry": True, "n_prompts": len(reference["per_sample"]),
            "claim_scope": "paired prefill-inclusive eager replay, not native live or CUDA graph gate overhead",
            "arms": by_arm, "shadow_counters": {mode: r["shadow_probe"] for mode,r in reports.items()}}


def main():
    p = argparse.ArgumentParser()
    for mode in ("off", "gpu", "cpu"):
        p.add_argument("--"+mode, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = compare({mode: load(getattr(a, mode)) for mode in ("off","gpu","cpu")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
