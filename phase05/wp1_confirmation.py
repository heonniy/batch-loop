"""Phase-05 WP1: fresh N32/B8/T8 BF16 confirmation under the approved B2 contract.

Hard checks are defined in phase05/WP1_PREREGISTRATION.md. Depth and
next-token flips versus the B1 reference are measured outcomes only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT / "phase05/GPU_ASSIGNMENT_20261017.json"
OLD_GATE_A_MANIFESTS = {
    model: ROOT / f"phase04/results/20260917T055047Z_8e6ce7b_bf16_resolution/{model}_counterexamples/COUNTEREXAMPLE_MANIFEST.json"
    for model in ("ouro", "huginn")}
PHASE04R_MANIFESTS = {
    model: ROOT / f"phase04/results/20260917T060824Z_50910e6_bf16_resolution/{model}_fresh_validation/validation_manifest.json"
    for model in ("ouro", "huginn")}
CONTRACT = {
    "ouro": {"revision": "37bc7e8ff85a90d813dc096383405f521b6a67b0", "threshold": 0.2, "rmax": 4,
             "source_sha256": "59e7523e9fe23f75bde78bb914d1a4d24f3997ae02bca510bbaef947356c447b"},
    "huginn": {"revision": "bb6621b65e90b6a4b9b29ef88dc83866d450470c", "threshold": 0.12, "rmax": 16,
               "source_sha256": "a1d447da93c605a6dc11fbc17cbc7185663f4a41ae12335b34b5847edf0f84aa"},
}
N, B, T, SEED = 32, 8, 8, 1
PERMUTATION_SEEDS = (41, 53)


# ---------------------------------------------------------------- pure checks

def fresh_prompts() -> list[str]:
    topics = ("Summarize what changed in the log below and why it matters. ",
              "List the constraints a careful planner would check first. ",
              "Explain the tradeoff in one compact paragraph. ",
              "Continue the numbered procedure with the next safe action. ",
              "Give a short answer, then justify it briefly. ")
    return [f"Phase05 confirmation item {2000 + index}: " + topics[index % 5] * (1 + index % 3)
            for index in range(N)]


def orders() -> dict[str, list[int]]:
    base = list(range(N))
    result = {"fifo": base}
    for seed in PERMUTATION_SEEDS:
        shuffled = base.copy()
        random.Random(seed).shuffle(shuffled)
        result[f"permutation_seed_{seed}"] = shuffled
    result["fifo_repeat"] = base
    return result


def recompute_depth(model_key: str, scores: list[float], threshold: float, rmax: int) -> int:
    """Independent exit comparator: first passing recurrence, else the Rmax cap."""
    for depth, score in enumerate(scores, 1):
        passed = score >= threshold if model_key == "ouro" else score < threshold
        if passed or depth == rmax:
            return depth
    return -1  # scores ended before an exit: invalid trace


def cohort_errors(selected: list[int], depths: list[int], widths: list[int],
                  kv_rows: list[list[int]], member_rows: list[list[int]], batch_size: int) -> list[str]:
    errors = []
    if len(selected) != batch_size or len(set(selected)) != len(selected):
        errors.append("cohort size or duplicate members")
    if not depths or len(widths) != max(depths):
        errors.append("width count differs from max depth")
    if widths and widths[0] != len(selected):
        errors.append("first width differs from cohort size")
    if widths != sorted(widths, reverse=True):
        errors.append("widths not shrink-only")
    expected = [sum(depth >= r for depth in depths) for r in range(1, max(depths, default=0) + 1)]
    if widths != expected:
        errors.append("width at r differs from members with depth >= r")
    for kv, members in zip(kv_rows, member_rows):
        if kv != members or not set(members) <= set(selected):
            errors.append("KV rows differ from active requests or refill detected")
            break
    return errors


def row_identity_errors(executed: list[tuple[int, int]], tokens: int, n: int) -> list[str]:
    counts = Counter(executed)
    expected = {(t, rid) for t in range(tokens) for rid in range(n)}
    errors = []
    if set(counts) != expected:
        errors.append(f"missing={len(expected - set(counts))} extra={len(set(counts) - expected)}")
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        errors.append(f"duplicated request-tokens={len(duplicates)}")
    return errors


def ordering_flags(reference: dict[tuple[int, int], int], live: dict[tuple[int, int], int]) -> dict:
    deltas = Counter(live[key] - reference[key] for key in reference)
    large_gap_inversions = []
    tokens = sorted({t for t, _ in reference})
    for t in tokens:
        rids = sorted(rid for tt, rid in reference if tt == t)
        for i, a in enumerate(rids):
            for b in rids[i + 1:]:
                gap = reference[(t, a)] - reference[(t, b)]
                if abs(gap) >= 2 and (live[(t, a)] - live[(t, b)]) * gap < 0:
                    large_gap_inversions.append({"token_idx": t, "pair": [a, b], "B1_gap": gap})
    max_abs = max((abs(d) for d in deltas), default=0)
    return {"delta_histogram": {str(k): v for k, v in sorted(deltas.items())},
            "max_abs_delta": max_abs, "gap_ge2_inversions": large_gap_inversions,
            "status": ("ordering materially unstable" if max_abs >= 2 or large_gap_inversions
                       else "ordering stable at observed scope")}


# ---------------------------------------------------------------- GPU runtime

def gpu_identity(role: str) -> dict:
    import torch
    assignment = json.loads(ASSIGNMENT.read_text())["device_map"]
    entry = assignment[role]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(entry["physical_index"]) or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly the mapped physical GPU must be visible")
    query = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=index,uuid,name,memory.total,memory.used",
         "--format=csv,noheader"], text=True).strip()
    fields = [part.strip() for part in query.split(",")]
    if int(fields[0]) != entry["physical_index"] or fields[1] != entry["uuid"] or "H100" not in fields[2]:
        raise RuntimeError(f"GPU identity mismatch: {query}")
    return {"role": role, "lane": entry["lane"], "physical_index": entry["physical_index"],
            "uuid": fields[1], "name": fields[2], "memory_total": fields[3],
            "memory_used_at_start": fields[4], "CUDA_VISIBLE_DEVICES": visible,
            "assignment_sha256": hashlib.sha256(ASSIGNMENT.read_bytes()).hexdigest()}


def _live(model_key, model, tokenizer, prompts, known_tokens, threshold, order):
    import torch
    from phase02_runtime.common import run_interval
    from phase04_runtime.validate import _adapter, _capture
    adapter = _adapter(model_key, model, tokenizer, list(range(N)), SEED, "profile")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    adapter.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
    records, states, cohorts, executed = {}, {}, [], []
    with torch.inference_mode():
        for token_idx, known in enumerate(known_tokens):
            adapter.begin_token()
            for first in range(0, N, B):
                selected = order[first:first + B]
                rows = torch.tensor(selected, dtype=torch.long, device="cuda")
                adapter.backend.set_active(rows)
                ids = torch.tensor([known[rid] for rid in selected], dtype=torch.long,
                                   device="cuda").unsqueeze(1)
                embedded, initial_state = adapter.prelude(ids)
                result = run_interval(adapter, embedded, initial_state, rows, rows,
                                      exit_source="live", threshold=threshold,
                                      instrumentation="profile", debug_ids=True,
                                      debug_hook=_capture(states, token_idx))
                adapter.backend.set_active(rows)
                logits = adapter.coda_logits(result.final_state)[:, -1, :]
                next_ids = logits.argmax(dim=-1)
                depths = [int(d) for d in result.exit_depths.tolist()]
                cohorts.append({"token_idx": token_idx, "members": selected, "depths": depths,
                                "widths": [e["active_batch_size"] for e in result.events],
                                "kv_rows": [e["active_kv_rows"] for e in result.events],
                                "member_rows": [e["active_request_ids"] for e in result.events]})
                for local, rid in enumerate(selected):
                    executed.append((token_idx, rid))
                    mine = [e for e in result.events if rid in e["active_request_ids"]]
                    records[(token_idx, rid)] = {
                        "depth": depths[local], "next_token": int(next_ids[local]),
                        "logits": logits[local].detach().float().cpu(),
                        "scores": [float(e["exit_score"][e["active_request_ids"].index(rid)]) for e in mine],
                        "margins": [float(e["exit_margin"][e["active_request_ids"].index(rid)]) for e in mine]}
            adapter.commit_token()
    return {"records": records, "states": states, "cohorts": cohorts, "executed": executed}, adapter


def _rel_l2(a, b) -> float:
    import torch
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12))


def run(args: argparse.Namespace) -> int:
    import torch
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    from phase04_runtime.data_contract import effective_token_hashes, sha256_file
    from phase04_runtime.validate import _kv_compare, _reference

    started = time.perf_counter()
    torch.set_num_threads(8)
    key = args.model
    contract = CONTRACT[key]
    role = f"{key}_{args.role}"
    hard: dict[str, list[str]] = {}
    gpu = gpu_identity(role)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "phase02_runtime",
                                     "phase04_runtime", "phase05"], text=True, cwd=ROOT).strip()
    adapter_type = HuginnAdapter if key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if key == "huginn"
               else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    hard["provenance"] = [msg for ok, msg in (
        (next(model.parameters()).dtype == torch.bfloat16, "weights are not BF16"),
        (adapter_type.model_revision == contract["revision"], "model revision mismatch"),
        (sha256_file(imported) == contract["source_sha256"], "imported model source changed"),
        (initial.rmax == contract["rmax"], "Rmax mismatch"),
        (not dirty, "uncommitted runtime/phase05 source")) if not ok]

    prompts = fresh_prompts()
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    hashes = effective_token_hashes(encoded["input_ids"], encoded["attention_mask"])
    old = {row["sha256"] for row in json.loads(OLD_GATE_A_MANIFESTS[key].read_text())["cohort_effective_inputs"]}
    old |= set(json.loads(PHASE04R_MANIFESTS[key].read_text())["effective_input_hashes"])
    hard["input_identity"] = [msg for ok, msg in (
        (len(set(hashes)) == N, "duplicate effective inputs"),
        (not set(hashes) & old, "cohort overlaps an earlier Gate-A/Phase-04R cohort")) if not ok]

    threshold = contract["threshold"]
    reference, known, reference_adapters = _reference(
        key, model, tokenizer, prompts, T, threshold, SEED, int(encoded["input_ids"].shape[1]))
    ref = reference["records"]
    ref_depth = {k: v["depth"] for k, v in ref.items()}
    hard["reference_checks"] = [
        f"B1 comparator mismatch at {k}" for k, v in ref.items()
        if recompute_depth(key, v["scores"], threshold, contract["rmax"]) != v["depth"]]
    hard["reference_checks"] += [f"B1 nonfinite at {k}" for k, v in ref.items()
                                 if not all(map(math.isfinite, v["scores"]))
                                 or not bool(torch.isfinite(v["logits"]).all())]
    reference_h0 = {(rid, pos): digest for adapter, _ in reference_adapters
                    for (rid, pos), digest in getattr(adapter, "state_hashes", {}).items()}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_token, summaries, conditions, live_by_name = [], [], {}, {}
    for name, order in orders().items():
        cond_start = time.perf_counter()
        live, adapter = _live(key, model, tokenizer, prompts, known, threshold, order)
        torch.cuda.synchronize()
        recs = live["records"]
        errors = row_identity_errors(live["executed"], T, N)
        for index, cohort in enumerate(live["cohorts"]):
            slot = index % (N // B)
            if cohort["members"] != order[slot * B:(slot + 1) * B]:
                errors.append(f"cohort {index} membership differs from declared order")
            errors += [f"cohort {index}: {e}" for e in cohort_errors(
                cohort["members"], cohort["depths"], cohort["widths"],
                cohort["kv_rows"], cohort["member_rows"], B)]
        for k, v in recs.items():
            if recompute_depth(key, v["scores"], threshold, contract["rmax"]) != v["depth"]:
                errors.append(f"live comparator mismatch at {k}")
            if not all(map(math.isfinite, v["scores"])) or not bool(torch.isfinite(v["logits"]).all()):
                errors.append(f"live nonfinite score/logit at {k}")
        if not all(bool(torch.isfinite(s).all()) for s in live["states"].values()):
            errors.append("nonfinite captured recurrent state")
        kv_max, kv_rel, kv_structural, kv_mismatches = _kv_compare(
            reference_adapters, adapter, atol=0.05, rtol=0.02)
        if kv_structural:
            errors.append(f"KV structural mismatches={kv_structural}")
        h0_mismatches = []
        if key == "huginn":
            live_h0 = getattr(adapter, "state_hashes", {})
            h0_mismatches = [f"{rid}:{pos}" for (rid, pos), d in reference_h0.items() if live_h0.get((rid, pos)) != d]
            if not reference_h0 or h0_mismatches:
                errors.append(f"h0 identity mismatches={len(h0_mismatches)} (reference hashes={len(reference_h0)})")
        hard[f"condition:{name}"] = errors

        common = [s for s in reference["states"] if s in live["states"]]
        live_depth = {k: v["depth"] for k, v in recs.items()}
        flags = ordering_flags(ref_depth, live_depth)
        frozen_ric = sum(max(ref_depth[(c["token_idx"], rid)] for rid in c["members"]) for c in live["cohorts"])
        summaries.append({
            "model": key, "lane": gpu["lane"], "condition": name, "B": B, "positions": N * T,
            "depth_flips": sum(live_depth[k] != ref_depth[k] for k in ref),
            "next_token_flips": sum(recs[k]["next_token"] != ref[k]["next_token"] for k in ref),
            "reference_B1_work": sum(ref_depth.values()), "actual_live_work": sum(live_depth.values()),
            "actual_live_RIC": sum(max(c["depths"]) for c in live["cohorts"]),
            "frozen_B1_depth_grouping_RIC": frozen_ric,
            "max_abs_depth_delta": flags["max_abs_delta"],
            "gap_ge2_inversions": len(flags["gap_ge2_inversions"]),
            "ordering_status": flags["status"],
            "max_hidden_relative_l2": max((_rel_l2(reference["states"][s], live["states"][s]) for s in common), default=0.0),
            "max_logit_relative_l2": max(_rel_l2(ref[k]["logits"], recs[k]["logits"]) for k in ref),
            "max_kv_relative_l2": kv_rel, "max_kv_abs_error": kv_max,
            "max_gate_abs_diff": max(abs(a - b) for k in ref for a, b in zip(ref[k]["scores"], recs[k]["scores"])),
            "hard_check_errors": len(errors), "condition_wall_s": time.perf_counter() - cond_start})
        conditions[name] = {"ordering_flags": flags, "kv_value_mismatches_first20": kv_mismatches[:20],
                            "h0_mismatches": h0_mismatches[:20], "hard_errors": errors[:50]}
        for t in range(T):
            for rid in range(N):
                r, l = ref[(t, rid)], recs[(t, rid)]
                per_token.append({
                    "model": key, "lane": gpu["lane"], "condition": name, "B": B,
                    "token_idx": t, "request_id": rid, "input_token": known[t][rid],
                    "reference_depth": r["depth"], "live_depth": l["depth"],
                    "reference_next_token": r["next_token"], "live_next_token": l["next_token"],
                    "reference_min_abs_margin": min(abs(m) for m in r["margins"]),
                    "reference_scores": json.dumps(r["scores"]), "live_scores": json.dumps(l["scores"]),
                    "live_logits_sha256": hashlib.sha256(l["logits"].numpy().tobytes()).hexdigest()})
        live_by_name[name] = {k: (v["depth"], v["next_token"], tuple(v["scores"]), v["logits"]) for k, v in recs.items()}
        del live, adapter
        torch.cuda.empty_cache()

    a, b = live_by_name["fifo"], live_by_name["fifo_repeat"]
    hard["duplicate_control"] = [f"fifo_repeat differs at {k}" for k in a
                                 if a[k][:3] != b[k][:3] or not torch.equal(a[k][3], b[k][3])][:50]

    failed = {name: errs for name, errs in hard.items() if errs}
    status = "PASS" if not failed else "FAIL"
    for name, rows in (("per_token.csv", per_token), ("condition_summary.csv", summaries)):
        with (args.output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    gate = {"status": status, "gate": "phase05/WP1_PREREGISTRATION.md semantic Gate A (B2)",
            "model": key, "lane": gpu["lane"], "failed_checks": failed,
            "check_names": sorted(hard), "conditions": conditions,
            "note": "Depth/next-token flips vs B1 are measured outcomes under B2, not gate failures."}
    (args.output_dir / "gate_report.json").write_text(json.dumps(gate, indent=2) + "\n")
    manifest = {
        "execution_kind": "real-no-refill-runtime BF16 correctness confirmation; no scored wall time",
        "source_sha": source_sha, "argv": sys.argv, "gpu": gpu, "model": key,
        "model_revision": contract["revision"], "imported_model_source": str(imported),
        "imported_model_source_sha256": sha256_file(imported),
        "model_config_sha256": hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True,
                                                         default=str).encode()).hexdigest(),
        "precision": "torch.bfloat16", "threshold": threshold, "rmax": initial.rmax,
        "kv_policy": initial.kv_policy, "h0_seed": SEED if key == "huginn" else None,
        "N": N, "B": B, "T": T, "orders": orders(), "prompt_cap": 64,
        "cohort_tokenized_width": int(encoded["input_ids"].shape[1]),
        "effective_input_hashes": hashes,
        "disjoint_from": [str(OLD_GATE_A_MANIFESTS[key].relative_to(ROOT)), str(PHASE04R_MANIFESTS[key].relative_to(ROOT))],
        "input_policy": "B1-generated greedy token IDs teacher-forced into every live condition; no EOS",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.import_module("transformers").__version__,
        "python": sys.version, "elapsed_wall_s": time.perf_counter() - started}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": status, "model": key, "lane": gpu["lane"],
                      "failed_checks": sorted(failed),
                      "depth_flips": [s["depth_flips"] for s in summaries],
                      "next_token_flips": [s["next_token_flips"] for s in summaries],
                      "elapsed_s": round(manifest["elapsed_wall_s"], 1)}), flush=True)
    return 0 if status == "PASS" else 1


# ---------------------------------------------------------------- replicas

def compare_replicas(primary: Path, replica: Path) -> dict:
    def load(path):
        with (path / "per_token.csv").open() as stream:
            return {(r["condition"], r["token_idx"], r["request_id"]): r for r in csv.DictReader(stream)}
    a, b = load(primary), load(replica)
    if set(a) != set(b):
        return {"status": "KEY_MISMATCH", "primary": str(primary), "replica": str(replica)}
    fields = ("reference_depth", "live_depth", "reference_next_token", "live_next_token",
              "reference_scores", "live_scores", "live_logits_sha256")
    return {"primary": str(primary), "replica": str(replica), "positions": len(a),
            "bit_identical_fraction": {f: sum(a[k][f] == b[k][f] for k in a) / len(a) for f in fields}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--model", required=True, choices=("ouro", "huginn"))
    runner.add_argument("--role", required=True, choices=("primary", "replication"))
    runner.add_argument("--output-dir", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--pair", nargs=2, action="append", type=Path, required=True)
    compare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(run(args))
    report = [compare_replicas(p, r) for p, r in args.pair]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
