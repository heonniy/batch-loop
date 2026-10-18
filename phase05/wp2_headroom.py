"""Phase-05 WP2: physical no-refill regrouping headroom (lockstep decode boundary).

Every scored run physically executes its own grouping with live BF16 exits on
the phase02 survivor-compaction runtime. All policies are teacher-forced with
the same B1 greedy token IDs, so input trajectories are identical; actual live
depth/work and next-token differences are measured, never corrected.
See phase05/WP2_PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from phase05.wp1_confirmation import ASSIGNMENT, CONTRACT, cohort_errors, row_identity_errors

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("fifo", "random", "static_first", "history_last", "history_mean", "oracle_b1")
BATCHES = (4, 8, 16, 32, 64)
N, T, REPEATS, PROMPT_CAP = 256, 64, 3, 64
SPLIT_SEED, PRIOR_SEED, PRIOR_PREFIX, H0_SEED = 20261017, 1, 4096, 1
SPLITS = {"wp2_headroom": (0, 256), "wp3_train": (256, 1280),
          "wp3_validation": (1280, 1792), "wp3_test": (1792, 2304),
          "wp4_confirm": (2304, 2816), "wp5_test": (2816, 3328),
          "wp6_test": (3328, 3840), "wp6_q_test": (3840, 4352), "wp6_eos_test": (4352, 4864)}


# ---------------------------------------------------------------- pure parts

def split_prompts(all_prompts: list[str], split: str) -> list[str]:
    """Deterministic request split, excluding prompts that earlier phases could have used.

    Earlier phases drew the first <=PRIOR_PREFIX prompts of a seed-1 shuffle
    (poc.load_prompts); those are removed before the seed-20261017 shuffle.
    """
    prior = all_prompts.copy()
    random.Random(PRIOR_SEED).shuffle(prior)
    excluded = set(prior[:PRIOR_PREFIX])
    pool = sorted(set(all_prompts) - excluded)
    random.Random(SPLIT_SEED).shuffle(pool)
    start, stop = SPLITS[split]
    if len(pool) < stop:
        raise ValueError("dataset too small for the declared split")
    return pool[start:stop]


def policy_order(policy: str, t: int, n: int, own_history: list[list[int]],
                 b1_depth: list[list[int]], rep_seed: int) -> list[int]:
    """Cohort order for decode step t. Only oracle_b1 reads step-t information."""
    ids = list(range(n))
    if policy == "fifo" or (t == 0 and policy in ("static_first", "history_last", "history_mean")):
        return ids
    if policy == "random":
        random.Random(f"{rep_seed}:{t}").shuffle(ids)
        return ids
    if policy == "oracle_b1":
        return sorted(ids, key=lambda r: (b1_depth[t][r], r))
    if policy == "static_first":
        return sorted(ids, key=lambda r: (own_history[0][r], r))
    if policy == "history_last":
        return sorted(ids, key=lambda r: (own_history[t - 1][r], r))
    if policy == "history_mean":
        return sorted(ids, key=lambda r: (sum(h[r] for h in own_history[:t]) / t, r))
    raise ValueError(policy)


def run_plan(batches=BATCHES, repeats=REPEATS) -> list[dict]:
    """Interleaved, rep-rotated execution order; B=N no-composition control and a warmup."""
    plan = [{"kind": "warmup", "B": 64, "policy": "fifo", "rep": -1}]
    for rep in range(repeats):
        order = list(batches)
        random.Random(f"wp2-plan:{rep}").shuffle(order)
        for b in order:
            k = (rep * 2) % len(POLICIES)
            for policy in POLICIES[k:] + POLICIES[:k]:
                plan.append({"kind": "scored", "B": b, "policy": policy, "rep": rep})
        plan.append({"kind": "control_no_composition", "B": N, "policy": "fifo", "rep": rep})
    return plan


# ---------------------------------------------------------------- GPU runtime

def _gpu(role: str) -> dict:
    from phase05.wp1_confirmation import gpu_identity
    return gpu_identity(role)


def _new_adapter(key, model, tokenizer, rows: int, ids: list[int], instrumentation: str):
    import torch
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    if key == "huginn":
        return HuginnAdapter(model, tokenizer, rows, state_init="native-seeded", identity_seed=H0_SEED,
                             canonical_ids=torch.tensor(ids, device="cuda"),
                             instrumentation=instrumentation, capture_state_hashes=False)
    return OuroAdapter(model, tokenizer, rows, instrumentation=instrumentation)


def b1_reference(key, model, tokenizer, prompts, threshold) -> dict:
    """Per-request B=1 live greedy run: canonical depths, tokens and B=1 decode wall."""
    import torch
    from phase02_runtime.common import run_interval
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_CAP)
    width = int(encoded["input_ids"].shape[1])
    depth = [[0] * N for _ in range(T)]
    inputs = [[0] * N for _ in range(T)]
    nexts = [[0] * N for _ in range(T)]
    decode_s = 0.0
    with torch.inference_mode():
        for rid, prompt in enumerate(prompts):
            adapter = _new_adapter(key, model, tokenizer, 1, [rid], "benchmark")
            one = tokenizer([prompt], return_tensors="pt", padding="max_length",
                            truncation=True, max_length=width)
            current = adapter.prefill(one["input_ids"].cuda(), one["attention_mask"].cuda().bool())
            row = torch.tensor([0], dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            for t in range(T):
                inputs[t][rid] = int(current.reshape(-1)[0])
                adapter.begin_token()
                adapter.backend.set_active(row)
                embedded, state = adapter.prelude(current.reshape(1, 1))
                result = run_interval(adapter, embedded, state, row, row, exit_source="live",
                                      threshold=threshold, instrumentation="benchmark", debug_ids=False)
                adapter.backend.set_active(row)
                current = adapter.coda(result.final_state).reshape(1, 1)
                adapter.commit_token()
                depth[t][rid] = int(result.exit_depths[0])
                nexts[t][rid] = int(current[0, 0])
            torch.cuda.synchronize()
            decode_s += time.perf_counter() - start
            del adapter
    return {"depth": depth, "input": inputs, "next": nexts, "decode_wall_s": decode_s, "width": width}


def batched_run(key, model, tokenizer, encoded, ref, threshold, B, policy, rep) -> dict:
    import torch
    from phase02_runtime.common import run_interval
    adapter = _new_adapter(key, model, tokenizer, N, list(range(N)), "benchmark")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        adapter.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
        known = torch.tensor(ref["input"], dtype=torch.long, device="cuda")  # [T, N]
        history: list[list[int]] = []
        next_out = torch.empty((T, N), dtype=torch.long, device="cuda")
        cohorts, executed = [], []
        scheduler_s = 0.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in range(T):
            s0 = time.perf_counter()
            order = policy_order(policy, t, N, history, ref["depth"], rep)
            groups = [order[i:i + B] for i in range(0, N, B)]
            scheduler_s += time.perf_counter() - s0
            adapter.begin_token()
            step_depth = torch.empty(N, dtype=torch.long, device="cuda")
            step = []
            for members in groups:
                rows = torch.tensor(members, dtype=torch.long, device="cuda")
                adapter.backend.set_active(rows)
                embedded, state = adapter.prelude(known[t].index_select(0, rows).unsqueeze(1))
                result = run_interval(adapter, embedded, state, rows, rows, exit_source="live",
                                      threshold=threshold, instrumentation="benchmark", debug_ids=False)
                adapter.backend.set_active(rows)
                next_out[t].index_copy_(0, rows, adapter.coda(result.final_state).reshape(-1))
                step_depth.index_copy_(0, rows, result.exit_depths)
                step.append((members, [e["active_batch_size"] for e in result.events]))
            adapter.commit_token()
            depths_t = step_depth.tolist()  # one host sync per decode step, all policies
            history.append(depths_t)
            for members, widths in step:
                cohorts.append({"t": t, "members": members, "widths": widths,
                                "depths": [depths_t[r] for r in members]})
                executed += [(t, r) for r in members]
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - start
        nexts = next_out.tolist()
    errors = row_identity_errors(executed, T, N)
    for c in cohorts:
        errors += cohort_errors(c["members"], c["depths"], c["widths"], [], [], len(c["members"]))
    ref_d, ref_n = ref["depth"], ref["next"]
    live_work = sum(map(sum, history))
    return {
        "decode_wall_s": decode_s, "scheduler_cpu_s": scheduler_s,
        "tokens_per_s": N * T / decode_s,
        "live_work": live_work, "b1_work": sum(map(sum, ref_d)),
        "live_RIC": sum(max(c["depths"]) for c in cohorts),
        "frozen_b1_RIC_same_grouping": sum(max(ref_d[c["t"]][r] for r in c["members"]) for c in cohorts),
        "core_row_slots": sum(sum(c["widths"]) for c in cohorts),
        "depth_flips": sum(history[t][r] != ref_d[t][r] for t in range(T) for r in range(N)),
        "next_token_flips": sum(nexts[t][r] != ref_n[t][r] for t in range(T) for r in range(N)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "hard_check_errors": errors[:20], "n_hard_errors": len(errors),
        "live_depth": history, "live_next": nexts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True, choices=("primary", "replication"))
    parser.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="unscored engineering check: N/T reduced")
    args = parser.parse_args()
    global N, T
    if args.smoke:
        N, T = 32, 4
    import torch
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    from phase04_runtime.data_contract import effective_token_hashes, sha256_file

    torch.set_num_threads(4)
    key, contract = args.model, CONTRACT[args.model]
    gpu = _gpu(f"{key}_{args.role}")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--",
                                     "phase02_runtime", "phase04_runtime", "phase05", "poc.py"],
                                    text=True, cwd=ROOT).strip()
    if dirty and not args.smoke:
        raise RuntimeError("scored WP2 runs require committed source")
    adapter_type = HuginnAdapter if key == "huginn" else OuroAdapter
    initial = (adapter_type.load(1, state_init="native-seeded") if key == "huginn" else adapter_type.load(1))
    model, tokenizer = initial.model, initial.tokenizer
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    if (next(model.parameters()).dtype != torch.bfloat16 or adapter_type.model_revision != contract["revision"]
            or sha256_file(imported) != contract["source_sha256"]):
        raise RuntimeError("model provenance differs from the WP1-confirmed contract")
    threshold = contract["threshold"]

    import datasets
    from poc import _first_text
    name = {"alpaca": "tatsu-lab/alpaca",
            "sharegpt": "ewof/sharegpt-instruct-unfiltered-deduped"}[args.dataset]
    texts = []
    for row in datasets.load_dataset(name, split="train"):
        try:
            texts.append(_first_text(dict(row)))
        except ValueError:
            continue
    prompts = split_prompts(texts, "wp2_headroom")[:N]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_CAP)
    hashes = effective_token_hashes(encoded["input_ids"], encoded["attention_mask"])

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    status = out / "status.jsonl"

    def log(obj):
        obj["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with status.open("a") as stream:
            stream.write(json.dumps(obj) + "\n")
        print(json.dumps(obj), flush=True)

    manifest = {
        "execution_kind": ("UNSCORED_SMOKE" if args.smoke else
                           "physical no-refill lockstep regrouping headroom, live BF16 exits, teacher-forced B1 tokens"),
        "preregistration": "phase05/WP2_PREREGISTRATION.md", "source_sha": source_sha, "argv": sys.argv,
        "gpu": gpu, "cpu_affinity": sorted(os.sched_getaffinity(0)), "model": key,
        "model_revision": contract["revision"], "imported_model_source_sha256": sha256_file(imported),
        "precision": "torch.bfloat16", "threshold": threshold, "rmax": initial.rmax,
        "kv_policy": initial.kv_policy, "h0_seed": H0_SEED if key == "huginn" else None,
        "dataset": name, "dataset_rows_parsed": len(texts), "split": "wp2_headroom",
        "split_rule": {"prior_seed": PRIOR_SEED, "prior_prefix_excluded": PRIOR_PREFIX,
                       "split_seed": SPLIT_SEED, "splits": SPLITS},
        "N": N, "T": T, "prompt_cap": PROMPT_CAP, "cohort_tokenized_width": int(encoded["input_ids"].shape[1]),
        "effective_input_hashes": hashes, "policies": POLICIES, "B": BATCHES, "repeats": REPEATS,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.import_module("transformers").__version__}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    log({"event": "b1_reference_start"})
    ref = b1_reference(key, model, tokenizer, prompts, threshold)
    with gzip.open(out / "b1_reference.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["token_idx", "request_id", "input_token", "b1_depth", "b1_next_token"])
        for t in range(T):
            for r in range(N):
                writer.writerow([t, r, ref["input"][t][r], ref["depth"][t][r], ref["next"][t][r]])
    log({"event": "b1_reference_done", "decode_wall_s": ref["decode_wall_s"],
         "b1_work": sum(map(sum, ref["depth"])), "label": "B=1 control (per-request sequential)"})

    plan = run_plan(BATCHES if not args.smoke else (4, 8), REPEATS if not args.smoke else 1)
    summary_path = out / "runs.csv"
    fields = ["run_id", "kind", "B", "policy", "rep", "status", "decode_wall_s", "scheduler_cpu_s",
              "tokens_per_s", "live_work", "b1_work", "live_RIC", "frozen_b1_RIC_same_grouping",
              "core_row_slots", "depth_flips", "next_token_flips", "peak_allocated_gib", "n_hard_errors", "error"]
    with summary_path.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    depth_stream = gzip.open(out / "live_tokens.csv.gz", "wt", newline="")
    depth_writer = csv.writer(depth_stream)
    depth_writer.writerow(["run_id", "token_idx", "request_id", "live_depth", "live_next_token"])
    hard_failures = 0
    for run_id, cell in enumerate(plan):
        row = {"run_id": run_id, **cell, "status": "OK", "error": ""}
        try:
            result = batched_run(key, model, tokenizer, encoded, ref, threshold, cell["B"], cell["policy"], cell["rep"])
            for t in range(T):
                for r in range(N):
                    depth_writer.writerow([run_id, t, r, result["live_depth"][t][r], result["live_next"][t][r]])
            row.update({k: result[k] for k in fields if k in result})
            if result["n_hard_errors"]:
                row["status"], row["error"] = "HARD_CHECK_FAIL", json.dumps(result["hard_check_errors"])
                hard_failures += 1
        except torch.cuda.OutOfMemoryError as error:
            row["status"], row["error"] = "OOM", str(error)[:300]
        except Exception as error:  # retained, never filtered
            row["status"], row["error"] = "ERROR", f"{type(error).__name__}: {error}"[:300]
        finally:
            torch.cuda.empty_cache()
        with summary_path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore").writerow(row)
        log({"event": "run", **{k: row.get(k) for k in ("run_id", "kind", "B", "policy", "rep", "status",
                                                         "decode_wall_s", "live_RIC", "depth_flips")}})
        if hard_failures:
            log({"event": "lane_stopped", "reason": "hard correctness check failed"})
            break
    depth_stream.close()
    log({"event": "lane_done", "hard_failures": hard_failures})
    sys.exit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()
