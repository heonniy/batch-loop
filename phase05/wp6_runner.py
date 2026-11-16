"""Phase-05 WP6: extensions of the WP5 ranker result (see phase05/WP6_PREREGISTRATION.md).

6a cross-dataset transfer, 6b ranker without trajectory features, 6c another exit
threshold, 6d EOS-terminated variable-length decoding. All policies run on the
host-sync-free GPU serving path (wp4_gpu_serving.gpu_run), live BF16 exits,
teacher-forced B1 tokens, strict no-refill, per-run hard checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np

from phase05.wp3_predictor import COHORT, EVAL_B

REPEATS = 3


def ranker_specs(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        name, path = value.split("=", 1)
        out[name] = Path(path)
    return out


def main() -> None:
    import torch
    from phase05.wp1_confirmation import CONTRACT, gpu_identity
    from phase05.wp2_headroom import split_prompts
    from phase05.wp3_predictor import _check_source, _dataset_texts, _encode, _load, _projection, _sha
    from phase05.wp4_gpu_serving import gpu_run
    from phase05.wp5_ranker import GpuRanker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=("transfer_notraj", "threshold", "eos"))
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True)
    parser.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    parser.add_argument("--split", required=True, choices=("wp6_test", "wp6_q_test", "wp6_eos_test", "wp3_validation"))
    parser.add_argument("--feature-root", required=True, type=Path, help="lane dir holding <split>/ arrays")
    parser.add_argument("--ranker", action="append", default=[], help="name=ranker_dir (repeatable)")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--use-eos-lengths", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-ranker", help="Gate 0: audit this ranker on --split (validation) instead of scoring")
    parser.add_argument("--audit-random-lengths", action="store_true", help="Gate 0 for the variable-length path")
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    key = args.model
    threshold = args.threshold if args.threshold is not None else CONTRACT[key]["threshold"]
    gpu = gpu_identity(f"{key}_{args.role}")
    source_sha = _check_source(args.smoke)
    rankers, selections = {}, {}
    for name, root in ranker_specs(args.ranker).items():
        sel = json.loads((root / "ranker_selection.json").read_text())
        if _sha(root / "ranker_checkpoint.pt") != sel["checkpoint_sha256"]:
            raise RuntimeError(f"ranker {name} changed")
        rankers[name], selections[name] = GpuRanker(root / "ranker_checkpoint.pt"), {"root": str(root), **sel}
    _, model, tokenizer = _load(key)
    projection = _projection(key, model)
    d = args.feature_root / args.split
    man = json.loads((d / "feature_manifest.json").read_text())
    if abs(man["threshold"] - threshold) > 1e-12:
        raise RuntimeError(f"labels were collected at threshold {man['threshold']}, run uses {threshold}")
    labels_all = np.load(d / "depth_labels.npy").astype(np.int64)
    tokens_all = np.load(d / "token_ids.npy").astype(np.int64)
    lengths_all = np.load(d / "lengths.npy") if args.use_eos_lengths else None
    prompts_all = split_prompts(_dataset_texts(args.dataset), args.split)[:len(man["prompt_sha256"])]
    if [hashlib.sha256(p.encode()).hexdigest() for p in prompts_all] != man["prompt_sha256"]:
        raise RuntimeError("prompts differ from collected labels")
    size = min(COHORT, len(prompts_all))
    cohorts = tuple(range(len(prompts_all) // size))[:2]
    steps = args.tokens or labels_all.shape[0]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    base = {"preregistration": "phase05/WP6_PREREGISTRATION.md", "study": args.study, "source_sha": source_sha,
            "argv": sys.argv, "gpu": gpu, "cpu_affinity": sorted(os.sched_getaffinity(0)), "model": key,
            "dataset": args.dataset, "split": args.split, "threshold": threshold, "eos_lengths": args.use_eos_lengths,
            "rankers": selections, "label_manifest_sha256": _sha(d / "feature_manifest.json"), "torch": torch.__version__}

    def inputs(c):
        sl = slice(c * size, (c + 1) * size)
        lengths = None if lengths_all is None else np.minimum(lengths_all[sl], steps)
        return _encode(tokenizer, prompts_all[sl]), labels_all[:steps, sl], tokens_all[:steps, sl], lengths

    if args.audit_ranker:  # Gate 0: live GPU scores vs offline scores on B1 arrays (validation only)
        ranker = rankers[args.audit_ranker]
        encoded, labels, tokens, lengths = inputs(0)
        if args.audit_random_lengths:
            lengths = np.random.default_rng(0).integers(1, steps + 1, size=size)
        result = gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, 8, "ranker_gpu", None,
                         projection, ranker=ranker, capture_scores=True, lengths=lengths)
        scal = np.load(d / "scalars.npy")[:steps, :size].astype(np.float32)
        fulls = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ranker.full_names}
        live_lengths = np.full(size, steps) if lengths is None else lengths
        rho, k = [], 0
        rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")
        with torch.inference_mode():
            for t in range(steps):
                act = np.nonzero(live_lengths > t)[0]
                if not len(act):
                    continue
                cur = torch.as_tensor(tokens[t], device="cuda")
                prev = torch.as_tensor(tokens[t - 1], device="cuda") if t else torch.full_like(cur, ranker.vocab)
                off = ranker.score(torch.as_tensor(scal[t], device="cuda"), cur, prev,
                                   {n: torch.as_tensor(np.asarray(f[t, :size], dtype=np.float32), device="cuda")
                                    for n, f in fulls.items()}).cpu().numpy()[act]
                live = result["scores"][k][act]
                k += 1
                if len(act) > 2 and off.std() and live.std():
                    rho.append(float(np.corrcoef(rank(off), rank(live))[0, 1]))
        summary = {"execution_kind": "UNTIMED Gate-0 ranker audit", **base, "audit_ranker": args.audit_ranker,
                   "random_lengths": bool(args.audit_random_lengths), "B": 8, "steps_compared": len(rho),
                   "n_hard_errors": result["n_hard_errors"], "hard_errors": result["hard_check_errors"],
                   "mean_spearman_live_vs_offline": statistics.mean(rho), "min_spearman": min(rho)}
        summary["status"] = ("PASS" if summary["mean_spearman_live_vs_offline"] >= 0.95 and not result["n_hard_errors"]
                             else "FAIL")
        (out / "ranker_audit.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        print(json.dumps({k2: summary[k2] for k2 in ("status", "mean_spearman_live_vs_offline", "min_spearman",
                                                      "n_hard_errors")}), flush=True)
        sys.exit(0 if summary["status"] == "PASS" else 1)

    policies = ("fifo", "history_last", *[f"ranker_{n}" for n in rankers], "oracle_b1")
    plan = [{"kind": "warmup", "cohort": 0, "B": 32, "policy": "fifo", "rep": -1}]
    for rep in range(1 if args.smoke else REPEATS):
        for c in cohorts:
            order = list(EVAL_B)
            random.Random(f"wp6-plan:{args.study}:{rep}:{c}").shuffle(order)
            for b in order:
                k = rep % len(policies)
                plan += [{"kind": "scored", "cohort": c, "B": b, "policy": p, "rep": rep}
                         for p in policies[k:] + policies[:k]]
    if args.smoke:
        plan = [c for c in plan if c["B"] in (32, 64)]
    (out / "run_manifest.json").write_text(json.dumps({
        "execution_kind": "UNSCORED_SMOKE" if args.smoke else "sealed physical no-refill, live BF16 exits",
        **base, "cohorts": [[c * size, (c + 1) * size] for c in cohorts], "policies": policies, "B": EVAL_B,
        "repeats": REPEATS, "T": steps}, indent=2, default=float) + "\n")
    fields = ["run_id", "kind", "cohort", "B", "policy", "rep", "status", "decode_wall_s", "live_work", "b1_work",
              "live_RIC", "frozen_b1_RIC_same_grouping", "depth_flips", "next_token_flips", "active_tokens",
              "peak_allocated_gib", "n_hard_errors", "error"]
    with (out / "runs.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    live_rows, hard = [], 0
    for run_id, cell in enumerate(plan):
        encoded, labels, tokens, lengths = inputs(cell["cohort"])
        row = {"run_id": run_id, **cell, "status": "OK", "error": ""}
        try:
            policy = cell["policy"]
            ranker = rankers.get(policy[len("ranker_"):]) if policy.startswith("ranker_") else None
            result = gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, cell["B"],
                             "ranker_gpu" if ranker is not None else policy, None, projection,
                             ranker=ranker, lengths=lengths)
            row.update({k: result[k] for k in fields if k in result})
            live_rows.append((run_id, np.asarray(result["live_depth"], dtype=np.int16)))
            if result["n_hard_errors"]:
                row["status"], row["error"] = "HARD_CHECK_FAIL", json.dumps(result["hard_check_errors"])
                hard += 1
        except torch.cuda.OutOfMemoryError as error:
            row["status"], row["error"] = "OOM", str(error)[:300]
        except Exception as error:  # retained, never filtered
            row["status"], row["error"] = "ERROR", f"{type(error).__name__}: {error}"[:300]
        finally:
            torch.cuda.empty_cache()
        with (out / "runs.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore").writerow(row)
        print(json.dumps({k: row.get(k) for k in ("run_id", "kind", "cohort", "B", "policy", "rep", "status",
                                                  "decode_wall_s", "live_RIC", "error")}), flush=True)
        if hard:
            print(json.dumps({"event": "lane_stopped", "reason": "hard correctness check failed"}), flush=True)
            break
    np.savez_compressed(out / "live_depths.npz", **{f"run_{i}": x for i, x in live_rows})
    print(json.dumps({"event": "lane_done", "hard_failures": hard}), flush=True)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
