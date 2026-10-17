"""Phase-05 WP3: causal current-token depth-propensity predictor under B2.

collect  (GPU) — per-request B=1 live greedy runs on a declared split, writing the
                 Phase-03 causal feature schema (phase03_predictor.physical._features,
                 the same function the physical runtime calls) and canonical depths.
select   (CPU) — train Phase-03 candidates C0-C9 on wp3_train, choose ONLY on
                 wp3_validation, freeze checkpoints. Never opens test arrays.
physical (GPU) — sealed wp3_test cohorts: FIFO / history_last / frozen predictor /
                 oracle_b1, each executing its own grouping with live BF16 exits.
See phase05/WP3_PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from phase05.wp1_confirmation import CONTRACT, cohort_errors, row_identity_errors
from phase05.wp2_headroom import H0_SEED, SPLITS, split_prompts

ROOT = Path(__file__).resolve().parents[1]
T, PROMPT_CAP, PROJECTION_DIM, PROJECTION_SEED = 64, 64, 64, 1
COHORT = 256
EVAL_B = (8, 16, 32, 64)
PHYSICAL_POLICIES = ("fifo", "history_last", "predictor", "oracle_b1")
REPEATS = 3
TRAJECTORY_CANDIDATES = {"C2", "C6", "C7", "C8", "C9"}
UNCERTAINTY_CANDIDATES = {"C3", "C6", "C7", "C8", "C9"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_texts(dataset: str) -> list[str]:
    import datasets
    from poc import _first_text
    name = {"alpaca": "tatsu-lab/alpaca", "sharegpt": "ewof/sharegpt-instruct-unfiltered-deduped"}[dataset]
    texts = []
    for row in datasets.load_dataset(name, split="train"):
        try:
            texts.append(_first_text(dict(row)))
        except ValueError:
            continue
    return texts


def _load(key: str):
    import torch
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    from phase04_runtime.data_contract import sha256_file
    adapter_type = HuginnAdapter if key == "huginn" else OuroAdapter
    initial = adapter_type.load(1, state_init="native-seeded") if key == "huginn" else adapter_type.load(1)
    model = initial.model
    imported = Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
    c = CONTRACT[key]
    if (next(model.parameters()).dtype != torch.bfloat16 or adapter_type.model_revision != c["revision"]
            or sha256_file(imported) != c["source_sha256"] or initial.rmax != c["rmax"]):
        raise RuntimeError("model provenance differs from the WP1-confirmed contract")
    return initial, model, initial.tokenizer


def _projection(key: str, model):
    import torch
    hidden = (model.transformer.wte.weight.shape[1] if key == "huginn"
              else model.model.embed_tokens.weight.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    return torch.randn(hidden, PROJECTION_DIM, generator=generator).cuda()


def _encode(tokenizer, prompts):
    return tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_CAP)


def _check_source(allow_dirty: bool) -> str:
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--",
                                     "phase02_runtime", "phase03_predictor", "phase04_runtime", "phase05", "poc.py"],
                                    text=True, cwd=ROOT).strip()
    if dirty and not allow_dirty:
        raise RuntimeError("scored WP3 GPU runs require committed source")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()


# ---------------------------------------------------------------- collect

def collect_split(key, model, tokenizer, prompts, out: Path, threshold, projection, smoke_tokens=None) -> dict:
    import torch
    from phase02_runtime.common import run_interval
    from phase03_predictor.collect_huginn import SCALARS
    from phase03_predictor.physical import _features
    from phase05.wp2_headroom import _new_adapter
    tokens = smoke_tokens or T
    n = len(prompts)
    out.mkdir(parents=True, exist_ok=False)
    shape = (tokens, n)
    scalar_arr = np.lib.format.open_memmap(out / "scalars.npy", mode="w+", dtype="float32", shape=(*shape, len(SCALARS)))
    emb_arr = np.lib.format.open_memmap(out / "embedding.npy", mode="w+", dtype="float16", shape=(*shape, PROJECTION_DIM))
    rep_arr = np.lib.format.open_memmap(out / "prelude.npy", mode="w+", dtype="float16", shape=(*shape, PROJECTION_DIM))
    tokens_arr = np.zeros(shape, dtype=np.int32)
    next_arr = np.zeros(shape, dtype=np.int32)
    labels = np.zeros(shape, dtype=np.int16)
    encoded = _encode(tokenizer, prompts)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    ns = SimpleNamespace(model=key)
    kind = "huginn_convergence" if key == "huginn" else "ouro_gate"
    began = time.perf_counter()
    with torch.inference_mode():
        for rid in range(n):
            canonical = rid % COHORT  # physical cohorts use canonical ids 0..255
            adapter = _new_adapter(key, model, tokenizer, 1, [canonical], "benchmark")
            current = adapter.prefill(encoded["input_ids"][rid:rid + 1].cuda(),
                                      encoded["attention_mask"][rid:rid + 1].cuda().bool()).reshape(1, 1)
            previous_logits = adapter.prefill_logits[:, -1, :]
            previous_hidden = torch.zeros((1, PROJECTION_DIM), dtype=torch.float32, device="cuda")
            history, trajectory = [[]], [[]]
            row = torch.tensor([0], dtype=torch.long, device="cuda")
            for t in range(tokens):
                adapter.begin_token()
                adapter.backend.set_active(row)
                embedded, state = adapter.prelude(current)
                scalar, emb, rep = _features(ns, adapter, current, previous_logits, history, trajectory,
                                             previous_hidden, embedded, [prompt_lengths[rid]], projection, "C9")
                sink: dict[int, list[float]] = {}
                result = run_interval(adapter, embedded, state, row, row, exit_source="live", threshold=threshold,
                                      instrumentation="benchmark", debug_ids=False,
                                      trajectory_sink=sink, trajectory_kind=kind)
                adapter.backend.set_active(row)
                logits = adapter.coda_logits(result.final_state)
                nxt = logits.argmax(dim=-1).reshape(1, 1)
                adapter.commit_token()
                depth = int(result.exit_depths[0])
                scalar_arr[t, rid], emb_arr[t, rid], rep_arr[t, rid] = scalar[0], emb[0], rep[0]
                tokens_arr[t, rid], next_arr[t, rid], labels[t, rid] = int(current[0, 0]), int(nxt[0, 0]), depth
                previous_logits = logits[:, -1, :]
                if key == "ouro":
                    previous_hidden = result.final_state.float().squeeze(1) @ projection
                history[0].append(depth)
                trajectory[0] = sink[0]
                current = nxt
            del adapter
            if (rid + 1) % 64 == 0:
                print(json.dumps({"event": "collect_progress", "split_dir": out.name, "done": rid + 1, "of": n,
                                  "elapsed_s": round(time.perf_counter() - began, 1)}), flush=True)
    for arr in (scalar_arr, emb_arr, rep_arr):
        arr.flush()
    np.save(out / "token_ids.npy", tokens_arr)
    np.save(out / "b1_next_token.npy", next_arr)
    np.save(out / "depth_labels.npy", labels)
    return {"requests": n, "tokens": tokens, "prompt_token_counts": prompt_lengths,
            "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
            "elapsed_wall_s": time.perf_counter() - began,
            "artifact_sha256": {f: _sha(out / f) for f in ("scalars.npy", "embedding.npy", "prelude.npy",
                                                           "token_ids.npy", "b1_next_token.npy", "depth_labels.npy")}}


def cmd_collect(args) -> None:
    import torch
    from phase03_predictor.collect_huginn import SCALARS
    from phase05.wp1_confirmation import gpu_identity
    torch.set_num_threads(4)
    key = args.model
    gpu = gpu_identity(f"{key}_{args.role}")
    source_sha = _check_source(args.smoke)
    initial, model, tokenizer = _load(key)
    projection = _projection(key, model)
    texts = _dataset_texts(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "projection.npy", projection.cpu().numpy())
    base = {"model": key, "dataset": args.dataset, "source_sha": source_sha, "argv": sys.argv, "gpu": gpu,
            "threshold": CONTRACT[key]["threshold"], "rmax": initial.rmax, "kv_policy": initial.kv_policy,
            "h0_seed": H0_SEED if key == "huginn" else None, "h0_canonical_id": "request_index % 256",
            "scalar_names": list(SCALARS), "projection_dim": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
            "feature_function": "phase03_predictor.physical._features (same call as physical runtime)",
            "feature_causality": "all features computed after Prelude/embedding, before current-token recurrence",
            "prompt_cap": PROMPT_CAP, "padding": "max_length 64", "T": T, "splits_rule": SPLITS,
            "smoke": args.smoke}
    for split in args.splits:
        prompts = split_prompts(texts, split)
        if args.smoke:
            prompts = prompts[:64]
        meta = collect_split(key, model, tokenizer, prompts, args.output_dir / split,
                             CONTRACT[key]["threshold"], projection, 4 if args.smoke else None)
        manifest = {**base, "split": split, "status": "COMPLETE", **meta}
        (args.output_dir / split / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"event": "split_done", "split": split, "elapsed_s": round(meta["elapsed_wall_s"], 1)}),
              flush=True)


# ---------------------------------------------------------------- select (CPU, validation only)

def lockstep_recovery(labels: np.ndarray, scores: np.ndarray, batches=EVAL_B) -> dict[int, float]:
    """Frozen-label lockstep RIC recovery per B, cohort blocks of 256 requests (as in physical)."""
    from phase03_predictor.tournament import _recovery, _ric
    result = {}
    for b in batches:
        totals = np.zeros(3, dtype=np.int64)
        for start in range(0, labels.shape[1], COHORT):
            totals += np.asarray(_ric(labels[:, start:start + COHORT], scores[:, start:start + COHORT], b))
        result[b] = _recovery(*map(int, totals))
    return result


def cmd_select(args) -> None:
    import torch
    from phase03_predictor.tournament import (CANDIDATES, _prepare, _rank_metrics, _scores, _train_one,
                                              assemble_candidate_features)
    torch.set_num_threads(8)
    root = args.feature_root
    manifests = {s: json.loads((root / s / "feature_manifest.json").read_text()) for s in ("wp3_train", "wp3_validation")}
    for split, manifest in manifests.items():
        for name, digest in manifest["artifact_sha256"].items():
            if _sha(root / split / name) != digest:
                raise RuntimeError(f"checksum mismatch {split}/{name}")
    if set(manifests["wp3_train"]["prompt_sha256"]) & set(manifests["wp3_validation"]["prompt_sha256"]):
        raise RuntimeError("train/validation prompt overlap")
    manifest = {**manifests["wp3_train"], "model": manifests["wp3_train"]["model"]}

    def arrays(split):
        d = root / split
        return (np.load(d / "scalars.npy").astype(np.float32), np.load(d / "embedding.npy").astype(np.float32),
                np.load(d / "prelude.npy").astype(np.float32), np.load(d / "depth_labels.npy").astype(np.int16))
    tr, va = arrays("wp3_train"), arrays("wp3_validation")
    names = manifest["scalar_names"]
    for scal, _, _, lab in (tr, va):  # causal audit: feature at t sees only depths < t
        prev = scal[..., names.index("previous_depth")]
        if not (np.all(prev[0] == 0) and np.array_equal(prev[1:], lab[:-1])):
            raise RuntimeError("causal previous-depth audit failed")
    n_tr = tr[3].shape[1]
    labels = np.concatenate((tr[3], va[3]), axis=1)
    train_ids, val_ids = list(range(n_tr)), list(range(n_tr, labels.shape[1]))
    features = assemble_candidate_features(np.concatenate((tr[0], va[0]), 1), np.concatenate((tr[1], va[1]), 1),
                                           np.concatenate((tr[2], va[2]), 1), manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    rmax = manifest["rmax"]
    for candidate in CANDIDATES:
        start = time.perf_counter()
        if candidate == "C0":
            scores = np.concatenate((tr[0], va[0]), 1)[:, val_ids, names.index("previous_depth")]
            params, config, predict_s = 0, {"rule": "previous own depth (history_last)"}, 0.0
        else:
            x, mean, std = _prepare(features[candidate], train_ids)
            variants = ((0, 1e-3), (64, 1e-3)) if candidate in {"C4", "C5"} else ((64, 1e-3), (256, 3e-4))
            selected = None
            for width, lr in variants:
                head, config = _train_one(candidate, x, labels, train_ids, val_ids, 16, width, lr,
                                          args.epochs, args.seed, rmax)
                if selected is None or config["validation_ric_recovery"] > selected[1]["validation_ric_recovery"]:
                    selected = (head, config)
            head, config = selected
            torch.save({"state_dict": head.state_dict(), "input_dim": x.shape[-1],
                        "output_dim": rmax - 1 if candidate == "C7" else 1, "config": config,
                        "feature_mean": mean, "feature_std": std}, args.output_dir / f"{candidate}_checkpoint.pt")
            scores, predict_s = _scores(head, x[:, val_ids], candidate)
            params = sum(p.numel() for p in head.parameters())
        recovery = lockstep_recovery(labels[:, val_ids], scores)
        spear, kendall = _rank_metrics(labels[:, val_ids], scores)
        rows.append({"candidate": candidate, "split": "validation", "params": params,
                     **{f"recovery_B{b}": r for b, r in recovery.items()},
                     "mean_recovery": float(np.nanmean(list(recovery.values()))) if any(np.isfinite(list(recovery.values()))) else -1.0, "spearman": spear,
                     "kendall_tau_b": kendall, "predict_cpu_s_per_token_step": predict_s / labels.shape[0],
                     "train_wall_s": time.perf_counter() - start, "config": json.dumps(config)})
        print(json.dumps({k: rows[-1][k] for k in ("candidate", "mean_recovery", "spearman")}), flush=True)
    winner = max(rows, key=lambda r: (r["mean_recovery"], -r["params"]))
    with (args.output_dir / "validation_tournament.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "selection.json").write_text(json.dumps({
        "winner": winner["candidate"], "winner_mean_validation_recovery": winner["mean_recovery"],
        "rule": "max mean frozen-label lockstep RIC recovery over B in {8,16,32,64} on wp3_validation; tie -> fewer params",
        "test_arrays_opened": False, "seed": args.seed, "epochs_max": args.epochs,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "checkpoint_sha256": {p.name: _sha(p) for p in sorted(args.output_dir.glob("*_checkpoint.pt"))},
        "feature_manifest_sha256": {s: _sha(root / s / "feature_manifest.json") for s in manifests}},
        indent=2) + "\n")
    print(json.dumps({"winner": winner["candidate"], "mean_recovery": winner["mean_recovery"]}), flush=True)


# ---------------------------------------------------------------- physical (sealed test)

def physical_run(key, model, tokenizer, encoded, labels, tokens, prompt_lengths, threshold, B, policy,
                 predictor, projection) -> dict:
    import torch
    from phase02_runtime.common import run_interval
    from phase03_predictor.physical import _features
    from phase05.wp2_headroom import _new_adapter
    n = labels.shape[1]
    ns = SimpleNamespace(model=key)
    candidate = predictor.candidate if predictor is not None else None
    need_traj = policy == "predictor" and candidate in TRAJECTORY_CANDIDATES
    kind = "huginn_convergence" if key == "huginn" else "ouro_gate"
    adapter = _new_adapter(key, model, tokenizer, n, list(range(n)), "benchmark")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        adapter.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
        known = torch.tensor(tokens, dtype=torch.long, device="cuda")
        previous_logits = adapter.prefill_logits[:, -1, :].clone()
        previous_hidden = torch.zeros((n, PROJECTION_DIM), dtype=torch.float32, device="cuda")
        histories = [[] for _ in range(n)]
        trajectories = [[] for _ in range(n)]
        next_out = torch.empty((len(tokens), n), dtype=torch.long, device="cuda")
        cohorts, executed = [], []
        feature_s = sort_s = 0.0
        width = int(model.transformer.wte.weight.shape[1] if key == "huginn" else model.model.embed_tokens.weight.shape[1])
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in range(len(tokens)):
            current = known[t].unsqueeze(1)
            adapter.begin_token()
            embedded_all = torch.empty((n, 1, width), dtype=torch.bfloat16, device="cuda")
            state_all = torch.empty_like(embedded_all)
            for first in range(0, n, B):  # common staged Prelude in fixed FIFO chunks of B, every policy
                rows = torch.arange(first, first + B, dtype=torch.long, device="cuda")
                adapter.backend.set_active(rows)
                e, s = adapter.prelude(current.index_select(0, rows))
                embedded_all.index_copy_(0, rows, e)
                state_all.index_copy_(0, rows, s)
            s0 = time.perf_counter()
            ids = np.arange(n)
            if policy == "fifo":
                order = ids
            elif policy == "oracle_b1":
                order = np.lexsort((ids, labels[t]))
            elif policy == "history_last":
                order = np.lexsort((ids, np.array([h[-1] if h else 0 for h in histories])))
            else:
                scalar, emb, rep = _features(ns, adapter, current, previous_logits, histories, trajectories,
                                             previous_hidden, embedded_all, prompt_lengths, projection, candidate)
                scores = predictor.score(scalar, emb, rep)
                feature_s += time.perf_counter() - s0
                s1 = time.perf_counter()
                order = np.lexsort((ids, scores))
                sort_s += time.perf_counter() - s1
            if policy != "predictor":
                sort_s += time.perf_counter() - s0
            step_depth = torch.empty(n, dtype=torch.long, device="cuda")
            step = []
            sink: dict[int, list[float]] | None = {} if need_traj else None
            for first in range(0, n, B):
                members = order[first:first + B].tolist()
                rows = torch.tensor(members, dtype=torch.long, device="cuda")
                result = run_interval(adapter, embedded_all.index_select(0, rows), state_all.index_select(0, rows),
                                      rows, rows, exit_source="live", threshold=threshold, instrumentation="benchmark",
                                      debug_ids=False, trajectory_sink=sink, trajectory_kind=kind if need_traj else None)
                adapter.backend.set_active(rows)
                logits = adapter.coda_logits(result.final_state)
                next_out[t].index_copy_(0, rows, logits.argmax(dim=-1).reshape(-1))
                if policy == "predictor":
                    previous_logits.index_copy_(0, rows, logits[:, -1, :])
                    if key == "ouro":
                        previous_hidden.index_copy_(0, rows, result.final_state.float().squeeze(1) @ projection)
                step_depth.index_copy_(0, rows, result.exit_depths)
                step.append((members, [e["active_batch_size"] for e in result.events]))
            adapter.commit_token()
            depths_t = step_depth.tolist()
            for r in range(n):
                histories[r].append(depths_t[r])
                if sink is not None:
                    trajectories[r] = sink[r]
            for members, widths in step:
                cohorts.append({"t": t, "members": members, "widths": widths, "depths": [depths_t[r] for r in members]})
                executed += [(t, r) for r in members]
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - start
        nexts = next_out.cpu().numpy()
    errors = row_identity_errors(executed, len(tokens), n)
    for c in cohorts:
        errors += cohort_errors(c["members"], c["depths"], c["widths"], [], [], len(c["members"]))
    live = np.array([[h[t] for h in histories] for t in range(len(tokens))])
    return {"decode_wall_s": decode_s, "feature_and_predict_s": feature_s, "sort_s": sort_s,
            "live_work": int(live.sum()), "b1_work": int(labels.sum()),
            "live_RIC": sum(max(c["depths"]) for c in cohorts),
            "frozen_b1_RIC_same_grouping": sum(max(int(labels[c["t"], r]) for r in c["members"]) for c in cohorts),
            "depth_flips": int((live != labels).sum()), "next_token_flips": int((nexts[:-1] != tokens[1:]).sum()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "n_hard_errors": len(errors), "hard_check_errors": errors[:20], "live_depth": live}


def physical_plan(repeats=REPEATS, cohorts=(0, 1)) -> list[dict]:
    import random
    plan = [{"kind": "warmup", "cohort": 0, "B": 32, "policy": "fifo", "rep": -1}]
    for rep in range(repeats):
        for cohort in cohorts:
            order = list(EVAL_B)
            random.Random(f"wp3-plan:{rep}:{cohort}").shuffle(order)
            for b in order:
                k = rep % len(PHYSICAL_POLICIES)
                for policy in PHYSICAL_POLICIES[k:] + PHYSICAL_POLICIES[:k]:
                    plan.append({"kind": "scored", "cohort": cohort, "B": b, "policy": policy, "rep": rep})
    return plan


def cmd_physical(args) -> None:
    import torch
    from phase03_predictor.physical import Predictor
    from phase05.wp1_confirmation import gpu_identity
    torch.set_num_threads(4)
    key = args.model
    gpu = gpu_identity(f"{key}_{args.role}")
    source_sha = _check_source(args.smoke)
    selection = json.loads((args.selection_root / "selection.json").read_text())
    for name, digest in selection["checkpoint_sha256"].items():
        if _sha(args.selection_root / name) != digest:
            raise RuntimeError(f"frozen checkpoint changed: {name}")
    feature_manifest = json.loads((args.selection_root / "feature_manifest.json").read_text())
    if feature_manifest["model"] != key:
        raise RuntimeError("selection belongs to another model")
    predictor = Predictor(selection["winner"], args.selection_root, feature_manifest)
    initial, model, tokenizer = _load(key)
    projection = _projection(key, model)
    collected = np.load(args.feature_root / "projection.npy")
    if not np.array_equal(collected, projection.cpu().numpy()):
        raise RuntimeError("projection differs from the collected features")
    test_dir = args.feature_root / "wp3_test"
    test_manifest = json.loads((test_dir / "feature_manifest.json").read_text())
    labels_all = np.load(test_dir / "depth_labels.npy").astype(np.int64)
    tokens_all = np.load(test_dir / "token_ids.npy").astype(np.int64)
    texts = _dataset_texts(args.dataset)
    prompts_all = split_prompts(texts, "wp3_test")[:len(test_manifest["prompt_sha256"])]
    if [hashlib.sha256(p.encode()).hexdigest() for p in prompts_all] != test_manifest["prompt_sha256"]:
        raise RuntimeError("test prompts differ from collected test features")
    cohorts = (0,) if args.smoke else (0, 1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_manifest.json").write_text(json.dumps({
        "execution_kind": "UNSCORED_SMOKE" if args.smoke else "sealed-test physical no-refill, live BF16 exits",
        "preregistration": "phase05/WP3_PREREGISTRATION.md", "source_sha": source_sha, "argv": sys.argv,
        "gpu": gpu, "cpu_affinity": sorted(os.sched_getaffinity(0)), "model": key, "dataset": args.dataset,
        "selection": selection, "predictor": selection["winner"], "cohorts": [[c * COHORT, (c + 1) * COHORT] for c in cohorts],
        "policies": PHYSICAL_POLICIES, "B": EVAL_B, "repeats": REPEATS, "T": T,
        "test_feature_manifest_sha256": _sha(test_dir / "feature_manifest.json"),
        "torch": torch.__version__}, indent=2) + "\n")
    fields = ["run_id", "kind", "cohort", "B", "policy", "rep", "status", "decode_wall_s", "feature_and_predict_s",
              "sort_s", "live_work", "b1_work", "live_RIC", "frozen_b1_RIC_same_grouping", "depth_flips",
              "next_token_flips", "peak_allocated_gib", "n_hard_errors", "error"]
    with (out / "runs.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    live_rows = []
    plan = physical_plan(1 if args.smoke else REPEATS, cohorts)
    if args.smoke:
        plan = [c for c in plan if c["B"] in (32, 64)]
    hard = 0
    for run_id, cell in enumerate(plan):
        sl = slice(cell["cohort"] * COHORT, (cell["cohort"] + 1) * COHORT)
        prompts = prompts_all[sl]
        encoded = _encode(tokenizer, prompts)
        tokens = tokens_all[:args.tokens or T, sl]
        labels = labels_all[:args.tokens or T, sl]
        row = {"run_id": run_id, **cell, "status": "OK", "error": ""}
        try:
            result = physical_run(key, model, tokenizer, encoded, labels, tokens,
                                  encoded["attention_mask"].sum(dim=1).tolist(), CONTRACT[key]["threshold"],
                                  cell["B"], cell["policy"], predictor, projection)
            row.update({k: result[k] for k in fields if k in result})
            live_rows.append((run_id, result["live_depth"].astype(np.int16)))
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
    np.savez_compressed(out / "live_depths.npz", **{f"run_{i}": d for i, d in live_rows})
    print(json.dumps({"event": "lane_done", "hard_failures": hard}), flush=True)
    sys.exit(1 if hard else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True, choices=("ouro", "huginn"))
    c.add_argument("--role", required=True, choices=("primary", "replication"))
    c.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    c.add_argument("--splits", nargs="+", required=True, choices=("wp3_train", "wp3_validation", "wp3_test"))
    c.add_argument("--output-dir", required=True, type=Path)
    c.add_argument("--smoke", action="store_true")
    s = sub.add_parser("select")
    s.add_argument("--feature-root", required=True, type=Path)
    s.add_argument("--output-dir", required=True, type=Path)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("physical")
    p.add_argument("--model", required=True, choices=("ouro", "huginn"))
    p.add_argument("--role", required=True, choices=("primary", "replication"))
    p.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    p.add_argument("--feature-root", required=True, type=Path)
    p.add_argument("--selection-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tokens", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    {"collect": cmd_collect, "select": cmd_select, "physical": cmd_physical}[args.command](args)


if __name__ == "__main__":
    main()
