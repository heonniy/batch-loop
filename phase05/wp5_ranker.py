"""Phase-05 WP5: a stronger causal current-token depth ranker ("R").

Inputs, all available before the current token's recurrence:
  * Phase-03 scalar block (38): position, own depth history, previous-token
    uncertainty and previous-token convergence/gate trajectory;
  * token-ID target encodings (current, previous, hashed bigram) of the canonical
    B1 depth, fit on wp3_train only (out-of-fold for train rows);
  * FULL-dimensional states instead of 64-d random projections: current-token
    embedding, previous token's final recurrent state, and (Huginn) the Prelude
    output; each passes through a learned low-rank projection.
Trained with the Phase-03 weighted pairwise loss within a token step, early-stopped
and tuned on wp3_validation only. See phase05/WP5_PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from phase05.wp3_predictor import COHORT, EVAL_B, lockstep_recovery

ROOT = Path(__file__).resolve().parents[1]
BIGRAM_BUCKETS = 1 << 21
TE_SMOOTHING = 20.0
FOLDS = 5
GRID = ({"proj": 64, "width": 256, "lr": 1e-3}, {"proj": 128, "width": 256, "lr": 3e-4},
        {"proj": 256, "width": 512, "lr": 3e-4})


# ---------------------------------------------------------------- token target encodings

def token_keys(tokens: np.ndarray, vocab: int) -> dict[str, np.ndarray]:
    """[T,N] current/previous/bigram keys; t=0 previous token is the sentinel `vocab`."""
    prev = np.vstack([np.full((1, tokens.shape[1]), vocab), tokens[:-1]]).astype(np.int64)
    cur = tokens.astype(np.int64)
    return {"cur": cur, "prev": prev, "bigram": (prev * 1000003 + cur) % BIGRAM_BUCKETS}


def te_table(keys: np.ndarray, y: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.bincount(keys.ravel(), weights=y.ravel().astype(np.float64), minlength=size)
    count = np.bincount(keys.ravel(), minlength=size).astype(np.float64)
    prior = float(y.mean())
    return ((total + TE_SMOOTHING * prior) / (count + TE_SMOOTHING)).astype(np.float32), np.log1p(count).astype(np.float32)


def te_sizes(vocab: int) -> dict[str, int]:
    return {"cur": vocab + 1, "prev": vocab + 1, "bigram": BIGRAM_BUCKETS}


def te_features(keys: dict[str, np.ndarray], tables: dict) -> np.ndarray:
    return np.stack([tables[k][j][keys[k]] for k in ("cur", "prev", "bigram") for j in (0, 1)], axis=-1)


def te_train_oof(keys: dict[str, np.ndarray], y: np.ndarray, vocab: int) -> np.ndarray:
    """Out-of-fold (by request) encodings for training rows, so no row sees its own label."""
    out = np.zeros(y.shape + (6,), dtype=np.float32)
    fold = np.arange(y.shape[1]) % FOLDS
    sizes = te_sizes(vocab)
    for f in range(FOLDS):
        fit, apply = fold != f, fold == f
        tables = {k: te_table(keys[k][:, fit], y[:, fit], sizes[k]) for k in keys}
        out[:, apply] = te_features({k: v[:, apply] for k, v in keys.items()}, tables)
    return out


# ---------------------------------------------------------------- model

def build_model(scalar_dim: int, full_dims: list[int], proj: int, width: int):
    import torch
    from torch import nn

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList(nn.Sequential(nn.Linear(d, proj), nn.LayerNorm(proj)) for d in full_dims)
            self.mlp = nn.Sequential(nn.Linear(scalar_dim + proj * len(full_dims), width), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, scalar, fulls):
            parts = [scalar] + [p(x) for p, x in zip(self.proj, fulls)]
            return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
    return Ranker()


class Data:
    """Standardized tensors on the GPU: scalar [T,N,S] float32, fulls list of [T,N,H] float16."""

    def __init__(self, scalar, fulls, labels):
        self.scalar, self.fulls, self.labels = scalar, fulls, labels


def load_split(split_dir: Path, full_names: list[str]):
    scalar = np.load(split_dir / "scalars.npy").astype(np.float32)
    fulls = [np.load(split_dir / f"{name}.npy", mmap_mode="r") for name in full_names]
    return scalar, fulls, np.load(split_dir / "token_ids.npy"), np.load(split_dir / "depth_labels.npy").astype(np.int64)


def full_stats(arr) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])
    mean, std = flat.mean(0), flat.std(0)
    std[std < 1e-4] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def to_gpu(scalar, fulls, labels, s_mean, s_std, f_stats):
    import torch
    return Data(torch.as_tensor((scalar - s_mean) / s_std, device="cuda"),
                [torch.as_tensor(((np.asarray(f, dtype=np.float32) - m) / s).astype(np.float16), device="cuda")
                 for f, (m, s) in zip(fulls, f_stats)],
                torch.as_tensor(labels, device="cuda"))


def score_all(model, data: Data) -> np.ndarray:
    import torch
    model.eval()
    with torch.inference_mode():
        return np.stack([model(data.scalar[t], [f[t].float() for f in data.fulls]).float().cpu().numpy()
                         for t in range(data.scalar.shape[0])])


def pairwise_loss(scores, depths, generator, count=4096):
    import torch
    import torch.nn.functional as F
    n = depths.shape[0]
    left = torch.randint(0, n, (count,), device="cuda", generator=generator)
    right = torch.randint(0, n, (count,), device="cuda", generator=generator)
    diff = (depths[right] - depths[left]).float()
    usable = diff != 0
    if not bool(usable.any()):
        return scores.sum() * 0
    weight = diff[usable].abs().clamp(max=8) / 8
    return (weight * F.softplus(-torch.sign(diff[usable]) * (scores[right[usable]] - scores[left[usable]]))).mean()


def train_one(train: Data, val: Data, val_labels: np.ndarray, cfg: dict, seed: int, epochs: int,
              ablate_full: bool = False) -> tuple[object, dict]:
    import torch
    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fulls_dims = [] if ablate_full else [f.shape[-1] for f in train.fulls]
    model = build_model(train.scalar.shape[-1], fulls_dims, cfg["proj"], cfg["width"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps, n = train.scalar.shape[0], train.scalar.shape[1]
    best, best_state, best_epoch, patience = -1.0, None, -1, 0
    val_data = val if not ablate_full else Data(val.scalar, [], val.labels)
    for epoch in range(epochs):
        model.train()
        for t in torch.randperm(steps, generator=torch.Generator().manual_seed(seed + epoch)).tolist():
            for chunk in torch.randperm(n, device="cuda", generator=gen).split(COHORT):
                opt.zero_grad()
                fulls = [] if ablate_full else [f[t].index_select(0, chunk).float() for f in train.fulls]
                scores = model(train.scalar[t].index_select(0, chunk), fulls)
                loss = pairwise_loss(scores, train.labels[t].index_select(0, chunk), gen)
                loss.backward()
                opt.step()
        rec = lockstep_recovery(val_labels, score_all(model, val_data))
        value = float(np.nanmean(list(rec.values())))
        if value > best + 1e-6:
            best, best_epoch, patience = value, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best_state)
    return model, {**cfg, "best_epoch": best_epoch, "validation_mean_recovery": best, "ablate_full": ablate_full}


def cmd_train(args) -> None:
    import torch
    torch.set_num_threads(8)
    lane = args.feature_root
    manifest = json.loads((lane / "wp3_train" / "feature_manifest.json").read_text())
    key = manifest["model"]
    full_names = manifest["full_dim_arrays"]
    for split in ("wp3_train", "wp3_validation"):
        m = json.loads((lane / split / "feature_manifest.json").read_text())
        for name, digest in m["artifact_sha256"].items():
            if hashlib.sha256((lane / split / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"checksum mismatch {split}/{name}")
    vocab = int(args.vocab)
    s_tr, f_tr, tok_tr, y_tr = load_split(lane / "wp3_train", full_names)
    s_va, f_va, tok_va, y_va = load_split(lane / "wp3_validation", full_names)
    k_tr, k_va = token_keys(tok_tr, vocab), token_keys(tok_va, vocab)
    sizes = te_sizes(vocab)
    tables = {k: te_table(k_tr[k], y_tr, sizes[k]) for k in k_tr}
    s_tr = np.concatenate((s_tr, te_train_oof(k_tr, y_tr, vocab)), -1)
    s_va = np.concatenate((s_va, te_features(k_va, tables)), -1)
    s_mean, s_std = s_tr.reshape(-1, s_tr.shape[-1]).mean(0), s_tr.reshape(-1, s_tr.shape[-1]).std(0)
    s_std[s_std < 1e-5] = 1.0
    f_stats = [full_stats(f) for f in f_tr]
    train = to_gpu(s_tr, f_tr, y_tr, s_mean, s_std, f_stats)
    val = to_gpu(s_va, f_va, y_va, s_mean, s_std, f_stats)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, best = [], None
    started = time.perf_counter()
    for i, cfg in enumerate(GRID):
        model, info = train_one(train, val, y_va, cfg, args.seed, args.epochs)
        rows.append(info)
        print(json.dumps({"event": "grid", **info}), flush=True)
        if best is None or info["validation_mean_recovery"] > best[1]["validation_mean_recovery"]:
            best = (model, info)
    ablation_model, ablation = train_one(train, val, y_va, best[1], args.seed, args.epochs, ablate_full=True)
    print(json.dumps({"event": "ablation_no_full_dim", **ablation}), flush=True)
    model, info = best
    val_scores = score_all(model, val)
    rec = lockstep_recovery(y_va, val_scores)
    torch.save({"state_dict": model.state_dict(), "config": info, "scalar_dim": int(s_tr.shape[-1]),
                "full_names": full_names, "full_dims": [int(f.shape[-1]) for f in f_tr],
                "scalar_mean": s_mean, "scalar_std": s_std, "full_stats": f_stats,
                "te_tables": {k: tables[k] for k in tables}, "vocab": vocab, "bigram_buckets": BIGRAM_BUCKETS},
               args.output_dir / "ranker_checkpoint.pt")
    summary = {"model": key, "dataset": manifest["dataset"], "grid": rows, "selected": info,
               "validation_recovery_by_B": {str(b): v for b, v in rec.items()},
               "ablation_without_full_dim_states": ablation,
               "selection_rule": "max mean validation lockstep RIC recovery over B in {8,16,32,64}",
               "test_arrays_opened": False, "seed": args.seed, "epochs_max": args.epochs,
               "train_wall_s": time.perf_counter() - started,
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()}
    summary["checkpoint_sha256"] = hashlib.sha256((args.output_dir / "ranker_checkpoint.pt").read_bytes()).hexdigest()
    (args.output_dir / "ranker_selection.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({"event": "selected", "validation_mean_recovery": info["validation_mean_recovery"],
                      "by_B": summary["validation_recovery_by_B"]}), flush=True)


# ---------------------------------------------------------------- GPU serving

class GpuRanker:
    """Frozen ranker + encodings + standardization on device; no host sync."""

    def __init__(self, checkpoint_path: Path):
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.model = build_model(ckpt["scalar_dim"], ckpt["full_dims"], cfg["proj"], cfg["width"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.cuda().eval()
        dev = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device="cuda")
        self.s_mean, self.s_std = dev(ckpt["scalar_mean"]), dev(ckpt["scalar_std"])
        self.f_stats = [(dev(m), dev(s)) for m, s in ckpt["full_stats"]]
        self.full_names = ckpt["full_names"]
        self.tables = {k: (dev(v[0]), dev(v[1])) for k, v in ckpt["te_tables"].items()}
        self.vocab, self.buckets = int(ckpt["vocab"]), int(ckpt["bigram_buckets"])

    def score(self, scalar38, cur_tok, prev_tok, fulls: dict):
        import torch
        keys = {"cur": cur_tok, "prev": prev_tok, "bigram": (prev_tok * 1000003 + cur_tok) % self.buckets}
        te = torch.stack([self.tables[k][j][keys[k]] for k in ("cur", "prev", "bigram") for j in (0, 1)], dim=-1)
        scalar = (torch.cat((scalar38, te), dim=-1) - self.s_mean) / self.s_std
        xs = [(fulls[name] - m) / s for name, (m, s) in zip(self.full_names, self.f_stats)]
        return self.model(scalar, xs)


POLICIES5 = ("fifo", "history_last", "predictor_gpu", "ranker_gpu", "oracle_b1")


def cmd_physical(args) -> None:
    """Gate-0 wiring audit on wp3_validation, or the scored physical run on sealed wp5_test."""
    import csv
    import os
    import random
    import statistics
    import sys
    import torch
    from phase05.wp1_confirmation import CONTRACT, gpu_identity
    from phase05.wp2_headroom import split_prompts
    from phase05.wp3_predictor import _check_source, _dataset_texts, _encode, _load, _projection, _sha
    from phase05.wp4_gpu_serving import GpuPredictor, gpu_run
    torch.set_num_threads(4)
    key = args.model
    gpu = gpu_identity(f"{key}_{args.role}")
    source_sha = _check_source(args.smoke)
    sel = json.loads((args.ranker_root / "ranker_selection.json").read_text())
    if _sha(args.ranker_root / "ranker_checkpoint.pt") != sel["checkpoint_sha256"]:
        raise RuntimeError("frozen ranker changed")
    w3sel = json.loads((args.wp3_selection_root / "selection.json").read_text())
    for name, digest in w3sel["checkpoint_sha256"].items():
        if _sha(args.wp3_selection_root / name) != digest:
            raise RuntimeError(f"WP3 checkpoint changed: {name}")
    ranker = GpuRanker(args.ranker_root / "ranker_checkpoint.pt")
    predictor = GpuPredictor(args.wp3_selection_root, key)
    _, model, tokenizer = _load(key)
    projection = _projection(key, model)
    if not np.array_equal(np.load(args.wp3_feature_root / "projection.npy"), projection.cpu().numpy()):
        raise RuntimeError("projection differs from the WP3 features")
    split = "wp3_validation" if args.audit_only else "wp5_test"
    d = args.feature_root / split
    man = json.loads((d / "feature_manifest.json").read_text())
    labels_all = np.load(d / "depth_labels.npy").astype(np.int64)
    tokens_all = np.load(d / "token_ids.npy").astype(np.int64)
    prompts_all = split_prompts(_dataset_texts(args.dataset), split)[:len(man["prompt_sha256"])]
    if [hashlib.sha256(p.encode()).hexdigest() for p in prompts_all] != man["prompt_sha256"]:
        raise RuntimeError("prompts differ from collected labels")
    size = min(COHORT, len(prompts_all))
    cohorts = tuple(range(len(prompts_all) // size))[:2]
    steps = args.tokens or labels_all.shape[0]
    threshold = CONTRACT[key]["threshold"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    base = {"preregistration": "phase05/WP5_PREREGISTRATION.md", "source_sha": source_sha, "argv": sys.argv,
            "gpu": gpu, "cpu_affinity": sorted(os.sched_getaffinity(0)), "model": key, "dataset": args.dataset,
            "ranker_selection": sel, "wp3_predictor": w3sel["winner"], "split": split, "torch": torch.__version__}

    def inputs(c):
        sl = slice(c * size, (c + 1) * size)
        return _encode(tokenizer, prompts_all[sl]), labels_all[:steps, sl], tokens_all[:steps, sl]

    if args.audit_only:  # live BF16 batched ranker scores vs offline scores from B1-collected arrays
        encoded, labels, tokens = inputs(0)
        result = gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, 8, "ranker_gpu", predictor,
                         projection, ranker=ranker, capture_scores=True)
        scal = np.load(d / "scalars.npy")[:steps, :size].astype(np.float32)
        fulls = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ranker.full_names}
        rho = []
        with torch.inference_mode():
            for t in range(steps):
                cur = torch.as_tensor(tokens[t], device="cuda")
                prev = torch.as_tensor(tokens[t - 1], device="cuda") if t else torch.full_like(cur, ranker.vocab)
                off = ranker.score(torch.as_tensor(scal[t], device="cuda"), cur, prev,
                                   {n: torch.as_tensor(np.asarray(f[t, :size], dtype=np.float32), device="cuda")
                                    for n, f in fulls.items()}).cpu().numpy()
                live = result["scores"][t]
                r = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")
                rho.append(float(np.corrcoef(r(off), r(live))[0, 1]))
        summary = {"execution_kind": "UNTIMED ranker wiring audit on wp3_validation cohort 0", **base,
                   "B": 8, "steps": steps, "n_hard_errors": result["n_hard_errors"],
                   "mean_spearman_live_vs_offline": statistics.mean(rho), "min_spearman": min(rho), "per_step": rho}
        summary["status"] = "PASS" if summary["mean_spearman_live_vs_offline"] >= 0.95 and not result["n_hard_errors"] else "FAIL"
        (out / "ranker_audit.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        print(json.dumps({k: summary[k] for k in ("status", "mean_spearman_live_vs_offline", "min_spearman")}), flush=True)
        sys.exit(0 if summary["status"] == "PASS" else 1)

    plan = [{"kind": "warmup", "cohort": 0, "B": 32, "policy": "fifo", "rep": -1}]
    for rep in range(1 if args.smoke else 3):
        for c in cohorts:
            order = list(EVAL_B)
            random.Random(f"wp5-plan:{rep}:{c}").shuffle(order)
            for b in order:
                k = rep % len(POLICIES5)
                plan += [{"kind": "scored", "cohort": c, "B": b, "policy": p, "rep": rep}
                         for p in POLICIES5[k:] + POLICIES5[:k]]
    if args.smoke:
        plan = [c for c in plan if c["B"] in (32, 64)]
    (out / "run_manifest.json").write_text(json.dumps({
        "execution_kind": "UNSCORED_SMOKE" if args.smoke else "sealed wp5_test physical no-refill, live BF16 exits",
        **base, "cohorts": [[c * size, (c + 1) * size] for c in cohorts], "policies": POLICIES5, "B": EVAL_B,
        "repeats": 3, "T": steps}, indent=2, default=float) + "\n")
    fields = ["run_id", "kind", "cohort", "B", "policy", "rep", "status", "decode_wall_s", "feature_and_predict_s",
              "sort_s", "live_work", "b1_work", "live_RIC", "frozen_b1_RIC_same_grouping", "depth_flips",
              "next_token_flips", "peak_allocated_gib", "n_hard_errors", "error"]
    with (out / "runs.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    live_rows, hard = [], 0
    for run_id, cell in enumerate(plan):
        encoded, labels, tokens = inputs(cell["cohort"])
        row = {"run_id": run_id, **cell, "status": "OK", "error": ""}
        try:
            result = gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, cell["B"], cell["policy"],
                             predictor, projection, ranker=ranker)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--feature-root", required=True, type=Path)
    t.add_argument("--output-dir", required=True, type=Path)
    t.add_argument("--vocab", required=True, type=int)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("physical")
    p.add_argument("--model", required=True, choices=("ouro", "huginn"))
    p.add_argument("--role", required=True, choices=("primary", "replication"))
    p.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    p.add_argument("--feature-root", required=True, type=Path, help="WP5 lane features (wp3_validation, wp5_test)")
    p.add_argument("--ranker-root", required=True, type=Path)
    p.add_argument("--wp3-feature-root", required=True, type=Path)
    p.add_argument("--wp3-selection-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--tokens", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    {"train": cmd_train, "physical": cmd_physical}[args.command](args)


if __name__ == "__main__":
    main()
