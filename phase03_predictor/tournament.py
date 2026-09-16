"""CPU-only C0-C9 request-split RIC tournament over frozen causal features.

C9 uses a documented max-sensitive adjacent-microbatch swap surrogate: for
each score-sorted neighboring B32 pair, it penalizes an inversion only when
exchanging the deepest item in the earlier batch for the shallowest item in
the later batch would reduce the sum of true microbatch maxima. A small
weighted-pairwise term prevents the sparse swap objective from stalling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CANDIDATES = tuple(f"C{index}" for index in range(10))


class Head(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int):
        super().__init__()
        self.net = (nn.Linear(input_dim, output_dim) if width == 0 else
                    nn.Sequential(nn.Linear(input_dim, width), nn.GELU(),
                                  nn.Linear(width, 64), nn.GELU(), nn.Linear(64, output_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indices(names: tuple[str, ...], wanted: tuple[str, ...]) -> list[int]:
    return [names.index(name) for name in wanted]


def assemble_candidate_features(scalar: np.ndarray, embedding: np.ndarray,
                                prelude: np.ndarray, manifest: dict) -> dict[str, np.ndarray]:
    """Build identical train and physical feature views from causal raw arrays."""
    names = tuple(manifest["scalar_names"])
    position = ("token_position", "context_length")
    depth = ("previous_depth", "previous_available", "depth_mean_2", "depth_std_2",
             "depth_min_2", "depth_max_2", "depth_mean_4", "depth_std_4",
             "depth_min_4", "depth_max_4", "depth_mean_8", "depth_std_8",
             "depth_min_8", "depth_max_8", "depth_trend")
    trajectory = ("previous_trajectory_length",) + tuple(
        f"previous_convergence_{index + 1}" for index in range(16))
    uncertainty = ("sampled_surprisal", "entropy", "top1_probability", "top1_top2_margin")
    def slice_(fields: tuple[str, ...]) -> np.ndarray:
        return scalar[..., _indices(names, fields)]
    base = slice_(position)
    fused = np.concatenate((scalar, embedding, prelude), axis=-1)
    representation_probe = (np.concatenate((base, prelude), axis=-1)
                            if manifest["model"].startswith("huginn") else
                            np.concatenate((base, embedding, prelude), axis=-1))
    return {
        "C1": slice_(position + depth),
        "C2": slice_(position + ("previous_depth", "previous_available") + trajectory),
        "C3": slice_(position + uncertainty),
        "C4": np.concatenate((base, embedding), axis=-1),
        "C5": representation_probe,
        "C6": fused, "C7": fused, "C8": fused, "C9": fused,
    }


def _feature_sets(root: Path, manifest: dict) -> dict[str, np.ndarray]:
    scalar = np.asarray(np.load(root / "scalars.npy", mmap_mode="r"), dtype=np.float32)
    embedding = np.asarray(np.load(root / "embedding.npy", mmap_mode="r"), dtype=np.float32)
    prelude = np.asarray(np.load(root / "prelude.npy", mmap_mode="r"), dtype=np.float32)
    return assemble_candidate_features(scalar, embedding, prelude, manifest)


def _ric(depths: np.ndarray, scores: np.ndarray, batch_size: int) -> tuple[int, int, int]:
    if depths.shape != scores.shape or depths.ndim != 2 or depths.shape[1] % batch_size:
        raise ValueError("RIC requires equal [token,request] matrices and full microbatches")
    fifo = pred = oracle = 0
    ids = np.arange(depths.shape[1])
    for d, s in zip(depths, scores):
        fifo += int(d.reshape(-1, batch_size).max(axis=1).sum())
        predicted = np.lexsort((ids, s))
        pred += int(d[predicted].reshape(-1, batch_size).max(axis=1).sum())
        oracle += int(np.sort(d).reshape(-1, batch_size).max(axis=1).sum())
    return fifo, pred, oracle


def _recovery(fifo: int, pred: int, oracle: int) -> float:
    return (fifo - pred) / (fifo - oracle) if fifo != oracle else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        result[order[left:right]] = (left + right - 1) / 2
        left = right
    return result


def _rank_metrics(depths: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    spearman = []
    kendall = []
    upper = np.triu_indices(depths.shape[1], 1)
    for d, s in zip(depths, scores):
        rd, rs = _rank(d), _rank(s)
        corr = np.corrcoef(rd, rs)[0, 1] if rd.std() and rs.std() else 0.0
        spearman.append(float(corr))
        dd = np.sign(d[:, None] - d[None, :])[upper]
        ss = np.sign(s[:, None] - s[None, :])[upper]
        denom = math.sqrt(float(np.count_nonzero(dd)) * float(np.count_nonzero(ss)))
        kendall.append(float((dd * ss).sum() / denom) if denom else 0.0)
    return float(np.mean(spearman)), float(np.mean(kendall))


def _scores(head: Head, x: np.ndarray, candidate: str) -> tuple[np.ndarray, float]:
    head.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        result = head(torch.from_numpy(np.ascontiguousarray(x.reshape(-1, x.shape[-1]))))
        if candidate == "C7":
            result = 1.0 + torch.sigmoid(result).sum(dim=-1)
        else:
            result = result.squeeze(-1)
        scores = result.numpy().reshape(x.shape[:2])
    return scores, time.perf_counter() - start


def _prepare(raw: np.ndarray, train_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = raw[:, train_ids].reshape(-1, raw.shape[-1])
    mean = training.mean(axis=0)
    std = training.std(axis=0)
    std[std < 1e-5] = 1.0
    scaled = (raw - mean) / std
    return np.ascontiguousarray(scaled, dtype=np.float32), mean, std


def _pairwise_loss(scores: torch.Tensor, depths: torch.Tensor, rng: np.random.Generator,
                   count: int = 1024) -> torch.Tensor:
    n = len(depths)
    left = torch.from_numpy(rng.integers(0, n, size=count))
    right = torch.from_numpy(rng.integers(0, n, size=count))
    difference = depths[right] - depths[left]
    usable = difference != 0
    if not bool(usable.any()):
        return scores.sum() * 0
    signed = torch.sign(difference[usable])
    weights = difference[usable].abs().clamp(max=8) / 8
    return (weights * F.softplus(-signed * (scores[right[usable]] - scores[left[usable]]))).mean()


def _listwise_loss(scores: torch.Tensor, depths: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Max-sensitive adjacent-group swap regret surrogate, score gradient only."""
    order = torch.argsort(scores.detach(), stable=True)
    n = len(order)
    if n % batch_size:
        raise ValueError("listwise training requests must make full B groups")
    terms = []
    for start in range(0, n - batch_size, batch_size):
        first = order[start:start + batch_size]
        second = order[start + batch_size:start + 2 * batch_size]
        first_depths = depths[first]
        second_depths = depths[second]
        i = first[torch.argmax(first_depths)]
        j = second[torch.argmin(second_depths)]
        if depths[i] <= depths[j]:
            continue
        old_max = first_depths.max() + second_depths.max()
        swapped_first = torch.maximum(first_depths[first != i].max(), depths[j])
        swapped_second = torch.maximum(second_depths[second != j].max(), depths[i])
        gain = (old_max - swapped_first - swapped_second).clamp(min=0)
        if float(gain) > 0:
            terms.append(gain * F.softplus(scores[j] - scores[i]))
    return torch.stack(terms).mean() if terms else scores.sum() * 0


def _train_one(candidate: str, x: np.ndarray, labels: np.ndarray,
               train_ids: list[int], validation_ids: list[int], batch_size: int,
               width: int, lr: float, epochs: int, seed: int,
               rmax: int = 16) -> tuple[Head, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output_dim = rmax - 1 if candidate == "C7" else 1
    head = Head(x.shape[-1], output_dim, width)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    train_x = torch.from_numpy(np.ascontiguousarray(x[:, train_ids]))
    train_y = torch.from_numpy(np.ascontiguousarray(labels[:, train_ids], dtype=np.float32))
    validation_x = x[:, validation_ids]
    validation_y = labels[:, validation_ids]
    best = -float("inf")
    best_state: dict | None = None
    best_epoch = -1
    patience = 0
    for epoch in range(epochs):
        head.train()
        if candidate in {"C8", "C9"}:
            for token_idx in rng.permutation(train_x.shape[0]):
                optimizer.zero_grad()
                scores = head(train_x[token_idx]).squeeze(-1)
                loss = _pairwise_loss(scores, train_y[token_idx], rng)
                if candidate == "C9":
                    loss = loss * 0.25 + _listwise_loss(scores, train_y[token_idx], batch_size)
                loss.backward()
                optimizer.step()
        else:
            total = train_x.reshape(-1, train_x.shape[-1])
            truth = train_y.reshape(-1)
            for selection in torch.randperm(len(truth)).split(1024):
                optimizer.zero_grad()
                output = head(total[selection])
                if candidate == "C7":
                    targets = (truth[selection, None] > torch.arange(1, rmax)).float()
                    loss = F.binary_cross_entropy_with_logits(output, targets)
                else:
                    loss = F.huber_loss(output.squeeze(-1), truth[selection], delta=2.0)
                loss.backward()
                optimizer.step()
        prediction, _ = _scores(head, validation_x, candidate)
        fifo, pred, oracle = _ric(validation_y, prediction, batch_size)
        recovery = _recovery(fifo, pred, oracle)
        if recovery > best + 1e-9:
            best, best_epoch = recovery, epoch + 1
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    if best_state is None:
        raise AssertionError("no valid validation RIC selection")
    head.load_state_dict(best_state)
    return head, {"width": width, "learning_rate": lr, "weight_decay": 1e-4,
                  "best_epoch": best_epoch, "validation_ric_recovery": best,
                  "selection": "maximum validation RIC recovery, early-stop patience 5"}


def _metrics(candidate: str, labels: np.ndarray, scores: np.ndarray, batch_size: int,
             params: int, elapsed: float) -> dict:
    fifo, pred, oracle = _ric(labels, scores, batch_size)
    spear, kendall = _rank_metrics(labels, scores)
    start = time.perf_counter()
    for s in scores:
        np.lexsort((np.arange(len(s)), s))
    sort_s = time.perf_counter() - start
    return {
        "candidate": candidate, "fifo_ric": fifo, "predictor_ric": pred, "oracle_ric": oracle,
        "ric_reduction_vs_fifo": (fifo - pred) / fifo,
        "oracle_ric_recovery": _recovery(fifo, pred, oracle),
        "spearman": spear, "kendall_tau_b": kendall,
        "depth_mae": float(np.abs(scores - labels).mean()) if candidate not in {"C8", "C9"} else float("nan"),
        "exact_depth_accuracy": float((np.rint(scores) == labels).mean()) if candidate not in {"C8", "C9"} else float("nan"),
        "predictor_params": params, "prediction_cpu_s": elapsed, "sort_cpu_s": sort_s,
        "num_requests": labels.shape[1], "num_tokens": labels.shape[0], "batch_size": batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = json.loads((args.feature_root / "feature_manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("feature collection is not complete")
    for filename, expected in manifest["artifact_sha256"].items():
        if _sha256(args.feature_root / filename) != expected:
            raise RuntimeError(f"feature artifact checksum mismatch: {filename}")
    labels = np.asarray(np.load(args.feature_root / "depth_labels.npy", mmap_mode="r"), dtype=np.int16)
    scalar_audit = np.load(args.feature_root / "scalars.npy", mmap_mode="r")
    names = tuple(manifest["scalar_names"])
    previous_depth = scalar_audit[..., names.index("previous_depth")]
    trajectory_length = scalar_audit[..., names.index("previous_trajectory_length")]
    if not (np.all(previous_depth[0] == 0) and np.array_equal(previous_depth[1:], labels[:-1])
            and np.array_equal(trajectory_length, previous_depth)):
        raise RuntimeError("causal previous-depth/trajectory audit failed")
    if np.any(labels < 1) or np.any(labels > manifest["rmax"]):
        raise RuntimeError("depth labels outside model recurrent range")
    split = manifest["request_split"]
    all_ids = [rid for ids in split.values() for rid in ids]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != labels.shape[1]:
        raise RuntimeError("request split overlap or omission")
    if min(len(split[name]) for name in split) % args.batch_size:
        raise RuntimeError("each request split must be divisible by B")
    feature_sets = _feature_sets(args.feature_root, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for candidate in CANDIDATES:
        if candidate == "C0":
            scores = np.asarray(np.load(args.feature_root / "scalars.npy", mmap_mode="r")[:, :, 2],
                                dtype=np.float32)
            for split_name in ("validation", "test"):
                ids = split[split_name]
                rows.append({"split": split_name, **_metrics(candidate, labels[:, ids], scores[:, ids],
                                                              args.batch_size, 0, 0.0)})
            print(f"{candidate} test RIC recovery={rows[-1]['oracle_ric_recovery']:.4f}", flush=True)
            continue
        x, mean, std = _prepare(feature_sets[candidate], split["train"])
        variants = ((0, 1e-3), (64, 1e-3)) if candidate in {"C4", "C5"} else ((64, 1e-3), (256, 3e-4))
        selected = None
        for width, lr in variants:
            head, config = _train_one(candidate, x, labels, split["train"], split["validation"],
                                      args.batch_size, width, lr, args.epochs, args.seed,
                                      manifest["rmax"])
            if selected is None or config["validation_ric_recovery"] > selected[1]["validation_ric_recovery"]:
                selected = (head, config)
        head, config = selected
        config.update({"candidate": candidate, "features": candidate, "base_model_frozen": True,
                       "split_by_request": True, "loss": ("ordinal_cumulative_bce" if candidate == "C7" else
                       "weighted_pairwise_logistic" if candidate == "C8" else
                       "max_sensitive_adjacent_group_swap_plus_pairwise" if candidate == "C9" else "huber")})
        checkpoint = args.output_dir / f"{candidate}_checkpoint.pt"
        torch.save({"state_dict": head.state_dict(), "input_dim": x.shape[-1],
                    "output_dim": manifest["rmax"] - 1 if candidate == "C7" else 1,
                    "config": config, "feature_mean": mean, "feature_std": std}, checkpoint)
        (args.output_dir / f"{candidate}_config.json").write_text(json.dumps(config, indent=2) + "\n")
        params = sum(param.numel() for param in head.parameters())
        for split_name in ("validation", "test"):
            ids = split[split_name]
            scores, elapsed = _scores(head, x[:, ids], candidate)
            rows.append({"split": split_name, **_metrics(candidate, labels[:, ids], scores,
                                                          args.batch_size, params, elapsed)})
        print(f"{candidate} test RIC recovery={rows[-1]['oracle_ric_recovery']:.4f}", flush=True)
    with (args.output_dir / "predictor_tournament.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "execution_kind": "saved-data-analysis", "feature_root": str(args.feature_root),
        "feature_manifest_sha256": _sha256(args.feature_root / "feature_manifest.json"),
        "seed": args.seed, "batch_size": args.batch_size, "epochs_max": args.epochs,
        "source_sha": manifest["source_sha"], "no_model_forward": True,
        "candidate_count": len(CANDIDATES), "rows": len(rows),
        "C9_surrogate": "max-sensitive adjacent-B32 swap regret plus 0.25 weighted pairwise",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
