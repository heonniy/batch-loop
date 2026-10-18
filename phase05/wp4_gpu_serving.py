"""Phase-05 WP4: the WP3-frozen predictors served with on-GPU, host-sync-free features.

Same checkpoints, same feature definitions as phase03_predictor.physical._features,
but every feature buffer (history, previous-token uncertainty, previous trajectory,
projections), the head and the sort stay on the GPU. Only the survivor-compaction
syncs inside run_interval remain, and those are common to every policy.
See phase05/WP4_PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from phase05.wp1_confirmation import CONTRACT, cohort_errors, row_identity_errors
from phase05.wp3_predictor import (COHORT, EVAL_B, PROJECTION_DIM, T, _check_source, _dataset_texts, _encode,
                                   _load, _projection, _sha, physical_run)
from phase05.wp2_headroom import split_prompts

POLICIES = ("fifo", "history_last", "predictor_gpu", "predictor_host", "oracle_b1")
FUSED = {"C6", "C7", "C8", "C9"}
REPEATS = 3


class GpuPredictor:
    """Phase-03 fused feature view + frozen head, evaluated entirely on the GPU."""

    def __init__(self, selection_root: Path, key: str):
        import torch
        from phase03_predictor.tournament import Head
        selection = json.loads((selection_root / "selection.json").read_text())
        self.candidate = selection["winner"]
        if self.candidate not in FUSED:
            raise ValueError("GPU serving implements the fused C6-C9 view only")
        ckpt = torch.load(selection_root / f"{self.candidate}_checkpoint.pt", map_location="cpu", weights_only=False)
        self.head = Head(ckpt["input_dim"], ckpt["output_dim"], ckpt["config"]["width"])
        self.head.load_state_dict(ckpt["state_dict"])
        self.head = self.head.cuda().eval()
        self.mean = torch.as_tensor(np.asarray(ckpt["feature_mean"], dtype=np.float32), device="cuda")
        self.std = torch.as_tensor(np.asarray(ckpt["feature_std"], dtype=np.float32), device="cuda")
        self.key = key

    def score(self, x):
        import torch
        out = self.head((x - self.mean) / self.std)
        return 1 + torch.sigmoid(out).sum(dim=-1) if self.candidate == "C7" else out.squeeze(-1)


def uncertainty(logits, token_ids):
    """Same arithmetic as phase03_predictor.physical._uncertainty_batch, kept on device."""
    import torch
    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1)
    probs = torch.exp(values - log_z[:, None])
    top = torch.topk(values, 2, dim=-1).values
    surprisal = log_z - values.gather(1, token_ids.long().view(-1, 1)).squeeze(1)
    entropy = log_z - (probs * values).sum(dim=-1)
    top_probs = torch.exp(top - log_z[:, None])
    return torch.stack((surprisal, entropy, top_probs[:, 0], top_probs[:, 0] - top_probs[:, 1]), dim=-1)


def scalar_features(t: int, history, prompt_lengths, unc, prev_traj):
    """[N,38] in phase03 SCALARS order from on-device buffers (history: [T,N] long)."""
    import torch
    n = prompt_lengths.shape[0]
    zeros = torch.zeros(n, device=prompt_lengths.device)
    prev = history[t - 1].float() if t > 0 else zeros
    cols = [torch.full((n,), float(t), device=prompt_lengths.device), prompt_lengths + t, prev,
            torch.full((n,), float(t > 0), device=prompt_lengths.device)]
    for width in (2, 4, 8):
        seg = history[max(0, t - width):t].float()
        if seg.shape[0]:
            cols += [seg.mean(0), seg.std(0, unbiased=False), seg.min(0).values, seg.max(0).values]
        else:
            cols += [zeros] * 4
    cols.append(history[t - 1].float() - history[t - 2].float() if t > 1 else zeros)
    return torch.cat((torch.stack(cols, dim=1), unc, prev.unsqueeze(1), prev_traj), dim=1)


def gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, B, policy, predictor, projection,
            audit_predictor=None, ranker=None, capture_scores=False, lengths=None) -> dict:
    import torch
    from phase02_runtime.common import run_interval
    from phase05.wp2_headroom import _new_adapter
    n, steps = labels.shape[1], len(tokens)
    adapter = _new_adapter(key, model, tokenizer, n, list(range(n)), "benchmark")
    width = int(model.transformer.wte.weight.shape[1] if key == "huginn" else model.model.embed_tokens.weight.shape[1])
    want = policy in ("predictor_gpu", "ranker_gpu")
    need_traj = policy == "predictor_gpu" or (policy == "ranker_gpu" and ranker.uses_trajectory)
    lengths = np.full(n, steps) if lengths is None else np.asarray(lengths)
    active_np = [np.nonzero(lengths > t)[0] for t in range(steps)]  # requests still decoding (EOS-causal)
    captured = []
    audit = audit_predictor is not None
    audit_rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        adapter.prefill(encoded["input_ids"].cuda(), encoded["attention_mask"].cuda().bool())
        known = torch.as_tensor(tokens, dtype=torch.long, device="cuda")
        oracle = torch.as_tensor(labels, dtype=torch.long, device="cuda")
        plen = encoded["attention_mask"].sum(dim=1).float().cuda()
        history = torch.zeros((steps, n), dtype=torch.long, device="cuda")
        next_out = torch.empty((steps, n), dtype=torch.long, device="cuda")
        unc = uncertainty(adapter.prefill_logits[:, -1, :], known[0]) if want else None
        prev_traj = torch.zeros((n, 16), device="cuda")
        cur_traj = torch.zeros((n, 16), device="cuda")
        prev_hidden = torch.zeros((n, PROJECTION_DIM), device="cuda")
        prev_final = torch.zeros((n, width), device="cuda") if policy == "ranker_gpu" else None
        host_logits = adapter.prefill_logits[:, -1, :].clone() if audit else None
        host_traj = [[] for _ in range(n)]
        holder = {}

        def pre_hook(recurrence, request_ids, before, _adapter):
            holder["before"] = before

        def post_hook(recurrence, request_ids, after, _adapter):
            if key == "huginn":
                b, a = holder["before"].float(), after.float()
                value = ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)).mean(dim=-1)
            else:
                value = torch.sigmoid(model.model.early_exit_gate(after)).float().reshape(-1)
            cur_traj[request_ids, recurrence - 1] = value

        ids = torch.arange(n, device="cuda")
        orders, step_widths = [], []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in range(steps):
            if not len(active_np[t]):
                orders.append(ids[:0])
                step_widths.append([])
                continue
            act = torch.as_tensor(active_np[t], device="cuda")
            n_act = len(active_np[t])
            current = known[t].unsqueeze(1)
            adapter.begin_token()
            embedded_all = torch.empty((n, 1, width), dtype=torch.bfloat16, device="cuda")
            state_all = torch.empty_like(embedded_all)
            for first in range(0, n_act, B):  # common staged Prelude, FIFO chunks of B, every policy
                rows = act[first:first + B]
                adapter.backend.set_active(rows)
                e, s = adapter.prelude(current.index_select(0, rows))
                embedded_all.index_copy_(0, rows, e)
                state_all.index_copy_(0, rows, s)
            if policy == "fifo":
                order = act
            elif policy == "oracle_b1":
                order = act[torch.argsort(oracle[t][act], stable=True)]
            elif policy == "history_last":
                order = act[torch.argsort(history[t - 1][act], stable=True)] if t else act
            elif policy == "ranker_gpu":  # WP5 ranker: scalars + token encodings + full-dimensional states
                emb_raw = embedded_all if key == "ouro" else model.transformer.wte(current) * model.emb_scale
                fulls = {"emb_full": emb_raw.float().squeeze(1), "prevh_full": prev_final,
                         "pre_full": embedded_all.float().squeeze(1)}
                prev_tok = known[t - 1] if t else torch.full((n,), ranker.vocab, dtype=torch.long, device="cuda")
                scores = ranker.score(scalar_features(t, history, plen, unc, prev_traj), known[t], prev_tok, fulls)
                order = act[torch.argsort(scores[act], stable=True)]
                if capture_scores:
                    captured.append(scores.float())
            else:
                emb_raw = embedded_all if key == "ouro" else model.transformer.wte(current) * model.emb_scale
                emb = emb_raw.float().squeeze(1) @ projection
                rep = prev_hidden if key == "ouro" else embedded_all.float().squeeze(1) @ projection
                x = torch.cat((scalar_features(t, history, plen, unc, prev_traj), emb, rep), dim=1)
                scores = predictor.score(x)
                order = act[torch.argsort(scores[act], stable=True)]
                if audit:
                    audit_rows.append(_audit_step(t, key, adapter, current, host_logits, history, host_traj,
                                                  prev_hidden, embedded_all, plen, projection, audit_predictor,
                                                  scores, x, B))
            orders.append(order)
            widths_t = []
            step_sink: dict[int, list[float]] | None = {} if audit else None
            for first in range(0, n_act, B):
                rows = order[first:first + B]
                result = run_interval(adapter, embedded_all.index_select(0, rows), state_all.index_select(0, rows),
                                      rows, rows, exit_source="live", threshold=threshold, instrumentation="benchmark",
                                      debug_ids=False, trajectory_sink=step_sink,
                                      trajectory_kind=("huginn_convergence" if key == "huginn" else "ouro_gate")
                                      if audit else None,
                                      debug_pre_hook=pre_hook if need_traj else None,
                                      debug_hook=post_hook if need_traj else None)
                adapter.backend.set_active(rows)
                logits = adapter.coda_logits(result.final_state)[:, -1, :]
                next_out[t].index_copy_(0, rows, logits.argmax(dim=-1))
                history[t].index_copy_(0, rows, result.exit_depths)
                if want:
                    if t + 1 < steps:
                        unc.index_copy_(0, rows, uncertainty(logits, known[t + 1].index_select(0, rows)))
                    if key == "ouro":
                        prev_hidden.index_copy_(0, rows, result.final_state.float().squeeze(1) @ projection)
                    if prev_final is not None:
                        prev_final.index_copy_(0, rows, result.final_state.float().squeeze(1))
                    if audit:
                        host_logits.index_copy_(0, rows, logits)
                widths_t.append([e["active_batch_size"] for e in result.events])
            if n_act < n:  # EOS-finished requests: masked placeholder position (WP6 variable length)
                done = np.setdiff1d(np.arange(n), active_np[t])
                adapter.backend.retire_rows(torch.as_tensor(done, device="cuda"))
            adapter.commit_token()
            if need_traj:
                prev_traj, cur_traj = cur_traj, prev_traj
                cur_traj.zero_()
                if audit:
                    for r in range(n):
                        host_traj[r] = step_sink[r]
            step_widths.append(widths_t)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - start
        live = history.cpu().numpy()
        nexts = next_out.cpu().numpy()
        orders = [o.cpu().numpy() for o in orders]
        captured = [c.cpu().numpy() for c in captured]
    errors, executed = [], []
    ric = frozen = 0
    for t in range(steps):
        for k, first in enumerate(range(0, len(orders[t]), B)):
            members = orders[t][first:first + B].tolist()
            depths = [int(live[t, r]) for r in members]
            executed += [(t, r) for r in members]
            errors += cohort_errors(members, depths, step_widths[t][k], [], [], len(members))
            ric += max(depths)
            frozen += max(int(labels[t, r]) for r in members)
    expected = {(t, int(r)) for t in range(steps) for r in active_np[t]}
    if sorted(executed) != sorted(expected):
        errors.insert(0, f"active request-tokens executed != expected ({len(executed)} vs {len(expected)})")
    valid = np.array([[lengths[r] > t for r in range(n)] for t in range(steps)])
    nxt_valid = valid[:-1] & valid[1:]
    return {"decode_wall_s": decode_s, "feature_and_predict_s": 0.0, "sort_s": 0.0,
            "live_work": int(live[valid].sum()), "b1_work": int(labels[valid].sum()), "live_RIC": ric,
            "frozen_b1_RIC_same_grouping": frozen, "depth_flips": int((live != labels)[valid].sum()),
            "next_token_flips": int((nexts[:-1] != tokens[1:])[nxt_valid].sum()), "active_tokens": int(valid.sum()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "n_hard_errors": len(errors), "hard_check_errors": errors[:20], "live_depth": live, "audit": audit_rows,
            "scores": captured}


def _audit_step(t, key, adapter, current, host_logits, history, host_traj, prev_hidden, embedded_all, plen,
                projection, host_predictor, gpu_scores, gpu_x, B) -> dict:
    """Untimed equivalence check: WP3 host path on the same live state vs the GPU path."""
    from phase03_predictor.physical import _features
    hist = [history[:t, r].tolist() for r in range(history.shape[1])]
    scalar, emb, rep = _features(SimpleNamespace(model=key), adapter, current, host_logits, hist, host_traj,
                                 prev_hidden, embedded_all, plen.long().tolist(), projection, host_predictor.candidate)
    host_x = np.concatenate((scalar, emb, rep), axis=1)
    host_scores = host_predictor.score(scalar, emb, rep)
    g = gpu_scores.float().cpu().numpy()
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")
    rho = float(np.corrcoef(rank(g), rank(host_scores))[0, 1]) if g.std() and host_scores.std() else 1.0
    g_order, h_order = np.argsort(g, kind="stable"), np.lexsort((np.arange(len(g)), host_scores))
    same = np.mean([set(g_order[i:i + B]) == set(h_order[i:i + B]) for i in range(0, len(g), B)])
    return {"t": t, "max_abs_feature_diff": float(np.abs(gpu_x.cpu().numpy() - host_x).max()),
            "max_abs_score_diff": float(np.abs(g - host_scores).max()), "score_spearman": rho,
            "identical_cohort_fraction": float(same)}


def plan(repeats=REPEATS, cohorts=(0, 1)) -> list[dict]:
    import random
    out = [{"kind": "warmup", "cohort": 0, "B": 32, "policy": "fifo", "rep": -1}]
    for rep in range(repeats):
        for cohort in cohorts:
            order = list(EVAL_B)
            random.Random(f"wp4-plan:{rep}:{cohort}").shuffle(order)
            for b in order:
                k = rep % len(POLICIES)
                for policy in POLICIES[k:] + POLICIES[:k]:
                    out.append({"kind": "scored", "cohort": cohort, "B": b, "policy": policy, "rep": rep})
    return out


def main() -> None:
    import torch
    from phase03_predictor.physical import Predictor
    from phase05.wp1_confirmation import gpu_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("ouro", "huginn"))
    parser.add_argument("--role", required=True, choices=("primary", "replication", "gpu0", "gpu1", "gpu2", "gpu3"))
    parser.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    parser.add_argument("--feature-root", required=True, type=Path, help="WP3 lane feature dir (projection)")
    parser.add_argument("--confirm-root", required=True, type=Path, help="collected wp4_confirm B1 labels")
    parser.add_argument("--selection-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true", help="untimed GPU-vs-host equivalence gate")
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    key = args.model
    gpu = gpu_identity(f"{key}_{args.role}")
    source_sha = _check_source(args.smoke)
    selection = json.loads((args.selection_root / "selection.json").read_text())
    for name, digest in selection["checkpoint_sha256"].items():
        if _sha(args.selection_root / name) != digest:
            raise RuntimeError(f"frozen checkpoint changed: {name}")
    feature_manifest = json.loads((args.selection_root / "feature_manifest.json").read_text())
    host_predictor = Predictor(selection["winner"], args.selection_root, feature_manifest)
    gpu_predictor = GpuPredictor(args.selection_root, key)
    initial, model, tokenizer = _load(key)
    projection = _projection(key, model)
    if not np.array_equal(np.load(args.feature_root / "projection.npy"), projection.cpu().numpy()):
        raise RuntimeError("projection differs from the collected features")
    confirm = args.confirm_root / "wp4_confirm"
    confirm_manifest = json.loads((confirm / "feature_manifest.json").read_text())
    labels_all = np.load(confirm / "depth_labels.npy").astype(np.int64)
    tokens_all = np.load(confirm / "token_ids.npy").astype(np.int64)
    prompts_all = split_prompts(_dataset_texts(args.dataset), "wp4_confirm")[:len(confirm_manifest["prompt_sha256"])]
    if [hashlib.sha256(p.encode()).hexdigest() for p in prompts_all] != confirm_manifest["prompt_sha256"]:
        raise RuntimeError("confirmation prompts differ from collected labels")
    cohort_size = min(COHORT, len(prompts_all))
    cohorts = tuple(range(len(prompts_all) // cohort_size))[:2]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    steps = args.tokens or T
    threshold = CONTRACT[key]["threshold"]

    def cohort_inputs(c):
        sl = slice(c * cohort_size, (c + 1) * cohort_size)
        return _encode(tokenizer, prompts_all[sl]), labels_all[:steps, sl], tokens_all[:steps, sl]

    base_manifest = {"preregistration": "phase05/WP4_PREREGISTRATION.md", "source_sha": source_sha, "argv": sys.argv,
                     "gpu": gpu, "cpu_affinity": sorted(os.sched_getaffinity(0)), "model": key,
                     "dataset": args.dataset, "predictor": selection["winner"], "selection": selection,
                     "confirm_manifest_sha256": _sha(confirm / "feature_manifest.json"), "torch": torch.__version__}
    if args.audit_only:
        encoded, labels, tokens = cohort_inputs(0)
        result = gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, 8, "predictor_gpu",
                         gpu_predictor, projection, audit_predictor=host_predictor)
        audit = result["audit"]
        summary = {"execution_kind": "UNTIMED equivalence audit", **base_manifest,
                   "B": 8, "steps": len(audit), "n_hard_errors": result["n_hard_errors"],
                   "min_score_spearman": min(a["score_spearman"] for a in audit),
                   "mean_score_spearman": statistics.mean(a["score_spearman"] for a in audit),
                   "mean_identical_cohort_fraction": statistics.mean(a["identical_cohort_fraction"] for a in audit),
                   "max_abs_feature_diff": max(a["max_abs_feature_diff"] for a in audit),
                   "max_abs_score_diff": max(a["max_abs_score_diff"] for a in audit), "per_step": audit}
        summary["status"] = ("PASS" if summary["mean_score_spearman"] >= 0.999 and summary["min_score_spearman"] >= 0.99
                             and summary["mean_identical_cohort_fraction"] >= 0.95 and not result["n_hard_errors"]
                             else "FAIL")
        (out / "equivalence_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("status", "mean_score_spearman", "min_score_spearman",
                                                  "mean_identical_cohort_fraction", "max_abs_feature_diff",
                                                  "max_abs_score_diff")}), flush=True)
        sys.exit(0 if summary["status"] == "PASS" else 1)

    (out / "run_manifest.json").write_text(json.dumps({
        "execution_kind": "UNSCORED_SMOKE" if args.smoke else "fresh-cohort physical no-refill, live BF16 exits",
        **base_manifest, "cohorts": [[c * cohort_size, (c + 1) * cohort_size] for c in cohorts],
        "policies": POLICIES, "B": EVAL_B, "repeats": REPEATS, "T": steps}, indent=2) + "\n")
    fields = ["run_id", "kind", "cohort", "B", "policy", "rep", "status", "decode_wall_s", "feature_and_predict_s",
              "sort_s", "live_work", "b1_work", "live_RIC", "frozen_b1_RIC_same_grouping", "depth_flips",
              "next_token_flips", "peak_allocated_gib", "n_hard_errors", "error"]
    with (out / "runs.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    live_rows, hard = [], 0
    cells = plan(1 if args.smoke else REPEATS, cohorts)
    if args.smoke:
        cells = [c for c in cells if c["B"] in (32, 64)]
    for run_id, cell in enumerate(cells):
        encoded, labels, tokens = cohort_inputs(cell["cohort"])
        row = {"run_id": run_id, **cell, "status": "OK", "error": ""}
        try:
            if cell["policy"] == "predictor_host":  # the WP3 implementation, unchanged
                result = physical_run(key, model, tokenizer, encoded, labels, tokens,
                                      encoded["attention_mask"].sum(dim=1).tolist(), threshold, cell["B"],
                                      "predictor", host_predictor, projection)
            else:
                result = gpu_run(key, model, tokenizer, encoded, labels, tokens, threshold, cell["B"],
                                 cell["policy"], gpu_predictor, projection)
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
    np.savez_compressed(out / "live_depths.npz", **{f"run_{i}": d for i, d in live_rows})
    print(json.dumps({"event": "lane_done", "hard_failures": hard}), flush=True)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
