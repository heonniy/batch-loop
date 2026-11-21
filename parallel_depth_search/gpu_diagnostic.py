"""Opt-in real-H200 teacher-forced depth prediction/correction diagnostic.

Uses the pinned PoC WaveFront batch engine with FULL depth-indexed KV. This is
NOT a runnable production decoder. Exact AR future states are computed only for
diagnosis/reference and NEVER passed to the predictor beyond the allowed
warmup. An exact commit repairs current-token KV before advancing. Timed
solver-only time must not be represented as actual serving speedup: the
diagnostic also pays for the reference, cache commit, and repeated coda.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from parallel_depth_poc import wf_engine as W
from parallel_depth_poc.depth_diag import cohort_texts, kl_top1, rel
from .numerics import PREDICTORS, CORRECTORS, ALLOCATIONS, predict, CorrectionStepper


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNRESOLVED"


def sync():
    torch.cuda.synchronize()


def elapsed_ms(fn):
    sync()
    t0 = time.perf_counter()
    result = fn()
    sync()
    return result, 1000 * (time.perf_counter() - t0)


def validate(a):
    if not a.allow_gpu:
        raise SystemExit("GPU execution requires --allow-gpu; no implicit background jobs")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not detected; emit NOT_RUN:no_device_access")
    dev = torch.cuda.get_device_name()
    if "H200" not in dev and not a.allow_other_gpu:
        raise SystemExit(f"requires explicitly allocated H200; detected {dev}. "
                         "Override with --allow-other-gpu only for labeled device comparison")
    if a.batch < 1 or a.prompt_len < 1 or a.decode_steps < 1:
        raise ValueError("batch, prompt-len and decode-steps must be positive")
    if a.model.startswith("ouro") and a.depth not in (2, 3, 4):
        raise ValueError("Ouro study capped at native depth 4; no invented deep Ouro")
    if a.model.startswith("huginn") and a.depth not in (8, 16, 24, 32):
        raise ValueError("Huginn supported study depths 8/16/24/32, confirm native revision")
    if not 1 <= a.warmup <= a.depth:
        raise ValueError("warmup depth must be 1..D")
    if a.predictor == "geometric" and a.warmup < 2:
        raise ValueError("geometric needs >=2 exact anchors")
    if a.allocation == "residual_topk" and a.rounds < 2:
        raise ValueError("residual_topk requires at least a full initial round")
    if a.rounds < 0:
        raise ValueError("rounds must be >=0")
    if a.corrector.startswith("newton") and a.allocation != "full":
        # Sparse triangular Newton needs boundary handling per skipped depth.
        # This diagnostic can calculate it but do not silently call it exact.
        pass
    return dev


@torch.inference_mode()
def run(a):
    dev = validate(a)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if a.predictor == "prev_shift" and a.offset < 0:
        raise ValueError("invalid prompt offset")
    lm, tok, batch_cls, kind = W.load_model(a.model, torch.bfloat16 if a.dtype == "bfloat16" else torch.float32)
    texts = cohort_texts(a.cohort, max(a.offset + 4 * a.batch, a.batch))
    selected = texts[a.offset:]
    ids = W.make_cohort(tok, selected, a.prompt_len, a.batch)
    D, B = a.depth, a.batch
    eng = W.DepthEngine(lm, batch_cls, kind, D, B, a.prompt_len + a.decode_steps + 8)
    meta = {
        "status": "GPU_DIAGNOSTIC_ONLY", "model": a.model, "model_checkpoint": W.MODEL_IDS[a.model],
        "depth": D, "B": B, "L": a.prompt_len, "dtype": a.dtype, "gpu_name": dev,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "batch_loop_git_sha": git_sha(Path(__file__).resolve().parents[1]),
        "wavefront_git_sha": git_sha(Path(W.WF_ROOT)),
        "prompt_sha256": sha256_text("\n".join(selected[:B])),
        "cohort": a.cohort, "offset": a.offset, "seed": a.seed,
        "predictor": a.predictor, "corrector": a.corrector,
        "warmup": a.warmup, "rounds": a.rounds, "allocation": a.allocation,
        "width": a.width, "scale": a.scale, "omega": a.omega, "history": a.history,
        "kv_policy": "FULL per-depth, exact serial commit after diagnostics",
        "no_future_state_in_predictor": True,
        "not_end_to_end_serving": True,
        "not_training": True,
        "note": "Exact reference and serial commit costs are separately measured, not usable at inference for free.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    next_logits, prefill_ms = elapsed_ms(lambda: eng.prefill(ids))
    cur = next_logits.argmax(-1).tolist()
    prior_exact = None
    results = []
    mismatch_exact_commit = 0
    max_logit_error = 0.0
    for t in range(a.decode_steps):
        h0, prelude_ms = elapsed_ms(lambda: eng.prelude(cur))
        # One *measured* teacher-forced ground truth, never fed into predictor
        # beyond independent exact m-step warmup computed below.
        ref, serial_ms = elapsed_ms(lambda: eng.serial(h0))
        truth, head_ms = elapsed_ms(lambda: eng.logits(ref[D]))
        lref = truth.detach().clone()
        def warm_exact():
            vals = [h0]
            h = h0
            for r in range(1, a.warmup + 1):
                h = eng.step_many(h.unsqueeze(0), torch.tensor([r], device=h.device))[0]
                vals.append(h)
            return torch.stack(vals)
        prefix, warmup_ms = elapsed_ms(warm_exact)
        if not torch.equal(prefix, ref[:a.warmup + 1]):
            raise RuntimeError("warmup prefix differs from serial reference; BF16 path not reproducible")
        if a.predictor == "prev_shift" and prior_exact is None:
            # Do not fabricate earlier token states; consume first token as a
            # warmup for previous-token predictor and mark the skipped trial.
            _, commit_ms = elapsed_ms(lambda: eng.commit(h0))
            prior_exact = ref
            cur = lref.argmax(-1).tolist()
            results.append({"t": t, "status": "skip_no_previous_token", "commit_ms": commit_ms})
            continue
        z, init_ms = elapsed_ms(lambda: predict(prefix, D, a.predictor, prior_exact))
        jvp = None
        if a.jvp_probe:
            # C3 milestone: is A_r = alpha*I a usable Jacobian surrogate? Perturb the EXACT input of
            # every depth by the predictor's actual error direction delta_r = z[r-1]-ref[r-1] and
            # compare the true response F_r(ref+delta)-F_r(ref) with alpha*delta (one physical call;
            # its candidate KV is overwritten by the exact commit below).
            dsd = list(range(a.warmup + 1, D + 1))
            dl = torch.stack([z[d - 1] - ref[d - 1] for d in dsd])
            ok = dl.float().flatten(1).norm(dim=-1) > 0
            if bool(ok.any()):
                ids_t = torch.tensor(dsd, device=z.device)
                out_p = eng.step_many(torch.stack([ref[d - 1] for d in dsd]) + dl, ids_t)
                resp = (out_p.float() - torch.stack([ref[d] for d in dsd]).float())
                dlf = dl.float()
                nrm = lambda x: x.norm(dim=-1)
                alpha_opt = (resp * dlf).sum(-1) / dlf.square().sum(-1).clamp_min(1e-12)   # [R,B]
                err_id = nrm(resp - dlf) / nrm(resp).clamp_min(1e-12)
                err_sc = nrm(resp - a.scale * dlf) / nrm(resp).clamp_min(1e-12)
                err_opt = nrm(resp - alpha_opt[..., None] * dlf) / nrm(resp).clamp_min(1e-12)
                gain = nrm(resp) / nrm(dlf).clamp_min(1e-12)
                q = lambda x: [float(v) for v in torch.quantile(x.flatten().float(), torch.tensor([.1, .5, .9], device=x.device))]
                jvp = dict(depths=dsd, gain_q10_50_90=q(gain), alpha_opt_q10_50_90=q(alpha_opt),
                           rel_err_identity_q=q(err_id), rel_err_scaled_q=q(err_sc), rel_err_best_scalar_q=q(err_opt),
                           cos_resp_delta_q=q((resp * dlf).sum(-1) / (nrm(resp) * nrm(dlf)).clamp_min(1e-12)),
                           per_depth_gain_median=[float(x) for x in gain.median(-1).values],
                           per_depth_alpha_median=[float(x) for x in alpha_opt.median(-1).values])
        stepper = CorrectionStepper(D, a.warmup, a.corrector, eng.step_many,
                                    omega=a.omega, scale=a.scale, history=a.history,
                                    damping=a.damping, ridge=a.ridge)
        rounds = []
        activity = None
        for k in range(a.rounds):
            # First residual_topk iteration is full: otherwise no admissible
            # observed residual exists to rank candidate depth rows.
            allocation = "full" if a.allocation == "residual_topk" and k == 0 else a.allocation
            (new, updates), ms = elapsed_ms(
                lambda: stepper.step(z, policy=allocation, width=a.width, activity=activity))
            z = new
            activity = updates
            logits, coda_ms = elapsed_ms(lambda: eng.logits(z[D]))
            kl, agree = kl_top1(lref, logits)
            rounds.append({
                "k": k + 1,
                "solver_wall_ms": ms,
                "coda_wall_ms": coda_ms,
                "top1_agreement": float(agree.float().mean()),
                "kl_mean": float(kl.mean()),
                "hidden_rel_l2_mean": float(rel(z[D], ref[D]).mean()),
                "max_online_update": float(updates.max()),
                "core_calls_cumulative": stepper.calls,
                "core_rows_cumulative": stepper.eval_rows,
                "not_cache_safe_as_standalone_decoder": True,
            })
        final, final_head_ms = elapsed_ms(lambda: eng.logits(z[D]))
        kl, agree = kl_top1(lref, final)
        # Exact-commit is the *only* cache-safe continuation in this diagnostic.
        (committed, lg_committed), commit_ms = elapsed_ms(lambda: eng.commit(h0))
        same_state = bool(torch.equal(ref, committed))
        same_logit = bool(torch.equal(lref, lg_committed))
        if not same_state or not same_logit:
            mismatch_exact_commit += 1
            max_logit_error = max(max_logit_error, float((lref - lg_committed).abs().max()))
        results.append({
            "t": t, "reference_serial_ms": serial_ms, "reference_head_ms": head_ms,
            "prelude_ms": prelude_ms, "warmup_ms": warmup_ms, "predictor_ms": init_ms,
            "final_head_ms": final_head_ms, "commit_ms": commit_ms,
            "solver_only_ms": warmup_ms + init_ms + sum(x["solver_wall_ms"] + x["coda_wall_ms"] for x in rounds) + final_head_ms,
            "actual_diagnostic_ms_excluding_prefill": prelude_ms + serial_ms + head_ms + warmup_ms + init_ms +
                   sum(x["solver_wall_ms"] + x["coda_wall_ms"] for x in rounds) + final_head_ms + commit_ms,
            "top1_agreement": float(agree.float().mean()), "kl_mean": float(kl.mean()),
            "final_hidden_rel_l2_mean": float(rel(z[D], ref[D]).mean()),
            "exact_commit_trajectory_bitwise": same_state, "exact_commit_logits_bitwise": same_logit,
            "total_solver_fine_calls": stepper.calls, "total_solver_fine_rows": stepper.eval_rows,
            "rounds": rounds, "jvp_probe": jvp
        })
        prior_exact = committed
        # Advance with original exact AR, never with approximate candidate.
        cur = lg_committed.argmax(-1).tolist()
        if t % 8 == 0:
            print(f"[gpu_diagnostic] t={t} B={B} D={D} method={a.corrector} rounds={len(rounds)}", flush=True)
    summary = {
        "status": "DIAGNOSTIC_RECORDED", "not_real_speedup": True,
        "n_prompts": B, "n_token_steps": a.decode_steps, "n_valid_steps": sum("rounds" in x for x in results),
        "prefill_wall_ms": prefill_ms, "exact_commit_mismatch_count": mismatch_exact_commit,
        "max_commit_logit_error": max_logit_error,
        "hbm_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "hbm_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "avg_top1": sum(x.get("top1_agreement", 0) for x in results) / max(1, sum("rounds" in x for x in results)),
        "records": results, "metadata": meta,
        "warning": "NOT a full no-refill B>1 E2E benchmark, no original-AR cache-safe speed claim."
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(out.with_suffix(".json")), "status": summary["status"],
                      "valid_steps": summary["n_valid_steps"], "avg_top1": summary["avg_top1"],
                      "exact_commit_mismatches": mismatch_exact_commit}))
    if mismatch_exact_commit:
        raise RuntimeError("exact-commit parity failed; no inference claim allowed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=W.MODEL_IDS.keys())
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--cohort", default="gsm8k3", choices=["gsm8k3", "long"])
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--predictor", choices=PREDICTORS, default="constant")
    p.add_argument("--corrector", choices=CORRECTORS, default="jacobi")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--allocation", choices=ALLOCATIONS, default="full")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--omega", type=float, default=0.75)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--damping", type=float, default=0.8)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--jvp-probe", action="store_true", help="C3: directional response vs scalar-Jacobian surrogate")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--out", required=True, help="output stem, writes .json and .meta.json")
    p.add_argument("--allow-gpu", action="store_true")
    p.add_argument("--allow-other-gpu", action="store_true", help="label non-H200 comparison explicitly")
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
