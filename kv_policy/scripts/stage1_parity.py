"""Stage 1 on real checkpoints: native-HF parity of the FULL engine and policy semantics checks.

(a) logits parity, engine FULL (parallel and token-by-token prompt) vs native no-cache forward,
    fp32 and bf16, on real benchmark prompts + gold continuations;
(b) LAST/AVERAGE engine vs independent native-forward dictionary-cache reference (fp32);
(c) greedy generation, engine FULL (B=1 and left-padded B=4) vs native no-cache greedy (bf16);
(d) batch-shape sensitivity of the engine (bf16, B=4 vs B=1 logprobs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kvp.jobs import gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.reference import reference_logits  # noqa: E402

N_CONT = 48


def cmp(eng: torch.Tensor, ref: torch.Tensor, P: int | None = None) -> dict:
    d = (eng - ref).abs()
    lp_e, lp_r = torch.log_softmax(eng, -1), torch.log_softmax(ref, -1)
    kl = (lp_r.exp() * (lp_r - lp_e)).sum(-1)
    out = {"max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
           "top1_agreement": (eng.argmax(-1) == ref.argmax(-1)).float().mean().item(),
           "max_kl": kl.max().item(), "n_positions": eng.shape[0]}
    if P is not None:
        out["max_abs_logit_diff_prompt"] = d[:P].max().item()
        out["max_abs_logit_diff_cont"] = d[P:].max().item()
    return out


@torch.inference_mode()
def native_logits(spec, model, run, ids, uid):
    x = torch.tensor([ids], device="cuda")
    if spec.family == "ouro":
        return model(x, exit_at_step=spec.native_depth - 1, use_cache=False).logits[0].float().cpu()
    noise = run.bb.example_noise(uid, len(ids)).to("cuda", next(model.parameters()).dtype)[None]
    return model(x, input_states=noise, num_steps=spec.native_depth).logits[0].float().cpu()


@torch.inference_mode()
def native_greedy(spec, model, run, prompt, uid, n):
    ids = list(prompt)
    for _ in range(n):
        ids.append(int(native_logits(spec, model, run, ids, uid)[-1].argmax()))
    return ids[len(prompt):]


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"model": args.model, "gpu": None, "sections": {}}
    t_start = time.time()

    # ---------------- fp32: (a) + (b)
    spec, tok, model = load(args.model, dtype=torch.float32)
    run = make_runner(spec, tok, model)
    report["gpu"] = gpu_identity()
    cases = []
    for ds in ("gsm8k", "math500"):
        items, _ = prepare_items(args.model, ds, tok)
        for it in items[:2]:
            cases.append((ds, it["uid"], it["prompt_ids"], it["target_ids"][:N_CONT]))
    sec = []
    for ds, uid, p, c in cases:
        nat = native_logits(spec, model, run, p + c, uid)
        for mode in ("decode", "strict"):
            eng = run.teacher_forced_logits(p, c, uid, "full", mode)
            sec.append({"dataset": ds, "uid": uid, "mode": mode, "P": len(p), **cmp(eng, nat, len(p))})
    report["sections"]["a_full_vs_native_fp32"] = sec

    short_p = tok.encode("Q: A box has 12 red and 7 blue marbles. Half of the red ones are removed. "
                         "How many marbles are left?\nA:", add_special_tokens=False)
    short_c = tok.encode(" Half of 12 is 6, so 6 red marbles remain. 6 + 7 = 13. The answer is 13.",
                         add_special_tokens=False)
    sec = []
    for (label, p, c) in [("short_prompt", short_p, short_c), ("benchmark_prompt", cases[0][2], cases[0][3][:16])]:
        for pol in ("full", "last", "average"):
            for mode in ("decode", "strict"):
                if label == "benchmark_prompt" and mode == "strict":
                    continue  # dictionary reference is O(T^2) python; strict covered by short prompt
                noise = run.bb.example_noise("ref", len(p) + len(c)) if spec.family == "huginn" else None
                ref = reference_logits(model, spec.family, run.bb.R, p, c, pol, mode, noise)
                eng = run.teacher_forced_logits(p, c, "ref", pol, mode)
                sec.append({"case": label, "policy": pol, "mode": mode, "P": len(p), **cmp(eng, ref, len(p))})
    report["sections"]["b_policy_engine_vs_reference_fp32"] = sec
    del model, run
    torch.cuda.empty_cache()

    # ---------------- bf16: (a) + (c) + (d)
    spec, tok, model = load(args.model, dtype=torch.bfloat16)
    run = make_runner(spec, tok, model)
    sec = []
    for ds, uid, p, c in cases:
        nat = native_logits(spec, model, run, p + c, uid)
        for mode in ("decode", "strict"):
            eng = run.teacher_forced_logits(p, c, uid, "full", mode)
            sec.append({"dataset": ds, "uid": uid, "mode": mode, "P": len(p), **cmp(eng, nat, len(p))})
    report["sections"]["a_full_vs_native_bf16"] = sec

    prompts = [c[2] for c in cases]
    uids = [c[1] for c in cases]
    nat_gen = [native_greedy(spec, model, run, p, u, N_CONT) for p, u in zip(prompts, uids)]
    eos = set()
    never = lambda b, ids: False  # noqa: E731 - fixed-length comparison, no stopping
    b1 = [run.generate([p], [u], "full", "decode", N_CONT, eos, never).gen_ids[0] for p, u in zip(prompts, uids)]
    b4 = run.generate(prompts, uids, "full", "decode", N_CONT, eos, never).gen_ids
    sec = []
    for i, (u, n, x, y) in enumerate(zip(uids, nat_gen, b1, b4)):
        sec.append({"uid": u, "native_vs_engine_b1_first_divergence": first_div(n, x),
                    "native_vs_engine_b4_first_divergence": first_div(n, y),
                    "engine_b1_vs_b4_first_divergence": first_div(x, y),
                    "native_tokens": n, "engine_b1_tokens": x, "engine_b4_tokens": y,
                    "native_text": tok.decode(n), "engine_b1_text": tok.decode(x)})
    report["sections"]["c_greedy_native_vs_engine_bf16"] = sec

    sec = []
    targets = [c[3] for c in cases]
    for pol in ("full", "last", "average"):
        bat = run.score(prompts, targets, uids, pol, "decode").target_logprobs
        for i in range(len(prompts)):
            one = run.score([prompts[i]], [targets[i]], [uids[i]], pol, "decode").target_logprobs[0]
            diff = max(abs(a - b) for a, b in zip(one, bat[i]))
            sec.append({"policy": pol, "uid": uids[i], "max_abs_logprob_diff_b4_vs_b1": diff,
                        "sum_logprob_b1": sum(one), "sum_logprob_b4": sum(bat[i])})
    report["sections"]["d_batch_shape_sensitivity_bf16"] = sec
    report["wall_s"] = time.time() - t_start
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
