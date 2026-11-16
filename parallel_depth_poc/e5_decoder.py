"""E5: physically integrated depth-parallel decoder vs exact fixed-depth AR (GPU, opt-in, B=1).

PRE-REGISTERED RUN RULE (fixed before reading any E2/E3 result): a (model, D) cell is run only if
its bf16_b16_L256 OR bf16_b1_L256 projection upper bound (summarize_gpu.py, best Jacobi init, at
mean k_stable) exceeds 1.15x. Otherwise E5 is recorded `not_run (gate)` for that cell.

Arms (each prompt: fresh prefill per arm, arm order shuffled per prompt, same tokenizer/prompt/
depth/greedy/stop strings/FULL KV):
  AR          exact fixed depth: D serial width-1 core calls + coda per token.
  jac_exact   real h1 + D-1 Jacobi rounds (width D each) + coda: bit-identical to AR by construction
              (measures pure overhead of the parallel formulation).
  jac_g{tau}  real h1, then Jacobi rounds; stop at the first round k with residual < tau AND an
              unchanged greedy final-depth token vs round k-1 (coda per round, counted); if never,
              run to D-1 rounds (exact fallback). Commits the CANDIDATE states/KV of the stop round
              -> approximate when it stops early; quality measured against AR.
  trunc{D'}   exact AR at a shallower fixed depth D' (independent loop-skipping quality baseline).
Predictor init for Jacobi: `--init` (linear | prev_shift | constant).
Metrics: wall per arm (CUDA-synchronised per token), tokens/s, TPOT p50/p90, core calls/rows,
coda calls, peak HBM, token agreement vs AR (full-sequence exact, first mismatch), GSM8K accuracy
(strict + flexible) on the generation truncated at the stop strings. Writes <out>.jsonl + meta.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from . import wf_engine as W
from .depth_diag import env_info


def stops_hit(tok, gen, stops, window=24):
    tail = tok.decode(gen[-window:], skip_special_tokens=False)
    return any(s in tail for s in stops)


@torch.no_grad()
def decode(eng: W.DepthEngine, tok, ids, arm, a, eos, stops):
    D = eng.D
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    last = eng.prefill(ids)
    cur = last.argmax(-1).tolist()
    gen = [cur[0]]
    tpot, rounds = [], []
    prev_traj = None
    while len(gen) < a.max_new and gen[-1] != eos and not stops_hit(tok, gen, stops):
        t0 = time.perf_counter()
        h0 = eng.prelude(cur)
        if arm == "AR" or arm.startswith("trunc"):
            h = h0
            for r in range(1, D + 1):
                h = eng.step_many(h.unsqueeze(0), torch.tensor([r]))[0]
            lg = eng.logits(h)
            k_used = D
        else:
            tau = 0.0 if arm == "jac_exact" else float(arm[5:])
            h1 = eng.step_many(h0.unsqueeze(0), torch.tensor([1]))[0]
            if a.init == "prev_shift" and prev_traj is not None:
                z = torch.cat([h0.unsqueeze(0), prev_traj[1:] + (h1 - prev_traj[1]).unsqueeze(0)])
            elif a.init == "constant":
                z = torch.stack([h0] + [h1] * D)
            else:
                z = torch.stack([h0] + [h1 + (r - 1) * (h1 - h0) for r in range(1, D + 1)])
            z[1] = h1
            ds = torch.arange(1, D + 1)
            prev_arg = None
            k_used = D - 1
            for k in range(1, D):
                new = torch.cat([h0.unsqueeze(0), eng.step_many(z[:-1], ds)])
                resid = ((new[1:].float() - z[1:].float()).norm(dim=-1)
                         / new[1:].float().norm(dim=-1).clamp_min(1e-8)).amax().item() if tau > 0 else 1.0
                z = new
                if tau > 0:
                    lg = eng.logits(z[D])
                    arg = int(lg.argmax(-1)[0])
                    if resid < tau and arg == prev_arg:
                        k_used = k
                        break
                    prev_arg = arg
            if tau == 0:                                        # gated arms already hold the last round's logits
                lg = eng.logits(z[D])
            prev_traj = z
            k_used = 1 + k_used                                   # + the real h1 call
        eng.pos += 1
        cur = lg.argmax(-1).tolist()
        gen.append(cur[0])
        torch.cuda.synchronize()
        tpot.append((time.perf_counter() - t0) * 1e3)
        rounds.append(k_used)
    torch.cuda.synchronize()
    return gen, time.perf_counter() - t_start, tpot, rounds


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--arms", required=True, help="e.g. AR,jac_exact,jac_g0.01,trunc16")
    p.add_argument("--init", default="linear", choices=["linear", "prev_shift", "constant"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    from experiments.exp3_accuracy import tasks as ds
    lm, tok, batch_cls, kind = W.load_model(a.model, torch.bfloat16)
    eos = lm.config.eos_token_id if isinstance(lm.config.eos_token_id, int) else None
    rows = ds.load_dataset_prompts("gsm8k", 3, a.offset + a.n)[a.offset:]
    arms = a.arms.split(",")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(a), **env_info(a)), open(str(out) + ".meta.json", "w"), indent=1)
    fh = open(str(out) + ".jsonl", "w")
    # warm-up (not recorded)
    w_ids = tok(rows[0]["prompt"], return_tensors="pt").input_ids.cuda()
    for arm in arms:
        D = int(arm[5:]) if arm.startswith("trunc") else a.depth
        eng = W.DepthEngine(lm, batch_cls, kind, D, 1, w_ids.shape[1] + 40)
        decode(eng, tok, w_ids, arm, argparse.Namespace(**{**vars(a), "max_new": 16}), eos, [])
    for i, r in enumerate(rows):
        ids = tok(r["prompt"], return_tensors="pt").input_ids.cuda()
        order = list(arms)
        random.Random(f"{a.seed}:{i}").shuffle(order)
        recs = {}
        for arm in order:
            D = int(arm[5:]) if arm.startswith("trunc") else a.depth
            eng = W.DepthEngine(lm, batch_cls, kind, D, 1, ids.shape[1] + a.max_new + 8)
            torch.cuda.reset_peak_memory_stats()
            gen, wall, tpot, rounds = decode(eng, tok, ids, arm, a, eos, r["stop"])
            txt = tok.decode(gen, skip_special_tokens=True)
            ok, _, extra = ds.score("gsm8k", txt, r["gold"], r["stop"], r.get("meta"))
            recs[arm] = dict(prompt_idx=a.offset + i, arm=arm, order=order.index(arm), tokens=gen, n_tokens=len(gen),
                             wall_s=wall, tok_per_s=len(gen) / wall,
                             tpot_p50_ms=sorted(tpot)[len(tpot) // 2] if tpot else None,
                             tpot_p90_ms=sorted(tpot)[int(0.9 * (len(tpot) - 1))] if tpot else None,
                             rounds_mean=(sum(rounds) / len(rounds)) if rounds else None,
                             core_calls=eng.n.core_calls, core_rows=eng.n.core_rows, coda_calls=eng.n.coda_calls,
                             hbm_peak_gb=torch.cuda.max_memory_allocated() / 2**30,
                             correct_strict=bool(ok), correct_flex=bool(extra.get("correct_flex")))
        ar = recs.get("AR")
        for arm, rec in recs.items():
            if ar:
                n = min(len(ar["tokens"]), len(rec["tokens"]))
                fm = next((j for j in range(n) if ar["tokens"][j] != rec["tokens"][j]), None)
                rec["exact_to_AR"] = rec["tokens"] == ar["tokens"]
                rec["first_mismatch_vs_AR"] = fm if fm is not None else (None if rec["exact_to_AR"] else n)
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"[e5] {i} " + " ".join(f"{k}:{v['tok_per_s']:.1f}" for k, v in recs.items()), flush=True)
    fh.close()


if __name__ == "__main__":
    main()
