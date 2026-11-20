"""Free-running, no-refill, batched (B>=1) end-to-end decoder: depth-parallel arms vs fixed-depth AR.

Every arm decodes the SAME cohort of B real task prompts (variable lengths, no truncation, no
padding; per-request prefill into one FULL depth-indexed KV bank; finished requests leave the
union, no refill), greedy, same stop strings/EOS/max_new, fresh engine per arm, arm order
shuffled per cohort. Timing: CUDA-synchronised wall of prefill (TTFT) and of the whole decode
loop (all solver, coda, gate, commit, Python and stop-check overhead included).

Arms (comma list):
  AR                 fixed depth D: D serial width-n core calls + coda per token (exact baseline).
  trunc{d}           fixed depth d<D (quality-matched truncation baseline).
  jac_exact          exact h1 + D-1 Jacobi sweeps (width D x n), coda once. Exact in real
                     arithmetic; BF16 row shape differs from AR (fidelity measured, not assumed).
  jac_g{tau}         exact h1, then corrector sweeps; stop at first round where every active row
                     has residual < tau AND an unchanged greedy token vs the previous round (coda per
                     round, counted); fallback D-1 sweeps. Candidate KV carried (APPROXIMATE).
  jac_g{tau}_f       same, but rows that pass the gate are frozen (removed from later sweeps of
                     this token): per-example adaptive width (allocation lane A7).
  jac_g{tau}_rep     gated token, then exact serial repair of the depth KV (rows > k+1 recomputed
                     serially): cache-exact, token from the candidate (V2 lane).
Options: --init {linear,constant,prev_shift}, --corrector {jacobi,damped,anderson,newton_scaled,
newton_identity} (non-Jacobi correctors via numerics.CorrectionStepper, full allocation).
Writes <out>.jsonl (one row per (prompt, arm)) and <out>.meta.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from parallel_depth_poc import wf_engine as W
from parallel_depth_poc.depth_diag import env_info
from .numerics import CorrectionStepper
from .vengine import VarEngine, load


def sync():
    torch.cuda.synchronize()


def rel_rows(new, old):                       # [R,n,H] -> [n]  max over depths
    return ((new.float() - old.float()).norm(dim=-1) / new.float().norm(dim=-1).clamp_min(1e-8)).amax(0)


class Arm:
    def __init__(self, name: str, D: int):
        self.name = name
        self.depth = D
        self.kind = "AR"
        self.tau = 0.0
        self.freeze = self.repair = False
        if name.startswith("trunc"):
            self.kind, self.depth = "AR", int(name[5:])
        elif name == "jac_exact":
            self.kind = "jac"
        elif name.startswith("jac_g"):
            self.kind = "jac"
            body = name[5:]
            if body.endswith("_f"):
                self.freeze, body = True, body[:-2]
            if body.endswith("_rep"):
                self.repair, body = True, body[:-4]
            self.tau = float(body)
        elif name != "AR":
            raise ValueError(name)


@torch.no_grad()
def token_step(eng: VarEngine, arm: Arm, a, rows, cur, prev_traj, stats):
    D = eng.D
    h0 = eng.prelude(rows, cur)
    if arm.kind == "AR":
        h = h0
        for r in range(1, D + 1):
            h = eng.step_many(h.unsqueeze(0), [r])[0]
        stats["rounds"].append(D)
        return eng.logits(h), None
    n = len(rows)
    h1 = eng.step_many(h0.unsqueeze(0), [1])[0]
    if a.init == "prev_shift" and prev_traj is not None and all(prev_traj["ok"][b] for b in rows):
        pt = prev_traj["z"][:, rows]
        z = torch.cat([h0.unsqueeze(0), pt[1:] + (h1 - pt[1]).unsqueeze(0)])
    elif a.init == "constant":
        z = torch.stack([h0] + [h1] * D)
    else:
        z = torch.stack([h0] + [h1 + (r - 1) * (h1 - h0) for r in range(1, D + 1)])
    z[1] = h1
    ds = list(range(1, D + 1))
    stepper = None
    if a.corrector != "jacobi":
        stepper = CorrectionStepper(D, 1, a.corrector, eng.step_many, omega=a.omega,
                                    scale=a.scale, history=a.history, damping=a.damping)
    active = torch.ones(n, dtype=torch.bool, device=h0.device)
    prev_arg = None
    k_used = D - 1
    lg = None
    for k in range(1, D):
        if stepper is not None:
            new, _ = stepper.step(z, policy="full")
        elif arm.freeze and not bool(active.all()):
            idx = active.nonzero().squeeze(-1)
            sub = idx.tolist()
            keep_rows, keep_tok, keep_e = eng.rows, eng.tok_ids, eng.e
            eng.rows = [keep_rows[j] for j in sub]
            eng.tok_ids = [keep_tok[j] for j in sub]
            eng.e = None if keep_e is None else keep_e[idx]
            out = eng.step_many(z[:-1, idx], ds)
            eng.rows, eng.tok_ids, eng.e = keep_rows, keep_tok, keep_e
            new = z.clone()
            new[1:, idx] = out
        else:
            new = torch.cat([h0.unsqueeze(0), eng.step_many(z[:-1], ds)])
        res = rel_rows(new[1:], z[1:]) if arm.tau > 0 else None
        z = new
        if arm.tau > 0:
            lg = eng.logits(z[D])
            arg = lg.argmax(-1)
            if prev_arg is not None:
                ok = (res < arm.tau) & (arg == prev_arg)
                if arm.freeze:
                    active &= ~ok
                    if not bool(active.any()):
                        k_used = k
                        break
                elif bool(ok.all()):
                    k_used = k
                    break
            prev_arg = arg
    if arm.tau == 0:
        lg = eng.logits(z[D])
    stats["rounds"].append(1 + k_used)
    if arm.repair and k_used < D - 1:
        # rows z[0..k_used+1] are exact in real arithmetic after k_used sweeps; recompute the rest
        # serially so every depth slot holds KV of the exact trajectory (token stays candidate's).
        start = min(k_used + 1, D)
        h = z[start]
        for r in range(start + 1, D + 1):
            h = eng.step_many(h.unsqueeze(0), [r])[0]
            z[r] = h
        eng.logits(z[D])                                       # exact coda KV write (logits unused)
    return lg, z


@torch.no_grad()
def decode_cohort(lm, bcls, kind, tok, prompts, stops, arm: Arm, a, eos):
    B = len(prompts)
    S = max(p.numel() for p in prompts) + a.max_new + 8
    eng = VarEngine(lm, bcls, kind, arm.depth, B, S)
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    lg = eng.prefill_rows(prompts)
    cur_all = lg.argmax(-1).tolist()
    sync()
    ttft = time.perf_counter() - t0
    gens = [[c] for c in cur_all]
    done = [False] * B
    stats = {"rounds": [], "tpot": []}
    prev = {"z": None, "ok": [False] * B} if a.init == "prev_shift" else None

    def finished(b):
        g = gens[b]
        if len(g) >= a.max_new or (eos is not None and g[-1] == eos):
            return True
        tail = tok.decode(g[-24:], skip_special_tokens=False)
        return any(s in tail for s in stops[b])

    for b in range(B):
        done[b] = finished(b)
    t1 = time.perf_counter()
    while not all(done):
        rows = [b for b in range(B) if not done[b]]
        ts = time.perf_counter()
        lg, z = token_step(eng, arm, a, rows, [gens[b][-1] for b in rows], prev, stats)
        eng.advance()
        nxt = lg.argmax(-1).tolist()
        if prev is not None and z is not None:
            if prev["z"] is None:
                prev["z"] = torch.zeros((eng.D + 1, B, z.shape[-1]), dtype=z.dtype, device=z.device)
            prev["z"][:, rows] = z
            for b in rows:
                prev["ok"][b] = True
        for j, b in enumerate(rows):
            gens[b].append(nxt[j])
            done[b] = finished(b)
        stats["tpot"].append((time.perf_counter() - ts) * 1e3)
    sync()
    dec = time.perf_counter() - t1
    return dict(gens=gens, ttft_s=ttft, decode_s=dec, wall_s=ttft + dec, steps=len(stats["tpot"]),
                tpot=stats["tpot"], rounds=stats["rounds"], core_calls=eng.n.core_calls,
                core_rows=eng.n.core_rows, coda_calls=eng.n.coda_calls, coda_rows=eng.n.coda_rows,
                hbm_peak_gb=torch.cuda.max_memory_allocated() / 2**30)


def score(ds, task, text, row):
    ok, pred, extra = ds.score(task, text, row["gold"], row["stop"], row.get("meta"))
    out = dict(correct_strict=bool(ok), pred=pred)
    if task == "gsm8k":
        out["correct_flex"] = bool(extra.get("correct_flex"))
    else:
        out["correct_boxed"] = bool(extra.get("correct_boxed", extra.get("math_verify", ok)))
        out["extra_keys"] = sorted(extra)
    return out


def pct(v, p):
    s = sorted(v)
    return s[int(p * (len(s) - 1))] if s else None


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--task", default="gsm8k", choices=["gsm8k", "math500"])
    p.add_argument("--fewshot", type=int, default=None)
    p.add_argument("--split", required=True, choices=["pilot", "dev", "test"])
    p.add_argument("--idx-start", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--init", default="linear", choices=["linear", "constant", "prev_shift"])
    p.add_argument("--corrector", default="jacobi",
                   choices=["jacobi", "damped", "anderson", "newton_scaled", "newton_identity"])
    p.add_argument("--omega", type=float, default=0.75)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--damping", type=float, default=0.8)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    from experiments.exp3_accuracy import tasks as ds
    lm, tok, bcls, kind = load(a.model, a.dtype)
    eos = lm.config.eos_token_id if isinstance(lm.config.eos_token_id, int) else None
    fs = a.fewshot if a.fewshot is not None else (3 if a.task == "gsm8k" else 4)
    rows = ds.load_dataset_prompts(a.task, fs, a.idx_start + a.n)[a.idx_start:a.idx_start + a.n]
    arms = [Arm(x, a.depth) for x in a.arms.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(vars(a), **env_info(a), fewshot_used=fs, status="E2E_FREE_RUNNING_NO_REFILL",
                timing="CUDA-synchronised wall: prefill (TTFT) + full decode loop incl. all overhead")
    json.dump(meta, open(str(out) + ".meta.json", "w"), indent=1)
    fh = open(str(out) + ".jsonl", "w")
    # warm-up every arm once on a tiny cohort (not recorded)
    wp = [tok(rows[0]["prompt"], return_tensors="pt").input_ids[0]] * min(a.batch, 2)
    wa = argparse.Namespace(**{**vars(a), "max_new": 8})
    for arm in arms:
        decode_cohort(lm, bcls, kind, tok, wp, [rows[0]["stop"]] * len(wp), arm, wa, eos)
    nco = (len(rows) + a.batch - 1) // a.batch
    for c in range(nco):
        chunk = rows[c * a.batch:(c + 1) * a.batch]
        prompts = [tok(r["prompt"], return_tensors="pt").input_ids[0] for r in chunk]
        order = list(arms)
        random.Random(f"{a.seed}:{c}").shuffle(order)
        recs = {}
        for arm in order:
            try:
                r = decode_cohort(lm, bcls, kind, tok, prompts, [x["stop"] for x in chunk], arm, a, eos)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                recs[arm.name] = dict(oom=True, error=str(e)[:200])
                continue
            recs[arm.name] = r
            torch.cuda.empty_cache()
        ar = recs.get("AR")
        for arm in arms:
            r = recs[arm.name]
            if r.get("oom"):
                fh.write(json.dumps(dict(cohort=c, arm=arm.name, B=len(chunk), oom=True, error=r["error"])) + "\n")
                continue
            ntok = sum(len(g) for g in r["gens"])
            base = dict(cohort=c, arm=arm.name, order=[x.name for x in order].index(arm.name), B=len(chunk),
                        model=a.model, depth=arm.depth, task=a.task, split=a.split,
                        ttft_s=r["ttft_s"], decode_s=r["decode_s"], wall_s=r["wall_s"], steps=r["steps"],
                        cohort_tokens=ntok, tok_per_s=ntok / r["wall_s"], decode_tok_per_s=ntok / r["decode_s"],
                        tpot_p50_ms=pct(r["tpot"], .5), tpot_p90_ms=pct(r["tpot"], .9),
                        rounds_mean=sum(r["rounds"]) / max(1, len(r["rounds"])),
                        core_calls=r["core_calls"], core_rows=r["core_rows"], coda_calls=r["coda_calls"],
                        coda_rows=r["coda_rows"], hbm_peak_gb=r["hbm_peak_gb"])
            for j, row in enumerate(chunk):
                g = r["gens"][j]
                rec = dict(base, prompt_idx=a.idx_start + c * a.batch + j, tokens=g, n_tokens=len(g),
                           **score(ds, a.task, tok.decode(g, skip_special_tokens=True), row))
                if ar and not ar.get("oom"):
                    ga = ar["gens"][j]
                    n = min(len(ga), len(g))
                    fm = next((i for i in range(n) if ga[i] != g[i]), None)
                    rec["exact_to_AR"] = g == ga
                    rec["first_mismatch_vs_AR"] = fm if fm is not None else (None if g == ga else n)
                fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"[e2e] cohort {c}/{nco} " + " ".join(
            f"{k}:{(sum(len(g) for g in v['gens']) / v['wall_s']):.1f}" if not v.get("oom") else f"{k}:OOM"
            for k, v in recs.items()), flush=True)
    fh.close()


if __name__ == "__main__":
    main()
