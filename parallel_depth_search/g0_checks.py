"""G0 foundation gates on the real frozen models (GPU, opt-in).

--check native : (complements parallel_depth_poc.e0_native_parity, which covers FP32 engine-vs-HF)
   (1) VarEngine (per-request prefill, variable length) vs PoC DepthEngine on an equal-length
       cohort, FP32 AR rollout: tokens identical, max |logit diff|.
   (2) BF16 batch-shape/order sensitivity on variable-length GSM8K prompts: AR rollout of each
       prompt at B=1, inside a B=16 cohort, and inside the same cohort with permuted row order.
       Reports token agreement, first mismatch and logit drift. (Characterisation, not a gate:
       BF16 GEMM row-shape effects are expected; FP32 gives the parity gate.)
--check kv : candidate-KV isolation and commit audit (per dtype), on variable-length prompts:
   ref   plain AR rollout                                   -> tokens, per-depth bank digest
   cand  3 Jacobi candidate rounds + candidate coda each token, then exact serial commit
         -> must equal ref bit-for-bit (tokens, every depth slot, coda slot)
   carry E5-style approximate carry (Jacobi stop at tau, candidate KV kept, no commit)
         -> divergence vs ref reported (explains E5 non-exactness)
   jexact D-1 Jacobi rounds from exact h1 (no recompute): exact in real arithmetic, row-shape
         differs from AR -> bitwise / tolerance comparison vs ref
Writes <out>.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from parallel_depth_poc import wf_engine as W
from parallel_depth_poc.depth_diag import cohort_texts, env_info
from .vengine import VarEngine, bank_digest, bank_snapshot, bank_rel_diff, load


def gsm8k_prompts(tok, idx: list[int]):
    from experiments.exp3_accuracy import tasks as ds
    rows = ds.load_dataset_prompts("gsm8k", 3, max(idx) + 1)
    return [tok(rows[i]["prompt"], return_tensors="pt").input_ids[0] for i in idx]


@torch.no_grad()
def ar_rollout(eng: VarEngine, prompts, new: int):
    lg = eng.prefill_rows(prompts)
    rows = list(range(eng.B))
    cur = lg.argmax(-1).tolist()
    toks, logits = [cur], [lg]
    for _ in range(new - 1):
        h0 = eng.prelude(rows, cur)
        h = eng.serial(h0)[-1]
        lg = eng.logits(h)
        eng.advance()
        cur = lg.argmax(-1).tolist()
        toks.append(cur)
        logits.append(lg)
    return torch.tensor(toks).T, torch.stack(logits, 1)          # [B,new], [B,new,V]


@torch.no_grad()
def jacobi_rollout(eng: VarEngine, prompts, new: int, mode: str, tau: float = 0.1, rounds: int = 3):
    lg = eng.prefill_rows(prompts)
    rows = list(range(eng.B))
    cur = lg.argmax(-1).tolist()
    toks, logits, used = [cur], [lg], []
    D = eng.D
    for _ in range(new - 1):
        h0 = eng.prelude(rows, cur)
        h1 = eng.step_many(h0.unsqueeze(0), [1])[0]
        z = torch.stack([h0] + [h1 + (r - 1) * (h1 - h0) for r in range(1, D + 1)])
        z[1] = h1
        ds = list(range(1, D + 1))
        nmax = {"cand": rounds, "carry": D - 1, "jexact": D - 1}[mode]
        prev_arg, k_used = None, nmax
        for k in range(1, nmax + 1):
            new_z = torch.cat([h0.unsqueeze(0), eng.step_many(z[:-1], ds)])
            res = ((new_z[1:].float() - z[1:].float()).norm(dim=-1)
                   / new_z[1:].float().norm(dim=-1).clamp_min(1e-8)).amax()
            z = new_z
            if mode in ("cand", "carry"):
                lg_c = eng.logits(z[D])                                     # candidate coda write
                arg = lg_c.argmax(-1)
                if mode == "carry" and prev_arg is not None and float(res) < tau and torch.equal(arg, prev_arg):
                    k_used = k
                    break
                prev_arg = arg
        if mode == "cand":
            h = eng.serial(h0)[-1]                                          # exact commit rewrites slots
            lg = eng.logits(h)
        elif mode == "carry":
            lg = lg_c
        else:
            lg = eng.logits(z[D])
        used.append(k_used)
        eng.advance()
        cur = lg.argmax(-1).tolist()
        toks.append(cur)
        logits.append(lg)
    return torch.tensor(toks).T, torch.stack(logits, 1), used


def first_mismatch(a, b):
    out = []
    for i in range(a.shape[0]):
        m = (a[i] != b[i]).nonzero()
        out.append(int(m[0]) if len(m) else None)
    return out


def check_native(a, res):
    # (1) VarEngine vs DepthEngine, FP32, equal-length cohort
    lm, tok, bcls, kind = load(a.model, "float32")
    D = a.depths[-1]
    ids = W.make_cohort(tok, cohort_texts("gsm8k3", 64), a.prompt_len, 8)
    de = W.DepthEngine(lm, bcls, kind, D, 8, a.prompt_len + a.new + 8)
    lg = de.prefill(ids)
    cur = lg.argmax(-1)
    t_de, l_de = [cur], [lg]
    for _ in range(a.new - 1):
        _, lg = de.commit(de.prelude(cur.tolist()))
        cur = lg.argmax(-1)
        t_de.append(cur)
        l_de.append(lg)
    t_de = torch.stack(t_de, 1).cpu()
    l_de = torch.stack(l_de, 1)
    ve = VarEngine(lm, bcls, kind, D, 8, a.prompt_len + a.new + 8)
    t_ve, l_ve = ar_rollout(ve, list(ids), a.new)
    res["varengine_vs_depthengine_fp32"] = dict(
        depth=D, tokens_identical=bool(torch.equal(t_de, t_ve)),
        max_abs_logit_diff=float((l_de - l_ve).abs().amax()), positions=int(t_de.numel()))
    del lm, de, ve
    torch.cuda.empty_cache()
    # (2) BF16 batch shape / order
    lm, tok, bcls, kind = load(a.model, "bfloat16")
    idx = list(range(a.offset, a.offset + 16))
    prompts = gsm8k_prompts(tok, idx)
    S = max(p.numel() for p in prompts) + a.new + 8
    perm = list(range(16))
    random.Random(a.seed).shuffle(perm)
    cells = {}
    for D in a.depths:
        b1 = []
        b1l = []
        for p in prompts:
            e = VarEngine(lm, bcls, kind, D, 1, S)
            t, l = ar_rollout(e, [p], a.new)
            b1.append(t[0])
            b1l.append(l[0])
        b1 = torch.stack(b1)
        b1l = torch.stack(b1l)
        e = VarEngine(lm, bcls, kind, D, 16, S)
        b16, b16l = ar_rollout(e, prompts, a.new)
        e = VarEngine(lm, bcls, kind, D, 16, S)
        bp, bpl = ar_rollout(e, [prompts[i] for i in perm], a.new)
        inv = [perm.index(i) for i in range(16)]
        bp, bpl = bp[inv], bpl[inv]
        def cmp(x, y, lx, ly):
            fm = first_mismatch(x, y)
            # logit drift only over the common-prefix positions (identical inputs)
            drift = []
            for i in range(16):
                n = fm[i] if fm[i] is not None else a.new
                n = max(n, 1)
                drift.append(float((lx[i, :n] - ly[i, :n]).abs().amax()))
            return dict(seq_exact_rate=sum(f is None for f in fm) / 16,
                        token_agree=float((x == y).float().mean()), first_mismatch=fm,
                        max_logit_drift_common_prefix=max(drift))
        cells[f"D{D}"] = dict(b1_vs_b16=cmp(b1, b16, b1l, b16l), b16_vs_permuted=cmp(b16, bp, b16l, bpl))
        print(D, json.dumps(cells[f"D{D}"])[:300], flush=True)
    res["bf16_batch_order"] = dict(prompt_idx=idx, permutation=perm, new=a.new, cells=cells)


def check_kv(a, res):
    lm, tok, bcls, kind = load(a.model, a.dtype)
    idx = list(range(a.offset, a.offset + a.batch))
    prompts = gsm8k_prompts(tok, idx)
    S = max(p.numel() for p in prompts) + a.new + 8
    cells = {}
    for D in a.depths:
        e = VarEngine(lm, bcls, kind, D, a.batch, S)
        t_ref, l_ref = ar_rollout(e, prompts, a.new)
        d_ref, s_ref = bank_digest(e), bank_snapshot(e)
        del e
        out = {}
        for mode in ("cand", "carry", "jexact"):
            e = VarEngine(lm, bcls, kind, D, a.batch, S)
            t, l, used = jacobi_rollout(e, prompts, a.new, mode, tau=a.tau)
            dg = bank_digest(e)
            fm = first_mismatch(t, t_ref)
            same_inputs = all(f is None for f in fm)
            rd = bank_rel_diff(bank_snapshot(e), s_ref) if same_inputs else None
            out[mode] = dict(
                tokens_identical=bool(torch.equal(t, t_ref)), seq_exact_rate=sum(f is None for f in fm) / a.batch,
                first_mismatch=fm, logits_bitwise=bool(torch.equal(l, l_ref)),
                max_abs_logit_diff=float((l - l_ref).abs().amax()) if same_inputs else None,
                depth_banks_bitwise={k: dg[k] == d_ref[k] for k in d_ref},
                bank_max_rel_diff=rd, rounds_mean=sum(used) / len(used))
            del e
            torch.cuda.empty_cache()
        c = out["cand"]
        out["gates"] = dict(
            candidate_cache_isolated=c["tokens_identical"] and c["logits_bitwise"],
            per_depth_bank_parity=all(v for k, v in c["depth_banks_bitwise"].items() if k != "coda"),
            coda_cache_parity=c["depth_banks_bitwise"].get("coda", True))
        cells[f"D{D}"] = out
        print(D, json.dumps(out["gates"]), json.dumps({m: out[m]["seq_exact_rate"] for m in ("cand", "carry", "jexact")}), flush=True)
    res["kv"] = dict(prompt_idx=idx, dtype=a.dtype, tau=a.tau, cells=cells)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", required=True, choices=["native", "kv"])
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depths", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--new", type=int, default=24)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    a.depths = [int(x) for x in a.depths.split(",")]
    a.depth = a.depths[-1]
    torch.manual_seed(a.seed)
    res = dict(config=vars(a) | env_info(a))
    (check_native if a.check == "native" else check_kv)(a, res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
