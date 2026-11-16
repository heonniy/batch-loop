"""E2/E3/E4 (+ E0 KV-isolation) online diagnostics on the real frozen core (GPU, opt-in).

Protocol per decode step t (cohort of B equal-length prompts, fixed depth D, greedy):
  1. h0 = prelude(current tokens). EXACT serial trajectory ref[0..D] (writes exact KV), exact
     logits at a depth subset S (always incl. D).
  2. E2 predictors (no future information; h1 is the real first core call):
       constant (h1), linear (h1 + (r-1)(h1-h0)), prev (exact trajectory of token t-1, t>=1),
       prev_shift (prev[r] + h1_t - prev[1], t>=1).
     per depth: relative L2, cosine; at depths S: KL(exact||pred) and top-1 agreement.
  3. E3 Jacobi (init in {constant, linear, prev_shift}) for k = 1..D rounds; every round is ONE
     physical step_many call over all D depths. After each round: final-depth relative L2,
     final-logit top-1 agreement and KL vs exact, online residual (max_r ||z^k_r - z^{k-1}_r|| /
     ||z^k_r||). Block Parareal (correction G(new)+F(old)-G(old) accumulated in FP32) (b in {2,4,8}, b<D, D%b==0; coarse G in {identity, onestep:
     G_j = F_{jb+1}}), rounds 1..D/b, same metrics, explicit sequential-call accounting.
  4. E4: perturb h_r (r in {1, D//2, D-1}) by eps*||h_r||*u (eps: FP32 {1e-4,1e-3,1e-2}; BF16
     {1e-2,1e-1} because BF16 cannot represent smaller per-element perturbations), propagate
     serially; ||dh_{r+k}||/||dh_r|| for k=1..D-r, final-logit KL and top-1 flip; and an
     unperturbed re-run from h_r (numerical noise floor, should be 0).
  5. commit: recompute the exact serial trajectory (rewrites every slot at t), check it is
     bit-identical to ref, exact coda, greedy next token.
After the rollout, a fresh prefill + plain fixed-depth AR rollout (no solver branches) must give
identical tokens: candidate KV never leaked into committed KV (E0 item 3).

Writes <out>.metrics.jsonl.gz (one row per (t, b)), <out>.summary.json, <out>.config.json.
No timing claims: wall time here includes diagnostic branches by design.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch

from . import wf_engine as W

INITS = ("constant", "linear", "prev_shift")


def rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:          # [B,H] -> [B]
    return (a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1).clamp_min(1e-8)


def kl_top1(lref: torch.Tensor, lp: torch.Tensor):                    # [B,V] -> ([B],[B])
    a = torch.log_softmax(lref, -1)
    b = torch.log_softmax(lp, -1)
    kl = (a.exp() * (a - b)).sum(-1).clamp_min(0)
    return kl, (lref.argmax(-1) == lp.argmax(-1))


def depth_subset(D: int) -> list[int]:
    return sorted({s for s in (2, D // 4, D // 2, (3 * D) // 4, D) if 2 <= s <= D}) or [D]


def lst(x: torch.Tensor, nd: int = 5):
    return [round(float(v), nd) for v in x.tolist()]


@torch.no_grad()
def run(a) -> None:
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    torch.manual_seed(a.seed)
    lm, tok, batch_cls, kind = W.load_model(a.model, dtype)
    texts = cohort_texts(a.cohort, a.offset + 4 * a.batch)[a.offset:]
    ids = W.make_cohort(tok, texts, a.prompt_len, a.batch)
    eng = W.DepthEngine(lm, batch_cls, kind, a.depth, a.batch, a.prompt_len + a.decode_steps + 8)
    D, B = a.depth, a.batch
    S = depth_subset(D)
    blocks = [b for b in (2, 4, 8) if b < D and D % b == 0]
    pert_r = sorted({r for r in (1, D // 2, D - 1) if 1 <= r < D})
    gen = torch.Generator(device="cuda").manual_seed(a.seed + 7)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = gzip.open(str(out) + ".metrics.jsonl.gz", "wt")
    last = eng.prefill(ids)
    cur = last.argmax(-1).tolist()
    tokens = [cur]
    prev_traj = None
    nondet = 0
    t0 = time.time()
    for t in range(a.decode_steps):
        h0 = eng.prelude(cur)
        ref = eng.serial(h0)                                         # [D+1,B,H] exact
        lref = {s: eng.logits(ref[s]) for s in S}
        lD = lref[D]
        rows = [dict(t=t, b=b, token=cur[b]) for b in range(B)]

        # ---------------- E2 predictors ----------------
        h1 = ref[1]
        preds = {"constant": torch.stack([h0] + [h1] * D),
                 "linear": torch.stack([h0] + [h1 + (r - 1) * (h1 - h0) for r in range(1, D + 1)])}
        if prev_traj is not None:
            preds["prev"] = torch.cat([h0.unsqueeze(0), prev_traj[1:]])
            preds["prev_shift"] = torch.cat([h0.unsqueeze(0), prev_traj[1:] + (h1 - prev_traj[1]).unsqueeze(0)])
        for name, P in preds.items():
            r_l2 = torch.stack([rel(P[r], ref[r]) for r in range(1, D + 1)], 1)          # [B,D]
            cos = torch.stack([torch.nn.functional.cosine_similarity(P[r].float(), ref[r].float(), -1)
                               for r in range(1, D + 1)], 1)
            kls, agr = [], []
            for s in S:
                k, g = kl_top1(lref[s], eng.logits(P[s]))
                kls.append(k); agr.append(g)
            kls, agr = torch.stack(kls, 1), torch.stack(agr, 1)
            for b in range(B):
                rows[b].setdefault("pred", {})[name] = dict(rel=lst(r_l2[b]), cos=lst(cos[b], 6),
                                                           kl=lst(kls[b]), top1=agr[b].int().tolist())

        # ---------------- E3 Jacobi ----------------
        ds_all = torch.arange(1, D + 1)
        for init in INITS:
            if init == "prev_shift" and "prev_shift" not in preds:
                continue
            z = preds[init].clone()
            z[1] = h1                                                  # first step is truly evaluated
            res = {k: [] for k in ("rel", "top1", "kl", "resid")}
            for k in range(1, D + 1):
                nxt = eng.step_many(z[:-1], ds_all)
                new = torch.cat([h0.unsqueeze(0), nxt])
                upd = torch.stack([rel(z[r], new[r]) for r in range(1, D + 1)], 1)  # ||new-old||/||new||
                z = new
                kl, g = kl_top1(lD, eng.logits(z[D]))
                res["rel"].append(rel(z[D], ref[D])); res["top1"].append(g)
                res["kl"].append(kl); res["resid"].append(upd.amax(1))
            for key in res:
                res[key] = torch.stack(res[key], 1)                                     # [B,D]
            exact_D = rel(z[D], ref[D])
            for b in range(B):
                rows[b].setdefault("jacobi", {})[init] = dict(
                    rel=lst(res["rel"][b]), top1=res["top1"][b].int().tolist(), kl=lst(res["kl"][b]),
                    resid=lst(res["resid"][b]), relD_after_D=float(exact_D[b]))

        # ---------------- E3 Parareal ----------------
        for bs in blocks:
            nb = D // bs
            bid = torch.arange(nb)
            for gname in ("identity", "onestep"):
                def G(states, block_ids, _bs=bs, _g=gname):
                    if _g == "identity":
                        return states
                    return eng.step_many(states, block_ids * _bs + 1)
                q = [h0]
                for j in range(nb):
                    q.append(G(q[-1].unsqueeze(0), bid[j:j + 1])[0])
                q = torch.stack(q)
                res = {k: [] for k in ("rel", "top1", "kl", "resid")}
                for k in range(1, nb + 1):
                    work = q[:-1]
                    for loc in range(bs):
                        work = eng.step_many(work, bid * bs + loc + 1)
                    g_old = G(q[:-1], bid)
                    newer = [h0]
                    for j in range(nb):
                        corr = (G(newer[-1].unsqueeze(0), bid[j:j + 1])[0].float()
                                + work[j].float() - g_old[j].float())        # fp32 accumulate (no bf16 cancellation)
                        newer.append(corr.to(h0.dtype))
                    new_q = torch.stack(newer)
                    upd = torch.stack([(new_q[j].float() - q[j].float()).norm(dim=-1)
                                       / new_q[j].float().norm(dim=-1).clamp_min(1e-8) for j in range(1, nb + 1)], 1)
                    q = new_q
                    kl, g = kl_top1(lD, eng.logits(q[nb]))
                    res["rel"].append(rel(q[nb], ref[D])); res["top1"].append(g)
                    res["kl"].append(kl); res["resid"].append(upd.amax(1))
                for key in res:
                    res[key] = torch.stack(res[key], 1)
                for b in range(B):
                    rows[b].setdefault("parareal", {})[f"b{bs}_{gname}"] = dict(
                        rel=lst(res["rel"][b]), top1=res["top1"][b].int().tolist(), kl=lst(res["kl"][b]),
                        resid=lst(res["resid"][b]))

        # ---------------- E4 perturbation ----------------
        for r in pert_r:
            base = eng.serial(h0, start=r, h_start=ref[r])
            floor = rel(base[D], ref[D])
            # BF16 per-element resolution is ~4e-3 relative: smaller perturbations are quantised away
            for eps in ((1e-4, 1e-3, 1e-2) if h0.dtype == torch.float32 else (1e-2, 1e-1)):
                u = torch.randn(ref[r].shape, generator=gen, device="cuda", dtype=torch.float32)
                u = u / u.norm(dim=-1, keepdim=True)
                d0 = eps * ref[r].float().norm(dim=-1, keepdim=True) * u
                hp = (ref[r].float() + d0).to(ref.dtype)
                d0n = (hp.float() - ref[r].float()).norm(dim=-1).clamp_min(1e-12)       # realised |dh_r|
                tr = eng.serial(h0, start=r, h_start=hp)
                amp = torch.stack([(tr[r + k].float() - ref[r + k].float()).norm(dim=-1) / d0n
                                   for k in range(1, D - r + 1)], 1)
                kl, g = kl_top1(lD, eng.logits(tr[D]))
                for b in range(B):
                    rows[b].setdefault("perturb", {})[f"r{r}_e{eps:g}"] = dict(
                        amp=lst(amp[b]), kl=float(kl[b]), flip=int(not bool(g[b])), floor=float(floor[b]))

        # ---------------- commit ----------------
        traj, lg = eng.commit(h0)
        diff = (traj.float() - ref.float()).abs().amax().item()
        nondet += int(diff != 0.0)
        nxt = lg.argmax(-1).tolist()
        for b in range(B):
            rows[b]["commit_bitexact"] = diff == 0.0
            rows[b]["next"] = nxt[b]
            fh.write(json.dumps(rows[b]) + "\n")
        prev_traj = traj
        cur = nxt
        tokens.append(cur)
        if t % 8 == 0:
            print(f"[diag] t={t} {time.time() - t0:.0f}s calls={eng.n.core_calls}", flush=True)
    fh.close()

    # ---------------- E0 item 3: plain AR rollout must reproduce the tokens ----------------
    eng2 = W.DepthEngine(lm, batch_cls, kind, D, B, a.prompt_len + a.decode_steps + 8)
    cur2 = eng2.prefill(ids).argmax(-1).tolist()
    toks2 = [cur2]
    for t in range(a.decode_steps):
        h0 = eng2.prelude(cur2)
        _, lg = eng2.commit(h0)
        cur2 = lg.argmax(-1).tolist()
        toks2.append(cur2)
    same = [[toks2[t][b] == tokens[t][b] for t in range(len(tokens))] for b in range(B)]
    first_mm = [next((t for t, s in enumerate(row) if not s), None) for row in same]
    summary = dict(status="done", wall_s=time.time() - t0, core_calls=eng.n.core_calls,
                   core_rows=eng.n.core_rows, coda_calls=eng.n.coda_calls,
                   commit_nondeterministic_steps=nondet,
                   kv_isolation={"tokens_identical_to_plain_AR": all(all(r) for r in same),
                                 "first_mismatch_per_seq": first_mm},
                   tokens=tokens, depth_subset=S, parareal_blocks=blocks, perturb_depths=pert_r)
    json.dump(summary, open(str(out) + ".summary.json", "w"))
    print(json.dumps({k: v for k, v in summary.items() if k != "tokens"}), flush=True)


def cohort_texts(name: str, n: int) -> list[str]:
    from experiments.exp3_accuracy import tasks as ds
    if name == "gsm8k3":
        return [r["prompt"] for r in ds.load_dataset_prompts("gsm8k", 3, n)]
    if name == "long":       # synthetic long context: MATH-500 4-shot prompt i + GSM8K 8-shot prompt i
        m = ds.load_dataset_prompts("math500", 4, n)
        g = ds.load_dataset_prompts("gsm8k", 8, n)
        return [m[i]["prompt"] + "\n\n" + g[i]["prompt"] for i in range(n)]
    raise ValueError(name)


def env_info(a) -> dict:
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, text=True).strip()
        except Exception:
            return None
    return dict(gpu=torch.cuda.get_device_name(), driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
                torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                source_commit=sh("git -C " + str(Path(__file__).resolve().parents[1]) + " rev-parse HEAD"),
                wf_commit=sh(f"git -C {W.WF_ROOT} rev-parse HEAD"), model_id=W.MODEL_IDS[a.model],
                kv_policy="FULL depth-indexed (one KV slot per recurrent depth; author-native per-step cache)",
                attention_backend="FA varlen (bf16) / SDPA fallback (fp32), WaveFront fa4_attention")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--cohort", default="gsm8k3", choices=["gsm8k3", "long"])
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(a), **env_info(a)), open(a.out + ".config.json", "w"), indent=1)
    run(a)


if __name__ == "__main__":
    main()
