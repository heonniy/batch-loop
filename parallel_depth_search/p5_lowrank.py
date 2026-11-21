"""P5: low-rank ridge trajectory predictor fitted on a TRAIN cohort, evaluated on a disjoint cohort
(GPU, opt-in, teacher-forced diagnostic; frozen model weights, the predictor is a small offline fit).

Model (per depth r, rank k, ridge lam):  h_r ~= h1 + c_r + B_r P^T (h1 - h0)
  P [H,k] = top-k right singular vectors of the train deltas X = h1 - h0 (no future states needed
  at inference: only the real h0, h1), B_r [H,k] and c_r by ridge regression on exact train
  trajectories. Train cohort = pilot prompts offset --train-offset (B x T tokens), eval cohort =
  disjoint pilot prompts offset --offset. Inference cost: one [n,H]x[H,k] + D x [n,k]x[k,H] matmul.
Per eval token: exact reference, predictions {constant, lowrank_k*}, final-depth top-1/KL at
round 0 (prediction only) and after Jacobi rounds 1..K (one physical step_many per round), and the
measured predictor wall time. Exact commit advances (no candidate state leaks).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from parallel_depth_poc import wf_engine as W
from parallel_depth_poc.depth_diag import cohort_texts, env_info, kl_top1, rel


@torch.no_grad()
def rollout_states(eng, ids, T):
    """Exact AR rollout; returns list of trajectories [D+1,B,H] (one per token step)."""
    lg = eng.prefill(ids)
    cur = lg.argmax(-1).tolist()
    trajs = []
    for _ in range(T):
        traj, lg = eng.commit(eng.prelude(cur))
        trajs.append(traj.float())
        cur = lg.argmax(-1).tolist()
    return trajs


def fit(trajs, D, k, lam):
    Z = torch.cat(trajs, dim=1)                                   # [D+1,N,H]
    X = Z[1] - Z[0]
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    P = Vh[:k].T                                                  # [H,k]
    F = torch.cat([X @ P, torch.ones(X.shape[0], 1, device=X.device)], 1)   # [N,k+1]
    G = F.T @ F + lam * torch.eye(k + 1, device=X.device)
    maps = []
    for r in range(2, D + 1):
        Y = Z[r] - Z[1]
        Wr = torch.linalg.solve(G, F.T @ Y)                       # [k+1,H]
        maps.append(Wr)
    return P, torch.stack(maps)                                   # [D-1,k+1,H]


def predict(P, M, h0, h1, D):
    f = torch.cat([(h1.float() - h0.float()) @ P, torch.ones(h0.shape[0], 1, device=h0.device)], 1)
    out = h1.float().unsqueeze(0) + torch.einsum("nk,rkh->rnh", f, M)
    return torch.cat([h0.unsqueeze(0), h1.unsqueeze(0), out.to(h0.dtype)])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--train-offset", type=int, default=64)
    p.add_argument("--train-tokens", type=int, default=64)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--ranks", default="8,32,128")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    torch.manual_seed(a.seed)
    D, B = a.depth, a.batch
    lm, tok, bcls, kind = W.load_model(a.model, torch.bfloat16)
    texts = cohort_texts("gsm8k3", 128)
    tr_ids = W.make_cohort(tok, texts[a.train_offset:], a.prompt_len, B)
    ev_ids = W.make_cohort(tok, texts[a.offset:a.train_offset], a.prompt_len, B)
    assert not set(map(tuple, tr_ids.tolist())) & set(map(tuple, ev_ids.tolist())), "train/eval overlap"
    eng = W.DepthEngine(lm, bcls, kind, D, B, a.prompt_len + max(a.train_tokens, a.decode_steps) + 8)
    trajs = rollout_states(eng, tr_ids, a.train_tokens)
    ranks = [int(x) for x in a.ranks.split(",")]
    models = {k: fit(trajs, D, k, a.lam * len(trajs) * B) for k in ranks}
    # train-fit quality (in-sample, for reference only)
    del trajs
    eng = W.DepthEngine(lm, bcls, kind, D, B, a.prompt_len + a.decode_steps + 8)
    lg = eng.prefill(ev_ids)
    cur = lg.argmax(-1).tolist()
    recs = []
    for t in range(a.decode_steps):
        h0 = eng.prelude(cur)
        ref = eng.serial(h0)
        lref = eng.logits(ref[D])
        h1 = ref[1]
        inits = {"constant": torch.stack([h0] + [h1] * D)}
        tms = {"constant": 0.0}
        for k, (P, M) in models.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            inits[f"lowrank{k}"] = predict(P, M, h0, h1, D)
            torch.cuda.synchronize(); tms[f"lowrank{k}"] = 1e3 * (time.perf_counter() - t0)
        row = {"t": t}
        for name, z in inits.items():
            curve = []
            per_depth_rel = [float(rel(z[r], ref[r]).mean()) for r in range(2, D + 1)]
            for k in range(a.rounds + 1):
                if k > 0:
                    z = torch.cat([h0.unsqueeze(0), h1.unsqueeze(0),
                                   eng.step_many(z[1:-1], torch.arange(2, D + 1))])
                kl, ag = kl_top1(lref, eng.logits(z[D]))
                curve.append(dict(k=k, top1=float(ag.float().mean()), kl=float(kl.mean()),
                                  rel=float(rel(z[D], ref[D]).mean())))
            row[name] = dict(curve=curve, predictor_ms=tms[name], rel_by_depth=per_depth_rel)
        _, lg = eng.commit(h0)
        cur = lg.argmax(-1).tolist()
        recs.append(row)
        print(f"[p5] t={t} " + " ".join(f"{n}:{row[n]['curve'][0]['top1']:.2f}/{row[n]['curve'][1]['top1']:.2f}" for n in inits), flush=True)
    names = list(recs[0].keys() - {"t"})
    summ = {n: dict(top1_by_round=[sum(r[n]["curve"][k]["top1"] for r in recs) / len(recs) for k in range(a.rounds + 1)],
                    kl_by_round=[sum(r[n]["curve"][k]["kl"] for r in recs) / len(recs) for k in range(a.rounds + 1)],
                    rel_final_by_round=[sum(r[n]["curve"][k]["rel"] for r in recs) / len(recs) for k in range(a.rounds + 1)],
                    predictor_ms=sum(r[n]["predictor_ms"] for r in recs) / len(recs)) for n in names}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(config=vars(a) | env_info(a), status="GPU_DIAGNOSTIC_ONLY_TEACHER_FORCED",
                   train=dict(offset=a.train_offset, tokens=a.train_tokens, B=B), summary=summ, records=recs),
              open(str(out) + ".json", "w"), indent=1)
    print(json.dumps({n: [round(x, 3) for x in s["top1_by_round"]] for n, s in summ.items()}))


if __name__ == "__main__":
    main()
