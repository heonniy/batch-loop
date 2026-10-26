"""E2 (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md): failure localisation, one job = (model, dataset, prompt).

Models: ouro-1.4b, ouro-2.6b (R=4), huginn-0125 (native R=32) and the separately labelled Huginn
depth sweep huginn-0125-r{4,8,16}. Prompts: 16 GSM8K + 16 MATH-500 from the Stage-1 selection.
Prefix: each model's own greedy FULL generation (E0 / R32 / E2 depth-sweep receipts); sites at
decode tokens g = 10/50/90% of the continuation. All interventions act on the current token only,
with history frozen from a B=1 FULL pass (kvp.xdepth):

 a  fixed-query cross-depth attention: layer l, step r, history read from step s (all s in S):
    relative change of the attention output and of the layer output vs s = r
 b  single-call intervention: layer l reads terminal history at step r (equivalently, its output
    is shifted by Δ = out_T − out_ref); Δ and two matched-norm Gaussian controls are propagated through
    the remaining steps; relative error per step end, final KL / top-1
 c  persistent reads: every call of step r reads terminal history; and every call of every step
    (the current token decoded under LAST semantics with a FULL history)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import MixedKV, claim  # noqa: E402
from kvp.xdepth import make_fn, run_all  # noqa: E402

MODELS = ("ouro-1.4b", "ouro-2.6b", "huginn-0125", "huginn-0125-r4", "huginn-0125-r8", "huginn-0125-r16")
STAGE1_SEL = "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json"
HUGINN_R32_RUN = "results/20261023T060125Z_huginn_r32_h200x8"
N_PER_DS = 16
TOKEN_FRACS = (0.1, 0.5, 0.9)
N_RAND = 2


def loop_subset(R):
    if R <= 4:
        return list(range(1, R + 1))
    base = {1, 2, 4, 8, 16, 24, R // 2, R - 1, R}
    return sorted(x for x in base if 1 <= x <= R)


def receipts(path: Path, model: str, ds: str):
    out = {}
    for f in sorted((path / "raw").glob(f"{model}__{ds}__gen__full-decode__b*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def source_run(run: Path, model: str) -> Path:
    if model == "huginn-0125":
        return Path(HUGINN_R32_RUN)
    if model.startswith("huginn-0125-r"):
        return run / "e2_huginn_depth"
    return run


def kl_top1(lb, lx):
    a, b = torch.log_softmax(lb.double(), -1), torch.log_softmax(lx.double(), -1)
    return (a.exp() * (a - b)).sum(-1), a.argmax(-1) == b.argmax(-1)


def site(fn, key):
    R, L = fn.R, fn.L
    cap = {}
    with torch.no_grad():
        hs = run_all(fn, fn.init(), cap=cap)
        base_logits = fn.readout(hs[-1])
        loops = loop_subset(R)
        # a. fixed-query cross-depth attention
        fq = []
        for l in range(L):
            for r in loops:
                x_in, a_ref, out_ref = cap[(l, r)]
                for s in loops:
                    if s == r:
                        continue
                    a_s, out_s = fn.block_io(l, x_in, r, s)
                    fq.append({"layer": l, "r": r, "s": s,
                               "D_attn": float((a_s - a_ref).double().norm() / (a_ref.double().norm() + 1e-30)),
                               "D_out": float((out_s - out_ref).double().norm() / (out_ref.double().norm() + 1e-30)),
                               "attn_over_resid": float(a_ref.double().norm() / (x_in.double().norm() + 1e-30))})
        # b. single-call terminal reads + matched-norm random controls, all propagated in one batch
        grid = [(l, r) for l in range(L) for r in loops if r < R]
        N = len(grid) * (1 + N_RAND)
        E = hs[0].shape[-1]
        deltas = {}
        meta = []
        gen = torch.Generator().manual_seed(zlib.crc32(key.encode()))
        for gi, (l, r) in enumerate(grid):
            x_in, _, out_ref = cap[(l, r)]
            _, out_t = fn.block_io(l, x_in, r, R)
            d = (out_t - out_ref)[0, 0]
            dn = d.double().norm()
            vecs = [d] + [(torch.randn(E, generator=gen, dtype=torch.float64) * 1.0).to(d.device) for _ in range(N_RAND)]
            vecs = [vecs[0]] + [(v / v.norm() * dn).to(d.dtype) for v in vecs[1:]]
            for k, v in enumerate(vecs):
                row = gi * (1 + N_RAND) + k
                deltas.setdefault((l, r), torch.zeros(N, 1, E, dtype=d.dtype, device=d.device))[row, 0] = v
                meta.append({"layer": l, "r": r, "kind": "kv" if k == 0 else f"rand{k}",
                             "E_inj": float(dn / (out_ref.double().norm() + 1e-30))})
        z = fn.init().expand(N, -1, -1).contiguous()
        traj = []
        for s in range(1, R + 1):
            z = fn.step(z, s, inject=deltas)
            ref = hs[s - 1]
            traj.append(((z - ref).double().norm(dim=-1)[:, 0] / (ref.double().norm() + 1e-30)).cpu())
        lg = fn.readout(z)
        kl, same = kl_top1(base_logits.expand_as(lg), lg)
        decay = []
        for i, m in enumerate(meta):
            ends = [float(traj[s - 1][i]) for s in range(m["r"], R + 1)]  # step r end .. step R end
            decay.append({**m, "E_step_end": ends, "E_final": ends[-1], "kl": float(kl[i]), "top1_same": bool(same[i])})
        # c. persistent reads
        pers = []
        for label, rt in [(f"step{r}_all_layers", frozenset((l, r) for l in range(L))) for r in loops if r < R] + \
                         [("all_steps_all_layers", frozenset((l, s) for l in range(L) for s in range(1, R + 1)))]:
            hz = run_all(fn, fn.init(), read_terminal=rt)
            e = [float((h - b).double().norm() / (b.double().norm() + 1e-30)) for h, b in zip(hz, hs)]
            k_, s_ = kl_top1(base_logits, fn.readout(hz[-1]))
            pers.append({"label": label, "E_step_end": e, "kl": float(k_[0]), "top1_same": bool(s_[0])})
    return {"fixed_q": fq, "decay": decay, "persistent": pers, "state_norm": [float(h.double().norm()) for h in hs]}


def run_job(ctx, job, out: Path, run_dir: Path):
    spec, tok, model, run, items = ctx
    bb = run.bb
    ds, uid = job["dataset"], job["uid"]
    it = items[ds][uid]
    rec = receipts(source_run(run_dir, job["model"]), job["model"], ds)[uid]
    prompt, cont = it["prompt_ids"], rec["gen_token_ids"]
    P, n = len(prompt), len(cont)
    gs = sorted({min(n - 1, max(1, round(f * n))) for f in TOKEN_FRACS}) if n >= 2 else []
    T = P + (max(gs) if gs else 1)
    dev = run.device
    kv = MixedKV(["last"], torch.ones(1, T, dtype=torch.bool), bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D, run.dtype, dev)
    pos, valid = torch.arange(T, device=dev)[None], torch.ones(1, T, dtype=torch.bool, device=dev)
    aux = bb.make_aux([uid], [0], T)
    tgt = torch.tensor(cont, device=dev)
    t0 = time.time()
    with torch.inference_mode():
        run._prefill_parallel(torch.tensor([prompt], device=dev), pos, valid, kv, aux, P)
    sites = []
    for g in range(1, (max(gs) if gs else 0) + 1):
        c = P + g - 1
        if g in gs:
            fn = make_fn(bb, kv, int(cont[g - 1]), c, aux[0, c] if bb.family == "huginn" else None)
            with torch.no_grad():
                lg_fn = fn.readout(run_all(fn, fn.init())[-1])[0]
            with torch.inference_mode():
                lg_eng = run._step(tgt[g - 1: g], c, pos, valid, kv, aux)[0, 0]
            k_, s_ = kl_top1(lg_eng[None], lg_fn[None])
            s = site(fn, f"{job['model']}:{uid}:{g}")
            s.update({"g": g, "c": c, "parity": {"max_abs_logit_diff": float((lg_fn - lg_eng).abs().max()),
                                                 "kl": float(k_[0]), "top1_same": bool(s_[0])}})
            sites.append(s)
        else:
            with torch.inference_mode():
                run._step(tgt[g - 1: g], c, pos, valid, kv, aux)
    res = {"job": job, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "R": bb.R, "L": bb.n_rec, "P": P, "n": n,
           "gs": gs, "full_correct": rec["correct"], "sites": sites, "wall_s": time.time() - t0}
    out.write_text(json.dumps(res))


def make_jobs(args):
    d = Path(args.run_dir) / "e2"
    d.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(STAGE1_SEL).read_text())
    jobs = []
    for m in args.models.split(","):
        for ds in ("gsm8k", "math500"):
            for u in sel[ds][:N_PER_DS * 2:2]:  # every other item of the stratified (sorted) selection
                jobs.append({"name": f"e2__{m}__{ds}__{u.replace('/', '_').replace('.json', '')}", "model": m,
                             "dataset": ds, "uid": u})
    (d / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"{len(jobs)} E2 jobs -> {d}")


def work(args):
    d = Path(args.run_dir) / "e2"
    raw = d / "raw"
    raw.mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    log = open(d / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    jobs = json.loads((d / "jobs.json").read_text())
    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only.split(","))]
    if args.models_filter:
        jobs = [j for j in jobs if j["model"] in set(args.models_filter.split(","))]
    ctx = None
    for job in jobs:
        if (raw / f"{job['name']}.json").exists() or not claim(d / "claims", job["name"]):
            continue
        t = time.time()
        try:
            if ctx is None or ctx[0].key != job["model"]:
                ctx = None
                torch.cuda.empty_cache()
                spec, tok, model = load(job["model"], dtype=torch.bfloat16)
                for p in model.parameters():
                    p.requires_grad_(False)
                items = {ds: {it["uid"]: it for it in prepare_items(job["model"], ds, tok)[0]} for ds in ("gsm8k", "math500")}
                ctx = (spec, tok, model, make_runner(spec, tok, model), items)
            tmp = raw / f".{job['name']}.tmp"
            run_job(ctx, job, tmp, Path(args.run_dir))
            os.replace(tmp, raw / f"{job['name']}.json")
            log.write(f"done {job['name']} {time.time() - t:.1f}s\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (d / "errors").mkdir(exist_ok=True)
            (d / "errors" / f"{job['name']}.json").write_text(json.dumps({"job": job, "traceback": traceback.format_exc()}))
            log.write(f"FAILED {job['name']}\n{traceback.format_exc()}\n")
            torch.cuda.empty_cache()
    log.write("no pending jobs; exit\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-jobs", "work"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--models-filter", default="")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    make_jobs(args) if args.cmd == "make-jobs" else work(args)


if __name__ == "__main__":
    main()
