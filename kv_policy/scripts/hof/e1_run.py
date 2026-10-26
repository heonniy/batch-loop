"""E1 (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md): training-depth signal characterisation under FULL.

One job = (model, dataset, item); B=1 FULL (keep-all) teacher-forced pass over the frozen gold target.
  A  per-depth gold next-token NLL / entropy / top-1 at every target position (Ouro: native per-step
     lm_head(norm h_r); Huginn: its coda readout of z_r, see kvp.depthprobe)
  B  Ouro official ACT exit distribution p_{i,r} at every prompt and target position
  C  cross-depth K/V geometry (all loop pairs for Ouro R=4; loops {1,2,4,8,16,24,31,32} for Huginn)
  D  Ouro only, Stage-1 64-item selection: effect on the next W gold predictions of substituting the
     historical K/V of ONE source token i (all its per-step planes -> its terminal plane) inside an
     otherwise FULL cache, vs per-token features (exit pdf, depth NLL, KV distance). Source index i =
     cache column of token x_i; prediction positions are the W targets after the cut (all later than i).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.depthprobe import ProbeHuginn, ProbeOuro, ouro_exit_pdf, pairwise_kv_geometry  # noqa: E402
from kvp.engine import Runner  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import MixedKV, claim, run_rows  # noqa: E402

MODELS = ("huginn-0125", "ouro-1.4b", "ouro-2.6b")
STAGE1_SEL = "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json"
HUGINN_GEO_LOOPS = [0, 1, 3, 7, 15, 23, 30, 31]
D_W, D_CUT_MAX, D_SRC = 32, 64, 32
D_BATCH = 8


def r6(x):
    return [float(f"{v:.6g}") for v in x]


def run_job(ctx, job, out: Path):
    spec, tok, model, run, prun, items = ctx
    bb = prun.bb
    ds, uid = job["dataset"], job["uid"]
    it = items[ds][uid]
    prompt, cont = it["prompt_ids"], it["target_ids"]
    P, n = len(prompt), len(cont)
    T = P + n
    dev = run.device
    kv = MixedKV(["last"], torch.ones(1, T, dtype=torch.bool), bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D, run.dtype, dev)
    pos, valid = torch.arange(T, device=dev)[None], torch.ones(1, T, dtype=torch.bool, device=dev)
    aux = bb.make_aux([uid], [0], T)
    tgt = torch.tensor(cont, device=dev)
    ids_all = torch.tensor(prompt + cont, device=dev)
    nll, ent, top1, pdf_rows = [], [], [], []
    res = {"job": job, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "P": P, "n": n, "R": bb.R,
           "level": it["doc"].get("level")}
    t0 = time.time()

    def depth_stats(lg, target):
        lp = torch.log_softmax(lg.double(), -1)  # [R, ..., V]
        nl = -lp.gather(-1, target.expand(*lp.shape[:-1])[..., None])[..., 0]
        en = -(lp.exp() * lp).sum(-1)
        return nl, en, lp.argmax(-1) == target

    bb.record = True
    with torch.inference_mode():
        lg = prun._prefill_parallel(ids_all[None, :P], pos, valid, kv, aux, P)
        if bb.family == "ouro":
            dl, gates = bb.depth_logits([s[0] for s in bb.states])  # [R, P, V], [R, P]
            res["prompt_exit_pdf"] = [r6(x) for x in ouro_exit_pdf(gates).T.tolist()]  # [P][R]
            nxt = ids_all[1:P + 1]
            a, b_, c_ = depth_stats(dl, nxt)  # predicting x_{i+1} at every prompt position i
            res["prompt_depth_nll"] = [r6(x) for x in a.T.tolist()]  # [P][R]
            first = (a[:, -1], b_[:, -1], c_[:, -1])
            del dl
        else:
            st = [s[0, -1:][None] for s in bb.states]  # R x [1, 1, E] at the last prompt position
            first = None
        for g in range(n):
            if g == 0 and first is not None:
                a, b_, c_ = first
                pdf_rows.append(res["prompt_exit_pdf"][-1])
            else:
                if g > 0:
                    lg_step = prun._step(tgt[g - 1: g], P + g - 1, pos, valid, kv, aux)
                if bb.family == "ouro":
                    dl, gates = bb.depth_logits([s[0, 0] for s in bb.states])  # [R, V], [R]
                    pdf_rows.append(r6(ouro_exit_pdf(gates).tolist()))
                else:
                    c = P + g - 1 if g > 0 else P - 1
                    if g == 0:
                        dl = depth_logits_prefill_last(bb, kv, st, P)
                    else:
                        dl = bb.depth_logits_decode(bb.states, kv, c)
                a, b_, c_ = depth_stats(dl, tgt[g])
            nll.append(a.float())
            ent.append(b_.float())
            top1.append(c_)
    bb.record = False
    N = torch.stack(nll, 1).cpu()  # [R, n]
    res["depth_nll"] = [r6(x) for x in N.tolist()]
    res["depth_entropy"] = [r6(x) for x in torch.stack(ent, 1).cpu().tolist()]
    res["depth_top1"] = torch.stack(top1, 1).cpu().int().tolist()
    if bb.family == "ouro":
        res["target_exit_pdf"] = pdf_rows  # [n][R] (pdf of the position that predicts target g)
    loops = list(range(bb.R)) if bb.family == "ouro" else HUGINN_GEO_LOOPS
    res["geo_loops"] = [l + 1 for l in loops]
    geo = {}
    for seg, sl in (("prompt", slice(0, P)), ("cont", slice(P, T - 1))):
        for k0 in range(sl.start, sl.stop, 256):  # chunk columns to bound memory, then weight
            sub = slice(k0, min(k0 + 256, sl.stop))
            gg = pairwise_kv_geometry(kv, 0, T, loops, {seg: sub})
            w = sub.stop - sub.start
            for k, v in gg.items():
                geo[k] = geo.get(k, 0) + v * w
        for k in list(geo):
            if k.endswith("|" + seg):
                geo[k] = geo[k] / max(1, sl.stop - sl.start)
    res["kv_geometry"] = {k: v.tolist() for k, v in geo.items()}
    if bb.family == "ouro" and job.get("with_D"):
        res["D"] = experiment_d(prun, kv, prompt, cont, uid, res)
    res["wall_s"] = time.time() - t0
    out.write_text(json.dumps(res))


def depth_logits_prefill_last(bb, kv, states, P):
    """Huginn depth readout for the last prompt position (predicting the first target)."""
    return bb.depth_logits_decode(states, kv, P - 1)


def experiment_d(prun, kv_full, prompt, cont, uid, res):
    bb = prun.bb
    P, n = len(prompt), len(cont)
    if n < 16 + D_W:
        return {"skipped": f"n={n} < {16 + D_W}"}
    cut = min(D_CUT_MAX, n - D_W)
    H = P + cut
    rng = random.Random(f"E1D:{uid}")
    recent = list(range(H - 8, H))
    cont_cols = [c for c in range(P, H) if c not in recent]
    prm = rng.sample(range(P), 16)
    cs = rng.sample(cont_cols, min(8, len(cont_cols)))
    srcs = sorted(set(prm + cs + recent))
    rest = [c for c in range(H) if c not in srcs]
    srcs += rng.sample(rest, D_SRC - len(srcs)) if len(srcs) < D_SRC else []
    srcs = sorted(srcs)[:D_SRC]
    seq = cont[: cut + D_W]
    T = P + len(seq)
    tgt = torch.tensor(seq, device=prun.device)
    # per-source KV distance (layer-mean of ||K_R - K_1|| / ||K_R||, V likewise) from the FULL pass
    kvd = {}
    for c in srcs:
        dk = torch.stack([(kb[-1, 0, :, c] - kb[0, 0, :, c]).float().norm(dim=-1).mean() / kb[-1, 0, :, c].float().norm(dim=-1).mean()
                          for kb in kv_full.rk])
        dv = torch.stack([(vb[-1, 0, :, c] - vb[0, 0, :, c]).float().norm(dim=-1).mean() / vb[-1, 0, :, c].float().norm(dim=-1).mean()
                          for vb in kv_full.rv])
        kvd[c] = (float(dk.mean()), float(dv.mean()))
    specs = [("dup_full", None)] + [(f"src:{c}", c) for c in srcs] + [("all_terminal", "all")]
    lpF = {}
    out = []
    for i in range(0, len(specs), D_BATCH - 1):
        chunk = specs[i: i + D_BATCH - 1]
        rows = [{"base": "last", "keep": "all", "uid": uid}]
        for _, c in chunk:
            if c is None:
                keep = "all"
            elif c == "all":
                keep = None
            else:
                keep = [True] * T
                keep[c] = False
            rows.append({"base": "last", "keep": keep, "uid": uid})
        kl, dn = [[] for _ in rows], [[] for _ in rows]

        def on_step(g, lg):
            if g < cut:
                return
            lp = torch.log_softmax(lg.double(), -1)
            ref = lp[0]
            for b in range(len(rows)):
                kl[b].append(float((ref.exp() * (ref - lp[b])).sum()))
                dn[b].append(float(-lp[b, tgt[g]] + ref[tgt[g]]))

        run_rows(prun, prompt, seq, rows, on_step)
        for b, (name, c) in enumerate(chunk, start=1):
            rec = {"name": name, "kl_mean": sum(kl[b]) / len(kl[b]), "dnll_mean": sum(dn[b]) / len(dn[b])}
            if isinstance(c, int):
                rec.update({"col": c, "in_prompt": c < P, "dist_to_window": H - c,
                            "kv_dist_K": kvd[c][0], "kv_dist_V": kvd[c][1]})
                if c < P:
                    rec["exit_pdf"] = res["prompt_exit_pdf"][c]
                    rec["depth_nll"] = res["prompt_depth_nll"][c]  # predicts x_{c+1} (cont[0] for c = P-1)
                else:
                    g_src = c - P + 1  # column c holds cont[c-P]; its next-token target is cont[c-P+1] = target g_src
                    rec["exit_pdf"] = res["target_exit_pdf"][g_src] if g_src < len(res["target_exit_pdf"]) else None
                    rec["depth_nll"] = [res["depth_nll"][r][g_src] for r in range(len(res["depth_nll"]))] if g_src < n else None
            out.append(rec)
    return {"cut": cut, "W": D_W, "H": H, "sources": out}


def make_jobs(args):
    d = Path(args.run_dir) / "e1"
    d.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(STAGE1_SEL).read_text())
    from transformers import AutoTokenizer
    from kvp.specs import MODELS as SPECS

    jobs = []
    for m in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(SPECS[m].repo, revision=SPECS[m].revision, trust_remote_code=True)
        for ds in ("gsm8k", "math500"):
            for it in prepare_items(m, ds, tok)[0]:
                cost = (len(it["target_ids"]) * (32 if m.startswith("huginn") else 4) * (2 if m == "ouro-2.6b" else 1))
                jobs.append({"name": f"e1__{m}__{ds}__{it['uid'].replace('/', '_').replace('.json', '')}", "model": m,
                             "dataset": ds, "uid": it["uid"], "with_D": m.startswith("ouro") and it["uid"] in sel[ds],
                             "cost": cost + (100000 if m.startswith("ouro") and it["uid"] in sel[ds] else 0)})
    jobs.sort(key=lambda j: (j["model"], -j["cost"]))
    (d / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"{len(jobs)} E1 jobs -> {d}")


def work(args):
    d = Path(args.run_dir) / "e1"
    raw = d / "raw"
    raw.mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    log = open(d / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    jobs = json.loads((d / "jobs.json").read_text())
    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only.split(","))]
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
                run = make_runner(spec, tok, model)
                probe = (ProbeOuro if spec.family == "ouro" else ProbeHuginn)(run.bb)
                items = {ds: {it["uid"]: it for it in prepare_items(job["model"], ds, tok)[0]} for ds in ("gsm8k", "math500")}
                ctx = (spec, tok, model, run, Runner(probe, run.pad_id), items)
            tmp = raw / f".{job['name']}.tmp"
            run_job(ctx, job, tmp)
            os.replace(tmp, raw / f"{job['name']}.json")
            log.write(f"done {job['name']} {time.time() - t:.1f}s\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (d / "errors").mkdir(exist_ok=True)
            (d / "errors" / f"{job['name']}.json").write_text(json.dumps({"job": job, "traceback": traceback.format_exc()}))
            log.write(f"FAILED {job['name']}\n{traceback.format_exc()}\n")
            if ctx is not None:
                ctx[4].bb.record = False
            torch.cuda.empty_cache()
    log.write("no pending jobs; exit\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-jobs", "work"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    make_jobs(args) if args.cmd == "make-jobs" else work(args)


if __name__ == "__main__":
    main()
