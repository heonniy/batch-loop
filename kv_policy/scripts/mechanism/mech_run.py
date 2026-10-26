"""Huginn R32 attention-mechanism pilot (HUGINN_R32_ATTENTION_MECHANISM_PLAN.md), one job per prompt.

Items: the Stage-1 stratified 32 GSM8K + 32 MATH-500 selection; prefix = prompt + the frozen greedy
FULL/decode continuation of the R32 quality run (same uid/position initial latents everywhere).

pass 1  rows [FULL, LAST/convert] in lockstep, traced (kvp.mechanism.TraceHuginn):
        B: fixed-query counterfactuals on the FULL row at every decode token, loop, core layer
        C: FULL-vs-LAST observed errors per loop / layer, prelude & coda boundaries, logits KL/argmax
        A: KV geometry of the FULL row's per-loop history after the pass (all prompt+continuation columns)
        hook check (n <= 256): identical untraced pass; max |logit difference| must be 0
pass 2  D: B=1 FULL pass with single-call interventions at 8 continuation tokens,
        loops {1,2,4,8,16,24,31,32} x 4 core layers.
Writes raw/<name>.json (aggregates + D records) and raw/<name>.npz (per-step B/C arrays, per-column
geometry means over heads, per-head geometry summaries).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kvp.engine import Runner  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.mechanism import B_METRICS, C_METRICS, TraceHuginn, Tracer, intervention_pass, kv_geometry  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import claim, run_rows  # noqa: E402

MODEL = "huginn-0125"
BASE_RUN = "results/20261023T060125Z_huginn_r32_h200x8"
STAGE1_RUN = "results/20261023T075134Z_huginn_r32_oracle_stage1"
D_LOOPS = (1, 2, 4, 8, 16, 24, 31, 32)
D_TOKENS = 8


def gen_records(base: Path, dataset: str) -> dict[str, dict]:
    out = {}
    for f in sorted((base / "raw").glob(f"{MODEL}__{dataset}__gen__full-decode__b*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def make_jobs(args):
    d = Path(args.run_dir) / "mechanism"
    d.mkdir(parents=True, exist_ok=True)
    sel = json.loads((Path(args.stage1_run) / "parity" / "selection.json").read_text())
    jobs = []
    for ds, uids in sel.items():
        recs = gen_records(Path(args.base_run), ds)
        for u in uids:
            jobs.append({"name": f"mech__{ds}__{u.replace('/', '_').replace('.json', '')}", "dataset": ds, "uid": u,
                         "cost": recs[u]["n_gen_tokens"]})
    jobs.sort(key=lambda j: -j["cost"])
    (d / "jobs.json").write_text(json.dumps(jobs, indent=0))
    (d / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"{len(jobs)} mechanism jobs -> {d}")


def run_job(ctx, job, out_json: Path, out_npz: Path, base: Path):
    spec, tok, model, run, trun, items = ctx
    bb = trun.bb
    ds, uid = job["dataset"], job["uid"]
    it = items[ds][uid]
    rec = gen_records(base, ds)[uid]
    prompt, cont = it["prompt_ids"], rec["gen_token_ids"]
    P, n = len(prompt), len(cont)
    rows = [{"base": "last", "keep": "all", "uid": uid}, {"base": "last", "keep": None, "uid": uid}]
    t0 = time.time()

    # ---- pass 1: B + C (+ A afterwards)
    tr = Tracer(bb)
    bb.tracer = tr
    kl, same, logits_keep = [], [], []
    check = n <= 256

    def on_step(g, lg):
        lp = torch.log_softmax(lg.float(), -1).double()
        kl.append((lp[0].exp() * (lp[0] - lp[1])).sum())
        same.append(lp[0].argmax() == lp[1].argmax())
        if check:
            logits_keep.append(lg.clone())
        if g > 0:
            tr.end_step()

    kv = run_rows(trun, prompt, cont, rows, on_step, keep_kv=True)
    bb.tracer = None
    hook_check = None
    if check:
        ref = []
        run_rows(run, prompt, cont, rows, lambda g, lg: ref.append(lg.clone()))
        hook_check = float((torch.stack(ref) - torch.stack(logits_keep)).abs().max())
    t1 = time.time()
    upto = P + n - 1  # columns whose per-loop K,V were written
    geo = kv_geometry(kv, 0, upto)  # [n_core, R, C, H]
    del kv
    torch.cuda.empty_cache()
    seg = {"prompt": slice(0, P), "cont": slice(P, upto)}
    geo_summary = {}
    for m, x in geo.items():
        for sname, sl in seg.items():
            part = x[:, :, sl]  # [n_core, R, C_seg, H]
            if part.shape[2] == 0:
                continue
            flat = part.reshape(part.shape[0], part.shape[1], -1)
            ok = ~torch.isnan(flat)
            geo_summary[f"{m}|{sname}|mean"] = torch.nanmean(flat, -1).cpu().tolist()
            qv = torch.tensor([0.5, 0.9], device=flat.device)
            qs = torch.stack([torch.quantile(flat[j, s][ok[j, s]], qv) if ok[j, s].any()
                              else torch.full((2,), float("nan"), device=flat.device)
                              for j in range(flat.shape[0]) for s in range(flat.shape[1])])
            geo_summary[f"{m}|{sname}|median"] = qs[:, 0].view(flat.shape[0], flat.shape[1]).cpu().tolist()
            geo_summary[f"{m}|{sname}|p90"] = qs[:, 1].view(flat.shape[0], flat.shape[1]).cpu().tolist()
    geo_col = {m: torch.nanmean(x, -1).cpu().numpy().astype(np.float32) for m, x in geo.items()}  # [n_core, R, C]
    geo_head = {f"{m}|{sname}": torch.nanmean(x[:, :, sl], 2).cpu().numpy().astype(np.float32)
                for m, x in geo.items() for sname, sl in seg.items() if x[:, :, sl].shape[2]}  # [n_core, R, H]
    del geo
    torch.cuda.empty_cache()

    # ---- pass 2: D
    steps = sorted({int(round(1 + k * (n - 2) / (D_TOKENS - 1))) for k in range(D_TOKENS)}) if n >= D_TOKENS + 1 else []
    grid = [(j, r - 1) for j in range(bb.n_core) for r in D_LOOPS]
    d_recs = intervention_pass(trun, prompt, cont, uid, set(steps), grid) if steps else []
    t2 = time.time()

    b = torch.stack(tr.b_steps).numpy().astype(np.float32)  # [n-1, R, n_core, |B|]
    c = torch.stack(tr.c_steps).numpy().astype(np.float32)
    plain_keys = sorted(tr.plain_steps[0]) if tr.plain_steps else []
    plain = np.array([[p[k] for k in plain_keys] for p in tr.plain_steps], dtype=np.float32)
    np.savez_compressed(out_npz, b=b, c=c, plain=plain, kl=torch.stack(kl).cpu().numpy(),
                        argmax_same=torch.stack(same).cpu().numpy(),
                        **{f"geo_col__{k}": v for k, v in geo_col.items()},
                        **{f"geo_head__{k}": v for k, v in geo_head.items()})
    res = {"job": job, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "P": P, "n": n,
           "full_correct": rec["correct"], "full_finish": rec["finish_reason"], "level": it["doc"].get("level"),
           "b_metrics": list(B_METRICS), "c_metrics": list(C_METRICS), "plain_keys": plain_keys,
           "hook_check_max_abs_logit_diff": hook_check,
           "nan_b": int(np.isnan(b).sum()), "nan_c": int(np.isnan(c).sum()),
           "kl_mean": float(torch.stack(kl).mean()), "argmax_same_rate": float(torch.stack(same).float().mean()),
           "geo_summary": geo_summary, "d_steps": steps, "d_records": d_recs,
           "npz_sha256": None, "wall_s": {"pass1_and_geometry": t1 - t0, "pass2": t2 - t1}}
    res["npz_sha256"] = hashlib.sha256(Path(out_npz).read_bytes()).hexdigest()
    out_json.write_text(json.dumps(res))


def work(args):
    d = Path(args.run_dir) / "mechanism"
    raw = d / "raw"
    raw.mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    log = open(d / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    jobs = json.loads((d / "jobs.json").read_text())
    ctx = None
    for job in jobs:
        if (raw / f"{job['name']}.json").exists() or not claim(d / "claims", job["name"]):
            continue
        t = time.time()
        try:
            if ctx is None:
                spec, tok, model = load(MODEL, dtype=torch.bfloat16)
                run = make_runner(spec, tok, model)
                trun = Runner(TraceHuginn(run.bb), run.pad_id)
                items = {ds: {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]} for ds in ("gsm8k", "math500")}
                ctx = (spec, tok, model, run, trun, items)
            tmp_j, tmp_n = raw / f".{job['name']}.tmp.json", raw / f".{job['name']}.tmp.npz"
            run_job(ctx, job, tmp_j, tmp_n, Path(args.base_run))
            os.replace(tmp_n, raw / f"{job['name']}.npz")
            os.replace(tmp_j, raw / f"{job['name']}.json")
            log.write(f"done {job['name']} {time.time() - t:.1f}s\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (d / "errors").mkdir(exist_ok=True)
            (d / "errors" / f"{job['name']}.json").write_text(json.dumps({"job": job, "traceback": traceback.format_exc()}))
            log.write(f"FAILED {job['name']}\n{traceback.format_exc()}\n")
            if ctx is not None:
                ctx[4].bb.tracer = None
            torch.cuda.empty_cache()
    log.write("no pending jobs; exit\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-jobs", "work"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-run", default=BASE_RUN)
    ap.add_argument("--stage1-run", default=STAGE1_RUN)
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()
    make_jobs(args) if args.cmd == "make-jobs" else work(args)


if __name__ == "__main__":
    main()
