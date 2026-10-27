"""Step 2 (OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md): recovery vs persistent-context mismatch, fixed prefix.

One job = (model, dtype, item). Prefix = the model's own frozen greedy FULL generation (E0 receipts;
Huginn R32 from its R32 run, first 512 tokens). Rows in one lockstep batch (kvp.rowbanks, same latent):
  FULL          every bank per-step
  SWAP_L0R1     FULL except the (layer 0, recurrence 1) bank reads terminal history (persistent, all tokens)
  LAST          LAST-CONVERT
  PROT_L0R1     LAST-CONVERT except (layer 0, recurrence 1) keeps per-step history
  PROT_R1ALL    LAST-CONVERT except every layer's recurrence-1 bank
The one-call swap (plan condition A) is the E2 reference (results/20261026T080106Z_.../analysis/e2_*).
Per generated position: KL(FULL || row), top-1 agreement, ΔNLL of the FULL token; per recurrent step:
relative hidden-state error of the current token vs the FULL row (Ouro: normalised state after each
UT step; Huginn: latent after each loop). Teacher forcing on a frozen prefix is NOT a rollout; rollout
behaviour comes from the Step-1 generations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.depthprobe import ProbeHuginn, ProbeOuro  # noqa: E402
from kvp.engine import Runner  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import claim  # noqa: E402
from kvp.rowbanks import run_bank_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = {"ouro-1.4b": "results/20261026T080106Z_huginn_ouro_full_last_convert",
            "ouro-2.6b": "results/20261026T080106Z_huginn_ouro_full_last_convert",
            "huginn-0125": "results/20261023T060125Z_huginn_r32_h200x8"}
SEL = "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json"
ROWS = ("FULL", "SWAP_L0R1", "LAST", "PROT_L0R1", "PROT_R1ALL")
HUGINN_MAX = 512


def full_receipts(model, ds):
    out = {}
    for f in sorted((ROOT / RECEIPTS[model] / "raw").glob(f"{model}__{ds}__gen__full-decode__b*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def row_sets(L, R):
    allb = frozenset((l, r) for l in range(L) for r in range(R - 1))
    return [allb, allb - {(0, 0)}, frozenset(), frozenset({(0, 0)}), frozenset((l, 0) for l in range(L))]


def run_job(ctx, job, out):
    spec, tok, model, prun, items = ctx
    bb = prun.bb
    ds, uid = job["dataset"], job["uid"]
    it = items[ds][uid]
    rec = full_receipts(job["model"], ds)[uid]
    cont = rec["gen_token_ids"][: HUGINN_MAX if spec.family == "huginn" else None]
    sets = row_sets(bb.n_rec, bb.R)
    B = len(sets)
    kl, top1, dnll, herr = [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)]
    tgt = torch.tensor(cont, device=prun.device)
    bb.record = True

    def on_step(g, lg):
        lp = torch.log_softmax(lg.double(), -1)
        ref = lp[0]
        k = (ref.exp()[None] * (ref[None] - lp)).sum(-1)
        a = lp.argmax(-1) == ref.argmax()
        d = -lp[:, tgt[g]] + ref[tgt[g]]
        if g > 0:  # per-step hidden error of the current (decode) token vs the FULL row
            st = torch.stack([s[:, -1].float() for s in bb.states])  # [R, B, E]
            e = (st - st[:, :1]).norm(dim=-1) / (st[:, :1].norm(dim=-1) + 1e-12)  # [R, B]
        for b in range(B):
            kl[b].append(float(k[b]))
            top1[b].append(bool(a[b]))
            dnll[b].append(float(d[b]))
            if g > 0:
                herr[b].append([float(x) for x in e[:, b]])

    t0 = time.time()
    run_bank_rows(prun, it["prompt_ids"], cont, sets, uid, on_step)
    bb.record = False
    res = {"job": job, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "R": bb.R, "L": bb.n_rec, "P": len(it["prompt_ids"]),
           "n": len(cont), "full_correct": rec["correct"], "rows": list(ROWS), "wall_s": time.time() - t0}
    for b, name in enumerate(ROWS):
        res[name] = {"kl": [float(f"{x:.6g}") for x in kl[b]], "top1": top1[b], "dnll": [round(x, 6) for x in dnll[b]],
                     "hidden_err_by_step_mean": [sum(h[s] for h in herr[b]) / max(1, len(herr[b])) for s in range(bb.R)],
                     "hidden_err_final": [round(h[-1], 6) for h in herr[b]]}
    out.write_text(json.dumps(res))


def make_jobs(args):
    d = Path(args.run_dir) / "step2"
    d.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / SEL).read_text())
    jobs = []
    for m in ("ouro-1.4b", "ouro-2.6b", "huginn-0125"):
        for ds in ("gsm8k", "math500"):
            for i, u in enumerate(sel[ds]):
                for dt in ("bfloat16", "float32"):
                    if dt == "float32" and (i >= 4 or m.startswith("huginn")):
                        continue
                    jobs.append({"name": f"s2__{m}__{dt}__{ds}__{u.replace('/', '_').replace('.json', '')}", "model": m,
                                 "dtype": dt, "dataset": ds, "uid": u})
    (d / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"{len(jobs)} step-2 jobs -> {d}")


def work(args):
    d = Path(args.run_dir) / "step2"
    raw = d / "raw"
    raw.mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    log = open(d / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    jobs = sorted(json.loads((d / "jobs.json").read_text()), key=lambda j: (j["model"], j["dtype"]))
    ctx, key = None, None
    for job in jobs:
        if (raw / f"{job['name']}.json").exists() or not claim(d / "claims", job["name"]):
            continue
        t = time.time()
        try:
            if key != (job["model"], job["dtype"]):
                ctx = None
                torch.cuda.empty_cache()
                spec, tok, model = load(job["model"], dtype=getattr(torch, job["dtype"]))
                run = make_runner(spec, tok, model)
                probe = (ProbeOuro if spec.family == "ouro" else ProbeHuginn)(run.bb)
                items = {ds: {it["uid"]: it for it in prepare_items(job["model"], ds, tok)[0]} for ds in ("gsm8k", "math500")}
                ctx, key = (spec, tok, model, Runner(probe, run.pad_id), items), (job["model"], job["dtype"])
            tmp = raw / f".{job['name']}.tmp"
            run_job(ctx, job, tmp)
            os.replace(tmp, raw / f"{job['name']}.json")
            log.write(f"done {job['name']} {time.time() - t:.1f}s\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (d / "errors").mkdir(exist_ok=True)
            (d / "errors" / f"{job['name']}.json").write_text(json.dumps({"job": job, "traceback": traceback.format_exc()}))
            log.write(f"FAILED {job['name']}\n{traceback.format_exc()}\n")
            if ctx is not None:
                ctx[3].bb.record = False
            torch.cuda.empty_cache()
    log.write("no pending jobs; exit\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-jobs", "work"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()
    make_jobs(args) if args.cmd == "make-jobs" else work(args)


if __name__ == "__main__":
    main()
