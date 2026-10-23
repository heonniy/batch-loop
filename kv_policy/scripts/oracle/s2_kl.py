"""Stage 2 (HUGINN_R32_KV_ORACLE_PLAN.md): fixed-prefix full-distribution divergence, Huginn R32 BF16.

For every frozen item the SAME prefix x = prompt + continuation is replayed in one lockstep batch
(kvp.oracle.run_rows, B1-equivalent rows, no padding) under
    full      FULL (every token keeps its per-loop K,V)
    last      LAST/convert
    average   AVERAGE/convert (secondary)
    full_alt  FULL with an independent initial latent draw (noise floor: latent + nothing else)
and at every position the full-vocabulary log-softmax of each row is compared with FULL:
    KL_t = sum_v p_F(v) [log p_F(v) - log p_X(v)]  (fp64 accumulation over fp32 log-probs)
    dNLL_t = -log p_X(x_{t+1}) + log p_F(x_{t+1});  top1_changed_t = 1[argmax p_F != argmax p_X].
Continuation sources: "full_gen" = the frozen greedy FULL/decode generation of the R32 quality run
(primary, all 200+200 items); sensitivity on the Stage-1 64-item selection: "gold" (reference
solution tokens) and "last_gen" (the LAST/convert generation of the same run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import claim, run_rows  # noqa: E402

MODEL = "huginn-0125"
BASE_RUN = "results/20261023T060125Z_huginn_r32_h200x8"
STAGE1_RUN = "results/20261023T075134Z_huginn_r32_oracle_stage1"
ROWS = ("full", "last", "average", "full_alt")


@lru_cache(maxsize=None)
def gen_records(base: Path, dataset: str, policy: str, mode: str) -> dict[str, dict]:
    out = {}
    for f in sorted((base / "raw").glob(f"{MODEL}__{dataset}__gen__{policy}-{mode}__b*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def sanitize(uid: str) -> str:
    return uid.replace("/", "_").replace(".json", "")


def make_jobs(args):
    d = Path(args.run_dir) / "stage2"
    d.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_run)
    sel = json.loads((Path(args.stage1_run) / "parity" / "selection.json").read_text())
    jobs = []
    for ds in ("gsm8k", "math500"):
        full = gen_records(base, ds, "full", "decode")
        for uid, r in full.items():
            jobs.append({"name": f"s2__full_gen__{ds}__{sanitize(uid)}", "dataset": ds, "uid": uid,
                         "source": "full_gen", "cost": r["n_gen_tokens"]})
        for uid in sel[ds]:
            for src in ("gold", "last_gen"):
                jobs.append({"name": f"s2__{src}__{ds}__{sanitize(uid)}", "dataset": ds, "uid": uid,
                             "source": src, "cost": 600 if src == "gold" else full[uid]["n_gen_tokens"]})
    jobs.sort(key=lambda j: (-j["cost"], j["name"]))
    (d / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"{len(jobs)} stage-2 jobs -> {d}")


def run_job(ctx, job, out: Path, base: Path):
    spec, tok, model, run, items = ctx
    ds, uid, src = job["dataset"], job["uid"], job["source"]
    it = items[ds][uid]
    full_rec = gen_records(base, ds, "full", "decode")[uid]
    if src == "full_gen":
        cont = full_rec["gen_token_ids"]
    elif src == "last_gen":
        cont = gen_records(base, ds, "last", "convert")[uid]["gen_token_ids"]
    else:
        cont = it["target_ids"]
    rows = [{"base": "last", "keep": "all", "uid": uid}, {"base": "last", "keep": None, "uid": uid},
            {"base": "average", "keep": None, "uid": uid}, {"base": "last", "keep": "all", "uid": uid + "#alt1"}]
    acc = {k: [] for k in ("kl", "dnll", "chg", "nll_f", "ent_f", "tgt_top1_f", "norm")}
    tgt = torch.tensor(cont, device=run.device)

    def on_step(g, logits):
        lp = torch.log_softmax(logits.float(), -1).double()
        pf = lp[0].exp()
        acc["kl"].append((pf[None] * (lp[0][None] - lp[1:])).sum(-1))           # [3]
        nll = -lp[:, tgt[g]]
        acc["dnll"].append(nll[1:] - nll[0])
        top = lp.argmax(-1)
        acc["chg"].append(top[1:] != top[0])
        acc["nll_f"].append(nll[0])
        acc["ent_f"].append(-(pf * lp[0]).sum())
        acc["tgt_top1_f"].append(top[0] == tgt[g])
        acc["norm"].append((lp.exp().sum(-1) - 1).abs().max())

    t0 = time.time()
    run_rows(run, it["prompt_ids"], cont, rows, on_step)
    st = {k: torch.stack(v).cpu() for k, v in acc.items()}
    res = {"job": job, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "wall_s": time.time() - t0,
           "P": len(it["prompt_ids"]), "n": len(cont), "rows": list(ROWS),
           "full_correct": full_rec["correct"], "full_finish": full_rec["finish_reason"],
           "full_n_gen": full_rec["n_gen_tokens"], "level": it["doc"].get("level"),
           "subject": it["doc"].get("subject"),
           "any_nan": bool(torch.isnan(st["kl"]).any() or torch.isnan(st["dnll"]).any()),
           "normalization_max_err": float(st["norm"].max()),
           "nll_full": [round(float(x), 6) for x in st["nll_f"]],
           "entropy_full": [round(float(x), 6) for x in st["ent_f"]],
           "target_is_full_top1": [bool(x) for x in st["tgt_top1_f"]]}
    for j, name in enumerate(ROWS[1:]):
        res[f"kl_{name}"] = [float(f"{x:.6g}") for x in st["kl"][:, j].tolist()]
        res[f"dnll_{name}"] = [round(x, 6) for x in st["dnll"][:, j].tolist()]
        res[f"top1chg_{name}"] = [int(x) for x in st["chg"][:, j].tolist()]
    out.write_text(json.dumps(res))


def work(args):
    d = Path(args.run_dir) / "stage2"
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
                items = {ds: {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]} for ds in ("gsm8k", "math500")}
                ctx = (spec, tok, model, make_runner(spec, tok, model), items)
            tmp = raw / f".{job['name']}.tmp"
            run_job(ctx, job, tmp, Path(args.base_run))
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
    ap.add_argument("--base-run", default=BASE_RUN)
    ap.add_argument("--stage1-run", default=STAGE1_RUN)
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()
    make_jobs(args) if args.cmd == "make-jobs" else work(args)


if __name__ == "__main__":
    main()
