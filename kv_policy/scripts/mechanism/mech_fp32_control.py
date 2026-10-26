"""Precision control for the mechanism pilot: rerun C (FULL vs LAST hidden error, logits KL) and D
(single-call interventions) in FP32 on a few short pilot prompts. If the ~1.4% loop-32 error plateau
seen in BF16 for every intervention loop is a BF16 rounding floor of the recurrence, it must drop in FP32.
Same prompts, prefix, latents and intervention grid as mech_run.py; writes mechanism/fp32_control/*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from kvp.engine import Runner  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.mechanism import C_METRICS, TraceHuginn, Tracer, intervention_pass  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import run_rows  # noqa: E402
from mech_run import D_LOOPS, D_TOKENS, MODEL, gen_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-run", default="results/20261023T060125Z_huginn_r32_h200x8")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--uids", required=True, help="comma-separated dataset:uid")
    args = ap.parse_args()
    out = Path(args.run_dir) / "mechanism" / f"{args.dtype}_control"
    out.mkdir(parents=True, exist_ok=True)
    spec, tok, model = load(MODEL, dtype=getattr(torch, args.dtype))
    run = make_runner(spec, tok, model)
    trun = Runner(TraceHuginn(run.bb), run.pad_id)
    items = {ds: {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]} for ds in ("gsm8k", "math500")}
    for spec_ in args.uids.split(","):
        ds, uid = spec_.split(":", 1)
        it, rec = items[ds][uid], gen_records(Path(args.base_run), ds)[uid]
        prompt, cont = it["prompt_ids"], rec["gen_token_ids"]
        n = len(cont)
        t0 = time.time()
        tr = Tracer(trun.bb, exp_b=False)
        trun.bb.tracer = tr
        kl = []

        def on_step(g, lg):
            lp = torch.log_softmax(lg.double(), -1)
            kl.append(float((lp[0].exp() * (lp[0] - lp[1])).sum()))
            if g > 0:
                tr.end_step()

        rows = [{"base": "last", "keep": "all", "uid": uid}, {"base": "last", "keep": None, "uid": uid}]
        run_rows(trun, prompt, cont, rows, on_step)
        trun.bb.tracer = None
        c = torch.stack(tr.c_steps).numpy()
        steps = sorted({int(round(1 + k * (n - 2) / (D_TOKENS - 1))) for k in range(D_TOKENS)})
        grid = [(j, r - 1) for j in range(trun.bb.n_core) for r in D_LOOPS]
        d = intervention_pass(trun, prompt, cont, uid, set(steps), grid)
        iE = C_METRICS.index("err_out")
        res = {"dataset": ds, "uid": uid, "dtype": args.dtype, "n": n, "source_sha": SOURCE_SHA, "gpu": gpu_identity(),
               "c_err_out_last_layer_by_loop": np.nanmean(c[:, :, -1, iE], 0).tolist(),
               "kl_mean_decode": float(np.mean(kl[1:])), "d_records": d, "wall_s": time.time() - t0}
        (out / f"{ds}__{uid.replace('/', '_').replace('.json', '')}.json").write_text(json.dumps(res))
        print("done", uid, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
