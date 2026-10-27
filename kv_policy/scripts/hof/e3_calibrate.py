"""E3 calibration: per-bank protection gain on prompts DISJOINT from the frozen evaluation items.

For one Ouro model: 16 GSM8K-test + 16 MATH-500 items outside the frozen 200-item subsets (seeded),
rendered with the same protocol. Teacher-forced gold NLL (token-weighted) under LAST-CONVERT, FULL,
and LAST + protection of exactly one (layer l, step r < R) bank, for every bank. Gain_b =
NLL(LAST) - NLL(LAST + b). Cumulative top-k sets are then re-scored as wholes (gains need not add).
Writes <run>/e3/calibration_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.banks import REGISTRY  # noqa: E402
from kvp.data import load_gsm8k_test, load_math500_test, load_subset  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.prompts import encode_prompt, gold_target, protocol, render  # noqa: E402

N_CAL = 16
BATCH = 8


def calib_items(model_key, family, tok):
    out = []
    for ds, loader in (("gsm8k", load_gsm8k_test), ("math500", load_math500_test)):
        docs, _ = loader()
        frozen = set(load_subset(ds)[1]["uids"])
        pool = [d for d in docs if d["uid"] not in frozen]
        rng = random.Random(f"e3calib:{ds}")
        proto = protocol(family, ds)
        for d in rng.sample(pool, N_CAL):
            text = render(d, proto, tok)
            ids = encode_prompt(text, tok)
            tgt = gold_target(d, proto)
            whole = tok.encode(text + tgt, add_special_tokens=False)
            t_ids = whole[len(ids):] if whole[: len(ids)] == ids else tok.encode(tgt, add_special_tokens=False)
            out.append({"dataset": ds, "uid": d["uid"], "prompt_ids": ids, "target_ids": t_ids[:256]})
    return out


def score(run, items, policy, mode):
    tot, ntok = 0.0, 0
    per_ds = {}
    for ds in ("gsm8k", "math500"):
        its = sorted([x for x in items if x["dataset"] == ds], key=lambda x: -len(x["prompt_ids"]))
        s_ds, n_ds = 0.0, 0
        for i in range(0, len(its), BATCH):
            b = its[i: i + BATCH]
            res = run.score([x["prompt_ids"] for x in b], [x["target_ids"] for x in b], [x["uid"] for x in b], policy, mode)
            for lp in res.target_logprobs:
                s_ds += -sum(lp)
                n_ds += len(lp)
        per_ds[ds] = s_ds / n_ds
        tot += s_ds
        ntok += n_ds
    return tot / ntok, per_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    spec, tok, model = load(args.model, dtype=torch.bfloat16)
    run = make_runner(spec, tok, model)
    items = calib_items(args.model, spec.family, tok)
    R, L = run.bb.R, run.bb.n_rec
    t0 = time.time()
    out = {"model": args.model, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "R": R, "L": L,
           "items": [{k: x[k] for k in ("dataset", "uid")} | {"prompt_tokens": len(x["prompt_ids"]), "target_tokens": len(x["target_ids"])}
                     for x in items]}
    out["nll_last"], out["nll_last_by_ds"] = score(run, items, "last", "convert")
    out["nll_full"], out["nll_full_by_ds"] = score(run, items, "full", "decode")
    gains = []
    for l in range(L):
        for r in range(R - 1):
            name = f"bankcal{l}x{r}"
            REGISTRY[(args.model, name)] = frozenset({(l, r)})
            nll, _ = score(run, items, name, "convert")
            gains.append({"layer": l, "step": r, "nll": nll, "gain": out["nll_last"] - nll})
    gains.sort(key=lambda g: -g["gain"])
    out["single_bank"] = gains
    cum = []
    for k in sorted({1, 2, 4, L // 4, L // 2, L}):
        cells = frozenset((g["layer"], g["step"]) for g in gains[:k])
        REGISTRY[(args.model, f"bankcum{k}")] = cells
        nll, by = score(run, items, f"bankcum{k}", "convert")
        cum.append({"k": k, "nll": nll, "by_ds": by, "cells": sorted(cells)})
    first = frozenset((l, 0) for l in range(L))
    REGISTRY[(args.model, "bankfirstcal")] = first
    out["first_loop_nll"], out["first_loop_by_ds"] = score(run, items, "bankfirstcal", "convert")
    out["cumulative_topk"] = cum
    out["wall_s"] = time.time() - t0
    d = Path(args.run_dir) / "e3"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"calibration_{args.model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("nll_last", "nll_full", "first_loop_nll")}), [(c["k"], round(c["nll"], 4)) for c in cum],
          gains[:5])


if __name__ == "__main__":
    main()
