"""E0 step 5: audit discordant FULL/LAST-CONVERT items by repeating them at B1 in BF16 and FP32.

For one (model, dtype): pick up to K FULL-only-correct and K LAST-only-correct items per dataset from
the batched E0 receipts (seeded), regenerate both policies at batch size 1, grade with the same
evaluator, and record whether correctness / tokens match the batched receipt. Writes
<run>/repeats/<model>__<dtype>.json; `--report` collects them into analysis/e0_repeats.md.
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

from kvp.grading import grade, truncate_at_stops  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import eos_ids, load, make_runner  # noqa: E402
from kvp.prompts import protocol  # noqa: E402
from e0_analyze import HUGINN_RUN, load as load_receipts  # noqa: E402

K = 6


def pick(rec, ds):
    uids = sorted({u for (d, p, u) in rec if d == ds and p == "full-decode"})
    fo = [u for u in uids if rec[(ds, "full-decode", u)]["correct"] and not rec[(ds, "last-convert", u)]["correct"]]
    lo = [u for u in uids if rec[(ds, "last-convert", u)]["correct"] and not rec[(ds, "full-decode", u)]["correct"]]
    rng = random.Random(f"e0repeat:{ds}")
    return rng.sample(fo, min(K, len(fo))) + rng.sample(lo, min(K, len(lo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    if args.report:
        return report(run)
    src = Path(__file__).resolve().parents[2] / HUGINN_RUN if args.model.startswith("huginn") else run
    rec = load_receipts(src, args.model, "gen")
    spec, tok, model = load(args.model, dtype=getattr(torch, args.dtype))
    runner = make_runner(spec, tok, model)
    eos = eos_ids(spec, tok, model)
    eos_s = sorted({tok.decode([e], skip_special_tokens=False) for e in eos})
    out = {"model": args.model, "dtype": args.dtype, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "items": []}
    t0 = time.time()
    for ds in ("gsm8k", "math500"):
        items = {it["uid"]: it for it in prepare_items(args.model, ds, tok)[0]}
        proto = protocol(spec.family, ds)
        stops = tuple(proto.until) + tuple(eos_s)
        for u in pick(rec, ds):
            it = items[u]
            row = {"dataset": ds, "uid": u}
            for pol, mode in (("full", "decode"), ("last", "convert")):
                res = runner.generate([it["prompt_ids"]], [u], pol, mode, proto.max_new_tokens, eos,
                                      lambda b, ids: any(s in tok.decode(ids, skip_special_tokens=False) for s in stops))
                ids, fin = res.gen_ids[0], res.finish[0]
                g = grade(ds, truncate_at_stops(tok.decode(ids, skip_special_tokens=False), stops), it["doc"], fin)
                b = rec[(ds, f"{pol}-{mode}", u)]
                row[f"{pol}"] = {"correct": g["correct"], "n": len(ids), "finish": fin,
                                 "batched_correct": b["correct"], "identical_to_batched": ids == b["gen_token_ids"]}
            out["items"].append(row)
    out["wall_s"] = time.time() - t0
    (run / "repeats").mkdir(exist_ok=True)
    (run / "repeats" / f"{args.model}__{args.dtype}.json").write_text(json.dumps(out, indent=1))


def report(run: Path):
    md = ["## E0 audit: discordant items repeated at B1 (BF16 and FP32)", "",
          "Up to 6 FULL-only and 6 LAST-only correct items per model x dataset, regenerated at batch size 1. "
          "'same verdict' = B1 correctness equals the batched E0 receipt.", "",
          "| model | dtype | items | FULL same verdict | LAST-C same verdict | FULL identical tokens | LAST-C identical tokens | "
          "discordance kept at B1 (FULL≠LAST correctness) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for p in sorted((run / "repeats").glob("*.json")):
        r = json.loads(p.read_text())
        it = r["items"]
        if not it:
            md.append(f"| {r['model']} | {r['dtype']} | 0 | | | | | |")
            continue
        md.append(f"| {r['model']} | {r['dtype']} | {len(it)} | {sum(x['full']['correct'] == x['full']['batched_correct'] for x in it)} | "
                  f"{sum(x['last']['correct'] == x['last']['batched_correct'] for x in it)} | "
                  f"{sum(x['full']['identical_to_batched'] for x in it)} | {sum(x['last']['identical_to_batched'] for x in it)} | "
                  f"{sum(x['full']['correct'] != x['last']['correct'] for x in it)} |")
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "e0_repeats.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
