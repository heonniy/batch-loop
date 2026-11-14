#!/usr/bin/env python3
"""Post-hoc accuracy re-scoring of the frontier receipts (CPU only; receipts are not modified).

Why: the runner's `correct` field is the task's primary lm-eval metric as replicated in WaveFront
exp3 `tasks.py` — for math500 that is minerva `exact_match`, which only accepts the literal phrase
"Final Answer: The final answer is X. I hope it is correct.". Base Ouro/Huginn usually end with
`\\boxed{X}` (or "Final Answer: The answer is \\boxed{X}") instead, so almost every answer becomes
`[invalidanswer]`. The upstream secondary metric (math_verify) is None because the package is not
installed in the run env.

Added metrics (decoded from the stored token ids with the target tokenizer, truncated at the same
stop strings):
  math500 `boxed`: last \\boxed{...} of the generation -> the same minerva normalize_final_answer and
                   is_equiv as the strict metric (gold unchanged).
  gsm8k   `flex` : lm-eval flexible-extract (already computed by tasks.score, previously discarded).
Output: <out>.json and <out>.md, per cell x arm: n, strict (recorded), recomputed strict, added metric.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dsd_run as base  # noqa: E402  (exposes WaveFront exp3 tasks as base.ds)

TOK = {"ouro-1.4b": "ByteDance/Ouro-1.4B", "huginn-3.5b": "tomg-group-umd/huginn-0125"}


def boxed_ok(gen, gold, stop):
    t = base.ds._truncate_at_stops(gen, stop)
    b = base.ds.last_boxed_only_string(t)
    if b is None:
        return False
    inner = base.ds.remove_boxed(b)
    if inner is None:
        return False
    pred = base.ds.normalize_final_answer(inner)
    return pred == gold or base.ds.is_equiv(pred, gold)


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    toks, golds, res = {}, {}, {}
    metas = sorted(glob.glob(f"{a.raw}/*/*.jsonl.meta.json"))
    cells = defaultdict(list)
    for mp in metas:
        args = json.load(open(mp))["args"]
        if args["dataset"] not in ("gsm8k", "math500"):
            continue
        name = Path(mp).name.replace(".jsonl.meta.json", "")
        lane = Path(mp).parent.name
        key = f"{lane}/" + base_re(name)
        cells[key].append((mp.replace(".meta.json", ".gz"), args))
    for key, files in sorted(cells.items()):
        args = files[0][1]
        ds, fs, model = args["dataset"], args["num_fewshot"], args["model"]
        if model not in toks:
            toks[model] = AutoTokenizer.from_pretrained(TOK[model])
        if (ds, fs) not in golds:
            golds[(ds, fs)] = base.ds.load_dataset_prompts(ds, fs, 300 if ds == "gsm8k" else 50)
        rows = golds[(ds, fs)]
        acc = defaultdict(lambda: [0, 0, 0, 0])  # n, recorded strict, recomputed strict, added
        for path, _ in files:
            for line in gzip.open(path, "rt"):
                r = json.loads(line)
                g = rows[int(r["prompt_id"].split(":")[1])]
                txt = toks[model].decode(r["tokens"], skip_special_tokens=True)
                ok, _, extra = base.ds.score(ds, txt, g["gold"], g["stop"], g.get("meta"))
                add = extra["correct_flex"] if ds == "gsm8k" else boxed_ok(txt, g["gold"], g["stop"])
                c = acc[r["arm"]]
                c[0] += 1; c[1] += bool(r["correct"]); c[2] += bool(ok); c[3] += bool(add)
        res[key] = {arm: {"n": n, "strict_recorded": s / n, "strict_recomputed": s2 / n,
                          ("flex" if ds == "gsm8k" else "boxed"): x / n}
                    for arm, (n, s, s2, x) in sorted(acc.items())}
        print(key, {k: round(v.get("boxed", v.get("flex")), 3) for k, v in res[key].items()}, flush=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("# Post-hoc accuracy re-scoring (see rescore_accuracy.py docstring)\n\n"
                "strict = lm-eval primary metric as recorded (gsm8k strict-match / math500 minerva exact_match); "
                "added = gsm8k flexible-extract / math500 last-\\\\boxed{} with minerva normalization + is_equiv.\n\n"
                "| cell | arm | n | strict (recorded) | strict (recomputed) | added metric |\n|---|---|---:|---:|---:|---:|\n")
        for key, arms in res.items():
            for arm, v in arms.items():
                add = v.get("boxed", v.get("flex"))
                f.write(f"| {key} | {arm} | {v['n']} | {v['strict_recorded']:.3f} | {v['strict_recomputed']:.3f} | {add:.3f} |\n")


def base_re(name):
    import re
    return re.sub(r"_(o\d+n\d+(r\d+)?|i\d+n\d+|n\d+)$", "", name)


if __name__ == "__main__":
    main()
