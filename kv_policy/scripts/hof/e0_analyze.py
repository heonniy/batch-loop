"""E0 analysis: FULL vs LAST-CONVERT for Huginn R32 (reused run) and Ouro-1.4B / Ouro-2.6B (this run).

Writes analysis/e0_pairs.csv (one row per model x dataset x item) and analysis/e0_quality.md.
Paired statistics: accuracy difference with a 95% bootstrap CI over problems, exact McNemar
p-value on discordant pairs, token-weighted gold NLL and per-problem paired ΔNLL with bootstrap CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import zlib
from pathlib import Path

HUGINN_RUN = "results/20261023T060125Z_huginn_r32_h200x8"
CAPS = {"gsm8k": 512, "math500": 2048}
# retained recurrent KV per token (bytes, bf16): planes x n_rec_layers x {K,V} x H x D x 2
ARCH = {"huginn-0125": {"R": 32, "n_rec": 4, "n_plain": 4, "H": 55, "D": 96},
        "ouro-1.4b": {"R": 4, "n_rec": 24, "n_plain": 0, "H": 16, "D": 128},
        "ouro-2.6b": {"R": 4, "n_rec": 48, "n_plain": 0, "H": 16, "D": 128}}
VARIANTS = {"gsm8k": ("lm_eval_strict_match", "lm_eval_flexible_extract"),
            "math500": ("math_verify", "minerva_is_equiv", "normalized_string_equal")}


def load(run: Path, model: str, task: str):
    out = {}
    for f in sorted((run / "raw").glob(f"{model}__*__{task}__*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[(r["dataset"], f"{r['policy']}-{r['mode']}", r["uid"])] = r
    return out


def boot(vals, key, n=2000):
    rng = random.Random(zlib.crc32(key.encode()))
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(n))
    return st.mean(vals), ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--huginn-run", default=HUGINN_RUN)
    args = ap.parse_args()
    run = Path(args.run_dir)
    sources = {"huginn-0125": Path(args.huginn_run), "ouro-1.4b": run, "ouro-2.6b": run}
    rows, md = [], []
    md += ["# E0 — FULL vs LAST-CONVERT (Huginn R32, Ouro-1.4B R4, Ouro-2.6B R4)", "",
           f"Ouro: this run (`{run}`). Huginn R32: reused `{args.huginn_run}` (same frozen items, prompts, evaluator, "
           "caps 512/2048, FULL = full/decode, LAST-CONVERT = last/convert; same engine semantics). Greedy, fixed depth, "
           "BF16, early exit disabled. 'LAST-CONVERT' is **our** policy: loop-major FULL prefill, then every past token "
           "keeps only its terminal-step K/V; the official Ouro sharing semantics are not established, so degradation is "
           "attributed to this policy only.", "",
           "| model | dataset | n | acc FULL | acc LAST-C | Δ (LAST−FULL) [95% CI] | FULL-only / LAST-only | McNemar p | "
           "gold NLL FULL / LAST (tok-weighted) | ΔNLL per problem [CI] | cap hit FULL / LAST | parse fail FULL / LAST | "
           "mean gen len FULL / LAST | identical generations | first divergence median [IQR] |",
           "|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|---|---:|---|"]
    variant_lines = []
    for model, src in sources.items():
        gen, nll = load(src, model, "gen"), load(src, model, "nll")
        for ds in ("gsm8k", "math500"):
            uids = sorted({u for (d, p, u) in gen if d == ds and p == "full-decode"})
            if not uids:
                continue
            F = [gen[(ds, "full-decode", u)] for u in uids]
            L = [gen[(ds, "last-convert", u)] for u in uids]
            NF = [nll.get((ds, "full-decode", u)) for u in uids]
            NL = [nll.get((ds, "last-convert", u)) for u in uids]
            for u, f, l, nf, nl in zip(uids, F, L, NF, NL):
                rows.append({"model": model, "dataset": ds, "uid": u, "prompt_tokens": f["prompt_tokens"],
                             "full_correct": f["correct"], "last_correct": l["correct"],
                             "full_n_gen": f["n_gen_tokens"], "last_n_gen": l["n_gen_tokens"],
                             "full_finish": f["finish_reason"], "last_finish": l["finish_reason"],
                             "full_parse_ok": f["parse_ok"], "last_parse_ok": l["parse_ok"],
                             "first_divergence": first_div(f["gen_token_ids"], l["gen_token_ids"]),
                             "full_nll_mean": nf["nll_mean"] if nf else None, "last_nll_mean": nl["nll_mean"] if nl else None,
                             "target_tokens": nf["target_tokens"] if nf else None,
                             **{f"full_{v}": f.get(v) for v in VARIANTS[ds]}, **{f"last_{v}": l.get(v) for v in VARIANTS[ds]}})
            d = [float(l["correct"]) - float(f["correct"]) for f, l in zip(F, L)]
            m, lo, hi = boot(d, f"acc{model}{ds}")
            b = sum(f["correct"] and not l["correct"] for f, l in zip(F, L))
            c = sum(l["correct"] and not f["correct"] for f, l in zip(F, L))
            tw = lambda X: sum(x["nll_mean"] * x["target_tokens"] for x in X) / sum(x["target_tokens"] for x in X)  # noqa: E731
            dn = [nl["nll_mean"] - nf["nll_mean"] for nf, nl in zip(NF, NL) if nf and nl]
            dm, dlo, dhi = boot(dn, f"nll{model}{ds}")
            fd = sorted(x for x in (first_div(f["gen_token_ids"], l["gen_token_ids"]) for f, l in zip(F, L)) if x is not None)
            q = lambda p: fd[min(len(fd) - 1, int(p * len(fd)))] if fd else None  # noqa: E731
            cap = CAPS[ds]
            md.append(f"| {model} | {ds} | {len(uids)} | {st.mean(float(x['correct']) for x in F):.3f} | "
                      f"{st.mean(float(x['correct']) for x in L):.3f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {b} / {c} | {mcnemar(b, c):.1e} | "
                      f"{tw([x for x in NF if x]):.4f} / {tw([x for x in NL if x]):.4f} | {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}] | "
                      f"{sum(x['n_gen_tokens'] >= cap for x in F)} / {sum(x['n_gen_tokens'] >= cap for x in L)} | "
                      f"{sum(not x['parse_ok'] for x in F)} / {sum(not x['parse_ok'] for x in L)} | "
                      f"{st.mean(x['n_gen_tokens'] for x in F):.0f} / {st.mean(x['n_gen_tokens'] for x in L):.0f} | "
                      f"{len(uids) - len(fd)} | {q(0.5)} [{q(0.25)}, {q(0.75)}] |")
            variant_lines.append(f"- {model} {ds}: " + ", ".join(
                f"{v} FULL {st.mean(float(bool(x.get(v))) for x in F):.3f} / LAST {st.mean(float(bool(x.get(v))) for x in L):.3f}"
                for v in VARIANTS[ds]))
    md += ["", "Scorer variants (same generations):", ""] + variant_lines + [""]
    md += ["## KV footprint (analytic, BF16, per token)", "",
           "| model | recurrent KV FULL | recurrent KV LAST-CONVERT (retained during decode) | prelude/coda KV | retained total LAST/FULL | "
           "prefill transient (both policies) |", "|---|---:|---:|---:|---:|---|"]
    for m, a in ARCH.items():
        unit = a["n_rec"] * 2 * a["H"] * a["D"] * 2
        plain = a["n_plain"] * 2 * a["H"] * a["D"] * 2
        full, last = a["R"] * unit, unit
        md.append(f"| {m} | {full / 1024:.0f} KiB | {last / 1024:.1f} KiB | {plain / 1024:.1f} KiB | "
                  f"{(last + plain) / (full + plain):.3f} | prompt needs all {a['R']} planes during the parallel FULL prefill |")
    md += ["", "Retained = after Convert, during decode. The prefill peak is NOT reduced by Convert (the prompt is prefilled "
           "loop-major with every per-step plane). The reference engine stores a FULL-sized broadcast layout for every "
           "policy, so its measured peaks are not policy footprints.", ""]
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    with open(out / "e0_pairs.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    rep = run / "analysis" / "e0_repeats.md"
    if rep.exists():
        md += rep.read_text().splitlines()
    (out / "e0_quality.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
