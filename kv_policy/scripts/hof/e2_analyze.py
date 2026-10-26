"""E2 analysis -> analysis/e2_cross_depth_attention.csv, e2_error_propagation.csv, e2_persistent.csv, e2_localisation.md.

Per-prompt means (over its 3 decode sites and the grid cell) -> mean over prompts with a 95% bootstrap CI.
Recovery is compared within model and, across models, only through R-normalised quantities
(per-remaining-step decay factor (E_final / E_inj)^(1/(R - r))) and the depth-matched Huginn sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import zlib
from collections import defaultdict
from pathlib import Path

ORDER = ("ouro-1.4b", "ouro-2.6b", "huginn-0125-r4", "huginn-0125-r8", "huginn-0125-r16", "huginn-0125")
N_BOOT = 1000


def boot(vals, key):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(zlib.crc32(key.encode()))
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(N_BOOT))
    return st.mean(vals), ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT) - 1], len(vals)


def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def f3(t, f=".3g"):
    return f"{t[0]:{f}} [{t[1]:{f}}, {t[2]:{f}}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    recs = [json.loads(p.read_text()) for p in sorted((run / "e2" / "raw").glob("e2__*.json"))]
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    models = [m for m in ORDER if any(r["job"]["model"] == m for r in recs)]
    counts = ", ".join(f"{m}: {sum(r['job']['model'] == m for r in recs)}" for m in models)
    md = ["# E2 — failure localisation (current-token interventions, frozen FULL history)", "",
          f"Run `{run}`: {len(recs)} prompt records ({counts}); "
          "3 decode sites per prompt on each model's own FULL greedy prefix; BF16. Huginn r4/r8/r16 = separately labelled "
          "non-native depth sweep. E = relative state error vs the unperturbed FULL trajectory; KL = KL(FULL ‖ intervened) "
          "of the final logits. Values: mean over prompts [95% bootstrap CI].", ""]
    par = [s["parity"]["max_abs_logit_diff"] for r in recs for s in r["sites"]]
    md += [f"Parity (functional single-token map vs engine, BF16): max |Δlogit| {max(par):.3g} over {len(par)} sites, top-1 "
           f"identical {sum(s['parity']['top1_same'] for r in recs for s in r['sites'])}/{len(par)}.", ""]

    fq_rows, dec_rows, pers_rows = [], [], []
    for r in recs:
        m, R = r["job"]["model"], r["R"]
        agg_fq, agg_dec, agg_p = defaultdict(list), defaultdict(list), defaultdict(list)
        for s in r["sites"]:
            for x in s["fixed_q"]:
                agg_fq[(x["layer"], x["r"], x["s"])].append(x)
            for x in s["decay"]:
                agg_dec[(x["layer"], x["r"], x["kind"][:4])].append(x)
            for x in s["persistent"]:
                agg_p[x["label"]].append(x)
        for (l, rr, ss), xs in agg_fq.items():
            fq_rows.append({"model": m, "R": R, "uid": r["job"]["uid"], "layer": l, "r": rr, "s": ss,
                            "D_attn": st.mean(x["D_attn"] for x in xs), "D_out": st.mean(x["D_out"] for x in xs),
                            "attn_over_resid": st.mean(x["attn_over_resid"] for x in xs)})
        for (l, rr, kind), xs in agg_dec.items():
            ei, ef = st.mean(x["E_inj"] for x in xs), st.mean(x["E_final"] for x in xs)
            rem = R - rr
            dec_rows.append({"model": m, "R": R, "uid": r["job"]["uid"], "layer": l, "r": rr, "kind": kind,
                             "E_inj": ei, "E_final": ef, "recovery_ratio": ef / ei if ei else float("nan"),
                             "per_step_factor": (ef / ei) ** (1 / rem) if ei and rem else float("nan"),
                             "kl": st.mean(x["kl"] for x in xs), "top1_same": st.mean(float(x["top1_same"]) for x in xs)})
        for lab, xs in agg_p.items():
            pers_rows.append({"model": m, "R": R, "uid": r["job"]["uid"], "label": lab,
                              "E_final": st.mean(x["E_step_end"][-1] for x in xs), "kl": st.mean(x["kl"] for x in xs),
                              "top1_same": st.mean(float(x["top1_same"]) for x in xs)})
    write_csv(out / "e2_cross_depth_attention.csv", fq_rows)
    write_csv(out / "e2_error_propagation.csv", dec_rows)
    write_csv(out / "e2_persistent.csv", pers_rows)

    def per_prompt(rows, m, cond, key):
        d = defaultdict(list)
        for x in rows:
            if x["model"] == m and cond(x):
                d[x["uid"]].append(x[key])
        return [st.mean(v) for v in d.values()]

    md += ["## a. Fixed-query cross-depth attention (layer mean): relative attention-output change when step r reads step-s history", ""]
    for m in models:
        R = next(x["R"] for x in fq_rows if x["model"] == m)
        rs = sorted({x["r"] for x in fq_rows if x["model"] == m})
        md.append(f"- **{m}** (R={R}): reading terminal history (s=R): " + ", ".join(
            f"r={rr}: {boot(per_prompt(fq_rows, m, lambda x, rr=rr: x['r'] == rr and x['s'] == R, 'D_attn'), f'fq{m}{rr}')[0]:.3f}"
            for rr in rs if rr < R)
            + "; reading step-1 history from the terminal step (r=R, s=1): "
            + f"{boot(per_prompt(fq_rows, m, lambda x: x['r'] == R and x['s'] == 1, 'D_attn'), f'fqr{m}')[0]:.3f}")
    md.append("")

    md += ["## b. Single-call terminal read (one layer, one step) vs matched-norm random shift, propagated to the output", "",
           "| model | r | E_inj (kv) | E_final kv | E_final random | per-step decay kv | final KL kv | final KL random | top-1 kept kv |",
           "|---|---:|---|---|---|---|---|---|---:|"]
    for m in models:
        R = next(x["R"] for x in dec_rows if x["model"] == m)
        for rr in sorted({x["r"] for x in dec_rows if x["model"] == m}):
            kv = lambda key: boot(per_prompt(dec_rows, m, lambda x: x["r"] == rr and x["kind"] == "kv", key), f"d{m}{rr}{key}")  # noqa: E731
            rd = lambda key: boot(per_prompt(dec_rows, m, lambda x: x["r"] == rr and x["kind"] == "rand", key), f"r{m}{rr}{key}")  # noqa: E731
            md.append(f"| {m} | {rr} | {kv('E_inj')[0]:.3g} | {f3(kv('E_final'))} | {rd('E_final')[0]:.3g} | "
                      f"{kv('per_step_factor')[0]:.3g} | {f3(kv('kl'))} | {rd('kl')[0]:.3g} | {kv('top1_same')[0]:.3f} |")
    md += ["", "Layer x step sensitivity (final KL of the single-call terminal read, mean over prompts; top-5 cells per model):", ""]
    for m in models:
        cells = defaultdict(list)
        for x in dec_rows:
            if x["model"] == m and x["kind"] == "kv":
                cells[(x["layer"], x["r"])].append(x["kl"])
        top = sorted(cells.items(), key=lambda kv: -st.mean(kv[1]))[:5]
        md.append(f"- {m}: " + "; ".join(f"layer {l} step {rr}: KL {st.mean(v):.3g}" for (l, rr), v in top))
    md.append("")

    md += ["## c. Persistent terminal reads for the current token (history unchanged)", "",
           "| model | reads replaced | E_final | final KL | top-1 kept |", "|---|---|---|---|---:|"]
    for m in models:
        labs = sorted({x["label"] for x in pers_rows if x["model"] == m},
                      key=lambda s: (s.startswith("all"), int(s[4:].split("_")[0]) if s.startswith("step") else 0))
        for lab in labs:
            e = boot(per_prompt(pers_rows, m, lambda x: x["label"] == lab, "E_final"), f"p{m}{lab}E")
            k = boot(per_prompt(pers_rows, m, lambda x: x["label"] == lab, "kl"), f"p{m}{lab}K")
            t = boot(per_prompt(pers_rows, m, lambda x: x["label"] == lab, "top1_same"), f"p{m}{lab}T")
            md.append(f"| {m} | {lab} | {f3(e)} | {f3(k)} | {t[0]:.3f} |")
    md += ["", "Single-call shifts are recovered under an unchanged history if E_final ≪ E_inj; persistent rows show whether "
           "replacing every read of a step moves the token to a different output (changed fixed point) rather than a "
           "recoverable perturbation.", ""]
    (out / "e2_localisation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
