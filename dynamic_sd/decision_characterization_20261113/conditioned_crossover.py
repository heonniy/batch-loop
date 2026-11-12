#!/usr/bin/env python3
"""Stage B — reference-D-conditioned paired crossover of draft depth d (fixed K) and length K (fixed d).

POST-HOC: D = native AR reference exit depth of the token being processed at the checkpoint
(`ref_next_depths[0]`), i.e. a post-verification quantity, NOT a pre-decision feature.
Stratifying on it (or on accepted length, an outcome) describes association only.

Per prefix and action, utility U = Σ committed tokens / Σ measured ms over its 3 timing repeats
(all actions at a prefix start from the identical committed-prefix KV snapshot). Comparisons are
paired per prefix; bin ratios are aggregate (Σ tok / Σ ms) with prompt-cluster bootstrap CIs.

Fields available in the receipts (SOURCE MEASURED): committed_tokens, wall_ms, info.n_acc,
info.depths (final exit depth of every DtV source), info.verify_ticks, AR info.depth, k, pos,
ref_next_depths. DERIVED FROM RECEIPTS (exact algebra on the recorded depths):
  core_rows    = Σ depths_i                         (DtV) / depth (AR)
  draft_calls  = Σ min(d, depths_i)                  (each source runs ≤ d single-row draft steps)
  core_calls   = draft_calls + verify_ticks
  rejected_rows= Σ depths_i for sources after the first rejected draft
MISSING (not reconstructed): per-stage Core/Prelude/Coda wall, CUDA-event stage timings, CPU stall,
graph bucket, prelude/coda wall. Stage-time attribution is therefore NOT reported.

Usage: conditioned_crossover.py --cf-dir DIR --out-dir OUT [--min-n 30]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from cf_receipts import ReceiptError, parse_action, read_rows, tok_ms, validate

MISSING_FIELDS = ["stage wall: Core / Prelude / Coda (ms)", "CUDA-event timings", "CPU stall / host sync time",
                  "graph bucket", "per-call prelude/coda counts beyond K+1 implied by the action"]


def d_bins(model):
    if model == "huginn":
        return [("≤14", lambda D: D <= 14), ("15", lambda D: D == 15), ("16", lambda D: D == 16),
                ("17", lambda D: D == 17), ("18–20", lambda D: 18 <= D <= 20), ("21–31 (tail)", lambda D: 21 <= D <= 31),
                ("32 (no exit)", lambda D: D == 32)]
    return [(str(v), (lambda v: lambda D: D == v)(v)) for v in (1, 2, 3, 4)]


def derived(action, trial):
    kind, d, K = parse_action(action)
    info = trial["info"]
    if kind == "AR":
        return {"core_rows": info["depth"], "core_calls": info["depth"], "n_acc": None, "rejected_rows": 0}
    deps = info["depths"]
    n_acc = info["n_acc"]
    rejected = n_acc < K and trial["committed_tokens"] == n_acc + 1
    rej_rows = sum(deps[n_acc + 1:]) if rejected else 0   # same slice as dsd_run.dtv_live discard_rows
    draft_calls = sum(min(d, x) for x in deps)
    return {"core_rows": sum(deps), "core_calls": draft_calls + info["verify_ticks"], "n_acc": n_acc,
            "rejected_rows": rej_rows}


def boot_ratio(items, n=2000, seed=0):
    """items: list of (prompt, tok_hi, ms_hi, tok_lo, ms_lo) -> (point, lo, hi) of aggregate U_hi/U_lo."""
    byp = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for p, a, b, c, d in items:
        v = byp[p]; v[0] += a; v[1] += b; v[2] += c; v[3] += d
    keys = sorted(byp)
    def r(ks):
        s = [sum(byp[k][i] for k in ks) for i in range(4)]
        return (s[0] / s[1]) / (s[2] / s[3])
    point = r(keys)
    if len(keys) < 2:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    bs = sorted(r([rng.choice(keys) for _ in keys]) for _ in range(n))
    return point, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def classify(rows):
    """rows: [(bin, n, point, lo, hi)] -> crossover label over adequately-sized bins."""
    ok = [x for x in rows if not x[5]]
    above = [x for x in ok if x[3] > 1]
    below = [x for x in ok if x[4] < 1]
    pts_above = [x for x in ok if x[2] > 1]
    pts_below = [x for x in ok if x[2] < 1]
    if above and below:
        return "reproducible crossover"
    if pts_above and pts_below:
        return "noisy crossover (point estimates cross 1, CIs do not separate)"
    if all(x[3] > 1 for x in ok) and ok:
        return "no crossover — higher option dominates in every bin"
    if all(x[4] < 1 for x in ok) and ok:
        return "no crossover — lower option dominates in every bin"
    return "no crossover (same sign of point estimate, some CIs include 1)"


def compare(cell, model, a_lo, a_hi, min_n, by="D"):
    bins = d_bins(model)
    per_bin = {b: [] for b, _ in bins}
    lat = {b: [] for b, _ in bins}
    acc = {b: [[], []] for b, _ in bins}
    rows_ = {b: [[], []] for b, _ in bins}
    for p in cell["prefixes"].values():
        D = p["meta"]["ref_next_depths"][0]
        b = next(name for name, f in bins if f(D))
        th, mh = tok_ms(p["actions"][a_hi])
        tl, ml = tok_ms(p["actions"][a_lo])
        per_bin[b].append((p["meta"]["prompt_idx"], th, mh, tl, ml))
        n = len(p["actions"][a_hi])
        lat[b].append(mh / n - ml / n)
        dh = derived(a_hi, p["actions"][a_hi][0]); dl = derived(a_lo, p["actions"][a_lo][0])
        acc[b][0].append(dl["n_acc"]); acc[b][1].append(dh["n_acc"])
        rows_[b][0].append(dl["core_rows"]); rows_[b][1].append(dh["core_rows"])
    out = []
    for b, _ in bins:
        items = per_bin[b]
        if not items:
            continue
        pt, lo, hi = boot_ratio(items)
        nprom = len({x[0] for x in items})
        sparse = len(items) < min_n or nprom < 5
        mean = lambda v: (sum(v) / len(v)) if v and v[0] is not None else None
        out.append((b, len(items), pt, lo, hi, sparse, nprom, sum(lat[b]) / len(lat[b]),
                    mean(acc[b][0]), mean(acc[b][1]), mean(rows_[b][0]), mean(rows_[b][1])))
    return out


def acceptance_strata(cell, a_lo, a_hi):
    """Confounding control: within accepted-length strata of the LOWER action (an outcome; post-hoc),
    how does U_hi/U_lo vary? Reports common support (n per stratum)."""
    strata = defaultdict(list)
    for p in cell["prefixes"].values():
        dl = derived(a_lo, p["actions"][a_lo][0])
        th, mh = tok_ms(p["actions"][a_hi]); tl, ml = tok_ms(p["actions"][a_lo])
        strata[dl["n_acc"]].append((p["meta"]["prompt_idx"], th, mh, tl, ml))
    res = []
    for s in sorted(strata, key=lambda x: (x is None, x)):
        pt, lo, hi = boot_ratio(strata[s])
        res.append((s, len(strata[s]), pt, lo, hi))
    return res


def context_strata(cell, a_lo, a_hi):
    """Stratify by generated-token index k (position in the continuation) tertiles."""
    ks = sorted(p["meta"]["k"] for p in cell["prefixes"].values())
    cut = [ks[len(ks) // 3], ks[2 * len(ks) // 3]]
    st = defaultdict(list)
    for p in cell["prefixes"].values():
        k = p["meta"]["k"]
        s = 0 if k < cut[0] else (1 if k < cut[1] else 2)
        th, mh = tok_ms(p["actions"][a_hi]); tl, ml = tok_ms(p["actions"][a_lo])
        st[s].append((p["meta"]["prompt_idx"], th, mh, tl, ml))
    names = [f"k<{cut[0]}", f"{cut[0]}≤k<{cut[1]}", f"k≥{cut[1]}"]
    return [(names[s], len(st[s]), *boot_ratio(st[s])) for s in sorted(st)]


def pairs_for(grid):
    ds = sorted({parse_action(a)[1] for a in grid if a != "AR"})
    Ks = sorted({parse_action(a)[2] for a in grid if a != "AR"})
    dpairs = [(f"DtV:d{d1}:K{K}", f"DtV:d{d2}:K{K}") for K in Ks for d1, d2 in zip(ds, ds[1:])]
    kpairs = [(f"DtV:d{d}:K{K1}", f"DtV:d{d}:K{K2}") for d in ds for K1, K2 in zip(Ks, Ks[1:])]
    arpairs = [("AR", f"DtV:d{d}:K{K}") for d in ds for K in Ks]
    return dpairs, kpairs, arpairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", default="ouro_fixed,ouro_tau0.4,ouro_tau0.1,huginn_fixed,huginn_eps0.1")
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    L = ["# Stage B — reference-D-conditioned crossover (POST-HOC, DERIVED FROM RECEIPTS)", "",
         "D = native AR reference exit depth of the token processed at the checkpoint (post-verification; not a "
         "pre-decision feature). Ratio = aggregate U(hi)/U(lo), U = Σ committed tokens / Σ measured ms, paired on "
         "the identical committed-prefix snapshot; 95%% CI = prompt-cluster bootstrap. `sparse` = n<%d prefixes or "
         "<5 prompts. Δlat = mean per-round latency hi − lo (ms). acc = mean accepted drafts; rows = core rows/round "
         "(derived)." % a.min_n, "",
         "**Missing fields (not reconstructed, so no stage-time attribution):** " + "; ".join(MISSING_FIELDS), ""]
    js = {"missing_fields": MISSING_FIELDS, "cells": {}}
    for c in a.cells.split(","):
        model = "huginn" if c.startswith("huginn") else "ouro"
        try:
            cell = validate(read_rows(Path(a.cf_dir) / f"{c}.jsonl.gz"), expected_repeats=3)
        except ReceiptError as e:
            L.append(f"**{c}: validation failure — {e}**"); continue
        dpairs, kpairs, arpairs = pairs_for(cell["actions"])
        dist = defaultdict(int)
        for p in cell["prefixes"].values():
            dist[p["meta"]["ref_next_depths"][0]] += 1
        L += [f"## {c}", "", "reference-D distribution of checkpoints: " +
              ", ".join(f"D={k}:{v}" for k, v in sorted(dist.items())), ""]
        js["cells"][c] = {"D_dist": dict(dist), "d": {}, "K": {}, "AR": {}}
        for title, pairs, key in (("draft depth d at fixed K", dpairs, "d"), ("length K at fixed d", kpairs, "K"),
                                  ("SD vs AR", arpairs, "AR")):
            L += [f"### {title}", "", "| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |",
                  "|---|---|---:|---:|---:|---|---|---|"]
            for lo_a, hi_a in pairs:
                res = compare(cell, model, lo_a, hi_a, a.min_n)
                lab = classify(res)
                js["cells"][c][key][f"{lo_a}->{hi_a}"] = {"label": lab, "bins": [
                    {"bin": r[0], "n": r[1], "prompts": r[6], "ratio": r[2], "lo": r[3], "hi": r[4], "sparse": r[5],
                     "dlat_ms": r[7], "acc_lo": r[8], "acc_hi": r[9], "rows_lo": r[10], "rows_hi": r[11]} for r in res]}
                for r in res:
                    f2 = lambda x: "-" if x is None else f"{x:.2f}"
                    L.append(f"| {lo_a} → {hi_a} | {r[0]} | {r[1]} ({r[6]}) | {r[2]:.3f} [{r[3]:.3f}, {r[4]:.3f}] | "
                             f"{r[7]:+.1f} | {f2(r[8])}→{f2(r[9])} | {r[10]:.1f}→{r[11]:.1f} | {'sparse' if r[5] else ''} |")
                L.append(f"| **{lo_a} → {hi_a}** | **{lab}** | | | | | | |")
            L.append("")
        # confounding controls on the main d comparison
        main = ("DtV:d1:K4", "DtV:d2:K4") if model == "ouro" else ("DtV:d4:K8", "DtV:d8:K8")
        L += [f"### confounding controls for {main[0]} → {main[1]} (post-hoc)", "",
              "| stratum | n | U_hi/U_lo [95% CI] |", "|---|---:|---:|"]
        for s, n, pt, lo, hi in acceptance_strata(cell, *main):
            L.append(f"| accepted(lo)={s} | {n} | {pt:.3f} [{lo:.3f}, {hi:.3f}] |")
        for s, n, pt, lo, hi in context_strata(cell, *main):
            L.append(f"| {s} | {n} | {pt:.3f} [{lo:.3f}, {hi:.3f}] |")
        L.append("")
    L += ["*Not done (needs GPU, diagnostic only):* forced counterfactual verification budget / matched replay depth "
          "at a fixed committed prefix. Any forced-depth arm has a different target; not run."]
    (out / "crossover.md").write_text("\n".join(L) + "\n")
    json.dump(js, open(out / "crossover.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
