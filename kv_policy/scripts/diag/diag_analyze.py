"""Analysis for OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md Steps 1-3.

  pilot   <run>/pilot raw receipts -> analysis/step1_policies.csv, step1_items.csv, step1_report.md
          per model x dataset x policy: accuracy, paired Δ vs FULL and vs LAST-CONVERT (95% bootstrap CI
          over items, exact McNemar), recovered fraction of the LAST->FULL accuracy and gold-NLL gaps,
          parse failures, cap hits, first greedy divergence from FULL, retained recurrent KV.
  step2   <run>/step2 raw -> analysis/step2_isolation.csv, step2_report.md
  memory  analytic retained-KV layout per policy -> analysis/step3_memory.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import zlib
from collections import defaultdict
from pathlib import Path

CAPS = {"gsm8k": 512, "math500": 2048}
ARCH = {"ouro-1.4b": {"L": 24, "R": 4, "plain": 0, "H": 16, "D": 128},
        "ouro-2.6b": {"L": 48, "R": 4, "plain": 0, "H": 16, "D": 128},
        "huginn-0125": {"L": 4, "R": 32, "plain": 4, "H": 55, "D": 96}}
POS_BINS = [(1, 64), (65, 256), (257, 1024), (1025, 2048)]


def boot(vals, key, n=2000):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(zlib.crc32(key.encode()))
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(n))
    return st.mean(vals), ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def load(run: Path, task: str):
    out = {}
    for f in sorted((run / "raw").glob(f"*__{task}__*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[(r["model"], r["dataset"], f"{r['policy']}-{r['mode']}", r["uid"])] = r
    return out


def retained_planes(model, policy, banks):
    a = ARCH[model]
    if policy == "full-decode":
        return a["L"] * a["R"]
    if policy in ("last-convert",):
        return a["L"]
    if policy == "last-decode":
        return None  # prompt keeps all planes, generated tokens one: depends on lengths
    cells = banks.get(model, {}).get(policy.split("-")[0])
    return None if cells is None else a["L"] + len(cells)


E0_RUN = "results/20261026T080106Z_huginn_ouro_full_last_convert"


def pilot(run: Path, out: Path, sub: str = "pilot"):
    pr = run / sub
    gen, nll = load(pr, "gen"), load(pr, "nll")
    if sub == "full":  # FULL and LAST-CONVERT receipts come from E0 (same main batching, same engine path)
        e0 = Path(__file__).resolve().parents[2] / E0_RUN
        for d, src in ((gen, load(e0, "gen")), (nll, load(e0, "nll"))):
            for k, v in src.items():
                if k[2] in ("full-decode", "last-convert"):
                    d.setdefault(k, v)
    banks = json.loads((pr / "banks.json").read_text())
    rows, items = [], []
    title = ("Step 1 — localized protection pilot (Stage-1 32 GSM8K + 32 MATH500 items" if sub == "pilot" else
             "Step 1 — full frozen evaluation (GSM8K-200 + MATH500-200; FULL / LAST-CONVERT receipts reused from E0")
    md = [f"# {title}, BF16, greedy, caps 512/2048)", "",
          "Base for every bank policy: LAST-CONVERT (FULL prefill, then terminal history) except the protected (layer, "
          "recurrence) banks, which keep per-recurrence K/V for every historical token (kvp.banks; verified against the "
          "dictionary reference). `last-decode` is a diagnostic alternate reading of the paper (prompt keeps every "
          "recurrence's K/V) and is NOT part of the primary comparison. Recovered fraction = (x − LAST) / (FULL − LAST).", "",
          "| model | dataset | policy | protected banks | retained rec. KV / FULL | acc | Δ vs FULL [CI] | Δ vs LAST [CI] | "
          "acc recovered | gold NLL | NLL recovered | parse fail | cap hit | identical to FULL | first div. median |",
          "|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    models = sorted({k[0] for k in gen})
    for m in models:
        for ds in ("gsm8k", "math500"):
            uids = sorted({k[3] for k in gen if k[0] == m and k[1] == ds and k[2] == "full-decode"})
            if not uids:
                continue
            pols = sorted({k[2] for k in gen if k[0] == m and k[1] == ds},
                          key=lambda p: (p != "full-decode", p != "last-convert", p != "last-decode", p))
            F = {u: gen[(m, ds, "full-decode", u)] for u in uids}
            Lr = {u: gen.get((m, ds, "last-convert", u)) for u in uids}
            nF = {u: nll.get((m, ds, "full-decode", u)) for u in uids}
            nL = {u: nll.get((m, ds, "last-convert", u)) for u in uids}
            tw = lambda X: sum(x["nll_mean"] * x["target_tokens"] for x in X) / sum(x["target_tokens"] for x in X)  # noqa: E731
            accF = st.mean(float(F[u]["correct"]) for u in uids)
            accL = st.mean(float(Lr[u]["correct"]) for u in uids if Lr[u]) if all(Lr.values()) else float("nan")
            nllF = tw([nF[u] for u in uids if nF[u]])
            nllL = tw([nL[u] for u in uids if nL[u]]) if all(nL.values()) else float("nan")
            for p in pols:
                G = {u: gen.get((m, ds, p, u)) for u in uids}
                if not all(G.values()):
                    continue
                N = [nll.get((m, ds, p, u)) for u in uids]
                acc = st.mean(float(G[u]["correct"]) for u in uids)
                dF = boot([float(G[u]["correct"]) - float(F[u]["correct"]) for u in uids], f"F{m}{ds}{p}")
                dL = boot([float(G[u]["correct"]) - float(Lr[u]["correct"]) for u in uids], f"L{m}{ds}{p}") if all(Lr.values()) else (float("nan"),) * 3
                nl = tw([x for x in N if x]) if all(N) else float("nan")
                fd = sorted(x for x in (first_div(F[u]["gen_token_ids"], G[u]["gen_token_ids"]) for u in uids) if x is not None)
                planes = retained_planes(m, p, banks)
                cells = banks.get(m, {}).get(p.split("-")[0])
                shown = "" if not cells else (json.dumps(cells) if len(cells) <= 6 else f"{len(cells)} banks")
                row = {"model": m, "dataset": ds, "policy": p, "n": len(uids), "protected": json.dumps(cells) if cells else "",
                       "retained_rec_planes": planes, "retained_frac_of_full": planes / (ARCH[m]["L"] * ARCH[m]["R"]) if planes else None,
                       "acc": acc, "d_vs_full": dF[0], "d_vs_full_lo": dF[1], "d_vs_full_hi": dF[2],
                       "d_vs_last": dL[0], "d_vs_last_lo": dL[1], "d_vs_last_hi": dL[2],
                       "mcnemar_vs_full": mcnemar(sum(F[u]["correct"] and not G[u]["correct"] for u in uids),
                                                  sum(G[u]["correct"] and not F[u]["correct"] for u in uids)),
                       "acc_recovered": (acc - accL) / (accF - accL) if accF != accL else float("nan"),
                       "nll": nl, "nll_recovered": (nllL - nl) / (nllL - nllF) if nllL == nllL and nllL != nllF else float("nan"),
                       "parse_fail": sum(not G[u]["parse_ok"] for u in uids),
                       "cap_hit": sum(G[u]["n_gen_tokens"] >= CAPS[ds] for u in uids),
                       "identical_to_full": len(uids) - len(fd), "first_div_median": fd[len(fd) // 2] if fd else None}
                rows.append(row)
                for u in uids:
                    items.append({"model": m, "dataset": ds, "policy": p, "uid": u, "correct": G[u]["correct"],
                                  "full_correct": F[u]["correct"], "n_gen": G[u]["n_gen_tokens"], "finish": G[u]["finish_reason"],
                                  "parse_ok": G[u]["parse_ok"], "first_div_vs_full": first_div(F[u]["gen_token_ids"], G[u]["gen_token_ids"]),
                                  "nll_mean": (nll.get((m, ds, p, u)) or {}).get("nll_mean")})
                f = lambda x: "—" if x is None or x != x else f"{x:.3f}"  # noqa: E731
                md.append(f"| {m} | {ds} | {p} | {shown or '—'} | {f(row['retained_frac_of_full'])} | {acc:.3f} | "
                          f"{dF[0]:+.3f} [{dF[1]:+.3f}, {dF[2]:+.3f}] | {dL[0]:+.3f} [{dL[1]:+.3f}, {dL[2]:+.3f}] | "
                          f"{f(row['acc_recovered'])} | {f(nl)} | {f(row['nll_recovered'])} | {row['parse_fail']} | {row['cap_hit']} | "
                          f"{row['identical_to_full']} | {row['first_div_median']} |")
            rnd = [r for r in rows if r["model"] == m and r["dataset"] == ds and r["policy"].startswith("bankrand")]
            if rnd:
                md.append(f"| {m} | {ds} | **random single bank ({len(rnd)} seed{'s' if len(rnd) > 1 else ''})** | — | {f(rnd[0]['retained_frac_of_full'])} | "
                          f"{st.mean(r['acc'] for r in rnd):.3f} (min {min(r['acc'] for r in rnd):.3f}, max {max(r['acc'] for r in rnd):.3f}) | | | "
                          f"{st.mean(r['acc_recovered'] for r in rnd):.3f} | {st.mean(r['nll'] for r in rnd):.3f} | "
                          f"{st.mean(r['nll_recovered'] for r in rnd):.3f} | | | | |")
    pre = "step1" if sub == "pilot" else "step1_full"
    for name, data in ((f"{pre}_policies.csv", rows), (f"{pre}_items.csv", items)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    notes = pr / "banks_notes.json"
    if notes.exists():
        md += ["", "Calibration-selected sets (disjoint prompts, rule fixed before the pilot): " + notes.read_text().replace("\n", " ")]
    (out / f"{pre}_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def step2(run: Path, out: Path):
    recs = [json.loads(p.read_text()) for p in sorted((run / "step2" / "raw").glob("s2__*.json"))]
    rows = []
    md = ["# Step 2 — recovery vs persistent-context mismatch (frozen FULL prefix, teacher forcing)", "",
          "Rows share prefix, latent and batch. Mean over items of per-item means [95% bootstrap CI over items]. "
          "Hidden error = ||h_row − h_FULL|| / ||h_FULL|| of the current token after each recurrent step. The one-call swap "
          "(condition A) is the E2 reference.", ""]
    for m in ("ouro-1.4b", "ouro-2.6b", "huginn-0125"):
        for dt in ("bfloat16", "float32"):
            g = [r for r in recs if r["job"]["model"] == m and r["job"]["dtype"] == dt]
            if not g:
                continue
            R = g[0]["R"]
            md += [f"## {m} ({dt}, {len(g)} items)", "", "| row | KL(FULL‖row) | top-1 agree | ΔNLL (FULL tokens) | hidden err after step 1 | "
                   "after mid step | final step | KL by position " + " / ".join(f"{a}-{b}" for a, b in POS_BINS) + " |",
                   "|---|---|---|---|---:|---:|---:|---|"]
            for name in g[0]["rows"]:
                kl = boot([st.mean(r[name]["kl"]) for r in g], f"k{m}{dt}{name}")
                t1 = boot([st.mean(map(float, r[name]["top1"])) for r in g], f"t{m}{dt}{name}")
                dn = boot([st.mean(r[name]["dnll"]) for r in g], f"d{m}{dt}{name}")
                he = [st.mean(r[name]["hidden_err_by_step_mean"][s] for r in g) for s in (0, R // 2 - 1, R - 1)]
                pb = []
                for a, b in POS_BINS:
                    v = [st.mean(r[name]["kl"][a - 1:b]) for r in g if len(r[name]["kl"]) >= a]
                    pb.append(f"{st.mean(v):.3g} (n={len(v)})" if v else "—")
                rows.append({"model": m, "dtype": dt, "row": name, "items": len(g), "kl": kl[0], "kl_lo": kl[1], "kl_hi": kl[2],
                             "top1": t1[0], "dnll": dn[0], "herr_step1": he[0], "herr_mid": he[1], "herr_final": he[2]})
                md.append(f"| {name} | {kl[0]:.3g} [{kl[1]:.3g}, {kl[2]:.3g}] | {t1[0]:.3f} | {dn[0]:+.3g} | {he[0]:.3f} | {he[1]:.3f} | "
                          f"{he[2]:.3f} | {' / '.join(pb)} |")
            md.append("")
    with open(out / "step2_isolation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "step2_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def memory(run: Path, out: Path):
    banks = json.loads((run / "pilot" / "banks.json").read_text())
    md = ["# Step 3 — retained KV layout (analytic, BF16)", "",
          "Per token. Recurrent plane = one (layer, recurrence) K+V slot: 2 x H x D x 2 bytes. Retained = during decode after "
          "Convert. Every policy here needs ALL planes for the prompt during the parallel FULL prefill (transient peak "
          "unchanged by Convert). The reference engine allocates a FULL-sized broadcast store for every policy, so measured "
          "engine peaks are not policy footprints; a compact store was not benchmarked and no serving speed is claimed.", "",
          "| model | policy | protected banks | retained recurrent planes / token | bytes / token | vs FULL | vs LAST-CONVERT |",
          "|---|---|---|---:|---:|---:|---:|"]
    for m, a in ARCH.items():
        plane = 2 * a["H"] * a["D"] * 2
        plain = a["plain"] * plane
        full = a["L"] * a["R"]
        entries = [("FULL", full, "all"), ("LAST-CONVERT", a["L"], "—")]
        for p, cells in sorted(banks.get(m, {}).items()):
            entries.append((p, a["L"] + len(cells), json.dumps(cells) if len(cells) <= 6 else f"{len(cells)} banks"))
        for name, planes, cells in entries:
            md.append(f"| {m} | {name} | {cells} | {planes} | {(planes * plane + plain) / 1024:.1f} KiB | "
                      f"{(planes * plane + plain) / (full * plane + plain):.3f} | {(planes * plane + plain) / (a['L'] * plane + plain):.3f} |")
    (out / "step3_memory.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["pilot", "full", "step2", "memory", "all"])
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    for p in (["pilot", "full", "step2", "memory"] if args.part == "all" else [args.part]):
        {"pilot": pilot, "full": lambda r, o: pilot(r, o, "full"), "step2": step2, "memory": memory}[p](run, out)


if __name__ == "__main__":
    main()
