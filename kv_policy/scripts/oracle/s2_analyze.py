"""Stage 2 analysis: position-binned fixed-prefix divergence with problem-clustered bootstrap CIs.

Generated position g = 1..n (g-th continuation token; its distribution reads g-1 continuation
tokens of history). Every bin reports surviving problems and tokens (censoring is explicit), a
token-weighted mean with a 95% CI from resampling PROBLEMS, a fixed-cohort view (problems that
reach the bin's upper range) to separate position effects from composition, per-request
normalised curves and first-disagreement distributions. `full_alt` (FULL with an independent
initial latent) is the noise floor every LAST / AVERAGE number should be read against.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import zlib
from pathlib import Path

BINS = [(1, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048)]
POLS = ("last", "average", "full_alt")
N_BOOT = 1000


def load(run: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run / "stage2" / "raw").glob("s2__*.json"))]


def boot_ci(groups: list[tuple[float, int]], seed: int) -> tuple[float, float, float]:
    """groups: per-problem (sum, count). Token-weighted mean, 95% CI by resampling problems."""
    tot_s, tot_n = sum(s for s, _ in groups), sum(n for _, n in groups)
    if tot_n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(N_BOOT):
        smp = [groups[rng.randrange(len(groups))] for _ in groups]
        n = sum(x[1] for x in smp)
        vals.append(sum(x[0] for x in smp) / n if n else float("nan"))
    vals.sort()
    return tot_s / tot_n, vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def bin_rows(recs, ds, src, cohort_min=0, tag="all"):
    out = []
    for pol in POLS:
        for lo, hi in BINS:
            groups = {m: [] for m in ("kl", "dnll", "chg")}
            n_tok = n_prob = 0
            for r in recs:
                if r["n"] < max(lo, cohort_min):
                    continue
                a, b = lo - 1, min(hi, r["n"])
                if b <= a:
                    continue
                n_prob += 1
                n_tok += b - a
                groups["kl"].append((sum(r[f"kl_{pol}"][a:b]), b - a))
                groups["dnll"].append((sum(r[f"dnll_{pol}"][a:b]), b - a))
                groups["chg"].append((sum(r[f"top1chg_{pol}"][a:b]), b - a))
            if not n_prob:
                continue
            row = {"dataset": ds, "source": src, "cohort": tag, "policy": pol, "bin": f"{lo}-{hi}",
                   "problems": n_prob, "tokens": n_tok}
            for m, g in groups.items():
                mean, lo_ci, hi_ci = boot_ci(g, zlib.crc32(repr((ds, src, tag, pol, lo, m)).encode()))
                row.update({f"{m}_mean": mean, f"{m}_ci_lo": lo_ci, f"{m}_ci_hi": hi_ci})
            out.append(row)
    return out


def first_disagreement(r, pol):
    for i, x in enumerate(r[f"top1chg_{pol}"]):
        if x:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    recs = load(run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    # ---- per item
    items = []
    for r in recs:
        row = {"dataset": r["job"]["dataset"], "uid": r["job"]["uid"], "source": r["job"]["source"], "P": r["P"],
               "n": r["n"], "full_correct": r["full_correct"], "full_finish": r["full_finish"],
               "level": r["level"], "subject": r["subject"], "any_nan": r["any_nan"],
               "normalization_max_err": r["normalization_max_err"],
               "nll_full_mean": st.mean(r["nll_full"]), "entropy_full_mean": st.mean(r["entropy_full"])}
        for pol in POLS:
            row[f"kl_{pol}_mean"] = st.mean(r[f"kl_{pol}"])
            row[f"kl_{pol}_max"] = max(r[f"kl_{pol}"])
            row[f"dnll_{pol}_mean"] = st.mean(r[f"dnll_{pol}"])
            row[f"top1chg_{pol}_rate"] = st.mean(r[f"top1chg_{pol}"])
            row[f"first_disagreement_{pol}"] = first_disagreement(r, pol)
        items.append(row)
    with open(out / "stage2_kl_by_item.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)

    # ---- position bins (all problems, and fixed cohorts)
    pos_rows = []
    for src in ("full_gen", "gold", "last_gen"):
        for ds in ("gsm8k", "math500"):
            sub = [r for r in recs if r["job"]["source"] == src and r["job"]["dataset"] == ds]
            if not sub:
                continue
            pos_rows += bin_rows(sub, ds, src)
            if src == "full_gen":
                for cm in ((129, 513) if ds == "gsm8k" else (257, 513, 1025)):
                    pos_rows += bin_rows(sub, ds, src, cm, f"n>={cm}")
    with open(out / "stage2_kl_by_position.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pos_rows[0]))
        w.writeheader()
        w.writerows(pos_rows)

    # ---- report
    md = ["# Huginn R32 Stage 2 — fixed-prefix distribution divergence (FULL vs LAST / AVERAGE)", "",
          f"Run `{run}`. Same prefix for every row; continuation = frozen greedy FULL/decode generation of "
          "the R32 quality run unless stated. KL = KL(p_FULL || p_X) in nats; ΔNLL = NLL_X − NLL_FULL of the "
          "FULL-generated next token; top-1 changed = argmax differs. `full_alt` = FULL with an independent "
          "initial latent (noise floor). Means are token-weighted; [95% CI] from resampling problems "
          f"({N_BOOT} draws). g = 1..n is the generated position.", "",
          f"Records: {len(recs)}; NaN in any: {any(r['any_nan'] for r in recs)}; "
          f"max |Σp − 1|: {max(r['normalization_max_err'] for r in recs):.1e}.", ""]

    def table(rows, title):
        md.extend([f"### {title}", "", "| bin g | problems | tokens | policy | KL mean [CI] | ΔNLL mean [CI] | top-1 changed [CI] |",
                   "|---|---:|---:|---|---|---|---|"])
        for r in rows:
            md.append(f"| {r['bin']} | {r['problems']} | {r['tokens']} | {r['policy']} | "
                      f"{r['kl_mean']:.2e} [{r['kl_ci_lo']:.2e}, {r['kl_ci_hi']:.2e}] | "
                      f"{r['dnll_mean']:+.2e} [{r['dnll_ci_lo']:+.2e}, {r['dnll_ci_hi']:+.2e}] | "
                      f"{r['chg_mean']:.4f} [{r['chg_ci_lo']:.4f}, {r['chg_ci_hi']:.4f}] |")
        md.append("")

    for ds in ("gsm8k", "math500"):
        if not any(r["job"]["dataset"] == ds and r["job"]["source"] == "full_gen" for r in recs):
            continue
        md += [f"## {ds}", ""]
        sub = [r for r in recs if r["job"]["source"] == "full_gen" and r["job"]["dataset"] == ds]
        if not sub:
            continue
        ns = sorted(r["n"] for r in sub)
        md.append(f"{len(sub)} problems; continuation length median {ns[len(ns) // 2]}, max {ns[-1]}; "
                  f"truncated at cap: {sum(r['full_finish'] == 'length' for r in sub)}. Problems surviving per bin: "
                  + ", ".join(f"{lo}-{hi}: {sum(r['n'] >= lo for r in sub)}" for lo, hi in BINS) + ".")
        md.append("")
        table([r for r in pos_rows if r["dataset"] == ds and r["source"] == "full_gen" and r["cohort"] == "all"],
              f"{ds}: all problems (composition changes across bins)")
        for tag in sorted({r["cohort"] for r in pos_rows if r["dataset"] == ds and r["cohort"] != "all"}):
            table([r for r in pos_rows if r["dataset"] == ds and r["source"] == "full_gen" and r["cohort"] == tag
                   and r["policy"] in ("last", "full_alt")], f"{ds}: fixed cohort {tag} (same problems in every bin)")
        # normalised curves
        md += [f"### {ds}: per-request normalised position (mean over problems of the decile mean)", "",
               "| decile of g/n | " + " | ".join(f"KL {p}" for p in POLS) + " | " + " | ".join(f"top-1 chg {p}" for p in POLS) + " |",
               "|---|" + "---:|" * (2 * len(POLS))]
        for dcl in range(10):
            cells = []
            for metric in ("kl", "top1chg"):
                for p in POLS:
                    vals = []
                    for r in sub:
                        a, b = int(r["n"] * dcl / 10), max(int(r["n"] * (dcl + 1) / 10), int(r["n"] * dcl / 10) + 1)
                        seg = r[f"{metric}_{p}"][a:b]
                        if seg:
                            vals.append(sum(seg) / len(seg))
                    cells.append(f"{st.mean(vals):.2e}" if metric == "kl" else f"{st.mean(vals):.4f}")
            md.append(f"| {dcl * 10}-{dcl * 10 + 10}% | " + " | ".join(cells) + " |")
        md.append("")
        # first disagreement
        md += [f"### {ds}: first top-1 disagreement with FULL along the frozen FULL prefix", ""]
        for p in POLS:
            fd = [first_disagreement(r, p) for r in sub]
            hit = sorted(x for x in fd if x is not None)
            q = (lambda f: hit[min(len(hit) - 1, int(f * len(hit)))]) if hit else (lambda f: None)
            md.append(f"- {p}: {len(hit)}/{len(fd)} problems disagree somewhere; first disagreement g quartiles "
                      f"{q(0.25)}/{q(0.5)}/{q(0.75)}; never (censored at n): {len(fd) - len(hit)}")
        md.append("")
        # strata
        md += [f"### {ds}: item-level mean KL by stratum (mean over problems [min, max])", "",
               "| stratum | problems | KL last | KL average | KL full_alt | top-1 chg last | top-1 chg full_alt |", "|---|---:|---:|---:|---:|---:|---:|"]
        it = [r for r in items if r["dataset"] == ds and r["source"] == "full_gen"]
        strata = [("FULL correct", lambda r: r["full_correct"]), ("FULL incorrect", lambda r: not r["full_correct"]),
                  ("ended (eos/stop)", lambda r: r["full_finish"] != "length"), ("hit cap", lambda r: r["full_finish"] == "length")]
        if ds == "math500":
            strata += [(f"level {lv}", (lambda lv: lambda r: r["level"] == lv)(lv)) for lv in range(1, 6)]
        for name, f in strata:
            g = [r for r in it if f(r)]
            if g:
                md.append(f"| {name} | {len(g)} | {st.mean(r['kl_last_mean'] for r in g):.2e} | "
                          f"{st.mean(r['kl_average_mean'] for r in g):.2e} | {st.mean(r['kl_full_alt_mean'] for r in g):.2e} | "
                          f"{st.mean(r['top1chg_last_rate'] for r in g):.4f} | {st.mean(r['top1chg_full_alt_rate'] for r in g):.4f} |")
        md.append("")

    md += ["## Sensitivity: prefix source on the Stage-1 64-item selection", "",
           "| dataset | source | problems | tokens | KL last | KL average | KL full_alt | ΔNLL last | top-1 chg last | top-1 chg full_alt |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sel_uids = {(r["job"]["dataset"], r["job"]["uid"]) for r in recs if r["job"]["source"] == "gold"}
    for ds in ("gsm8k", "math500"):
        for src in ("full_gen", "last_gen", "gold"):
            g = [r for r in recs if r["job"]["source"] == src and r["job"]["dataset"] == ds
                 and (r["job"]["dataset"], r["job"]["uid"]) in sel_uids]
            if not g:
                continue
            ntok = sum(r["n"] for r in g)
            tw = lambda k: sum(sum(r[k]) for r in g) / ntok  # noqa: E731
            md.append(f"| {ds} | {src} | {len(g)} | {ntok} | {tw('kl_last'):.2e} | {tw('kl_average'):.2e} | "
                      f"{tw('kl_full_alt'):.2e} | {tw('dnll_last'):+.2e} | {tw('top1chg_last'):.4f} | {tw('top1chg_full_alt'):.4f} |")
    md += ["", "Caveats: per-bin means mix different problems (see fixed-cohort tables before reading a trend); "
           "the FULL-generated prefix favours FULL by construction (its own greedy path), so ΔNLL on `full_gen` "
           "is biased toward LAST looking worse; `last_gen` reverses that bias and `gold` is policy-neutral.", ""]
    (out / "stage2_kl_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
