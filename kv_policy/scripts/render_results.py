"""Fill kv_policy/RESULTS.template.md with tables generated from the published receipts.

Markers: {{ACC_MATRIX}}, {{DELTA_TABLE}}, {{STOPRULE_TABLE}}, {{NLLPOS_TABLE}}, {{STAGE1}}, {{STAGE3}},
{{RUN_DIRS}}. Every number in those tables is read from CSV/JSON receipts, never retyped.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CELLS = [("full", "decode"), ("last", "convert"), ("average", "convert"), ("last", "decode"),
         ("average", "decode"), ("last", "strict"), ("average", "strict"), ("full", "strict"),
         ("full_altbatch", "decode")]
LABEL = {("full", "decode"): "FULL", ("last", "convert"): "LAST conv", ("average", "convert"): "AVG conv",
         ("last", "decode"): "LAST dec", ("average", "decode"): "AVG dec", ("last", "strict"): "LAST strict",
         ("average", "strict"): "AVG strict", ("full", "strict"): "FULL strict (ctrl)",
         ("full_altbatch", "decode"): "FULL alt-batch (ctrl)"}
MODELS = ["ouro-1.4b", "ouro-2.6b", "huginn-0125"]


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def f(x, nd=3, sign=False):
    if x in (None, ""):
        return ""
    v = float(x)
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def acc_matrix(summ):
    idx = {(r["model"], r["dataset"], r["policy"], r["mode"]): r for r in summ}
    out = ["| model | dataset | " + " | ".join(LABEL[c] for c in CELLS) + " |", "|---|---|" + "---|" * len(CELLS)]
    for m in MODELS:
        for d in ("gsm8k", "math500"):
            cells = []
            for c in CELLS:
                r = idx.get((m, d, *c))
                cells.append("" if r is None else f"{float(r['accuracy']):.3f}")
            out.append(f"| {m} | {d} | " + " | ".join(cells) + " |")
    out += ["", "Teacher-forced token-weighted NLL (nats/token) on the same gold targets:", "",
            "| model | dataset | " + " | ".join(LABEL[c] for c in CELLS) + " |", "|---|---|" + "---|" * len(CELLS)]
    for m in MODELS:
        for d in ("gsm8k", "math500"):
            cells = []
            for c in CELLS:
                r = idx.get((m, d, *c))
                cells.append("" if r is None else f(r["nll_token_weighted"], 4))
            out.append(f"| {m} | {d} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def delta_table(pairs):
    out = ["| model | dataset | policy | region | Δacc vs FULL [95% CI] | FULL-only / policy-only | McNemar p | "
           "ΔNLL/token [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    order = {c: i for i, c in enumerate(CELLS)}
    rows = [r for r in pairs if r["baseline"] == "full/decode" and r["policy"] in ("last", "average")]
    rows.sort(key=lambda r: (MODELS.index(r["model"]), r["dataset"], order[(r["policy"], r["mode"])]))
    for r in rows:
        out.append(f"| {r['model']} | {r['dataset']} | {r['policy'].upper()} | {r['mode']} | "
                   f"{f(r['delta_acc'], 3, True)} [{f(r['delta_acc_ci95_lo'], 3, True)}, {f(r['delta_acc_ci95_hi'], 3, True)}] | "
                   f"{r['base_only_correct']} / {r['policy_only_correct']} | {float(r['mcnemar_exact_p']):.2g} | "
                   f"{f(r['delta_nll_token_weighted'], 4, True)} [{f(r['delta_nll_tw_ci95_lo'], 4, True)}, "
                   f"{f(r['delta_nll_tw_ci95_hi'], 4, True)}] |")
    ctrl = [r for r in pairs if (r["policy"], r["mode"]) in (("full", "strict"), ("full_altbatch", "decode"))]
    out += ["", "Numerical controls (same policy, different execution path / batch composition):", "",
            "| model | dataset | control | Δacc [95% CI] | ΔNLL/token [95% CI] |", "|---|---|---|---|---|"]
    for r in sorted(ctrl, key=lambda r: (MODELS.index(r["model"]), r["dataset"], r["policy"])):
        out.append(f"| {r['model']} | {r['dataset']} | {LABEL[(r['policy'], r['mode'])]} | "
                   f"{f(r['delta_acc'], 3, True)} [{f(r['delta_acc_ci95_lo'], 3, True)}, {f(r['delta_acc_ci95_hi'], 3, True)}] | "
                   f"{f(r['delta_nll_token_weighted'], 4, True)} [{f(r['delta_nll_tw_ci95_lo'], 4, True)}, "
                   f"{f(r['delta_nll_tw_ci95_hi'], 4, True)}] |")
    return "\n".join(out)


def stoprule_table(deg):
    out = ["| model | dataset | cell | degenerate | hit max tokens | acc main | acc stop-rule | answered→degenerated | "
           "degenerated w/o answer |", "|---|---|---|---|---|---|---|---|---|"]
    order = {c: i for i, c in enumerate(CELLS)}
    for r in sorted(deg, key=lambda r: (MODELS.index(r["model"]), r["dataset"], order[(r["policy"], r["mode"])])):
        if (r["policy"], r["mode"]) in (("full", "strict"), ("full_altbatch", "decode")):
            continue
        out.append(f"| {r['model']} | {r['dataset']} | {LABEL[(r['policy'], r['mode'])]} | {f(r['degenerate_rate'])} | "
                   f"{f(r['hit_max_tokens_rate'])} | {f(r['acc_main'])} | {f(r['acc_stop_rule'])} | "
                   f"{r['answered_then_degenerated']} | {r['degenerated_without_correct_answer']} |")
    return "\n".join(out)


def nllpos_table(pos):
    buckets = ["0-63", "64-127", "128-255", "256-511", "512-99999"]
    out = ["| model | dataset | cell | " + " | ".join(f"pos {b.replace('-99999', '+')}" for b in buckets) + " |",
           "|---|---|---|" + "---|" * len(buckets)]
    idx = {(r["model"], r["dataset"], r["policy"], r["mode"], r["positions"]): r for r in pos}
    for m in MODELS:
        for d in ("gsm8k", "math500"):
            for c in CELLS[1:7]:
                vals = []
                for b in buckets:
                    r = idx.get((m, d, *c, b))
                    vals.append("" if r is None else f"{float(r['delta_nll_per_token_vs_full']):+.3f} (n={r['n_problems']})")
                out.append(f"| {m} | {d} | {LABEL[c]} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--stage3", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    s2 = Path(args.stage2) / "analysis"
    text = (root / "RESULTS.template.md").read_text()
    repl = {
        "{{ACC_MATRIX}}": acc_matrix(read(s2 / "summary.csv")),
        "{{DELTA_TABLE}}": delta_table(read(s2 / "paired_vs_full.csv")),
        "{{STOPRULE_TABLE}}": stoprule_table(read(s2 / "degeneration.csv")),
        "{{NLLPOS_TABLE}}": nllpos_table(read(s2 / "nll_by_position_vs_full.csv")),
        "{{STAGE1}}": (Path(args.stage1) / "SUMMARY.md").read_text().split("\n", 1)[1],
        "{{STAGE3}}": (Path(args.stage3) / "SUMMARY.md").read_text().split("\n", 1)[1],
        "{{RUN_DIRS}}": "\n".join(f"- `{Path(p).resolve().relative_to(root)}`" for p in (args.stage1, args.stage2, args.stage3)),
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    (root / "RESULTS.md").write_text(text)
    print("wrote", root / "RESULTS.md")


if __name__ == "__main__":
    main()
