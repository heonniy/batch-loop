#!/usr/bin/env python3
"""Analysis of validated full-rollout frontier receipts (stdlib only).

Per cell (lane, model, exit, dataset), only cells with status passed / parity_failed are used:
  * throughput(arm) = Σ committed tokens / Σ wall (end-to-end) and / Σ decode_ms; speedup vs the cell's own AR
    (same target), paired prompt-cluster bootstrap 95% CI
  * K_opt (DtV only), runner-up, gap ratio with CI; 'plateau' when gap <= 2% or CI includes 1 (pre-registered)
  * WFD reported separately (its own geometry, no K)
  * quality: task accuracy per arm (scorer used), exact-match-to-AR fraction, first-mismatch median
  * per-prompt AR-vs-SD: fraction of prompts where AR is faster than the cell-level best DtV K
  * post-hoc prompt-level strata (NOT pre-action features): AR mean exit depth tertiles and best-K
    acceptance tertiles -> speedup of K_opt and per-stratum best K
  * held-out K: gsm8k3 pilot prompts (idx < 100) choose K, confirmation prompts (idx >= 100) evaluate
  * repeats lane: per-arm coefficient of variation of wall across the 5 repeats
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

GAP = 1.02


def load(cell):
    rows = []
    for p in cell["files"]:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def per_prompt(rows, field="measured_wall_ms"):
    """{prompt: {arm: [tok, ms]}} summed over repeats."""
    d = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for r in rows:
        v = d[r["prompt_id"]][r["arm"]]
        v[0] += r["tokens_committed"]
        v[1] += r[field]
    return d


def rate(pp, arm, keys):
    t = sum(pp[k][arm][0] for k in keys)
    m = sum(pp[k][arm][1] for k in keys)
    return t / m


def boot(pp, num, den, n=2000, seed=0):
    keys = sorted(pp)
    pt = rate(pp, num, keys) / rate(pp, den, keys)
    rng = random.Random(seed)
    bs = sorted(rate(pp, num, s) / rate(pp, den, s) for s in ([rng.choice(keys) for _ in keys] for _ in range(n)))
    return pt, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def kname(arm):
    return int(arm.split(":K")[1]) if arm.startswith("DtV") else None


def tertiles(vals):
    s = sorted(vals)
    return s[len(s) // 3], s[2 * len(s) // 3]


def analyze_cell(name, cell, n_boot):
    rows = load(cell)
    arms = cell["arms"]
    pp = per_prompt(rows)
    ppd = per_prompt(rows, "decode_ms")
    keys = sorted(pp)
    out = {"cell": name, "status": cell["status"], "n_prompts": len(keys), "arms": {}}
    for arm in arms:
        if arm == "AR":
            continue
        e2e = boot(pp, arm, "AR", n_boot)
        dec = boot(ppd, arm, "AR", n_boot)
        rs = [r for r in rows if r["arm"] == arm]
        acc = [r["correct"] for r in rs if r["correct"] is not None]
        acc_draft = sum(r.get("accepted_count", 0) for r in rs)
        drafts = sum(r.get("draft_tokens", 0) for r in rs)
        fms = [r["first_mismatch_vs_AR"] for r in rs if r["first_mismatch_vs_AR"] is not None]
        out["arms"][arm] = {"speedup_e2e": e2e, "speedup_decode": dec,
                            "tok_per_s": 1000 * rate(pp, arm, keys),
                            "acceptance": acc_draft / drafts if drafts else None,
                            "accuracy": sum(acc) / len(acc) if acc else None,
                            "exact_to_AR": sum(r["exact_match_to_AR"] for r in rs) / len(rs),
                            "first_mismatch_median": statistics.median(fms) if fms else None}
    ar = [r for r in rows if r["arm"] == "AR"]
    accs = [r["correct"] for r in ar if r["correct"] is not None]
    out["AR"] = {"tok_per_s": 1000 * rate(pp, "AR", keys), "accuracy": sum(accs) / len(accs) if accs else None,
                 "mean_tokens": statistics.mean(r["tokens_committed"] for r in ar),
                 "mean_depth": statistics.mean(d for r in ar for d in r["depths"]) if any(r["depths"] for r in ar) else None,
                 "prefill_share": sum(r["prefill_ms"] for r in ar) / sum(r["measured_wall_ms"] for r in ar)}
    dtv = [a for a in arms if a.startswith("DtV")]
    if dtv:
        ranked = sorted(dtv, key=lambda a: -out["arms"][a]["speedup_e2e"][0])
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        gap = boot(pp, best, second, n_boot) if second else None
        plateau = gap is None or gap[0] <= GAP or gap[1] <= 1.0
        near = [a for a in dtv if boot(pp, best, a, 500)[1] <= 1.0 or
                out["arms"][best]["speedup_e2e"][0] / out["arms"][a]["speedup_e2e"][0] <= GAP]
        out["K_opt"] = {"best": best, "second": second, "gap": gap, "plateau": plateau,
                        "tied_with_best": sorted(near, key=kname)}
        # AR vs SD per prompt
        out["ar_faster_than_bestK_frac"] = sum(
            1 for k in keys if pp[k]["AR"][0] / pp[k]["AR"][1] > pp[k][best][0] / pp[k][best][1]) / len(keys)
        out["ar_faster_than_all_sd_frac"] = sum(
            1 for k in keys if all(pp[k]["AR"][0] / pp[k]["AR"][1] > pp[k][a][0] / pp[k][a][1]
                                   for a in arms if a != "AR")) / len(keys)
        # post-hoc prompt strata
        depth = {r["prompt_id"]: statistics.mean(r["depths"]) for r in ar if r["depths"]}
        accb = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["arm"] == best:
                accb[r["prompt_id"]][0] += r.get("accepted_count", 0)
                accb[r["prompt_id"]][1] += r.get("draft_tokens", 0)
        acc_rate = {k: v[0] / v[1] for k, v in accb.items() if v[1]}
        strata = {}
        for sname, feat in (("AR mean exit depth", depth), ("acceptance of K_opt", acc_rate)):
            if len(set(feat.values())) < 3:
                continue
            c1, c2 = tertiles(list(feat.values()))
            for lab, sel in ((f"low (<{c1:.3g})", lambda v: v < c1), (f"mid", lambda v: c1 <= v < c2),
                             (f"high (>={c2:.3g})", lambda v: v >= c2)):
                ks = [k for k in keys if k in feat and sel(feat[k])]
                if len(ks) < 5:
                    continue
                sp = {a: rate(pp, a, ks) / rate(pp, "AR", ks) for a in arms if a != "AR"}
                bk = max(dtv, key=lambda a: sp[a])
                strata.setdefault(sname, []).append({"stratum": lab, "n": len(ks), "best_dtv": bk,
                                                     "best_dtv_speedup": sp[bk], "Kopt_speedup": sp[best],
                                                     "WFD_speedup": next((sp[a] for a in sp if a.startswith("WFD")), None)})
        out["posthoc_strata"] = strata
        # held-out K for gsm8k with pilot/confirm split
        idx = {k: int(k.split(":")[1]) for k in keys if k.startswith("gsm8k:")}
        cal = [k for k in keys if idx.get(k, -1) >= 0 and idx[k] < 100]
        ev = [k for k in keys if idx.get(k, -1) >= 100]
        if len(cal) >= 20 and len(ev) >= 20:
            kc = max(dtv, key=lambda a: rate(pp, a, cal))
            ke = max(dtv, key=lambda a: rate(pp, a, ev))
            sub = {k: pp[k] for k in ev}
            out["heldout_K"] = {"n_cal": len(cal), "n_eval": len(ev), "K_from_cal": kc, "K_best_on_eval": ke,
                                "cal_K_speedup_on_eval": boot(sub, kc, "AR", n_boot),
                                "regret_vs_eval_best": rate(pp, ke, ev) / rate(pp, kc, ev)}
    reps = sorted({r["repeat"] for r in rows})
    if len(reps) > 1:
        cv = {}
        for arm in arms:
            vals = defaultdict(list)
            for r in rows:
                if r["arm"] == arm:
                    vals[r["prompt_id"]].append(r["measured_wall_ms"])
            cv[arm] = statistics.median(statistics.pstdev(v) / statistics.mean(v) for v in vals.values())
        out["repeat_cv_median"] = cv
    return out


def f3(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def render(results, status):
    L = ["# Full-rollout frontier — derived tables", "",
         "SOURCE MEASURED rollouts → DERIVED tables. Speedup = Σtok/Σwall of arm ÷ same-cell AR (same target), "
         "paired prompt bootstrap 95% CI. `plateau` = best vs runner-up gap ≤2% or CI ∋ 1.", "",
         "## RUN_STATUS (validator)", "", "| cell | status | n prompts | notes |", "|---|---|---:|---|"]
    for c, r in status.items():
        note = "; ".join(r["reasons"][:2]) if r["reasons"] else ""
        if r.get("parity"):
            note += f" parity inexact {r['parity']['inexact']}/{r['parity']['n_rollouts']}"
        L.append(f"| {c} | {r['status']} | {r.get('n_prompts')} | {note} |")
    for c, o in results.items():
        L += ["", f"## {c} (n={o['n_prompts']}, {o['status']})", "",
              f"AR: {o['AR']['tok_per_s']:.1f} tok/s, acc {o['AR']['accuracy']}, mean len {o['AR']['mean_tokens']:.0f}, "
              f"mean exit depth {o['AR']['mean_depth']}, prefill share {100*o['AR']['prefill_share']:.1f}%", "",
              "| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, v in o["arms"].items():
            fmt = lambda x: "-" if x is None else f"{x:.3f}"
            L.append(f"| {arm} | {f3(v['speedup_e2e'])} | {f3(v['speedup_decode'])} | {v['tok_per_s']:.1f} | "
                     f"{fmt(v['acceptance'])} | {fmt(v['accuracy'])} | {v['exact_to_AR']:.2f} | {v['first_mismatch_median']} |")
        if "K_opt" in o:
            k = o["K_opt"]
            L += ["", f"K_opt **{k['best']}**, runner-up {k['second']}, gap {f3(k['gap']) if k['gap'] else '-'} → "
                      f"{'**plateau**' if k['plateau'] else 'distinct optimum'}; tied set {k['tied_with_best']}. "
                      f"Prompts where AR beats K_opt: {o['ar_faster_than_bestK_frac']:.2f}; AR beats every SD arm: "
                      f"{o['ar_faster_than_all_sd_frac']:.2f}."]
            if o.get("heldout_K"):
                h = o["heldout_K"]
                L.append(f"Held-out K: pilot (n={h['n_cal']}) picks {h['K_from_cal']} → on confirmation (n={h['n_eval']}) "
                         f"speedup {f3(h['cal_K_speedup_on_eval'])}; eval-best {h['K_best_on_eval']}, regret ×{h['regret_vs_eval_best']:.3f}.")
            for s, lst in o.get("posthoc_strata", {}).items():
                L += ["", f"post-hoc strata by {s} (prompt-level, outcome-derived; not pre-action):", "",
                      "| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |", "|---|---:|---|---:|---:|---:|"]
                for x in lst:
                    w = "-" if x["WFD_speedup"] is None else f"{x['WFD_speedup']:.3f}"
                    L.append(f"| {x['stratum']} | {x['n']} | {x['best_dtv']} | {x['best_dtv_speedup']:.3f} | "
                             f"{x['Kopt_speedup']:.3f} | {w} |")
        if o.get("repeat_cv_median"):
            L += ["", "median per-prompt wall CV over repeats: " +
                  ", ".join(f"{a}: {v:.3f}" for a, v in o["repeat_cv_median"].items())]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True, help="validator output json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    status = json.load(open(a.status))
    res = {c: analyze_cell(c, s, a.boot) for c, s in status.items() if s["status"] in ("passed", "parity_failed")}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out_dir) / "frontier.json", "w"), indent=1)
    md = render(res, status)
    (Path(a.out_dir) / "frontier.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
