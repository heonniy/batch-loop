#!/usr/bin/env python3
"""Pool several v9 run directories (one per model x dataset) into one cross-cell findings file (CPU only).

Usage: ffn_gate_causal_cross.py <out_dir> <run_dir> [<run_dir> ...]
Writes <out_dir>/V9_CROSS_FINDINGS.md, cross_windows.csv, cross_summary.json.
All CIs are prompt-UID bootstraps (windows within a prompt are correlated).
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _f(x, d=4):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}g}"


def boot(rows, key, n=2000, seed=20261103):
    units = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None or v == "" or (isinstance(v, float) and not math.isfinite(v)):
            continue
        units[r["uid"]].append(float(v))
    if len(units) < 3:
        return None
    means = {u: statistics.mean(v) for u, v in units.items()}
    uids = sorted(means)
    rng = random.Random(seed)
    draws = sorted(statistics.mean(means[rng.choice(uids)] for _ in uids) for _ in range(n))
    return [draws[int(0.025 * n)], draws[int(0.975 * n) - 1]]


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "") and math.isfinite(float(r[key]))]
    return statistics.mean(vals) if vals else None


def load(run: Path):
    man = json.loads((run / "manifest.json").read_text())
    rows = []
    for uid in man["evaluation_uids"]:
        p = run / "raw" / (uid.replace("/", "_") + ".json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for c in d["candidates"]:
            ev = c["evict_fixed"]
            rec = {"model": man["model"], "dataset": man["dataset"], "uid": uid, "stratum": c["stratum"],
                   "source": c["source"][0], "width": d["width"], "full_nll": d["full_fixed_nll"],
                   "evict_dnll": ev["delta_nll_vs_clean"], "evict_kl": ev["kl_to_clean"],
                   "full_em": float(d["full_fixed_grade"]["exact_match"]), "evict_em": float(c["evicted_fixed_grade"]["exact_match"])}
            for name, x in c["restorations"].items():
                rec[f"rec_{name}"] = x["relative_nll_recovery"]["nll_recovered"]
                rec[f"kl_{name}"] = x["kl_to_clean"]
                rec[f"patch_{name}"] = x["mean_relative_patch"]
            tr = c.get("gate_trace", {}).get("by_loop", [])
            for x in tr:
                rec[f"gatechg_loop{x['loop']}"] = x.get("gate_mean_relative_change")
                rec[f"ffnchg_loop{x['loop']}"] = x.get("ffn_mean_relative_change")
                rec[f"gateflip_loop{x['loop']}"] = x.get("gate_mean_sign_flip_frac")
            for j, dy in enumerate(c["dynamic"]):
                rec[f"thr_{j}"] = dy["threshold"]
                rec[f"depthchg_{j}"] = dy["n_depth_changed"] / max(1, len(dy["depth_deltas"]))
                rec[f"sumdepth_{j}"] = dy["sum_depth_delta"]
                rec[f"live_dnll_{j}"] = dy["evicted_live_nll"] - d["full_dynamic"][j]["nll_future"]
                rec[f"replay_dnll_{j}"] = dy["evicted_replay_nll"] - d["full_dynamic"][j]["nll_future"]
                rec[f"gate_replay_rec_{j}"] = dy["gate_restored_replay_nll_recovery"]
                rec[f"oracle_live_exitchg_{j}"] = dy["gate_restored_live_ORACLE_delta_depth_vs_evict"] / max(1, len(dy["depth_deltas"]))
                rec[f"answer_changed_{j}"] = float(dy["live_answer_changed"])
                rec[f"live_em_{j}"] = float(dy["evicted_live_grade"]["exact_match"])
                rec[f"full_live_em_{j}"] = float(dy["full_live_grade"]["exact_match"])
                rec[f"live_vs_fixed_flip_{j}"] = float(dy["live_vs_fixed_correctness_changed"])
            rows.append(rec)
    return man, rows


def main(out: Path, runs: list[Path]):
    out.mkdir(parents=True, exist_ok=True)
    allrows, mans = [], {}
    for run in runs:
        man, rows = load(run)
        mans[(man["model"], man["dataset"])] = man | {"run": str(run), "n_done": len({r["uid"] for r in rows})}
        allrows += rows
    cols = list(dict.fromkeys(k for r in allrows for k in r))
    with (out / "cross_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    md = ["# v9 cross-cell findings: Context-KV eviction → recurrent FFN gate → exit / answer", "",
          "Cells: " + "; ".join(f"{m}/{d} ({v['n_done']}/{len(v['evaluation_uids'])} prompts, thresholds "
                               + ", ".join(str(s['threshold']) for s in v['thresholds']['selected']) + ")" for (m, d), v in sorted(mans.items())), "",
          "All arms share the same LAST-plane-sharing cache (every historical recurrent read uses the terminal plane); EVICT masks a "
          "contiguous 4-token Context window for every generated query. Restorations patch ONLY the named FFN activation of the generated "
          "queries with the FULL run's value (privileged oracle). Positive 'recovered' = the patch removes eviction damage. CIs: prompt bootstrap.", ""]
    # 1. floors and penalties
    md += ["## 1. Eviction penalty and BF16 floor by stratum", "",
           "| model | dataset | stratum | n windows | n prompts | mean ΔNLL evict [CI] | mean KL evict | frac windows ΔNLL > 0.01 | EM full → evict |",
           "|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in allrows:
        groups[(r["model"], r["dataset"], r["stratum"])].append(r)
    summ = {}
    for (m, d, s), g in sorted(groups.items()):
        np_ = len({r["uid"] for r in g})
        row = {"n": len(g), "n_prompts": np_, "evict_dnll": mean(g, "evict_dnll"), "evict_dnll_ci": boot(g, "evict_dnll"), "evict_kl": mean(g, "evict_kl"),
               "frac_harm": statistics.mean(float(r["evict_dnll"] > 0.01) for r in g), "full_em": mean(g, "full_em"), "evict_em": mean(g, "evict_em")}
        summ[f"{m}/{d}/{s}"] = row
        ci = row["evict_dnll_ci"]
        md.append(f"| {m} | {d} | {s} | {len(g)} | {np_} | {_f(row['evict_dnll'])}" + (f" [{ci[0]:.3g},{ci[1]:.3g}]" if ci else "")
                  + f" | {_f(row['evict_kl'])} | {_f(row['frac_harm'], 2)} | {_f(row['full_em'], 2)} → {_f(row['evict_em'], 2)} |")
    md.append("")
    # 2. restorations
    arms = ["restore_gate", "restore_up", "shift_gate", "restore_ffn"]
    md += ["## 2. FFN restoration on the evicted branch: NLL recovered (positive = helps) and fraction of the penalty", "",
           "| model | dataset | stratum | penalty | gate [CI] | up | shifted-gate (direction control) | full FFN | gate/penalty | ffn/penalty | gate − up [CI] | gate − shift [CI] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, d, s), g in sorted(groups.items()):
        pen = mean(g, "evict_dnll")
        vals = {a: mean(g, f"rec_{a}") for a in arms}
        for r in g:
            r["gate_minus_up"] = (r.get("rec_restore_gate") or 0) - (r.get("rec_restore_up") or 0)
            r["gate_minus_shift"] = (r.get("rec_restore_gate") or 0) - (r.get("rec_shift_gate") or 0)
        ci = boot(g, "rec_restore_gate")
        c1, c2 = boot(g, "gate_minus_up"), boot(g, "gate_minus_shift")
        summ[f"{m}/{d}/{s}"].update({"rec": vals, "gate_ci": ci, "gate_minus_up": mean(g, "gate_minus_up"), "gate_minus_up_ci": c1,
                                     "gate_minus_shift": mean(g, "gate_minus_shift"), "gate_minus_shift_ci": c2})
        frac = lambda a: (_f(vals[a] / pen, 2) if pen and pen > 1e-3 and vals[a] is not None else "—")  # noqa: E731
        md.append(f"| {m} | {d} | {s} | {_f(pen)} | {_f(vals['restore_gate'])}" + (f" [{ci[0]:.3g},{ci[1]:.3g}]" if ci else "")
                  + f" | {_f(vals['restore_up'])} | {_f(vals['shift_gate'])} | {_f(vals['restore_ffn'])} | {frac('restore_gate')} | {frac('restore_ffn')} | "
                  f"{_f(mean(g, 'gate_minus_up'))}" + (f" [{c1[0]:.3g},{c1[1]:.3g}]" if c1 else "") + f" | {_f(mean(g, 'gate_minus_shift'))}"
                  + (f" [{c2[0]:.3g},{c2[1]:.3g}]" if c2 else "") + " |")
    md.append("")
    # 3. loop-local
    md += ["## 3. Loop-local gate restoration (NLL recovered when only the gates of the named loops are patched)", "",
           "| model | dataset | stratum | " + " | ".join(k for k in ("1–8 / 1–2", "9–16 / 3–4", "17–24", "25–32")) + " | all loops |", "|---|---|---|---|---|---|---|---|"]
    for (m, d, s), g in sorted(groups.items()):
        keys = sorted((k for k in g[0] if k.startswith("rec_gate_loops_")), key=lambda k: int(k.split("_")[3]))
        vals = [mean(g, k) for k in keys] + [None] * (4 - len(keys))
        summ[f"{m}/{d}/{s}"]["loop_local"] = {k: mean(g, k) for k in keys}
        md.append(f"| {m} | {d} | {s} | " + " | ".join(_f(v) for v in vals) + f" | {_f(mean(g, 'rec_restore_gate'))} |")
    md.append("")
    # 4. gate trace by loop (evicted vs full, no patch)
    md += ["## 4. How much does eviction change the recurrent FFN gate, by loop (relative L2 of pre-SiLU gate; sign-flip fraction)", "",
           "| model | dataset | stratum | loop → gate relative change / sign-flip fraction |", "|---|---|---|---|"]
    for (m, d, s), g in sorted(groups.items()):
        loops = sorted({int(k.split("loop")[1]) for k in g[0] if k.startswith("gatechg_loop")})
        sel = [l for l in loops if l in (1, 2, 3, 4, 8, 16, 24, 32)]
        md.append(f"| {m} | {d} | {s} | " + ", ".join(f"r{l}: {_f(mean(g, f'gatechg_loop{l}'), 3)} / {_f(mean(g, f'gateflip_loop{l}'), 3)}" for l in sel) + " |")
    md.append("")
    # 5. dynamic exit
    md += ["## 5. Native dynamic exit under eviction (clean-only thresholds): depth change, live vs replay vs gate-restored replay, answers", "",
           "| model | dataset | stratum | threshold | frac queries depth changed | mean Σdepth Δ | live ΔNLL | replay ΔNLL | gate-replay recovered | oracle gate-live: frac exits changed vs evict-live | frac answer changed (live full vs live evict) | EM full-live → evict-live | frac live≠fixed correctness |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, d, s), g in sorted(groups.items()):
        for j in range(2):
            if f"thr_{j}" not in g[0]:
                continue
            md.append(f"| {m} | {d} | {s} | {g[0][f'thr_{j}']} | {_f(mean(g, f'depthchg_{j}'), 3)} | {_f(mean(g, f'sumdepth_{j}'), 3)} | {_f(mean(g, f'live_dnll_{j}'))} | "
                      f"{_f(mean(g, f'replay_dnll_{j}'))} | {_f(mean(g, f'gate_replay_rec_{j}'))} | {_f(mean(g, f'oracle_live_exitchg_{j}'), 3)} | "
                      f"{_f(mean(g, f'answer_changed_{j}'), 2)} | {_f(mean(g, f'full_live_em_{j}'), 2)} → {_f(mean(g, f'live_em_{j}'), 2)} | {_f(mean(g, f'live_vs_fixed_flip_{j}'), 2)} |")
    md.append("")
    (out / "V9_CROSS_FINDINGS.md").write_text("\n".join(md))
    (out / "cross_summary.json").write_text(json.dumps({"cells": {f"{k[0]}/{k[1]}": v for k, v in mans.items()}, "strata": summ}, indent=1, default=float))
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), [Path(x).resolve() for x in sys.argv[2:]])
