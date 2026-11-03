#!/usr/bin/env python3
"""V10 cross-cell pooling (CPU): Full vs architecture-specific shared KV under Context eviction.

Usage: trajectory_sharing_cross.py <out_dir> <run_dir> [<run_dir> ...]
Writes <out_dir>/V10_CROSS_FINDINGS.md, cross_windows.csv, cross_summary.json.
CIs: prompt-UID bootstrap (windows of one prompt are correlated). Main (label-blind) and oracle strata are never pooled.
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


def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}g}"


def boot(rows, key, n=2000, seed=20261103):
    units = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        units[r["uid"]].append(float(v))
    if len(units) < 3:
        return None
    means = {u: statistics.mean(v) for u, v in units.items()}
    uids = sorted(means)
    rng = random.Random(seed)
    d = sorted(statistics.mean(means[rng.choice(uids)] for _ in uids) for _ in range(n))
    return [d[int(0.025 * n)], d[int(0.975 * n) - 1]]


def mean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return statistics.mean(v) if v else None


def ci(rows, key):
    c = boot(rows, key)
    return f"{_f(mean(rows, key))}" + (f" [{c[0]:.3g}, {c[1]:.3g}]" if c else "")


def load(run: Path):
    m = json.loads((run / "manifest.json").read_text())
    sh = m["shared_policy"]
    thr = m["thresholds"]["selected"]
    rows, prompts = [], []
    for uid in m["evaluation_uids"]:
        p = run / "raw" / (uid.replace("/", "_") + ".json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cp = d["clean_policy_comparison"]
        pr = {"model": m["model"], "dataset": m["dataset"], "uid": uid, "shared": sh,
              "clean_shared_minus_full_nll": cp["shared_minus_full_nll"], "clean_full_vs_shared_kl": cp["full_vs_shared_KL"]["mean_scored_kl"],
              "clean_full_vs_shared_traj_final": statistics.mean(cp["full_vs_shared_trajectory"]["per_query_final_relative_error"]),
              "full_fixed_em": float(d["clean"]["full"]["fixed_grade"]["exact_match"]),
              "shared_fixed_em": float(d["clean"][sh]["fixed_grade"]["exact_match"]),
              "full_floor": d["clean"]["full"]["repeat_logprob_floor"], "shared_floor": d["clean"][sh]["repeat_logprob_floor"]}
        for j, t in enumerate(thr):
            q = str(t["threshold"])
            for pol in ("full", sh):
                lv = d["clean"][pol]["live"][q]
                R = 32 if m["family"] == "huginn" else 4
                pr[f"{pol}_clean_live_mean_depth_{j}"] = statistics.mean(lv["arm"]["depths"])
                pr[f"{pol}_clean_live_early_{j}"] = statistics.mean(float(x < R) for x in lv["arm"]["depths"])
                pr[f"{pol}_clean_live_em_{j}"] = float(lv["greedy_grade"]["exact_match"])
                pr[f"{pol}_clean_live_dnll_vs_fixed_{j}"] = lv["arm"]["nll_future"] - d["clean"][pol]["fixed"]["nll_future"]
        prompts.append(pr)
        for c in d["candidates"]:
            it = c["fixed_eviction_sharing_interaction"]
            r = {"model": m["model"], "dataset": m["dataset"], "uid": uid, "stratum": c["stratum"], "shared": sh,
                 "full_pen": it["full_eviction_penalty"], "shared_pen": it["shared_eviction_penalty"],
                 "interaction": it["sharing_minus_full_eviction_penalty"]}
            for dd, x in c["depth_probe_interaction"].items():
                r[f"d{dd}_full"] = x["full_penalty"]
                r[f"d{dd}_shared"] = x["shared_penalty"]
                r[f"d{dd}_inter"] = x["shared_minus_full_penalty"]
            for pol, tag in (("full", "full"), (sh, "shared")):
                a = c["by_policy"][pol]
                tr = a["recurrent_trajectory"]
                r[f"{tag}_kl"] = a["fixed_kl"]["mean_scored_kl"]
                r[f"{tag}_E_final"] = tr["mean_relative_error_by_loop"][-1]
                r[f"{tag}_E_first"] = tr["mean_relative_error_by_loop"][1]
                r[f"{tag}_E_curve"] = tr["mean_relative_error_by_loop"]
                r[f"{tag}_abs_final"] = tr["mean_absolute_error_by_loop"][-1]
                r[f"{tag}_em_fixed_flip"] = float(a["evicted_fixed_grade"]["exact_match"] != a["clean_fixed_grade"]["exact_match"])
                r[f"{tag}_ans_fixed_changed"] = float(a["evicted_fixed_grade"]["extracted_answer"] != a["clean_fixed_grade"]["extracted_answer"])
                for j, t in enumerate(thr):
                    x = a["dynamic"][str(t["threshold"])]
                    r[f"{tag}_exitchg_{j}"] = x["depth_changed_fraction"]
                    r[f"{tag}_sumdepth_{j}"] = x["sum_depth_delta"]
                    r[f"{tag}_would_{j}"] = x["would_exit_from_fixed"]["fraction_changed"]
                    r[f"{tag}_live_pen_{j}"] = x["live_nll_eviction_penalty"]
                    r[f"{tag}_replay_pen_{j}"] = x["replay_nll_eviction_penalty"]
                    r[f"{tag}_lr_gap_{j}"] = x["live_minus_replay_nll_descriptive"]
                    r[f"{tag}_tok_changed_{j}"] = float(x["token_sequence_changed_vs_clean_live"])
                    r[f"{tag}_ans_changed_{j}"] = float(x["extracted_answer_changed_vs_clean_live"])
                    r[f"{tag}_corr_flip_{j}"] = float(x["correctness_changed_vs_clean_live"])
            for j in range(len(thr)):
                r[f"live_inter_{j}"] = r[f"shared_live_pen_{j}"] - r[f"full_live_pen_{j}"]
            rows.append(r)
    return m, rows, prompts


def main(out: Path, runs: list[Path]):
    out.mkdir(parents=True, exist_ok=True)
    cells, allrows, allprompts = {}, [], []
    for run in runs:
        m, rows, prompts = load(run)
        cells[(m["model"], m["dataset"])] = (m, rows, prompts)
        allrows += rows
        allprompts += prompts
    flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in allrows]
    cols = list(dict.fromkeys(k for r in flat for k in r))
    with (out / "cross_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flat)
    md = ["# V10 cross-cell findings: FULL vs architecture-specific shared recurrent KV × Context eviction × trajectories × exit", "",
          "Shared policy: Huginn = LAST plane for every query loop; Ouro = FIRST plane for query loop 1, LAST plane for loops ≥2. Same "
          "prompts, same frozen gold continuation, same 4-token windows (4 label-blind 'main' + 1 evidence + 1 distractor oracle per "
          "prompt). Interaction = (shared eviction ΔNLL) − (full eviction ΔNLL); positive = sharing makes the model MORE sensitive to "
          "eviction. CIs: prompt bootstrap (5 prompts per cell: wide).", ""]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        md.append(f"- {mo}/{ds}: {len(prompts)}/{len(m['evaluation_uids'])} prompts; thresholds "
                  + "; ".join(f"{t['threshold']:.3g} ({t.get('anchor', 'pooled')}, clean early full/shared "
                              + "/".join(f"{v:.2f}" for v in t.get("clean_early_fraction_by_policy", {}).values()) + ")" for t in m["thresholds"]["selected"]))
    md.append("")
    # 0. clean policy comparison
    md += ["## 0. Clean (no eviction) cost of the shared policy and numerical floors", "",
           "| model | dataset | shared − full NLL [CI] | KL(full‖shared) | final-loop state rel. diff full vs shared | EM fixed full → shared | repeat floor (logprob) |",
           "|---|---|---|---|---|---|---|"]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        md.append(f"| {mo} | {ds} | {ci(prompts, 'clean_shared_minus_full_nll')} | {_f(mean(prompts, 'clean_full_vs_shared_kl'))} | "
                  f"{_f(mean(prompts, 'clean_full_vs_shared_traj_final'))} | {_f(mean(prompts, 'full_fixed_em'), 2)} → {_f(mean(prompts, 'shared_fixed_em'), 2)} | "
                  f"{_f(max(max(p['full_floor'], p['shared_floor']) for p in prompts) if prompts else None)} |")
    md.append("")
    # 1. H2 interaction at native depth
    md += ["## 1. H2 — eviction penalty under FULL vs SHARED at native depth (fixed R), and the interaction", "",
           "| model | dataset | stratum | n | full ΔNLL [CI] | shared ΔNLL [CI] | interaction [CI] | KL full / shared | EM flips full / shared |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        for s in sorted({r["stratum"] for r in rows}):
            g = [r for r in rows if r["stratum"] == s]
            md.append(f"| {mo} | {ds} | {s} | {len(g)} | {ci(g, 'full_pen')} | {ci(g, 'shared_pen')} | {ci(g, 'interaction')} | "
                      f"{_f(mean(g, 'full_kl'))} / {_f(mean(g, 'shared_kl'))} | {_f(mean(g, 'full_em_fixed_flip'), 2)} / {_f(mean(g, 'shared_em_fixed_flip'), 2)} |")
    md.append("")
    # 2. matched depth
    md += ["## 2. H2 at matched forced depth (eviction ΔNLL full / shared, interaction), main + oracle strata pooled per label", "",
           "| model | dataset | stratum | depth → full / shared / interaction |", "|---|---|---|---|"]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        depths = sorted({int(k[1:].split("_")[0]) for k in rows[0] if k.startswith("d") and k.endswith("_inter")}) if rows else []
        for s in sorted({r["stratum"] for r in rows}):
            g = [r for r in rows if r["stratum"] == s]
            md.append(f"| {mo} | {ds} | {s} | " + "; ".join(f"d={d}: {_f(mean(g, f'd{d}_full'))} / {_f(mean(g, f'd{d}_shared'))} / {_f(mean(g, f'd{d}_inter'))}" for d in depths) + " |")
    md.append("")
    # 3. H1 trajectory
    md += ["## 3. H1 — eviction-induced change of the generated query's recurrent state (relative to the same policy's clean run)", "",
           "| model | dataset | stratum | E full: loop 1 → final | E shared: loop 1 → final | |Δh| final full / shared | per-loop curve (shared − full) at r=1, R/4, R/2, R |",
           "|---|---|---|---|---|---|---|"]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        for s in sorted({r["stratum"] for r in rows}):
            g = [r for r in rows if r["stratum"] == s]
            R = len(g[0]["full_E_curve"]) - 1
            idx = sorted({1, max(1, R // 4), max(1, R // 2), R})
            diff = [statistics.mean(r["shared_E_curve"][i] - r["full_E_curve"][i] for r in g) for i in idx]
            md.append(f"| {mo} | {ds} | {s} | {_f(mean(g, 'full_E_first'))} → {_f(mean(g, 'full_E_final'))} | {_f(mean(g, 'shared_E_first'))} → {_f(mean(g, 'shared_E_final'))} | "
                      f"{_f(mean(g, 'full_abs_final'))} / {_f(mean(g, 'shared_abs_final'))} | " + ", ".join(f"r{i}: {_f(v)}" for i, v in zip(idx, diff)) + " |")
    md.append("")
    # 4. clean live behaviour
    md += ["## 4. Clean native exit under each policy (same threshold for both) — realized depths and their cost", "",
           "| model | dataset | threshold (anchor) | clean mean depth full / shared | clean early fraction full / shared | live − fixed ΔNLL full / shared | greedy EM live full / shared |",
           "|---|---|---|---|---|---|---|"]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        sh = m["shared_policy"]
        for j, t in enumerate(m["thresholds"]["selected"]):
            md.append(f"| {mo} | {ds} | {t['threshold']:.3g} ({t.get('anchor', 'pooled')}) | {_f(mean(prompts, f'full_clean_live_mean_depth_{j}'))} / {_f(mean(prompts, f'{sh}_clean_live_mean_depth_{j}'))} | "
                      f"{_f(mean(prompts, f'full_clean_live_early_{j}'), 2)} / {_f(mean(prompts, f'{sh}_clean_live_early_{j}'), 2)} | "
                      f"{_f(mean(prompts, f'full_clean_live_dnll_vs_fixed_{j}'))} / {_f(mean(prompts, f'{sh}_clean_live_dnll_vs_fixed_{j}'))} | "
                      f"{_f(mean(prompts, f'full_clean_live_em_{j}'), 2)} / {_f(mean(prompts, f'{sh}_clean_live_em_{j}'), 2)} |")
    md.append("")
    # 5. H3/H4 exit under eviction
    md += ["## 5. H3/H4 — eviction × exit: would-exit (from fixed-depth traces), live depth change, live penalty, answers", "",
           "| model | dataset | stratum | threshold (anchor) | would-exit changed full / shared | live depth changed full / shared | Σdepth Δ full / shared | live ΔNLL full / shared | live interaction [CI] | live−replay full / shared | answer changed full / shared | correctness flip full / shared |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (mo, ds), (m, rows, prompts) in sorted(cells.items()):
        for s in sorted({r["stratum"] for r in rows}):
            g = [r for r in rows if r["stratum"] == s]
            for j, t in enumerate(m["thresholds"]["selected"]):
                md.append(f"| {mo} | {ds} | {s} | {t['threshold']:.3g} ({t.get('anchor', 'pooled')}) | {_f(mean(g, f'full_would_{j}'), 2)} / {_f(mean(g, f'shared_would_{j}'), 2)} | "
                          f"{_f(mean(g, f'full_exitchg_{j}'), 2)} / {_f(mean(g, f'shared_exitchg_{j}'), 2)} | {_f(mean(g, f'full_sumdepth_{j}'))} / {_f(mean(g, f'shared_sumdepth_{j}'))} | "
                          f"{_f(mean(g, f'full_live_pen_{j}'))} / {_f(mean(g, f'shared_live_pen_{j}'))} | {ci(g, f'live_inter_{j}')} | "
                          f"{_f(mean(g, f'full_lr_gap_{j}'))} / {_f(mean(g, f'shared_lr_gap_{j}'))} | {_f(mean(g, f'full_ans_changed_{j}'), 2)} / {_f(mean(g, f'shared_ans_changed_{j}'), 2)} | "
                          f"{_f(mean(g, f'full_corr_flip_{j}'), 2)} / {_f(mean(g, f'shared_corr_flip_{j}'), 2)} |")
    md.append("")
    (out / "V10_CROSS_FINDINGS.md").write_text("\n".join(md))
    (out / "cross_summary.json").write_text(json.dumps({"cells": [f"{a}/{b}" for a, b in cells], "n_windows": len(allrows), "n_prompts": len(allprompts)}, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), [Path(x).resolve() for x in sys.argv[2:]])
