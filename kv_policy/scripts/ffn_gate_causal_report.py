"""Offline v9 findings from committed per-prompt results; no GPU / no model load."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def mean_or_none(xs):
    vals=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def boot_by_prompt(rows, key, *, n=1000, seed=20261103):
    """Resample independent PROMPT UIDs, not correlated source windows."""
    units=defaultdict(list)
    for r in rows:
        if r.get(key) is not None and math.isfinite(float(r[key])):
            units[r["uid"]].append(float(r[key]))
    units={u:mean_or_none(v) for u,v in units.items()}
    if len(units)<3:
        return None
    uids=list(sorted(units))
    rng=random.Random(seed)
    draws=[statistics.mean(units[rng.choice(uids)] for _ in uids)
           for _ in range(n)]
    draws.sort()
    return [draws[int(n*0.025)],draws[int(n*0.975)-1]]


def create_findings(rows:list[dict], manifest:dict, complete:bool) -> str:
    lines=[
        "# v9 causal gate and dynamic exit: results readout",
        "",
        "Status: "+("all requested pilot units complete" if complete else "INCOMPLETE: missing/failed units"),
        "Model "+manifest["model"]+", dataset "+manifest["dataset"]+
        ", pilot UIDs "+str(len(manifest["evaluation_uids"]))+
        ", source window width "+str(manifest["width"]),
        "",
        "Important: FFN gate restoration is a privileged oracle, not a deployed policy.",
        "A positive gate repair is NOT, by itself, proof of loop-specific causality.",
        "Dynamic live experiments have actual exit decisions; replay arms force clean depths.",
        "The Huginn FP32 cosine exit is diagnostic on BF16 states.",
        "",
        "## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])",
        "",
        "| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    fmt=lambda x: "NA" if x is None else f"{x:.4g}"
    strata=defaultdict(list)
    for r in rows:
        strata[r["stratum"]].append(r)
    for strat,group in sorted(strata.items()):
        # Each entry is paired to the SAME Full and Evicted reference.
        vals=[mean_or_none([v.get("recovery_"+key) for v in group])
              for key in ("restore_gate","restore_up","shift_gate","restore_ffn")]
        lines.append("| "+strat+" | "+str(len(group))+" | "+fmt(mean_or_none([x.get("eviction_penalty") for x in group]))+
                     " | "+fmt(vals[0])+" "+str(boot_by_prompt(group,"recovery_restore_gate"))+
                     " | "+fmt(vals[1])+" | "+fmt(vals[2])+" | "+fmt(vals[3])+" |")
    lines += [
        "",
        "## Loop-local gate restoration (positive recovery = correct gate patch helps)",
        "",
        "| stratum | loops | mean NLL recovered | bootstrap by prompt |",
        "|---|---|---:|---|"
    ]
    for strat,group in sorted(strata.items()):
        keys=sorted((k for k in group[0] if k.startswith("recovery_gate_loops_")))
        for k in keys:
            lines.append("| "+strat+" | "+k.removeprefix("recovery_gate_loops_")+
                         " | "+fmt(mean_or_none([x.get(k) for x in group]))+
                         " | "+str(boot_by_prompt(group,k))+" |")
    lines += [
        "",
        "## Exit and answer interaction (same clean-only frozen threshold)",
        "",
        "| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |",
        "|---|---|---:|---:|---:|---:|---:|"
    ]
    for strat,group in sorted(strata.items()):
        for i,selected in enumerate(manifest["thresholds"]["selected"]):
            vals={
                "depth":mean_or_none([float(r.get(f"depth_changed_{i}",0)>0) for r in group]),
                "answer":mean_or_none([float(bool(r.get(f"answer_changed_{i}",False))) for r in group]),
                "live":mean_or_none([r.get(f"nll_live_{i}") for r in group]),
                "replay":mean_or_none([r.get(f"nll_replay_{i}") for r in group]),
                "gate":mean_or_none([r.get(f"nll_gate_replay_{i}") for r in group]),
            }
            lines.append("| "+strat+" | "+str(selected["threshold"])+" | "+
                         " | ".join(fmt(vals[k]) for k in ("depth","answer","live","replay","gate"))+" |")
    lines += [
        "",
        "## GO / NO-GO requirements",
        "",
        "One must observe a non-noise eviction penalty and a reproducible gate-only",
        "repair that beats up/shift controls, then verify loop-local effect beyond",
        "ordinary layer-wise FFN patching before claiming a loop-specific mechanism.",
        "If live exit changes depth but answers are stable, do not claim quality impact.",
        "If only ORACLE gate patches help, the method remains an offline diagnosis.",
        "This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.",
        ""
    ]
    return "\n".join(lines)


def write_findings(run:Path, rows:list[dict], manifest:dict, complete:bool):
    path=run/"analysis"/"FFN_GATE_CAUSAL_FINDINGS.md"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(create_findings(rows,manifest,complete))
    return path
