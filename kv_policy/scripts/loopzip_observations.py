#!/usr/bin/env python3
"""Render a falsifiable, source-backed LoopZip Phase-1/2 observation audit.

Reads ONLY actual completed model-run artifacts. Never invents a failure or
claims recurrent causality from an attention-policy screening observation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _flag(value):
    return value == "True" if value in ("True","False") else None


def _num(value):
    return float(value) if value not in (None,"") else None


def _table(rows):
    return "\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)


def render(run: Path, *, require_swaps: bool = False,
           top_n: int = 12) -> dict:
    run=run.resolve()
    completeness=json.loads((run/"completeness.json").read_text())
    if not completeness.get("complete"):
        raise RuntimeError("Refusing conclusions from incomplete Phase-1 run")
    failure=read_csv(run/"analysis"/"failure_cases.csv")
    quality=read_csv(run/"analysis"/"screening_quality.csv")
    manifest=json.loads((run/"run_manifest.json").read_text())
    if manifest["protocol"]!="loopzip-failure-first-fixed-v1":
        raise RuntimeError("Unknown Phase-1 protocol")
    expected={.10,.30,.50}
    if set(float(x["budget"]) for x in failure)!=expected:
        raise RuntimeError("Budget cells not exactly 10/30/50%")
    swap_done=(run/"swap_completeness.json").exists()
    if require_swaps and not swap_done:
        raise RuntimeError("Missing Phase-2 completion report")
    swap_results={}
    if swap_done:
        swap_complete=json.loads((run/"swap_completeness.json").read_text())
        if not swap_complete.get("complete"):
            raise RuntimeError("Refusing partial Phase-2 oracle conclusions")
        swap_dir=run/"swap_raw"
        for path in swap_dir.glob("*.json"):
            obj=json.loads(path.read_text())
            for s in obj["swap_targets"]:
                key=(obj["meta"]["model"],obj["meta"]["dataset"],
                     obj["meta"]["uid"],float(s["budget"]))
                if key in swap_results:
                    raise RuntimeError("Duplicate swap observation")
                swap_results[key]=s
    qkeys=set()
    for x in quality:
        k=(x["model"],x["dataset"],x["uid"],float(x["budget"]),x["policy"])
        if k in qkeys:
            raise RuntimeError("Repeated policy-budget request row")
        qkeys.add(k)
    groups=defaultdict(list)
    scored=[]
    for x in failure:
        budget=float(x["budget"])
        key=(x["model"],x["dataset"],x["uid"],budget)
        if x["category"]=="all_attention_wrong" and _flag(x["full_correct"]) is not True:
            raise RuntimeError("Invalid full-correct requirement for all-attention failure")
        swap=swap_results.get(key)
        gain=swap["local_oracle_gain"] if swap else None
        greedy=swap.get("best_greedy") if swap else None
        rescue=bool(greedy is not None and greedy.get("exact_match") is True
                    and gain is not None and gain>1e-8)
        r={
            "model":x["model"],"dataset":x["dataset"],"uid":x["uid"],
            "budget":budget,"category":x["category"],
            "full_correct":_flag(x["full_correct"]),
            "best_attention_delta_nll":_num(x["best_attn_delta_nll"]),
            "mean_attention_delta_nll":_num(x["mean_delta_nll"]),
            "local_swap_tested":swap is not None,
            "best_swap_nll_gain":gain,
            "greedy_answer_rescued":rescue if greedy is not None else None,
        }
        scored.append(r)
        groups[(r["model"],r["dataset"],budget)].append(r)
    summary=[]
    for (mk,ds,b),members in sorted(groups.items()):
        eligible=[x for x in members if x["full_correct"] is True]
        failed=[x for x in eligible if x["category"]=="all_attention_wrong"]
        probed=[x for x in failed if x["local_swap_tested"]]
        nll_rescues=[x for x in probed if x["best_swap_nll_gain"]>1e-8]
        confirmed=[x for x in nll_rescues if x["greedy_answer_rescued"] is True]
        summary.append({
            "model":mk,"dataset":ds,"budget":b,
            "n_requests":len(members),"n_full_correct":len(eligible),
            "n_all_attention_wrong":len(failed),
            "n_swap_probed_all_attention_wrong":len(probed),
            "n_positive_same_budget_swap_gain":len(nll_rescues),
            "n_confirmed_accuracy_rescues":len(confirmed),
            "n_attn_aggregation_failures":sum(
                m["category"]=="mean_wrong_other_attention_correct" for m in members),
            "n_full_wrong":sum(m["category"]=="full_wrong" for m in members)})
    lines=[
        "# LoopZip Phase 1/2 Observations — measured runs only",
        "",
        f"- Run directory: \`{run.name}\`",
        f"- Frozen source SHA: \`{manifest['code_sha']}\`",
        "- Budgets: **10%, 30%, 50% of aligned Context token positions**; other prompt roles unmasked.",
        "- Attention baselines: Mean, Last, Max, Recent+Mean, Random (three seeds).",
        f"- Phase 1 complete: **yes**; Phase 2 available: **{'yes' if swap_done else 'no'}**.",
        "- Full-size R-plane KV read masks, **not physical HBM savings or speedups**.",
        "",
        "## Budget-wise failure counts",
        "",
        _table([
            ["Model","Dataset","K%","Requests","Full-correct","All-Attn wrong",
             "Swap probed","NLL swap gain","Greedy rescued","Mean-only gap"],
            ["---"]*10,
            *[[s["model"],s["dataset"],int(s["budget"]*100),
               s["n_requests"],s["n_full_correct"],
               s["n_all_attention_wrong"],s["n_swap_probed_all_attention_wrong"],
               s["n_positive_same_budget_swap_gain"],
               s["n_confirmed_accuracy_rescues"],
               s["n_attn_aggregation_failures"]] for s in summary]]),
        "",
        "## Highest residual Attention failures (post-hoc case inventory)",
        "",
        "The cases below are sorted by signed future NLL loss of the BEST",
        "among the tested Attention baselines, not by a convenient weak baseline.",
        "This is selection for *mechanism study*, NOT an unbiased population estimate.",
        "",
    ]
    candidates=[r for r in scored
                if r["full_correct"] is True
                and r["category"]=="all_attention_wrong"
                and r["best_attention_delta_nll"] is not None]
    candidates.sort(key=lambda r:(-r["best_attention_delta_nll"],
                                   r["model"],r["dataset"],r["uid"],r["budget"]))
    if candidates:
        lines += [_table([
            ["Model","Dataset","UID","K%","Best-Attn ΔNLL",
             "Local-swap gain","Accuracy rescue"],
            ["---"]*7,
            *[[x["model"],x["dataset"],x["uid"],int(100*x["budget"]),
               f"{x['best_attention_delta_nll']:.5f}",
               (f"{x['best_swap_nll_gain']:.5f}"
                if x["best_swap_nll_gain"] is not None else "not tested"),
               ("yes" if x["greedy_answer_rescued"] else
                "no" if x["greedy_answer_rescued"] is False else "not scored")]
              for x in candidates[:top_n]])]
    else:
        lines += ["No Full-correct requests where all four Attention baselines",
                  "were wrong were observed in the completed screening cohort."]
    lines += [
        "",
        "## Interpretation constraints",
        "",
        "- If Mean fails but Last/Max works, this supports **aggregation**",
        "  sensitivity, not a general non-attention signal.",
        "- A positive same-budget swap gain shows a tested better KV set",
        "  for that request, not that the global optimum has been found.",
        "- Greedy answer recovery is only reported if optional greedy",
        "  decoding was actually performed in the swap stage.",
        "- A Phase-1/2 Attention selection gap alone does **NOT** prove",
        "  recurrent error amplification caused the failure. Phase 3",
        "  still needs immediate attention-output and loop-specific controls.",
        "- No observed residual gap is a valid NEGATIVE finding; do not",
        "  broaden budgets or choose more convenient problems post-hoc.",
    ]
    report={"schema":"loopzip-observation-audit-v1",
            "run":run.name,"sha":manifest["code_sha"],
            "phase2_available":swap_done,"groups":summary,
            "cases":candidates[:top_n],
            "counts":{"requests_by_budget_cells":len(failure),
                      "all_attention_failure_cases":len(candidates)}}
    out=run/"analysis"
    (out/"OBSERVATIONS.md").write_text("\n".join(lines)+"\n")
    (out/"OBSERVATIONS.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--require-swaps",action="store_true")
    ap.add_argument("--top-n",type=int,default=12)
    args=ap.parse_args()
    if args.top_n<1:
        ap.error("--top-n must be positive")
    result=render(Path(args.run_dir),require_swaps=args.require_swaps,
                  top_n=args.top_n)
    print(json.dumps(result["counts"],indent=2))


if __name__=="__main__":
    main()
