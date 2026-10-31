#!/usr/bin/env python3
"""LoopZip v5 Stage 0 (LOOPZIP_EVIDENCE_CONTROLLED_LOOP_SPECIFIC_PLAN.md §2): CPU-only evidence audit.

Reads a COMPLETE frozen Phase-1/2 run (jobs.json, raw/, swap_raw/), derives exact evidence
token sets from tokenizer offsets, and writes into <run>/analysis/:
  evidence_coverage_per_policy.csv, evidence_coverage_per_swap.csv,
  evidence_position_strata.csv, evidence_audit_integrity.json, EVIDENCE_COVERAGE_FINDINGS.md
No model is loaded, no source file is modified, budgets/requests are not reselected.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvp.loopzip_evidence_audit import (AlignmentError, coverage_row, evidence_position,  # noqa: E402
                                        evidence_spans, token_offsets)
from kvp.specs import MODELS  # noqa: E402

ATTN = ("mean_attn", "last_attn", "max_attn", "recent_attn")
CR_POS = {0: "front", 8: "middle", 15: "end"}


def family(policy_key: str) -> str:
    return "random" if policy_key.startswith("random") else policy_key


def boot_rate(units: dict, n=1000, seed=0):
    """units: unit -> (num, den). Request-grouped bootstrap of sum(num)/sum(den)."""
    keys = [k for k, (_, d) in units.items() if d]
    if not keys:
        return None, None
    tot = sum(units[k][0] for k in keys) / sum(units[k][1] for k in keys)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        s = [units[rng.choice(keys)] for _ in keys]
        draws.append(sum(a for a, _ in s) / sum(b for _, b in s))
    draws.sort()
    return tot, [draws[int(.025 * n)], draws[int(.975 * n) - 1]]


def write_csv(path: Path, rows: list[dict]):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols or ["empty"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    run = Path(a.run_dir).resolve()
    comp = json.loads((run / "completeness.json").read_text())
    scomp = json.loads((run / "swap_completeness.json").read_text())
    if not comp.get("complete") or not scomp.get("complete"):
        raise SystemExit("Phase-1 and Phase-2 completeness must both be TRUE")
    manifest = json.loads((run / "run_manifest.json").read_text())
    jobs = {j["key"]: j for j in json.loads((run / "jobs.json").read_text())}
    toks, evs, integrity = {}, {}, {"alignment_failures": [], "ambiguous_subspans": [],
                                    "boundary_tokens_total": 0}
    for key, j in jobs.items():
        if j["model"] not in toks:
            ck = MODELS[j["model"]]
            toks[j["model"]] = AutoTokenizer.from_pretrained(ck.repo, revision=ck.revision,
                                                             trust_remote_code=True)
        try:
            ev = evidence_spans(j, token_offsets(toks[j["model"]], j["prompt_text"], j["prompt_ids"]))
        except AlignmentError as e:
            integrity["alignment_failures"].append({"key": key, "uid": j["uid"], "error": str(e)})
            continue
        integrity["boundary_tokens_total"] += ev["boundary_tokens"]
        if ev["ambiguous"]:
            integrity["ambiguous_subspans"].append({"key": key, "uid": j["uid"], "spans": ev["ambiguous"]})
        evs[key] = ev

    rows, full = [], {}
    for p in sorted((run / "raw").glob("*.json")):
        res = json.loads(p.read_text())
        key = res["meta"]["key"]
        if key not in evs:
            continue
        j, ev = jobs[key], evs[key]
        full[key] = res["full"]
        fc = bool(res["full"].get("greedy_correct", res["full"].get("exact_match")))
        tp = j["attributes"].get("target_position")
        pos = evidence_position(ev, j["content_positions"])
        for s in res["screening"]:
            if s["n_retained"] != len(s["retained"]) or s["n_retained"] != j["budget_counts"][str(s["retention_requested"])]:
                raise SystemExit(f"Budget mismatch in {key}")
            rows.append({"model": j["model"], "dataset": j["dataset"], "uid": j["uid"], "key": key,
                         "budget": s["retention_requested"], "policy": s["policy_key"],
                         "policy_family": family(s["policy_key"]), "k": s["n_retained"],
                         "n_context": s["n_context"], "full_correct": fc,
                         "policy_correct": bool(s["greedy_correct"]), "delta_nll": s["delta_nll"],
                         "target_position_index": tp,
                         "target_location": CR_POS.get(tp) if j["dataset"] == "controlled_retrieval" else None,
                         "evidence_position_percentile": pos,
                         **coverage_row(s["retained"], ev, j["content_positions"], s.get("attention_scores"))})
    out = run / "analysis"
    write_csv(out / "evidence_coverage_per_policy.csv", rows)

    # ---- swaps
    swaps = []
    for p in sorted((run / "swap_raw").glob("*.json")):
        w = json.loads(p.read_text())
        key = w["meta"]["key"]
        if key not in evs:
            continue
        j, ev = jobs[key], evs[key]
        E, ids, code = set(ev["E"]), set(ev.get("id", [])), set(ev.get("code", []))
        tag = lambda t: ("none" if t is None else "code" if t in code else "id" if t in ids  # noqa: E731
                         else "evidence_other" if t in E else "nonevidence")
        for t in w["swap_targets"]:
            orig = next(s for s in t["swaps"] if s["type"] == "original")
            best = next(s for s in t["swaps"] if s["swap_id"] == t["best_swap_id"])
            co, cb = coverage_row(orig["retained"], ev, j["content_positions"]), \
                coverage_row(best["retained"], ev, j["content_positions"])
            swaps.append({"model": j["model"], "dataset": j["dataset"], "uid": j["uid"],
                          "budget": t["budget"], "role": t["role"], "category": t["category"],
                          "anchor_policy": t["anchor_policy"], "best_swap_id": t["best_swap_id"],
                          "local_oracle_gain": t["local_oracle_gain"],
                          "best_greedy_correct": (t.get("best_greedy") or {}).get("exact_match"),
                          "drop": best["drop"], "restore": best["restore"],
                          "drop_type": tag(best["drop"]), "restore_type": tag(best["restore"]),
                          "coverage_E_before": co["coverage_E"], "coverage_E_after": cb["coverage_E"],
                          "coverage_E_change": cb["coverage_E"] - co["coverage_E"],
                          "complete_E_before": co["complete_E"], "complete_E_after": cb["complete_E"],
                          "code_complete_before": co.get("code_complete"),
                          "code_complete_after": cb.get("code_complete")})
    write_csv(out / "evidence_coverage_per_swap.csv", swaps)

    # ---- strata / findings
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["dataset"], r["budget"], r["policy_family"])].append(r)
    strata, md_cells = [], []
    for (m, d, b, pf), rs in sorted(cells.items()):
        fc = [r for r in rs if r["full_correct"]]
        def units(sel, num):
            u = defaultdict(lambda: [0, 0])
            for r in sel:
                u[(r["model"], r["uid"])][1] += 1
                u[(r["model"], r["uid"])][0] += num(r)
            return {k: tuple(v) for k, v in u.items()}
        fail = [r for r in fc if not r["policy_correct"]]
        resid = [r for r in fail if r["complete_E"]]
        acc_c, ci_c = boot_rate(units([r for r in fc if r["complete_E"]], lambda r: r["policy_correct"]))
        acc_i, ci_i = boot_rate(units([r for r in fc if not r["complete_E"]], lambda r: r["policy_correct"]))
        row = {"model": m, "dataset": d, "budget": b, "policy_family": pf, "n_rows": len(rs),
               "full_correct_rows": len(fc), "flip_rows": len(fail),
               "flips_with_complete_evidence": len(resid),
               "complete_E_rate": sum(r["complete_E"] for r in rs) / len(rs),
               "mean_coverage_E": sum(r["coverage_E"] for r in rs) / len(rs),
               "acc_given_complete_E": acc_c, "acc_given_complete_E_ci95": ci_c,
               "n_complete_E_fullcorrect": sum(r["complete_E"] for r in fc),
               "acc_given_incomplete_E": acc_i, "acc_given_incomplete_E_ci95": ci_i}
        if d == "controlled_retrieval":
            row["code_complete_rate"] = sum(bool(r.get("code_complete")) for r in rs) / len(rs)
            row["flips_with_complete_code"] = sum(bool(r.get("code_complete")) for r in fail)
        strata.append(row)
    # CR target-location x policy
    loc = defaultdict(list)
    for r in rows:
        if r["dataset"] == "controlled_retrieval" and r["full_correct"]:
            loc[(r["model"], r["budget"], r["policy_family"], r["target_location"])].append(r)
    loc_rows = [{"model": m, "budget": b, "policy_family": pf, "target_location": l, "n": len(rs),
                 "accuracy": sum(r["policy_correct"] for r in rs) / len(rs),
                 "code_complete_rate": sum(bool(r.get("code_complete")) for r in rs) / len(rs),
                 "complete_E_rate": sum(r["complete_E"] for r in rs) / len(rs),
                 "acc_given_code_complete": (lambda s: sum(r["policy_correct"] for r in s) / len(s) if s else None)(
                     [r for r in rs if r.get("code_complete")])}
                for (m, b, pf, l), rs in sorted(loc.items())]
    write_csv(out / "evidence_position_strata.csv", strata + [{"section": "cr_target_location", **x} for x in loc_rows])

    attn_fail = [r for r in rows if r["policy_family"] in ATTN and r["full_correct"] and not r["policy_correct"]]
    integrity.update({
        "source_sha_of_audited_run": manifest.get("code_sha"),
        "audit_code_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         capture_output=True).stdout.strip(),
        "jobs": len(jobs), "audited_jobs": len(evs), "policy_rows": len(rows), "swap_targets": len(swaps),
        "phase1_complete": comp["complete"], "phase2_complete": scomp["complete"],
        "attention_flip_rows": len(attn_fail),
        "attention_flip_rows_with_complete_E": sum(r["complete_E"] for r in attn_fail),
        "labels_used_by_policies": False,
        "note": "Evidence labels/offsets are post-hoc diagnostics; Phase-1 selection never saw them."})
    (out / "evidence_audit_integrity.json").write_text(json.dumps(integrity, indent=2))

    fmt = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    cif = lambda c: "" if not c else f" [{c[0]:.2f},{c[1]:.2f}]"  # noqa: E731
    md = ["# LoopZip v5 Stage 0 — evidence coverage audit (CPU, existing results only)", "",
          f"Audited run: `{run.name}` (code {manifest.get('code_sha', '')[:7]}); "
          f"{len(evs)}/{len(jobs)} jobs aligned; alignment failures {len(integrity['alignment_failures'])}; "
          f"ambiguous CR subspans {len(integrity['ambiguous_subspans'])}.", "",
          f"Attention-policy flips (Full correct → policy wrong): **{len(attn_fail)}** rows; "
          f"with the COMPLETE evidence set retained: **{integrity['attention_flip_rows_with_complete_E']}**.", "",
          "Rows are budget×policy cells; the same question appears at 3 budgets (not independent). "
          "CIs: 1000 request-grouped bootstrap resamples.", "",
          "| model | dataset | b | policy | full-correct rows | flips | flips w/ complete E | complete-E rate | "
          "mean cov(E) | acc given complete E | acc given incomplete E |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in strata:
        md.append(f"| {s['model']} | {s['dataset']} | {s['budget']} | {s['policy_family']} | {s['full_correct_rows']} | "
                  f"{s['flip_rows']} | {s['flips_with_complete_evidence']} | {s['complete_E_rate']:.2f} | "
                  f"{s['mean_coverage_E']:.2f} | {fmt(s['acc_given_complete_E'])}{cif(s['acc_given_complete_E_ci95'])} "
                  f"(n={s['n_complete_E_fullcorrect']}) | {fmt(s['acc_given_incomplete_E'])}{cif(s['acc_given_incomplete_E_ci95'])} |")
    md += ["", "## Controlled Retrieval: target location × policy (Full-correct rows)", "",
           "| model | b | policy | location | n | acc | code complete | acc given code complete |", "|---|---|---|---|---|---|---|---|"]
    for x in loc_rows:
        md.append(f"| {x['model']} | {x['budget']} | {x['policy_family']} | {x['target_location']} | {x['n']} | "
                  f"{x['accuracy']:.2f} | {x['code_complete_rate']:.2f} | {fmt(x['acc_given_code_complete'])} |")
    sw = defaultdict(int)
    for s in swaps:
        sw[(s["role"], s["drop_type"], s["restore_type"])] += 1
    md += ["", "## Phase-2 best swaps: what was dropped / restored", "",
           "| role | drop | restore | count |", "|---|---|---|---|"]
    md += [f"| {r} | {d} | {t} | {c} |" for (r, d, t), c in sorted(sw.items())]
    ch = [s for s in swaps if s["coverage_E_change"] != 0]
    md += ["", f"Best swaps that changed evidence coverage: {len(ch)}/{len(swaps)}; "
           f"mean gain when coverage rose {fmt(sum(s['local_oracle_gain'] for s in ch if s['coverage_E_change'] > 0) / max(1, sum(s['coverage_E_change'] > 0 for s in ch)))}, "
           f"when unchanged {fmt(sum(s['local_oracle_gain'] for s in swaps if s['coverage_E_change'] == 0) / max(1, len(swaps) - len(ch)))}.",
           "", "Evidence offsets are post-hoc diagnostics; no recurrence-specific claim follows from this audit."]
    (out / "EVIDENCE_COVERAGE_FINDINGS.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: integrity[k] for k in ("audited_jobs", "policy_rows", "swap_targets",
                                                 "attention_flip_rows", "attention_flip_rows_with_complete_E")}
                     | {"alignment_failures": len(integrity["alignment_failures"]),
                        "ambiguous": len(integrity["ambiguous_subspans"])}, indent=2))


if __name__ == "__main__":
    main()
