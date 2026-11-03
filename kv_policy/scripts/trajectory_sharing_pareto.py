#!/usr/bin/env python3
"""V10 follow-up (journal E14): clean (no eviction) quality-vs-compute curves of native early exit under FULL vs the
architecture-specific shared recurrent KV, swept over a frozen threshold grid.

Clue from the main run: at the SAME threshold, shared-KV queries exit much earlier (Huginn LAST ≈16–20 loops vs FULL ≈30–32)
with ~0 NLL change. That is not a fair compute comparison: FULL at a looser threshold might reach the same depth equally
cheaply. Here both policies are swept over the same grid and compared at matched realized mean depth.

For every prompt (3 calibration + 5 evaluation UIDs of the main V10 cells — no eviction, so no outcome leakage), policy and
threshold: teacher-forced live decode of the gold continuation (realized depths, future NLL, KL to FULL-fixed) and a live
greedy answer (EM). Output: <run>/raw/<uid>.json and analysis/PARETO.md.

  CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_sharing_pareto.py --model huginn-0125 --dataset controlled_retrieval --run-dir results/<dir>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffn_gate_causal_sweep import gold_score, load_backbone, load_v5_jobs  # noqa: E402
from trajectory_sharing_sweep import _maps, _runner, _sha, _spec, _write  # noqa: E402

GRID = {"huginn": (1e-5, 3e-5, 6e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2),
        "ouro": (0.995, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)}


def run(a):
    out = Path(a.run_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    bb, tok, eos = load_backbone(a.model)
    fam = bb.family
    docs = load_v5_jobs(a.model, a.dataset)[: a.n_prompts]
    _write(out / "manifest.json", {"model": a.model, "dataset": a.dataset, "code_sha": _sha(), "grid": GRID[fam],
                                   "uids": [d["uid"] for d in docs], "greedy_tokens": a.greedy_tokens,
                                   "note": "clean only (no eviction); same grid for both KV policies"})
    for doc in docs:
        dest = out / "raw" / (doc["uid"].replace("/", "_") + ".json")
        if dest.exists():
            continue
        t0 = time.time()
        r = _runner(bb, doc, a.greedy_tokens)
        r.reference(eos, frozen_ids=list(doc["target_ids"]))
        rows = []
        full_fixed = r.trajectory_arm(plane_map=None)
        for pol, pmap in _maps(r).items():
            fx = full_fixed if pol == "full" else r.trajectory_arm(plane_map=pmap)
            g = r.greedy((), a.greedy_tokens, eos, plane_map=pmap)
            rows.append({"policy": pol, "threshold": None, "mean_depth": float(bb.R), "nll_future": fx["nll_future"],
                         "kl_to_full_fixed": r.full_vs_evicted_kl(full_fixed, fx)["mean_scored_kl"],
                         "em": float(gold_score(tok, doc["gold_answer"], g)["exact_match"]), "greedy_mean_depth": float(bb.R)})
            for q in GRID[fam]:
                spec = _spec(bb, q)
                lv = r.trajectory_arm(plane_map=pmap, spec=spec, dynamic=True)
                gd = r.greedy_dynamic(sources=(), spec=spec, max_new=a.greedy_tokens, eos_ids=eos, plane_map=pmap)
                rows.append({"policy": pol, "threshold": q, "mean_depth": statistics.mean(lv["depths"]),
                             "nll_future": lv["nll_future"], "kl_to_full_fixed": r.full_vs_evicted_kl(full_fixed, lv)["mean_scored_kl"],
                             "em": float(gold_score(tok, doc["gold_answer"], gd["token_ids"])["exact_match"]),
                             "greedy_mean_depth": statistics.mean(gd["depths"]) if gd["depths"] else None})
        _write(dest, {"uid": doc["uid"], "R": bb.R, "rows": rows, "elapsed_s": time.time() - t0})
        print("done", doc["uid"], flush=True)
    summarize(out)


def summarize(out: Path):
    m = json.loads((out / "manifest.json").read_text())
    data = [json.loads(p.read_text()) for p in sorted((out / "raw").glob("*.json"))]
    if not data:
        return
    R = data[0]["R"]
    md = [f"# Clean quality-vs-compute under native exit: FULL vs shared KV ({m['model']} / {m['dataset']}, {len(data)} prompts)", "",
          "Each row: one threshold, both policies; mean over prompts of realized mean depth (teacher-forced live), future NLL minus "
          "the FULL-fixed NLL of the same prompt, KL to FULL-fixed, greedy EM.", "",
          "| threshold | FULL depth | FULL ΔNLL | FULL KL | FULL EM | SHARED depth | SHARED ΔNLL | SHARED KL | SHARED EM |", "|---|---|---|---|---|---|---|---|---|"]
    pols = [p for p in dict.fromkeys(r["policy"] for r in data[0]["rows"])]
    base = {d["uid"]: next(r["nll_future"] for r in d["rows"] if r["policy"] == "full" and r["threshold"] is None) for d in data}

    def agg(pol, q):
        rs = [(d["uid"], next(r for r in d["rows"] if r["policy"] == pol and r["threshold"] == q)) for d in data]
        return (statistics.mean(r["mean_depth"] for _, r in rs), statistics.mean(r["nll_future"] - base[u] for u, r in rs),
                statistics.mean(r["kl_to_full_fixed"] for _, r in rs), statistics.mean(r["em"] for _, r in rs))
    curves = {p: [] for p in pols}
    for q in [None, *m["grid"]]:
        cells = []
        for p in pols:
            dpt, dn, kl, em = agg(p, q)
            curves[p].append((dpt, dn, kl, em, q))
            cells += [f"{dpt:.2f}", f"{dn:+.4f}", f"{kl:.2e}", f"{em:.2f}"]
        md.append(f"| {'fixed R' if q is None else q} | " + " | ".join(cells) + " |")
    # matched-depth comparison: for each shared point, interpolate FULL's KL at the same mean depth
    md += ["", "## Matched-depth comparison (FULL KL linearly interpolated at the SHARED point's realized mean depth)", "",
           "| SHARED threshold | depth | SHARED KL | FULL KL at same depth | SHARED ΔNLL | FULL ΔNLL at same depth |", "|---|---|---|---|---|---|"]
    full = sorted(curves["full"], key=lambda x: x[0])
    sh = [p for p in pols if p != "full"][0]

    def interp(dpt, idx):
        for a, b in zip(full, full[1:]):
            if a[0] <= dpt <= b[0] and b[0] > a[0]:
                w = (dpt - a[0]) / (b[0] - a[0])
                return a[idx] + w * (b[idx] - a[idx])
        return None
    for dpt, dn, kl, em, q in curves[sh]:
        fk, fn = interp(dpt, 2), interp(dpt, 1)
        md.append(f"| {q} | {dpt:.2f} | {kl:.2e} | {'out of FULL range' if fk is None else f'{fk:.2e}'} | {dn:+.4f} | {'—' if fn is None else f'{fn:+.4f}'} |")
    md.append(f"\nR = {R}. FULL depth range covered: {full[0][0]:.2f}–{full[-1][0]:.2f}.")
    (out / "analysis").mkdir(exist_ok=True)
    (out / "analysis" / "PARETO.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--greedy-tokens", type=int, default=24)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize(Path(a.run_dir))
    else:
        run(a)


if __name__ == "__main__":
    main()
