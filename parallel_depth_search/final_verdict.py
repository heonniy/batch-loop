"""Apply the FREEZE_20261121.md pre-registered decision rule to an e2e analysis file (CPU only).

usage: python -m parallel_depth_search.final_verdict --analysis RUN/analysis/F1_FULL_ROLLOUT.json [--write]
Per (cell, arm) vs AR on the same cohorts:
  supported_heldout_quality_matched : speedup CI low > 1.03 AND accuracy-delta CI low > -0.03
  contradicted_quality              : accuracy-delta CI high < -0.03
  no_speedup                        : speedup CI high <= 1.03 (and quality not contradicted)
  inconclusive                      : any CI crossing a margin
`exact` is never granted to a candidate arm (candidate KV carry is approximate, G0_KV/V2).
--write adds `verdicts`, `supported`, and top-level classification metrics of the best supported
candidate (or best candidate by speedup if none) back into the analysis JSON.
"""
import argparse
import json

SPEED_MIN, ACC_MARGIN = 1.03, -0.03
TRUNC = ("trunc",)


def verdict(r):
    lo, hi = r["speedup_ci"]
    dlo, dhi = r["acc_delta_ci"]
    if lo > SPEED_MIN and dlo > ACC_MARGIN:
        return "supported_heldout_quality_matched"
    if dhi < ACC_MARGIN:
        return "contradicted_quality"
    if hi <= SPEED_MIN:
        return "no_speedup"
    return "inconclusive"


def sort_key(c):
    f = c.split("|")
    return f[0], f[2], int(f[1][1:]), c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", required=True)
    p.add_argument("--write", action="store_true")
    p.add_argument("--champion", default="", help="'cell-substring:arm' for the node-level metrics; default = "
                   "fastest supported jac_* arm, else fastest jac_* arm")
    a = p.parse_args()
    m = json.load(open(a.analysis))
    V, sup, cands = {}, [], []
    print("OOM:", m.get("oom"))
    for c in sorted(m["cells"], key=sort_key):
        arms = m["cells"][c]
        print(c)
        ar = arms.get("AR")
        if ar:
            print(f"   {'AR':10s} acc={ar['acc_strict']:.3f} tok/s={ar['tok_per_s']:.1f} "
                  f"n={ar['n_prompts']} cohorts={ar['n_cohorts']} hbm={ar['hbm_peak_gb']:.0f}GB")
        for arm, r in arms.items():
            if arm == "AR":
                continue
            v = verdict(r)
            V[f"{c}|{arm}"] = v
            kind = "baseline_truncation" if arm.startswith(TRUNC) else "depth_parallel_candidate"
            rec = dict(cell=c, arm=arm, kind=kind, verdict=v, speedup=r["speedup"], speedup_ci=r["speedup_ci"],
                       acc_delta=r["acc_delta"], acc_delta_ci=r["acc_delta_ci"], seq_exact_rate=r["seq_exact_rate"])
            if kind == "depth_parallel_candidate":
                cands.append(rec)
                if v.startswith("supported"):
                    sup.append(rec)
            sl, sh = r["speedup_ci"]
            dl, dh = r["acc_delta_ci"]
            print(f"   {arm:10s} {r['speedup']:.2f}x [{sl:.2f},{sh:.2f}]  dacc={r['acc_delta']:+.3f} "
                  f"[{dl:+.3f},{dh:+.3f}]  acc={r['acc_strict']:.3f} exact={r['seq_exact_rate']:.2f} "
                  f"rounds={r['rounds_mean']:.2f}  {v}{'' if kind != 'baseline_truncation' else '  (baseline)'}")
    jac = [x for x in cands if x["arm"].startswith("jac")]
    jsup = [x for x in jac if x["verdict"].startswith("supported")]
    if a.champion:
        sub, arm = a.champion.rsplit(":", 1)
        best = next(x for x in cands if sub in x["cell"] and x["arm"] == arm)
    else:
        best = max(jsup or jac, key=lambda x: x["speedup"]) if jac else None
    print("supported depth-parallel candidates:", len(sup))
    if best:
        print("champion:", best["cell"], best["arm"], best["speedup_ci"], best["acc_delta_ci"], best["verdict"])
    if a.write:
        m["verdicts"] = V
        m["supported_candidates"] = sup
        m["champion"] = best
        if best:
            m.update(speedup=best["speedup"], speedup_ci_low=best["speedup_ci"][0],
                     accuracy_delta=best["acc_delta"], accuracy_delta_ci_low=best["acc_delta_ci"][0],
                     fidelity_tier="quality_noninferior" if best["verdict"].startswith("supported") else "approximate",
                     sequence_exact_rate=best["seq_exact_rate"], unverified_kv_fidelity=True)
        json.dump(m, open(a.analysis, "w"), indent=1)


if __name__ == "__main__":
    main()
