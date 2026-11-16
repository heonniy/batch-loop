#!/usr/bin/env python3
"""CPU summary of run 20261116_gpu receipts -> summary.json + summary.md (no model, no GPU).

Derived quantities (all from recorded receipts; definitions are fixed here, before reading results):
  Jacobi k_stable   : per (token, seq), smallest k such that the final-depth greedy token equals the exact
                      token at every round >= k (k=D always holds: exact). Effective sequential core rounds
                      = 1 (real h1) + k_stable; serial needs D.
  Jacobi gate(tau)  : online stop at the first round whose residual max_r ||z^k_r - z^{k-1}_r||/||z^k_r|| < tau
                      (else round D). Reported: mean rounds, top-1 agreement, mean KL at the stop.
  Parareal k_stable : same on Parareal rounds; sequential calls/round = b (fine, blocks batched)
                      + nb serial coarse calls (onestep G; identity G costs 0) + 1 batched G(q_old) call.
  Projection        : NOT an E5 measurement. Speed ratio implied by measured E1 component latencies:
                      serial = D * T_F(B);  jacobi = T_F(B) + k * T_F(D*B)   (T_F(D*B) = width-D call),
                      evaluated at the mean k_stable and at gate(tau) rounds. Excludes coda, predictor,
                      host packing differences, fallback and commit -> an upper bound.
  E4 amplification  : ||dh_{r+k}|| / ||dh_r||, median over (token, seq) per k; flip rate of final token.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

TAUS = (1e-1, 3e-2, 1e-2, 1e-3)


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def k_stable(top1):
    k = len(top1)
    for i in range(len(top1) - 1, -1, -1):
        if top1[i]:
            k = i + 1
        else:
            break
    return k if top1[-1] else len(top1) + 1          # last round not matching -> not converged


def gate(resid, tau):
    for i, r in enumerate(resid):
        if r < tau:
            return i
    return len(resid) - 1


def diag_cell(prefix):
    rows = [json.loads(l) for l in gzip.open(prefix + ".metrics.jsonl.gz", "rt")]
    summ = json.load(open(prefix + ".summary.json"))
    cfg = json.load(open(prefix + ".config.json"))
    D = cfg["depth"]
    out = dict(model=cfg["model"], depth=D, batch=cfg["batch"], L=cfg["prompt_len"], dtype=cfg["dtype"],
               n_rows=len(rows), kv_isolation=summ["kv_isolation"]["tokens_identical_to_plain_AR"],
               commit_nondeterministic_steps=summ["commit_nondeterministic_steps"],
               commit_bitexact_frac=mean([r["commit_bitexact"] for r in rows]),
               depth_subset=summ["depth_subset"])
    # E2
    pred = {}
    for name in ("constant", "linear", "prev", "prev_shift"):
        rs = [r["pred"][name] for r in rows if name in r.get("pred", {})]
        if not rs:
            continue
        per_depth = [dict(depth=d + 1, rel_p50=q([x["rel"][d] for x in rs], .5), rel_p95=q([x["rel"][d] for x in rs], .95),
                          cos_p50=q([x["cos"][d] for x in rs], .5)) for d in range(D)]
        at_s = [dict(depth=s, kl_mean=mean([x["kl"][i] for x in rs]), top1=mean([x["top1"][i] for x in rs]))
                for i, s in enumerate(summ["depth_subset"])]
        pred[name] = dict(n=len(rs), per_depth=per_depth, logits_at_depths=at_s)
    out["E2"] = pred
    # E3 Jacobi
    jac = {}
    for init in ("constant", "linear", "prev_shift"):
        rs = [r["jacobi"][init] for r in rows if init in r.get("jacobi", {})]
        if not rs:
            continue
        ks = [k_stable(x["top1"]) for x in rs]
        curve = [dict(k=k + 1, top1=mean([x["top1"][k] for x in rs]), kl_mean=mean([x["kl"][k] for x in rs]),
                      rel_p50=q([x["rel"][k] for x in rs], .5), resid_p50=q([x["resid"][k] for x in rs], .5))
                 for k in range(D)]
        gates = {}
        for tau in TAUS:
            ix = [gate(x["resid"], tau) for x in rs]
            gates[f"{tau:g}"] = dict(rounds_mean=mean([i + 1 for i in ix]),
                                     top1=mean([x["top1"][i] for x, i in zip(rs, ix)]),
                                     kl_mean=mean([x["kl"][i] for x, i in zip(rs, ix)]))
        jac[init] = dict(n=len(rs), k_stable_mean=mean(ks), k_stable_p50=q(ks, .5), k_stable_p90=q(ks, .9),
                         k_stable_max=max(ks), frac_k_stable_lt_Dm1=mean([k < D - 1 for k in ks]),
                         exact_after_D_rounds=mean([x["relD_after_D"] == 0.0 for x in rs]),
                         relD_after_D_max=max(x["relD_after_D"] for x in rs), curve=curve, gate=gates)
    out["E3_jacobi"] = jac
    par = {}
    for cfgname in sorted({k for r in rows for k in r.get("parareal", {})}):
        rs = [r["parareal"][cfgname] for r in rows]
        bs = int(cfgname.split("_")[0][1:])
        nb = D // bs
        g_cost = nb + 1 if cfgname.endswith("onestep") else 0
        init_cost = nb if cfgname.endswith("onestep") else 0
        ks = [k_stable(x["top1"]) for x in rs]
        par[cfgname] = dict(block=bs, n_blocks=nb, seq_calls_per_round=bs + g_cost, init_seq_calls=init_cost,
                            k_stable_mean=mean(ks), k_stable_p90=q(ks, .9),
                            seq_calls_to_stable_mean=init_cost + mean(ks) * (bs + g_cost),
                            top1_by_round=[mean([x["top1"][k] for x in rs]) for k in range(nb)],
                            kl_by_round=[mean([x["kl"][k] for x in rs]) for k in range(nb)])
    out["E3_parareal"] = par
    # E4
    pert = {}
    for key in sorted({k for r in rows for k in r.get("perturb", {})}):
        rs = [r["perturb"][key] for r in rows]
        n = len(rs[0]["amp"])
        pert[key] = dict(amp_p50_by_k=[q([x["amp"][k] for x in rs], .5) for k in range(n)],
                         amp_p90_by_k=[q([x["amp"][k] for x in rs], .9) for k in range(n)],
                         amp_final_p50=q([x["amp"][-1] for x in rs], .5), amp_final_max=max(x["amp"][-1] for x in rs),
                         frac_amp_final_gt1=mean([x["amp"][-1] > 1 for x in rs]),
                         flip_rate=mean([x["flip"] for x in rs]), kl_mean=mean([x["kl"] for x in rs]),
                         noise_floor_max=max(x["floor"] for x in rs))
    out["E4"] = pert
    out["_tokens"] = summ["tokens"]
    return out


def summarize_e5(d: Path, n_boot: int = 2000) -> dict:
    """E5: per model cell, per arm: throughput ratio vs AR = (sum tok/sum wall)_arm / (..)_AR with a
    paired prompt bootstrap 95% CI; TPOT p50; mean core rounds/token; exact-to-AR rate; accuracy."""
    import random
    out = {}
    cells = defaultdict(list)
    for f in sorted(glob.glob(str(d / "*.jsonl"))):
        cells[Path(f).stem.rsplit("_o", 1)[0]].extend(json.loads(l) for l in open(f))
    for cell, rows in cells.items():
        by = defaultdict(dict)
        for r in rows:
            by[r["prompt_idx"]][r["arm"]] = r
        prompts = sorted(p for p, a in by.items() if "AR" in a)
        arms = sorted({a for p in prompts for a in by[p]})
        rng = random.Random(0)
        res = {}
        for arm in arms:
            ps = [p for p in prompts if arm in by[p]]
            def ratio(sample):
                ta = sum(by[p][arm]["n_tokens"] for p in sample) / sum(by[p][arm]["wall_s"] for p in sample)
                tb = sum(by[p]["AR"]["n_tokens"] for p in sample) / sum(by[p]["AR"]["wall_s"] for p in sample)
                return ta / tb
            boots = sorted(ratio([rng.choice(ps) for _ in ps]) for _ in range(n_boot))
            rs = [by[p][arm] for p in ps]
            res[arm] = dict(n=len(ps), speed_vs_AR=ratio(ps), ci95=[boots[int(.025 * n_boot)], boots[int(.975 * n_boot)]],
                            tok_per_s=sum(r["n_tokens"] for r in rs) / sum(r["wall_s"] for r in rs),
                            tpot_p50_ms=st.median(r["tpot_p50_ms"] for r in rs if r["tpot_p50_ms"]),
                            rounds_mean=mean([r["rounds_mean"] for r in rs if r["rounds_mean"]]),
                            exact_to_AR=mean([bool(r.get("exact_to_AR")) for r in rs]),
                            acc_strict=mean([r["correct_strict"] for r in rs]), acc_flex=mean([r["correct_flex"] for r in rs]),
                            hbm_peak_gb=max(r["hbm_peak_gb"] for r in rs))
        out[cell] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.raw)
    res = dict(E0={}, E1={}, diag={}, status={})
    for f in sorted(glob.glob(str(raw / "e0/*.json"))):
        res["E0"][Path(f).stem] = json.load(open(f))["cells"]
    for f in sorted(glob.glob(str(raw / "e1/*.json"))):
        d = json.load(open(f))
        res["E1"][Path(f).stem] = [{k: v for k, v in c.items() if k in ("L", "B", "status", "S_oracle", "S_oracle_host",
                                                                     "hbm_peak_alloc_gb", "error")}
                                   | {"T_F_ms": {w: t["median_ms"] for w, t in c.get("T_F", {}).items()},
                                      "serial_D_ms": c.get("serial_D", {}).get("median_ms"),
                                      "coda_ms": c.get("coda", {}).get("median_ms")} for c in d["cells"]]
    for cfgf in sorted(glob.glob(str(raw / "diag/*/*.config.json"))):
        prefix = cfgf[: -len(".config.json")]
        name = str(Path(prefix).relative_to(raw / "diag"))
        if not Path(prefix + ".summary.json").exists():
            res["status"][name] = "failed_or_incomplete"
            continue
        res["diag"][name] = diag_cell(prefix)
        res["status"][name] = "done"
    # batch-shape lane: B=1 prompt p vs B=16 cohort row p (same prompt) -- token agreement
    for m in ("h_D32", "o14_D4", "o26_D4"):
        big = res["diag"].get(f"bf16_b16_L256/{m}")
        if not big:
            continue
        agree = []
        for p in range(4):
            one = res["diag"].get(f"bf16_b1_L256/{m}_p{p}")
            if one:
                t16 = [step[p] for step in big["_tokens"]]
                t1 = [step[0] for step in one["_tokens"]]
                first = next((i for i, (x, y) in enumerate(zip(t16, t1)) if x != y), None)
                agree.append(dict(prompt=p, identical=first is None, first_divergence=first))
        res["diag"][f"batch_shape/{m}"] = dict(b1_vs_b16=agree)
    # projections from E1 timings
    proj = {}
    for key, cell in res["diag"].items():
        if not key.startswith("bf16_") or "E3_jacobi" not in cell:
            continue
        m = {"huginn-3.5b": "h", "ouro-1.4b": "o14", "ouro-2.6b": "o26"}[cell["model"]]
        D, B, L = cell["depth"], cell["batch"], cell["L"]
        e1 = res["E1"].get(f"{m}_D{4 if m != 'h' else 32}_L{L}")
        if not e1 or D not in (4, 32):
            continue
        c = next((c for c in e1 if c["B"] == B and c["status"] == "ok"), None)
        if not c or str(D) not in c["T_F_ms"]:
            continue
        t1, tD = c["T_F_ms"]["1"], c["T_F_ms"][str(D)]
        serial = D * t1
        pj = {}
        for init, j in cell["E3_jacobi"].items():
            pj[init] = dict(at_k_stable=serial / (t1 + j["k_stable_mean"] * tD),
                            **{f"gate_{t}": dict(speed=serial / (t1 + g["rounds_mean"] * tD), top1=g["top1"])
                               for t, g in j["gate"].items()})
        proj[key] = dict(T_F1_ms=t1, T_FD_ms=tD, serial_ms=serial, jacobi=pj)
    res["projection_upper_bound"] = proj
    res["E5"] = summarize_e5(raw / "e5")
    for v in res["diag"].values():
        v.pop("_tokens", None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / "summary.json", "w"), indent=1)
    print(json.dumps(res["status"], indent=1))


if __name__ == "__main__":
    main()
