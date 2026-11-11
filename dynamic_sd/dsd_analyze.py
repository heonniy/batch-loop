"""Analysis of dsd_run.py outputs.  usage:
  python dsd_analyze.py calib <observe.json> [...]      # threshold -> realized-depth table
  python dsd_analyze.py cells <cell.json> [...]         # per-arm matrix + conditional stats
"""
import gzip
import json
import sys
from collections import Counter, defaultdict

import numpy as np


def load(p):
    """Reads a receipt, plain or gzip-compressed (*.json.gz, as committed)."""
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return json.load(f)


def calib(paths):
    for p in paths:
        J = load(p)
        a = J["meta"]["args"]
        T = a["T_total"]
        kind = "ouro" if "ouro" in a["model"] else "huginn"
        traj = defaultdict(dict)          # (sample,pos) -> step -> value(s)
        for r in J["per_sample"]:
            for rec in r["observe"]:
                traj[(r["idx"], rec[0])][rec[1]] = rec[2:]
        keys = list(traj)
        print(f"\n== {p}  model={a['model']} dtype={a['dtype']} tokens={len(keys)}")
        if kind == "ouro":
            grid = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            cdf_by_step = {s: np.array([traj[k].get(s, [np.nan])[0] for k in keys]) for s in range(1, T + 1)}
            for s in range(1, T):
                v = cdf_by_step[s]
                print(f"  cdf@step{s}: p10={np.nanpercentile(v,10):.3f} p50={np.nanpercentile(v,50):.3f} "
                      f"p90={np.nanpercentile(v,90):.3f}")
            for tau in grid:
                D = np.full(len(keys), T)
                for s in range(T - 1, 0, -1):
                    D = np.where(cdf_by_step[s] >= tau, s, D)
                h = Counter(D.tolist())
                print(f"  tau={tau:<5} meanD={D.mean():.2f} sd={D.std():.2f} hist={dict(sorted(h.items()))}")
        else:
            for name, col, smin in (("wfd_rel(÷prev)", 0, 2), ("native_latentdiff(÷cur)", 1, 1)):
                M = np.array([[traj[k].get(s, [np.nan, np.nan])[col] for s in range(1, T + 1)] for k in keys])
                print(f"  {name}: median by step " +
                      " ".join(f"{np.nanmedian(M[:, s-1]):.3g}" for s in (1, 2, 4, 8, 16, 24, 31)))
                for eps in (0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001):
                    D = np.full(len(keys), T)
                    for s in range(T - 1, smin - 1, -1):
                        D = np.where(M[:, s - 1] < eps, s, D)
                    print(f"    eps={eps:<6} meanD={D.mean():5.2f} sd={D.std():5.2f} "
                          f"p10={np.percentile(D,10):.0f} p50={np.percentile(D,50):.0f} "
                          f"p90={np.percentile(D,90):.0f} frac_full={np.mean(D==T):.2f}")


def cells(paths):
    for p in paths:
        J = load(p)
        a = J["meta"]["args"]
        S = J["summary"]
        P = J["per_sample"]
        T, d = a["T_total"], a["T_draft"]
        print(f"\n== {p.split('/')[-1]}  model={a['model']} ds={a['dataset']} exit={a['exit']} "
              f"tau={a['tau']} eps={a['eps']} T={T} d={d} K={a['K']} n={len(P)}")
        dep = [D for r in P for D in r.get("depths_ar", [])]
        if dep:
            h = Counter(dep)
            print(f"  AR depth: mean={np.mean(dep):.2f} sd={np.std(dep):.2f} "
                  f"hist={dict(sorted(h.items())) if T <= 8 else ''}")
        for arm, s in S.items():
            print(f"  {arm:7s} tok/s={s['tok_per_s']:7.1f} speedup={s.get('speedup_vs_ar', float('nan')):.3f} "
                  f"acc={s['acc']:.3f} Rcalls/tok={s['rcalls_per_tok']:.3f} rows/tok={s['rows_per_tok']:.3f} "
                  f"exact={s['exact_match_frac']}")
        # WFD conditional stats
        if any("ev_wfd" in r for r in P):
            acc_by = defaultdict(lambda: [0, 0])
            chain = Counter(); width = Counter()
            disc = halt = tail = ticks = bnd = 0
            ntok = 0
            for r in P:
                e = r["ev_wfd"]
                for (pos, ok, vD, dstep, dhalt, ci) in e["accept"]:
                    acc_by[vD][0] += ok; acc_by[vD][1] += 1
                for k, v in e["chain"].items(): chain[int(k)] += v
                for k, v in e["wave_width"].items(): width[int(k)] += v
                disc += e["discard_rows"]; halt += e["halted_wait_rows"]; tail += e.get("tail_rows", 0)
                ticks += e["n_ticks"]; bnd += e["boundary"]; ntok += len(r["tokens_wfd"])
            tot = sum(v[1] for v in acc_by.values()); okk = sum(v[0] for v in acc_by.values())
            wm = sum(k * v for k, v in width.items()) / max(sum(width.values()), 1)
            print(f"  WFD: accept={okk/max(tot,1):.3f} ({tot} verified) boundary/tok={bnd/ntok:.3f} "
                  f"ticks/tok={ticks/ntok:.3f} mean_wave_width={wm:.2f} discard_rows/tok={disc/ntok:.3f} "
                  f"halted_wait_rows/tok={halt/ntok:.3f} chain_hist={dict(sorted(chain.items()))}")
            print("       P(accept | verifier depth D): " +
                  " ".join(f"D{k}:{v[0]/v[1]:.2f}(n={v[1]})" for k, v in sorted(acc_by.items())))
        if any("ev_dtv" in r for r in P):
            acc_by = defaultdict(lambda: [0, 0])
            vts, mx, mean_src, nacc, rounds, disc, exact, ntok = [], [], [], [], 0, 0, 0, 0
            for r in P:
                e = r["ev_dtv"]
                disc += e["discard_rows"]; exact += e["exact_in_draft"]; ntok += len(r["tokens_dtv"])
                for (Kc, na, Ds, vt) in e["rounds"]:
                    rounds += 1; vts.append(vt); nacc.append(na)
                    mx.append(max(Ds)); mean_src.append(np.mean(Ds))
                    for i in range(min(na + 1, Kc)):        # censor after first reject
                        acc_by[Ds[i]][0] += int(i < na); acc_by[Ds[i]][1] += 1
            print(f"  DtV: rounds={rounds} mean_acc/round={np.mean(nacc):.2f} verify_ticks/round={np.mean(vts):.2f} "
                  f"E[max D]={np.mean(mx):.2f} E[mean D]={np.mean(mean_src):.2f} "
                  f"discard_rows/tok={disc/ntok:.3f} exact_in_draft/tok={exact/ntok:.3f}")
            print("       P(accept | source depth D): " +
                  " ".join(f"D{k}:{v[0]/v[1]:.2f}(n={v[1]})" for k, v in sorted(acc_by.items())))
        fm = [(r["idx"], r.get("first_mismatch_wfd"), r.get("first_mismatch_dtv")) for r in P]
        bad = [x for x in fm if x[1] is not None or x[2] is not None]
        if bad:
            print(f"  mismatching samples (idx, wfd_first, dtv_first): {bad[:12]}")


def msr(S, arm):
    if arm not in S:
        return "-"
    return f"{1000.0 / (S[arm]['tok_per_s'] * S[arm]['rcalls_per_tok']):.1f}"


def table(paths):
    """One row per cell: the headline matrix (markdown)."""
    hdr = ("| model | ds | exit | thr | d | K | n | meanD (sd) | AR t/s | WFD t/s | DtV t/s | WFD/AR | DtV/AR | "
           "R/tok AR | R/tok WFD | R/tok DtV | ms/R AR/WFD/DtV | acc AR/WFD/DtV | exact WFD/DtV |")
    print(hdr)
    print("|" + "---|" * hdr.count("|")[:-1] if False else "|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    for p in sorted(paths):
        try:
            J = load(p)
        except Exception:
            continue
        if "meta" not in J:
            continue
        a = J["meta"]["args"]
        if a["exit"] == "observe":
            continue
        S = J["summary"]
        dep = [D for r in J["per_sample"] for D in r.get("depths_ar", [])]
        md = f"{np.mean(dep):.2f} ({np.std(dep):.2f})" if dep else "-"
        thr = a["tau"] if a["tau"] is not None else (a["eps"] if a["eps"] is not None else "-")
        if a["exit"] == "replay":
            thr = "replay:" + a["replay_from"].split("/")[-1].replace(".json", "").split("_", 1)[-1]

        def g(arm, k, f="{:.1f}"):
            return f.format(S[arm][k]) if arm in S and S[arm].get(k) is not None else "-"
        m = a["model"].replace("huginn-3.5b", "Huginn").replace("ouro-", "Ouro-")
        print(f"| {m} | {a['dataset']} | {a['exit']}{'/fp32' if a['dtype']=='float32' else ''} | {thr} | "
              f"{a['T_draft']} | {a['K']} | {len(J['per_sample'])} | {md} | "
              f"{g('ar','tok_per_s')} | {g('wfd','tok_per_s')} | {g('dtv','tok_per_s')} | "
              f"{g('wfd','speedup_vs_ar','{:.2f}')} | {g('dtv','speedup_vs_ar','{:.2f}')} | "
              f"{g('ar','rcalls_per_tok','{:.2f}')} | {g('wfd','rcalls_per_tok','{:.2f}')} | "
              f"{g('dtv','rcalls_per_tok','{:.2f}')} | {msr(S,'ar')}/{msr(S,'wfd')}/{msr(S,'dtv')} | "
              f"{g('ar','acc','{:.2f}')}/{g('wfd','acc','{:.2f}')}/{g('dtv','acc','{:.2f}')} | "
              f"{g('wfd','exact_match_frac','{:.2f}')}/{g('dtv','exact_match_frac','{:.2f}')} |")


def straggler(paths):
    """DtV hindsight bound: verify ticks actually spent (max over all K+1 sources) vs ticks that
    were NEEDED (max over sources up to the first rejection; all if fully accepted). Also the
    share of verify ticks spent waiting on rows that were later discarded.  HINDSIGHT ONLY."""
    print("| cell | rounds | verify ticks/round | needed ticks/round | straggler share | E[maxD]-E[meanD] |")
    print("|---|---|---|---|---|---|")
    for p in sorted(paths):
        J = load(p)
        if "meta" not in J:
            continue
        a = J["meta"]["args"]
        d = a["T_draft"]
        spent = need = gap = 0.0
        n = 0
        for r in J["per_sample"]:
            e = r.get("ev_dtv")
            if not e:
                continue
            for (Kc, na, Ds, vt) in e["rounds"]:
                n += 1
                spent += max(0, max(Ds) - d)
                upto = Ds[: (na + 1) if na < Kc else Kc + 1]
                need += max(0, max(upto) - d)
                gap += max(Ds) - np.mean(Ds)
        if n:
            print(f"| {p.split('/')[-1].replace('.json','')} | {n} | {spent/n:.2f} | {need/n:.2f} | "
                  f"{1 - need/max(spent,1e-9):.3f} | {gap/n:.2f} |")


def bench(paths):
    for p in paths:
        J = load(p)
        print(f"\n== {p}  model={J['model']} prefix={J['prefix']}")
        for lay in ("same", "wave"):
            items = [(int(k.split("_")[1]), v) for k, v in J["res"].items() if k.startswith(lay)]
            print(f"  {lay}: " + "  ".join(f"n={n}:{v.get('median_ms', float('nan')):.2f}ms" for n, v in sorted(items)))


def batch(paths):
    for p in paths:
        J = load(p)
        agg = defaultdict(lambda: defaultdict(list))
        for x in J["res"]:
            key = (x["B"], x["tau"])
            a = x["ar"]
            agg[key]["ar_dec"].append(a["decode_wall"]); agg[key]["ar_R"].append(a["rcalls"])
            agg[key]["meanD"].append(a["mean_depth"]); agg[key]["maxD"].append(a["mean_max_depth"])
            for k, v in x.items():
                if k.startswith("dtv"):
                    agg[key][k + "_sp"].append(v["speedup_decode"]); agg[key][k + "_R"].append(v["rcalls"])
                    agg[key][k + "_vt"].append(v["verify_ticks_per_round"]); agg[key][k + "_acc"].append(v["accept"])
                    agg[key][k + "_match"].append(v["token_match_vs_ar"])
        print(f"\n== {p}  max_new={J['args']['max_new']} L={J['args']['L']} d={J['args']['d']}")
        print("| B | tau | AR meanD | AR E[max_b D] | AR ms/token-step | " +
              " | ".join(f"DtV K{K} x / acc / vt/round / match" for K in J["args"]["K"]) + " |")
        for (B, tau), g in sorted(agg.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
            mn = J["args"]["max_new"]
            cells_ = [f"{np.mean(g[f'dtv_K{K}_sp']):.2f} / {np.mean(g[f'dtv_K{K}_acc']):.2f} / "
                      f"{np.mean(g[f'dtv_K{K}_vt']):.2f} / {np.mean(g[f'dtv_K{K}_match']):.2f}" for K in J["args"]["K"]]
            print(f"| {B} | {tau} | {np.mean(g['meanD']):.2f} | {np.mean(g['maxD']):.2f} | "
                  f"{1000*np.mean(g['ar_dec'])/mn:.1f} | " + " | ".join(cells_) + " |")


def ci(paths, n_boot=2000):
    """Prompt-level bootstrap 95% CI of the speedup (ratio of summed tok / summed wall)."""
    rng = np.random.default_rng(0)
    print("| cell | n | WFD/AR [95% CI] | DtV/AR [95% CI] | AR acc [95% CI] |")
    print("|---|---|---|---|---|")
    for p in sorted(paths):
        J = load(p)
        if "meta" not in J or J["meta"]["args"]["exit"] == "observe":
            continue
        P = J["per_sample"]
        n = len(P)
        tok = {a: np.array([len(r[f"tokens_{a}"]) for r in P]) for a in ("ar", "wfd", "dtv") if f"tokens_{a}" in P[0]}
        wall = {a: np.array([r[f"wall_{a}"] for r in P]) for a in tok}
        acc = np.array([r["correct_ar"] for r in P], dtype=float)
        idx = rng.integers(0, n, size=(n_boot, n))

        def sp(a):
            if a not in tok:
                return "-"
            s = (tok[a][idx].sum(1) / wall[a][idx].sum(1)) / (tok["ar"][idx].sum(1) / wall["ar"][idx].sum(1))
            pt = (tok[a].sum() / wall[a].sum()) / (tok["ar"].sum() / wall["ar"].sum())
            return f"{pt:.2f} [{np.percentile(s, 2.5):.2f}, {np.percentile(s, 97.5):.2f}]"
        am = acc[idx].mean(1)
        print(f"| {p.split('/')[-2]}/{p.split('/')[-1].replace('.json','')} | {n} | {sp('wfd')} | {sp('dtv')} | "
              f"{acc.mean():.2f} [{np.percentile(am, 2.5):.2f}, {np.percentile(am, 97.5):.2f}] |")


if __name__ == "__main__":
    {"calib": calib, "cells": cells, "table": table, "straggler": straggler, "bench": bench, "ci": ci,
     "batch": batch}[sys.argv[1]](sys.argv[2:])
