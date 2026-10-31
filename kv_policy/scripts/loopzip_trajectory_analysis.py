"""LoopZip v6 analysis: completeness (fail closed), long tables, descriptive Stage A/B/C summaries.

Statistical unit = prompt. Sources and time steps within a prompt are NOT independent replicates:
intervals are bootstrap over prompts (per model x workload), never over sources/tokens.
No pattern labels are fitted here; Stage A outputs are distributions and full per-loop curves.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

T_BINS = (("early", 1, 16), ("medium", 17, 64), ("late", 65, 128))


def _json(path: Path, value):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=float))
    tmp.replace(path)


def _boot(vals, n=2000, seed=0):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    d = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return [d[int(.025 * n)], d[int(.975 * n) - 1]]


def _nanmean(x):
    x = np.asarray(x, dtype="float64")
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def completeness(run: Path):
    jobs = json.loads((run / "jobs.json").read_text())
    raw = run / "raw"
    missing = [j["key"] for j in jobs if not (raw / f"{j['key']}.json").exists()
               or not (raw / f"{j['key']}.npz").exists()]
    errs = sorted(p.stem for p in (run / "errors").glob("*.json")) if (run / "errors").exists() else []
    problems = []
    by_ref = defaultdict(list)
    for j in jobs:
        if j["kind"] == "B":
            by_ref[j["depends"]].append(j)
    for ref, chunks in by_ref.items():
        p = raw / f"{ref}.json"
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        srcs = [s for c in chunks for s in c["sources"]]
        if sorted(srcs) != sorted(meta["eligible"]) or len(set(srcs)) != len(srcs):
            problems.append(f"{ref}: source census mismatch")
        if not meta["validation"]["pass"]:
            problems.append(f"{ref}: G0 validation failed")
        for c in chunks:
            q = raw / f"{c['key']}.json"
            if q.exists() and json.loads(q.read_text())["n_tokens"] != meta["ref"]["n"]:
                problems.append(f"{c['key']}: token count mismatch")
    return {"complete": not missing and not errs and not problems, "missing": missing, "errors": errs,
            "problems": problems, "jobs": len(jobs)}


def stage_a(run: Path, out: Path):
    jobs = [j for j in json.loads((run / "jobs.json").read_text()) if j["kind"] in ("A", "Bref")]
    rows, summ = [], defaultdict(lambda: defaultdict(list))
    eos = defaultdict(lambda: [0, 0])
    for j in jobs:
        p = run / "raw" / f"{j['key']}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        meta = json.loads((run / "raw" / f"{j['key']}.json").read_text())
        roles = meta["roles"]
        P = len(roles)
        R = z["prompt_U"].shape[1] - 1
        eos[(j["model"], j["workload"])][0] += 1
        eos[(j["model"], j["workload"])][1] += meta["ref"]["stop"] == "eos"
        for phase, pre in (("prompt", "prompt_"), ("generated", "gen_")):
            n = z[pre + "U"].shape[0]
            for t in range(n):
                role = roles[t] if phase == "prompt" else "generated"
                pct = t / max(1, n - 1)
                for r in range(R + 1):
                    row = {"pool": j["kind"], "uid": j["uid"], "model": j["model"], "dataset": j["workload"],
                           "phase": phase, "t": t, "r": r, "r_rel": r / R, "role": role, "position_pct": pct,
                           "token_id": (meta["prompt_ids"][t] if phase == "prompt" else meta["ref"]["ids"][t])}
                    for k in ("h_norm", "U", "F", "C", "cos_init", "cos_term"):
                        row[k] = float(z[pre + k][t, r])
                    rows.append(row)
                    if j["kind"] == "A":
                        for k in ("U", "F", "C"):
                            summ[(j["model"], j["workload"], role, r)][k].append(row[k])
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(out / "normal_trajectory_long.parquet", index=False)
    dist = []
    for (m, w, role, r), d in sorted(summ.items()):
        e = {"model": m, "workload": w, "role": role, "r": r}
        for k, v in d.items():
            v = np.asarray(v, dtype="float64")
            v = v[np.isfinite(v)]
            if v.size:
                e[k] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
                        "q10": float(np.quantile(v, .1)), "q50": float(np.quantile(v, .5)),
                        "q90": float(np.quantile(v, .9))}
        dist.append(e)
    _json(out / "normal_trajectory_distribution.json",
          {"by_model_workload_role_loop": dist,
           "eos_rate": {f"{m}/{w}": {"n": a, "eos": b} for (m, w), (a, b) in eos.items()}})
    return df


def stage_bc(run: Path, out: Path):
    import pandas as pd
    jobs = json.loads((run / "jobs.json").read_text())
    refs = {j["key"]: j for j in jobs if j["kind"] == "Bref"}
    long_rows, per_src, kl_pos, hist = [], [], [], []
    for j in (j for j in jobs if j["kind"] == "B"):
        p = run / "raw" / f"{j['key']}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        rmeta = json.loads((run / "raw" / f"{j['depends']}.json").read_text())
        zr = np.load(run / "raw" / f"{j['depends']}.npz")
        roles, P = rmeta["roles"], len(rmeta["roles"])
        srcs, ts = z["sources"], z["clean_ts"]
        n, R = z["acc_E"].shape[1], z["acc_E"].shape[2] - 1
        for si, s in enumerate(srcs):
            s = int(s)
            feat = {"model": j["model"], "workload": j["workload"], "uid": j["uid"], "source": s,
                    "role": roles[s], "position": s, "position_pct": s / (P - 1), "age": P - s,
                    "attn_recv_mean": float(zr["src_recv_mean"][:, s].mean()),
                    "attn_last_query_mean": float(zr["src_last_query"][:, s].mean()),
                    "attn_last_query_lastloop": float(zr["src_last_query"][-1, s]),
                    "attn_recv_max": float(zr["src_recv_max"][:, s].max()),
                    "src_U_mean": _nanmean(zr["prompt_U"][s, 1:]), "src_F1": float(zr["prompt_F"][s, 1]),
                    "n_tokens": n}
            for name, a, b in T_BINS:
                sl = slice(a - 1, min(b, n))
                if a <= n:
                    feat[f"kl_{name}"] = _nanmean(z["acc_kl"][si, sl])
                    feat[f"E_R_{name}"] = _nanmean(z["acc_E"][si, sl, R])
                    feat[f"top1_{name}"] = _nanmean(z["acc_top1_changed"][si, sl])
            feat["kl_mean"] = _nanmean(z["acc_kl"][si])
            feat["E_R_mean"] = _nanmean(z["acc_E"][si, :, R])
            per_src.append(feat)
            for t in range(n):
                kl_pos.append({"model": j["model"], "workload": j["workload"], "uid": j["uid"], "source": s,
                               "t": t + 1, "kl": float(z["acc_kl"][si, t]),
                               "top1_changed": float(z["acc_top1_changed"][si, t]),
                               "nll_shift": float(z["acc_nll_shift"][si, t]),
                               "E_R": float(z["acc_E"][si, t, R])})
                for r in range(1, R + 1):
                    long_rows.append((j["model"], j["workload"], j["uid"], s, t + 1, r,
                                      float(z["acc_E"][si, t, r]), float(z["acc_U_ev"][si, t, r]),
                                      float(z["acc_dircos"][si, t, r])))
            for ti, t in enumerate(ts):
                for r in range(1, R + 1):
                    hist.append({"model": j["model"], "workload": j["workload"], "uid": j["uid"], "source": s,
                                 "t": int(t), "r": r, "r_rel": r / R,
                                 "E_accum": float(z["acc_E"][si, t - 1, r]), "E_clean": float(z["clean_E"][si, ti, r]),
                                 "H": float(z["acc_E"][si, t - 1, r] - z["clean_E"][si, ti, r]),
                                 "kl_accum": float(z["acc_kl"][si, t - 1]), "kl_clean": float(z["clean_kl"][si, ti])})
    pd.DataFrame(long_rows, columns=["model", "workload", "uid", "source", "t", "r", "E", "U_ev", "dircos"]
                 ).to_parquet(out / "eviction_trajectory_long.parquet", index=False)
    ps = pd.DataFrame(per_src)
    ps.to_csv(out / "per_source_128step_summary.csv", index=False)
    pd.DataFrame(kl_pos).to_csv(out / "longdecode_kl_by_position.csv", index=False)
    hd = pd.DataFrame(hist)
    hd.to_csv(out / "history_mode_difference.csv", index=False)
    return ps, hd


def _spearman(a, b):
    import pandas as pd
    s = pd.Series(a).rank().corr(pd.Series(b).rank())
    return None if s != s else float(s)


def analyze(run: Path):
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    comp = completeness(run)
    _json(run / "completeness.json", comp)
    _json(out / "errors_and_completeness.json", comp)
    jobs = json.loads((run / "jobs.json").read_text())
    rep = {"complete": comp["complete"]}
    md = ["# LoopZip v6 — persistent eviction and recurrent trajectories (observations)", "",
          f"complete={comp['complete']}; jobs {comp['jobs']}; missing {len(comp['missing'])}; "
          f"errors {len(comp['errors'])}; problems {len(comp['problems'])}.", "",
          "Full-sized reference KV + read masks; no HBM/latency claims. Intervals: bootstrap over prompts.", ""]
    has_a = any((run / "raw" / f"{j['key']}.npz").exists() for j in jobs if j["kind"] == "A")
    if has_a:
        df = stage_a(run, out)
        a = df[df["pool"] == "A"]
        md += ["## Stage A — normal trajectories (no eviction)", "",
               "Mean U_t(r) by model / workload / phase at relative depth (all loops kept in the parquet).", "",
               "| model | workload | phase | n tok | U(r=1) | U(mid) | U(R) | F(r=1) | C(mid) |", "|---|---|---|---|---|---|---|---|---|"]
        for (m, w, ph), g in a.groupby(["model", "dataset", "phase"]):
            R = int(g["r"].max())
            mid = R // 2
            f = lambda r, k: g[g["r"] == r][k].mean()  # noqa: E731
            md.append(f"| {m} | {w} | {ph} | {g[g['r'] == 0].shape[0]} | {f(1, 'U'):.4f} | {f(mid, 'U'):.4f} | "
                      f"{f(R, 'U'):.4f} | {f(1, 'F'):.4f} | {f(mid, 'C'):.3f} |")
        md += ["", "| model | role | n tok | U(r=1) | U(R) | F(r=1) |", "|---|---|---|---|---|---|"]
        for (m, role), g in a.groupby(["model", "role"]):
            R = int(g["r"].max())
            f = lambda r, k: g[g["r"] == r][k].mean()  # noqa: E731
            md.append(f"| {m} | {role} | {g[g['r'] == 0].shape[0]} | {f(1, 'U'):.4f} | {f(R, 'U'):.4f} | {f(1, 'F'):.4f} |")
        md.append("")
    has_b = any((run / "raw" / f"{j['key']}.npz").exists() for j in jobs if j["kind"] == "B")
    if has_b:
        ps, hd = stage_bc(run, out)
        md += ["## Stage B — exhaustive one-source persistent eviction (accumulated arm)", "",
               "Per prompt: mean over ALL eligible sources; then mean [95% CI] over prompts.", "",
               "| model | workload | prompts | sources | KL early | KL medium | KL late | E(t,R) late | top1 late |",
               "|---|---|---|---|---|---|---|---|---|"]
        summ = []
        for (m, w), g in ps.groupby(["model", "workload"]):
            per_prompt = g.groupby("uid").mean(numeric_only=True)
            e = {"model": m, "workload": w, "prompts": len(per_prompt), "sources": len(g)}
            cells = []
            for k in ("kl_early", "kl_medium", "kl_late", "E_R_late", "top1_late"):
                if k in per_prompt:
                    v = per_prompt[k].tolist()
                    e[k] = {"mean": float(np.nanmean(v)), "ci95": _boot(v)}
                    ci = e[k]["ci95"]
                    cells.append(f"{e[k]['mean']:.4g}" + (f" [{ci[0]:.3g},{ci[1]:.3g}]" if ci else ""))
                else:
                    cells.append("—")
            # within-prompt source heterogeneity and simple nonrecurrent correlates (descriptive)
            e["within_prompt_spearman_klmean"] = {
                f: _nanmean([_spearman(gg[f], gg["kl_mean"]) for _, gg in g.groupby("uid")])
                for f in ("position", "attn_recv_mean", "attn_last_query_mean", "src_U_mean")}
            e["within_prompt_cv_klmean"] = _nanmean([gg["kl_mean"].std() / gg["kl_mean"].mean()
                                                      for _, gg in g.groupby("uid")])
            summ.append(e)
            md.append(f"| {m} | {w} | {len(per_prompt)} | {len(g)} | " + " | ".join(cells) + " |")
        md += ["", "Within-prompt Spearman(source feature, mean KL), averaged over prompts (descriptive only):", "",
               "| model | workload | position | attn recv | attn last-query | src U mean | CV of KL |",
               "|---|---|---|---|---|---|---|"]
        f3 = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        for e in summ:
            s = e["within_prompt_spearman_klmean"]
            md.append(f"| {e['model']} | {e['workload']} | {f3(s['position'])} | {f3(s['attn_recv_mean'])} | "
                      f"{f3(s['attn_last_query_mean'])} | {f3(s['src_U_mean'])} | {f3(e['within_prompt_cv_klmean'])} |")
        md += ["", "## Stage C — accumulated vs clean-history (H = E_accum − E_clean at r = R)", "",
               "| model | t | mean E_accum | mean E_clean | mean H | KL accum | KL clean |", "|---|---|---|---|---|---|---|"]
        for (m, t), g in hd[hd["r_rel"] == 1.0].groupby(["model", "t"]):
            md.append(f"| {m} | {t} | {g['E_accum'].mean():.4g} | {g['E_clean'].mean():.4g} | {g['H'].mean():.4g} | "
                      f"{g['kl_accum'].mean():.4g} | {g['kl_clean'].mean():.4g} |")
        rep["stage_b"] = summ
        md += ["", "Descriptive census of a 4-prompt/model/workload pilot; not a powered generalization test. "
               "Correlations with position/attention are NOT evidence of recurrence-specific importance."]
    (out / "TRAJECTORY_OBSERVATIONS.md").write_text("\n".join(md) + "\n")
    _json(out / "summary.json", rep)
    return rep
