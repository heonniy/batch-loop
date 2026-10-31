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


def _load(run, key):
    return np.load(run / "raw" / f"{key}.npz"), json.loads((run / "raw" / f"{key}.json").read_text())


def stage_a(run: Path, out: Path):
    import pandas as pd
    jobs = [j for j in json.loads((run / "jobs.json").read_text()) if j["kind"] in ("A", "Bref")]
    frames, eos = [], defaultdict(lambda: [0, 0])
    keys = ("h_norm", "U", "F", "C", "cos_init", "cos_term")
    for j in jobs:
        if not (run / "raw" / f"{j['key']}.npz").exists():
            continue
        z, meta = _load(run, j["key"])
        if j["kind"] == "A":
            eos[(j["model"], j["workload"])][0] += 1
            eos[(j["model"], j["workload"])][1] += meta["ref"]["stop"] == "eos"
        for phase, pre, roles, ids in (("prompt", "prompt_", meta["roles"], meta["prompt_ids"]),
                                       ("generated", "gen_", None, meta["ref"]["ids"])):
            n, R1 = z[pre + "U"].shape
            R = R1 - 1
            t = np.repeat(np.arange(n), R1)
            r = np.tile(np.arange(R1), n)
            role = np.repeat(np.array(roles if roles else ["generated"] * n), R1)
            d = {"pool": j["kind"], "uid": j["uid"], "model": j["model"], "dataset": j["workload"], "phase": phase,
                 "t": t, "r": r, "r_rel": r / R, "role": role, "position_pct": t / max(1, n - 1),
                 "token_id": np.repeat(np.asarray(ids[:n]), R1)}
            d |= {k: z[pre + k].reshape(-1).astype("float64") for k in keys}
            frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / "normal_trajectory_long.parquet", index=False)
    dist = []
    a = df[df["pool"] == "A"]
    for (m, w, role, r), g in a.groupby(["model", "dataset", "role", "r"]):
        e = {"model": m, "workload": w, "role": role, "r": int(r)}
        for k in ("U", "F", "C"):
            v = g[k].to_numpy()
            v = v[np.isfinite(v)]
            if v.size:
                e[k] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
                        "q10": float(np.quantile(v, .1)), "q50": float(np.quantile(v, .5)),
                        "q90": float(np.quantile(v, .9))}
        dist.append(e)
    _json(out / "normal_trajectory_distribution.json",
          {"by_model_workload_role_loop": dist,
           "eos_rate": {f"{m}/{w}": {"n": x, "eos": y} for (m, w), (x, y) in eos.items()}})
    return df


def stage_bc(run: Path, out: Path):
    import pandas as pd
    jobs = json.loads((run / "jobs.json").read_text())
    long_f, per_src, kl_f, hist_f = [], [], [], []
    for j in (j for j in jobs if j["kind"] == "B"):
        if not (run / "raw" / f"{j['key']}.npz").exists():
            continue
        z, _ = _load(run, j["key"])
        zr, rmeta = _load(run, j["depends"])
        roles, P = rmeta["roles"], len(rmeta["roles"])
        srcs, ts = z["sources"].astype(int), z["clean_ts"].astype(int)
        N, n, R1 = z["acc_E"].shape
        R = R1 - 1
        base = {"model": j["model"], "workload": j["workload"], "uid": j["uid"]}
        for si, s in enumerate(srcs):
            feat = {**base, "source": int(s), "role": roles[s], "position": int(s), "position_pct": s / (P - 1),
                    "age": P - s, "attn_recv_mean": float(zr["src_recv_mean"][:, s].mean()),
                    "attn_last_query_mean": float(zr["src_last_query"][:, s].mean()),
                    "attn_last_query_lastloop": float(zr["src_last_query"][-1, s]),
                    "attn_recv_max": float(zr["src_recv_max"][:, s].max()),
                    "src_U_mean": _nanmean(zr["prompt_U"][s, 1:]), "src_F1": float(zr["prompt_F"][s, 1]),
                    "n_tokens": n}
            for name, lo, hi in T_BINS:
                if lo <= n:
                    sl = slice(lo - 1, min(hi, n))
                    feat[f"kl_{name}"] = _nanmean(z["acc_kl"][si, sl])
                    feat[f"E_R_{name}"] = _nanmean(z["acc_E"][si, sl, R])
                    feat[f"top1_{name}"] = _nanmean(z["acc_top1_changed"][si, sl])
            feat["kl_mean"] = _nanmean(z["acc_kl"][si])
            feat["E_R_mean"] = _nanmean(z["acc_E"][si, :, R])
            per_src.append(feat)
        tt = np.tile(np.arange(1, n + 1), N)
        ss = np.repeat(srcs, n)
        kl_f.append(pd.DataFrame({**base, "source": ss, "t": tt, "kl": z["acc_kl"].reshape(-1),
                                  "top1_changed": z["acc_top1_changed"].reshape(-1),
                                  "nll_shift": z["acc_nll_shift"].reshape(-1), "E_R": z["acc_E"][:, :, R].reshape(-1)}))
        long_f.append(pd.DataFrame({**base, "source": np.repeat(srcs, n * R),
                                    "t": np.tile(np.repeat(np.arange(1, n + 1), R), N),
                                    "r": np.tile(np.arange(1, R + 1), N * n),
                                    "E": z["acc_E"][:, :, 1:].reshape(-1), "U_ev": z["acc_U_ev"][:, :, 1:].reshape(-1),
                                    "dircos": z["acc_dircos"][:, :, 1:].reshape(-1)}))
        Ea = z["acc_E"][:, ts - 1, 1:]          # (N, len(ts), R)
        Ec = z["clean_E"][:, :, 1:]
        K = len(ts)
        hist_f.append(pd.DataFrame({**base, "source": np.repeat(srcs, K * R),
                                    "t": np.tile(np.repeat(ts, R), N), "r": np.tile(np.arange(1, R + 1), N * K),
                                    "r_rel": np.tile(np.arange(1, R + 1) / R, N * K),
                                    "E_accum": Ea.reshape(-1), "E_clean": Ec.reshape(-1), "H": (Ea - Ec).reshape(-1),
                                    "kl_accum": np.repeat(z["acc_kl"][:, ts - 1].reshape(-1), R),
                                    "kl_clean": np.repeat(z["clean_kl"].reshape(-1), R)}))
    pd.concat(long_f, ignore_index=True).to_parquet(out / "eviction_trajectory_long.parquet", index=False)
    ps = pd.DataFrame(per_src)
    ps.to_csv(out / "per_source_128step_summary.csv", index=False)
    pd.concat(kl_f, ignore_index=True).to_csv(out / "longdecode_kl_by_position.csv", index=False)
    hd = pd.concat(hist_f, ignore_index=True)
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
