"""Analysis of the Huginn R32 attention-mechanism pilot -> analysis/*.csv, figures, mechanism_report.md.

Every summary is a mean over PROMPTS of a per-prompt quantity (token-mean within the prompt), with a
95% bootstrap CI over prompts; the number of contributing prompts is reported. Loops are 1-based
(loop 32 = terminal). Layers are the 4 physical core layers, never pooled for attention metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import zlib
from pathlib import Path

import numpy as np

N_BOOT = 1000
SHOW_LOOPS = (1, 2, 4, 8, 16, 24, 28, 31, 32)
POS_BINS = [(1, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, 2048)]


def boot(vals, key):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(zlib.crc32(key.encode()))
    ms = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(N_BOOT))
    return st.mean(vals), ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT) - 1], len(vals)


def fmt(m, lo, hi, n=None, f=".3g"):
    return f"{m:{f}} [{lo:{f}}, {hi:{f}}]"


def load(run: Path):
    recs = []
    for p in sorted((run / "mechanism" / "raw").glob("mech__*.json")):
        r = json.loads(p.read_text())
        r["_npz"] = dict(np.load(p.with_suffix(".npz")))
        recs.append(r)
    return recs


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    recs = load(run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    B, C = recs[0]["b_metrics"], recs[0]["c_metrics"]
    R = recs[0]["_npz"]["b"].shape[1]
    NJ = recs[0]["_npz"]["b"].shape[2]
    groups = {"gsm8k": [r for r in recs if r["job"]["dataset"] == "gsm8k"],
              "math500": [r for r in recs if r["job"]["dataset"] == "math500"], "all": recs}
    groups = {k: v for k, v in groups.items() if v}

    # ---------------------------------------------------------------- A: geometry
    geo_rows = []
    for gname, g in groups.items():
        for key in sorted(g[0]["geo_summary"]):
            metric, seg, stat = key.split("|")
            for j in range(NJ):
                for s in range(R):
                    m, lo, hi, n = boot([r["geo_summary"][key][j][s] for r in g if key in r["geo_summary"]], f"A{gname}{key}{j}{s}")
                    geo_rows.append({"group": gname, "metric": metric, "segment": seg, "stat": stat, "layer": j,
                                     "loop": s + 1, "mean": m, "ci_lo": lo, "ci_hi": hi, "prompts": n})
    write_csv(out / "kv_geometry_by_loop.csv", geo_rows)
    head_rows = []
    for metric in ("dK", "dV", "cosK", "cosV"):
        for seg in ("prompt", "cont"):
            arr = np.stack([r["_npz"][f"geo_head__{metric}|{seg}"] for r in recs if f"geo_head__{metric}|{seg}" in r["_npz"]])
            mean = np.nanmean(arr, 0)  # [NJ, R, H]
            for j in range(NJ):
                for s in range(R):
                    hv = mean[j, s]
                    head_rows.append({"metric": metric, "segment": seg, "layer": j, "loop": s + 1, "head_min": float(np.nanmin(hv)),
                                      "head_median": float(np.nanmedian(hv)), "head_max": float(np.nanmax(hv)),
                                      "head_argmax": int(np.nanargmax(hv)), "per_head": json.dumps([round(float(x), 5) for x in hv])})
    write_csv(out / "kv_geometry_per_head.csv", head_rows)
    pos_rows = []
    for gname, g in groups.items():
        for metric in ("dK", "dV"):
            for j in range(NJ):
                for loop in (1, 8, 16, 24, 31):
                    for lo_, hi_ in POS_BINS:
                        vals = []
                        for r in g:
                            col = r["_npz"][f"geo_col__{metric}"][j, loop - 1]  # [C]
                            a, b = r["P"] + lo_ - 1, min(r["P"] + hi_, col.shape[0])
                            if b > a:
                                vals.append(float(np.nanmean(col[a:b])))
                        m, lo, hi, n = boot(vals, f"Apos{gname}{metric}{j}{loop}{lo_}")
                        pos_rows.append({"group": gname, "metric": metric, "layer": j, "loop": loop, "bin_g": f"{lo_}-{hi_}",
                                         "mean": m, "ci_lo": lo, "ci_hi": hi, "prompts": n})
    write_csv(out / "kv_geometry_by_position.csv", pos_rows)

    # ---------------------------------------------------------------- B / C per loop & layer
    def per_loop(arrkey, names, tag):
        rows = []
        for gname, g in groups.items():
            for mi, mname in enumerate(names):
                for j in list(range(NJ)) + ["mean"]:
                    for s in range(R):
                        vals = []
                        for r in g:
                            a = r["_npz"][arrkey]
                            if a.shape[0] == 0:
                                continue
                            v = a[:, s, :, mi].mean(-1) if j == "mean" else a[:, s, j, mi]
                            vals.append(float(np.nanmean(v)))
                        m, lo, hi, n = boot(vals, f"{tag}{gname}{mname}{j}{s}")
                        rows.append({"group": gname, "metric": mname, "layer": j, "loop": s + 1, "mean": m, "ci_lo": lo,
                                     "ci_hi": hi, "prompts": n})
        return rows

    b_rows = per_loop("b", B, "B")
    c_rows = per_loop("c", C, "C")
    write_csv(out / "fixed_q_attention_by_loop.csv", b_rows)
    # boundaries + output
    bnd = []
    for gname, g in groups.items():
        for k in g[0]["plain_keys"] + ["kl", "argmax_same"]:
            vals = []
            for r in g:
                z = r["_npz"]
                if k in ("kl", "argmax_same"):
                    vals.append(float(np.mean(z[k][1:].astype(float))))  # decode steps (g >= 1)
                else:
                    vals.append(float(np.mean(z["plain"][:, r["plain_keys"].index(k)])))
            m, lo, hi, n = boot(vals, f"bnd{gname}{k}")
            bnd.append({"group": gname, "metric": k, "layer": "", "loop": "", "mean": m, "ci_lo": lo, "ci_hi": hi, "prompts": n})
    write_csv(out / "hidden_error_by_loop.csv", c_rows + bnd)
    # fixed-Q by position (loops 1, 16, 31, per layer mean)
    bpos = []
    iD = B.index("D_swap")
    for gname, g in groups.items():
        for loop in (1, 8, 16, 31):
            for lo_, hi_ in POS_BINS:
                vals = [float(np.nanmean(r["_npz"]["b"][lo_ - 1:hi_, loop - 1, :, iD])) for r in g if r["_npz"]["b"].shape[0] >= lo_]
                m, lo, hi, n = boot(vals, f"Bpos{gname}{loop}{lo_}")
                bpos.append({"group": gname, "metric": "D_swap(layer-mean)", "loop": loop, "bin_g": f"{lo_}-{hi_}",
                             "mean": m, "ci_lo": lo, "ci_hi": hi, "prompts": n})
    write_csv(out / "fixed_q_attention_by_position.csv", bpos)

    # ---------------------------------------------------------------- D
    d_rows = []
    for r in recs:
        for x in r["d_records"]:
            d_rows.append({"dataset": r["job"]["dataset"], "uid": r["job"]["uid"], **{k: v for k, v in x.items() if k != "loop_end_traj"},
                           "contraction_postR_over_postlayer": x["post_R"] / x["post_layer"] if x["post_layer"] > 0 else float("nan"),
                           "loop_end_traj": json.dumps([round(v, 6) for v in x["loop_end_traj"]])})
    write_csv(out / "intervention_effects.csv", d_rows)
    d_sum = []
    for gname, g in groups.items():
        for j in list(range(NJ)) + ["mean"]:
            for loop in sorted({x["loop"] for x in d_rows}):
                row = {"group": gname, "layer": j, "loop": loop}
                for k in ("imm_attn", "post_layer", "post_R", "post_core_lnf", "final_lnf", "kl_ref_int", "argmax_same"):
                    vals = []
                    for r in g:
                        xs = [x for x in r["d_records"] if x["loop"] == loop and (j == "mean" or x["layer"] == j)]
                        if xs:
                            vals.append(st.mean(float(x[k]) for x in xs))
                    m, lo, hi, n = boot(vals, f"D{gname}{j}{loop}{k}")
                    row.update({k: m, f"{k}_ci_lo": lo, f"{k}_ci_hi": hi})
                row["prompts"] = n
                d_sum.append(row)
    write_csv(out / "intervention_summary.csv", d_sum)

    # ---------------------------------------------------------------- figures
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def curve(ax, rows, metric, group, label_fmt, layers):
            for j in layers:
                pts = sorted((x["loop"], x["mean"], x["ci_lo"], x["ci_hi"]) for x in rows
                             if x["group"] == group and x["metric"] == metric and x["layer"] == j)
                xs = [p[0] for p in pts]
                ax.plot(xs, [p[1] for p in pts], label=label_fmt.format(j))
                ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], alpha=0.2)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric in zip(axs, ("dK", "dV")):
            for j in range(NJ):
                pts = sorted((x["loop"], x["mean"]) for x in geo_rows if x["group"] == "all" and x["metric"] == metric
                             and x["segment"] == "cont" and x["stat"] == "median" and x["layer"] == j)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=f"layer {j}")
            ax.set_title(f"A: median {metric} vs terminal (continuation)"), ax.set_xlabel("loop r"), ax.legend()
        curve(axs[2], b_rows, "D_swap", "all", "layer {}", range(NJ))
        axs[2].set_title("B: fixed-Q D_swap (post-proj, rel.)"), axs[2].set_xlabel("loop r"), axs[2].legend()
        fig.tight_layout(), fig.savefig(out / "fig_A_B_by_loop.png", dpi=110), plt.close(fig)
        figs.append("fig_A_B_by_loop.png")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        iE = C.index("err_out")
        for r in recs:
            a = r["_npz"]["c"]
            if a.shape[0]:
                axs[0].plot(range(1, R + 1), np.nanmean(a[:, :, NJ - 1, iE], 0), lw=0.5, alpha=0.4,
                            color="C0" if r["job"]["dataset"] == "gsm8k" else "C1")
        curve(axs[0], c_rows, "err_out", "all", "mean (layer {})", [NJ - 1])
        axs[0].set_yscale("log"), axs[0].set_title("C: FULL-vs-LAST hidden error (blue GSM8K, orange MATH)")
        axs[0].set_xlabel("loop r")
        for loop in (1, 4, 16, 31):
            traj = [json.loads(x["loop_end_traj"]) for x in d_rows if x["loop"] == loop]
            if traj:
                axs[1].plot(range(1, R + 1), np.mean(np.array(traj), 0), label=f"intervene at loop {loop}")
        axs[1].set_yscale("log"), axs[1].set_title("D: hidden error at loop end after one intervention"), axs[1].legend()
        axs[1].set_xlabel("loop r")
        fig.tight_layout(), fig.savefig(out / "fig_C_D_error_propagation.png", dpi=110), plt.close(fig)
        figs.append("fig_C_D_error_propagation.png")
    except ImportError:
        pass

    # ---------------------------------------------------------------- report
    def get(rows, **kw):
        for x in rows:
            if all(x.get(k) == v for k, v in kw.items()):
                return x
        return None

    md = ["# Huginn R32 — mechanism of FULL-vs-terminal KV robustness (pilot)", "",
          f"Run `{run}`: {len(recs)} prompts ({len(groups.get('gsm8k', []))} GSM8K + {len(groups.get('math500', []))} MATH-500, Stage-1 "
          "stratified selection), frozen greedy FULL prefix, BF16 model, FP32 instrumentation. Values: mean over prompts "
          "of per-prompt token means [95% bootstrap CI over prompts]. Loop 32 = terminal.", "",
          "Controls: hook check max |Δlogit| traced vs untraced = "
          f"{max((r['hook_check_max_abs_logit_diff'] or 0) for r in recs)} on "
          f"{sum(r['hook_check_max_abs_logit_diff'] is not None for r in recs)} prompts; NaN counts B/C = "
          f"{sum(r['nan_b'] for r in recs)}/{sum(r['nan_c'] for r in recs)}; prelude FULL-vs-LAST error "
          f"{max(float(np.max(r['_npz']['plain'][:, r['plain_keys'].index('prelude1')])) for r in recs):.1e} (must be 0).", ""]
    flagged = [r for r in recs if float(np.max(r["_npz"]["plain"][:, r["plain_keys"].index("prelude1")])) > 0]
    if flagged:
        clean = [r for r in recs if r not in flagged]
        iE = C.index("err_out")
        f_ = lambda g: (st.mean(float(np.nanmean(r["_npz"]["c"][:, R - 1, NJ - 1, iE])) for r in g),  # noqa: E731
                        st.mean(float(np.mean(r["_npz"]["kl"][1:])) for r in g))
        a_, b_ = f_(recs), f_(clean)
        md += [f"Control deviation: {len(flagged)} prompt(s) show a nonzero prelude FULL-vs-LAST difference although both rows "
               "have identical inputs: " + ", ".join(f"`{r['job']['uid']}` (P={r['P']})" for r in flagged) +
               ". The two rows of the same B=2 BF16 prefill differ numerically at this shape (batch-row kernel noise, not a "
               "policy effect), so this prompt's C/boundary numbers include that noise. Sensitivity: loop-32 hidden error / "
               f"KL with all prompts {a_[0]:.4g} / {a_[1]:.3e}, excluding flagged {b_[0]:.4g} / {b_[1]:.3e}. A/B/D use only the "
               "FULL row and are unaffected.", ""]
    md += ["## A. KV geometry on the FULL trajectory (continuation columns; median over columns×heads)", "",
           "| layer | metric | " + " | ".join(f"r={s}" for s in SHOW_LOOPS) + " |", "|---|---|" + "---:|" * len(SHOW_LOOPS)]
    for j in range(NJ):
        for metric in ("dK", "dV", "cosK", "cosV", "cK", "cV"):
            cells = []
            for s in SHOW_LOOPS:
                x = get(geo_rows, group="all", metric=metric, segment="cont", stat="median", layer=j, loop=s)
                cells.append(f"{x['mean']:.3f}" if x and x["mean"] == x["mean"] else "—")
            md.append(f"| {j} | {metric} | " + " | ".join(cells) + " |")
    md += ["", "Prompt columns (median dK / dV, layer-mean): " + ", ".join(
        f"r={s}: {st.mean(get(geo_rows, group='all', metric='dK', segment='prompt', stat='median', layer=j, loop=s)['mean'] for j in range(NJ)):.3f}/"
        f"{st.mean(get(geo_rows, group='all', metric='dV', segment='prompt', stat='median', layer=j, loop=s)['mean'] for j in range(NJ)):.3f}"
        for s in (1, 8, 16, 31)) + ". Per-head dispersion: `kv_geometry_per_head.csv`; by position: `kv_geometry_by_position.csv`.", ""]

    md += ["## B. Fixed-query attention counterfactual (FULL query; only past recurrent K,V swapped to loop 32)", "",
           "| layer | metric | " + " | ".join(f"r={s}" for s in SHOW_LOOPS) + " |", "|---|---|" + "---:|" * len(SHOW_LOOPS)]
    for j in range(NJ):
        for metric in ("D_swap", "D_konly", "D_vonly", "tv_mean", "cos_swap", "attn_over_resid", "Dswap_over_resid"):
            cells = []
            for s in SHOW_LOOPS:
                x = get(b_rows, group="all", metric=metric, layer=j, loop=s)
                cells.append(f"{x['mean']:.3g}")
            md.append(f"| {j} | {metric} | " + " | ".join(cells) + " |")
    prec = get(b_rows, group="all", metric="D_precision", layer="mean", loop=16)
    lay = get(b_rows, group="all", metric="D_layout", layer="mean", loop=16)
    md += ["", f"Controls (layer mean, r=16): BF16 engine vs FP32 direct attention D_precision {fmt(prec['mean'], prec['ci_lo'], prec['ci_hi'])}; "
           f"reversed-layout D_layout {fmt(lay['mean'], lay['ci_lo'], lay['ci_hi'])}; r=32 swap is an exact no-op. "
           "D_* = ||A_x − A_ref|| / ||A_ref|| after the output projection; Dswap_over_resid = ||A_swap − A_ref|| / ||block input||.", ""]

    md += ["## C. Actual FULL vs LAST sequential execution (same prefix and latent): observed differences", "",
           "| loop r | err_in | err_out (layer 3) | rel ΔA | cos(e_in, ΔA) | rel ΔM | cos(e_mid, ΔM) | attn/resid | mlp/resid |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SHOW_LOOPS:
        cell = lambda m, j="mean": get(c_rows, group="all", metric=m, layer=j, loop=s)["mean"]  # noqa: E731
        md.append(f"| {s} | {cell('err_in', 0):.3g} | {cell('err_out', NJ - 1):.3g} | {cell('rel_da'):.3g} | {cell('cos_err_da'):+.3f} | "
                  f"{cell('rel_dm'):.3g} | {cell('cos_err_dm'):+.3f} | {cell('attn_over_resid'):.3g} | {cell('mlp_over_resid'):.3g} |")
    md += ["", "(err_in at layer 0 of loop r = error entering loop r; ΔA, ΔM, cosines and ratios are layer means; see CSV per layer.)", "",
           "Boundaries (decode tokens): " + "; ".join(
               f"{x['metric']} {fmt(x['mean'], x['ci_lo'], x['ci_hi'])}" for x in bnd if x["group"] == "all") + ".", ""]
    for gname in [g for g in ("gsm8k", "math500") if g in groups]:
        e1 = get(c_rows, group=gname, metric="err_out", layer=NJ - 1, loop=1)
        e16 = get(c_rows, group=gname, metric="err_out", layer=NJ - 1, loop=16)
        e32 = get(c_rows, group=gname, metric="err_out", layer=NJ - 1, loop=R)
        kl = get(bnd, group=gname, metric="kl")
        md.append(f"- {gname}: hidden error at loop end r=1 {e1['mean']:.3g}, r=16 {e16['mean']:.3g}, r=32 {e32['mean']:.3g}; "
                  f"final logits KL {kl['mean']:.2e}.")
    md.append("")

    md += ["## D. Single-call intervention (one attention call reads terminal history; rest FULL)", "",
           "| intervened loop | imm. attn Δ | post-layer Δh | post-R32 Δh | final ln_f Δ | KL(ref‖int) | argmax same | prompts |",
           "|---:|---|---|---|---|---|---:|---:|"]
    for loop in sorted({x["loop"] for x in d_rows}):
        x = get(d_sum, group="all", layer="mean", loop=loop)
        md.append(f"| {loop} | {fmt(x['imm_attn'], x['imm_attn_ci_lo'], x['imm_attn_ci_hi'])} | "
                  f"{fmt(x['post_layer'], x['post_layer_ci_lo'], x['post_layer_ci_hi'])} | {fmt(x['post_R'], x['post_R_ci_lo'], x['post_R_ci_hi'])} | "
                  f"{fmt(x['final_lnf'], x['final_lnf_ci_lo'], x['final_lnf_ci_hi'])} | {fmt(x['kl_ref_int'], x['kl_ref_int_ci_lo'], x['kl_ref_int_ci_hi'])} | "
                  f"{x['argmax_same']:.4f} | {x['prompts']} |")
    md += ["", "Per layer: `intervention_summary.csv`; per token: `intervention_effects.csv`.", ""]

    ctrl = sorted((run / "mechanism" / "float32_control").glob("*.json"))
    if ctrl:
        fp = [json.loads(p.read_text()) for p in ctrl]
        bf = [r for r in recs if r["job"]["uid"] in {x["uid"] for x in fp}]
        md += ["## Precision control: same C and D in FP32 (short prompts: " + ", ".join(f"`{x['uid']}`" for x in fp) + ")", "",
               "| quantity | FP32 | BF16 (same prompts) |", "|---|---:|---:|"]
        for L in sorted({x["loop"] for x in fp[0]["d_records"]}):
            if L == R:
                continue
            for k in ("post_layer", "post_R", "kl_ref_int"):
                a = st.mean(st.mean(x[k] for x in r["d_records"] if x["loop"] == L) for r in fp)
                b = st.mean(st.mean(x[k] for x in r["d_records"] if x["loop"] == L) for r in bf)
                md.append(f"| D, intervene loop {L}: {k} | {a:.3g} | {b:.3g} |")
        iE = C.index("err_out")
        cf = np.mean([x["c_err_out_last_layer_by_loop"] for x in fp], 0)
        cb = np.mean([np.nanmean(r["_npz"]["c"][:, :, NJ - 1, iE], 0) for r in bf], 0)
        for L in (1, 4, 8, 16, 24, R):
            md.append(f"| C, FULL-vs-LAST hidden error at end of loop {L} | {cf[L - 1]:.3g} | {cb[L - 1]:.3g} |")
        md.append(f"| C, FULL-vs-LAST logits KL (decode mean) | {st.mean(x['kl_mean_decode'] for x in fp):.3g} | "
                  f"{st.mean(float(np.mean(r['_npz']['kl'][1:])) for r in bf):.3g} |")
        md.append("")

    md += ["## Negative / outlier cases", ""]
    worst_kl = sorted(recs, key=lambda r: -r["kl_mean"])[:5]
    md += ["Largest FULL-vs-LAST mean KL (C): " + "; ".join(
        f"{r['job']['uid']} KL {r['kl_mean']:.2e}, argmax same {r['argmax_same_rate']:.4f}, n={r['n']}, "
        f"err_out r32 {float(np.nanmean(r['_npz']['c'][:, R - 1, NJ - 1, C.index('err_out')])):.3g}" for r in worst_kl)]
    noncon = []
    for r in recs:
        a = r["_npz"]["c"][:, :, NJ - 1, C.index("err_out")]
        if a.shape[0] and np.nanmean(a[:, R - 1]) > np.nanmean(a[:, 15]):
            noncon.append(f"{r['job']['uid']} (r16 {np.nanmean(a[:, 15]):.3g} → r32 {np.nanmean(a[:, R - 1]):.3g})")
    md += ["", f"Prompts whose loop-end hidden error grows from r=16 to r=32: {len(noncon)}/{len(recs)}" + (": " + "; ".join(noncon[:10]) if noncon else ""),
           "", "Figures: " + ", ".join(f"`{f}`" for f in figs), ""]
    (out / "mechanism_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
