"""Summarize E2 raw trials: medians/IQR, stage shares, marginal batch scaling, knee B*, roofline estimates.

Bottleneck labels are assigned only where two independent signals agree:
  launch/host-dominated : Nsight Systems GPU busy fraction < 0.5 for the cell (profiled subset only), or
                          Core time within 10% of B=1 while achieved weight bandwidth < 25% of HBM peak;
  weight-memory-bound   : busy >= 0.5 AND achieved (weights+KV) bandwidth >= 50% of HBM peak AND AI < ridge;
  KV-dominated          : KV bytes > weight bytes AND marginal latency grows with L at fixed B;
  compute-bound         : achieved FLOP rate >= 50% of BF16 dense peak.
Otherwise the label is `unresolved`. H100 SXM peaks used: 3.35 TB/s HBM3, 989 TFLOP/s BF16 dense.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HF = Path.home() / ".cache/huggingface/hub"
SNAP = {"ouro14": HF / "models--ByteDance--Ouro-1.4B/snapshots/37bc7e8ff85a90d813dc096383405f521b6a67b0",
        "ouro26": HF / "models--ByteDance--Ouro-2.6B/snapshots/f8e38372d72a8aa7f4ea9f14b98739a311589088",
        "huginn": HF / "models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c"}
HBM_BPS, BF16_FLOPS = 3.35e12, 989e12
RIDGE = BF16_FLOPS / HBM_BPS
STAGES = ("prelude_ms", "core_ms", "core_kv_update_ms", "core_dispatch_ms", "core_wall_ms", "gate_ms", "exit_kv_ms",
          "compaction_ms", "coda_blocks_ms", "lm_head_ms")


def core_params(model: str) -> tuple[int, dict]:
    """Parameter counts by stage from safetensors headers (no weights loaded)."""
    from safetensors import safe_open
    snap = SNAP[model]
    files = sorted(snap.glob("*.safetensors"))
    groups = defaultdict(int)
    for f in files:
        with safe_open(str(f), "pt") as h:
            for name in h.keys():
                n = 1
                for d in h.get_slice(name).get_shape():
                    n *= d
                if model == "huginn":
                    g = ("core" if "core_block" in name else "prelude" if ".prelude." in name else
                         "coda" if ".coda." in name else "embedding" if "wte" in name else
                         "lm_head" if "lm_head" in name else "other")
                else:
                    g = ("core" if ".layers." in name else "embedding" if "embed_tokens" in name else
                         "lm_head" if "lm_head" in name else "other")
                groups[g] += n
    return groups["core"], dict(groups)


def kv_bytes_per_core_call(model: str, cfg: dict, B: int, L: int) -> float:
    if model == "huginn":  # core block of 4 physical layers, 55 heads x 96
        return 4 * 2 * B * L * cfg["n_embd"] * 2
    return cfg["num_hidden_layers"] * 2 * B * L * cfg["hidden_size"] * 2  # one depth slot per layer is read


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("e2_root", type=Path)
    a = p.parse_args()
    cells, status = defaultdict(list), {}
    for model in ("ouro14", "ouro26", "huginn"):
        for r in csv.DictReader((a.e2_root / model / "e2_stage_latency_raw.csv").open()):
            key = (model, int(r["B"]), int(r["L"]))
            status[key] = r["status"]
            if r["status"] == "OK":
                cells[key].append(r)
    gates = {(g["model"], g["B"], g["L"]): g for m in ("ouro14", "ouro26", "huginn")
             for g in json.loads((a.e2_root / m / "e2_correctness_gate.json").read_text())}
    nsys = {}
    if (a.e2_root / "nsys/e2_nsys_summary.csv").exists():
        for r in csv.DictReader((a.e2_root / "nsys/e2_nsys_summary.csv").open()):
            m, b, l = r["cell"].split("_")
            nsys[(m, int(b[1:]), int(l[1:]))] = float(r["gpu_busy_fraction"])
    params = {m: core_params(m) for m in SNAP}
    cfgs = {m: json.loads((SNAP[m] / "config.json").read_text()) for m in SNAP}
    rows = []
    for key in sorted(status):
        model, B, L = key
        base = {"model": model, "B": B, "L": L, "status": status[key], "evidence_class": "PHYSICAL_FORCED_DEPTH",
                "gate": gates.get(key, {}).get("status"), "row0_rel_l2_vs_B1": gates.get(key, {}).get("row0_rel_l2_vs_B1")}
        if status[key] != "OK":
            rows.append(base)
            continue
        rs = cells[key]
        for s in STAGES:
            v = sorted(float(r[s]) for r in rs)
            base[f"{s}_median"] = statistics.median(v)
            base[f"{s}_q25"] = v[len(v) // 4]
            base[f"{s}_q75"] = v[(3 * len(v)) // 4]
        base["n_trials"] = len(rs)
        base["peak_alloc_gib"] = max(float(r["peak_alloc_gib"]) for r in rs)
        boundary = base["prelude_ms_median"] + base["coda_blocks_ms_median"] + base["lm_head_ms_median"]
        base["core_share_of_core_plus_boundary"] = base["core_ms_median"] / (base["core_ms_median"] + boundary)
        core_p = params[model][0]
        core_p_per_call = core_p if model != "huginn" else core_p  # Huginn core block = 4 layers, one call
        wbytes = core_p_per_call * 2
        kvb = kv_bytes_per_core_call(model, cfgs[model], B, L)
        flops = 2 * core_p_per_call * B + 4 * B * L * (cfgs[model].get("hidden_size") or cfgs[model]["n_embd"]) * \
            (cfgs[model].get("num_hidden_layers") or 4)
        t = base["core_ms_median"] / 1e3
        base.update({"weight_bytes": wbytes, "kv_bytes": kvb, "flops": flops,
                     "arithmetic_intensity": flops / (wbytes + kvb),
                     "achieved_bw_frac_of_peak": (wbytes + kvb) / t / HBM_BPS,
                     "achieved_flops_frac_of_peak": flops / t / BF16_FLOPS,
                     "nsys_gpu_busy_fraction": nsys.get(key)})
        rows.append(base)
    by = {(r["model"], r["B"], r["L"]): r for r in rows}
    for r in rows:
        if r["status"] != "OK":
            continue
        b1 = by.get((r["model"], 1, r["L"]))
        prev = max((x for x in by.values() if x["model"] == r["model"] and x["L"] == r["L"] and x["status"] == "OK"
                    and x["B"] < r["B"]), key=lambda x: x["B"], default=None)
        r["core_rel_to_B1"] = r["core_ms_median"] / b1["core_ms_median"] if b1 and b1["status"] == "OK" else None
        r["marginal_ms_per_added_row"] = ((r["core_ms_median"] - prev["core_ms_median"]) / (r["B"] - prev["B"])
                                          if prev else None)
        busy = r["nsys_gpu_busy_fraction"]
        flat = r["core_rel_to_B1"] is not None and r["core_rel_to_B1"] <= 1.10
        if (busy is not None and busy < 0.5) or (flat and r["achieved_bw_frac_of_peak"] < 0.25):
            label = "launch/host-dominated"
        elif r["achieved_flops_frac_of_peak"] >= 0.5:
            label = "compute-bound"
        elif r["kv_bytes"] > r["weight_bytes"] and r["achieved_bw_frac_of_peak"] >= 0.5:
            label = "KV-bandwidth-bound"
        elif (busy is not None and busy >= 0.5) and r["achieved_bw_frac_of_peak"] >= 0.5 and r["arithmetic_intensity"] < RIDGE:
            label = "weight-memory-bound"
        else:
            label = "unresolved"
        r["bottleneck_label"] = label
    # knee: largest B whose Core median stays within +10% of B=1 (q75 vs q25 bounds as uncertainty)
    knees = []
    for model in SNAP:
        for L in sorted({r["L"] for r in rows if r["model"] == model}):
            ok = sorted((r for r in rows if r["model"] == model and r["L"] == L and r["status"] == "OK"), key=lambda r: r["B"])
            if not ok or ok[0]["B"] != 1:
                continue
            t1 = ok[0]["core_ms_median"]
            knee = max((r["B"] for r in ok if r["core_ms_median"] <= 1.10 * t1), default=None)
            lo = max((r["B"] for r in ok if r["core_ms_q75"] <= 1.10 * ok[0]["core_ms_q25"]), default=None)
            hi = max((r["B"] for r in ok if r["core_ms_q25"] <= 1.10 * ok[0]["core_ms_q75"]), default=None)
            knees.append({"model": model, "L": L, "objective": "largest B with Core median <= 1.10 x Core(B=1)",
                          "B_star": knee, "B_star_conservative": lo, "B_star_optimistic": hi,
                          "max_B_measured": ok[-1]["B"], "core_B1_ms": t1, "core_maxB_ms": ok[-1]["core_ms_median"]})
    out = a.e2_root
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "e2_stage_latency.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (out / "e2_knee.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(knees[0]))
        w.writeheader()
        w.writerows(knees)
    (out / "e2_params_by_stage.json").write_text(json.dumps({m: v[1] for m, v in params.items()}, indent=2) + "\n")
    plot(rows, out)
    print(json.dumps({"cells": len(rows), "oom": sum(r["status"] == "OOM" for r in rows),
                      "gate_fail": sum(r.get("gate") == "FAIL" for r in rows)}, indent=1))


def plot(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seq = ["#cde2fb", "#9ec5f2", "#6da6e8", "#3987e5", "#2a78d6", "#1c5fae", "#0f4483"]
    models = ("ouro14", "ouro26", "huginn")
    Ls = sorted({r["L"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, m in zip(axes, models):
        for i, L in enumerate(Ls):
            pts = sorted((r["B"], r["core_ms_median"]) for r in rows if r["model"] == m and r["L"] == L and r["status"] == "OK")
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=4, lw=2, color=seq[i], label=f"L={L}")
            oom = [r["B"] for r in rows if r["model"] == m and r["L"] == L and r["status"] == "OOM"]
            if oom and pts:
                ax.scatter([min(oom)], [pts[-1][1]], marker="x", color=seq[i], s=40)
        ax.set_xscale("log", base=2)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("active rows B (x = first OOM)")
        ax.grid(color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("one recurrent Core invocation (ms, median)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("E2 H100: Core latency vs active batch, by context (PHYSICAL, fixed state)", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(out / "e2_core_latency_vs_B.png", dpi=150, facecolor="#fcfcfb")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    colors = {"prelude": "#2a78d6", "core": "#eb6834", "coda blocks": "#1baf7a", "lm head": "#e87ba4"}
    for ax, m in zip(axes, models):
        sel = sorted((r for r in rows if r["model"] == m and r["L"] == 512 and r["status"] == "OK"), key=lambda r: r["B"])
        xs = [str(r["B"]) for r in sel]
        bottom = [0.0] * len(sel)
        for name, key in (("prelude", "prelude_ms_median"), ("core", "core_ms_median"),
                          ("coda blocks", "coda_blocks_ms_median"), ("lm head", "lm_head_ms_median")):
            vals = [r[key] for r in sel]
            ax.bar(xs, vals, bottom=bottom, color=colors[name], label=name, edgecolor="#fcfcfb", linewidth=1)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"{m}, L=512 (one Core call)", fontsize=10)
        ax.set_xlabel("B")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("ms (median)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e2_stage_breakdown_L512.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
