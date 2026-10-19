"""Single per-model/regime summary table joining E2 stage latency and E3 Oracle comparison (CPU only)."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_latency_oracle import ROOT, load_trace  # noqa: E402


def nearest_e2(e2_rows, model, B, L):
    cands = [r for r in e2_rows if r["model"] == model and r["status"] == "OK" and int(r["B"]) == B]
    return min(cands, key=lambda r: abs(int(r["L"]) - L)) if cands else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e1", required=True, type=Path)
    p.add_argument("--e2", required=True, type=Path)
    p.add_argument("--e3", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    e2 = list(csv.DictReader((a.e2 / "e2_stage_latency.csv").open()))
    e3 = list(csv.DictReader((a.e3 / "e3_oracle_comparison.csv").open()))
    traces = {}
    rows = []
    cells = sorted({(r["trace"], r["cell"]) for r in e3})
    for trace, cell in cells:
        if trace not in traces:
            traces[trace] = load_trace(ROOT / trace)
        tr = traces[trace]
        g = {r["policy"]: r for r in e3 if r["trace"] == trace and r["cell"] == cell}
        f, d, lat = g["fifo"], g["depth_oracle"], g["latency_oracle"]
        N, B, T, W = int(f["N"]), int(f["B"]), int(f["T"]), int(f["W"])
        depth = tr["depth"][:T, :N]
        L_mid = W + T // 2
        s = nearest_e2(e2, f["model"], min(B, 128), L_mid)
        s1 = nearest_e2(e2, f["model"], 1, L_mid)
        fl = lambda r, k: float(r[k]) if r and r.get(k) not in ("", None) else None
        bnd = (fl(s, "prelude_ms_median") or 0) + (fl(s, "coda_blocks_ms_median") or 0) + (fl(s, "lm_head_ms_median") or 0)
        rows.append({
            "model": f["model"], "trace": Path(trace).name, "threshold": tr["threshold"], "N": N, "B": B, "N_over_B": N // B,
            "T": T, "L_range": f"{W + 1}-{W + T}", "depth_mean": round(float(depth.mean()), 3), "depth_max": int(depth.max()),
            "e2_cell_used": f"B{s['B']}_L{s['L']}" if s else None,
            "core_share_E2": round(fl(s, "core_ms_median") / (fl(s, "core_ms_median") + bnd), 3) if s else None,
            "prelude_share_E2": round((fl(s, "prelude_ms_median") or 0) / (fl(s, "core_ms_median") + bnd), 3) if s else None,
            "coda_share_E2": round(((fl(s, "coda_blocks_ms_median") or 0) + (fl(s, "lm_head_ms_median") or 0)) /
                                   (fl(s, "core_ms_median") + bnd), 3) if s else None,
            "core_B_over_B1_E2": round(fl(s, "core_ms_median") / fl(s1, "core_ms_median"), 3) if s and s1 else None,
            "bottleneck_E2": s["bottleneck_label"] if s else None,
            "RIC_fifo": f["RIC"], "RIC_depth_oracle": d["RIC"], "RIC_latency_oracle": lat["RIC"],
            "pred_core_ms_fifo": round(float(f["pred_core_ms"])), "pred_core_ms_depth": round(float(d["pred_core_ms"])),
            "pred_core_ms_latency": round(float(lat["pred_core_ms"])),
            "fifo_pred_over_measured": round(float(f["fifo_pred_over_measured"]), 3),
            "latency_model_calibration": "ACCEPTED" if f["calibration_accepted"] == "True" else "REJECTED (>10% FIFO residual)",
            **{f"core_ms_{k}": round(float(g[k]["core_ms_median"])) for k in ("fifo", "depth_oracle", "latency_oracle")},
            **{f"wall_s_{k}": f"{float(g[k]['wall_s_median']):.2f} [{float(g[k]['wall_s_min']):.2f}-{float(g[k]['wall_s_max']):.2f}]"
               for k in ("fifo", "depth_oracle", "latency_oracle", "random")},
            "wall_reduction_depth": round(float(d["wall_reduction_vs_fifo"]), 4),
            "wall_reduction_latency": round(float(lat["wall_reduction_vs_fifo"]), 4),
            "wall_reduction_random": round(float(g["random"]["wall_reduction_vs_fifo"]), 4),
            "evidence_RIC": "TRACE_ONLY", "evidence_pred": "LATENCY_MODEL", "evidence_E2": "PHYSICAL_FORCED_DEPTH (stage timing)",
            "evidence_core_wall": "PHYSICAL_FORCED_DEPTH (frozen B1 depths; not live exit)"})
    with (a.out / "summary_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.copy(a.e1 / "e1_regime_cells.csv", a.out / "e1_regime_cells.csv")
    shutil.copy(a.e2 / "e2_stage_latency.csv", a.out / "e2_stage_latency.csv")
    shutil.copy(a.e3 / "e3_oracle_comparison.csv", a.out / "e3_oracle_comparison.csv")
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
