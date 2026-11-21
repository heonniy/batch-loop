"""Compact console view of analyze_nodes output for diagnostic (gpu_diagnostic) nodes."""
import json
import sys

for path in sys.argv[1:]:
    m = json.load(open(path))
    print("==", path.split("/")[-1], "samples", m.get("sample_count"))
    for k, c in sorted(m["cells"].items()):
        cur = c["curve"]
        pts = " ".join(f"k{x['k']}:{x['top1']:.3f}/{x['kl']:.3f}" for x in cur if x["k"] in (1, 2, 4, 8, 16) or x["k"] == len(cur))
        k99 = next((x["k"] for x in cur if x["top1"] >= 0.99), None)
        last = cur[-1]
        line = (f"{k:62s} serial {c['serial_ref_ms']:.1f}ms | {pts} | k@top1>=.99 {k99} | "
                f"rows/tok@last {last['rows']:.0f} vs serial {c['rows_per_serial_token']} | solver@last {last['solver_ms']:.1f}ms")
        if "jvp" in c:
            j = c["jvp"]
            line += (f" | JVP gain {j['gain_q10_50_90'][1]:.2f} a* {j['alpha_opt_q10_50_90'][1]:.2f} "
                     f"errI {j['rel_err_identity_q'][1]:.2f} errA {j['rel_err_scaled_q'][1]:.2f} errBest {j['rel_err_best_scalar_q'][1]:.2f}")
        print(line)
