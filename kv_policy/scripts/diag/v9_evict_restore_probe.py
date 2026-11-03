"""Evicted-branch diagnostics: per-component trace change vs clean, and restoration relpatch / NLL per mode."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ffn_gate_causal_sweep import load_backbone, load_v5_jobs, make_runner, last_plane_map, source_windows
model = sys.argv[1]
bb, tok, eos = load_backbone(model)
doc = load_v5_jobs(model, "controlled_retrieval")[1]
r, p = make_runner(bb, doc, len(doc["target_ids"]) + 2)
r.reference(eos, frozen_ids=list(doc["target_ids"]))
last = last_plane_map(r)
clean = r.run_gate_arm(sources=(), spec=r.spec, plane_map=last, mode="capture")
src = source_windows(p, doc["uid"], 4)[0]["positions"]
ev = r.run_gate_arm(sources=src, spec=r.spec, plane_map=last, mode="capture")
print("evict dNLL %+.4f" % (ev["nll_future"] - clean["nll_future"]))
comps = sorted({k[3] for k in clean["ffn_trace"]})
for c in comps:
    rel = [float((ev["ffn_trace"][k].float() - clean["ffn_trace"][k].float()).norm() / (clean["ffn_trace"][k].float().norm() + 1e-8))
           for k in clean["ffn_trace"] if k[3] == c]
    print(f"trace rel change {c}: mean {np.mean(rel):.4f} max {np.max(rel):.4f}")
for mode in ("restore_gate", "restore_up", "shift_gate", "restore_ffn"):
    arm = r.run_gate_arm(sources=src, spec=r.spec, plane_map=last, mode=mode, reference=clean["ffn_trace"])
    st = arm["patch_stats"].get("relative_patch_norm", [])
    print(f"{mode:13s} dNLL vs clean {arm['nll_future'] - clean['nll_future']:+.4f}  patches {arm['n_patched_components']}  relpatch mean {np.mean(st):.4f} max {np.max(st):.4f} first5 {[round(x,4) for x in st[:5]]}")
# reference integrity after the arms
chk = r.run_gate_arm(sources=(), spec=r.spec, plane_map=last, mode="capture")
print("clean trace unchanged after arms:", all(torch.equal(clean["ffn_trace"][k], chk["ffn_trace"][k]) for k in clean["ffn_trace"]))
