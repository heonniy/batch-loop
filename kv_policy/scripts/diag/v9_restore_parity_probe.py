"""Clean-branch parity of every restoration mode (reference = the clean trace itself)."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ffn_gate_causal_sweep import load_backbone, load_v5_jobs, make_runner, last_plane_map
model = sys.argv[1]
bb, tok, eos = load_backbone(model)
doc = load_v5_jobs(model, "controlled_retrieval")[0]
r, p = make_runner(bb, doc, len(doc["target_ids"]) + 2)
r.reference(eos, frozen_ids=list(doc["target_ids"]))
last = last_plane_map(r)
clean = r.run_gate_arm(sources=(), spec=r.spec, plane_map=last, mode="capture")
keys = list(clean["ffn_trace"].keys())[:6]
print("trace keys sample", keys, "n", len(clean["ffn_trace"]))
for k in keys[:3]:
    t = clean["ffn_trace"][k]; print(k, tuple(t.shape), t.dtype, float(t.float().norm()))
for mode in ("restore_gate", "restore_up", "shift_gate", "restore_ffn"):
    arm = r.run_gate_arm(sources=(), spec=r.spec, plane_map=last, mode=mode, reference=clean["ffn_trace"])
    gap = max(float((x - y).abs().max()) for x, y in zip(clean["logprobs"], arm["logprobs"]))
    st = arm["patch_stats"].get("relative_patch_norm", [])
    print(f"{mode:13s} max|dlogp| {gap:.3e}  patches {arm['n_patched_components']}  mean relpatch {np.mean(st) if st else None}")
# second capture to check trace stability
clean2 = r.run_gate_arm(sources=(), spec=r.spec, plane_map=last, mode="capture")
diffs = {c: max(float((clean["ffn_trace"][k].float() - clean2["ffn_trace"][k].float()).abs().max()) for k in clean["ffn_trace"] if k[3] == c) for c in {k[3] for k in clean["ffn_trace"]}}
print("capture repeat max abs diff by component", diffs)
