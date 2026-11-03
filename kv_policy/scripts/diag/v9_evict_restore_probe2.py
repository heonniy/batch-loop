"""Is the restore_up/restore_ffn explosion a bug or compounding dynamics? Self-reference parity on the evicted path,
and loop/layer-restricted patches."""
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
def run(mode, ref, **kw):
    a = r.run_gate_arm(sources=src, spec=r.spec, plane_map=last, mode=mode, reference=ref, **kw)
    st = a["patch_stats"].get("relative_patch_norm", [0.0])
    gap = max(float((x - y).abs().max()) for x, y in zip(ev["logprobs"], a["logprobs"]))
    return f"dNLL vs clean {a['nll_future']-clean['nll_future']:+.4f} vs evict {a['nll_future']-ev['nll_future']:+.4f} patches {a['n_patched_components']} relpatch mean {np.mean(st):.4f} max {np.max(st):.4f} max|dlogp vs evict| {gap:.2e}"
print("self-ref restore_up  (evict ref on evict path):", run("restore_up", ev["ffn_trace"]))
print("self-ref restore_ffn (evict ref on evict path):", run("restore_ffn", ev["ffn_trace"]))
for mode in ("restore_up", "restore_ffn", "restore_gate"):
    for kw in ({"selected_loops": [1]}, {"selected_loops": [bb.R]}, {"selected_layers": [0]}, {"selected_layers": [bb.n_rec - 1]}, {"selected_columns": [r.P]}):
        print(f"{mode:12s} {str(kw):32s}", run(mode, clean["ffn_trace"], **kw))
