"""Find the retaining allocation stacks for the per-decode CUDA memory growth."""
import json, sys, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from query_conditioned_sweep import V5_RUN, _load
from loopzip_failure_screen import make_prompt
from kvp.query_conditioned_probe import QueryConditionedRunner
model = sys.argv[1]
v5 = json.loads((V5_RUN / "jobs.json").read_text())
j = next(x for x in v5 if x["model"] == model and x["dataset"] == "controlled_retrieval")
tok, bb, eos = _load(model)
p = make_prompt(j)
r = QueryConditionedRunner(bb, p, j["uid"], horizon=40)
r.reference(eos, frozen_ids=list(j["target_ids"]))
r.decode((), r.spec)
torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
torch.cuda.memory._record_memory_history(max_entries=200000)
for i in range(2):
    r.decode((), r.spec)
torch.cuda.synchronize()
print("growth GB", (torch.cuda.memory_allocated() - base) / 2**30)
snap = torch.cuda.memory._snapshot()
agg = collections.Counter(); cnt = collections.Counter()
for seg in snap["segments"]:
    for blk in seg["blocks"]:
        if blk["state"] != "active_allocated":
            continue
        frames = blk.get("frames", [])
        key = " <- ".join(f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in frames if "kv_policy" in f["filename"])[:400]
        agg[key] += blk["size"]; cnt[key] += 1
for k, v in agg.most_common(8):
    print(f"{v/2**20:8.1f} MiB  n={cnt[k]:6d}  {k}")
