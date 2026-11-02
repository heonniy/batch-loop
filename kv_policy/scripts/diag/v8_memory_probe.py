"""Memory probe for the v8 plane-map decode path on a real checkpoint (one unit)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from query_conditioned_sweep import V5_RUN, _load  # noqa: E402
from loopzip_failure_screen import make_prompt  # noqa: E402
from kvp.query_conditioned_probe import QueryConditionedRunner  # noqa: E402
from kvp.plane_policies import plane_map_from_sets, plane_sets  # noqa: E402

model = sys.argv[1]
v5 = json.loads((V5_RUN / "jobs.json").read_text())
j = next(x for x in v5 if x["model"] == model and x["dataset"] == "controlled_retrieval")
tok, bb, eos = _load(model)
p = make_prompt(j)
r = QueryConditionedRunner(bb, p, j["uid"], horizon=40)
r.reference(eos, frozen_ids=list(j["target_ids"]))
R, P = bb.R, r.P
mem = lambda: torch.cuda.memory_allocated() / 2**30  # noqa: E731
print("after reference", f"{mem():.2f} GB", flush=True)
content = list(p.content_positions)
m = plane_map_from_sets(plane_sets("last", content, R, 2), R, P)
for i in range(6):
    t0 = time.time()
    d = r.decode((), r.spec, plane_map=m)
    print(f"decode {i}: {mem():.2f} GB  {time.time() - t0:.1f}s  nll {d['nll_future']:.3f}", flush=True)
for i in range(3):
    t0 = time.time()
    g = r.greedy((), 24, eos, plane_map=m)
    print(f"greedy {i}: {mem():.2f} GB  {time.time() - t0:.1f}s  len {len(g)}", flush=True)
for i in range(3):
    d = r.decode((), r.spec)
    print(f"plain decode {i}: {mem():.2f} GB", flush=True)
print(torch.cuda.memory_summary(abbreviated=True)[:1500])
