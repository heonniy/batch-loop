"""E0 BF16 batch-shape parity on the phase02 eager runtime (Phase-04 question re-checked on Phase-07 data).

The first 64 test requests of a collected lane are decoded together with LIVE exits at cohort width B (8, 32),
teacher-forced with the B=1 reference tokens. Live depths / next tokens are compared with the B=1 labels.
Differences are measured outcomes (PHYSICAL_LIVE_EXIT correctness evidence), not silently corrected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import MODELS, PROMPT_CAP, SPLITS, dataset_prompts, gpu_identity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-root", required=True, type=Path)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    import torch
    from phase02_runtime.common import run_interval
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    man = json.loads((a.features_root / a.lane / "test/manifest.json").read_text())
    meta = np.load(a.features_root / a.lane / "test/meta.npz")
    family, snap, rmax = MODELS[man["model"]]
    n, T = a.n, man["T"]
    labels, tokens, nexts = meta["depth"][:, :n], meta["token"][:, :n], meta["next_token"][:, :n]
    base = HuginnAdapter.load(1, snap, state_init="native-seeded") if family == "huginn" else OuroAdapter.load(1, snap)
    model, tok = base.model, base.tokenizer
    lo, hi = SPLITS["test"]
    prompts = dataset_prompts(man["dataset"])[lo:hi][:n]
    enc = tok(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_CAP)
    report = {"lane": a.lane, "gpu": gpu_identity(), "n": n, "T": T, "threshold": man["threshold"], "widths": {}}
    with torch.inference_mode():
        for B in (8, 32):
            ad = (HuginnAdapter(model, tok, n, state_init="native-seeded", identity_seed=1,
                                canonical_ids=torch.arange(n, device="cuda"), instrumentation="benchmark")
                  if family == "huginn" else OuroAdapter(model, tok, n, instrumentation="benchmark"))
            ad.prefill(enc["input_ids"].cuda(), enc["attention_mask"].cuda().bool())
            live = np.zeros((T, n), np.int64)
            nxt = np.zeros((T, n), np.int64)
            known = torch.as_tensor(tokens, device="cuda")
            for t in range(T):
                ad.begin_token()
                for first in range(0, n, B):
                    rows = torch.arange(first, min(n, first + B), device="cuda")
                    ad.backend.set_active(rows)
                    emb, st = ad.prelude(known[t].index_select(0, rows).unsqueeze(1))
                    res = run_interval(ad, emb, st, rows, rows, exit_source="live", threshold=man["threshold"],
                                       instrumentation="benchmark", debug_ids=False)
                    ad.backend.set_active(rows)
                    nxt[t, first:first + B] = ad.coda(res.final_state).reshape(-1).cpu().numpy()
                    live[t, first:first + B] = res.exit_depths.cpu().numpy()
                ad.commit_token()
            delta = Counter((live - labels).ravel().tolist())
            report["widths"][B] = {"depth_flips": int((live != labels).sum()), "positions": int(labels.size),
                                   "depth_flip_rate": float((live != labels).mean()),
                                   "delta_histogram": {str(k): v for k, v in sorted(delta.items())},
                                   "next_token_flips": int((nxt != nexts).sum()), "next_token_flip_rate": float((nxt != nexts).mean()),
                                   "b1_work": int(labels.sum()), "live_work": int(live.sum())}
            print(json.dumps({"B": B, **{k: report["widths"][B][k] for k in ("depth_flip_rate", "next_token_flip_rate")}}),
                  flush=True)
            del ad
            torch.cuda.empty_cache()
    report["gate_semantic"] = ("FAIL (batch-shape-dependent live depths persist; Phase-04 result reproduced)"
                               if any(v["depth_flips"] for v in report["widths"].values()) else "PASS (no flips observed)")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"e0_parity_eager_{a.lane}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
