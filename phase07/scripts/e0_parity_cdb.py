"""E0 live-exit B=1 vs B>1 parity on the pinned optimized CDB (CUDA graphs, paged FA3, async, no-refill).

The upstream engine runs its own live exit policy (Ouro cumulative hazard >= q; Huginn latent diff < tau),
immediate gate consumption, refill=False, with max_num_seqs = 1 (sequential B=1 reference) and 8 / 32
(native no-refill cohort = resident set). Per-request exit steps are recorded through a thin wrapper around
the engine's `_record_exit`; generated tokens are compared directly. Evidence: PHYSICAL_LIVE_EXIT correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cdb_ext"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdb_ext.cdb_common import MODELS, env_info, gpu_identity, load_model  # noqa: E402

PROMPT_TEXT = Path(__file__).resolve().parents[1] / "inputs/test_prompts.json"  # Phase-07 test split (frozen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("ouro14", "huginn"))
    ap.add_argument("--threshold", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=16)
    a = ap.parse_args()
    import torch
    from looped_cdb.continuous_depth_batching import ContinuousDepthBatchingConfig, ContinuousDepthBatchingEngine
    from looped_cdb.continuous_depth_batching import continuous_api as api
    model, tok = load_model(a.model)
    texts = json.loads(PROMPT_TEXT.read_text())["alpaca"]["prompts"][:a.n]
    ids = [tok(t)["input_ids"][:64] for t in texts]
    exits = defaultdict(list)
    orig = api.ContinuousDepthBatchingEngine._record_exit

    def rec(self, item, step):
        exits[item.state.request_id].append(int(step) + 1)
        return orig(self, item, step)
    api.ContinuousDepthBatchingEngine._record_exit = rec
    res = {}
    for width in (1, 8, 32):
        cfg = ContinuousDepthBatchingConfig(num_blocks=4096, max_num_seqs=max(width, 2),  # width-1 engines fail prefill-graph warmup upstream
                                            max_model_len=1024,
                                            max_recurrent_steps=MODELS[a.model]["rmax"], exit_threshold=a.threshold,
                                            refill=False, use_cuda_graph=True, use_cuda_graph_prefill=False, use_async_batching=True,
                                            delay_gate_consumption=False, min_recurrent_steps=1, kv_policy="single",
                                            kv_pressure_mode="none", safety_margin=0.0)
        eng = ContinuousDepthBatchingEngine.from_model(model, cdb_config=cfg, dtype=torch.bfloat16)
        eng.generate_batch(ids[:max(1, min(width, 4))], max_new_tokens=4, eos_token_id=None, warmup=True)
        exits.clear()
        if width == 1:  # B=1 reference: one request per generate call (a width-1 engine cannot co-prefill)
            toks, ex = [], []
            for one in ids:
                exits.clear()
                toks.append(list(eng.generate_batch([one], max_new_tokens=a.tokens + 1, eos_token_id=None)[0].generated_tokens))
                ex.append(next(iter(exits.values()), []))
        else:
            outs = eng.generate_batch(ids, max_new_tokens=a.tokens + 1, eos_token_id=None)
            toks = [list(o.generated_tokens) for o in outs]
            order = sorted(exits, key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))
            ex = [exits[k] for k in order]
        res[width] = {"tokens": toks, "exits": ex, "exit_hist": dict(Counter(x for v in ex for x in v))}
        del eng
        torch.cuda.empty_cache()
    ref = res[1]
    report = {"model": a.model, "threshold": a.threshold, "gpu": gpu_identity(), "env": env_info(), "n": a.n,
              "decode_tokens": a.tokens, "widths": {}}
    for width in (8, 32):
        r = res[width]
        tok_same = sum(x == y for x, y in zip(ref["tokens"], r["tokens"]))
        pos = sum(min(len(x), len(y)) for x, y in zip(ref["exits"], r["exits"]))
        flips = sum(int(p != q) for x, y in zip(ref["exits"], r["exits"]) for p, q in zip(x, y))
        report["widths"][width] = {"requests_with_identical_tokens": tok_same, "requests": a.n,
                                   "exit_step_flips": flips, "exit_positions_compared": pos,
                                   "exit_hist": r["exit_hist"], "b1_exit_hist": ref["exit_hist"]}
        print(json.dumps({"width": width, "identical_token_requests": tok_same, "exit_flips": flips, "positions": pos}))
    report["live_parity"] = ("FAIL" if any(v["exit_step_flips"] or v["requests_with_identical_tokens"] < a.n
                                            for v in report["widths"].values()) else "PASS")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"e0_parity_cdb_{a.model}_t{a.threshold}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
