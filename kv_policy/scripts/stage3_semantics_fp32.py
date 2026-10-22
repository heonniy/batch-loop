"""fp32 check that the vllm-rlt layouts implement the same KV semantics as the HF reference cells.

bf16 greedy parity cannot separate semantics from numerics when a policy has collapsed (near-tie
logits). Here both engines run in float32 on short prompts; agreement of whole greedy
continuations is evidence of identical semantics:
  last_exited  vs  HF FULL / decode
  shared       vs  HF LAST / strict      (and, as a negative control, vs HF LAST / decode)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from kvp.models import load, make_runner  # noqa: E402
from stage3_serving import build_engine, first_div, params, run_batch  # noqa: E402

PROMPTS = ["Q: Tom has 3 apples and buys 5 more. How many apples does he have?\nA:",
           "The capital of France is",
           "Q: A train travels 60 miles per hour for 3 hours. How far does it go?\nA:",
           "def fibonacci(n):\n",
           "Problem:\nCompute $7 \\times 8$.\n\nSolution:",
           "List three primary colors:",
           "Q: If x + 4 = 10, what is x?\nA:",
           "Once upon a time, in a small village,"]
N_TOK = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro-1.4b")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec, tok, model = load(args.model, dtype=torch.float32)
    run = make_runner(spec, tok, model)
    ids = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]
    hf = {}
    for pol, mode in (("full", "decode"), ("last", "strict"), ("last", "decode")):
        hf[f"{pol}/{mode}"] = [run.generate([p], [f"s{i}"], pol, mode, N_TOK, set(), lambda b, x: False).gen_ids[0]
                               for i, p in enumerate(ids)]
    del run, model
    gc.collect()
    torch.cuda.empty_cache()
    rep = {"model": args.model, "dtype": "float32", "backend": args.backend, "n_tokens": N_TOK, "results": []}
    for layout, refs in (("last_exited", ["full/decode"]), ("shared", ["last/strict", "last/decode"])):
        m, eng = build_engine(args.model, layout, 8, 8 * 2**30, dtype=torch.float32, backend=args.backend)
        outs, *_ = run_batch(eng, ids, params(spec, N_TOK, True), layout)
        for ref in refs:
            div = [first_div(h, o.token_ids) for h, o in zip(hf[ref], outs)]
            rep["results"].append({"layout": layout, "hf_cell": ref, "first_divergence": div,
                                   "exact": sum(d is None for d in div), "n": len(div),
                                   "rlt_text": [tok.decode(o.token_ids) for o in outs],
                                   "hf_text": [tok.decode(h) for h in hf[ref]]})
            print(layout, "vs", ref, "exact", sum(d is None for d in div), "/", len(div), div)
        eng.close()
        del eng, m
        gc.collect()
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
