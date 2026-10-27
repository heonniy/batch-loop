"""FULL/strict vs FULL/decode on real checkpoints (FULL_STRICT_BENICE_VALIDATION.md audit).

Same FULL read semantics, different prompt schedule (token-major vs loop-major parallel). For the first
4 GSM8K + 4 MATH500 pilot items: teacher-forced logits over prompt + 64 gold tokens (max |Δlogit|, KL,
top-1 agreement, gold NLL difference) and 64-token greedy generations (first divergence), in FP32 and
BF16. Writes <out>/mode_parity_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.jobs import gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402

SEL = Path(__file__).resolve().parents[2] / "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json"


def first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = json.loads(SEL.read_text())
    rows = []
    for dt in (torch.float32, torch.bfloat16):
        spec, tok, model = load(args.model, dtype=dt)
        run = make_runner(spec, tok, model)
        for ds in ("gsm8k", "math500"):
            items = {it["uid"]: it for it in prepare_items(args.model, ds, tok)[0]}
            for uid in sel[ds][:4]:
                it = items[uid]
                p, c = it["prompt_ids"], it["target_ids"][:64]
                with torch.inference_mode():
                    a = run.teacher_forced_logits(p, c, uid, "full", "decode").double()
                    b = run.teacher_forced_logits(p, c, uid, "full", "strict").double()
                    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
                    kl = (la.exp() * (la - lb)).sum(-1)
                    P = len(p)
                    tg = torch.tensor(c)
                    nll = lambda l: float(-l[P - 1: P - 1 + len(c)].gather(1, tg[:, None]).mean())  # noqa: E731
                    never = lambda bb, ids: False  # noqa: E731
                    ga = run.generate([p], [uid], "full", "decode", 64, set(), never).gen_ids[0]
                    gb = run.generate([p], [uid], "full", "strict", 64, set(), never).gen_ids[0]
                rows.append({"dtype": str(dt).split(".")[-1], "dataset": ds, "uid": uid, "P": P,
                             "max_abs_logit_diff": float((a - b).abs().max()), "max_kl": float(kl.max()),
                             "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).double().mean()),
                             "gold_nll_decode": nll(la), "gold_nll_strict": nll(lb),
                             "greedy64_first_divergence": first_div(ga, gb)})
        del model, run
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"model": args.model, "gpu": gpu_identity(), "rows": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
