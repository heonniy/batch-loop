"""E0: engine fixed-depth decode vs the model author's HF-native forward (GPU, opt-in).

For 16 deterministic prompts (equal length L) the engine (wf_engine.DepthEngine, FULL depth-indexed
KV, cached decode) greedily generates N tokens at fixed depth D. The SAME weights are then run
through the checkpoint's own `trust_remote_code` forward on the full sequence (prompt + generated,
no cache) at depth D, teacher-forced. We compare, at every generated position:
  greedy token (engine) vs argmax(HF logits), max |logit diff|, KL(HF || engine).
Huginn: HF draws its initial latent state with torch.randn; we pass the engine's seeded state
(`input_states`) so both start from the same h0 (prefill draw for prompt positions, per-position
draws for decode positions — exactly what the engine uses). Ouro: HF `total_ut_steps` set to D and
logits taken from the last UT step's hidden state.
Writes <out>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import wf_engine as W
from .depth_diag import cohort_texts, env_info


def ouro_mask_shim(lm) -> None:
    """Call-boundary compatibility shim (no math change): the checkpoint's remote code passes
    `input_embeds`/`cache_position` to transformers' create_*_mask, whose installed (5.22)
    signature expects `inputs_embeds` and has no `cache_position`."""
    import sys
    mod = sys.modules[type(lm.model).__module__]
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "_pd_shim", False):
            continue

        def f(*args, _orig=orig, **kw):
            if "input_embeds" in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return _orig(*args, **kw)
        f._pd_shim = True
        setattr(mod, name, f)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(W.MODEL_IDS))
    p.add_argument("--depths", required=True, help="comma list")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--new", type=int, default=32)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--out", required=True)
    p.add_argument("--allow-gpu", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu:
        raise SystemExit("explicit --allow-gpu required")
    a.depth = None
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    lm, tok, batch_cls, kind = W.load_model(a.model, dtype)
    ids = W.make_cohort(tok, cohort_texts("gsm8k3", 4 * a.n), a.prompt_len, a.n)
    res = dict(config=vars(a) | env_info(a), cells={})
    for D in [int(x) for x in a.depths.split(",")]:
        eng = W.DepthEngine(lm, batch_cls, kind, D, a.n, a.prompt_len + a.new + 8)
        lg0 = eng.prefill(ids)
        e_logits = [lg0]
        cur = lg0.argmax(-1)
        gen = [cur]
        for _ in range(a.new - 1):
            h0 = eng.prelude(cur.tolist())
            _, lg = eng.commit(h0)
            e_logits.append(lg)
            cur = lg.argmax(-1)
            gen.append(cur)
        gen = torch.stack(gen, 1)                                    # [n,new]
        e_logits = torch.stack(e_logits, 1)                          # [n,new,V]
        full = torch.cat([ids, gen[:, :-1]], 1)                      # positions L-1 .. L+new-2 predict gen
        L, N = ids.shape[1], full.shape[1]
        if kind == "huginn":
            s_pre = eng.blk._init_state_prefill(a.n, L)
            s_dec = eng.blk._init_state_tokens(list(range(L, N))).unsqueeze(0).expand(a.n, -1, -1)
            states = torch.cat([s_pre, s_dec], 1)
            outs = []
            for i in range(a.n):                                     # per row keeps memory small
                o = lm(input_ids=full[i:i + 1], input_states=states[i:i + 1],
                       num_steps=torch.tensor([D, 0]), use_cache=False)
                outs.append(o.logits[0].float())
            hf = torch.stack(outs)
        else:
            ouro_mask_shim(lm)
            lm.config.total_ut_steps = D
            lm.model.total_ut_steps = D
            o = lm.model(input_ids=full, use_cache=False)
            hs_list = o[1]
            assert len(hs_list) == D, len(hs_list)
            hf = lm.lm_head(hs_list[-1]).float()
        hf = hf[:, L - 1:L - 1 + a.new]                              # [n,new,V]
        top1 = (hf.argmax(-1) == gen)
        lp, lq = torch.log_softmax(hf, -1), torch.log_softmax(e_logits, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        mism = [(int(i), int(j)) for i, j in (~top1).nonzero().tolist()][:20]
        first = [next((j for j in range(a.new) if not top1[i, j]), None) for i in range(a.n)]
        res["cells"][f"D{D}"] = dict(
            positions=int(top1.numel()), greedy_agree=float(top1.float().mean()),
            n_mismatch=int((~top1).sum()), first_mismatch_per_prompt=first, mismatches=mism,
            max_abs_logit_diff=float((hf - e_logits).abs().amax()),
            median_abs_logit_diff=float((hf - e_logits).abs().amax(-1).median()),
            kl_mean=float(kl.mean()), kl_max=float(kl.max()))
        print(D, json.dumps(res["cells"][f"D{D}"])[:400], flush=True)
        del eng
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
