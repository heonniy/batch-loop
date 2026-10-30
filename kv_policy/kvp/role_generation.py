"""Optional greedy answer accuracy with clean full prefill and future-only masks.

Gold answers are NEVER fed during generation. This is a separate evaluation
from teacher-forced future NLL. Token-major full-sized KV costs are substantial.
"""
from __future__ import annotations

import re
import string

import torch

from .role_data import RolePrompt
from .role_sensitivity import RoleProbeKV
from .depth_eviction_probe import DepthProbeSpec, _ouro_one, _huginn_one


@torch.no_grad()
def greedy_answer(bb, uid: str, spec: DepthProbeSpec, prompt: RolePrompt,
                  max_new_tokens: int, *, decode_mode: str = "fixed",
                  source=None, eos_token_ids=()) -> dict:
    """Generated token 1 is predicted from the untouched last prompt query.

    Post-prefill evicted Content is masked only when processing generated
    query positions P..; the baseline and evicted branches use identical
    prompt states but may generate different token/depth trajectories.
    """
    if max_new_tokens < 2:
        raise ValueError("Generation requires at least two new tokens")
    if decode_mode not in ("fixed","dynamic"):
        raise ValueError("Unsupported decode mode")
    if spec.family != bb.family or spec.max_depth != bb.R:
        raise ValueError("Backbone/exit mismatch")
    p = len(prompt.token_ids)
    params = next(bb.model.parameters())
    kv = RoleProbeKV(bb, p+max_new_tokens, params.device, params.dtype,
                     prompt=prompt,
                     mode="postprefill" if source is not None else "none",
                     source=source)
    aux = bb.make_aux([uid],[0],p+max_new_tokens) if bb.family=="huginn" else None
    eos = {int(x) for x in eos_token_ids}
    generated = []
    depths = []
    for col in range(p + max_new_tokens - 1):
        tid = prompt.token_ids[col] if col < p else generated[col-p]
        ids = torch.tensor([[int(tid)]],dtype=torch.long,device=params.device)
        pos = torch.tensor([[col]],dtype=torch.long,device=params.device)
        causal = torch.ones(1,1,1,col+1,dtype=torch.bool,device=params.device)
        forced = bb.R if col < p or decode_mode=="fixed" else None
        if bb.family=="ouro":
            logits, dep = _ouro_one(bb,ids,pos,kv,col,causal,forced,spec)
        else:
            logits, dep = _huginn_one(bb,ids,pos,kv,aux,col,causal,forced,spec)
        kv.finish(col,dep)
        if col >= p - 1:
            nxt = int(torch.argmax(logits[0,0].float()).item())
            generated.append(nxt)
            if nxt in eos:
                break
    return {"token_ids": generated,
            "decoded_tokens": len(generated),
            "decode_depths": kv.depths[p:],
            "actual_recurrence_calls": kv.actual_calls}


def _normalize_answer(text: str) -> str:
    text = text.casefold().strip()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _extract_answer(text: str) -> str:
    """Score only the short first line of an explicit Answer: response."""
    match = re.search(r"(?i)\banswer\s*:\s*([^\r\n]+)", text)
    if match:
        return match.group(1).strip()
    # Do not overclaim an exact match for unformatted long CoT completions.
    return text.splitlines()[0].strip() if text.strip() else ""


def score_free_answer(tokenizer, gold: str, rollout: dict) -> dict:
    output = tokenizer.decode(rollout["token_ids"],skip_special_tokens=True)
    extracted = _extract_answer(output)
    return {"exact_match": _normalize_answer(extracted)==_normalize_answer(gold),
            "generated_text": output,
            "extracted_answer": extracted,
            "generated_tokens": rollout["decoded_tokens"]}
