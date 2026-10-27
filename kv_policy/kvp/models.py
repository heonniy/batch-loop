"""Pinned model loading."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engine import HuginnBackbone, OuroBackbone, Runner
from .specs import MODELS, SELECTION_SEED


def load(model_key: str, device="cuda", dtype=torch.bfloat16):
    spec = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.repo, revision=spec.revision, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="sdpa" if spec.family == "ouro" else None,
    ).to(device).eval()
    return spec, tok, model


def make_runner(spec, tok, model) -> Runner:
    if spec.family == "ouro":
        bb = OuroBackbone(model)
        if bb.R != spec.native_depth:
            raise ValueError(f"total_ut_steps={bb.R} != pinned native depth {spec.native_depth}")
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    else:
        bb = HuginnBackbone(model, spec.native_depth, noise_seed=SELECTION_SEED)
        pad = model.config.pad_token_id
    run = Runner(bb, pad)
    run.model_key = spec.key
    return run


def eos_ids(spec, tok, model) -> set[int]:
    ids = set()
    gc = getattr(model, "generation_config", None)
    for src in (getattr(gc, "eos_token_id", None), tok.eos_token_id):
        if src is None:
            continue
        ids.update(src if isinstance(src, (list, tuple)) else [src])
    return ids
