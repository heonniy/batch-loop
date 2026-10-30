"""B1 physically-skipped recurrent inference and token-KV eviction oracle.

FULL-dynamic uses available historical KV_i^(min(query_depth, source_depth)).
After processing a source token only, uncomputed higher planes are filled with
its last *computed* KV. A source-eviction intervention masks that position from
all physical recurrent and ordinary attention layers, for all future queries.
The fixed-depth reference full cache is NOT compacted or physically evicted.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .engine import PolicyKV
from .native_exit import ExitRule, recorded_trace, replay_depth


@dataclass(frozen=True)
class DepthProbeSpec:
    family: str
    max_depth: int
    threshold: float
    min_depth: int = 2

    def __post_init__(self):
        if self.family not in ("ouro", "huginn") or self.max_depth < 2:
            raise ValueError("Invalid family/depth")
        if self.min_depth < 1 or self.min_depth > self.max_depth:
            raise ValueError("Invalid minimum recurrence")
        if self.family == "ouro" and not (0 < self.threshold <= 1):
            raise ValueError("Ouro Q-exit q must be in (0,1]")
        if self.family == "huginn" and self.threshold < 0:
            raise ValueError("Huginn cosine threshold must be nonnegative")

    def rule(self):
        return ExitRule(self.family, "q-exit" if self.family == "ouro" else "cosine",
                        self.threshold, self.min_depth)


class DepthOracleKV(PolicyKV):
    """R-plane quality store; causal single-token history eviction."""

    def __init__(self, bb, n_tokens, device, dtype, evicted=None, evaluation_start=0,
                 collect_attention=False):
        super().__init__("full", bb.R, bb.n_rec, bb.n_plain, 1, bb.H,
                         n_tokens, bb.D, dtype, device, region_start=0)
        self.evicted = evicted
        self.evaluation_start = evaluation_start
        self.collect_attention = collect_attention
        self.attention_pre = torch.zeros(n_tokens, dtype=torch.float64)
        self.attention_future = torch.zeros(n_tokens, dtype=torch.float64)
        self.depths = []
        self.actual_calls = 0

    def attend(self, layer, q, k, v, c0, c1, mask):
        if c1 != c0 + 1:
            raise ValueError("B1 single-token processing only")
        if self.evicted is not None and c0 >= self.evaluation_start:
            mask = mask.clone()
            mask[..., self.evicted] = False
        if self.collect_attention:
            kb, _ = self._buffers(layer)
            keys = kb[:, :, :c1].clone()
            keys[:, :, c0:c1] = k
            scores = (q.float() @ keys.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
            weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
            weights = weights[0, :, 0, :].mean(dim=0).detach().cpu().double()
            weights[c0] = 0.0  # past-token reception, not self attention
            target = self.attention_pre if c0 < self.evaluation_start else self.attention_future
            target[:c1] += weights
        return super().attend(layer, q, k, v, c0, c1, mask)

    def finish(self, col, depth):
        if not 1 <= depth <= self.R:
            raise ValueError("Exit depth out of bounds")
        for buf in (*self.rk, *self.rv):
            if depth < self.R:
                buf[depth:, :, :, col, :] = buf[depth-1, :, :, col, :].unsqueeze(0)
        self.depths.append(depth)


def choose_sources(n_prompt, n_candidates, seed, sinks=4):
    if n_prompt <= sinks or n_candidates < 1:
        raise ValueError("Need some nonsink prompt tokens")
    pool = list(range(sinks, n_prompt))
    return (pool if n_candidates >= len(pool) else
            sorted(random.Random(seed).sample(pool, n_candidates)))


def _ouro_one(bb, ids, pos, kv, col, mask, forced_depth, spec):
    model = bb.model
    m = model.model
    h = m.embed_tokens(ids)
    cos, sin = m.rotary_emb(h, pos)
    remaining, cumul = 1.0, 0.0
    for r in range(bb.R):
        for l, layer in enumerate(m.layers):
            res = h
            x = layer.input_layernorm(h)
            att = layer.self_attn
            q = att.q_proj(x).view(1, 1, -1, bb.D).transpose(1, 2)
            k = att.k_proj(x).view(1, 1, -1, bb.D).transpose(1, 2)
            v = att.v_proj(x).view(1, 1, -1, bb.D).transpose(1, 2)
            q, k = bb.apply_rope(q, k, cos, sin)
            o = kv.attend(("rec", l, r), q, k, v, col, col+1, mask)
            o = att.o_proj(o.transpose(1, 2).reshape(1, 1, -1))
            h = res + layer.input_layernorm_2(o)
            h = h + layer.post_attention_layernorm_2(
                layer.mlp(layer.post_attention_layernorm(h)))
        h = m.norm(h)
        # Official gate: lambda_r=sigmoid(early_exit_gate(h_r));
        # p_r=lambda_r * survival mass, except p_R=remaining mass.
        lam = torch.sigmoid(m.early_exit_gate(h)).float().item()
        if r < bb.R-1:
            cumul += lam * remaining
            remaining *= (1.0-lam)
        else:
            cumul = 1.0
        kv.actual_calls += 1
        depth = r+1
        if forced_depth is not None:
            if depth == forced_depth:
                break
        elif depth >= spec.min_depth and (
                depth == bb.R or (spec.threshold < 1 and cumul >= spec.threshold)):
            break
    return model.lm_head(h).float(), depth


def _huginn_one(bb, ids, pos, kv, aux, col, mask, forced_depth, spec):
    tr = bb.model.transformer
    freqs = bb.model.freqs_cis[0][pos]
    e = tr.wte(ids) * bb.emb_scale
    for l, block in enumerate(tr.prelude):
        e = bb._block(block, e, freqs, kv, ("plain", l, 0), col, col+1, mask)
    x = aux[:, col:col+1]
    prev = x
    for r in range(bb.R):
        x = tr.adapter(torch.cat([x, e], dim=-1))
        for l, block in enumerate(tr.core_block):
            x = bb._block(block, x, freqs, kv, ("rec", l, r), col, col+1, mask)
        # Official cosine evaluator arithmetic (including model dtype).
        cosine = ((x * prev).sum(dim=-1) / x.norm(dim=-1) /
                  prev.norm(dim=-1)).mean(dim=1).item()
        distance = 1.0 - cosine
        prev = x
        kv.actual_calls += 1
        depth = r+1
        if forced_depth is not None:
            if depth == forced_depth:
                break
        elif depth >= spec.min_depth and (
                distance < spec.threshold or depth == bb.R):
            break
    x = tr.ln_f(x)
    for l, block in enumerate(tr.coda):
        x = bb._block(block, x, freqs, kv, ("plain", bb.n_pre+l, 0),
                      col, col+1, mask)
    x = tr.ln_f(x)
    return bb.model.lm_head(x).float(), depth


@torch.no_grad()
def run_prefix(bb, ids: Sequence[int], uid: str, spec: DepthProbeSpec, *,
               prompt_len: int, forced_depths=None, evicted=None,
               collect_attention=False):
    """Teacher-forced B1 score of gold suffix. No future gold used by exit."""
    if not 2 <= prompt_len < len(ids):
        raise ValueError("Need prompt and nonempty gold suffix")
    if spec.family != bb.family or spec.max_depth != bb.R:
        raise ValueError("Backbone vs criterion mismatch")
    if forced_depths is not None and len(forced_depths) != len(ids)-1:
        raise ValueError("Replay trace length mismatch")
    if evicted is not None and not 0 <= evicted < prompt_len:
        raise ValueError("Source must be in historical prompt")
    p = next(bb.model.parameters())
    device, dtype = p.device, p.dtype
    kv = DepthOracleKV(bb, len(ids)-1, device, dtype, evicted,
                       prompt_len, collect_attention)
    aux = bb.make_aux([uid], [0], len(ids)-1) if bb.family == "huginn" else None
    loss = []
    for col, tid in enumerate(ids[:-1]):
        tok = torch.tensor([[int(tid)]], device=device)
        pos = torch.tensor([[col]], dtype=torch.long, device=device)
        mask = torch.ones((1, 1, 1, col+1), dtype=torch.bool, device=device)
        forced = None if forced_depths is None else int(forced_depths[col])
        if bb.family == "ouro":
            logits, depth = _ouro_one(bb, tok, pos, kv, col, mask, forced, spec)
        else:
            logits, depth = _huginn_one(bb, tok, pos, kv, aux, col, mask, forced, spec)
        kv.finish(col, depth)
        if col >= prompt_len-1:
            logp = F.log_softmax(logits[0, 0].float(), dim=-1)
            loss.append(float(-logp[int(ids[col+1])].item()))
    return dict(depths=kv.depths, nll=sum(loss)/len(loss),
                loss_by_future_token=loss,
                attention_pre=kv.attention_pre.tolist(),
                attention_future=kv.attention_future.tolist(),
                actual_recurrence_calls=kv.actual_calls,
                reference_full_allocated=True)


@torch.no_grad()
def single_source_oracle(bb, ids: Sequence[int], uid: str, spec: DepthProbeSpec,
                         model_revision: str, *, prompt_len: int,
                         candidates, replay_tolerance=1e-4):
    """Record FULL dynamic trace once, then reuse for all evictions."""
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate source IDs")
    if any(not 0 <= i < prompt_len for i in candidates):
        raise ValueError("Source must belong to prompt")
    base = run_prefix(bb, ids, uid, spec, prompt_len=prompt_len,
                      collect_attention=True)
    trace = recorded_trace(model_revision, spec.rule(), uid, ids[:-1],
                           base["depths"], spec.max_depth, policy="full-dynamic")
    frozen = [replay_depth(trace, i, model_revision, uid)
              for i in range(len(ids)-1)]
    control = run_prefix(bb, ids, uid, spec, prompt_len=prompt_len,
                         forced_depths=frozen)
    diff = max(abs(a-b) for a, b in zip(base["loss_by_future_token"],
                                      control["loss_by_future_token"]))
    if diff > replay_tolerance:
        raise RuntimeError("Record/replay parity failed: max delta NLL "+str(diff))
    rows = []
    for source in candidates:
        result = run_prefix(bb, ids, uid, spec, prompt_len=prompt_len,
                            forced_depths=frozen, evicted=source)
        rows.append(dict(uid=uid, source_position=source,
                         source_token_id=int(ids[source]),
                         exit_depth=int(frozen[source]),
                         attention_pre=float(base["attention_pre"][source]),
                         attention_pre_per_opportunity=(
                            float(base["attention_pre"][source]) /
                            max(1, prompt_len-source-1)),
                         attention_future_ORACLE=float(base["attention_future"][source]),
                         delta_nll=result["nll"]-base["nll"],
                         baseline_nll=base["nll"], evicted_nll=result["nll"],
                         age_at_eval=prompt_len-source,
                         replay_depths_identical=(result["depths"]==frozen)))
    return dict(trace=trace,
                reference=dict(nll=base["nll"],
                               replay_max_abs_loss_delta=diff,
                               actual_recurrence_calls=base["actual_recurrence_calls"],
                               reference_full_allocated=True),
                candidate_rows=rows)
