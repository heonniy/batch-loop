"""Fast Phase-1/2 evaluator: clean FULL prefill once; reuse it for all policies.

Important: this is a FULL-sized reference KV bank. We DO NOT physically free
evicted keys. Only future generated-token Attention reads are masked. The
prefix is computed *once*, and every policy branch replays generated slots
into the same reusable scratch region. No KV writes occur to the prefix after
it has been finalized; stale future scratch is ignored by the causal mask.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .role_data import RolePrompt
from .role_sensitivity import RoleProbeKV
from .depth_eviction_probe import DepthProbeSpec, _ouro_one, _huginn_one
from .role_generation import score_free_answer


class CachedFullPrefill:
    """One request, one checkpoint, one clean prompt; many policy/Oracle decodes.

    Initially FIXED-depth decode only to disentangle cache policy from exit
    decisions. The existing v3 recurrent oracle handles later dynamic probes.
    """

    def __init__(self, bb, spec: DepthProbeSpec, prompt: RolePrompt,
                 target_ids: Sequence[int], uid: str, *,
                 max_generation_tokens: int = 32):
        if not uid or spec.family != bb.family or spec.max_depth != bb.R:
            raise ValueError("Request UID / backbone / exit criterion mismatch")
        if max_generation_tokens < 2 or len(target_ids) < 2:
            raise ValueError("Need at least two gold and generation tokens")
        if not prompt.content_positions:
            raise ValueError("Cannot compress a request with no Context tokens")
        self.bb, self.spec, self.prompt, self.uid = bb, spec, prompt, uid
        self.target_ids = tuple(int(x) for x in target_ids)
        self.P = len(prompt.token_ids)
        self.max_generation_tokens = max_generation_tokens
        self.params = next(bb.model.parameters())
        self.device = self.params.device
        # One additional slot is harmless and avoids an off-by-one allocation.
        self.T = self.P + max(len(self.target_ids), max_generation_tokens) + 1
        self.kv = RoleProbeKV(bb, self.T, self.device, self.params.dtype,
                              prompt=prompt,mode="none",observe=True)
        self.aux = (bb.make_aux([uid],[0],self.T)
                    if bb.family == "huginn" else None)
        self._last_prompt_logits = None
        self._prefix_calls = None
        self._prefix_depths = None
        self._prefill()

    def _step(self, col: int, tid: int):
        ids=torch.tensor([[int(tid)]],dtype=torch.long,device=self.device)
        pos=torch.tensor([[col]],dtype=torch.long,device=self.device)
        mask=torch.ones(1,1,1,col+1,dtype=torch.bool,device=self.device)
        if self.bb.family == "ouro":
            logits, depth = _ouro_one(self.bb,ids,pos,self.kv,col,mask,
                                      self.bb.R,self.spec)
        else:
            logits, depth = _huginn_one(self.bb,ids,pos,self.kv,self.aux,col,
                                        mask,self.bb.R,self.spec)
        self.kv.finish(col,depth)
        return logits

    @torch.no_grad()
    def _prefill(self):
        for col,tid in enumerate(self.prompt.token_ids):
            logits=self._step(col,tid)
        self._last_prompt_logits=logits.detach()
        self._prefix_calls=self.kv.actual_calls
        self._prefix_depths=tuple(self.kv.depths)
        if len(self._prefix_depths)!=self.P or any(d!=self.bb.R
                                                    for d in self._prefix_depths):
            raise RuntimeError("Expected full-depth prompt recurrence")
        self.clean_attention = self.kv.observed_attention()
        if set(int(k) for k in self.clean_attention["content_attention_by_source"]
               ) != set(self.prompt.content_positions):
            raise RuntimeError("Clean Context Attention missing positions")

    def _reset(self, evicted_positions: Sequence[int] = ()):
        sources=tuple(sorted(int(p) for p in evicted_positions))
        if len(sources)!=len(set(sources)):
            raise ValueError("Eviction source list contains duplicates")
        if not set(sources)<=set(self.prompt.content_positions):
            raise ValueError("Only Context positions may be evicted")
        self.kv.mode = "postprefill" if sources else "none"
        self.kv.source = sources
        self.kv.observe = False
        # Prompt K/V and normal end-of-prompt logits remain untouched. For
        # every branch the generated slots are overwritten before any read.
        self.kv.depths = list(self._prefix_depths)
        self.kv.actual_calls = self._prefix_calls

    @torch.no_grad()
    def score_gold(self, evicted_positions: Sequence[int] = ()) -> dict:
        self._reset(evicted_positions)
        first=float(-F.log_softmax(
            self._last_prompt_logits[0,0].float(),dim=-1
        )[self.target_ids[0]].item())
        losses=[first]
        for i in range(1,len(self.target_ids)):
            col=self.P+i-1
            logits=self._step(col,self.target_ids[i-1])
            lp=F.log_softmax(logits[0,0].float(),dim=-1)
            losses.append(float(-lp[self.target_ids[i]].item()))
        if len(losses)!=len(self.target_ids) or not all(
                math.isfinite(v) for v in losses):
            raise RuntimeError("Invalid frozen gold NLL observations")
        return {
            "nll_all":sum(losses)/len(losses),
            "nll_future":sum(losses[1:])/len(losses[1:]),
            "first_target_nll":first,
            "token_losses":losses,
            "n_gold_tokens":len(losses),
            "gold_recurrence_calls":self.kv.actual_calls-self._prefix_calls,
            "n_masked_context":len(tuple(evicted_positions))}

    @torch.no_grad()
    def greedy(self, tokenizer, gold_answer: str,
               evicted_positions: Sequence[int] = (),
               max_generation_tokens: int | None = None) -> dict:
        self._reset(evicted_positions)
        limit=(self.max_generation_tokens if max_generation_tokens is None
               else int(max_generation_tokens))
        if not 2 <= limit <= self.max_generation_tokens:
            raise ValueError("Invalid greedy generation bound")
        first=int(self._last_prompt_logits[0,0].float().argmax().item())
        generated=[first]
        eos_id=getattr(tokenizer,"eos_token_id",None)
        eos=set(eos_id if isinstance(eos_id,(list,tuple)) else
                (() if eos_id is None else (eos_id,)))
        if first not in eos:
            for i in range(1,limit):
                col=self.P+i-1
                logits=self._step(col,generated[i-1])
                nxt=int(logits[0,0].float().argmax().item())
                generated.append(nxt)
                if nxt in eos:
                    break
        answer=score_free_answer(tokenizer,gold_answer,
                                 {"token_ids":generated,
                                  "decoded_tokens":len(generated)})
        return {**answer,"token_ids":generated,
                "decode_recurrence_calls":self.kv.actual_calls-self._prefix_calls,
                "n_masked_context":len(tuple(evicted_positions))}

    @torch.no_grad()
    def assert_clean_reference_parity(self, tolerance: float = 1e-4):
        """Reusable scratch vs independent v3 reference, BEFORE any measurement."""
        from .role_sensitivity import role_forward
        cached=self.score_gold(())
        independent=role_forward(self.bb,self.uid,self.spec,self.prompt,
                                 self.target_ids,decode_mode="fixed")
        diff=max(abs(a-b) for a,b in zip(cached["token_losses"],
                                        independent["token_losses"]))
        if diff>tolerance:
            raise RuntimeError(f"Snapshot/reference scoring mismatch: {diff}")
        # A second cached pass checks stale generated-token scratch and reset.
        again=self.score_gold(())
        delta=max(abs(a-b) for a,b in zip(cached["token_losses"],
                                         again["token_losses"]))
        if delta>tolerance:
            raise RuntimeError(f"Reusable cache replay mismatch: {delta}")
        return {"reference_max_abs_nll_diff":diff,
                "repeated_cache_max_abs_nll_diff":delta,
                "first_token_loss":cached["first_target_nll"]}
