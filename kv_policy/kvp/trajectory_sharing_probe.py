"""V10: model-agnostic per-generated-query recurrent state and exit traces.

Reuses the v7/v9 validated full-prefill / post-prefill masking / full-size
KV read-redirection engine; adds no new KV semantics. All observations are
made at the model-native observer used in v6:
  Huginn: recurrent core output BEFORE Coda, 0..R;
  Ouro: recurrent model.norm output, 0..R.
No hidden-state trajectory from an early-exited query is extrapolated.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .ffn_gate_causal import GateCausalRunner


class TrajectorySharingRunner(GateCausalRunner):
    def __init__(self, *args, **kwargs):
        self._tracing = False
        self._states = []
        self._exit_stats = []
        self._logprob_trace = []
        super().__init__(*args, **kwargs)

    def _step_dyn(self, col, tid, spec, forced):
        logits, depth = super()._step_dyn(col, tid, spec, forced)
        if self._tracing:
            # observe_initial() clears 'seen' for each generated query.
            if not all(self.cap.seen[:depth+1]):
                raise RuntimeError("The recurrent observer missed a state")
            H = self.cap.buf[:depth+1].detach().float().cpu().clone()
            self._states.append(H)
            self._exit_stats.append(self.exit_stats(H))
            self._logprob_trace.append(
                F.log_softmax(logits[0,0].float(),dim=-1).detach().cpu())
        return logits, depth

    @torch.no_grad()
    def trajectory_arm(self, *, sources: Sequence[int]=(), spec=None,
                       plane_map=None, dynamic: bool=False,
                       forced_depths: Sequence[int] | None=None):
        """Paired same-token teacher-forced decode; no free generation.

        'fixed' uses R native recurrences, 'live' genuinely applies the exit
        rule, and 'replay' forces FULL-LIVE depths for this *same* cache policy.
        """
        if self._tracing or self._ffn_hooks is not None or self._exit_collect is not None:
            raise RuntimeError("Nested trajectory/FFN exit probes are not supported")
        self._states,self._exit_stats,self._logprob_trace=[],[],[]
        self._tracing=True
        try:
            result=self.decode(
                sources, self.spec if spec is None else spec,
                plane_map=plane_map, dynamic=dynamic,
                forced_depths=forced_depths)
            states,stats,logps=list(self._states),list(self._exit_stats),list(self._logprob_trace)
        finally:
            self._tracing=False
            self._states,self._exit_stats,self._logprob_trace=[],[],[]
        n=len(self.ref_ids)
        if not (len(states)==len(stats)==len(logps)==len(result["depths"])==n):
            raise RuntimeError("Misaligned recurrence trajectory/output/depth receipts")
        if any(x.shape[0] != d+1 for x,d in zip(states,result["depths"])):
            raise RuntimeError("Recurrent trajectory not aligned to executed depth")
        return {**result,"states":[x.numpy() for x in states],
                "exit_stats":stats,"logprobs":logps}

    @torch.no_grad()
    def check_baseline_reference(self, arm, tol:float=1e-4):
        """No-mask FULL historical KV read must reproduce the independent ref."""
        ref=self.ref_lp
        if len(ref)!=len(arm["logprobs"]):
            raise RuntimeError("Reference length differs")
        diff=max(float((x.float()-y.float()).abs().max()) for x,y in zip(
            ref,arm["logprobs"]))
        if diff>tol:
            raise RuntimeError(f"Full historical KV reference parity failed: {diff}")
        return diff

    @staticmethod
    def full_vs_evicted_kl(clean,evicted,*,drop_last:bool=True):
        if len(clean["logprobs"])!=len(evicted["logprobs"]):
            raise ValueError("Paired decoded tokens differ")
        vals=[]
        for x,y in zip(clean["logprobs"],evicted["logprobs"]):
            x,y=x.float(),y.float()
            vals.append(float((x.exp()*(x-y)).sum()))
        kept=vals[:-1] if drop_last else vals
        return {"per_query":vals,
                "mean_scored_kl":sum(kept)/len(kept) if kept else None}
