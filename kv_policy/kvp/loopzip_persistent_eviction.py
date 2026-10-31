"""LoopZip v6 Stage A/B/C reference runner: normal trajectories and persistent one-source eviction.

One untouched full-depth prompt prefill per runner. Then:
- ``reference``: FULL fixed-depth continuation (greedy to EOS/horizon, or teacher-forced frozen IDs),
  storing per-token states h_(t,0..R) and fp32 next-token log-probs; snapshot of generated KV.
- ``accumulated(i)``: PRIMARY arm. Source i is unreadable for every generated query column, every
  recurrent loop, every recurrent layer AND every plain (prelude/coda) layer; the SAME token IDs are
  teacher-forced, so generated-token KV changes and persists (real no-refill semantics).
- ``clean_probes(i, ts)``: COUNTERFACTUAL diagnostic. For each t the FULL generated history is
  restored and only query token t (all its loops/layers) cannot read i; its KV is discarded.
Full-sized reference KV + read masks: no HBM claims. Every masked step asserts the mask was applied
in exactly n_rec*R recurrent + n_plain plain attention calls.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .depth_eviction_probe import DepthProbeSpec, _huginn_one, _ouro_one
from .loopzip_trajectory_recorder import StateCapture, logit_metrics, paired_metrics, trajectory_metrics
from .role_data import RolePrompt
from .role_sensitivity import RoleProbeKV

ELIGIBLE_ROLES = ("content", "instruction", "question")


def fixed_spec(bb) -> DepthProbeSpec:
    if bb.family == "ouro":
        return DepthProbeSpec("ouro", bb.R, 1.0, min_depth=2)
    return DepthProbeSpec("huginn", bb.R, 0.0, min_depth=2, cosine_dtype="float32")


def eligible_sources(prompt: RolePrompt) -> list[int]:
    """Frozen rule: every token aligned to Context/Instruction/Question; 'other' (format/special/
    boundary-crossing) tokens are excluded and counted separately."""
    return [i for i, r in enumerate(prompt.roles) if r in ELIGIBLE_ROLES]


class PersistentEvictionRunner:
    def __init__(self, bb, prompt: RolePrompt, uid: str, *, horizon: int = 128,
                 record_prefill_attention: bool = True):
        if horizon < 2:
            raise ValueError("Need a horizon of at least two tokens")
        self.bb, self.prompt, self.uid = bb, prompt, uid
        self.spec = fixed_spec(bb)
        self.R, self.P, self.horizon = bb.R, len(prompt.token_ids), horizon
        self.n_plain = int(getattr(bb, "n_plain", 0))
        self.n_rec = int(bb.n_rec)
        p = next(bb.model.parameters())
        self.device = p.device
        self.T = self.P + horizon + 1
        self.kv = RoleProbeKV(bb, self.T, self.device, p.dtype, prompt=prompt, mode="none", observe=False)
        self.aux = bb.make_aux([uid], [0], self.T) if bb.family == "huginn" else None
        self.cap = StateCapture(self.kv, self.R)
        self._mask = None            # (frozenset sources, first col, last col)
        self._applied = 0
        self._attn = record_prefill_attention
        self.recv_sum = torch.zeros(self.R, self.P, dtype=torch.float64, device=self.device)
        self.recv_max = torch.zeros(self.R, self.P, dtype=torch.float32, device=self.device)
        self.last_q = torch.zeros(self.R, self.P, dtype=torch.float64, device=self.device)
        orig = self.kv.attend

        def attend(layer, q, k, v, c0, c1, mask):
            if c0 < self.P and self._attn and layer[0] == "rec":
                kb, _ = self.kv._buffers(layer)
                keys = kb[:, :, :c1].clone()
                keys[:, :, c0:c1] = k
                w = torch.softmax(q.float() @ keys.float().transpose(-2, -1) / math.sqrt(q.shape[-1]),
                                  dim=-1)[0, :, 0, :].mean(0)
                r = layer[2]
                self.recv_sum[r, :c0] += w[:c0].double() / self.n_rec
                self.recv_max[r, :c0] = torch.maximum(self.recv_max[r, :c0], w[:c0])
                if c0 == self.P - 1:
                    self.last_q[r, :c0] += w[:c0].double() / self.n_rec
            m = self._mask
            if m is not None and m[1] <= c0 <= m[2]:
                if c0 in m[0]:
                    raise RuntimeError("Never mask the current token's own KV")
                mask = mask.clone()
                mask[..., sorted(m[0])] = False
                self._applied += 1
            return orig(layer, q, k, v, c0, c1, mask)

        self.kv.attend = attend
        self._prefill()

    # ------------------------------------------------------------------ core
    def _step(self, col: int, tid: int):
        ids = torch.tensor([[int(tid)]], dtype=torch.long, device=self.device)
        pos = torch.tensor([[col]], dtype=torch.long, device=self.device)
        mask = torch.ones(1, 1, 1, col + 1, dtype=torch.bool, device=self.device)
        self._applied = 0
        if self.bb.family == "ouro":
            logits, depth = _ouro_one(self.bb, ids, pos, self.kv, col, mask, self.R, self.spec)
        else:
            logits, depth = _huginn_one(self.bb, ids, pos, self.kv, self.aux, col, mask, self.R, self.spec)
        if depth != self.R:
            raise RuntimeError("Fixed full depth expected")
        self.kv.finish(col, depth)
        m = self._mask
        want = (self.n_rec * self.R + self.n_plain) if (m is not None and m[1] <= col <= m[2]) else 0
        if self._applied != want:
            raise RuntimeError(f"Mask applied {self._applied} times at col {col}, expected {want}")
        return logits, self.cap.take()

    def _kv_digest(self, lo: int, hi: int) -> float:
        bufs = [b[..., lo:hi, :] for b in (*self.kv.rk, *self.kv.rv)]
        bufs += [b[..., lo:hi, :] for b in (*self.kv.pk, *self.kv.pv)]
        return float(sum(b.float().abs().sum() for b in bufs))

    @torch.no_grad()
    def _prefill(self):
        rows = []
        for col, tid in enumerate(self.prompt.token_ids):
            logits, H = self._step(col, tid)
            rows.append({k: v.cpu() for k, v in trajectory_metrics(H).items()})
        self.prompt_metrics = rows
        self.last_prompt_lp = F.log_softmax(logits[0, 0].float(), -1)
        self.prefill_depths = list(self.kv.depths)
        self.prompt_kv_digest = self._kv_digest(0, self.P)
        self._attn = False

    def source_features(self) -> dict:
        """Clean-prefill, annotation-free per-source features (attention received from LATER prompt
        queries, averaged over recurrent layers/heads; final-prompt-token attention per loop)."""
        n_later = torch.tensor([self.P - 1 - i for i in range(self.P)], dtype=torch.float64,
                               device=self.device).clamp(min=1)
        mean = (self.recv_sum / n_later).cpu()
        return {"recv_mean": mean, "recv_max": self.recv_max.cpu(), "last_query": self.last_q.cpu()}

    def _reset_decode(self, n_done: int):
        self.kv.depths = list(self.prefill_depths) + [self.R] * n_done

    # ------------------------------------------------------------------ reference
    @torch.no_grad()
    def reference(self, eos_ids: set[int], frozen_ids: Sequence[int] | None = None) -> dict:
        self._mask = None
        self._reset_decode(0)
        y = [int(self.last_prompt_lp.argmax())] if frozen_ids is None else [int(frozen_ids[0])]
        Hs, lps, metrics = [], [], []
        stop = "horizon"
        argmax_agree = 0
        if frozen_ids is None and y[0] in eos_ids:
            raise RuntimeError("Reference continuation is empty (EOS first)")
        while True:
            t = len(y)
            logits, H = self._step(self.P + t - 1, y[-1])
            lp = F.log_softmax(logits[0, 0].float(), -1)
            Hs.append(H)
            lps.append(lp)
            metrics.append({k: v.cpu() for k, v in trajectory_metrics(H).items()})
            if t >= self.horizon if frozen_ids is None else t >= len(frozen_ids):
                break
            nxt = int(lp.argmax())
            if frozen_ids is None:
                if nxt in eos_ids:
                    stop = "eos"
                    break
                y.append(nxt)
            else:
                argmax_agree += int(nxt == int(frozen_ids[t]))
                y.append(int(frozen_ids[t]))
        if frozen_ids is not None:
            stop = "frozen"
        self.ref_ids = y
        self.ref_H = torch.stack(Hs)
        self.ref_lp = torch.stack(lps)
        self.ref_metrics = metrics
        n = len(y)
        self.snapshot = [(b, b[..., self.P:self.P + n, :].clone())
                         for b in (*self.kv.rk, *self.kv.rv, *self.kv.pk, *self.kv.pv)]
        return {"ids": y, "n": n, "stop": stop,
                "frozen_argmax_agreement": (argmax_agree / max(1, n - 1)) if frozen_ids is not None else None,
                "first_token_lp": float(self.last_prompt_lp[y[0]])}

    def _restore(self):
        for b, s in self.snapshot:
            b[..., self.P:self.P + s.shape[-2], :] = s

    # ------------------------------------------------------------------ interventions
    def _observe(self, t: int, logits, H) -> dict:
        y = self.ref_ids
        pm = paired_metrics(H, self.ref_H[t - 1])
        lm = logit_metrics(self.ref_lp[t - 1], logits[0, 0], y[t] if t < len(y) else None)
        return {"pm": pm, "lm": lm}

    @torch.no_grad()
    def accumulated(self, sources: Sequence[int]) -> dict:
        """Persistent mask on all generated query columns; same token IDs; altered KV persists."""
        src = frozenset(int(s) for s in sources)
        n = len(self.ref_ids)
        self._mask = (src, self.P, self.P + n) if src else None
        self._reset_decode(0)
        E = torch.empty(n, self.R + 1, device=self.device)
        U = torch.empty_like(E)
        D = torch.empty_like(E)
        tok = []
        try:
            for t in range(1, n + 1):
                logits, H = self._step(self.P + t - 1, self.ref_ids[t - 1])
                o = self._observe(t, logits, H)
                E[t - 1], U[t - 1], D[t - 1] = o["pm"]["E"], o["pm"]["U_ev"], o["pm"]["dircos"]
                tok.append(o["lm"])
        finally:
            self._mask = None
        return {"E": E.cpu(), "U_ev": U.cpu(), "dircos": D.cpu(), "token": tok}

    @torch.no_grad()
    def clean_probes(self, sources: Sequence[int], ts: Sequence[int]) -> dict:
        """Pristine FULL generated history, mask only the queried token t (all its loops/layers)."""
        src = frozenset(int(s) for s in sources)
        n = len(self.ref_ids)
        out = {}
        try:
            for t in ts:
                if not 1 <= t <= n:
                    continue
                self._restore()
                self._reset_decode(t - 1)
                col = self.P + t - 1
                self._mask = (src, col, col) if src else None
                logits, H = self._step(col, self.ref_ids[t - 1])
                o = self._observe(t, logits, H)
                out[t] = {"E": o["pm"]["E"].cpu(), "U_ev": o["pm"]["U_ev"].cpu(),
                          "dircos": o["pm"]["dircos"].cpu(), "token": o["lm"]}
        finally:
            self._mask = None
            self._restore()
        return out

    # ------------------------------------------------------------------ G0 validation
    @torch.no_grad()
    def validate(self, src_a: int, src_b: int, t_probe: int = 2, tolerance: float = 1e-4,
                 independent: bool = True) -> dict:
        from .role_sensitivity import role_forward
        rep = {}
        none = self.accumulated(())
        rep["nomask_max_E"] = float(none["E"].nan_to_num().abs().max())
        rep["nomask_max_kl"] = max(abs(x["kl"]) for x in none["token"])
        a1 = self.accumulated((src_a,))
        b1 = self.accumulated((src_b,))
        a2 = self.accumulated((src_a,))
        rep["source_switch_replay_max_abs"] = float((a1["E"] - a2["E"]).nan_to_num().abs().max())
        rep["source_switch_kl_max_abs"] = max(abs(x["kl"] - y["kl"]) for x, y in zip(a1["token"], a2["token"]))
        rep["a_vs_b_differs"] = bool((a1["E"] - b1["E"]).nan_to_num().abs().max() > 0)
        rep["masked_max_E"] = float(a1["E"].nan_to_num().max())
        # clean-history probe == sequential pristine replay with mask only at column P+t-1
        t = min(t_probe, len(self.ref_ids))
        cp = self.clean_probes((src_a,), (t,))[t]
        col = self.P + t - 1
        self._reset_decode(0)
        self._mask = None
        for s in range(1, t):
            self._step(self.P + s - 1, self.ref_ids[s - 1])
        self._mask = (frozenset((src_a,)), col, col)
        try:
            logits, H = self._step(col, self.ref_ids[t - 1])
        finally:
            self._mask = None
        o = self._observe(t, logits, H)
        rep["clean_probe_vs_recompute_max_abs"] = float((o["pm"]["E"] - cp["E"].to(self.device)).abs().max())
        rep["clean_probe_kl_vs_recompute"] = abs(o["lm"]["kl"] - cp["token"]["kl"])
        # t=1 under the clean probe and accumulated arm is the same computation
        rep["accum_vs_clean_t1_max_abs"] = float((a1["E"][0] - self.clean_probes((src_a,), (1,))[1]["E"]).abs().max())
        rep["prompt_kv_digest_drift"] = abs(self._kv_digest(0, self.P) - self.prompt_kv_digest)
        if independent:
            ind = role_forward(self.bb, self.uid, self.spec, self.prompt, self.ref_ids, decode_mode="fixed")
            ref_nll = [-float(self.last_prompt_lp[self.ref_ids[0]])] + [
                -float(self.ref_lp[i][self.ref_ids[i + 1]]) for i in range(len(self.ref_ids) - 1)]
            rep["independent_nll_max_abs"] = max(abs(a - b) for a, b in zip(ref_nll, ind["token_losses"]))
        fail = [k for k in ("nomask_max_E", "nomask_max_kl", "source_switch_replay_max_abs",
                            "source_switch_kl_max_abs", "clean_probe_vs_recompute_max_abs",
                            "clean_probe_kl_vs_recompute", "accum_vs_clean_t1_max_abs",
                            "prompt_kv_digest_drift", "independent_nll_max_abs")
                if k in rep and rep[k] > tolerance]
        if not rep["a_vs_b_differs"] or rep["masked_max_E"] <= 0:
            fail.append("mask_has_no_effect")
        rep["pass"] = not fail
        rep["failures"] = fail
        return rep
