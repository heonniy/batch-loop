"""v7 query-conditioned Context-KV sensitivity runner (QUERY_CONDITIONED_RECURRENT_KV_PLAN.md).

Extends the validated v6 ``PersistentEvictionRunner`` (one untouched full-depth prefill, full-sized
reference KV + read masks, no HBM claim) with:

* **Clean per-query attention census** at *watched* columns (prompt anchors q and every generated
  query t of the FULL reference): per recurrent loop r and source i the head-mean clean attention
  a_q(i,r), the closed-form *local attention-output perturbation* if i were unreadable at that loop,
  ``delta_i = w_i/(1-w_i) * ||o - v_i||`` (summed over heads/layers, normalised by ||o||), and the mean
  value norm. These are one-pass, label-free quantities; the generated-column versions are future
  ORACLE features (never available at prompt-completion time).
* **Query-conditioned intervention** ``query_probe(col, sources, loops, plain)``: recompute ONE prompt
  (or generated) column on its pristine clean history with the sources unreadable at the selected
  recurrent loops (and optionally the plain prelude/coda layers); the column's own KV write and the
  depth bookkeeping are rolled back afterwards, so the prefix is never contaminated. Returns the
  full h_q(r) error curve, update-direction cosine, next-token KL/top-1 change and model-native exit
  statistics (Ouro gate lambda_r; Huginn consecutive-state cosine in fp32 and model dtype).
* **Dynamic-exit arms** ``decode(sources, spec, forced_depths, dynamic)``: teacher-forced reference
  continuation with native early exit (Ouro Q-exit / Huginn cosine), recorded-trace replay or fixed
  depth, with the same post-prefill read mask; ``greedy`` for free-generation answer checks.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .depth_eviction_probe import DepthProbeSpec, _huginn_one, _ouro_one
from .loopzip_persistent_eviction import PersistentEvictionRunner
from .loopzip_trajectory_recorder import logit_metrics, paired_metrics, trajectory_metrics
from .role_data import RolePrompt

PREREG_LOOPS = {"ouro": (1, 2, 3, 4), "huginn": (1, 2, 4, 8, 16, 24, 32)}


def anchor_columns(prompt: RolePrompt) -> dict[str, int]:
    """Preregistered query anchors: last aligned Question token (else last Instruction token) and,
    as an explicitly separate ablation, the last prompt token (may be a format/chat marker)."""
    q = max(prompt.question_positions) if prompt.question_positions else max(prompt.instruction_positions)
    return {"question_last": int(q), "prompt_last": len(prompt.token_ids) - 1}


class QueryConditionedRunner(PersistentEvictionRunner):
    def __init__(self, bb, prompt: RolePrompt, uid: str, *, horizon: int = 64,
                 anchors: dict[str, int] | None = None, record_prefill_attention: bool = True):
        self.anchors = dict(anchors) if anchors is not None else anchor_columns(prompt)
        P = len(prompt.token_ids)
        if any(not 0 <= c < P for c in self.anchors.values()):
            raise ValueError("Anchor outside the prompt")
        self._watch: set[int] = set(self.anchors.values())
        self._recording = True
        self.local: dict[int, dict[str, torch.Tensor]] = {}
        self.clean_H: dict[int, torch.Tensor] = {}
        self.clean_lp: dict[int, torch.Tensor] = {}
        self.prompt_exit: list[dict] = []
        super().__init__(bb, prompt, uid, horizon=horizon, record_prefill_attention=record_prefill_attention)

    # ------------------------------------------------------------------ clean census hook
    def _install_watch(self):
        inner = self.kv.attend
        n_rec, n_plain = self.n_rec, self.n_plain

        def attend(layer, q, k, v, c0, c1, mask):
            if self._recording and c0 in self._watch:
                rec = self.local.get(c0)
                if rec is None:
                    z = lambda: torch.zeros(self.R, c0, dtype=torch.float64, device=self.device)  # noqa: E731
                    rec = self.local[c0] = {"attn": z(), "attn_max": z(), "delta": z(), "delta_max": z(),
                                            "vnorm": z(), "onorm": torch.zeros(self.R, dtype=torch.float64,
                                                                               device=self.device),
                                            "plain_attn": torch.zeros(max(1, n_plain), c0, dtype=torch.float64,
                                                                      device=self.device),
                                            "plain_delta": torch.zeros(max(1, n_plain), c0, dtype=torch.float64,
                                                                       device=self.device)}
                kb, vb = self.kv._buffers(layer)
                keys = kb[:, :, :c1].clone()
                keys[:, :, c0:c1] = k
                vals = vb[:, :, :c1].clone()
                vals[:, :, c0:c1] = v
                w = torch.softmax(q.float() @ keys.float().transpose(-2, -1) / math.sqrt(q.shape[-1]),
                                  dim=-1)[0, :, 0, :]                                   # (H, c1)
                V = vals[0].float()                                                     # (H, c1, D)
                o = torch.einsum("hj,hjd->hd", w, V)                                    # (H, D)
                diff = (o[:, None, :] - V).norm(dim=-1)                                 # (H, c1)
                delta = w / (1.0 - w).clamp_min(1e-6) * diff
                onorm = o.norm(dim=-1).sum().clamp_min(1e-9)
                if layer[0] == "rec":
                    r = layer[2]
                    rec["attn"][r] += w[:, :c0].mean(0).double() / n_rec
                    rec["attn_max"][r] = torch.maximum(rec["attn_max"][r], w[:, :c0].max(0).values.double())
                    rec["delta"][r] += (delta[:, :c0].sum(0) / onorm).double() / n_rec
                    rec["delta_max"][r] = torch.maximum(rec["delta_max"][r], (delta[:, :c0] / onorm).max(0).values.double())
                    rec["vnorm"][r] += V[:, :c0].norm(dim=-1).mean(0).double() / n_rec
                    rec["onorm"][r] += onorm.double() / n_rec
                else:
                    l = layer[1]
                    rec["plain_attn"][l] += w[:, :c0].mean(0).double()
                    rec["plain_delta"][l] += (delta[:, :c0].sum(0) / onorm).double()
            return inner(layer, q, k, v, c0, c1, mask)

        self.kv.attend = attend

    def exit_stats(self, H: torch.Tensor) -> dict[str, list[float]]:
        """Model-native exit statistics recomputed from the captured states h_0..h_R.
        Ouro: lambda_r = sigmoid(early_exit_gate(h_r)) and the official cumulative mass (p_R = rest).
        Huginn: 1 - cos(h_r, h_{r-1}) in fp32 and in model dtype (official evaluator arithmetic)."""
        if self.bb.family == "ouro":
            g = self.bb.model.model.early_exit_gate
            lam = torch.sigmoid(g(H[1:].to(g.weight.dtype)).float()).reshape(-1).tolist()
            cum, rem, cumul = [], 1.0, 0.0
            for r, l in enumerate(lam):
                if r < self.R - 1:
                    cumul += l * rem
                    rem *= (1.0 - l)
                else:
                    cumul = 1.0
                cum.append(cumul)
            return {"lambda": lam, "cumulative": cum}
        a, b = H[1:], H[:-1]
        d32 = (1.0 - (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).tolist()
        dt = next(self.bb.model.parameters()).dtype
        a16, b16 = H[1:].to(dt), H[:-1].to(dt)
        d16 = (1.0 - ((a16 * b16).sum(-1) / a16.norm(dim=-1) / b16.norm(dim=-1)).float()).tolist()
        return {"distance_fp32": d32, "distance_model_dtype": d16}

    @torch.no_grad()
    def _prefill(self):
        self._install_watch()
        rows = []
        for col, tid in enumerate(self.prompt.token_ids):
            logits, H = self._step(col, tid)
            rows.append({k: v.cpu() for k, v in trajectory_metrics(H).items()})
            self.prompt_exit.append(self.exit_stats(H))
            if col in self._watch:
                self.clean_H[col] = H.clone()
                self.clean_lp[col] = F.log_softmax(logits[0, 0].float(), -1)
        self.prompt_metrics = rows
        self.last_prompt_lp = F.log_softmax(logits[0, 0].float(), -1)
        self.prefill_depths = list(self.kv.depths)
        self.prompt_kv_digest = self._kv_digest(0, self.P)
        self._attn = False
        self._recording = False

    @torch.no_grad()
    def reference(self, eos_ids: set[int], frozen_ids: Sequence[int] | None = None) -> dict:
        """FULL fixed-depth reference with the clean census also recorded at every generated query."""
        self._watch |= set(range(self.P, self.T))
        self._recording = True
        try:
            out = super().reference(eos_ids, frozen_ids)
        finally:
            self._recording = False
            self._watch = set(self.anchors.values())
        self.ref_exit = [self.exit_stats(H) for H in self.ref_H]
        return out

    def census(self, col: int) -> dict[str, torch.Tensor]:
        """Clean attention / local-perturbation census of query column col (anchor or generated)."""
        rec = self.local[col]
        return {k: v.cpu() for k, v in rec.items()}

    # ------------------------------------------------------------------ query-conditioned probe
    def _column_token(self, col: int) -> int:
        return int(self.prompt.token_ids[col]) if col < self.P else int(self.ref_ids[col - self.P])

    def _next_token(self, col: int) -> int | None:
        if col + 1 < self.P:
            return int(self.prompt.token_ids[col + 1])
        t = col + 1 - self.P
        return int(self.ref_ids[t]) if t < len(self.ref_ids) else None

    def _clean(self, col: int):
        if col < self.P:
            return self.clean_H[col], self.clean_lp[col]
        t = col - self.P
        return self.ref_H[t], self.ref_lp[t]

    @torch.no_grad()
    def query_probe(self, col: int, sources: Sequence[int], loops: Sequence[int] | None = None,
                    plain: bool = True) -> dict:
        """Recompute ONE query column on pristine history with `sources` unreadable at `loops`
        (None = every recurrent loop) and, if `plain`, in the plain layers. Rolls back the column's
        KV write and depth bookkeeping. `sources` must precede `col`."""
        src = frozenset(int(s) for s in sources)
        if any(s >= col for s in src):
            raise ValueError("Sources must precede the probed query column")
        if col >= self.P and col - self.P >= len(self.ref_ids):
            raise ValueError("Column outside prompt + reference continuation")
        bufs = (*self.kv.rk, *self.kv.rv, *self.kv.pk, *self.kv.pv)
        snap = [b[..., col, :].clone() for b in bufs]
        depths = list(self.kv.depths)
        if col >= self.P:
            self._restore()
            self._reset_decode(col - self.P)
        else:
            self.kv.depths = list(self.prefill_depths[:col])
        self._mask = (src, col, col, None if loops is None else frozenset(int(r) for r in loops), plain) \
            if src else None
        try:
            logits, H = self._step(col, self._column_token(col))
        finally:
            self._mask = None
            for b, s in zip(bufs, snap):
                b[..., col, :] = s
            self.kv.depths = depths
            if col >= self.P:
                self._restore()
        Hc, lpc = self._clean(col)
        pm = paired_metrics(H, Hc)
        lm = logit_metrics(lpc, logits[0, 0], self._next_token(col))
        return {"E": pm["E"].cpu(), "U_ev": pm["U_ev"].cpu(), "dircos": pm["dircos"].cpu(), "token": lm,
                "exit": self.exit_stats(H), "delta_norm": (H - Hc).norm(dim=-1).cpu()}

    # ------------------------------------------------------------------ dynamic-exit arms
    def _step_dyn(self, col: int, tid: int, spec: DepthProbeSpec, forced: int | None):
        ids = torch.tensor([[int(tid)]], dtype=torch.long, device=self.device)
        pos = torch.tensor([[col]], dtype=torch.long, device=self.device)
        mask = torch.ones(1, 1, 1, col + 1, dtype=torch.bool, device=self.device)
        self._applied = 0
        if self.bb.family == "ouro":
            logits, depth = _ouro_one(self.bb, ids, pos, self.kv, col, mask, forced, spec)
        else:
            logits, depth = _huginn_one(self.bb, ids, pos, self.kv, self.aux, col, mask, forced, spec)
        self.kv.finish(col, depth)
        m = self._mask
        if m is not None and m[1] <= col <= m[2]:
            want = self.n_rec * depth + (self.n_plain if m[4] else 0)
            if self._applied != want:
                raise RuntimeError(f"Mask applied {self._applied} times at col {col}, expected {want}")
        return logits, depth

    @torch.no_grad()
    def decode(self, sources: Sequence[int], spec: DepthProbeSpec, *, forced_depths: Sequence[int] | None = None,
               dynamic: bool = False) -> dict:
        """Teacher-forced reference ids with post-prefill read mask on `sources` (all loops + plain).
        dynamic=True: genuinely re-evaluated native exit at each generated query (spec thresholds);
        forced_depths: recorded-trace replay; neither: fixed full depth. Generated KV is restored after."""
        src = frozenset(int(s) for s in sources)
        n = len(self.ref_ids)
        if forced_depths is not None and len(forced_depths) != n:
            raise ValueError("Replay trace must cover every generated query")
        if dynamic and forced_depths is not None:
            raise ValueError("Choose live exit or recorded replay, not both")
        self._mask = (src, self.P, self.P + n, None, True) if src else None
        self._reset_decode(0)
        depths, losses, kls = [], [], []
        try:
            for t in range(1, n + 1):
                col = self.P + t - 1
                forced = None if dynamic else (self.R if forced_depths is None else int(forced_depths[t - 1]))
                logits, depth = self._step_dyn(col, self.ref_ids[t - 1], spec, forced)
                depths.append(int(depth))
                lp = F.log_softmax(logits[0, 0].float(), -1)
                kls.append(float((self.ref_lp[t - 1].exp() * (self.ref_lp[t - 1] - lp)).sum()))
                if t < n:
                    losses.append(float(-lp[self.ref_ids[t]]))
        finally:
            self._mask = None
            self._restore()
            self._reset_decode(0)
        first = float(-self.last_prompt_lp[self.ref_ids[0]])
        return {"depths": depths, "token_losses": [first] + losses, "first_target_nll": first,
                "nll_future": sum(losses) / max(1, len(losses)), "kl_by_t": kls,
                "total_depth": sum(depths)}

    @torch.no_grad()
    def greedy(self, sources: Sequence[int], max_new: int, eos_ids: set[int]) -> list[int]:
        """Fixed-depth greedy free generation under the post-prefill read mask; generated KV restored."""
        src = frozenset(int(s) for s in sources)
        if max_new < 2 or self.P + max_new > self.T:
            raise ValueError("Generation bound exceeds allocated KV")
        self._mask = (src, self.P, self.P + max_new, None, True) if src else None
        self._reset_decode(0)
        out = [int(self.last_prompt_lp.argmax())]
        try:
            if out[0] not in eos_ids:
                for i in range(1, max_new):
                    logits, _ = self._step_dyn(self.P + i - 1, out[-1], self.spec, self.R)
                    nxt = int(logits[0, 0].float().argmax())
                    out.append(nxt)
                    if nxt in eos_ids:
                        break
        finally:
            self._mask = None
            self._restore()
            self._reset_decode(0)
        return out

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate_v7(self, src_a: int, src_b: int, live_spec: DepthProbeSpec | None = None,
                    tolerance: float = 1e-4) -> dict:
        rep = {}
        qa = self.anchors["question_last"]
        none = self.query_probe(qa, ())
        rep["probe_nomask_max_E"] = float(none["E"].nan_to_num().abs().max())
        rep["probe_nomask_kl"] = abs(none["token"]["kl"])
        before = self._kv_digest(0, self.P)
        a = self.query_probe(qa, (src_a,))
        b = self.query_probe(qa, (src_b,))
        a2 = self.query_probe(qa, (src_a,))
        rep["probe_replay_max_abs"] = float((a["E"] - a2["E"]).nan_to_num().abs().max())
        rep["probe_a_vs_b_differs"] = bool((a["E"] - b["E"]).nan_to_num().abs().max() > 0)
        rep["probe_masked_max_E"] = float(a["E"].nan_to_num().max())
        rep["probe_prompt_kv_drift"] = abs(self._kv_digest(0, self.P) - before)
        one = self.query_probe(qa, (src_a,), loops=(2,), plain=False)
        rep["one_loop_E_before_masked_loop_is_zero"] = float(one["E"][1]) == 0.0 and float(one["E"][2]) > 0.0
        # Re-running the full prefix step after probing must still reproduce the clean states.
        again = self.query_probe(qa, ())
        rep["probe_after_mask_nomask_max_E"] = float(again["E"].nan_to_num().abs().max())
        # Fixed decode reproduces the reference; forced-R replay == fixed; accumulated == fixed-evicted.
        fx = self.decode((), self.spec)
        ref_nll = [-float(self.last_prompt_lp[self.ref_ids[0]])] + [
            -float(self.ref_lp[i][self.ref_ids[i + 1]]) for i in range(len(self.ref_ids) - 1)]
        rep["fixed_nomask_max_abs_nll"] = max(abs(x - y) for x, y in zip(fx["token_losses"], ref_nll))
        rp = self.decode((), self.spec, forced_depths=[self.R] * len(self.ref_ids))
        rep["forced_R_vs_fixed_max_abs_nll"] = max(abs(x - y) for x, y in zip(fx["token_losses"], rp["token_losses"]))
        ev = self.decode((src_a,), self.spec)
        acc = self.accumulated((src_a,))
        self._restore()
        shifts = [x["nll_shift"] for x in acc["token"] if x["nll_shift"] is not None]
        rep["accumulated_vs_fixed_decode_max_abs"] = max(
            abs((ev["token_losses"][i + 1] - ref_nll[i + 1]) - shifts[i]) for i in range(len(shifts)))
        rep["first_target_invariant"] = abs(ev["first_target_nll"] - ref_nll[0])
        if live_spec is not None:
            l1 = self.decode((), live_spec, dynamic=True)
            l2 = self.decode((), live_spec, dynamic=True)
            rep["live_repeat_identical"] = l1["depths"] == l2["depths"] and max(
                abs(x - y) for x, y in zip(l1["token_losses"], l2["token_losses"])) == 0
            rr = self.decode((), live_spec, forced_depths=l1["depths"])
            rep["live_vs_replay_max_abs_nll"] = max(abs(x - y) for x, y in zip(l1["token_losses"], rr["token_losses"]))
            rep["live_depths"] = l1["depths"]
        rep["prompt_kv_digest_drift"] = abs(self._kv_digest(0, self.P) - self.prompt_kv_digest)
        fail = [k for k in ("probe_nomask_max_E", "probe_nomask_kl", "probe_replay_max_abs", "probe_prompt_kv_drift",
                            "probe_after_mask_nomask_max_E", "fixed_nomask_max_abs_nll",
                            "forced_R_vs_fixed_max_abs_nll", "accumulated_vs_fixed_decode_max_abs",
                            "first_target_invariant", "live_vs_replay_max_abs_nll", "prompt_kv_digest_drift")
                if k in rep and rep[k] > tolerance]
        if not rep["probe_a_vs_b_differs"] or rep["probe_masked_max_E"] <= 0:
            fail.append("probe_mask_has_no_effect")
        if not rep["one_loop_E_before_masked_loop_is_zero"]:
            fail.append("one_loop_mask_leaks_before_its_loop")
        if live_spec is not None and not rep["live_repeat_identical"]:
            fail.append("live_exit_not_repeatable")
        rep["pass"] = not fail
        rep["failures"] = fail
        return rep
