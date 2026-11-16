"""Frozen Huginn/Ouro depth-indexed core operator on the pinned WaveFront batch engine (GPU).

This module wraps WaveFront's *unmodified* batch blocks (`HuginnBlocksBatch`, `OuroBlocksBatch`)
to expose the solver.py interface

    step_many(states[R,B,H], depth_ids[R]) -> [R,B,H]

for ONE decode position shared by a cohort of B equal-length sequences.

KV semantics (FULL depth-indexed KV, both families, = author-native per-step cache):
  * each recurrent depth r (1-based) owns physical KV slots (depth slot r-1, every core layer);
  * a row (seq b, position t, depth r) writes K/V for (slot r-1, b, t) and then attends to
    positions 0..t of slot r-1 in sequence b. All rows of a call are written before any read.
  * Rows at the SAME position but DIFFERENT depths write disjoint slots, so one physical call can
    evaluate every depth of the current token (one Jacobi sweep) with correct per-depth context.
  * Candidate K/V for the current position is only ever visible to the current position: later
    positions are not decoded until `commit()`, which recomputes the exact serial trajectory and
    rewrites every slot (t, r) and the coda slot from exact states. Past-position KV is never
    written by a candidate row (fail-closed: `assert_position` checks every row's position).
  * Huginn coda attention writes the coda slot at position t; `logits()` may be called on candidate
    states for diagnostics, and the final exact coda call in `commit()` overwrites it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

WF_ROOT = os.environ.get("WF_ROOT", os.path.expanduser("~/dsd_work/wavefront-decoding"))
if WF_ROOT not in sys.path:
    sys.path.insert(0, WF_ROOT)

MODEL_IDS = {"huginn-3.5b": "tomg-group-umd/huginn-0125", "ouro-1.4b": "ByteDance/Ouro-1.4B",
             "ouro-2.6b": "ByteDance/Ouro-2.6B"}


def load_model(model: str, dtype: torch.dtype):
    """Returns (causal_lm, tokenizer, batch_cls, kind)."""
    if model.startswith("ouro"):
        from src.decoding.ouro_blocks_batch import OuroBlocksBatch
        from src.recursive_models.ouro_wrapper import OuroWrapper
        w = OuroWrapper.from_pretrained(MODEL_IDS[model], dtype=dtype, device="cuda")
        return w.model, w.tokenizer, OuroBlocksBatch, "ouro"
    if model == "huginn-3.5b":
        from src.decoding.huginn_blocks_batch import HuginnBlocksBatch
        from src.recursive_models.huginn_wrapper import HuginnWrapper
        w = HuginnWrapper.from_pretrained(dtype=dtype, device="cuda")
        return w.model, w.tokenizer, HuginnBlocksBatch, "huginn"
    raise ValueError(model)


@dataclass
class Counters:
    core_calls: int = 0
    core_rows: int = 0
    coda_calls: int = 0

    def add(self, rows: int) -> None:
        self.core_calls += 1
        self.core_rows += rows


class DepthEngine:
    def __init__(self, lm, batch_cls, kind: str, depth: int, batch: int, max_seq_len: int):
        from src.decoding.state import ActiveToken
        self._AT = ActiveToken
        self.kind, self.D, self.B = kind, depth, batch
        self.blk = batch_cls(lm, T_total=depth, batch_size=batch, max_seq_len=max_seq_len)
        self.lm = lm
        self.cache = None
        self.pos: Optional[int] = None
        self.e: Optional[torch.Tensor] = None       # Huginn prelude output (static injection) [B,H]
        self.tok_ids: Optional[list[int]] = None
        self.n = Counters()

    # -- prefill / prelude --------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        self.cache, last_logits = self.blk.prefill(ids)
        self.pos = ids.shape[1]
        return last_logits

    @torch.no_grad()
    def prelude(self, tok_ids: list[int]) -> torch.Tensor:
        """Embed the current tokens at self.pos; returns h0 [B,H]."""
        assert len(tok_ids) == self.B
        toks = [self._AT(token_id=int(t), position=self.pos, role="verify") for t in tok_ids]
        self.blk.prelude_union(list(range(self.B)), toks, self.cache)
        self.tok_ids = [int(t) for t in tok_ids]
        h0 = torch.stack([t.hidden for t in toks])
        self.e = None if toks[0].static_state is None else torch.stack([t.static_state for t in toks])
        return h0

    # -- core -----------------------------------------------------------------------------------
    @torch.no_grad()
    def step_many(self, states: torch.Tensor, depth_ids: torch.Tensor) -> torch.Tensor:
        """states [R,B,H] are the INPUTS to depths depth_ids[R] (1-based); returns outputs [R,B,H]."""
        R, B, H = states.shape
        assert B == self.B and len(depth_ids) == R
        ds = [int(d) for d in depth_ids.tolist()]
        assert all(1 <= d <= self.D for d in ds), ds
        toks, seq_ids = [], []
        for r in range(R):
            for b in range(B):
                t = self._AT(token_id=self.tok_ids[b], position=self.pos, role="verify")
                t.hidden = states[r, b]
                t.static_state = None if self.e is None else self.e[b]
                t.step = ds[r] - 1
                toks.append(t)
                seq_ids.append(b)
        self.blk.advance_union(seq_ids, toks, self.cache)
        self.n.add(R * B)
        return torch.stack([t.hidden for t in toks]).view(R, B, H)

    @torch.no_grad()
    def serial(self, h0: torch.Tensor, start: int = 0, h_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exact serial trajectory [D+1,B,H] (row 0 = h0). With start>0, continue from h_start at depth
        `start` (rows < start are copied from nothing: returned tensor rows 0..start-1 are zeros)."""
        out = [torch.zeros_like(h0)] * start + [h0 if start == 0 else h_start]
        h = out[-1]
        for r in range(start + 1, self.D + 1):
            h = self.step_many(h.unsqueeze(0), torch.tensor([r]))[0]
            out.append(h)
        return torch.stack(out)

    # -- coda -----------------------------------------------------------------------------------
    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """[B,H] final-depth-style hidden -> [B,V] float logits (Huginn coda writes coda slot at pos)."""
        self.n.coda_calls += 1
        if self.kind == "ouro":
            return self.lm.lm_head(h.unsqueeze(0)).float()[0]
        blk, model = self.blk, self.lm
        n = self.B
        positions = [self.pos] * n
        seq_ids = list(range(n))
        x = model.transformer.ln_f(h.unsqueeze(0))
        freqs = blk._freqs_at(positions)
        cu_q1 = torch.arange(0, n + 1, dtype=torch.int32, device=h.device)
        for i, block in enumerate(model.transformer.coda):
            slot = blk.slot_coda(i)
            hh = block.norm_1(x)
            q, k, v = blk._qkv(block.attn, hh, freqs)
            slots = [slot] * n
            self.cache.write_tokens(slots, seq_ids, positions, k[0], v[0])
            out = blk._union_attn(q, self.cache, slots, seq_ids, positions, cu_q1, None, None)
            x = blk._sandwich_tail(block, x, block.attn.proj(out.reshape(1, n, -1)))
        x = model.transformer.ln_f(x)
        return model.lm_head(x).float()[0]

    # -- commit ---------------------------------------------------------------------------------
    @torch.no_grad()
    def commit(self, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute the exact serial trajectory (rewrites all depth slots at pos), exact coda,
        advance position. Returns (trajectory [D+1,B,H], logits [B,V])."""
        traj = self.serial(h0)
        lg = self.logits(traj[-1])
        self.pos += 1
        return traj, lg


def make_cohort(tokenizer, texts: list[str], length: int, B: int) -> torch.Tensor:
    """B equal-length prompts: tokenize each text and keep its LAST `length` tokens (left
    truncation keeps the question/answer cue at the end). Texts shorter than `length` are skipped."""
    rows = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids[0]
        if ids.numel() >= length:
            rows.append(ids[-length:])
        if len(rows) == B:
            break
    if len(rows) < B:
        raise ValueError(f"only {len(rows)} prompts reach {length} tokens")
    return torch.stack(rows).cuda()
