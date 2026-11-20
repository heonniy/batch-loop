"""Variable-length, active-subset depth engine for free-running batched decoding (GPU).

Extends the PoC `parallel_depth_poc.wf_engine.DepthEngine` contract to what a real no-refill
offline batch needs:
  * each request keeps its OWN prompt length / position (no left truncation, no padding tokens);
  * prefill is per request (author-native dense causal prefill of the unmodified WaveFront
    blocks at B=1), copied into row b of one shared FULL depth-indexed KV bank;
  * every decode call runs only the ACTIVE requests (finished rows leave the union; no refill);
  * `step_many(states[R,n,H], depths[R])` evaluates R depths x n active rows in ONE
    `advance_union` call (all writes before reads -> one Jacobi sweep with correct per-depth KV).
KV semantics are unchanged from the PoC: depth r of (request b, position p) owns slot r-1;
candidate rows only ever write the CURRENT position of their own request.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from parallel_depth_poc import wf_engine as W


@dataclass
class Counters:
    core_calls: int = 0
    core_rows: int = 0
    coda_calls: int = 0
    coda_rows: int = 0
    prelude_calls: int = 0


class VarEngine:
    def __init__(self, lm, batch_cls, kind: str, depth: int, batch: int, max_seq_len: int):
        from src.decoding.state import ActiveToken
        self._AT = ActiveToken
        self.lm, self.kind, self.D, self.B, self.S = lm, kind, depth, batch, max_seq_len
        self.blk = batch_cls(lm, T_total=depth, batch_size=batch, max_seq_len=max_seq_len)
        self._blk1 = batch_cls(lm, T_total=depth, batch_size=1, max_seq_len=max_seq_len)
        self.cache = self.blk.create_cache()
        self.pos = [0] * batch
        self.prompt_len = [0] * batch
        self.rows: list[int] = []
        self.tok_ids: list[int] = []
        self.e = None
        self.n = Counters()

    # ------------------------------------------------------------------ prefill
    @torch.no_grad()
    def prefill_rows(self, prompts: list[torch.Tensor]) -> torch.Tensor:
        """prompts: B 1-D LongTensors (any lengths). Returns next-token logits [B,V] (float)."""
        assert len(prompts) == self.B
        out = []
        for b, ids in enumerate(prompts):
            L = int(ids.numel())
            if L + 1 >= self.S:
                raise ValueError(f"prompt {b} length {L} exceeds max_seq_len {self.S}")
            c1, lg = self._blk1.prefill(ids.view(1, -1).cuda())
            self.cache.K[:, b, :L].copy_(c1.K[:, 0, :L])
            self.cache.V[:, b, :L].copy_(c1.V[:, 0, :L])
            for s in range(self.cache.n_slots):
                self.cache.seqlen[s][b] = max(c1.seqlen[s][0], 0)
            self.pos[b] = self.prompt_len[b] = L
            out.append(lg[0])
            del c1
        return torch.stack(out)

    # ------------------------------------------------------------------ prelude
    @torch.no_grad()
    def prelude(self, rows: list[int], tok_ids: list[int]) -> torch.Tensor:
        """Embed the current token of each active row at its own position; returns h0 [n,H]."""
        assert len(rows) == len(tok_ids) and rows
        toks = [self._AT(token_id=int(t), position=self.pos[b], role="verify") for b, t in zip(rows, tok_ids)]
        self.blk.prelude_union(list(rows), toks, self.cache)
        self.rows, self.tok_ids = list(rows), [int(t) for t in tok_ids]
        self.n.prelude_calls += 1
        h0 = torch.stack([t.hidden for t in toks])
        self.e = None if toks[0].static_state is None else torch.stack([t.static_state for t in toks])
        return h0

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def step_many(self, states: torch.Tensor, depth_ids) -> torch.Tensor:
        """states [R,n,H] = INPUTS to depths depth_ids[R] (1-based) for the active rows."""
        R, n, H = states.shape
        ds = [int(d) for d in (depth_ids.tolist() if torch.is_tensor(depth_ids) else depth_ids)]
        assert n == len(self.rows) and len(ds) == R and all(1 <= d <= self.D for d in ds), ds
        toks, seq_ids = [], []
        for r in range(R):
            for j, b in enumerate(self.rows):
                t = self._AT(token_id=self.tok_ids[j], position=self.pos[b], role="verify")
                t.hidden = states[r, j]
                t.static_state = None if self.e is None else self.e[j]
                t.step = ds[r] - 1
                toks.append(t)
                seq_ids.append(b)
        self.blk.advance_union(seq_ids, toks, self.cache)
        self.n.core_calls += 1
        self.n.core_rows += R * n
        return torch.stack([t.hidden for t in toks]).view(R, n, H)

    @torch.no_grad()
    def serial(self, h0: torch.Tensor, start: int = 0, h_start=None, upto: int | None = None) -> torch.Tensor:
        """Exact serial trajectory rows start..upto (default D); returns [D+1,n,H] (rows < start zero)."""
        upto = self.D if upto is None else upto
        h = h0 if start == 0 else h_start
        out = [torch.zeros_like(h0)] * start + [h]
        for r in range(start + 1, upto + 1):
            h = self.step_many(h.unsqueeze(0), [r])[0]
            out.append(h)
        return torch.stack(out)

    # ------------------------------------------------------------------ coda / head
    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """[n,H] final-depth hidden of the active rows -> [n,V] float logits. Huginn coda writes the
        coda slot at the current position (overwritten by any later exact coda call)."""
        n = h.shape[0]
        self.n.coda_calls += 1
        self.n.coda_rows += n
        if self.kind == "ouro":
            return self.lm.lm_head(h.unsqueeze(0)).float()[0]
        blk, model = self.blk, self.lm
        positions = [self.pos[b] for b in self.rows]
        seq_ids = list(self.rows)
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

    def advance(self) -> None:
        for b in self.rows:
            self.pos[b] += 1

    # ------------------------------------------------------------------ KV audit helpers
    def core_slots(self) -> list[list[int]]:
        """slot ids per recurrent depth r=1..D (list index r-1)."""
        if self.kind == "huginn":
            n_core = self.blk.n_core
            return [[self.blk.slot_core(r, l) for l in range(n_core)] for r in range(self.D)]
        n_layers = len(self.blk.inner.layers)
        return [[self.blk.slot_of(r, l) for l in range(n_layers)] for r in range(self.D)]

    def coda_slots(self) -> list[int]:
        if self.kind == "huginn":
            return [self.blk.slot_coda(i) for i in range(self.blk.n_coda)]
        return []

    def decode_bank(self, slots: list[int], b: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K,V of request b over its DECODED positions only [len(slots), T, Hkv, Dh]."""
        lo, hi = self.prompt_len[b], self.pos[b]
        idx = torch.tensor(slots, device=self.cache.K.device)
        return self.cache.K[idx, b, lo:hi], self.cache.V[idx, b, lo:hi]


def bank_digest(eng: VarEngine) -> dict:
    """sha256 per depth over all requests' decoded-position K/V (bitwise identity check)."""
    out = {}
    groups = {f"d{r + 1}": s for r, s in enumerate(eng.core_slots())}
    if eng.coda_slots():
        groups["coda"] = eng.coda_slots()
    for name, slots in groups.items():
        h = hashlib.sha256()
        for b in range(eng.B):
            k, v = eng.decode_bank(slots, b)
            h.update(k.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            h.update(v.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[name] = h.hexdigest()
    return out


def bank_snapshot(eng: VarEngine) -> dict:
    groups = {f"d{r + 1}": s for r, s in enumerate(eng.core_slots())}
    if eng.coda_slots():
        groups["coda"] = eng.coda_slots()
    snap = {}
    for name, slots in groups.items():
        snap[name] = [tuple(x.clone() for x in eng.decode_bank(slots, b)) for b in range(eng.B)]
    return snap


def bank_rel_diff(a: dict, b: dict) -> dict:
    """max over requests of ||Ka-Kb||/||Kb|| (and V) per depth group."""
    out = {}
    for name in a:
        worst = 0.0
        for (ka, va), (kb, vb) in zip(a[name], b[name]):
            if ka.shape != kb.shape:
                worst = float("inf")
                continue
            for x, y in ((ka, kb), (va, vb)):
                d = (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)
                worst = max(worst, float(d))
        out[name] = worst
    return out


def load(model: str, dtype: str):
    return W.load_model(model, {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype])
