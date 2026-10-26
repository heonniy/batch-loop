"""Cross-depth KV interventions on one current token, for both families (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md, E2).

`make_fn(bb, kv, token, c, x0)` returns a family-specific single-token recurrence with a common API:
  init()                         -> state before step 1 ([1, 1, E]; Ouro: token embedding, Huginn: latent x0)
  step(h, s, ...)                -> state after step s (1-based)
  readout(h)                     -> logits [N, V]
  block_io(l, x_in, s, src)      -> (attention output a, block output) of layer l at step s when its
                                    attention reads the FULL history of step `src` (src = s is FULL)
Everything outside the current token (history K/V of all layers/steps, Huginn prelude input e and coda
history, position) is frozen from a B=1 FULL keep-all pass; no cache is written. Rows are independent
branches. `inject[(l, s)] = delta [N, 1, E]` adds delta to the output of layer l at step s;
`read_terminal = {(l, s), ...}` makes those attention calls read the terminal-step history (persistent
LAST-style reads for the current token only). `cap[(l, s)] = (x_in, a, out)` records the baseline.
Ouro layers = all decoder layers (each runs every step); Huginn layers = the 4 core blocks.
"""

from __future__ import annotations

import torch

from .stability import _attend, attn_half, make_ctx, readout as huginn_readout


class OuroFn:
    def __init__(self, bb, kv, token: int, c: int):
        self.bb, self.m = bb, bb.model.model
        self.R, self.L = bb.R, bb.L
        dev = bb.model.lm_head.weight.device
        self.h0 = self.m.embed_tokens(torch.tensor([[token]], device=dev))
        self.cos, self.sin = self.m.rotary_emb(self.h0, torch.tensor([[c]], device=dev))
        self.hk = [[kv.rk[l][s][0, :, :c] for s in range(self.R)] for l in range(self.L)]
        self.hv = [[kv.rv[l][s][0, :, :c] for s in range(self.R)] for l in range(self.L)]

    def init(self):
        return self.h0

    def _attn(self, layer, x, Kh, Vh):
        N = x.shape[0]
        at = layer.self_attn
        D = self.bb.D
        q = at.q_proj(x).view(N, 1, -1, D).transpose(1, 2)
        k = at.k_proj(x).view(N, 1, -1, D).transpose(1, 2)
        v = at.v_proj(x).view(N, 1, -1, D).transpose(1, 2)
        q, k = self.bb.apply_rope(q, k, self.cos, self.sin)
        o = _attend(q, k, v, Kh, Vh)
        return at.o_proj(o.transpose(1, 2).reshape(N, 1, -1))

    def _layer(self, l, h, Kh, Vh):
        layer = self.m.layers[l]
        a = self._attn(layer, layer.input_layernorm(h), Kh, Vh)
        h1 = h + layer.input_layernorm_2(a)
        return a, h1 + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h1)))

    def block_io(self, l, x_in, s, src):
        return self._layer(l, x_in, self.hk[l][src - 1], self.hv[l][src - 1])

    def step(self, h, s, inject=None, read_terminal=frozenset(), cap=None):
        for l in range(self.L):
            src = self.R if (l, s) in read_terminal else s
            x_in = h
            a, h = self._layer(l, h, self.hk[l][src - 1], self.hv[l][src - 1])
            if cap is not None:
                cap[(l, s)] = (x_in, a, h)
            if inject is not None and (l, s) in inject:
                h = h + inject[(l, s)]
        return self.m.norm(h)

    def readout(self, h):
        return self.bb.model.lm_head(h).float()[:, 0]


class HuginnFn:
    def __init__(self, bb, kv, token: int, c: int, x0):
        self.bb = bb
        self.ctx = make_ctx(bb, kv, token, c, x0)
        self.R, self.L = bb.R, bb.n_core
        self.tr = bb.model.transformer

    def init(self):
        return self.ctx.x0

    def block_io(self, j, x_in, s, src):
        blk = self.tr.core_block[j]
        a = attn_half(self.bb, blk, x_in, self.ctx.freqs, self.ctx.rec_k[j][src - 1], self.ctx.rec_v[j][src - 1])
        xm = blk.norm_2(a + x_in)
        return a, blk.norm_4(blk.mlp(blk.norm_3(xm)) + xm)

    def step(self, z, s, inject=None, read_terminal=frozenset(), cap=None):
        x = self.tr.adapter(torch.cat([z, self.ctx.e.expand(z.shape[0], -1, -1)], dim=-1))
        for j in range(self.L):
            src = self.R if (j, s) in read_terminal else s
            x_in = x
            a, x = self.block_io(j, x, s, src)
            if cap is not None:
                cap[(j, s)] = (x_in, a, x)
            if inject is not None and (j, s) in inject:
                x = x + inject[(j, s)]
        return x

    def readout(self, z):
        return huginn_readout(self.ctx, z)


def make_fn(bb, kv, token: int, c: int, x0=None):
    return OuroFn(bb, kv, token, c) if bb.family == "ouro" else HuginnFn(bb, kv, token, c, x0)


def run_all(fn, h, start: int = 0, end: int | None = None, **kw):
    """States after steps start+1..end."""
    end = fn.R if end is None else end
    out = []
    for s in range(start + 1, end + 1):
        h = fn.step(h, s, **kw)
        out.append(h)
    return out
