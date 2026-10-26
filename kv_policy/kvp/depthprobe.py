"""Depth probes for the Huginn-vs-Ouro study (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md, E1).

Drop-in backbones (identical arithmetic to kvp.engine) that record the current tokens' state after
every recurrent step:
  Ouro   : h_r = norm(h) after UT step r (the model re-normalises after every step and its own
           per-step exit logits are lm_head(h_r); gate_r = early_exit_gate(h_r)) -> native per-step
           logits and the official ACT exit distribution (sigmoid gates -> p_r with the remaining
           mass on the last step, modeling_ouro.OuroForCausalLM.forward).
  Huginn : z_r = latent after loop r (before ln_f). Its depth-r prediction is ln_f -> coda -> ln_f ->
           lm_head on z_r, the path the model's own adaptive-compute generation uses; the coda reads
           the frozen coda history and is evaluated functionally (no cache writes).
"""

from __future__ import annotations

import torch

from .engine import HuginnBackbone, OuroBackbone


class ProbeOuro(OuroBackbone):
    def __init__(self, base: OuroBackbone):  # noqa: D107
        self.__dict__.update(base.__dict__)
        self.record = False
        self.states: list[torch.Tensor] = []

    def forward(self, ids, pos, kv, c0, c1, mask, aux, finalize: bool):
        m = self.model.model
        h = m.embed_tokens(ids)
        cos, sin = m.rotary_emb(h, pos)
        B, S, _ = h.shape
        states = []
        for r in range(self.R):
            for l, layer in enumerate(m.layers):
                res = h
                x = layer.input_layernorm(h)
                at = layer.self_attn
                q = at.q_proj(x).view(B, S, -1, self.D).transpose(1, 2)
                k = at.k_proj(x).view(B, S, -1, self.D).transpose(1, 2)
                v = at.v_proj(x).view(B, S, -1, self.D).transpose(1, 2)
                q, k = self.apply_rope(q, k, cos, sin)
                o = kv.attend(("rec", l, r), q, k, v, c0, c1, mask)
                o = at.o_proj(o.transpose(1, 2).reshape(B, S, -1))
                h = res + layer.input_layernorm_2(o)
                res = h
                h = res + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
            h = m.norm(h)
            if self.record:
                states.append(h)
        if self.record:
            self.states = states
        if finalize:
            kv.finalize(c0)
        return self.model.lm_head(h).float()

    def depth_logits(self, states):
        """states: list of R tensors [..., E] -> logits [R, ..., V] (fp32) and gate logits [R, ...]."""
        lg = torch.stack([self.model.lm_head(s).float() for s in states])
        gates = torch.stack([self.model.model.early_exit_gate(s).float()[..., 0] for s in states])
        return lg, gates


def ouro_exit_pdf(gates: torch.Tensor) -> torch.Tensor:
    """Official Ouro ACT exit distribution from per-step gate logits [R, ...] -> pdf [R, ...]."""
    lam = torch.sigmoid(gates.double())
    remaining = torch.ones_like(lam[0])
    pdf = []
    for i in range(lam.shape[0]):
        if i < lam.shape[0] - 1:
            pdf.append(lam[i] * remaining)
            remaining = remaining * (1.0 - lam[i])
        else:
            pdf.append(remaining)
    return torch.stack(pdf)


class ProbeHuginn(HuginnBackbone):
    def __init__(self, base: HuginnBackbone):  # noqa: D107
        self.__dict__.update(base.__dict__)
        self.record = False
        self.states: list[torch.Tensor] = []

    def forward(self, ids, pos, kv, c0, c1, mask, aux, finalize: bool):
        tr = self.model.transformer
        freqs = self.model.freqs_cis[0][pos]
        e = tr.wte(ids) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            e = self._block(blk, e, freqs, kv, ("plain", i, 0), c0, c1, mask)
        x = aux[:, c0:c1]
        states = []
        for s in range(self.R):
            x = tr.adapter(torch.cat([x, e], dim=-1))
            for j, blk in enumerate(tr.core_block):
                x = self._block(blk, x, freqs, kv, ("rec", j, s), c0, c1, mask)
            if self.record:
                states.append(x)
        if self.record:
            self.states = states
        x = tr.ln_f(x)
        for i, blk in enumerate(tr.coda):
            x = self._block(blk, x, freqs, kv, ("plain", self.n_pre + i, 0), c0, c1, mask)
        x = tr.ln_f(x)
        if finalize:
            kv.finalize(c0)
        return self.model.lm_head(x).float()

    def depth_logits_decode(self, states, kv, c: int):
        """Single-token decode at column c (B=1): states R x [1, 1, E] -> logits [R, V].
        The coda reads its frozen history [0, c) plus the probed token's own K/V (no cache write)."""
        from .stability import TokenCtx, readout

        dev = self.model.lm_head.weight.device
        freqs = self.model.freqs_cis[0][torch.tensor([[c]], device=dev)]
        ctx = TokenCtx(self, c, freqs, None, None, None, None,
                       [kv.pk[self.n_pre + i][0, :, :c] for i in range(self.n_coda)],
                       [kv.pv[self.n_pre + i][0, :, :c] for i in range(self.n_coda)])
        return readout(ctx, torch.cat(states, 0))  # [R, V]


def pairwise_kv_geometry(kv, row: int, upto: int, loops: list[int], segs: dict) -> dict:
    """Cross-depth K/V geometry of a keep-all (FULL) store: for every recurrent layer, loop pair
    (r, s) in `loops` (0-based) and segment, mean over columns x heads of
      dist = ||X^(r) - X^(s)|| / ||X^(s)||   and   cos(X^(r), X^(s)),  X in {K, V}.
    Returns {f"{X}_{stat}|{seg}": tensor [n_layers, len(loops), len(loops)]}."""
    out = {}
    for name, bufs in (("K", kv.rk), ("V", kv.rv)):
        for seg, sl in segs.items():
            d_all, c_all = [], []
            for buf in bufs:
                x = buf[loops, row, :, sl].float()  # [L, H, C, D]
                n = x.norm(dim=-1)  # [L, H, C]
                d = (x[:, None] - x[None]).norm(dim=-1) / (n[None] + 1e-6)  # [r, s, H, C] (normalised by s)
                cs = torch.nn.functional.cosine_similarity(x[:, None], x[None], dim=-1)
                d_all.append(d.mean((-1, -2)))
                c_all.append(cs.mean((-1, -2)))
            out[f"{name}_dist|{seg}"] = torch.stack(d_all).cpu()
            out[f"{name}_cos|{seg}"] = torch.stack(c_all).cpu()
    return out
