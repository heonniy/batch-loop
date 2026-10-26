"""Functional single-token Huginn recurrence for perturbation / Jacobian analysis
(HUGINN_R32_PERTURBATION_STABILITY_PLAN.md).

For one current token at column c of a FULL teacher-forced pass, `TokenCtx` freezes everything
outside the current token's recurrent state: the prompt + prefix history K,V of every layer
(read from the FULL keep-all store, columns [0, c), never written), the token's prelude output e
(re-injected by the adapter every loop), its position (RoPE) and its initial latent x0.

    z_0 = x0;   z_s = F_s(z_{s-1}),  F_s(z) = core_s(adapter([z, e]))   (s = 1..R, 1-based loops)
    logits = head(coda(ln_f(z_R)))

core_s runs the 4 core blocks at loop s; each block's attention reads the frozen FULL history at
loop s plus the token's OWN current K,V (computed from z). Rows of z are independent branches
(batch dimension N), so many perturbations of the same token run in one call and torch.func
JVP/VJP see a pure function of z. Attention is computed explicitly (softmax in max(dtype, fp32)), not by SDPA;
`parity` compares the result with the engine's own output for the same token.

Boundary convention: z_r = state after loop r (= input of loop r+1). An injection at loop r
perturbs z_r and re-executes loops r+1..R.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenCtx:
    bb: object                 # HuginnBackbone
    c: int                     # column of the current token
    freqs: torch.Tensor        # [1, 1, 1, D/2, 2] RoPE for position c
    e: torch.Tensor            # [1, 1, E] prelude output (adapter input), frozen
    x0: torch.Tensor           # [1, 1, E] initial latent
    rec_k: list                # rec_k[j][s] -> [H, c, D] frozen FULL history (layer j, 0-based loop s)
    rec_v: list
    coda_k: list               # [H, c, D] per coda block
    coda_v: list


def _attend(q, k_own, v_own, Kh, Vh):
    """q, k_own, v_own: [N, H, 1, D]; Kh, Vh: [H, c, D] shared history. Returns [N, H, 1, D]."""
    D = q.shape[-1]
    acc = torch.promote_types(q.dtype, torch.float32)
    qf = q.to(acc)
    s_hist = torch.einsum("nhqd,hkd->nhqk", qf, Kh.to(acc)) * D ** -0.5
    s_self = (qf * k_own.to(acc)).sum(-1, keepdim=True) * D ** -0.5
    p = torch.softmax(torch.cat([s_hist, s_self], -1), -1)
    out = torch.einsum("nhqk,hkd->nhqd", p[..., :-1], Vh.to(acc)) + p[..., -1:] * v_own.to(acc)
    return out.to(q.dtype)


def attn_half(bb, block, x, freqs, Kh, Vh):
    """x [N, 1, E] -> attention branch output a (post-projection) [N, 1, E]."""
    N, S, E = x.shape
    at = block.attn
    q, k, v = at.Wqkv(block.norm_1(x)).split(at.chunks, dim=2)
    q = q.view(N, S, at.n_head, at.head_dim)
    k = k.view(N, S, at.n_kv_heads, at.head_dim)
    v = v.view(N, S, at.n_kv_heads, at.head_dim)
    if at.config.qk_bias:
        qb, kb = at.qk_bias.split(1, dim=0)
        q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
    q, k = bb.apply_rope(q, k, freqs_cis=freqs.expand(N, *freqs.shape[1:]))
    o = _attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), Kh, Vh)
    return at.proj(o.transpose(1, 2).reshape(N, S, E))


def block_fwd(bb, block, x, freqs, Kh, Vh, cap: dict | None = None, patch: dict | None = None, key=None):
    """One sandwich block for N independent single-token rows x [N, 1, E] (same math as
    HuginnBackbone._block):  xm = norm_2(a(x) + x);  out = norm_4(mlp(norm_3(xm)) + xm).
    cap[key] = (a, m); patch[key] = {"a": t} / {"m": t} replaces that branch output."""
    a = attn_half(bb, block, x, freqs, Kh, Vh)
    if patch is not None and key in patch and "a" in patch[key]:
        a = patch[key]["a"]
    xm = block.norm_2(a + x)
    m = block.mlp(block.norm_3(xm))
    if patch is not None and key in patch and "m" in patch[key]:
        m = patch[key]["m"]
    if cap is not None:
        cap[key] = (a, m)
    return block.norm_4(m + xm)


def sublayer_maps(ctx, s: int):
    """The one-loop map F_s split at residual boundaries, as a list of (name, fn) whose
    composition equals loop_fn(ctx, ., s): adapter, then per core layer j the attention half
    (x -> norm_2(a + x)) and the MLP half (xm -> norm_4(mlp(norm_3 xm) + xm))."""
    bb, tr = ctx.bb, ctx.bb.model.transformer
    maps = [("adapter", lambda z: tr.adapter(torch.cat([z, ctx.e.expand(z.shape[0], -1, -1)], dim=-1)))]
    for j, blk in enumerate(tr.core_block):
        Kh, Vh = ctx.rec_k[j][s - 1], ctx.rec_v[j][s - 1]
        maps.append((f"attn{j}", lambda x, blk=blk, Kh=Kh, Vh=Vh: blk.norm_2(attn_half(bb, blk, x, ctx.freqs, Kh, Vh) + x)))
        maps.append((f"mlp{j}", lambda xm, blk=blk: blk.norm_4(blk.mlp(blk.norm_3(xm)) + xm)))
    return maps


def loop_fn(ctx: TokenCtx, z, s: int, e_scale: float = 1.0, terminal_layers=(), cap=None, patch=None):
    """F_s for 1-based loop s: z [N, 1, E] -> [N, 1, E]. terminal_layers: core layers whose
    attention reads the terminal (loop R) history instead of loop s (one KV-swap read).
    e_scale scales the re-injected prelude embedding (diagnostic ablation; 1.0 = model)."""
    bb, tr = ctx.bb, ctx.bb.model.transformer
    R = bb.R
    e = ctx.e.expand(z.shape[0], -1, -1) * e_scale
    x = tr.adapter(torch.cat([z, e], dim=-1))
    for j, blk in enumerate(tr.core_block):
        src = R - 1 if j in terminal_layers else s - 1
        x = block_fwd(bb, blk, x, ctx.freqs, ctx.rec_k[j][src], ctx.rec_v[j][src], cap, patch, ("rec", j, s))
    return x


def readout(ctx: TokenCtx, z):
    """Final logits [N, V] from z_R."""
    bb, tr = ctx.bb, ctx.bb.model.transformer
    x = tr.ln_f(z)
    for i, blk in enumerate(tr.coda):
        x = block_fwd(bb, blk, x, ctx.freqs, ctx.coda_k[i], ctx.coda_v[i])
    return bb.model.lm_head(tr.ln_f(x)).float()[:, 0]


def trajectory(ctx: TokenCtx, z, start: int, end: int | None = None, **kw):
    """States after loops start+1..end (1-based; z = state after loop `start`). Returns list of
    end-start tensors."""
    end = ctx.bb.R if end is None else end
    out = []
    for s in range(start + 1, end + 1):
        z = loop_fn(ctx, z, s, **kw)
        out.append(z)
    return out


def prelude_e(bb, kv, ids_tok, freqs, c: int):
    """Prelude output for the current token from the frozen plain history [0, c)."""
    tr = bb.model.transformer
    x = tr.wte(ids_tok) * bb.emb_scale  # [1, 1, E]
    for i, blk in enumerate(tr.prelude):
        x = block_fwd(bb, blk, x, freqs, kv.pk[i][0, :, :c], kv.pv[i][0, :, :c])
    return x


def make_ctx(bb, kv, token: int, c: int, x0) -> TokenCtx:
    """kv: keep-all store of a B=1 FULL pass in which columns [0, c) are final."""
    dev = bb.model.lm_head.weight.device
    freqs = bb.model.freqs_cis[0][torch.tensor([[c]], device=dev)]
    e = prelude_e(bb, kv, torch.tensor([[token]], device=dev), freqs, c)
    R = bb.R
    rec_k = [[kv.rk[j][s][0, :, :c] for s in range(R)] for j in range(bb.n_core)]
    rec_v = [[kv.rv[j][s][0, :, :c] for s in range(R)] for j in range(bb.n_core)]
    coda_k = [kv.pk[bb.n_pre + i][0, :, :c] for i in range(bb.n_coda)]
    coda_v = [kv.pv[bb.n_pre + i][0, :, :c] for i in range(bb.n_coda)]
    return TokenCtx(bb, c, freqs, e, x0.reshape(1, 1, -1), rec_k, rec_v, coda_k, coda_v)


def rel(a, b, dim=-1):
    return (a - b).norm(dim=dim) / (b.norm(dim=dim) + 1e-12)
