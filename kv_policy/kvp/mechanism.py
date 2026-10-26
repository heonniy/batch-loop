"""Trace instrumentation for the Huginn R32 attention-mechanism study (HUGINN_R32_ATTENTION_MECHANISM_PLAN.md).

`TraceHuginn` is a drop-in HuginnBackbone (same arithmetic, same KV store calls) that, during
single-token decode steps, hands every block's tensors to a `Tracer`. Tracing only reads tensors;
the forward result is unchanged (tests/test_mechanism.py).

Experiment B (fixed query, row 0 = FULL): at decode column c, recurrent layer j, loop s, with the
FULL query q_F^(s) and the current token's own K,V, recompute attention in FP32 over past columns
    ref  : past K^(s), V^(s)       (what FULL reads)
    swap : past K^(R), V^(R)       (terminal; what LAST would read for the same query)
    konly: past K^(R), V^(s)       vonly: past K^(s), V^(R)
    lay  : ref with past columns in reversed storage order (layout control, must equal ref)
and project with an FP32 copy of the output projection. The engine's own BF16 output of the same
call is the precision-floor control.
Experiment C (rows 0 = FULL, 1 = LAST, same prefix and latent): per (loop, layer) relative errors of
the block input / post-attention / output, observed attention and MLP update differences and their
alignment with the incoming error. These are observed differences, not isolated causal effects.
Experiment D: `intervene = (j, s)` makes that one attention call read terminal history
(past K^(R), V^(R)) while every other call and the cache writes stay FULL.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .engine import HuginnBackbone

EPS = 1e-12
B_METRICS = ("D_swap", "D_konly", "D_vonly", "Dhead_swap", "cos_swap", "normratio_swap", "tv_mean", "tv_max",
             "entropy_ref", "selfmass_ref", "selfmass_swap", "D_precision", "D_layout", "attn_over_resid",
             "Dswap_over_resid")
C_METRICS = ("err_in", "rel_da", "cos_err_da", "err_sum1", "err_mid", "rel_dm", "cos_err_dm", "err_sum2", "err_out",
             "attn_over_resid", "mlp_over_resid")


def _rel(a, b):
    return (a - b).norm() / (b.norm() + EPS)


def _cos(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + EPS)


class Tracer:
    """Collects per-decode-step metrics. step_* lists hold one tensor per traced step."""

    def __init__(self, bb: HuginnBackbone, exp_b: bool = True, exp_c: bool = True, keep_vectors: bool = False):
        self.bb, self.exp_b, self.exp_c, self.keep_vectors = bb, exp_b, exp_c, keep_vectors
        self.R, self.nj = bb.R, bb.n_core
        self.proj32: dict[int, torch.Tensor] = {}
        self.b_steps, self.c_steps, self.plain_steps, self.vec_steps = [], [], [], []
        self._new()

    def _new(self):
        dev = self.bb.model.lm_head.weight.device
        self.cur_b = torch.full((self.R, self.nj, len(B_METRICS)), float("nan"), device=dev)
        self.cur_c = torch.full((self.R, self.nj, len(C_METRICS)), float("nan"), device=dev)
        self.cur_plain: dict[str, float] = {}
        self.cur_vec: dict = {}

    def end_step(self):
        self.b_steps.append(self.cur_b.cpu())
        self.c_steps.append(self.cur_c.cpu())
        self.plain_steps.append({k: float(v) for k, v in self.cur_plain.items()})
        if self.keep_vectors:
            self.vec_steps.append(self.cur_vec)
        self._new()

    # ---------------------------------------------------------------- experiment B (row 0)
    def _w32(self, j, at):
        if j not in self.proj32:
            self.proj32[j] = at.proj.weight.detach().float()
        return self.proj32[j]

    def fixed_q(self, j, s, at, q, k, v, kv, c0, o_engine, x_in):
        R = self.R
        qf = q[0].float()  # [H, 1, D]
        ks, vs = k[0].float(), v[0].float()
        kr, vr = kv.rk[j][s][0, :, :c0].float(), kv.rv[j][s][0, :, :c0].float()
        kt, vt = kv.rk[j][R - 1][0, :, :c0].float(), kv.rv[j][R - 1][0, :, :c0].float()
        scale = qf.shape[-1] ** -0.5

        def attn(kp, vp):
            K, V = torch.cat([kp, ks], 1), torch.cat([vp, vs], 1)
            p = torch.softmax((qf @ K.transpose(1, 2)) * scale, -1)  # [H, 1, c0+1]
            return p @ V, p[:, 0]

        W = self._w32(j, at)
        H, D = qf.shape[0], qf.shape[-1]
        proj = lambda o: F.linear(o.transpose(0, 1).reshape(H * D), W)  # noqa: E731
        o_ref, p_ref = attn(kr, vr)
        o_sw, p_sw = attn(kt, vt)
        o_k, _ = attn(kt, vr)
        o_v, _ = attn(kr, vt)
        o_lay, _ = attn(kr.flip(1), vr.flip(1))
        A_ref, A_sw, A_k, A_v, A_lay = proj(o_ref), proj(o_sw), proj(o_k), proj(o_v), proj(o_lay)
        A_eng = proj(o_engine[0].float())  # engine's BF16 attention output (pre-proj) of the same call
        tv = 0.5 * (p_ref - p_sw).abs().sum(-1)
        ent = -(p_ref * (p_ref + EPS).log()).sum(-1)
        dh = ((o_ref - o_sw).norm(dim=-1) / (o_ref.norm(dim=-1) + EPS)).mean()
        xin = x_in[0, 0].float().norm()
        vals = [_rel(A_sw, A_ref), _rel(A_k, A_ref), _rel(A_v, A_ref), dh, _cos(A_sw, A_ref),
                A_sw.norm() / (A_ref.norm() + EPS), tv.mean(), tv.max(), ent.mean(),
                p_ref[:, -1].mean(), p_sw[:, -1].mean(), _rel(A_eng, A_ref), _rel(A_lay, A_ref),
                A_ref.norm() / (xin + EPS), (A_sw - A_ref).norm() / (xin + EPS)]
        self.cur_b[s, j] = torch.stack(vals)

    # ---------------------------------------------------------------- experiment C (rows 0, 1)
    def errors(self, j, s, x, a, xm, m, out):
        f = lambda t: t[:, 0].float()  # noqa: E731  [B, E]
        x, a, xm, m, out = f(x), f(a), f(xm), f(m), f(out)
        s1, s2 = a + x, m + xm
        e_in, e_mid = x[1] - x[0], xm[1] - xm[0]
        vals = [_rel(x[1], x[0]), _rel(a[1], a[0]), _cos(e_in, a[1] - a[0]), _rel(s1[1], s1[0]), _rel(xm[1], xm[0]),
                _rel(m[1], m[0]), _cos(e_mid, m[1] - m[0]), _rel(s2[1], s2[0]), _rel(out[1], out[0]),
                a[0].norm() / (x[0].norm() + EPS), m[0].norm() / (xm[0].norm() + EPS)]
        self.cur_c[s, j] = torch.stack(vals)

    def plain(self, name, t):
        if t.shape[0] >= 2 and self.exp_c:
            self.cur_plain[name] = _rel(t[1, 0].float(), t[0, 0].float())  # tensor; materialised in end_step
        if self.keep_vectors:
            self.cur_vec[name] = t[:, 0].detach().clone()


class TraceHuginn(HuginnBackbone):
    def __init__(self, base: HuginnBackbone):  # noqa: D107 - share the wrapped backbone's state
        self.__dict__.update(base.__dict__)
        self.tracer: Tracer | None = None
        self.intervene: tuple[int, int] | None = None

    def _block(self, block, x, freqs, kv, layer, c0, c1, mask):
        B, S, _ = x.shape
        at = block.attn
        q, k, v = at.Wqkv(block.norm_1(x)).split(at.chunks, dim=2)
        q = q.view(B, S, at.n_head, at.head_dim)
        k = k.view(B, S, at.n_kv_heads, at.head_dim)
        v = v.view(B, S, at.n_kv_heads, at.head_dim)
        if at.config.qk_bias:
            qb, kb = at.qk_bias.split(1, dim=0)
            q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k = self.apply_rope(q, k, freqs_cis=freqs)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        kind, j, s = layer
        if self.intervene is not None and kind == "rec" and (j, s) == self.intervene:
            kb_, vb_ = kv.rk[j], kv.rv[j]
            kb_[s][:, :, c0:c1], vb_[s][:, :, c0:c1] = kh, vh  # same cache write as FULL
            K = torch.cat([kb_[self.R - 1][:, :, :c0], kh], 2)
            V = torch.cat([vb_[self.R - 1][:, :, :c0], vh], 2)
            o = F.scaled_dot_product_attention(qh, K, V, attn_mask=mask)
        else:
            o = kv.attend(layer, qh, kh, vh, c0, c1, mask)
        a = at.proj(o.transpose(1, 2).reshape(B, S, self.E))
        xm = block.norm_2(a + x)
        m = block.mlp(block.norm_3(xm))
        out = block.norm_4(m + xm)
        tr = self.tracer
        if tr is not None and S == 1:
            if kind == "rec":
                if tr.exp_b:
                    tr.fixed_q(j, s, at, qh, kh, vh, kv, c0, o, x)
                if tr.exp_c and B >= 2:
                    tr.errors(j, s, x, a, xm, m, out)
                if tr.keep_vectors:
                    tr.cur_vec[("rec", j, s)] = (a[:, 0].detach().clone(), out[:, 0].detach().clone())
            else:
                tr.plain(f"{'prelude' if j < self.n_pre else 'coda'}{j if j < self.n_pre else j - self.n_pre}", out)
        return out

    def forward(self, ids, pos, kv, c0, c1, mask, aux, finalize: bool):
        tr_ = self.model.transformer
        freqs = self.model.freqs_cis[0][pos]
        e = tr_.wte(ids) * self.emb_scale
        for i, blk in enumerate(tr_.prelude):
            e = self._block(blk, e, freqs, kv, ("plain", i, 0), c0, c1, mask)
        x = aux[:, c0:c1]
        for s in range(self.R):
            x = tr_.adapter(torch.cat([x, e], dim=-1))
            for j, blk in enumerate(tr_.core_block):
                x = self._block(blk, x, freqs, kv, ("rec", j, s), c0, c1, mask)
        x = tr_.ln_f(x)
        if self.tracer is not None and x.shape[1] == 1:
            self.tracer.plain("post_core_lnf", x)
        for i, blk in enumerate(tr_.coda):
            x = self._block(blk, x, freqs, kv, ("plain", self.n_pre + i, 0), c0, c1, mask)
        x = tr_.ln_f(x)
        if self.tracer is not None and x.shape[1] == 1:
            self.tracer.plain("final_lnf", x)
        if finalize:
            kv.finalize(c0)
        return self.model.lm_head(x).float()


def kv_geometry(kv, row: int, upto: int) -> dict[str, torch.Tensor]:
    """Experiment A on the FULL trajectory (row with per-loop history kept), columns [0, upto).

    Returns per-(layer j, loop s, column, head) arrays [n_core, R, C, H] of
      dK = ||K^(s) - K^(R)|| / (||K^(s)|| + eps), dV likewise, cosK, cosV (vs terminal loop R),
      cK = ||K^(s+1) - K^(s)|| / (||K^(s)|| + eps) (consecutive; last loop = nan), cV likewise.
    Same token and position for every loop, so RoPE is aligned; K is post-RoPE as stored."""
    out = {k: [] for k in ("dK", "dV", "cosK", "cosV", "cK", "cV")}
    for kbuf, vbuf in zip(kv.rk, kv.rv):
        for name, buf in (("K", kbuf), ("V", vbuf)):
            x = buf[:, row, :, :upto].float().transpose(1, 2)  # [R, C, H, D]
            term = x[-1:]
            n = x.norm(dim=-1)
            out[f"d{name}"].append(((x - term).norm(dim=-1) / (n + 1e-6)))
            out[f"cos{name}"].append(F.cosine_similarity(x, term.expand_as(x), dim=-1))
            cons = torch.full_like(n, float("nan"))
            cons[:-1] = (x[1:] - x[:-1]).norm(dim=-1) / (n[:-1] + 1e-6)
            out[f"c{name}"].append(cons)
    return {k: torch.stack(v) for k, v in out.items()}  # [n_core, R, C, H]


@torch.inference_mode()
def intervention_pass(runner, prompt: list[int], cont: list[int], uid: str, steps: set[int],
                      grid: list[tuple[int, int]]) -> list[dict]:
    """Experiment D. B=1 FULL teacher-forced pass over `cont`. Before the real FULL step that feeds
    cont[g-1] (g in `steps`), run one extra forward of that token per (j, s) in `grid` in which only
    attention call (layer j, loop s) reads terminal history; then run the real step, which rewrites
    every cache entry of that column (keep-all store: finalize is a no-op), so later steps are
    unaffected. Compares each intervention with the real step of the same token."""
    from .oracle import MixedKV

    bb = runner.bb
    assert isinstance(bb, TraceHuginn)
    P, n = len(prompt), len(cont)
    T = P + n
    dev = runner.device
    kv = MixedKV(["last"], torch.ones(1, T, dtype=torch.bool), bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D,
                 runner.dtype, dev)
    ids = torch.tensor([prompt], device=dev)
    pos = torch.arange(T, device=dev)[None]
    valid = torch.ones(1, T, dtype=torch.bool, device=dev)
    aux = bb.make_aux([uid], [0], T)
    runner._prefill_parallel(ids, pos, valid, kv, aux, P)
    tgt = torch.tensor(cont, device=dev)
    out = []
    last_core = bb.n_core - 1
    for g in range(1, n):
        c = P + g - 1
        tok = tgt[g - 1: g]
        if g not in steps:
            runner._step(tok, c, pos, valid, kv, aux)
            continue
        ints = []
        for js in grid:
            bb.intervene = js
            bb.tracer = Tracer(bb, exp_b=False, exp_c=False, keep_vectors=True)
            lg = runner._step(tok, c, pos, valid, kv, aux)[0, 0]
            ints.append((js, lg, bb.tracer.cur_vec))
        bb.intervene = None
        bb.tracer = Tracer(bb, exp_b=False, exp_c=False, keep_vectors=True)
        ref_lg = runner._step(tok, c, pos, valid, kv, aux)[0, 0]
        ref = bb.tracer.cur_vec
        bb.tracer = None
        lr = torch.log_softmax(ref_lg.double(), -1)
        for (j, s), lg, vec in ints:
            li = torch.log_softmax(lg.double(), -1)
            f = lambda key, i: vec[key][i][0].float()  # noqa: E731
            g_ = lambda key, i: ref[key][i][0].float()  # noqa: E731
            traj = [float(_rel(f(("rec", last_core, r), 1), g_(("rec", last_core, r), 1))) for r in range(bb.R)]
            rec = {"g": g, "layer": j, "loop": s + 1,
                   "imm_attn": float(_rel(f(("rec", j, s), 0), g_(("rec", j, s), 0))),
                   "post_layer": float(_rel(f(("rec", j, s), 1), g_(("rec", j, s), 1))),
                   "post_R": traj[-1], "loop_end_traj": traj,
                   "post_core_lnf": float(_rel(vec["post_core_lnf"][0].float(), ref["post_core_lnf"][0].float())),
                   "final_lnf": float(_rel(vec["final_lnf"][0].float(), ref["final_lnf"][0].float())),
                   "kl_ref_int": float((lr.exp() * (lr - li)).sum()), "argmax_same": bool(lr.argmax() == li.argmax()),
                   "target_logprob_ref": float(lr[tgt[g]]), "target_logprob_int": float(li[tgt[g]])}
            out.append(rec)
    return out
