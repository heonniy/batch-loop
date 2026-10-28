"""Reference fixed-depth execution engine with FULL / LAST / AVERAGE recurrent-KV policies.

Terminology
-----------
* column c      -- token slot in the left-padded batch; position(b, c) = c - pad_b.
* loop r        -- recurrent iteration 0..R-1 of the shared core (Ouro: all L layers; Huginn: the
                   4-layer core block). Non-recurrent Huginn prelude/coda layers have ordinary
                   single-version caches and are not affected by the policy.
* policy region -- columns whose historical K/V is subject to the policy:
                   mode "convert": every column. The prompt is prefilled in parallel with its own
                                   per-loop K/V (as FULL), then the prompt K/V is converted to the
                                   policy version; decoding reads only policy versions. Retained
                                   memory = one plane (the Ouro paper's 4x claim / plan definition).
                   mode "decode":  c >= P only (generated / teacher-forced tokens); the prompt
                                   keeps every loop version (hybrid, less memory saving).
                   mode "strict":  every column; the prompt is processed token-by-token so that
                                   prompt tokens already read policy versions (vllm-rlt `shared`).

Semantics (per recurrent physical layer l, query of current token t at loop r):
* own position t: reads its own loop-r K/V for every policy (the only causally available version).
* history i < t in the full region: K_i^{(r)}, V_i^{(r)}.
* history i < t in the policy region:
      FULL    -> K_i^{(r)}, V_i^{(r)}
      LAST    -> K_i^{(R-1)}, V_i^{(R-1)}            (final computed loop)
      AVERAGE -> mean_r' K_i^{(r')}, mean_r' V_i^{(r')}  (K and V averaged separately, fp32 mean)
  The policy version of token i is written only after i's last loop (no future-loop leakage).

Storage ("broadcast" layout): every recurrent layer keeps an [R, B, H, T, D] buffer so that all
policies run the *same* attention kernel on the *same* shapes; LAST/AVERAGE finalisation overwrites
every loop plane of a policy-region column with the policy version. Physical memory of this
reference layout is therefore FULL-sized for every policy; compact policy-sized storage is checked
separately in tests/test_compact_storage.py and measured on the serving engine (Stage 3).
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- KV store


class PolicyKV:
    def __init__(self, policy: str, R: int, n_rec: int, n_plain: int, B: int, H: int, T: int, D: int,
                 dtype, device, region_start: int):
        if policy not in ("full", "last", "first", "average"):
            raise ValueError(policy)
        self.policy, self.R, self.region_start = policy, R, region_start
        self.rk = [torch.zeros(R, B, H, T, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.rv = [torch.zeros(R, B, H, T, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.pk = [torch.zeros(B, H, T, D, dtype=dtype, device=device) for _ in range(n_plain)]
        self.pv = [torch.zeros(B, H, T, D, dtype=dtype, device=device) for _ in range(n_plain)]
        self.reads: dict[str, int] = {"policy_region_reads": 0}

    def _buffers(self, layer):
        kind, idx, r = layer
        if kind == "rec":
            return self.rk[idx][r], self.rv[idx][r]
        return self.pk[idx], self.pv[idx]

    def attend(self, layer, q, k, v, c0: int, c1: int, mask):
        """Write K/V of columns [c0, c1) for `layer` and attend over columns [0, c1)."""
        kb, vb = self._buffers(layer)
        kb[:, :, c0:c1] = k
        vb[:, :, c0:c1] = v
        return F.scaled_dot_product_attention(q, kb[:, :, :c1], vb[:, :, :c1], attn_mask=mask)

    def finalize(self, c: int):
        """Called after the last loop of column c (sequential execution only)."""
        if self.policy == "full" or c < self.region_start:
            return
        for kbuf, vbuf in zip(self.rk, self.rv):
            for buf in (kbuf, vbuf):
                if self.policy in ("last", "first"):
                    ver = buf[self.R - 1 if self.policy == "last" else 0, :, :, c]
                else:
                    ver = buf[:, :, :, c].to(torch.promote_types(buf.dtype, torch.float32)).mean(0).to(buf.dtype)
                buf[:, :, :, c] = ver.unsqueeze(0)

    def convert_prefix(self, P: int):
        """Mode "convert": after the parallel prefill, replace prompt K/V by the policy version."""
        if self.policy == "full":
            return
        for kbuf, vbuf in zip(self.rk, self.rv):
            for buf in (kbuf, vbuf):
                if self.policy in ("last", "first"):
                    ver = buf[self.R - 1 if self.policy == "last" else 0, :, :, :P]
                else:
                    ver = buf[:, :, :, :P].to(torch.promote_types(buf.dtype, torch.float32)).mean(0).to(buf.dtype)
                buf[:, :, :, :P] = ver.unsqueeze(0)

    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.rk, *self.rv, *self.pk, *self.pv))


class CompactPolicyKV(PolicyKV):
    """Policy-sized storage for LAST / AVERAGE (memory check; same semantics as PolicyKV).

    Recurrent layers keep R loop planes only for the full region [0, region_start) and ONE plane
    for the policy region. The current token's per-loop K/V live in an R-slot scratch until
    finalize() writes the policy version. Attention concatenates [full-region plane r | policy plane
    | own current K/V], so reads differ from PolicyKV only by kernel input layout.
    """

    def __init__(self, policy, R, n_rec, n_plain, B, H, T, D, dtype, device, region_start, prefill_cols=None):
        if policy == "full":
            raise ValueError("FULL is already policy-sized in PolicyKV")
        self.policy, self.R, self.region_start = policy, R, region_start
        # per-loop planes exist for the prompt in "decode" (kept) and "convert" (transient) modes
        F_ = region_start if prefill_cols is None else prefill_cols
        self.offset = region_start  # first column held in the single policy plane
        G = T - region_start
        self.rk = [torch.zeros(R, B, H, F_, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.rv = [torch.zeros(R, B, H, F_, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.ck = [torch.zeros(B, H, G, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.cv = [torch.zeros(B, H, G, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.sk = [torch.zeros(R, B, H, 1, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.sv = [torch.zeros(R, B, H, 1, D, dtype=dtype, device=device) for _ in range(n_rec)]
        self.pk = [torch.zeros(B, H, T, D, dtype=dtype, device=device) for _ in range(n_plain)]
        self.pv = [torch.zeros(B, H, T, D, dtype=dtype, device=device) for _ in range(n_plain)]
        self.reads = {}
        self.transient_peak_bytes = self.bytes_allocated()

    def attend(self, layer, q, k, v, c0, c1, mask):
        kind, idx, r = layer
        if kind != "rec":
            return super().attend(layer, q, k, v, c0, c1, mask)
        F_ = self.rk[idx].shape[3]
        if c1 <= F_:  # parallel prompt prefill (per-loop prompt planes still present)
            self.rk[idx][r][:, :, c0:c1] = k
            self.rv[idx][r][:, :, c0:c1] = v
            return F.scaled_dot_product_attention(q, self.rk[idx][r][:, :, :c1], self.rv[idx][r][:, :, :c1],
                                                  attn_mask=mask)
        assert c1 == c0 + 1 and c0 >= self.offset, "policy region is processed one column at a time"
        self.sk[idx][r], self.sv[idx][r] = k, v
        g = c0 - self.offset
        full = self.rk[idx][r][:, :, :self.offset], self.rv[idx][r][:, :, :self.offset]
        keys = torch.cat([full[0], self.ck[idx][:, :, :g], k], dim=2)
        vals = torch.cat([full[1], self.cv[idx][:, :, :g], v], dim=2)
        return F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask)

    def convert_prefix(self, P: int):
        for i in range(len(self.rk)):
            for planes, plane in ((self.rk, self.ck), (self.rv, self.cv)):
                buf = planes[i][:, :, :, :P]
                if self.policy in ("last", "first"):
                    ver = buf[self.R - 1 if self.policy == "last" else 0]
                else:
                    ver = buf.to(torch.promote_types(buf.dtype, torch.float32)).mean(0).to(buf.dtype)
                plane[i][:, :, :P] = ver
            # drop the transient per-loop prompt planes
            self.rk[i] = self.rk[i][:, :, :, :0].contiguous()
            self.rv[i] = self.rv[i][:, :, :, :0].contiguous()

    def finalize(self, c: int):
        if c < self.region_start:
            return
        g = c - self.offset
        for sk, sv, ck, cv in zip(self.sk, self.sv, self.ck, self.cv):
            for scratch, plane in ((sk, ck), (sv, cv)):
                if self.policy in ("last", "first"):
                    ver = scratch[self.R - 1 if self.policy == "last" else 0]
                else:
                    ver = scratch.to(torch.promote_types(scratch.dtype, torch.float32)).mean(0).to(scratch.dtype)
                plane[:, :, g: g + 1] = ver

    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (*self.rk, *self.rv, *self.ck, *self.cv, *self.pk, *self.pv))

    def scratch_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.sk, *self.sv))


# ----------------------------------------------------------------------------- backbones


class OuroBackbone:
    family = "ouro"

    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.R = cfg.total_ut_steps
        self.L = cfg.num_hidden_layers
        self.H = cfg.num_key_value_heads
        self.D = cfg.head_dim
        if cfg.num_attention_heads != self.H:
            raise ValueError("engine assumes MHA (no GQA repeat)")
        if any(t != "full_attention" for t in cfg.layer_types):
            raise ValueError("sliding-window layers not supported")
        mod = sys.modules[type(model).__module__]
        self.apply_rope = mod.apply_rotary_pos_emb
        self.n_rec, self.n_plain = self.L, 0

    def make_aux(self, batch_uids, pads, T):
        return None

    def forward(self, ids, pos, kv: PolicyKV, c0, c1, mask, aux, finalize: bool):
        m = self.model.model
        h = m.embed_tokens(ids)
        cos, sin = m.rotary_emb(h, pos)
        B, S, _ = h.shape
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
            h = m.norm(h)  # Ouro normalises after every UT step; step-R output feeds the head
        if finalize:
            kv.finalize(c0)
        return self.model.lm_head(h).float()


class HuginnBackbone:
    family = "huginn"

    def __init__(self, model, num_steps: int, noise_seed: int):
        self.model = model
        cfg = model.config
        self.R = num_steps
        self.H = cfg.num_key_value_heads
        self.D = cfg.n_embd // cfg.num_attention_heads
        self.E = cfg.n_embd
        if cfg.num_attention_heads != self.H:
            raise ValueError("engine assumes MHA")
        if getattr(cfg, "test_time_noise", 0):
            raise ValueError("test_time_noise must be 0")
        mod = sys.modules[type(model).__module__]
        self.apply_rope = mod.apply_rotary_emb_complex_like
        tr = model.transformer
        self.n_pre, self.n_core, self.n_coda = len(tr.prelude), len(tr.core_block), len(tr.coda)
        self.n_rec, self.n_plain = self.n_core, self.n_pre + self.n_coda
        self.noise_seed = noise_seed
        self.std = cfg.init_values["std"]
        self.emb_scale = float(model.emb_scale)

    # Deterministic per-(example, position) initial latent state, independent of batch shape,
    # policy, mode and sequence length (generated in fixed 256-position blocks).
    def noise_block(self, uid: str, block: int) -> torch.Tensor:
        seed = int(hashlib.sha256(f"{self.noise_seed}:{uid}:{block}".encode()).hexdigest()[:15], 16)
        g = torch.Generator().manual_seed(seed)
        x = torch.empty(256, self.E)
        torch.nn.init.trunc_normal_(x, mean=0.0, std=self.std, a=-3 * self.std, b=3 * self.std, generator=g)
        return x * self.emb_scale

    def example_noise(self, uid: str, length: int) -> torch.Tensor:
        blocks = [self.noise_block(uid, b) for b in range((length + 255) // 256)]
        return torch.cat(blocks)[:length]

    def make_aux(self, batch_uids, pads, T):
        dev, dt = self.model.lm_head.weight.device, self.model.lm_head.weight.dtype
        noise = torch.zeros(len(batch_uids), T, self.E, dtype=dt, device=dev)
        for b, (uid, pad) in enumerate(zip(batch_uids, pads)):
            noise[b, pad:] = self.example_noise(uid, T - pad).to(dev, dt)
        return noise

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
        o = kv.attend(layer, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), c0, c1, mask)
        o = at.proj(o.transpose(1, 2).reshape(B, S, self.E))
        x = block.norm_2(o + x)
        return block.norm_4(block.mlp(block.norm_3(x)) + x)

    def forward(self, ids, pos, kv: PolicyKV, c0, c1, mask, aux, finalize: bool):
        tr = self.model.transformer
        freqs = self.model.freqs_cis[0][pos]  # [B, S, 1, D/2, 2]
        e = tr.wte(ids) * self.emb_scale
        for i, blk in enumerate(tr.prelude):
            e = self._block(blk, e, freqs, kv, ("plain", i, 0), c0, c1, mask)
        x = aux[:, c0:c1]
        for s in range(self.R):
            x = tr.adapter(torch.cat([x, e], dim=-1))
            for j, blk in enumerate(tr.core_block):
                x = self._block(blk, x, freqs, kv, ("rec", j, s), c0, c1, mask)
        x = tr.ln_f(x)
        for i, blk in enumerate(tr.coda):
            x = self._block(blk, x, freqs, kv, ("plain", self.n_pre + i, 0), c0, c1, mask)
        x = tr.ln_f(x)
        if finalize:
            kv.finalize(c0)
        return self.model.lm_head(x).float()


# ----------------------------------------------------------------------------- batch runner


@dataclass
class BatchResult:
    gen_ids: list[list[int]] = field(default_factory=list)
    finish: list[str] = field(default_factory=list)
    target_logprobs: list[list[float]] = field(default_factory=list)
    first_logits: torch.Tensor | None = None
    stats: dict = field(default_factory=dict)


class Runner:
    def __init__(self, backbone, pad_id: int, store: str = "broadcast"):
        self.bb = backbone
        self.store = store
        p = next(backbone.model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.pad_id = pad_id
        self.cuda = self.device.type == "cuda"

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _reset_peak(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak(self):
        return torch.cuda.max_memory_allocated(self.device) if self.cuda else 0

    def _setup(self, prompts, extra, policy, mode, uids):
        B = len(prompts)
        P = max(len(p) for p in prompts)
        T = P + extra
        pads = [P - len(p) for p in prompts]
        ids = torch.full((B, P), self.pad_id, dtype=torch.long)
        for b, p in enumerate(prompts):
            ids[b, pads[b]:] = torch.tensor(p)
        cols = torch.arange(T)
        pos_full = (cols[None, :] - torch.tensor(pads)[:, None]).clamp_min(0)
        valid = cols[None, :] >= torch.tensor(pads)[:, None]
        if mode not in ("decode", "strict", "convert"):
            raise ValueError(mode)
        region = P if mode == "decode" else 0
        if policy.startswith("bank"):  # E3 selective (layer x step) protection, see kvp.banks
            from .banks import REGISTRY, BankKV

            if mode not in ("convert", "strict"):
                raise ValueError("bank policies are defined for modes 'convert' and 'strict'")
            kv = BankKV(REGISTRY[(getattr(self, "model_key", None), policy)], self.bb.R, self.bb.n_rec, self.bb.n_plain,
                        B, self.bb.H, T, self.bb.D, self.dtype, self.device)
        elif self.store == "compact" and policy != "full":
            kv = CompactPolicyKV(policy, self.bb.R, self.bb.n_rec, self.bb.n_plain, B, self.bb.H, T, self.bb.D,
                                 self.dtype, self.device, region, prefill_cols=P if mode != "strict" else 0)
        else:
            kv = PolicyKV(policy, self.bb.R, self.bb.n_rec, self.bb.n_plain, B, self.bb.H, T, self.bb.D,
                          self.dtype, self.device, region)
        aux = self.bb.make_aux(uids, pads, T)
        return ids.to(self.device), pos_full.to(self.device), valid.to(self.device), kv, aux, P, T

    def _prefill_parallel(self, ids, pos_full, valid, kv, aux, P):
        causal = torch.ones(P, P, dtype=torch.bool, device=self.device).tril()
        mask = causal[None] & valid[:, None, :P]
        mask |= torch.eye(P, dtype=torch.bool, device=self.device)[None]  # pad rows: self only (no NaN)
        return self.bb.forward(ids, pos_full[:, :P], kv, 0, P, mask[:, None], aux, finalize=False)

    def _step(self, tok, c, pos_full, valid, kv, aux):
        mask = valid[:, None, None, : c + 1].clone()
        mask[:, :, :, c] = True
        return self.bb.forward(tok[:, None], pos_full[:, c: c + 1], kv, c, c + 1, mask, aux, finalize=True)

    def _prefill(self, ids, pos_full, valid, kv, aux, P, mode, keep_all=False):
        if mode in ("decode", "convert"):
            logits = self._prefill_parallel(ids, pos_full, valid, kv, aux, P)
            if mode == "convert":
                kv.convert_prefix(P)
            return logits if keep_all else logits[:, -1]
        outs = []
        for c in range(P):
            lg = self._step(ids[:, c], c, pos_full, valid, kv, aux)[:, 0]
            if keep_all:
                outs.append(lg)
        return torch.stack(outs, 1) if keep_all else lg

    @torch.inference_mode()
    def generate(self, prompts, uids, policy, mode, max_new, eos_ids, stop_check) -> BatchResult:
        """Greedy decoding. `stop_check(b, ids)` -> True when row b's text hit a stop string."""
        t0 = time.perf_counter()
        self._reset_peak()
        ids, pos_full, valid, kv, aux, P, T = self._setup(prompts, max_new, policy, mode, uids)
        bytes_at_alloc = kv.bytes_allocated()
        logits = self._prefill(ids, pos_full, valid, kv, aux, P, mode)
        self._sync()
        t_prefill = time.perf_counter() - t0
        B = len(prompts)
        res = BatchResult(gen_ids=[[] for _ in range(B)], finish=["length"] * B)
        res.first_logits = logits.detach().float().cpu()
        done = [False] * B
        eos = torch.tensor(sorted(eos_ids), device=self.device)
        steps = 0
        for t in range(max_new):
            tok = logits.argmax(-1)
            tok_cpu = tok.tolist()
            is_eos = torch.isin(tok, eos).tolist()
            for b in range(B):
                if done[b]:
                    continue
                res.gen_ids[b].append(tok_cpu[b])
                if is_eos[b]:
                    done[b], res.finish[b] = True, "eos"
                elif stop_check(b, res.gen_ids[b]):
                    done[b], res.finish[b] = True, "stop_string"
            if all(done) or t == max_new - 1:
                break
            logits = self._step(tok, P + t, pos_full, valid, kv, aux)[:, 0]
            steps += 1
        self._sync()
        res.stats = {"B": B, "P": P, "T_alloc": T, "decode_steps": steps, "prefill_s": t_prefill,
                     "total_s": time.perf_counter() - t0, "kv_store_bytes": kv.bytes_allocated(),
                     "kv_store_bytes_at_alloc": bytes_at_alloc,
                     "peak_mem_bytes": self._peak()}
        return res

    @torch.inference_mode()
    def score(self, prompts, targets, uids, policy, mode) -> BatchResult:
        """Teacher-forced log-probabilities of `targets` given `prompts` (gold tokens are fed as
        policy-region tokens exactly as generated tokens would be)."""
        t0 = time.perf_counter()
        self._reset_peak()
        n = max(len(t) for t in targets)
        ids, pos_full, valid, kv, aux, P, T = self._setup(prompts, n, policy, mode, uids)
        logits = self._prefill(ids, pos_full, valid, kv, aux, P, mode)
        B = len(prompts)
        tgt = torch.full((B, n), self.pad_id, dtype=torch.long)
        for b, t in enumerate(targets):
            tgt[b, : len(t)] = torch.tensor(t)
        tgt = tgt.to(self.device)
        lps = []
        for i in range(n):
            lps.append(torch.log_softmax(logits, -1).gather(1, tgt[:, i: i + 1])[:, 0])
            if i == n - 1:
                break
            logits = self._step(tgt[:, i], P + i, pos_full, valid, kv, aux)[:, 0]
        lp = torch.stack(lps, 1).cpu()
        res = BatchResult()
        res.target_logprobs = [lp[b, : len(t)].tolist() for b, t in enumerate(targets)]
        self._sync()
        res.stats = {"B": B, "P": P, "T_alloc": T, "total_s": time.perf_counter() - t0,
                     "kv_store_bytes": kv.bytes_allocated(),
                     "peak_mem_bytes": self._peak()}
        return res

    @torch.inference_mode()
    def teacher_forced_logits(self, prompt, cont, uid, policy, mode):
        """B=1 logits for every position of prompt+cont (for parity tests). Modes "decode" and
        "convert" use the parallel prompt prefill, "strict" feeds the prompt token-by-token."""
        n = len(cont)
        ids, pos_full, valid, kv, aux, P, T = self._setup([prompt], n, policy, mode, [uid])
        outs = [self._prefill(ids, pos_full, valid, kv, aux, P, mode, keep_all=True)[0]]
        ct = torch.tensor(cont, device=self.device)
        for i in range(n):
            outs.append(self._step(ct[i: i + 1], P + i, pos_full, valid, kv, aux)[0])
        return torch.cat(outs, 0).float().cpu()
