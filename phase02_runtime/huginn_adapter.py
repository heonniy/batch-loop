"""Huginn-3.5B physical survivor adapter with shared recurrent KV.

The local attention wrapper replaces the upstream flex-attention path that
rejects head dimension 96. It uses explicit causal/padding masks and SDPA's
math backend; no cached Hugging Face source is edited.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase02_runtime.kv import HuginnSharedKV
from phase04_runtime.identity import HuginnStateIdentity, state_sha256
from phase04_runtime.policy import huginn_decision


HUGINN_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
HUGINN_SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--tomg-group-umd--huginn-0125/snapshots" / HUGINN_REVISION
)


def _sdpa_head96_forward(attn, x, freqs_cis, block_idx, mask=None, past_key_values=None):
    if mask is not None:
        raise ValueError("Phase-2 Huginn expects its own explicit padding/causal mask")
    if not isinstance(past_key_values, HuginnSharedKV):
        raise TypeError("Phase-2 Huginn attention requires HuginnSharedKV")
    batch, seq, emb = x.shape
    q, k, v = attn.Wqkv(x).split(attn.chunks, dim=2)
    q = q.view(batch, seq, attn.n_head, attn.head_dim)
    k = k.view(batch, seq, attn.n_kv_heads, attn.head_dim)
    v = v.view(batch, seq, attn.n_kv_heads, attn.head_dim)
    if attn.config.qk_bias:
        q_bias, k_bias = attn.qk_bias.split(1, dim=0)
        q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(k.dtype)
    rotate = attn._phase02_rotary
    q, k = rotate(q, k, freqs_cis=freqs_cis)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    k, v = past_key_values.update(k, v, block_idx)
    bias = past_key_values.attention_bias(seq, k.shape[2], q.dtype)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
    diagnostic_observer = getattr(attn, "_phase04_sdpa_observer", None)
    if diagnostic_observer is not None:
        diagnostic_observer(q, k, v, bias, y)
    y = y.transpose(1, 2).reshape(batch, seq, emb).contiguous()
    return attn.proj(y)


class HuginnAdapter:
    rmax = 16
    kv_policy = "shared-physical-layer"
    model_revision = HUGINN_REVISION

    def __init__(self, model, tokenizer, num_rows: int, state_init: str = "zero",
                 *, identity_seed: int | None = None, canonical_ids: torch.Tensor | None = None,
                 instrumentation: str = "profile", capture_state_hashes: bool = False):
        if state_init not in {"zero", "native-seeded"}:
            raise ValueError(state_init)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.num_rows = num_rows
        self.state_init = state_init
        if model.config.n_layers_in_prelude != 2:
            raise ValueError("Huginn core index convention requires exactly two Prelude layers")
        if model.config.n_heads != model.config.num_key_value_heads:
            raise ValueError("Huginn local attention wrapper does not support GQA")
        if identity_seed is not None and state_init != "native-seeded":
            raise ValueError("identity-keyed h0 requires native-seeded initialization")
        if (identity_seed is None) != (canonical_ids is None):
            raise ValueError("identity seed and canonical IDs must be supplied together")
        if canonical_ids is not None and canonical_ids.shape != (num_rows,):
            raise ValueError("one canonical ID is required per persistent KV row")
        if identity_seed is not None and model.config.test_time_noise != 0:
            raise ValueError("identity-keyed recurrent noise is not implemented for this checkpoint")
        self.canonical_ids = canonical_ids
        self.identity = (None if identity_seed is None else HuginnStateIdentity(
            seed=identity_seed, standard_deviation=model.config.init_values["std"],
            embedding_scale=model.emb_scale))
        self.instrumentation = instrumentation
        self.capture_state_hashes = capture_state_hashes
        self.state_hashes: dict[tuple[int, int | str], str] = {}
        self.prompt_valid: torch.Tensor | None = None
        self.last_exit_decision = None
        self.backend = HuginnSharedKV(
            num_rows, self.device,
            n_prelude=model.config.n_layers_in_prelude,
            n_recurrent=model.config.n_layers_in_recurrent_block,
            n_coda=model.config.n_layers_in_coda,
            instrumentation=instrumentation,
        )
        self.position_ids: torch.Tensor | None = None
        self.prev_for_gate: torch.Tensor | None = None
        self.prefill_logits: torch.Tensor | None = None
        remote = importlib.import_module(model.__class__.__module__)
        rotary = remote.apply_rotary_emb_complex_like
        all_blocks = (
            list(model.transformer.prelude) + list(model.transformer.core_block) + list(model.transformer.coda)
        )
        for block in all_blocks:
            block.attn._phase02_rotary = rotary
            block.attn.forward = types.MethodType(_sdpa_head96_forward, block.attn)

    @classmethod
    def load(cls, num_rows: int, snapshot: Path = HUGINN_SNAPSHOT,
             state_init: str = "zero", *, identity_seed: int | None = None,
             canonical_ids: torch.Tensor | None = None,
             instrumentation: str = "profile", capture_state_hashes: bool = False) -> "HuginnAdapter":
        if not snapshot.exists():
            raise FileNotFoundError(snapshot)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        return cls(model, tokenizer, num_rows, state_init, identity_seed=identity_seed,
                   canonical_ids=canonical_ids, instrumentation=instrumentation,
                   capture_state_hashes=capture_state_hashes)

    def _freqs(self, rows: torch.Tensor) -> torch.Tensor:
        assert self.position_ids is not None
        positions = self.position_ids.index_select(0, rows)
        return self.model.freqs_cis[0, positions]

    def _initial_state(self, embedded: torch.Tensor) -> torch.Tensor:
        if self.state_init == "zero":
            return torch.zeros_like(embedded)
        if self.identity is None:
            return self.model.initialize_state(embedded)
        assert self.canonical_ids is not None and self.position_ids is not None
        rows = self.backend.active_rows
        ids = self.canonical_ids.index_select(0, rows)
        if self.backend.phase == "prefill":
            assert self.prompt_valid is not None
            valid = self.prompt_valid.index_select(0, rows)
            state = self.identity.prefill(ids, valid, embedded.shape[-1],
                                          embedded.dtype, embedded.device)
            if self.capture_state_hashes:
                for local, request_id in enumerate(ids.tolist()):
                    self.state_hashes[(request_id, "prefill")] = state_sha256(state[local, valid[local]])
            return state
        if self.backend.phase == "token":
            positions = self.position_ids.index_select(0, rows)
            state = self.identity.decode(ids, positions, embedded.shape[-1],
                                         embedded.dtype, embedded.device)
            if self.capture_state_hashes:
                for local, (request_id, position) in enumerate(zip(ids.tolist(), positions[:, 0].tolist())):
                    self.state_hashes[(request_id, position)] = state_sha256(state[local])
            return state
        raise RuntimeError("Huginn h0 sampled outside prefill/token")

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, prompt_valid: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != self.num_rows:
            raise ValueError("prefill must initialize all persistent Huginn rows")
        self.backend.begin_prefill(prompt_valid)
        self.prompt_valid = prompt_valid
        self.position_ids = (prompt_valid.long().cumsum(dim=1) - 1).clamp_min(0)
        freqs = self._freqs(self.backend.active_rows)
        embedded = self.model.transformer.wte(input_ids)
        if self.model.emb_scale != 1:
            embedded = embedded * self.model.emb_scale
        for idx, block in enumerate(self.model.transformer.prelude):
            embedded = block(embedded, freqs, idx, None, self.backend)
        state = self._initial_state(embedded)
        for recurrence in range(1, self.rmax + 1):
            idx = 1 + (recurrence - 1) * self.backend.n_recurrent
            state, _ = self.model.core_block_forward(
                state, embedded, freqs, None, self.backend,
                torch.tensor(idx, dtype=torch.long), recurrence - 1,
            )
        state = self.model.transformer.ln_f(state)
        for idx, block in enumerate(self.model.transformer.coda, 1):
            state = block(state, freqs, -idx, None, self.backend)
        state = self.model.transformer.ln_f(state)
        logits = self.model.lm_head(state[:, -1:, :]).float()
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("non-finite Huginn prefill logits")
        self.prefill_logits = logits
        self.backend.finish_prefill()
        return logits.argmax(dim=-1)

    def begin_token(self) -> None:
        self.backend.begin_token()
        assert self.backend.key_valid is not None
        self.position_ids = (self.backend.key_valid.long().sum(dim=1) - 1).unsqueeze(1)

    def prelude(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = self._freqs(self.backend.active_rows)
        embedded = self.model.transformer.wte(token_ids)
        if self.model.emb_scale != 1:
            embedded = embedded * self.model.emb_scale
        for idx, block in enumerate(self.model.transformer.prelude):
            embedded = block(embedded, freqs, idx, None, self.backend)
        return embedded, self._initial_state(embedded)

    def recurrent_step(self, state: torch.Tensor, embedded: torch.Tensor, recurrence: int) -> torch.Tensor:
        freqs = self._freqs(self.backend.active_rows)
        idx = 1 + (recurrence - 1) * self.backend.n_recurrent
        out, _ = self.model.core_block_forward(
            state, embedded, freqs, None, self.backend,
            torch.tensor(idx, dtype=torch.long), recurrence - 1,
        )
        return out

    def live_exit(self, before: torch.Tensor, after: torch.Tensor, rows: torch.Tensor,
                  recurrence: int, threshold: float) -> torch.Tensor:
        del rows
        decision = huginn_decision(before, after, recurrence, self.rmax, threshold,
                                   validate_finite=self.instrumentation != "benchmark")
        self.last_exit_decision = decision
        return decision.mask

    @torch.inference_mode()
    def coda_logits(self, final_state: torch.Tensor) -> torch.Tensor:
        freqs = self._freqs(self.backend.active_rows)
        state = self.model.transformer.ln_f(final_state)
        for idx, block in enumerate(self.model.transformer.coda, 1):
            state = block(state, freqs, -idx, None, self.backend)
        state = self.model.transformer.ln_f(state)
        logits = self.model.lm_head(state).float()
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("non-finite Huginn final logits")
        return logits

    @torch.inference_mode()
    def coda(self, final_state: torch.Tensor) -> torch.Tensor:
        return self.coda_logits(final_state).argmax(dim=-1)

    def commit_token(self) -> None:
        self.backend.commit_token()
