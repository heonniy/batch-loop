"""Ouro-1.4B one-recurrence adapter with physical survivor compute."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase02_runtime.kv import OuroLastExitedKV
from phase04_runtime.policy import ouro_decision


OURO14_REVISION = "37bc7e8ff85a90d813dc096383405f521b6a67b0"
OURO14_SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots" / OURO14_REVISION
)


class OuroAdapter:
    rmax = 4
    kv_policy = "last-exited-depth-indexed"
    model_revision = OURO14_REVISION

    def __init__(self, model, tokenizer, num_rows: int, *, instrumentation: str = "profile"):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.num_rows = num_rows
        if model.config.total_ut_steps != self.rmax:
            raise ValueError("Ouro configured UT steps do not match the physical Rmax")
        if any(kind != "full_attention" for kind in model.config.layer_types):
            raise ValueError("Ouro sliding/local attention requires a separately validated mask")
        if model.config.num_attention_heads != model.config.num_key_value_heads:
            raise ValueError("Ouro eager attention path requires matched Q/KV heads")
        self.instrumentation = instrumentation
        self.last_exit_decision = None
        self.backend = OuroLastExitedKV(
            num_rows, self.device, num_layers=model.config.num_hidden_layers, rmax=self.rmax,
            instrumentation=instrumentation,
        )
        self.position_ids: torch.Tensor | None = None
        self.remaining: torch.Tensor | None = None
        self.prefill_logits: torch.Tensor | None = None

    @classmethod
    def load(cls, num_rows: int, snapshot: Path = OURO14_SNAPSHOT,
             *, instrumentation: str = "profile") -> "OuroAdapter":
        if not snapshot.exists():
            raise FileNotFoundError(snapshot)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="eager",
        )
        # Both FIFO and Oracle use this exact, explicit attention path.
        model.config._attn_implementation = "eager"
        return cls(model, tokenizer, num_rows, instrumentation=instrumentation)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, prompt_valid: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != self.num_rows:
            raise ValueError("prefill must initialize all persistent cache rows")
        self.backend.begin_prefill(prompt_valid)
        seq_len = input_ids.shape[1]
        positions = (prompt_valid.long().cumsum(dim=1) - 1).clamp_min(0)
        hidden = self.model.model.embed_tokens(input_ids)
        rope = self.model.model.rotary_emb(hidden, positions)
        mask = self.backend.attention_bias(seq_len, seq_len, hidden.dtype)
        cache_position = torch.arange(seq_len, device=self.device)
        for recurrence in range(1, self.rmax + 1):
            for layer in self.model.model.layers:
                hidden = layer(
                    hidden, attention_mask=mask, position_ids=positions,
                    past_key_value=self.backend, use_cache=True,
                    cache_position=cache_position, position_embeddings=rope,
                    current_ut=recurrence - 1,
                )
            hidden = self.model.model.norm(hidden)
        self.backend.finish_prefill()
        logits = self.model.lm_head(hidden[:, -1:, :])
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("non-finite Ouro prefill logits")
        self.prefill_logits = logits.float()
        return logits.argmax(dim=-1)

    def begin_token(self) -> None:
        self.backend.begin_token()
        assert self.backend.key_valid is not None
        self.position_ids = (self.backend.key_valid.long().sum(dim=1) - 1).unsqueeze(1)
        self.remaining = torch.ones(self.num_rows, device=self.device, dtype=torch.float32)

    def prelude(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.model.model.embed_tokens(token_ids)
        return embedded, embedded

    def recurrent_step(self, state: torch.Tensor, embedded: torch.Tensor, recurrence: int) -> torch.Tensor:
        del embedded
        assert self.position_ids is not None and self.backend.key_valid is not None
        rows = self.backend.active_rows
        positions = self.position_ids.index_select(0, rows)
        rope = self.model.model.rotary_emb(state, positions)
        seq_len = self.backend.key_valid.shape[1]
        mask = self.backend.attention_bias(1, seq_len, state.dtype)
        cache_position = torch.tensor([seq_len - 1], device=self.device)
        hidden = state
        for layer in self.model.model.layers:
            hidden = layer(
                hidden, attention_mask=mask, position_ids=positions,
                past_key_value=self.backend, use_cache=True,
                cache_position=cache_position, position_embeddings=rope,
                current_ut=recurrence - 1,
            )
        return self.model.model.norm(hidden)

    def live_exit(self, before: torch.Tensor, after: torch.Tensor, rows: torch.Tensor,
                  recurrence: int, threshold: float) -> torch.Tensor:
        del before
        assert self.remaining is not None
        previous = self.remaining.index_select(0, rows)
        decision, remaining = ouro_decision(
            self.model.model.early_exit_gate(after), previous,
            recurrence, self.rmax, threshold,
            validate_finite=self.instrumentation != "benchmark",
        )
        self.remaining.index_copy_(0, rows, remaining)
        self.last_exit_decision = decision
        return decision.mask

    @torch.inference_mode()
    def coda_logits(self, final_state: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(final_state).float()
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("non-finite Ouro final logits")
        return logits

    @torch.inference_mode()
    def coda(self, final_state: torch.Tensor) -> torch.Tensor:
        return self.coda_logits(final_state).argmax(dim=-1)

    def commit_token(self) -> None:
        self.backend.commit_token()
