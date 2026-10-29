"""Analytic logical-KV memory accounting for the knob-screening experiment.

Assumes the same K/V vector shape and dtype for every attention layer.
This is accurate for the investigated Huginn/Ouro reference layouts and is
NOT the actual broadcast-buffer allocation, prefill scratch, or GPU peak HBM.
Dory's sliding-window attention must be handled by a separate exact budget.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVLayout:
    plain_layers: int
    recurrent_layers: int
    depth: int

    def __post_init__(self):
        if min(self.plain_layers, self.recurrent_layers) < 0 or self.depth < 1:
            raise ValueError("Invalid layer counts/depth")
        if self.plain_layers + self.recurrent_layers == 0:
            raise ValueError("At least one attention layer required")

    @property
    def full_planes(self) -> int:
        return self.plain_layers + self.recurrent_layers * self.depth

    @property
    def last_planes(self) -> int:
        return self.plain_layers + self.recurrent_layers

    def last_over_full(self) -> float:
        return self.last_planes / self.full_planes

    def protected_over_full(self, n_protected_banks: int) -> float:
        """Each extra recurrent layer×recurrence plane counts once."""
        if n_protected_banks < 0 or n_protected_banks > self.recurrent_layers * (self.depth - 1):
            raise ValueError("Invalid bank count")
        return (self.last_planes + n_protected_banks) / self.full_planes

    def postlast_window_over_full(self, sequence_tokens: int, window: int,
                                  sinks: int = 4, prompt_tokens: int = 0,
                                  keep_prompt: bool = False) -> float:
        """Logical active-KV ratio at END of sequence for window+sink policy."""
        if sequence_tokens < 1 or window < 1 or sinks < 0:
            raise ValueError("Invalid sequence/window/sink")
        if prompt_tokens < 0 or prompt_tokens > sequence_tokens:
            raise ValueError("Prompt length out of range")
        surviving = set(range(max(0, sequence_tokens - window), sequence_tokens))
        surviving.update(range(min(sequence_tokens, sinks)))
        if keep_prompt:
            surviving.update(range(prompt_tokens))
        return self.last_over_full() * len(surviving) / sequence_tokens


def layout_for_model(model: str) -> KVLayout:
    if model.startswith("huginn-0125"):
        if model == "huginn-0125":
            depth = 32
        else:
            depth = int(model.rsplit("-r", 1)[1])
        return KVLayout(plain_layers=4, recurrent_layers=4, depth=depth)
    if model == "ouro-1.4b":
        return KVLayout(plain_layers=0, recurrent_layers=24, depth=4)
    if model == "ouro-2.6b":
        return KVLayout(plain_layers=0, recurrent_layers=48, depth=4)
    raise ValueError(f"Unsupported analytic reference layout: {model}")
