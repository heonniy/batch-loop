"""Canonical request/position-keyed Huginn initial latent states.

The cached model's initial-state distribution is truncated normal with its
configured standard deviation and embedding scale. This component changes
which random draw belongs to a request, not the marginal distribution.
"""

from __future__ import annotations

import hashlib

import torch


class HuginnStateIdentity:
    def __init__(self, *, seed: int, standard_deviation: float, embedding_scale: float):
        if standard_deviation <= 0 or embedding_scale <= 0:
            raise ValueError("Huginn state distribution must have positive scale")
        self.seed = seed
        self.std = standard_deviation
        self.embedding_scale = embedding_scale

    def _generator(self, request_id: int, purpose: str, position: int, device: torch.device
                   ) -> torch.Generator:
        key = f"phase04-h0-v1:{self.seed}:{request_id}:{purpose}:{position}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
        return torch.Generator(device=device).manual_seed(value)

    def _draw(self, shape: tuple[int, ...], request_id: int, purpose: str,
              position: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        result = torch.empty(shape, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(result, mean=0.0, std=self.std,
                                    a=-3 * self.std, b=3 * self.std,
                                    generator=self._generator(request_id, purpose, position, device))
        return result * self.embedding_scale

    def prefill(self, canonical_ids: torch.Tensor, valid: torch.Tensor,
                hidden: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if valid.ndim != 2 or canonical_ids.shape != (valid.shape[0],):
            raise ValueError("prefill identity/valid-mask shape mismatch")
        output = torch.empty((*valid.shape, hidden), dtype=dtype, device=device)
        for local, request_id in enumerate(canonical_ids.tolist()):
            indices = torch.nonzero(valid[local], as_tuple=False).flatten()
            length = int(indices.numel())
            if length:
                output[local, indices] = self._draw((length, hidden), int(request_id),
                                                    "prefill", 0, dtype, device)
            if length != valid.shape[1]:
                pad = torch.nonzero(~valid[local], as_tuple=False).flatten()
                output[local, pad] = self._draw((len(pad), hidden), int(request_id),
                                                "prefill_padding", 0, dtype, device)
        return output

    def decode(self, canonical_ids: torch.Tensor, positions: torch.Tensor,
               hidden: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if positions.shape != (len(canonical_ids), 1):
            raise ValueError("decode identity positions must be [B,1]")
        return torch.stack([
            self._draw((1, hidden), int(request_id), "decode", int(position), dtype, device)
            for request_id, position in zip(canonical_ids.tolist(), positions[:, 0].tolist())
        ])


def state_sha256(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
