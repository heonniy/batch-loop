"""Selective (layer x step) KV-bank protection for E3 (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md).

A policy named "bank<NAME>" (no '-', so it fits the "<policy>-<mode>" job naming) is LAST-CONVERT
except that the protected recurrent banks (layer l, 0-based step r < R-1) keep their own per-step
K/V for every historical token: a query of step r at layer l reads K/V^(r) if (l, r) is protected,
the terminal K/V^(R-1) otherwise. Unprotected = LAST-CONVERT, all protected = FULL. The prompt is
prefilled exactly as FULL (loop-major, every plane) and then converted.
Retained recurrent KV per token = (n_layers + |protected|) planes vs n_layers * R for FULL.
Masks are registered per model: REGISTRY[(model_key, policy)] = set of (l, r).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .engine import PolicyKV

REGISTRY: dict[tuple[str, str], frozenset] = {}


def load_registry(path: Path):
    data = json.loads(Path(path).read_text())
    for model, pols in data.items():
        for pol, cells in pols.items():
            REGISTRY[(model, pol)] = frozenset((int(l), int(r)) for l, r in cells)


class BankKV(PolicyKV):
    def __init__(self, protected: frozenset, R, n_rec, n_plain, B, H, T, D, dtype, device):
        super().__init__("last", R, n_rec, n_plain, B, H, T, D, dtype, device, region_start=0)
        self.protected = protected

    def _apply(self, cols):
        for i, (kbuf, vbuf) in enumerate(zip(self.rk, self.rv)):
            for buf in (kbuf, vbuf):
                term = buf[self.R - 1, :, :, cols]
                for r in range(self.R - 1):
                    if (i, r) not in self.protected:
                        buf[r, :, :, cols] = term

    def convert_prefix(self, P: int):
        self._apply(slice(0, P))

    def finalize(self, c: int):
        self._apply(slice(c, c + 1))
