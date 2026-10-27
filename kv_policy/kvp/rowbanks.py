"""Per-row protected-bank caches for fixed-prefix comparisons (OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md Step 2).

`RowBankKV` is kvp.banks.BankKV with one protected (layer, step) set PER BATCH ROW, so FULL (every
bank protected), LAST-CONVERT (none), "persistent swap of one bank" (all but one) and "protect one
bank" (only one) run in lockstep on the same prefix, latent and batch shape. Base mode is convert:
the prompt is prefilled exactly as FULL, then each row's unprotected banks are converted to the
terminal step; every later token is finalised the same way after its own last step.
"""

from __future__ import annotations

import torch

from .engine import PolicyKV


class RowBankKV(PolicyKV):
    def __init__(self, protected: list[frozenset], R, n_rec, n_plain, H, T, D, dtype, device):
        B = len(protected)
        super().__init__("last", R, n_rec, n_plain, B, H, T, D, dtype, device, region_start=0)
        m = torch.zeros(B, n_rec, R, dtype=torch.bool)
        for b, cells in enumerate(protected):
            for l, r in cells:
                m[b, l, r] = True
        m[:, :, R - 1] = True  # the terminal step is always its own version
        self.prot = m.to(device)

    def _apply(self, cols):
        for i, (kbuf, vbuf) in enumerate(zip(self.rk, self.rv)):
            for buf in (kbuf, vbuf):
                term = buf[self.R - 1, :, :, cols]
                for r in range(self.R - 1):
                    keep = self.prot[:, i, r].view(-1, *([1] * (term.dim() - 1)))
                    buf[r, :, :, cols] = torch.where(keep, buf[r, :, :, cols], term)

    def convert_prefix(self, P: int):
        self._apply(slice(0, P))

    def finalize(self, c: int):
        self._apply(slice(c, c + 1))


@torch.inference_mode()
def run_bank_rows(runner, prompt: list[int], cont: list[int], protected: list[frozenset], uid: str, on_step):
    """Teacher-force `cont` after `prompt` for one row per protected set (same latent `uid`).
    on_step(g, logits [B, V]) gets the distribution predicting cont[g]."""
    bb = runner.bb
    B, P, n = len(protected), len(prompt), len(cont)
    T = P + max(n - 1, 0) + 1
    kv = RowBankKV(protected, bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D, runner.dtype, runner.device)
    dev = runner.device
    ids = torch.tensor([prompt] * B, device=dev)
    pos = torch.arange(T, device=dev)[None].expand(B, T)
    valid = torch.ones(B, T, dtype=torch.bool, device=dev)
    aux = bb.make_aux([uid] * B, [0] * B, T)
    logits = runner._prefill_parallel(ids, pos, valid, kv, aux, P)[:, -1]
    kv.convert_prefix(P)
    tgt = torch.tensor(cont, device=dev)
    for g in range(n):
        on_step(g, logits)
        if g == n - 1:
            break
        logits = runner._step(tgt[g].expand(B), P + g, pos, valid, kv, aux)[:, 0]
