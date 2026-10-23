"""Mixed per-token recurrent-KV caches for the Huginn R32 divergence / restoration oracle study.

`MixedKV` extends the broadcast reference store (kvp.engine.PolicyKV, mode "convert") with a
per-(row, column) restore mask and a per-row base policy:

* base "last" / "average": every historical column holds its terminal / loop-mean K,V on every loop
  plane (exactly PolicyKV "last"/"average" convert), EXCEPT columns with keep[b, c] = True, which
  keep their own per-loop K^{(r)}, V^{(r)}: a recurrent query at loop r reads K_c^{(r)}, V_c^{(r)}
  for those columns (FULL semantics for that token only).
* keep all  == FULL;  keep none == the base policy.

Restored per-loop K,V of prompt columns come from the parallel FULL prefill of the same prefix;
those of later columns are computed causally in the mixed run itself (each column's own forward
already reads the mixed history). The current token always reads its own loop-r K,V. Prelude/coda
caches are single-version and unaffected. Rows of one call share the prompt and teacher-forced
continuation (no padding), so they differ only by policy / keep mask / initial latent.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .engine import PolicyKV


class MixedKV(PolicyKV):
    def __init__(self, bases: list[str], keep: torch.Tensor, R, n_rec, n_plain, H, T, D, dtype, device):
        B = len(bases)
        super().__init__("last", R, n_rec, n_plain, B, H, T, D, dtype, device, region_start=0)
        if any(b not in ("last", "average") for b in bases):
            raise ValueError(bases)
        self.keep = keep.to(device=device, dtype=torch.bool)  # [B, T]
        self.is_avg = torch.tensor([b == "average" for b in bases], device=device)

    def _policy_version(self, buf: torch.Tensor) -> torch.Tensor:
        """buf: [R, B, H, ..., D] per-loop planes -> per-row policy version [B, H, ..., D]."""
        last = buf[self.R - 1]
        if not bool(self.is_avg.any()):
            return last
        avg = buf.to(torch.promote_types(buf.dtype, torch.float32)).mean(0).to(buf.dtype)
        sel = self.is_avg.view(-1, *([1] * (last.dim() - 1)))
        return torch.where(sel, avg, last)

    def convert_prefix(self, P: int):
        keep = self.keep[:, :P].view(1, -1, 1, P, 1)
        for kbuf, vbuf in zip(self.rk, self.rv):
            for buf in (kbuf, vbuf):
                part = buf[:, :, :, :P]
                ver = self._policy_version(part).unsqueeze(0)
                buf[:, :, :, :P] = torch.where(keep, part, ver)

    def finalize(self, c: int):
        keep = self.keep[:, c].view(1, -1, 1, 1)
        for kbuf, vbuf in zip(self.rk, self.rv):
            for buf in (kbuf, vbuf):
                part = buf[:, :, :, c]
                ver = self._policy_version(part).unsqueeze(0)
                buf[:, :, :, c] = torch.where(keep, part, ver)


@torch.inference_mode()
def run_rows(runner, prompt: list[int], cont: list[int], rows: list[dict], on_step, keep_kv: bool = False):
    """Teacher-force `cont` after `prompt` for several rows in lockstep (one batch, no padding).

    rows: {"base": "last"|"average", "keep": bool tensor [P + len(cont)] | "all" | None,
           "uid": latent-noise key}. on_step(g, logits[B, V]) is called with the distribution that
    predicts cont[g] (g = 0 is the prefill output at the last prompt token). Returns the store if
    keep_kv (per-loop planes of keep-all rows stay readable)."""
    bb = runner.bb
    B, P, n = len(rows), len(prompt), len(cont)
    T = P + max(n - 1, 0) + 1
    keep = torch.zeros(B, T, dtype=torch.bool)
    for b, r in enumerate(rows):
        k = r.get("keep")
        if isinstance(k, str) and k == "all":
            keep[b] = True
        elif k is not None:
            keep[b, : len(k)] = torch.as_tensor(k, dtype=torch.bool)[:T]
    kv = MixedKV([r["base"] for r in rows], keep, bb.R, bb.n_rec, bb.n_plain, bb.H, T, bb.D,
                 runner.dtype, runner.device)
    dev = runner.device
    ids = torch.tensor([prompt] * B, device=dev)
    pos_full = torch.arange(T, device=dev)[None].expand(B, T)
    valid = torch.ones(B, T, dtype=torch.bool, device=dev)
    aux = bb.make_aux([r["uid"] for r in rows], [0] * B, T)
    logits = runner._prefill_parallel(ids, pos_full, valid, kv, aux, P)[:, -1]
    kv.convert_prefix(P)
    tgt = torch.tensor(cont, device=dev)
    for g in range(n):
        on_step(g, logits)
        if g == n - 1:
            break
        logits = runner._step(tgt[g].expand(B), P + g, pos_full, valid, kv, aux)[:, 0]
    return kv if keep_kv else None


def convergence_scores(kv: MixedKV, row: int, upto: int) -> dict[str, torch.Tensor]:
    """Per-column non-convergence of the per-loop K,V of a keep-all (FULL) row, columns [0, upto).

    dist_to_terminal[c] = mean over rec layers, heads, loops r < R-1, and {K, V} of
                          ||X_c^{(r)} - X_c^{(R-1)}|| / ||X_c^{(R-1)}||
    consecutive[c]      = same with ||X_c^{(r)} - X_c^{(r-1)}||, r = 1..R-1, over ||X_c^{(R-1)}||
    (L2 over head_dim; equal layers/heads for K and V)."""
    dist, cons = [], []
    for kbuf, vbuf in zip(kv.rk, kv.rv):
        for buf in (kbuf, vbuf):
            x = buf[:, row, :, :upto].float()  # [R, H, C, D]
            term = x[-1]
            denom = term.norm(dim=-1).clamp_min(1e-6)  # [H, C]
            dist.append(((x[:-1] - term).norm(dim=-1) / denom).mean((0, 1)))
            cons.append(((x[1:] - x[:-1]).norm(dim=-1) / denom).mean((0, 1)))
    return {"dist_to_terminal": torch.stack(dist).mean(0).cpu(), "consecutive": torch.stack(cons).mean(0).cpu()}


def claim(d: Path, name: str) -> bool:
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.environ.get('CUDA_VISIBLE_DEVICES')} {os.getpid()} {time.time()}\n".encode())
    os.close(fd)
    return True
