"""LoopZip v6: memory-bounded recurrent-trajectory metrics (LOOPZIP_PERSISTENT_RECURRENCE_TRAJECTORY_PLAN.md).

States h_(t,0..R) are captured at the SAME observer checkpoint in every branch:
- Ouro: h_0 = token embedding, h_r = output of ``model.norm`` after full UT recurrence r (the
  model-native normalized state that feeds loop r+1 and the LM head at r=R).
- Huginn: h_0 = deterministic per-(UID, position) initial latent, h_r = core-block output after
  recurrence r (before ``ln_f``/coda). Prelude output is reported only as a norm summary.
All metrics are computed on GPU in fp32 and moved to CPU as small per-token vectors.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

EPS = 1e-8


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + EPS)


@torch.no_grad()
def trajectory_metrics(H: torch.Tensor) -> dict[str, torch.Tensor]:
    """H: (R+1, d) fp32 states h_0..h_R of ONE query token. Returns length-(R+1) vectors.

    Index r follows the plan: U[r] (r>=1), C[r] (1<=r<R) and are NaN where undefined.
    U_t(r) = ||h_r-h_{r-1}|| / ||h_{r-1}||; F_t(r) = ||h_r-h_R|| / ||h_R||;
    C_t(r) = cos(h_r-h_{r-1}, h_{r+1}-h_r).
    """
    H = H.float()
    R = H.shape[0] - 1
    nan = torch.full((1,), float("nan"), device=H.device)
    norm = H.norm(dim=-1)
    d = H[1:] - H[:-1]                                     # (R, d): update into loop r
    U = torch.cat([nan, d.norm(dim=-1) / (norm[:-1] + EPS)])
    Fd = (H - H[-1]).norm(dim=-1) / (norm[-1] + EPS)
    C = torch.cat([nan, _cos(d[:-1], d[1:]), nan]) if R >= 2 else torch.full((R + 1,), float("nan"), device=H.device)
    return {"h_norm": norm, "U": U, "F": Fd, "C": C,
            "cos_init": _cos(H, H[0].expand_as(H)), "cos_term": _cos(H, H[-1].expand_as(H))}


@torch.no_grad()
def paired_metrics(Hev: torch.Tensor, Hfull: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same-token Full vs Evicted at every r: E (relative state error), U_ev, update-direction cosine."""
    Hev, Hfull = Hev.float(), Hfull.float()
    nan = torch.full((1,), float("nan"), device=Hev.device)
    E = (Hev - Hfull).norm(dim=-1) / (Hfull.norm(dim=-1) + EPS)
    dev, dfu = Hev[1:] - Hev[:-1], Hfull[1:] - Hfull[:-1]
    U_ev = torch.cat([nan, dev.norm(dim=-1) / (Hev[:-1].norm(dim=-1) + EPS)])
    dircos = torch.cat([nan, _cos(dev, dfu)])
    return {"E": E, "U_ev": U_ev, "dircos": dircos}


@torch.no_grad()
def logit_metrics(lp_full: torch.Tensor, logits_ev: torch.Tensor, next_id: int | None) -> dict[str, float]:
    """fp32 KL(p_full || p_ev), max |logit diff| (on log-probs), top-1 change, entropy shift, NLL shift."""
    lp_ev = F.log_softmax(logits_ev.float().reshape(-1), -1)
    lp_full = lp_full.float().reshape(-1)
    p_full = lp_full.exp()
    kl = float((p_full * (lp_full - lp_ev)).sum())
    ent_f = float(-(p_full * lp_full).sum())
    ent_e = float(-(lp_ev.exp() * lp_ev).sum())
    out = {"kl": kl, "max_logprob_diff": float((lp_full - lp_ev).abs().max()),
           "top1_changed": bool(lp_full.argmax() != lp_ev.argmax()), "entropy_shift": ent_e - ent_f,
           "nll_shift": None}
    if next_id is not None:
        out["nll_shift"] = float(lp_full[next_id] - lp_ev[next_id])
    return out


class StateCapture:
    """Installs observe_initial/observe_state on a KV object; holds h_0..h_R of the current column."""

    def __init__(self, kv, R: int):
        self.buf = None
        self.seen = [False] * (R + 1)
        self.prelude_norm = None
        self.R = R
        kv.observe_initial = self._initial
        kv.observe_state = self._state
        kv.observe_prelude = self._prelude

    def _initial(self, col, h):
        self.seen = [False] * (self.R + 1)
        if self.buf is None:
            self.buf = torch.zeros(self.R + 1, h.numel(), dtype=torch.float32, device=h.device)
        self.buf[0] = h.reshape(-1).float()
        self.seen[0] = True

    def _state(self, col, r, h):
        self.buf[r] = h.reshape(-1).float()
        self.seen[r] = True

    def _prelude(self, col, e):
        self.prelude_norm = e.float().norm()

    def take(self) -> torch.Tensor:
        if not all(self.seen):
            raise RuntimeError("Observer missed a recurrence checkpoint (fixed depth expected)")
        return self.buf.clone()
