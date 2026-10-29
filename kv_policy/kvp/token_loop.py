"""Per-token first-recurrence KV protection: a strict LAST reference policy.

Research question: Must EVERY historical token retain recurrence-1 KV to
recover the accuracy of Ouro's first-recurrence-all (50%-FULL) baseline?

For each completed token i, keep the terminal KV of every physical recurrent
layer. Optionally keep K/V at recurrence 1 for that token across ALL layers.
The decision is causal: UID/position/prompt membership or current token's
already computed first-vs-terminal KV distance. Current token self-KV stays at
the currently executing recurrence, as required by the shared-KV contract.

This is a FULL-SIZED BROADCAST *quality reference*, not physical compression.
"""
from __future__ import annotations

import hashlib
import re

import torch

from .engine import PolicyKV


TOKEN_POLICY = re.compile(
    r"tokL1(?:(hash)(0|20|40|60|80|100)(?:s([12]))?"
    r"|(prompt)|(decode)|(delta)(25|50|100))"
)


def parse_token_policy(name: str) -> dict:
    m = TOKEN_POLICY.fullmatch(name)
    if m is None:
        raise ValueError(f"Unsupported token-loop policy: {name}")
    if m.group(1):
        return {"kind": "hash", "rate": int(m.group(2)) / 100.0,
                "seed": int(m.group(3) or 0)}
    if m.group(4):
        return {"kind": "prompt"}
    if m.group(5):
        return {"kind": "decode"}
    return {"kind": "delta", "threshold": int(m.group(7)) / 100.0}


def stable_hash_fraction(uid: str, unpadded_pos: int, seed: int) -> float:
    raw = f"token-firstbank-20261029:{seed}:{uid}:{unpadded_pos}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / float(2**64)


class TokenFirstBankKV(PolicyKV):
    """LAST/strict plus selected recurrence-1 KV for each historical token.

    Each decision is made after ALL R recurrence calls have been performed
    for the token, before that token is read as history by the next token.
    This reference keeps a full-sized broadcast store; logical bytes are
    emitted separately and must never be interpreted as measured GPU HBM.
    """

    def __init__(self, policy: str, R: int, n_rec: int, n_plain: int,
                 B: int, H: int, T: int, D: int, dtype, device, *,
                 uids: list[str], pads: list[int], prompt_end: int):
        if R < 2 or n_plain != 0:
            raise ValueError("TokenFirstBankKV currently validated for Ouro R>=2 only")
        if len(uids) != B or len(pads) != B:
            raise ValueError("Mismatched UID/padding batch metadata")
        super().__init__("last", R, n_rec, n_plain, B, H, T, D, dtype, device, 0)
        self.device = torch.device(device)
        self.token_policy = policy
        self.rule = parse_token_policy(policy)
        self.uids = list(uids)
        self.pads = list(pads)
        self.prompt_end = prompt_end
        self.selected_token_positions: list[tuple[int, int, bool]] = []
        self.n_real_processed = 0
        self.n_selected = 0
        self.n_real_prompt = 0
        self.n_selected_prompt = 0

    def _decision(self, c: int) -> torch.Tensor:
        valid = [c >= p for p in self.pads]
        kind = self.rule["kind"]
        if kind == "hash":
            selected = [
                ok and stable_hash_fraction(uid, c - p, self.rule["seed"]) < self.rule["rate"]
                for ok, uid, p in zip(valid, self.uids, self.pads)
            ]
            return torch.tensor(selected, dtype=torch.bool, device=self.device)
        if kind == "prompt":
            return torch.tensor([ok and c < self.prompt_end for ok in valid],
                                dtype=torch.bool, device=self.device)
        if kind == "decode":
            return torch.tensor([ok and c >= self.prompt_end for ok in valid],
                                dtype=torch.bool, device=self.device)
        # Causal trajectory proxy measured on THIS token, after it completed
        # recurrence R: normalized combined first-vs-terminal K/V difference.
        numer = torch.zeros(len(valid), device=self.device, dtype=torch.float32)
        denom = torch.zeros_like(numer)
        for k, v in zip(self.rk, self.rv):
            for states in (k, v):
                first = states[0, :, :, c, :].float()
                terminal = states[-1, :, :, c, :].float()
                numer += (first - terminal).square().mean(dim=(1, 2))
                denom += terminal.square().mean(dim=(1, 2))
        score = torch.sqrt(numer / denom.clamp_min(1e-12))
        return (score >= self.rule["threshold"]) & torch.tensor(valid, dtype=torch.bool,
                                                                device=self.device)

    def finalize(self, c: int):
        """Only one recurrence-1 bank extra per selected token per physical layer."""
        if self.R < 2 or c < 0:
            return
        keep = self._decision(c)
        valid = [c >= p for p in self.pads]
        decisions = keep.detach().cpu().tolist()
        self.n_real_processed += sum(valid)
        self.n_selected += sum(decisions)
        if c < self.prompt_end:
            self.n_real_prompt += sum(valid)
            self.n_selected_prompt += sum(decisions)
        for b, ok in enumerate(valid):
            if ok:
                self.selected_token_positions.append((b, c - self.pads[b], bool(decisions[b])))
        # Entire recurrence 2..R-1 history is terminal regardless of selection.
        # Save the first recurrence before broadcasting terminal into its slot.
        for kbuf, vbuf in zip(self.rk, self.rv):
            for buf in (kbuf, vbuf):
                first = buf[0, :, :, c, :].clone()
                terminal = buf[self.R - 1, :, :, c, :].clone()
                buf[:self.R - 1, :, :, c, :] = terminal.unsqueeze(0)
                if bool(keep.any()):
                    buf[0, keep, :, c, :] = first[keep]

    def logical_summary(self, output_lengths: list[int] | None = None) -> dict:
        """Count actually needed rows/positions, excluding post-EOS ghost batch rows.

        The existing reference engine continues computing for rows that
        finished earlier; those extra KV planes must NOT be charged to the
        request's logical live cache. Each generated token after the final
        emitted token is not needed as history.
        """
        if output_lengths is None:
            n, p = self.n_real_processed, self.n_selected
            prompt_n, prompt_p = self.n_real_prompt, self.n_selected_prompt
        else:
            if len(output_lengths) != len(self.uids):
                raise ValueError("Expected one output length per batch row")
            n = p = prompt_n = prompt_p = 0
            for b, pos, selected in self.selected_token_positions:
                prompt_len = self.prompt_end - self.pads[b]
                valid_limit = prompt_len + max(0, output_lengths[b] - 1)
                if pos >= valid_limit:
                    continue
                n += 1
                p += int(selected)
                if pos < prompt_len:
                    prompt_n += 1
                    prompt_p += int(selected)
        ratio = ((n + p) / (self.R * n)) if n else None
        return {
            "policy": self.token_policy,
            "n_processed_real_token_positions": n,
            "n_protected_token_positions": p,
            "fraction_tokens_with_first_bank": p / n if n else None,
            "fraction_first_bank_in_prompt": prompt_p / prompt_n
            if prompt_n else None,
            "logical_recurrent_kv_vs_full": ratio,
            "logical_recurrent_kv_vs_loop1all": (n + p) / (2 * n) if n else None,
            "loop1all_saving_fraction": (n - p) / (2 * n) if n else None,
            "num_recurrent_layers": len(self.rk),
            "reference_full_allocated": True,
            "excluded_post_exit_rows": output_lengths is not None,
        }
