"""Independent policy reference: the *native* HF forward with a dictionary cache.

The cache stores every (layer, position, loop) K/V separately and materialises the history for
each update() call directly from the policy definition (no broadcast/finalize trick), so it is an
independent implementation against which kvp.engine is checked. B=1 only.
"""

from __future__ import annotations

import torch


class _DictPolicyStore:
    def __init__(self, policy: str, R: int, region_start: int, keep: frozenset = frozenset(),
                 banks: frozenset = frozenset()):
        self.policy, self.R, self.region_start = policy, R, region_start
        self.keep = keep  # positions that keep their per-loop versions (restoration oracle)
        self.banks = banks  # (layer, loop) banks that keep per-loop versions for every position
        self.entries: dict = {}  # (kind, layer) -> {pos: {loop: (k, v)}}
        self.p0 = 0  # number of tokens fully processed before the current forward call

    def _history_version(self, slots: dict, pos: int, r: int, kind: str, layer: int = -1):
        if kind == "plain":
            return slots[0]
        if (layer, r) in self.banks:
            return slots[r]
        if pos < self.region_start or self.policy == "full" or pos in self.keep:
            return slots[r]
        if self.policy in ("last", "first"):
            return slots[self.R - 1 if self.policy == "last" else 0]
        if self.policy in ("blend25", "blend50", "blend75"):
            alpha = int(self.policy[5:]) / 100.0
            dt = slots[0][0].dtype
            k = (alpha * slots[0][0].float() + (1.0-alpha) * slots[self.R - 1][0].float()).to(dt)
            v = (alpha * slots[0][1].float() + (1.0-alpha) * slots[self.R - 1][1].float()).to(dt)
            return k, v
        ks = torch.stack([slots[i][0] for i in range(self.R)]).double().mean(0)
        vs = torch.stack([slots[i][1] for i in range(self.R)]).double().mean(0)
        dt = slots[0][0].dtype
        return ks.to(dt), vs.to(dt)

    def store_and_read(self, kind: str, layer: int, r: int, k, v):
        """k, v: [1, H, S, D] for positions p0..p0+S-1 at loop r."""
        table = self.entries.setdefault((kind, layer), {})
        S = k.shape[2]
        for i in range(S):
            table.setdefault(self.p0 + i, {})[r] = (k[:, :, i], v[:, :, i])
        hist = [self._history_version(table[p], p, r, kind, layer) for p in range(self.p0)]
        hk = [h[0] for h in hist] + [k[:, :, i] for i in range(S)]
        hv = [h[1] for h in hist] + [v[:, :, i] for i in range(S)]
        return torch.stack(hk, dim=2), torch.stack(hv, dim=2)


class OuroRefCache:
    """Plain object (not a transformers Cache): OuroModel then uses it as-is. Subclassing the
    checkpoint's UniversalTransformerCache is impossible on transformers 4.54.1 (its __init__
    assigns to the read-only `key_cache` property) -- see the Stage-1 report."""

    def __init__(self, store: _DictPolicyStore, n_layers: int):
        self.store, self.n_layers = store, n_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        r, layer = divmod(layer_idx, self.n_layers)
        return self.store.store_and_read("rec", layer, r, key_states, value_states)

    def get_seq_length(self, layer_idx=0):
        return self.store.p0


class HuginnRefCache:
    def __init__(self, store: _DictPolicyStore, n_pre: int, n_core: int):
        self.store, self.n_pre, self.n_core = store, n_pre, n_core

    def update(self, key_states, value_states, step_idx_tensor, lookup_strategy=None):
        idx = int(step_idx_tensor)
        if idx < 0:
            return self.store.store_and_read("plain", 1000 + idx, 0, key_states, value_states)
        if idx < self.n_pre:
            return self.store.store_and_read("plain", idx, 0, key_states, value_states)
        s, j = divmod(idx - self.n_pre, self.n_core)
        return self.store.store_and_read("rec", j, s, key_states, value_states)

    def get_seq_length(self, step_idx=0):
        return self.store.p0


@torch.inference_mode()
def reference_logits(model, family: str, R: int, prompt: list[int], cont: list[int], policy: str,
                     mode: str, noise: torch.Tensor | None = None, keep: frozenset = frozenset(),
                     banks: frozenset = frozenset()) -> torch.Tensor:
    """Native-forward logits for every position of prompt+cont under (policy, mode).
    mode "decode":  prompt in one parallel call, policy region = continuation.
    mode "convert": prompt in one parallel call, policy region = everything (the prompt's per-loop
                    entries are used only inside its own call, later calls read policy versions).
    mode "strict":  every token in its own call, policy region = everything."""
    dev = next(model.parameters()).device
    P = len(prompt)
    store = _DictPolicyStore(policy, R, P if mode == "decode" else 0, keep, banks)
    if family == "ouro":
        cache = OuroRefCache(store, model.config.num_hidden_layers)
    else:
        tr = model.transformer
        cache = HuginnRefCache(store, len(tr.prelude), len(tr.core_block))
    seq = prompt + cont
    calls = [(0, P)] + [(p, p + 1) for p in range(P, len(seq))] if mode in ("decode", "convert") else \
        [(p, p + 1) for p in range(len(seq))]
    outs = []
    for a, b in calls:
        store.p0 = a
        ids = torch.tensor([seq[a:b]], device=dev)
        cpos = torch.arange(a, b, device=dev)
        if family == "ouro":
            out = model(ids, past_key_values=cache, use_cache=True, cache_position=cpos,
                        position_ids=cpos[None], attention_mask={"full_attention": None},
                        exit_at_step=R - 1)
        else:
            out = model(ids, input_states=noise[None, a:b].to(dev, next(model.parameters()).dtype),
                        num_steps=R, past_key_values=cache, use_cache=True, cache_position=cpos)
        outs.append(out.logits[0].float().cpu())
    return torch.cat(outs, 0)
