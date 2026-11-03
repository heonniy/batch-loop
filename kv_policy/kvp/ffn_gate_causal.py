"""Causal recurrent FFN Gate intervention for the pinned Huginn and Ouro reference runners.

Research-only, not a deployable KV policy. All interventions happen on the
CURRENT generated query, never during the untouched full-depth prompt prefill.
The FULL branch is executed with the SAME last-plane sharing (or FULL-bank)
map as the Evicted branch, so comparisons do not conflate two cache policies.

Architecture:
  OuroMLP: gate_proj(x), up_proj(x), down_proj(SiLU(gate) * up).
  Huginn GatedMLP: fc(x).chunk(2) -> gate, up; proj(SiLU(gate) * up).
Only CORE recurrent MLPs are hooked; no prelude/coda.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .query_conditioned_probe import QueryConditionedRunner

GATE_MODES = ("capture", "restore_gate", "restore_up", "shift_gate", "restore_ffn")
EPS = 1e-8


def _parts(out: torch.Tensor, family: str, component: str):
    """Return the component part and a composition function."""
    if family == "huginn" and component == "fc":
        if out.shape[-1] % 2:
            raise ValueError("Huginn fc output must split into equal gate/up halves")
        n = out.shape[-1] // 2
        return out[..., :n], out[..., n:], n
    return out, None, None


def patch_projection(out: torch.Tensor, clean: torch.Tensor, *, family: str,
                     component: str, mode: str) -> torch.Tensor:
    """Construct an exactly gate-only/up-only/matched-norm intervention.

    shift_gate circularly permutes the clean-evicted gate delta. It matches the
    perturbation L2 norm but not its feature direction (negative control).
    """
    if out.shape != clean.shape:
        raise ValueError("Full/Evicted FFN projections have different shapes")
    if mode == "restore_ffn":
        if component != "ffn":
            raise ValueError("Full FFN restoration requires whole-FFN output")
        return clean.to(out.dtype)
    if mode not in ("restore_gate", "restore_up", "shift_gate"):
        raise ValueError("Unsupported projection intervention")
    if family == "huginn":
        if component != "fc":
            raise ValueError("Huginn patch requires combined GatedMLP.fc")
        a, b, n = _parts(out, family, component)
        ac, bc = clean[..., :n], clean[..., n:]
        if mode == "restore_gate":
            return torch.cat((ac.to(out.dtype), b), dim=-1)
        if mode == "restore_up":
            return torch.cat((a, bc.to(out.dtype)), dim=-1)
        delta = ac.to(out.dtype) - a
        return torch.cat((a + torch.roll(delta, shifts=max(1, n//3), dims=-1), b), dim=-1)
    if component != ("up" if mode == "restore_up" else "gate"):
        raise ValueError("Ouro patch targets gate_proj or up_proj explicitly")
    if mode == "shift_gate":
        delta = clean.to(out.dtype) - out
        return out + torch.roll(delta, shifts=max(1, out.shape[-1]//3), dims=-1)
    return clean.to(out.dtype)


class RecurrentFFNHooks:
    """Hooks on the MODEL's actual recurrent FFNs; shared weights are indexed by call order.

    The runner invokes begin_column() before every current-token forward and
    end_column(depth) afterward. Each recurrent layer is called exactly once
    per executed loop. No hook is attached to an ordinary/prelude/coda layer.

    A captured trace key is (generated_column, 1-based_loop, physical_core_layer, component).
    Reference tensors are private, immutable copies with detached values.
    """
    def __init__(self, bb, *, mode: str, reference: Mapping | None = None,
                 selected_loops: Sequence[int] | None = None,
                 selected_columns: Sequence[int] | None = None,
                 selected_layers: Sequence[int] | None = None):
        if mode not in GATE_MODES:
            raise ValueError(mode)
        self.bb, self.mode = bb, mode
        self.family, self.R = bb.family, bb.R
        self.n_layers = bb.n_rec
        self.reference = reference or {}
        self.loops = None if selected_loops is None else frozenset(int(r) for r in selected_loops)
        self.cols = None if selected_columns is None else frozenset(int(c) for c in selected_columns)
        self.layers = None if selected_layers is None else frozenset(int(l) for l in selected_layers)
        if self.loops is not None and (not self.loops or min(self.loops) < 1 or max(self.loops) > self.R):
            raise ValueError("Invalid selected loop set")
        if self.layers is not None and (not self.layers or min(self.layers) < 0 or max(self.layers) >= self.n_layers):
            raise ValueError("Invalid selected physical layer set")
        self.trace: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self.stats: dict[str, list[float]] = defaultdict(list)
        self.calls: dict[tuple[int, str], int] = {}
        self.col: int | None = None
        self.patched = 0
        self.handles = []

    def _module_specs(self):
        core = (self.bb.model.model.layers if self.family == "ouro"
                else self.bb.model.transformer.core_block)
        if len(core) != self.n_layers:
            raise RuntimeError("Recurrent layer count differs from backbone")
        for l, layer in enumerate(core):
            mlp = layer.mlp
            if self.family == "ouro":
                for name, attr in (("gate", "gate_proj"), ("up", "up_proj")):
                    if not hasattr(mlp, attr):
                        raise RuntimeError(f"Ouro MLP lacks {attr}")
                    yield l, name, getattr(mlp, attr)
            elif self.family == "huginn":
                if not hasattr(mlp, "fc"):
                    raise RuntimeError("Huginn MLP lacks combined fc")
                yield l, "fc", mlp.fc
            else:
                raise ValueError(f"Unsupported family {self.family}")
            yield l, "ffn", mlp

    def begin_column(self, col: int):
        if self.col is not None:
            raise RuntimeError("Unclosed previous token in FFN hook")
        self.col, self.calls = int(col), {}

    def end_column(self, depth: int):
        if self.col is None:
            raise RuntimeError("Hook column not initialized")
        for l in range(self.n_layers):
            for c in (("gate", "up", "ffn") if self.family == "ouro" else ("fc", "ffn")):
                if self.calls.get((l, c), 0) != depth:
                    raise RuntimeError(f"Gate trace layer/call mismatch: {l}/{c}: "
                                       f"{self.calls.get((l,c),0)} != {depth}")
        self.col = None

    def _hook(self, layer: int, component: str):
        def handler(module, args, output):
            if self.col is None:
                raise RuntimeError("Gate hook was invoked without an active query token")
            r = self.calls.get((layer, component), 0) + 1
            self.calls[(layer, component)] = r
            if r > self.R:
                raise RuntimeError("More recurrent calls than configured")
            if ((self.cols is not None and self.col not in self.cols) or
                (self.loops is not None and r not in self.loops) or
                (self.layers is not None and layer not in self.layers)):
                return output
            key = (self.col, r, layer, component)
            if self.mode == "capture":
                self.trace[key] = output.detach().clone()
                return output
            action = (("ffn" if self.mode == "restore_ffn" else
                       ("fc" if self.family == "huginn" else
                        ("up" if self.mode == "restore_up" else "gate"))))
            if component != action:
                return output
            if key not in self.reference:
                raise KeyError(f"No full reference FFN activation for {key}")
            clean = self.reference[key].to(output.device, output.dtype)
            original = output.detach()
            changed = patch_projection(output, clean, family=self.family,
                                       component=component, mode=self.mode)
            self.patched += 1
            delta = (changed.detach().float() - original.float()).norm()
            denom = original.float().norm() + EPS
            self.stats["relative_patch_norm"].append(float((delta / denom).item()))
            return changed
        return handler

    @contextmanager
    def install(self):
        if self.handles:
            raise RuntimeError("Nested FFN hooks not supported")
        try:
            for layer, name, module in self._module_specs():
                self.handles.append(module.register_forward_hook(self._hook(layer, name)))
            yield self
            if self.col is not None:
                raise RuntimeError("Gate hook context left an unfinished column")
        finally:
            for h in self.handles:
                h.remove()
            self.handles.clear()
            self.col = None


class GateCausalRunner(QueryConditionedRunner):
    """Pairs FULL/EVICT/RESTORE on the same frozen target IDs and historical KV policy."""
    def __init__(self, *args, **kwargs):
        self._ffn_hooks: RecurrentFFNHooks | None = None
        self._lp_capture: list[torch.Tensor] | None = None
        self._exit_collect: list[dict] | None = None
        super().__init__(*args, **kwargs)

    def _step_dyn(self, col, tid, spec, forced):
        tracker = self._ffn_hooks
        if tracker is not None:
            tracker.begin_column(col)
        logits, depth = super()._step_dyn(col, tid, spec, forced)
        if tracker is not None:
            tracker.end_column(depth)
        if self._lp_capture is not None:
            self._lp_capture.append(F.log_softmax(logits[0, 0].float(), -1).detach().cpu())
        if self._exit_collect is not None:
            if depth != self.R:
                raise RuntimeError("Clean exit calibration needs complete fixed-depth traces")
            self._exit_collect.append(self.exit_stats(self.cap.take()))
        return logits, depth

    @torch.no_grad()
    def run_gate_arm(self, *, sources: Sequence[int], spec,
                     dynamic: bool = False, forced_depths: Sequence[int] | None = None,
                     plane_map=None, mode: str = "capture", reference: Mapping | None = None,
                     selected_loops: Sequence[int] | None = None,
                     selected_columns: Sequence[int] | None = None,
                     selected_layers: Sequence[int] | None = None) -> dict:
        if self._ffn_hooks is not None or self._lp_capture is not None:
            raise RuntimeError("Overlapping gate arms not allowed")
        h = RecurrentFFNHooks(self.bb, mode=mode, reference=reference,
                              selected_loops=selected_loops,
                              selected_columns=selected_columns,
                              selected_layers=selected_layers)
        self._ffn_hooks = h
        self._lp_capture = []
        try:
            with h.install():
                result = self.decode(sources, spec, dynamic=dynamic, forced_depths=forced_depths,
                                     plane_map=plane_map)
            lps = list(self._lp_capture)
        finally:
            self._ffn_hooks = None
            self._lp_capture = None
        if len(lps) != len(self.ref_ids):
            raise RuntimeError("Missing full output-distribution trace")
        return {**result, "logprobs": lps, "ffn_trace": h.trace,
                "n_patched_components": h.patched, "patch_stats": dict(h.stats),
                "ffn_mode": mode, "selected_loops": list(selected_loops) if selected_loops else None}

    @torch.no_grad()
    def clean_last_exit_observations(self, plane_map) -> list[dict]:
        """Gather CLEAN full-depth per-query native exit signals with the actual
        LAST-sharing cache read semantics. Thresholds are tuned only from these
        untouched observations on disjoint calibration prompts."""
        if self._exit_collect is not None or self._ffn_hooks is not None:
            raise RuntimeError("Nested exit calibration not allowed")
        self._exit_collect = []
        try:
            self.decode((), self.spec, plane_map=plane_map)
            observations = list(self._exit_collect)
        finally:
            self._exit_collect = None
        if len(observations) != len(self.ref_ids):
            raise RuntimeError("Incomplete clean LAST-sharing exit measurements")
        return observations

    @torch.no_grad()
    def greedy_dynamic(self, *, sources: Sequence[int], spec, max_new: int,
                       eos_ids: set[int], plane_map=None) -> dict:
        """True live exit and true free token branching, not recorded replay.

        This is intentionally NOT an FFN-restoration experiment: after a
        generated token differs, a clean-gold gate trace would no longer be
        aligned. The causal restoration is evaluated on frozen gold targets.
        """
        if self._ffn_hooks is not None:
            raise RuntimeError("Do not apply mismatched gold gate traces during free generation")
        if max_new < 2 or self.P + max_new > self.T:
            raise ValueError("Greedy horizon exceeds cache")
        src = frozenset(int(s) for s in sources)
        self._mask = (src, self.P, self.P + max_new, None, True) if src else None
        self._reset_decode(0)
        self._set_planes(plane_map)
        token_ids = [int(self.last_prompt_lp.argmax())]
        depths = []
        try:
            if token_ids[0] not in eos_ids:
                for i in range(1, max_new):
                    logits, depth = self._step_dyn(self.P + i - 1, token_ids[-1], spec, None)
                    depths.append(int(depth))
                    nxt = int(logits[0, 0].float().argmax())
                    token_ids.append(nxt)
                    if nxt in eos_ids:
                        break
        finally:
            self._mask = None
            self._set_planes(None)
            self._restore()
            self._reset_decode(0)
        return {"token_ids": token_ids, "depths": depths, "total_depth": sum(depths)}


@torch.no_grad()
def paired_logit_kl(clean: Sequence[torch.Tensor], altered: Sequence[torch.Tensor],
                    *, exclude_last_unscored: bool = True) -> dict:
    if len(clean) != len(altered) or not clean:
        raise ValueError("Unpaired logprob sequences")
    vals = []
    for c, a in zip(clean, altered):
        c, a = c.float(), a.float()
        if c.shape != a.shape:
            raise ValueError("Full and altered vocabulary logits differ")
        vals.append(float((c.exp() * (c - a)).sum()))
    kept = vals[:-1] if exclude_last_unscored else vals
    return {"kl_by_query": vals, "mean_future_kl": sum(kept)/len(kept) if kept else None}


def nll_recovery(full: float, evict: float, restore: float, tol: float = 1e-8) -> dict:
    """Positive => patch helps. Ratio reported only if eviction causes actual harm."""
    damage = evict - full
    repaired = evict - restore
    return {"eviction_penalty": damage, "nll_recovered": repaired,
            "fraction_recovered": (repaired / damage if damage > tol else None)}


@torch.no_grad()
def gate_trace_comparison(clean: Mapping, evicted: Mapping, family: str) -> dict:
    """Query×loop-resolved FFN internal response to EXACTLY the same source mask.

    Gate is pre-SiLU. Sign flips are a descriptive channel-activation
    diagnostic, NOT a sparse routing decision. Full versus Evict histories
    share the exact same token IDs and number of recurrences.
    """
    groups: dict[int,list[dict]] = defaultdict(list)
    for (col,r,l,component), a in clean.items():
        if component not in (("fc", "ffn") if family=="huginn" else ("gate", "ffn")):
            continue
        key=(col,r,l,component)
        if key not in evicted:
            raise RuntimeError(f"Missing evicted FFN trace {key}")
        b=evicted[key].to(a.device)
        if a.shape!=b.shape:
            raise RuntimeError(f"FFN activation shape mismatch {key}")
        if component=="fc":
            a=a[...,:a.shape[-1]//2]
            b=b[...,:b.shape[-1]//2]
        af,bf=a.float(),b.float()
        diff=(bf-af).norm()
        denom=af.norm()+EPS
        row={"col":col,"r":r,"layer":l,"component":"ffn" if component=="ffn" else "gate",
             "relative_change":float((diff/denom).item()),
             "abs_change":float(diff.item()),
             "sign_flip_frac":float(((af>0)!=(bf>0)).float().mean().item())}
        groups[r].append(row)
    by_loop=[]
    for r,rows in sorted(groups.items()):
        d={"loop":r,"n_observations":len(rows)}
        for cmp in ("gate","ffn"):
            rr=[x for x in rows if x["component"]==cmp]
            if rr:
                d[cmp+"_mean_relative_change"]=sum(x["relative_change"] for x in rr)/len(rr)
                d[cmp+"_mean_sign_flip_frac"]=sum(x["sign_flip_frac"] for x in rr)/len(rr)
                d[cmp+"_max_relative_change"]=max(x["relative_change"] for x in rr)
        by_loop.append(d)
    return {"by_loop":by_loop,"total_positions":sum(len(x) for x in groups.values()),
            "note":"Pre-SiLU gate sign changes are continuous-MLP features, not MoE expert switches"}
