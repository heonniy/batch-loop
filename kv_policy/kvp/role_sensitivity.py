"""B1 causal Content -> Instruction recurrent sensitivity and KV utility probe.

Two DIFFERENT interventions:
  (1) Instruction-phase counterfactual: hide an earlier Content KV only
      while processing Instruction/Question tokens in recurrent layers.
      This changes instruction hidden states and is NOT a deployed
      post-prefill eviction policy.
  (2) Realistic post-prefill eviction: run the ENTIRE prompt untouched,
      then hide a Content KV while scoring later generated tokens.

The same Full-depth prompt and frozen decode exit trace are used for
paired comparison. All caches here are FULL-sized broadcast references;
neither physical eviction nor H200 saved memory is claimed.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .depth_eviction_probe import DepthOracleKV, DepthProbeSpec, _ouro_one, _huginn_one
from .native_exit import recorded_trace, replay_depth
from .role_data import RolePrompt


class RoleProbeKV(DepthOracleKV):
    def __init__(self, bb, n_tokens, device, dtype, *,
                 prompt: RolePrompt, mode: str = "none", source: int | None = None,
                 perturbed_loop: int = 1, observe: bool = False,
                 tracked_positions: Sequence[int] = ()):
        if mode not in ("none", "instruction_all", "instruction_one_loop",
                        "postprefill"):
            raise ValueError("Unsupported role perturbation")
        allowed = set(prompt.content_positions)
        if mode == "postprefill":
            allowed.update(prompt.instruction_positions)
            allowed.update(prompt.distractor_positions)
        if mode != "none" and (source is None or source not in allowed):
            raise ValueError("Selected source is not in an allowed causal role span")
        if perturbed_loop < 1 or perturbed_loop > bb.R:
            raise ValueError("Perturbation loop out of range")
        self.prompt = prompt
        self.mode = mode
        self.source = source
        self.perturbed_loop = perturbed_loop
        self.observe = observe
        self.instruction_set = frozenset(prompt.instruction_positions)
        self.tracked = frozenset(tracked_positions)
        if len(prompt.token_ids) >= n_tokens:
            raise ValueError("Need at least two future target tokens")
        super().__init__(bb, n_tokens, device, dtype,
                         evicted=source if mode == "postprefill" else None,
                         evaluation_start=len(prompt.token_ids),
                         collect_attention=False)
        self.mass = torch.zeros(bb.R, 5, dtype=torch.float32, device=device)
        # content attention per content key, summed across instruction queries
        # and recurrent physical attention layers/heads.
        self.content_attention = torch.zeros(bb.R, len(prompt.token_ids),
                                             dtype=torch.float32, device=device)
        self.state_history: dict[int, dict[int, torch.Tensor]] = {}
        self.state_initial: dict[int, torch.Tensor] = {}
        self.prelude_state: dict[int, torch.Tensor] = {}

    def observe_initial(self, col: int, h: torch.Tensor):
        if col in self.tracked:
            self.state_initial[col] = h.detach().float().flatten().cpu().clone()

    def observe_prelude(self, col: int, h: torch.Tensor):
        if col in self.tracked:
            self.prelude_state[col] = h.detach().float().flatten().cpu().clone()

    def observe_state(self, col: int, depth: int, h: torch.Tensor):
        if col in self.tracked:
            self.state_history.setdefault(col, {})[depth] = (
                h.detach().float().flatten().cpu().clone())

    def _should_mask(self, layer, col: int) -> bool:
        if self.mode == "instruction_all":
            return col in self.instruction_set and layer[0] == "rec"
        if self.mode == "instruction_one_loop":
            return (col in self.instruction_set and layer[0] == "rec"
                    and layer[2]+1 == self.perturbed_loop)
        return False

    def attend(self, layer, q, k, v, c0, c1, mask):
        if c1 != c0+1:
            raise ValueError("Token-major reference only")
        if self.observe and layer[0] == "rec" and c0 in self.instruction_set:
            # Measure ORIGINAL attention before any intervention.
            kb, _ = self._buffers(layer)
            keys = kb[:, :, :c1].clone()
            keys[:, :, c0:c1] = k
            w = torch.softmax(
                (q.float() @ keys.float().transpose(-2,-1) /
                 math.sqrt(q.shape[-1])).masked_fill(~mask, float("-inf")),
                dim=-1)[0, :, 0, :].mean(dim=0)
            r = layer[2]
            roles = self.prompt.roles
            content = [i for i in self.prompt.content_positions if i < c0]
            distractor = [i for i in self.prompt.distractor_positions if i < c0]
            instruction = [i for i in self.prompt.instruction_positions if i < c0]
            self.mass[r,0] += w[content].sum() if content else 0
            self.mass[r,1] += w[instruction].sum() if instruction else 0
            self.mass[r,2] += w[distractor].sum() if distractor else 0
            self.mass[r,3] += w[c0]
            self.mass[r,4] += 1
            if content:
                self.content_attention[r,content] += w[content]
        if self._should_mask(layer,c0):
            mask = mask.clone()
            mask[...,self.source] = False
        return super().attend(layer,q,k,v,c0,c1,mask)

    def observed_attention(self) -> dict:
        mass = self.mass.detach().cpu().tolist()
        matrix = self.content_attention.detach().cpu()
        rows = []
        for i,m in enumerate(mass):
            n = m[4]
            rows.append({"loop":i+1,"n_query_layer_observations":int(n),
                         "content_mass":m[0]/n if n else None,
                         "instruction_previous_mass":m[1]/n if n else None,
                         "distractor_mass":m[2]/n if n else None,
                         "self_mass":m[3]/n if n else None,
                         "content_mass_per_token":m[0]/n/len(self.prompt.content_positions)
                         if n and self.prompt.content_positions else None})
        by_source = {str(p): [float(matrix[r,p])/mass[r][4]
                              if mass[r][4] else None
                              for r in range(len(mass))]
                     for p in self.prompt.content_positions}
        return {"loop_roles":rows,"content_attention_by_source":by_source}


def _norm_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(a-b) /
            torch.linalg.vector_norm(b).clamp_min(1e-9)).item()


def _trajectory_stats(observed: dict[int,torch.Tensor],
                      baseline: dict[int,torch.Tensor], max_depth: int) -> dict:
    if any(r not in observed or r not in baseline for r in range(1,max_depth+1)):
        raise RuntimeError("Missing required instruction-state loops")
    errs = [_norm_delta(observed[r],baseline[r]) for r in range(1,max_depth+1)]
    return {"hidden_error_by_loop":errs,
            "first_error":errs[0],"final_error":errs[-1],
            "final_over_first":errs[-1]/max(1e-9,errs[0])}


@torch.no_grad()
def role_forward(bb, uid: str, spec: DepthProbeSpec, prompt: RolePrompt,
                 target_ids: Sequence[int], *, decode_mode: str = "dynamic",
                 forced_depths: Sequence[int] | None = None,
                 role_mode: str = "none", source: int | None = None,
                 selected_loop: int = 1, observe: bool = False,
                 tracked_positions: Sequence[int] = ()) -> dict:
    """Full-depth B1 *prompt*; fixed or native-dynamic decoded gold targets.

    Unlike the earlier token-major DYNAMIC-prefill runner, all Instruction
    and Content positions execute R loops. In dynamic decode, only generated
    positions may use the exit gate. A frozen trace forces the *same*
    per-token depths in baseline and counterfactual.
    """
    if decode_mode not in {"fixed", "dynamic"}:
        raise ValueError("Invalid decode mode")
    if spec.family != bb.family or spec.max_depth != bb.R:
        raise ValueError("Backbone/exit mismatch")
    if len(target_ids) < 2:
        raise ValueError("Need >=2 gold future tokens for postprefill eviction")
    ids = list(prompt.token_ids) + list(target_ids)
    P = len(prompt.token_ids)
    if forced_depths is not None:
        if len(forced_depths) != len(ids)-1:
            raise ValueError("Recorded depth/token length mismatch")
        if any(int(x)!=bb.R for x in forced_depths[:P]):
            raise ValueError("Full-depth Prefill must be preserved")
    params = next(bb.model.parameters())
    kv = RoleProbeKV(bb, len(ids)-1, params.device, params.dtype,
                     prompt=prompt, mode=role_mode, source=source,
                     perturbed_loop=selected_loop, observe=observe,
                     tracked_positions=tracked_positions)
    aux = bb.make_aux([uid], [0], len(ids)-1) if bb.family == "huginn" else None
    losses = []
    for col,tid in enumerate(ids[:-1]):
        one = torch.tensor([[int(tid)]],dtype=torch.long,device=params.device)
        pos = torch.tensor([[col]],dtype=torch.long,device=params.device)
        mask = torch.ones(1,1,1,col+1,dtype=torch.bool,device=params.device)
        forced = (int(forced_depths[col]) if forced_depths is not None else
                  (bb.R if col < P or decode_mode == "fixed" else None))
        if bb.family=="ouro":
            logits,dep=_ouro_one(bb,one,pos,kv,col,mask,forced,spec)
        else:
            logits,dep=_huginn_one(bb,one,pos,kv,aux,col,mask,forced,spec)
        kv.finish(col,dep)
        if col >= P-1:
            lp=F.log_softmax(logits[0,0].float(),dim=-1)
            losses.append(float(-lp[int(ids[col+1])].item()))
    if len(losses)!=len(target_ids):
        raise RuntimeError("Gold target scoring misaligned")
    return {"nll_all":sum(losses)/len(losses),
            # The first target is predicted by the last prompt position
            # BEFORE post-prefill eviction; exclude it from the future label.
            "nll_future":sum(losses[1:])/len(losses[1:]),
            "first_target_nll":losses[0],
            "token_losses":losses,
            "depths":kv.depths,
            "state_history":kv.state_history,
            "state_initial":kv.state_initial,
            "prelude_state":kv.prelude_state,
            "attention":kv.observed_attention() if observe else None,
            "actual_recurrence_calls":kv.actual_calls}


@torch.no_grad()
def role_intervention_oracle(bb, uid: str, model_revision: str,
                             spec: DepthProbeSpec, prompt: RolePrompt,
                             target_ids: Sequence[int], candidates: Sequence[int],
                             *, decode_mode: str = "dynamic",
                             run_one_loop: bool = True,
                             live_exit: bool = True,
                             role_controls: Sequence[int] = (),
                             replay_tol: float = 1e-4) -> dict:
    """Paired single-Content interventions, independently under frozen state.

    All intervention observations are counterfactual oracles; they are
    NOT features available online before making eviction decisions.
    """
    candidates = tuple(candidates)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidates")
    if any(i not in prompt.content_positions for i in candidates):
        raise ValueError("Non-Content candidate")
    anchor = max(prompt.instruction_positions)
    tracked = (anchor,*candidates)
    common=dict(bb=bb,uid=uid,spec=spec,prompt=prompt,
                target_ids=target_ids,decode_mode=decode_mode)
    baseline=role_forward(**common,observe=True,
                          tracked_positions=tracked)
    depths=baseline["depths"]
    trace=recorded_trace(model_revision,spec.rule(),uid,
                         list(prompt.token_ids)+list(target_ids[:-1]),
                         depths,bb.R,policy="full-depth-prefill-"+decode_mode+"-decode")
    # Replay equality is required BEFORE any causal comparison.
    frozen=[replay_depth(trace,i,model_revision,uid)
            for i in range(len(depths))]
    replay=role_forward(**common,forced_depths=frozen)
    diff=max(abs(a-b) for a,b in zip(baseline["token_losses"],replay["token_losses"]))
    if diff>replay_tol:
        raise RuntimeError(f"Record/replay NLL parity failed: {diff}")
    first = baseline["state_history"].get(anchor,{})
    if len(first)!=bb.R:
        raise RuntimeError("Instruction anchor missing Full-prefill state trajectory")
    attn=baseline["attention"]["content_attention_by_source"]
    role_rows=baseline["attention"]["loop_roles"]
    rows=[]
    for source in candidates:
        prefill_all=role_forward(**common,forced_depths=frozen,
                                role_mode="instruction_all",source=source,
                                tracked_positions=(anchor,))
        persistent=_trajectory_stats(prefill_all["state_history"][anchor],
                                     first,bb.R)
        one=None
        if run_one_loop:
            prefill_once=role_forward(**common,forced_depths=frozen,
                                     role_mode="instruction_one_loop",
                                     source=source,selected_loop=1,
                                     tracked_positions=(anchor,))
            one=_trajectory_stats(prefill_once["state_history"][anchor],
                                  first,bb.R)
        post=role_forward(**common,forced_depths=frozen,
                          role_mode="postprefill",source=source)
        live=None
        if live_exit and decode_mode=="dynamic":
            live=role_forward(**common,role_mode="postprefill",source=source)
        base_decode=frozen[len(prompt.token_ids):]
        alt_decode=live["depths"][len(prompt.token_ids):] if live else []
        source_traj=baseline["state_history"].get(source,{})
        source_chg=[]
        for r in range(2,bb.R+1):
            if r in source_traj and r-1 in source_traj:
                source_chg.append(_norm_delta(source_traj[r],source_traj[r-1]))
        attention=attn[str(source)]
        rows.append({
            "uid":uid,"source_token_position":source,
            "source_role":"content",
            "attention_by_loop":attention,
            "attention_mean":sum(v for v in attention if v is not None) /
                max(1,sum(v is not None for v in attention)),
            "attention_max":max(v for v in attention if v is not None),
            "source_mean_recurrent_change":sum(source_chg)/len(source_chg)
                if source_chg else None,
            "prefill_instruction_error_by_loop":persistent["hidden_error_by_loop"],
            "prefill_persistent_final_error":persistent["final_error"],
            "prefill_persistent_recovery_ratio":persistent["final_over_first"],
            "prefill_one_loop_error_by_loop":
                one["hidden_error_by_loop"] if one else None,
            "prefill_one_loop_final_error":one["final_error"] if one else None,
            "postprefill_eviction_delta_future_nll":
                post["nll_future"]-baseline["nll_future"],
            "postprefill_eviction_delta_all_nll":
                post["nll_all"]-baseline["nll_all"],
            "live_exit_flip_fraction":
                (sum(a!=b for a,b in zip(base_decode,alt_decode))/len(base_decode))
                if alt_decode else None,
            "live_exit_signed_depth_shift":
                (sum(a-b for a,b in zip(alt_decode,base_decode))/len(base_decode))
                if alt_decode else None,
            "live_exit_delta_future_nll":
                (live["nll_future"]-baseline["nll_future"]) if live else None,
        })
    # Role-specific negative/comparator controls: Instruction and unrelated
    # Background tokens can be evicted AFTER prefill, but they are NOT
    # interpreted as Content-to-Instruction interventions.
    controls=[]
    for source in role_controls:
        if source in candidates or source not in set(prompt.instruction_positions) | set(prompt.distractor_positions):
            raise ValueError("Role comparator must be a distinct Instruction/Distractor position")
        post=role_forward(**common,forced_depths=frozen,
                          role_mode="postprefill",source=source)
        role=prompt.roles[source]
        controls.append({
            "uid":uid,"source_token_position":source,"source_role":role,
            "postprefill_eviction_delta_future_nll":
                post["nll_future"]-baseline["nll_future"]})
    return {
        "role_control_rows":controls,
        "trace":trace,
        "baseline":{"nll_all":baseline["nll_all"],
                    "nll_future":baseline["nll_future"],
                    "replay_max_abs_token_nll_delta":diff,
                    "decode_mode":decode_mode,
                    "full_prefill_depth":bb.R,
                    "attention_role_mass_by_loop":role_rows,
                    "source_roles_from_real_token_offsets":True,
                    "full_sized_reference_kv":True},
        "candidate_rows":rows}
