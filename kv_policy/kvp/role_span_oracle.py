"""Paired, span-level causal interventions for evidence-vs-distractor questions.

A masks earlier Context KV ONLY while processing later Instruction/Question.
B runs an untouched FULL prefill, masks the same KV window ONLY for future
gold target queries, then scores future NLL and recurrent-state errors.
These operations do NOT physically reclaim HBM.
"""
from __future__ import annotations

from typing import Sequence

from .role_data import RolePrompt
from .role_sensitivity import role_forward, _trajectory_stats, _norm_delta
from .depth_eviction_probe import DepthProbeSpec
from .native_exit import recorded_trace, replay_depth


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def role_evidence_oracle(bb, uid: str, model_revision: str, spec: DepthProbeSpec,
                         prompt: RolePrompt, target_ids: Sequence[int],
                         candidates: Sequence[dict], *, decode_mode="fixed",
                         run_one_loop=True, live_exit=True,
                         question_control=True, replay_tol=1e-4,
                         tokenizer=None, gold_answer=None,
                         max_generation_tokens=32) -> dict:
    if not candidates:
        raise ValueError("At least one labeled Context candidate required")
    ids = set()
    width = None
    for c in candidates:
        locs = tuple(c["positions"])
        if not locs or len(locs) != len(set(locs)):
            raise ValueError("Invalid evidence KV span")
        if c["id"] in ids:
            raise ValueError("Repeated evidence candidate ID")
        ids.add(c["id"])
        if c["role"] not in ("relevant_content", "irrelevant_content"):
            raise ValueError("Candidate must have a ground-truth evidence label")
        if any(p not in prompt.content_positions or
               prompt.roles[p] != c["role"] for p in locs):
            raise ValueError("Candidate crosses a role boundary")
        if width is None:
            width = len(locs)
        if len(locs) != width:
            raise ValueError("Evidence/distractor interventions must mask identical KV-token counts")
    if len({c["role"] for c in candidates}) != 2:
        raise ValueError("Paired evidence and distractor probes are mandatory")

    anchor = max(prompt.question_positions) if prompt.question_positions else max(prompt.instruction_positions)
    P = len(prompt.token_ids)
    future_positions = tuple(range(P, P + min(4, len(target_ids)-1)))
    tracked = tuple(sorted({anchor, *future_positions,
                            *(p for c in candidates for p in c["positions"])}))
    common = dict(bb=bb, uid=uid, spec=spec, prompt=prompt,
                  target_ids=target_ids, decode_mode=decode_mode)
    clean = role_forward(**common, observe=True, tracked_positions=tracked)
    trace = recorded_trace(model_revision, spec.rule(), uid,
                           list(prompt.token_ids)+list(target_ids[:-1]),
                           clean["depths"], bb.R,
                           policy="full-depth-prefill-"+decode_mode+"-decode")
    frozen = [replay_depth(trace, i, model_revision, uid)
              for i in range(len(clean["depths"]))]
    replay = role_forward(**common, forced_depths=frozen)
    parity = max(abs(x-y) for x, y in zip(clean["token_losses"], replay["token_losses"]))
    if parity > replay_tol:
        raise RuntimeError(f"Clean recorded-replay NLL parity failed: {parity}")
    clean_anchor = clean["state_history"].get(anchor, {})
    if len(clean_anchor) != bb.R:
        raise RuntimeError("Missing complete question anchor trajectory")
    clean_attention = clean["attention"]["content_attention_by_source"]
    base_decode = frozen[P:]
    baseline_greedy = None
    baseline_score = None
    eos = ()
    if tokenizer is not None:
        if not isinstance(gold_answer, str) or not gold_answer.strip():
            raise ValueError("Free generation needs an actual gold answer")
        from .role_generation import greedy_answer, score_free_answer
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos = (tuple(eos_id) if isinstance(eos_id, (list,tuple))
               else (() if eos_id is None else (eos_id,)))
        baseline_greedy = greedy_answer(
            bb,uid,spec,prompt,max_generation_tokens,
            decode_mode=decode_mode,eos_token_ids=eos)
        baseline_score = score_free_answer(tokenizer,gold_answer,baseline_greedy)

    rows = []
    for c in candidates:
        positions = tuple(c["positions"])
        masked_prefill = role_forward(**common, forced_depths=frozen,
                                      role_mode="instruction_all",
                                      source=positions, tracked_positions=(anchor,))
        persistent = _trajectory_stats(masked_prefill["state_history"][anchor],
                                       clean_anchor, bb.R)
        single = None
        if run_one_loop:
            masked_once = role_forward(**common, forced_depths=frozen,
                                       role_mode="instruction_one_loop",
                                       source=positions, selected_loop=1,
                                       tracked_positions=(anchor,))
            single = _trajectory_stats(masked_once["state_history"][anchor],
                                       clean_anchor, bb.R)
        evicted = role_forward(**common, forced_depths=frozen,
                               role_mode="postprefill", source=positions,
                               tracked_positions=future_positions)
        if abs(evicted["first_target_nll"] - clean["first_target_nll"]) > replay_tol:
            raise RuntimeError("First target must not change after post-prefill eviction")
        decode_errors = []
        for p in future_positions:
            d = frozen[p]
            full_h = clean["state_history"].get(p, {}).get(d)
            evicted_h = evicted["state_history"].get(p, {}).get(d)
            if full_h is None or evicted_h is None:
                raise RuntimeError("Generated state not recorded at matched depth")
            decode_errors.append(_norm_delta(evicted_h, full_h))
        live = (role_forward(**common, role_mode="postprefill", source=positions)
                if live_exit and decode_mode == "dynamic" else None)
        alt_decode = live["depths"][P:] if live else []
        source_changes = []
        for p in positions:
            trajectory = clean["state_history"].get(p, {})
            for r in range(2, bb.R+1):
                if r-1 in trajectory and r in trajectory:
                    source_changes.append(_norm_delta(trajectory[r], trajectory[r-1]))
        greedy_score = None
        if tokenizer is not None:
            generated = greedy_answer(bb,uid,spec,prompt,max_generation_tokens,
                                      decode_mode=decode_mode,source=positions,
                                      eos_token_ids=eos)
            if generated["token_ids"][0] != baseline_greedy["token_ids"][0]:
                raise RuntimeError("Post-prefill masking changed first generated token")
            greedy_score = score_free_answer(tokenizer,gold_answer,generated)
        attention_by_loop = [sum(clean_attention[str(p)][r] for p in positions)
                             for r in range(bb.R)]
        # Online-observable: source-specific attention alignment with the
        # QUESTION's normal clean recurrent state evolution (not intervention).
        question_change = [0.0] + [
            _norm_delta(clean_anchor[r], clean_anchor[r-1])
            for r in range(2, bb.R+1)]
        denom = sum(attention_by_loop)
        recurrent_alignment = (
            sum(a * delta for a, delta in zip(attention_by_loop, question_change))
            / denom if denom > 1e-12 else 0.0)
        rows.append({
            "uid": uid, "candidate_id": c["id"], "source_role": c["role"],
            "source": c.get("source", ""), "source_token_positions": list(positions),
            "source_token_position": positions[0], "n_masked_tokens": len(positions),
            "full_sentence_token_count": c.get("full_sentence_token_count"),
            "age_at_prefill_end": _mean([P-p for p in positions]),
            "answer_anchor_used": c.get("answer_anchor_used", False),
            "attention_by_loop": attention_by_loop,
            "attention_mean": _mean(attention_by_loop),
            "attention_max": max(attention_by_loop),
            "attention_recurrent_alignment": recurrent_alignment,
            "source_mean_recurrent_change": _mean(source_changes),
            "prefill_instruction_error_by_loop": persistent["hidden_error_by_loop"],
            "prefill_persistent_final_error": persistent["final_error"],
            "prefill_persistent_recovery_ratio": persistent["final_over_first"],
            "prefill_one_loop_error_by_loop":
                single["hidden_error_by_loop"] if single else None,
            "prefill_one_loop_final_error": single["final_error"] if single else None,
            "postprefill_eviction_delta_future_nll":
                evicted["nll_future"] - clean["nll_future"],
            "postprefill_eviction_delta_all_nll":
                evicted["nll_all"] - clean["nll_all"],
            "future_decode_recurrent_error_mean": _mean(decode_errors),
            "future_decode_recurrent_error_by_token": decode_errors,
            "live_exit_flip_fraction":
                _mean([a != b for a, b in zip(base_decode, alt_decode)])
                if alt_decode else None,
            "live_exit_signed_depth_shift":
                _mean([a-b for a, b in zip(alt_decode, base_decode)])
                if alt_decode else None,
            "live_exit_delta_future_nll":
                live["nll_future"] - clean["nll_future"] if live else None,
            "postprefill_greedy_exact_match":
                greedy_score["exact_match"] if greedy_score else None,
            "postprefill_greedy_answer_flip":
                (greedy_score["extracted_answer"] != baseline_score["extracted_answer"])
                if greedy_score else None,
            "postprefill_greedy_generated_text":
                greedy_score["generated_text"] if greedy_score else None,
        })
    controls = []
    if question_control:
        if not prompt.question_positions:
            raise ValueError("Question control requires an aligned Question token")
        source = max(prompt.question_positions)
        q = role_forward(**common, forced_depths=frozen,
                         role_mode="postprefill", source=source)
        controls.append({"uid": uid, "source_token_position": source,
                         "source_role": "question",
                         "postprefill_eviction_delta_future_nll":
                             q["nll_future"] - clean["nll_future"]})
    return {
        "role_control_rows": controls, "trace": trace,
        "baseline": {
            "nll_all": clean["nll_all"], "nll_future": clean["nll_future"],
            "first_target_nll": clean["first_target_nll"],
            "replay_max_abs_token_nll_delta": parity,
            "decode_mode": decode_mode, "full_prefill_depth": bb.R,
            "attention_role_mass_by_loop": clean["attention"]["loop_roles"],
            "source_roles_from_real_token_offsets": True,
            "full_sized_reference_kv": True,
            "n_question_tokens": len(prompt.question_positions),
            "n_relevant_content_tokens": len(prompt.relevant_positions),
            "n_irrelevant_content_tokens": len(prompt.irrelevant_positions),
            "greedy_exact_match":
                baseline_score["exact_match"] if baseline_score else None,
            "greedy_generated_text":
                baseline_score["generated_text"] if baseline_score else None},
        "candidate_rows": rows}
