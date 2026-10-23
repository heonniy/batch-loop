# Huginn R32: mechanism of Full-vs-Terminal KV robustness

Status: **experiment design; not executed**. Branch derives from `research/huginn-r32-kv-oracle-20261023`. All earlier result receipts and source files are preserved.

## Question

Why can a current token at recurrent loop r attend to **past tokens' loop-32 KV** instead of past tokens' loop-r KV, with almost unchanged predictions? Distinguish:
1. raw cross-loop KV redundancy;
2. attention readout insensitivity to KV substitution at a **fixed query**;
3. recovery of errors by later recurrent computation;
4. Coda / LM-head insensitivity to residual representation differences.

Do **not** infer that attention "dominates" hidden state based on norms alone. A residual update's *alignment* with error matters.

## Fixed conditions

Huginn-0125 pinned revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`; BF16 (with FP32 instrumentation/accumulation where feasible); R=32; Hugging Face reference engine; no vllm-rlt; H200 up to 8 independent workers; fixed UID/position latent initialization, tokenizer, prompts, greedy FULL-generated prefix. Use existing GSM8K and MATH-500 subsets, but start with 32 items each across correct/incorrect, short/long and MATH difficulty strata. Preserve policy conventions: FULL depth-indexed history, LAST-convert terminal history, same FULL prefill; ordinary prelude/coda caches. No training or early exit.

## Experiment A — KV geometry (observational)

For recurrent physical layer l, position i, head h, loop r:
  dK[l,i,h,r] = ||K^(r)-K^(32)||_2 / (||K^(r)||_2 + eps)
  dV[l,i,h,r] = ||V^(r)-V^(32)||_2 / (||V^(r)||_2 + eps)
  cosK = cosine(K^(r), K^(32)); cosV analogously.
Also inspect consecutive differences (r vs r+1) and per-head dispersion. Compute separately for prompt vs continuation and early vs late recurrence. Avoid using global average alone. Rotary/positional transforms must be aligned at the same token/head position. Compare the same **FULL trajectory** (not K from separate FULL and LAST trajectories).

## Experiment B — controlled attention counterfactual (causal, fixed Q)

At each current decode token t and loop r, obtain Q_F^(r) and current token self-K/V from the FULL trajectory. Compare:
  A_ref = Attn(Q_F^(r), past K_F^(r), past V_F^(r); same current-self K/V)
  A_swap = Attn(Q_F^(r), past K_F^(32), past V_F^(32); same current-self K/V)
  D_attn = ||A_ref - A_swap||_2 / (||A_ref||_2 + eps).
Keep causal masking, RoPE, attention scaling, positions and dtype identical; change *only past recurrent KV*. Record attention weight distributions, optionally separate key-only vs value-only swaps to attribute impact. Instrument **per recurrent attention layer** (4 physical Core layers of Huginn); do not incorrectly aggregate K/V across layers. No claim of actual performance change from this one-layer frozen-query intervention, which is diagnostic.

Control: replay FULL vs FULL (zero-error instrumentation), fixed Q with alternate identical KV layout, and numerical precision/noise floor. Verify a toy SDPA reference attention output against direct implementation before GPU sweep.

## Experiment C — actual sequential intervention and error propagation

Execute FULL and LAST with the same frozen prefix and initial latent. At each recurrent loop and physical sublayer log:
  e_h(r,l) = ||h_F(r,l)-h_L(r,l)||/(||h_F(r,l)||+eps)
  delta_a(r,l) = A_L(r,l)-A_F(r,l)
  delta_m(r,l) = M_L(r,l)-M_F(r,l)
and residual error alignment, e.g.
  cos(e_before, delta_a), cos(e_before, delta_m).
These are *observed differences*, not isolated attention/MLP causal effects because Q and states can change; label them accordingly. Compute before/after each residual update, prelude/core/coda boundaries, final logits KL and argmax. Explicitly test whether e_h decreases with recurrence and whether it remains large with small logits KL.

## Experiment D — attribution via controlled interventions (pilot if A–C worthwhile)

Starting from a FULL current-token trace, intervene at one recurrent layer and one loop:
- replace only the historical KV read (B), then let subsequent Core and Coda run;
- compare immediate attention difference, post-layer hidden error, post-R32 hidden error, final output KL;
- repeat interventions at early r in {1,2,4,8}, middle {16}, late {24,31,32}.
Avoid comparing independently regenerated prefixes. This discriminates small initial perturbation from later recovery. A paired current-state clamp or explicit activation patch may be added, but document any off-manifold effects.

## Outputs / analysis

- `analysis/kv_geometry_by_loop.csv` and per-head summary
- `analysis/fixed_q_attention_by_loop.csv`, key-only/value-only diagnostics
- `analysis/hidden_error_by_loop.csv`, residual alignment and output KL
- `analysis/intervention_effects.csv` for pilot D
- `analysis/mechanism_report.md` with per-request plots, bootstrap CI across **prompts**, number of prompts contributing at each token position, and negative cases
- reproducible manifest with checkpoint/dataset revisions, exact code SHA, seeds, hardware and instrumentation precision

Primary analysis should stratify by loop r and token position t, compare fixed cohorts when discussing depth/length trends, and distinguish attention-output *norm* from *direction* and from final prediction. Check that hooks do not change FULL greedy tokens or logits beyond precision tolerances.

## Decision rubric

- KV near-identical + fixed-Q outputs near-identical => raw KV redundancy plausible.
- KV different + fixed-Q outputs near-identical => attention readout insensitivity plausible.
- Fixed-Q outputs different + downstream hidden error contracts => recurrent recovery plausible.
- Hidden error remains high + final KL low => Coda/LM-head insensitivity plausible.
- Mixed observations => report mixed mechanisms rather than forcing one explanation.

Stage A–C = minimum; D optional until instrumentation and baselines pass. This experiment **characterizes mechanism** and does not by itself establish a new optimization algorithm or a novel contribution.

## Execution request for agent

Read `AGENTS.md`, this plan, existing `kv_policy/kvp/engine.py`, Stage-1 parity, Stage-2 fixed-prefix results and Stage-3 oracle report. Implement trace instrumentation under `kv_policy/` without overwriting baseline, add toy deterministic unit tests, then run a bounded 32+32 prompt pilot on the authorized H200 x8 host. Commit implementation, test logs and report separately. If GPU access is unavailable, stop at code/tests and mark the measurements not run; never fabricate results.
