# Huginn R32 — Perturbation Stability and Jacobian Analysis

**Status: experimental protocol, not an executed GPU run.** Based on completed Stage A–D mechanism pilot (64 prompts):
`kv_policy/results/20261023T125444Z_huginn_r32_attn_mechanism/analysis/mechanism_report.md`.
Use HF reference engine only; no vllm-rlt, early exit, retraining, or Ouro in this phase.
Pinned Huginn-0125 revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`, fixed depth R=32, authorized H200 x8.
Never overwrite earlier results.

## Research question

Prior observations: LAST has ~0.0275 relative hidden error at R32 (BF16);
single KV perturbation at loop 1 had post-layer error ~0.335 and post-R32
error ~0.000134 (FP32) versus ~0.014 (BF16).
Are these perturbations damped because of **a broad locally stable recurrent
dynamics**, or only because KV-sharing produces a special perturbation direction?
What model components contribute to recovery?

Do not assert universal robustness, global contraction, or learning causality
from a finite test set.

## Definitions and controls

At a selected physical recurrent layer/loop boundary r let z_r be the entire
current-token recurrent state at that boundary (freeze historical cache, prompt,
UID/position noise state, and future fixed token prefix). For a matched
counterfactual z'_r, report:
  E_s = ||z'_s-z_s||_2 / (||z_s||_2+eps), s = r,...,32
  G_{r:s} = ||z'_s-z_s||_2 / (||z'_r-z_r||_2+eps)
E is relative output error, G is absolute perturbation amplification; never
mix them or call G a singular value.

For a small unit direction v, the finite-difference local response is
  g_r(v) = ||F_r(z_r+eta*v)-F_r(z_r)||_2 / eta
where F_r is a well-defined **one-loop map**, all external inputs and
historical K/V held constant. Verify that finite differences converge across
eta in {1e-2,1e-3,1e-4} relative to state RMS and cross-check JVP when possible.
For true multi-step response, evaluate J_{s-1}...J_r*v via JVP composition
without materializing a dense Jacobian. A large g_r(v) does not imply the
full operator norm is >1, and a small sample does not prove contraction.

## Stage 0 — instrumentation and parity gate

- Read repository AGENTS.md, mechanism report, Huginn Backbone and current
  tests; implement isolated scripts under `kv_policy/scripts/stability/` and
  minimal reusable hooks under `kv_policy/kvp/`.
- Verify traced vs untraced FULL logits and per-layer hidden state; compare
  BF16 with FP32 control. Work around the previously documented H200 CUDA FP64
  softmax anomaly by **not using CUDA FP64 as oracle**.
- Ensure attention history is frozen and interventions do not alter prior
  tokens, RNG or future input tokens. Validate on tiny deterministic attention
  toy cases. Reject NaN/Inf and position/mask mismatches.
- Use FP32 as primary for dynamical attribution, BF16 to measure practical
  numerical floor. Match amplitude at injection, not just source norm.

## Stage 1 — multi-direction perturbation experiment

Pilot: 32 GSM8K + 32 MATH-500 prompts reused from mechanism pilot, stratified
by correctness, prompt length, MATH difficulty; 8 independent H200 workers.
Use fixed FULL continuation, and sample positions spanning early, mid and
late decode. Injection loops r = {1,2,4,8,16,24,31}, with later loops
r+1,...,32 re-executed under the same FULL recurrent history.

Intervention families at the **same hidden-state boundary**:
1. ORIGINAL KV-swap-induced direction: actual delta z_r from one historical
   full->terminal attention read, rescaled if needed.
2. Isotropic Gaussian direction, multiple seeds (>=8) per injection site.
3. Single-coordinate and random sparse-coordinate perturbations, normalized.
4. Attention-output-aligned direction: obtain the true attention output
   displacement, then inject it into z at the correct residual boundary.
5. MLP-output-aligned direction as a control.
6. Random direction orthogonal to the observed KV direction.
Choose fixed relative amplitudes e.g. 0.01,0.1,0.3 of ||z_r||; large
amplitudes measure *nonlinear basin behavior*, not local Jacobian. Verify
all directions receive the same actual post-injection L2 magnitude.

Record per-loop E_s, G_{r:s}, cosine(delta z_s, delta z_r) when dimensions
match, immediate attention/MLP residual deltas, final KL and top-1 agreement.
Aggregate by prompt, injection position, direction type and amplitude with
prompt-cluster bootstrap CI. Do not extrapolate "arbitrary values" from
small local perturbations.

## Stage 2 — Jacobian response / stability

- At loop r estimate JVP of the one-loop recurrence map in multiple
  directions; compare perturbation classes using equal-norm vectors.
- Estimate the dominant singular gain of a *multi-loop* Jacobian by power
  iteration on J^T J with autodiff JVP/VJP, if feasible; report iteration
  convergence and state dimensionality. If too costly, explicitly label
  sampled directional gain rather than spectral norm.
- Compare one-loop gains and composed gains from r to R32.
- Evaluate whether a small final response is due to repeated contraction
  or cancellation between expanding and shrinking phases.
- Run two precision controls and a repeat-seed control at matching sites.

## Stage 3 — mechanism attribution (controlled, limited)

Use diagnostic ablations only; these change the model function and are not
deployment policies. For each compare matched-baseline logits and
perturbation amplification **within that modified model**, not only absolute
accuracy against original FULL.
A. Repeated input injection: original, disabled after first loop,
   scaled to {0.5,1.5} where feasible, preserving normalization semantics.
B. Prelude input clamp: identical input across perturbation branches,
   verify no accidental RNG changes.
C. Residual branch diagnostics: attention-only / MLP-only *delta patches*
   at selected loops, not globally deleting learned weights.
D. Layerwise Jacobian gain from residual/attention/MLP boundaries to
   isolate which transformation contracts the observed directions.
These interventions can be out-of-distribution; results support sensitivity
attribution, not proof of training causality. Retraining prior/Truncated-BPTT
ablations would be a separate project and must not be claimed.

## Expected artifacts

Under `kv_policy/results/<new_run>/analysis/`:
- `perturbation_by_loop.csv`, `perturbation_by_prompt.csv`
- `jacobian_directional_gain.csv`, optional `jacobian_spectral_estimate.csv`
- `injection_ablation.csv`, `stability_report.md`
- run manifest: git SHA, exact model/data revisions, seeds, GPU UUIDs, dtype,
  command line, intervention loop/boundary and perturbation amplitude
- correctness tests, failures and representative counterexamples

## Decision criteria

A. KV directions contract, random/orthogonal directions do not:
   **KV-specific robustness**, not general attractor evidence.
B. Multiple independent directions contract with a reproducible
   multi-loop Jacobian gain <1:
   **local recurrent stability** supported, not global robustness.
C. Contraction changes materially after input-injection intervention:
   input injection plausibly contributes; further controlled study needed.
D. FP32 contracts but BF16 plateaus:
   separate mathematical dynamics from numerical error floor.
E. Gains depend strongly on perturbation amplitude, token, or prompt:
   report an empirical basin of recovery rather than one global property.

**Execution sequence:** implement+unit tests → FP32 8-prompt smoke test →
64-prompt bounded pilot on H200x8 → analyze + push results. Stop before
expensive full sweep if parity or finite-difference checks fail.

This commit is the plan only. Running and posting results require
execution on the H200 server by the separate experiment agent.
