# Full-rollout AR/SD/K frontier — results (run 2026-11-13)

Code commit `484f51aa` (runner, job matrix, validator, analyzer, tests) was pushed **before** any result.
All 161 pre-registered jobs (`jobs_20261113.txt`) ran to rc=0 on 8× H200 (18:18–23:11 KST); burners restored on all 8 GPUs afterwards.
Derived tables: `results_20261113/frontier.md` / `frontier.json`; validator output: `results_20261113/status.json`.
Raw receipts: `results_20261113/raw/<lane>/*.jsonl.gz` (+ `.meta.json`), worker/job logs in `job_logs.tar.gz`, checksums in `MANIFEST.sha256`.
Re-derive: `gunzip -k results_20261113/raw/*/*.gz && python validate_frontier_receipts.py --raw results_20261113/raw --out status.json && python analyze_frontier.py --status status.json --out-dir .`

Setting: B=1, FULL KV, greedy, same target per arm. Arms = AR, DtV(d,K) for K∈{1,2,4,8,16}, WFD(d); d fixed (Ouro 1, Huginn 4).
Speedup = Σtok/Σwall of arm ÷ same-cell AR, paired prompt bootstrap 95% CI. `plateau/tied` = best vs runner-up gap ≤2% or CI ∋ 1 (pre-registered).

## RUN_STATUS

| lane / arm | status |
|---|---|
| parity FP32 (18 cells: 2 models × 3 exits × {gsm8k3, math500, spec}, K2/4/8 + WFD) | **passed — 0 inexact of 768 SD rollouts** |
| s1 BF16 frontier (20 cells) | passed (validator: complete arm grid, prompt hash, stop-matcher calls = tokens, no fabricated stage times) |
| rep (5 timing repeats, 2 cells) | passed, median per-prompt CV 0.2–0.5% |
| S2 stage-time decomposition (draft/verifier/coda/gate/host-sync) | not_run — fields recorded as null, never fabricated |
| S3, S4 | not_run |
| Dory | blocked — nGPT arch (16/40/16 layers, per-loop KV) has no implementation in pinned WaveFront `4345209`; no verified per-token gate |
| LoopSpec official SGLang reproduction | not_run |
| BBH, HumanEval+, MBPP+ | not_run — LoopSpec custom prompts not ported; no code-exec sandbox |
| Ouro-2.6B | not_run |

## S1 frontier (BF16, e2e speedup vs AR)

| cell | n | AR tok/s | AR depth | K1 | K2 | K4 | K8 | K16 | K_opt [CI] | tied set | WFD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| ouro fixed gsm8k3 | 300 | 20.7 | 4.0 | 1.43 | 1.67 | **1.83** | 1.77 | 1.45 | K4 1.83 [1.81,1.85] | K4 | 2.21 |
| ouro fixed gsm8k8 | 50 | 20.6 | 4.0 | 1.43 | 1.65 | **1.82** | 1.75 | 1.39 | K4 1.82 [1.76,1.87] | K4,K8 | 2.21 |
| ouro fixed math500 | 50 | 20.6 | 4.0 | 1.46 | 1.71 | 1.93 | **1.93** | 1.57 | K8 1.93 [1.85,2.02] | K4,K8 | 2.32 |
| ouro fixed spec | 48 | 22.0 | 4.0 | 1.44 | 1.69 | **1.91** | 1.86 | 1.53 | K4 1.91 [1.79,2.02] | K4,K8 | 2.36 |
| ouro τ0.4 gsm8k3 | 300 | 32.1 | 2.5 | 1.22 | 1.31 | **1.35** | 1.23 | 0.97 | K4 1.35 [1.33,1.36] | K4 | 1.58 |
| ouro τ0.4 gsm8k8 | 50 | 31.9 | 2.5 | 1.20 | 1.32 | **1.33** | 1.25 | 0.93 | K4 1.33 [1.29,1.37] | K2,K4 | 1.58 |
| ouro τ0.4 math500 | 50 | 33.7 | 2.4 | 1.23 | 1.32 | **1.40** | 1.27 | 1.07 | K4 1.40 [1.35,1.44] | K4 | 1.57 |
| ouro τ0.4 spec | 48 | 37.1 | 2.3 | 1.21 | 1.31 | **1.35** | 1.22 | 0.95 | K4 1.35 [1.28,1.40] | K2,K4 | 1.49 |
| ouro τ0.1 gsm8k3 | 300 | 47.5 | 1.6 | **1.03** | 1.02 | 0.98 | 0.87 | 0.68 | K1 1.03 [1.01,1.05] | K1,K2 | 1.18 |
| ouro τ0.1 math500 | 50 | 50.1 | 1.6 | 0.99 | **1.03** | 1.01 | 0.87 | 0.72 | K2 1.03 [0.96,1.10] | K1,K2,K4 | 1.10 |
| huginn fixed gsm8k3 | 300 | 13.2 | 32 | 1.37 | 1.61 | **1.74** | 1.61 | 1.17 | K4 1.74 [1.69,1.80] | K4 | 1.85 |
| huginn fixed gsm8k8 | 50 | 13.3 | 32 | 1.34 | 1.58 | **1.67** | 1.54 | 1.10 | K4 1.67 [1.58,1.79] | K4 | 1.72 |
| huginn fixed math500 | 50 | 13.3 | 32 | 1.46 | 1.86 | 2.27 | **2.55** | 2.39 | K8 2.55 [2.29,2.74] | K8,K16 | 2.71 |
| huginn fixed spec | 48 | 13.8 | 32 | 1.48 | 1.87 | 2.33 | **2.58** | 2.41 | K8 2.58 [2.43,2.74] | K8 | 2.96 |
| huginn ε0.1 gsm8k3 | 300 | 22.0 | 17.0 | 1.23 | **1.35** | 1.33 | 1.14 | 0.78 | K2 1.35 [1.33,1.38] | K2,K4 | 1.60 |
| huginn ε0.1 gsm8k8 | 50 | 22.0 | 16.8 | 1.21 | **1.29** | 1.27 | 1.09 | 0.71 | K2 1.29 [1.24,1.34] | K2,K4 | 1.47 |
| huginn ε0.1 math500 | 50 | 24.6 | 14.9 | 1.27 | 1.46 | **1.67** | 1.66 | 1.53 | K4 1.67 [1.59,1.72] | K4,K8 | 1.76 |
| huginn ε0.1 spec | 48 | 25.5 | 14.8 | 1.29 | 1.46 | **1.61** | 1.60 | 1.37 | K4 1.61 [1.54,1.68] | K4,K8 | 1.91 |
| huginn ε0.2 gsm8k3 | 300 | 29.3 | 12.0 | 1.15 | **1.21** | 1.17 | 0.96 | 0.65 | K2 1.21 [1.19,1.23] | K2 | 1.41 |
| huginn ε0.2 math500 | 50 | 32.7 | 10.7 | 1.21 | 1.34 | **1.44** | 1.38 | 1.15 | K4 1.44 [1.36,1.48] | K4,K8 | 1.49 |

Held-out K (gsm8k3: K chosen on idx<100, evaluated on idx 100–299): the calibration K equals the eval-best K in all 6 exits (regret 1.000).

## Findings

1. **K_opt shrinks as the live exit gets shallower.** Ouro fixed (D=4) → τ0.4 (D≈2.5) → τ0.1 (D≈1.6): K_opt 4 → 4 → 1–2, best DtV speedup 1.83 → 1.35 → 1.03. Huginn fixed (D=32) → ε0.1 (D≈17) → ε0.2 (D≈12): gsm8k3 K_opt 4 → 2 → 2, speedup 1.74 → 1.35 → 1.21.
2. **Long-generation tasks prefer larger K.** On MATH-500 and Spec-Bench (≈400–700 tokens) acceptance at each K is higher and K_opt moves up (Huginn fixed: K8; Huginn ε0.1: K4) vs gsm8k (≈110 tokens; K2–K4).
3. **Plateaus are common.** 13 of 20 cells have a tied set of ≥2 K values under the pre-registered 2%/CI rule; a distinct optimum appears mainly in fixed-depth and τ0.4 gsm8k3 cells.
4. **AR is rarely faster per prompt.** The share of prompts where AR beats K_opt is ≤0.06 everywhere except Ouro τ0.1 (0.32 gsm8k3, 0.36 math500) and Huginn ε0.2 math500 (0.20) — i.e. exactly where the exit is shallow.
5. **WFD(d) is above every DtV K in every cell** (1.10–2.96×), with the same parity guarantee in FP32.
6. Speedups are unchanged by few-shot format (gsm8k3 vs gsm8k8 within CIs) and are stable across repeats (CV <0.5%).

## Accuracy re-score (post-hoc, 2026-11-14)

The recorded `correct` is the task's primary lm-eval metric as replicated in WaveFront exp3 `tasks.py`:
gsm8k strict-match, math500 minerva `exact_match`. Minerva exact_match only accepts the literal
"Final Answer: The final answer is X. I hope it is correct.". Base Ouro ends with `\boxed{X}` or
"Final Answer: The answer is \boxed{X}", so almost every answer was scored `[invalidanswer]`
(Ouro MATH-500 0.00–0.04). The upstream secondary metric (math_verify) was None because the package
is not installed in the run env.

`rescore_accuracy.py` decodes the stored token ids (CPU, receipts unchanged) and adds:
- for math500, the last `\boxed{}` with the same minerva normalization and is_equiv;
- for gsm8k, the lm-eval flexible-extract.

Recomputing the strict metric reproduces the recorded value in all 30 cells, so nothing was wrong
with the run itself. Full per-arm table: `results_20261113/accuracy_rescore.{md,json}`.

| cell (AR arm) | strict (recorded) | added metric |
|---|---:|---:|
| ouro fixed math500 / gsm8k3 / gsm8k8 | 0.04 / 0.62 / 0.62 | **0.64 / 0.78 / 0.70** |
| ouro τ0.4 math500 / gsm8k3 / gsm8k8 | 0.00 / 0.59 / 0.54 | **0.58 / 0.75 / 0.64** |
| ouro τ0.1 math500 / gsm8k3 | 0.00 / 0.46 | **0.34 / 0.58** |
| huginn fixed math500 / gsm8k3 / gsm8k8 | 0.16 / 0.27 / 0.30 | 0.12 / 0.36 / 0.32 |
| huginn ε0.1 math500 / gsm8k3 / gsm8k8 | 0.14 / 0.26 / 0.36 | 0.14 / 0.34 / 0.36 |
| huginn ε0.2 math500 / gsm8k3 | 0.10 / 0.18 | 0.08 / 0.24 |

- Huginn's MATH-500 accuracy is low under both metrics (0.08–0.16).
- The live exit costs accuracy, more at the shallower setting: on Ouro MATH-500, fixed 0.64 → τ0.4 0.58 → τ0.1 0.34.
- Across arms in the same cell, accuracy differs by ≤0.1 (BF16 drift, n=50). In FP32 parity the arms are identical.
- The parity lane uses max_new 256, so MATH answers are often truncated there (0.125).

## Caveats / deviations (pre-declared)

- **BF16 is not token-exact to AR** (per-arm exact_to_AR 0.12–0.86, mostly 0.6–0.7; FA varlen numerics with different query widths). Exactness is established only in the FP32 SDPA-fallback parity lane. BF16 accuracy differences between arms are numerical drift, not an SD effect.
- **Accuracy: use the post-hoc re-score, not the recorded `correct` field, for MATH-500** (see below). Speed results do not depend on it.
- Stop strings are applied during generation (lm-eval `until` semantics); WaveFront exp3 truncates only at scoring.
- Huginn exit rule = WaveFront relative-L2 ‖h_t−h_{t−1}‖/‖h_{t−1}‖ < ε (not Huginn's own normalization).
- One timing repeat per rollout outside the `rep` lane; MATH n=50 (not 100/200); Spec-Bench 48/480 stratified (8 per task).
- Post-hoc strata in `frontier.md` are outcome-derived and not pre-action features.
- Stage times (S2) are not instrumented; e2e and decode-only (wall − prefill) are the only timings.
