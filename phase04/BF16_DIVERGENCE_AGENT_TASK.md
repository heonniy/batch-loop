# Phase 04R executor task — localize and resolve BF16 live-exit divergence

This task starts from the reviewed failure commit `f84bd0c02855de2848e449f4c4eab45f4cbd2dbf`.
Read first:
1. `phase04/BF16_DIVERGENCE_RESOLUTION_PLAN.md`
2. `phase04/results/20260917T052341Z_c591671/IMPLEMENTATION_REVIEW.md`
3. `phase04/GATE_A_PARITY_RULE.md`
4. `phase04/DEPTH_PATTERN_PREDICTOR_PLAN.md`
5. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md`

## Immediate objective

Do **not** resume predictor collection. Determine whether the BF16 B1-versus-batched live-exit mismatch is:
- a real implementation/row/KV/mask/state bug;
- ordinary BF16 batch-shape numerical divergence that can flip a threshold decision;
- or an unresolved mixture.

The answer must come from the earliest divergent operation on the already saved counterexample rows, not from final-output speculation.

## Step 1 — freeze counterexamples

Build `COUNTEREXAMPLE_MANIFEST.json` from the existing raw Gate-A receipts. Include all decision mismatches and matched far-margin controls. Preserve exact request/token IDs, effective token inputs, h0 hashes where applicable, prior history/KV-policy signatures, exit scores/margins, source/model/config hashes and original evidence paths.

Do not regenerate labels or substitute new prompts for this localization stage.

## Step 2 — implement bounded first-divergence tracing

Add opt-in debug hooks that record per-target-row norms/hashes and errors at the boundaries listed in the resolution plan. The normal benchmark path must remain uninstrumented. Do not globally dump every hidden tensor.

Trace B1 and controlled B=8 for the same target input until the first component diverges, then dump only that component's target-row tensors. Repeat FP32 as a control. Preserve target row/KV identity and use the existing request-keyed Huginn h0 mechanism.

A run that merely reports a later exit mismatch is incomplete.

## Step 3 — batch-shape matrix

For each model, begin with one saved mismatch request/token and fixed filler rows. Run B=1/2/4/8, target row position first/middle/last, FIFO/fixed permutation, equal-length/original mixed-length, BF16/FP32. No forced exit.

Add duplicate-target and row-permutation invariants. Restore canonical order before comparing. Record whether differences follow tensor shape, target row position, or logical request/KV identity.

Do not expand to B16/32 or long horizons until this bounded matrix produces a discriminating result.

## Step 4 — isolate the first divergent operator

Once a first layer/subcomponent is identified, extract the exact current-backend input for that component and reproduce target-row B1 versus packed-B output in a minimal GPU diagnostic. Keep the current backend first. High-precision/deterministic alternatives are controls only.

Publish source SHA before execution. Add a regression test for any demonstrated implementation bug.

## Step 5 — choose Branch A/B/C explicitly

- **A, implementation bug:** fix only the proven bug, rerun frozen counterexamples, then rerun the original prospective N32/B8/T8 BF16 Gate A unchanged.
- **B, expected BF16 batch-shape numerics:** do not relabel old FAILs. Quantify flip rate versus B and exit margin on fresh validation requests, then draft a prospective semantic contract or a batch-dependent-depth research contract. Stop before applying that new rule unless it is committed before confirmation.
- **C, unresolved:** publish the bounded result and the next smallest discriminating test; do not start predictor training.

No threshold, q, checkpoint, KV policy, Rmax or primary precision change is an implementation fix.

## Step 6 — confirmation and reopening condition

Only after a justified fix or prospectively committed Branch-B rule, run a fresh Gate A on new N32/B8/T8 requests. Preserve every mismatch and raw receipt. Predictor work reopens only for model lanes that pass their preregistered new gate.

If reopened, return to the existing Phase-4 predictor plan: offline token prior, previous hidden/Prelude contextual features, then depth-aware microbatch composition. If live depth is proven batch-dependent, update predictor/oracle terminology and evaluation exactly as specified in the resolution plan before collection.

## GPU and process rules

Use only the exact physical-index/UUID/model-role mapping already recorded in `phase04/GPU_ASSIGNMENT_20260917.json`, unless a later explicit authorization supersedes it. Check running jobs first and do not interrupt unrelated processes. This task does not authorize indefinite background sweeps. When a bounded diagnostic finishes, release the GPU.

## Publication

Use new additive roots named `phase04/results/<timestamp>_<source_sha>_bf16_resolution/`. Keep source changes, measured results and review in separate commits. Required final handoff:
- first divergent operation per model;
- evidence for Branch A/B/C;
- exact fix if any;
- fresh Gate-A status if rerun;
- what remains BLOCKED;
- no predictor or speed claim unless the reopening condition was met.
