# Phase 05 agent handoff — strict no-refill on four H100s

## Objective

Build a defensible baseline for **offline decode-boundary cohort regrouping** while keeping the shared recurrent core **strictly no-refill**. The upstream implementation to study is https://github.com/kschwethelm/continuous-depth-batching . The destination is this repository only. Do not copy upstream code without checking license/attribution; prefer adapters or independently written experiments.

## Before changing anything

1. Read repository AGENTS.md, phase05/README.md, phase04/results/20260917T065413Z_e4113e5_rank_stability_cpu/RESULT_REVIEW.md and phase04/results/20260917T060824Z_50910e6_bf16_resolution/PROPOSED_B2_CONTRACT.md.
2. Review the already implemented phase02_runtime physical compaction and phase03_predictor modules. Reuse rather than duplicating.
3. Run the **CPU-only** test command below. If it fails, repair phase05 code with an isolated commit and publish the failure/fix. Do not suppress failing checks.
4. Record current branch/commit and inspect GPU resources *only once an explicit device-index/UUID authorization exists*. The user reports access to **four H100s** but their physical device identities are not specified in this turn.

    python -m unittest -v phase05.test_frozen_cohort_sim

## Immutable runtime contract

- Target: offline, no-refill, decode-boundary cohort composition.
- Recurrent-level: survivor removal within started cohort only; NEVER append new rows or merge survivors across cohorts in the same recurrent interval.
- Predictor controls group ranking, not exit gate/threshold or hidden state.
- Ouro: last-exited depth-indexed KV; Huginn: shared physical-layer KV.
- Match checkpoint, precision, KV, Rmax, thresholds, source revision, tokenization/sampling and prefill across policies.
- Do not assume exact BF16 depth invariance across batch shapes.
- Frozen B1 labels represent only a structural packing diagnostic. No fabricated live Oracle from unexecuted regrouping.

## Ordered work packages and stop gates

### WP0 — CPU prototype and existing evidence review (authorized without GPU)

Validate the Phase05 simulator's readiness dependencies, shrink-only widths, row identity, deterministic policy ordering, RIC and stage-cost accounting. Add coverage for non-divisible remainders, EOS/variable output length, identical-depth ties, and model-specific depth caps. Run a small synthetic sanity case and save labeled **CPU_FROZEN_TRACE_ABSTRACT_COST_NOT_LIVE** outputs; no speed claim.

Compare prior Phase04 CPU depth-rank evidence, identify remaining assumptions, and publish a concise written recommendation to approve/amend/reject B2. This task **does not itself activate B2**.

### STOP: explicit research-contract decision required

Do not launch GPU runs, restart predictor data collection, or claim new physical FIFO/Oracle results until the B2 research contract is explicitly approved/modified and a prospective validation gate is committed. Original BF16 Gate A remains FAIL.

### WP1 — gated small correctness confirmation (only after explicit approval)

With verified and authorized indices/UUIDs for 4 H100s, run a fresh disjoint N32/B8/T8 BF16 confirmation (one Ouro and one Huginn plus independent same-model replicas), capturing actual live exit depths, output IDs, KV identities/valid masks, h0 where relevant, and source hashes. Record FAILs. Stop a model lane on correctness failure.

### WP2 — physical no-refill headroom (only if WP1 passes)

Use a corrected integrated physical early-exit runtime, not Ouro's native post-hoc gate path. Compare FIFO, random, static-first and history-based regrouping, plus actual-executed Oracle/grouping checks. Predeclare dataset splits and the B/threshold grid before runs. Each grouping must execute its own live depth/work, with decode/E2E timing. Three repeats per successful scored cell; retain all failures/OOMs. Use four H100s as independent model/replica lanes, NOT automatic TP/DP or four-rank EP.

### WP3 — predictor (only if physical headroom warrants it)

Analyze depth-propensity ordering and temporal correlation using disjoint train/validation/test requests, select simple causal predictors on validation, freeze before fresh test. Include time for feature extraction, predictor, regrouping and output reordering. Retain a history-only baseline.

## Deliverables

Separate source, bounded result and review commits. Raw data and manifests must be immutable and auditable; publish phase05/results/<timestamp>_<source_sha>/RESULT_REVIEW.md and review_manifest.json with model-specific correctness gates, BF16 live-work deltas, RIC, stage and whole-model timings, detailed failures, comparison scope and GO/NO-GO decision.

## Critical acceptance criterion

Do not represent simulated RIC improvement, frozen B1 Oracle, post-hoc HF gate inference or extrapolated four-H100 capacity as an **observed integrated physical speedup**. If oracle headroom is negligible, report NO-GO rather than tune the data to generate a positive result.
