# Phase 05 WP4 — host-sync-free GPU serving of the WP3 predictors (preregistration)

Committed **before** any scored WP4 run. Authorization: user 2026-10-18 ("host 동기화가 없는 GPU 내 feature 계산으로 해봐"), after the WP3 NO-GO review.

## Hypothesis (from the WP3 diagnosis)

In WP3 the predictors cut live RIC by 4–10% but kept only 0–8% decode wall. The hypothesized loss is predictor-specific serving work: host syncs for per-recurrence trajectory capture, full-vocabulary logit copies, numpy feature assembly and a CPU head. **If that is the cause, serving the *same frozen predictors* with identical features computed entirely on the GPU should recover most of the RIC saving as wall time.**

## What changes and what does not

- **Unchanged:** WP3-frozen checkpoints and winners (Ouro×Alpaca C7, Huginn×Alpaca C8, Ouro×ShareGPT C9, Huginn×ShareGPT C8; SHA-256 verified at start), feature definitions, projection, WP1 contract (BF16, thresholds, Rmax, KV, h0), staged Prelude in FIFO chunks of B, live exits, teacher-forced B1 tokens, strict no-refill, hard checks. No retraining or reselection.
- **New serving path** (`phase05/wp4_gpu_serving.py:gpu_run`): history, previous-token uncertainty (computed per cohort right after Coda from that cohort's logits and the next teacher-forced token), previous trajectory (scattered by run_interval debug hooks), projections, the head and `argsort` all stay on device. No per-step host sync for any policy; the survivor-compaction syncs inside `run_interval` are common to every policy.
- `predictor_host` is the unchanged WP3 implementation (`wp3_predictor.physical_run`), run in the same session for a paired comparison.

## Fresh data

New split `wp4_confirm` = pool indices `[2304, 2816)` (`phase05/wp2_headroom.py:SPLITS`), never used before. Collected with the WP3 collector (B1 live greedy, T=64) for oracle labels and teacher-forced tokens; two cohorts of 256 per lane. Lanes as before: A Ouro×Alpaca GPU0, B Huginn×Alpaca GPU1, C Ouro×ShareGPT GPU2, D Huginn×ShareGPT GPU3.

## Gate 0 — equivalence audit (untimed, per lane, before scored runs)

On `wp4_confirm` cohort 0, B=8, T=64, `predictor_gpu` runs while the WP3 host feature path is evaluated on the same live state at every step (trajectory from run_interval's own host sink, logits from a full copy). **PASS** requires mean within-step score Spearman ≥ 0.999, minimum ≥ 0.99, mean identical-cohort fraction ≥ 0.95, and no hard-check error. A lane that fails Gate 0 is not scored.

## Scored grid

Policies `fifo`, `history_last`, `predictor_gpu`, `predictor_host`, `oracle_b1`; B ∈ {8,16,32,64}; 2 cohorts; 3 repeats; one warmup; B order shuffled per (rep, cohort) and policies rotated per rep. All runs, outliers, OOMs and errors are retained.

## Decision rules (fixed now)

Per lane, cohort and B: headroom(p) = 1 − median(p)/median(fifo).
- **Primary (unchanged WP3 rule, applied to `predictor_gpu`):** wins a cell if headroom ≥ 5% and all 3 repeats are faster than all 3 FIFO repeats; **practical GO per model** if it wins ≥ 2 of 4 B values in each of the 4 dataset×cohort blocks.
- **Serving-cost hypothesis:** *supported* if median `predictor_gpu` wall < median `predictor_host` wall in ≥ 75% of the 32 cells **and** the mean RIC-to-wall translation (wall saving ÷ live-RIC saving vs FIFO, cells with RIC saving ≥ 2%) rises from `predictor_host` to `predictor_gpu`. Otherwise *not supported*.
- Also reported: gain of `predictor_gpu` over `history_last`, recovery of oracle headroom, live work vs FIFO, per-run peak memory.

## Engineering smoke (declared)

Unscored smoke (64 `wp4_confirm` prompts, T=4) on GPU0/GPU1: Gate-0 audit PASS on both models (max |score diff| ≤ 3e-6, identical cohorts) and all five policies ran with 0 hard-check errors. Outputs stay in the session scratchpad.
