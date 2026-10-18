# Phase 05 WP5 — stronger causal current-token ranker (preregistration)

Committed **before** ranker training on the full train split and before any wp5_test array is read. Authorization: user 2026-10-18 ("더 강한 ranker로 개발해줘") after WP4 (serving cost fixed; ranking quality now binding).

## Development evidence that motivated the design (train/validation only)

- Token-ID target encodings added to the WP3 fused features gain only +1–2.6 pp validation RIC recovery (exploration, WP3 train/val arrays).
- WP3 features compress 2048-d (Ouro) / 5280-d (Huginn) states into 64-d random projections. WP5 re-collects train/validation with **full-dimensional** pre-recurrence states. The B1 rerun reproduces WP3 labels bit-identically.

## Ranker R (`phase05/wp5_ranker.py`)

- Inputs, all available before the current token's recurrence: the Phase-03 scalar block (38); token-ID target encodings of canonical B1 depth (current, previous, hashed bigram; mean and log-count; fit on wp3_train with 5-fold out-of-fold encodings for train rows); and full-dimensional states. The states are the current-token embedding and the previous token's final recurrent state, plus the Prelude output for Huginn. Each state goes through a learned Linear→LayerNorm projection, then an MLP produces the score.
- Training: weighted pairwise logistic loss within a token step on 256-request chunks (Phase-03 loss form), AdamW, seed 1, ≤ 20 epochs, early stop with patience 4.
- **Selection on wp3_validation only:** grid {proj 64 / width 256 / lr 1e-3; proj 128 / width 256 / lr 3e-4; proj 256 / width 512 / lr 3e-4}; pick max mean lockstep RIC recovery over B ∈ {8,16,32,64}. A no-full-dim ablation is reported. Checkpoint SHA-256 is committed before the physical run.

## Gate 0 (wiring audit, untimed, wp3_validation cohort 0, B=8, T=64)

The ranker is served live on the GPU (`wp4_gpu_serving.gpu_run`, policy `ranker_gpu`, no host sync). Its per-step scores are compared with offline scores of the same frozen ranker on the B1-collected validation arrays. **PASS** if mean within-step Spearman ≥ 0.95 (BF16 batch-shape differences allowed) and there are no hard-check errors. A lane failing Gate 0 is not scored.

## Sealed physical test

- Fresh split `wp5_test` = pool `[2816, 3328)` (collected with the train/val re-collection; arrays not read before the physical run), 2 cohorts × 256 per lane; lanes A–D as before.
- Policies: `fifo`, `history_last`, `predictor_gpu` (WP3-frozen C7/C8/C9 with WP4 GPU serving), `ranker_gpu` (R), `oracle_b1`. B ∈ {8,16,32,64}, 3 repeats, one warmup, rotated and shuffled order, all runs retained.

## Decision rules (fixed now)

- **Primary (unchanged rule, applied to `ranker_gpu`):** it wins a cell if its headroom vs FIFO is ≥ 5% and all 3 repeats are faster than all 3 FIFO repeats. **Practical GO per model** if it wins ≥ 2 of 4 B values in each of the 4 dataset×cohort blocks.
- **Stronger-ranker hypothesis — supported** if both hold: (a) median `ranker_gpu` wall is below median `predictor_gpu` wall in ≥ 75% of the 32 cells; (b) post-hoc offline wp5_test recovery of R exceeds that of the WP3 predictor in all 4 lanes.
- Also reported: recovery of oracle headroom, gain over `history_last`, RIC-to-wall translation, live work vs FIFO, the validation grid and the ablation.

## Engineering smoke (declared)

Unscored smoke (64-request splits, T=4, 2 epochs) on GPU0 (Ouro×Alpaca). The collector produced full-dim arrays and the WP3 labels reproduced; ranker training ran; Gate-0 audit PASS (Spearman 0.9996); all five policies ran with 0 hard-check errors. Outputs stay in the session scratchpad.
