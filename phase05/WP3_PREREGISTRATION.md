# Phase 05 WP3 — causal current-token predictor (preregistration)

Committed **before** any scored WP3 collection. Authorization: user 2026-10-17 ("WP3 제안한 방향대로 진행해줘") after the WP2 review (oracle headroom 10–35%, history policies ≤ 2.6%).

## Prior evidence this builds on (not repeated blindly)

Phase 03 (`phase03/predictor_tournament_results_20260916/RESULT_REVIEW.md`) trained the same C0–C9 family under a **frozen trace-forced depth** contract, synthetic-split N640/T32: best pairwise C8 recovered ≈18–21% of oracle RIC; physical wall recovery Ouro +11%, Huginn negative. WP3 re-tests this family under **B2 live exits**, fresh request-disjoint Alpaca/ShareGPT splits, T=64 and a B grid, with test sealed until selection is frozen. The feature function and model code are reused unchanged (`phase03_predictor.physical._features`, `phase03_predictor.tournament._train_one`/`Head`, `phase03_predictor.physical.Predictor`), so collection and serving compute identical features.

## Data and labels

- Splits from `phase05/wp2_headroom.py:split_prompts` (declared in WP2, untouched so far): `wp3_train` 1,024, `wp3_validation` 512, `wp3_test` 512 requests per dataset. WP2's `wp2_headroom` cohort is not used.
- Per-request B=1 live greedy run, T=64, prompt padded/truncated to 64, BF16, WP1 contract (Ouro q=0.2/Rmax 4; Huginn 0.12/Rmax 16, request-keyed h0 with canonical id = request index mod 256). Label = canonical B1 exit depth (B2). Features are computed after the Prelude/embedding and **before** the current token's recurrence: position, context length, own depth history, previous-token output uncertainty (surprisal/entropy/top1/margin), previous-token convergence/gate trajectory, 64-d seeded projections of the token embedding and Prelude output (Huginn) / previous final state (Ouro).
- Lanes: A Ouro×Alpaca GPU0, B Huginn×Alpaca GPU1, C Ouro×ShareGPT GPU2, D Huginn×ShareGPT GPU3 (CPU-pinned as WP2). Large feature arrays stay on local disk with SHA-256 in committed manifests; labels/tokens/manifests are committed.

## Selection (validation only; test arrays never opened)

Candidates C0 (previous depth = history_last) and C1–C9 exactly as Phase 03, same width/lr variants, seed 1, ≤20 epochs with patience 5. **Winner per lane = max mean frozen-label lockstep RIC recovery over B ∈ {8,16,32,64} on wp3_validation** (cohort blocks of 256); tie → fewer parameters. Checkpoints and `selection.json` (with SHA-256s) are committed before the physical test starts; `physical` refuses changed checkpoints.

## Physical sealed test

- Two independent test cohorts per lane: `wp3_test[0:256]`, `wp3_test[256:512]`.
- Policies: `fifo`, `history_last` (own previous live depth), `predictor` (frozen winner, live features from this run's own batched execution), `oracle_b1` (B1 test labels; clairvoyant reference). B ∈ {8,16,32,64}, 3 repeats, one warmup, interleaved order (B shuffled per rep/cohort, policies rotated per rep).
- Common staging for every policy: per decode step the Prelude/embedding of all N rows runs in fixed FIFO chunks of B, then each policy groups rows for core+Coda (Phase-03 staging). Live exits, teacher-forced B1 tokens, strict no-refill; per-run row/width hard checks stop the lane on failure.
- The decode wall timer includes feature extraction, predictor inference, sorting, logit/trajectory bookkeeping (predictor only) and all GPU stages.

## Decision rules (fixed now)

Per lane, cohort and B: headroom(policy) = 1 − median(policy)/median(fifo).
- **Predictor wins a cell** if headroom(predictor) ≥ 5% and all 3 predictor repeats are faster than all 3 FIFO repeats.
- **Practical GO (per model)** if the predictor wins ≥ 2 of the 4 B values in **each** of the 4 dataset×cohort blocks. Otherwise **NO-GO** for a practical predictor on this runtime.
- Also reported, not deciding: headroom vs `history_last`, recovery = headroom(predictor)/headroom(oracle_b1), live RIC/work vs FIFO (|Δwork| flagged if > 1%), feature+predict overhead share, and offline frozen test recovery of every candidate (computed only after the physical run).

## Engineering smoke (declared)

Unscored smoke (64 requests/split, T=4, 2 epochs) on GPU0/GPU1 verified collect → select → physical before this commit; outputs remain in the session scratchpad.
