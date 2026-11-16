# Phase 05 WP3 review — causal current-token predictor on a sealed test

**Decision (preregistered rule): practical predictor = NO-GO for both models.** The validation-selected predictors only win at B=8, and only for Ouro. Ouro saves **6.4–7.8%** decode wall against FIFO at B=8, winning in all 4 dataset×cohort blocks. Huginn never wins a cell (≤ 1.9%). The rule needs ≥ 2 winning B values per block. Oracle headroom is unchanged from WP2: 21–36% (Ouro) and 11–16% (Huginn). The predictors recover **at most ~23%** of it, matching Phase 03's ≈18–21% under a different (trace-forced) contract.

## Provenance

- Preregistration `WP3_PREREGISTRATION.md` and runner `79d0e8f`, committed before collection. Selection was frozen at `33e6ade` (checkpoints + SHA-256) before any test array was read. Analysis `84fb759` was committed before the physical results were read.
- Collection: per-request B=1 live greedy, T=64. Per lane: 1,024 train, 512 validation, 512 test requests. Feature arrays (214 MB) stay on local disk; SHA-256 is in each committed `feature_manifest.json`. Labels and tokens are committed.
- Physical: 4 lanes × 97 runs = **388/388 OK**, 0 OOM, 0 hard-check failures. Two disjoint test cohorts of 256 per lane; fifo / history_last / predictor / oracle_b1; B ∈ {8,16,32,64}; 3 repeats. Live BF16 exits and strict no-refill. Prelude staging is common to all policies, and feature, predictor and sort time is inside the timer.

## Selection (validation only) and post-hoc offline test check

| Lane | Winner | Validation mean RIC recovery | Offline test recovery (winner) | C0 = previous depth (test) |
| --- | --- | ---: | ---: | ---: |
| Ouro×Alpaca | C7 ordinal | 19.0% | 18.5% | 3.6% |
| Huginn×Alpaca | C8 pairwise | 28.1% | 29.8% | 13.5% |
| Ouro×ShareGPT | C9 listwise | 17.0% | 16.9% | 3.7% |
| Huginn×ShareGPT | C8 pairwise | 24.4% | 24.4% | 11.4% |

Validation matched test closely, so selection did not overfit. Fused current-token features (Prelude/embedding projection + uncertainty + history) beat history alone by 2–5×. Full table: `analysis/offline_test_recovery.csv`; validation: `selection/*/validation_tournament.csv`.

## Physical sealed test: decode-wall saving vs FIFO (median of 3; * = wins the cell)

| Lane / cohort | B | FIFO | history_last | **predictor** | oracle_b1 | predictor share of oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ouro×Alpaca c0 | 8 | 163.9 s | 1.1% | **7.8%\*** | 34.1% | 23% |
| | 16 | 88.9 s | −0.1% | 7.1% | 35.6% | 20% |
| | 32 | 46.7 s | −0.5% | 1.7% | 30.1% | 6% |
| | 64 | 25.3 s | 1.2% | −2.5% | 21.2% | — |
| Ouro×Alpaca c1 | 8 / 16 / 32 / 64 | | 2.4 / 1.2 / −0.2 / 0.8% | **7.8\*** / 6.7 / 3.2 / −2.0% | 33.6 / 34.3 / 30.6 / 22.8% | |
| Ouro×ShareGPT c0 | 8 / 16 / 32 / 64 | | 2.7 / 1.3 / 1.3 / 0.6% | **6.4\*** / 4.7 / 2.7 / −2.3% | 32.1 / 32.4 / 30.0 / 21.4% | |
| Ouro×ShareGPT c1 | 8 / 16 / 32 / 64 | | 2.5 / 2.5 / 0.6 / −1.1% | **6.6\*** / 5.3 / 1.6 / −2.9% | 32.1 / 33.3 / 30.4 / 22.1% | |
| Huginn×Alpaca c0 | 8 / 16 / 32 / 64 | | 1.8 / 2.2 / 2.7 / 2.0% | 1.2 / 1.9 / 1.0 / −0.5% | 14.0 / 16.2 / 15.2 / 12.4% | |
| Huginn×Alpaca c1 | 8 / 16 / 32 / 64 | | 2.1 / 1.8 / 1.8 / 1.8% | 0.9 / 0.8 / 0.6 / −1.3% | 14.1 / 14.8 / 13.8 / 10.9% | |
| Huginn×ShareGPT c0 | 8 / 16 / 32 / 64 | | 1.6 / 1.4 / 1.6 / 1.9% | 0.3 / 0.0 / −0.8 / −1.2% | 13.6 / 14.4 / 13.2 / 10.9% | |
| Huginn×ShareGPT c1 | 8 / 16 / 32 / 64 | | 2.1 / 2.1 / 2.5 / 1.7% | 0.6 / 0.6 / 0.0 / −1.4% | 14.1 / 15.5 / 15.2 / 12.1% | |

The two test cohorts agree within ~1.5 percentage points everywhere.

**Ouro B=16 misses on the all-repeats rule.** Its median clears 5% in 3 of 4 blocks (7.1 / 6.7 / 4.7 / 5.3%). In each Ouro block one predictor repeat was a slow outlier (93.9 / 94.6 / 95.0 s against 81.7–83.9 s for the others); one FIFO repeat was also slow (101.7 s). All repeats are retained and the rule is applied as written. Even counting B=16, the rule would not be met in Ouro×ShareGPT c0 (4.7%).

## Where the RIC saving goes (equal work: |Δ live work| < 0.05% everywhere)

| Lane (c0) | B | RIC saving pred / hist / oracle | Wall saving pred / hist / oracle |
| --- | ---: | --- | --- |
| Ouro×Alpaca | 8 | 9.8 / 2.0 / 32.6% | 7.8 / 1.1 / 34.1% |
| Ouro×Alpaca | 32 | 5.6 / 0.6 / 31.3% | 1.7 / −0.5 / 30.1% |
| Huginn×Alpaca | 8 | 5.1 / 2.1 / 16.2% | 1.2 / 1.8 / 14.0% |
| Huginn×Alpaca | 32 | 7.5 / 3.5 / 21.1% | 1.0 / 2.7 / 15.2% |

The oracle and history_last turn RIC savings into wall savings almost one-for-one. The predictor loses **≈ 2–6 percentage points** of wall time. Feature extraction + inference + sort measures **0.6–3.8%** of decode wall (`overhead_share`), and the share grows with B because there are fewer, shorter steps. The rest is in-loop bookkeeping that only the predictor needs: per-recurrence trajectory capture (C7/C8/C9 read previous convergence/gate trajectories, which forces a host sync per recurrent step) and copying full-vocabulary logits for the uncertainty features. On Huginn this cancels the predictor's larger RIC saving, so **history_last (≈ 2%) beats the predictor in wall time**. These mechanisms are a diagnosis, not a measured decomposition; they were not separately timed.

## Five-axis review

| Axis | Status |
| --- | --- |
| Measurement | VALID: 388/388 runs, two disjoint sealed cohorts, 3 repeats, all outliers retained |
| Implementation | Row/width hard checks pass; serving and collection use the same `_features` function |
| Simulator | Offline recovery reported only as a post-hoc reference; decision is physical |
| Comparison | Same inputs, B, staging, threshold, KV; equal live work |
| Hypothesis | Current-token predictors capture ~17–30% of oracle ordering offline but ≤ 8% wall; **NO-GO** by rule |

## Interpretation and limits

1. Ranking quality limits the gain: offline recovery is ≤ 30% of oracle RIC. Raising it is a model and feature problem, not a tuning problem; Phase 03 and WP3 agree on the ceiling of this feature family.
2. Serving cost is a second, independent limit. A predictor without trajectory/logit-sync features, or with them computed on the GPU without host syncs, could keep more of its RIC saving. That would need a **new preregistered study**; these results do not support it post hoc.
3. Scope limits as in WP2: eager runtime, lockstep, one threshold, 64-token prompt cap, no EOS, and B=N already fastest when B is unconstrained.

## Decision

Per `WP3_PREREGISTRATION.md`: **practical predictor NO-GO for Ouro and Huginn on this runtime.** The recorded positive is narrow: Ouro at B=8, 6.4–7.8% wall, replicated across 2 datasets × 2 sealed cohorts. Any follow-up (sync-free features, B=8-specific deployment, an optimized engine) needs a new explicit decision and preregistration.
