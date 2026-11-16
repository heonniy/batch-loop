# Phase 05 WP5 review — stronger causal current-token ranker R

**Decision (preregistered):**
- **Ouro-1.4B: practical ranker GO.** R wins B=8, 16 and 32 in all 4 dataset×cohort blocks of the sealed `wp5_test` split. It saves **10.0–14.8%** decode wall at B=8/16 and 6.5–8.6% at B=32 versus FIFO, with all repeats faster than all FIFO repeats.
- **Huginn-3.5B: NO-GO.** R saves 2.9–4.7%, below the 5% rule, in every cell.
- **Stronger-ranker hypothesis: supported.** R is faster than the GPU-served WP3 predictor in **32/32** cells. Its offline test recovery is higher in all 4 lanes (Ouro 26–29% vs 17–18%; Huginn 35–37% vs 28–30%).
- This is the first rule-passing practical result in Phase 05. Its scope is this eager runtime, lockstep no-refill, fixed B and the WP1 thresholds.

## Provenance

- Preregistration, ranker, GPU serving and analysis: `506523e`, committed before full training and before any `wp5_test` array was read.
- Full-dimensional collection: `fa5272d`. The B1 rerun reproduced the WP3 train/validation labels bit-identically in all 4 lanes.
- Rankers frozen at `2fe3f0f` (checkpoint SHA-256 in each `ranker_selection.json`). Lane driver: `0596e31`.
- **Gate 0 wiring audit: PASS 4/4.** On wp3_validation cohort 0 at B=8, live GPU-served scores match offline scores from the B1 arrays with mean within-step Spearman 0.9989–0.9996.
- Physical: 4 lanes × 121 runs = **484/484 OK**, 0 OOM, 0 hard-check failures. Fresh split `wp5_test` (pool 2816–3327), 2 cohorts × 256 per lane.
- Policies: fifo, history_last, predictor_gpu (WP3 C7/C8/C9, WP4 serving), ranker_gpu (R), oracle_b1. B ∈ {8,16,32,64}, 3 repeats.

## What made the ranker stronger (validation only)

| Lane | R | R without full-dim states | WP3 predictor | Offline **test** R / WP3 (post hoc) |
| --- | ---: | ---: | ---: | --- |
| Ouro×Alpaca | 30.1% | 17.2% | 19.0% (C7) | 29.3% / 18.2% |
| Huginn×Alpaca | 36.3% | 28.6% | 28.1% (C8) | 37.4% / 29.7% |
| Ouro×ShareGPT | 27.3% | 16.8% | 17.0% (C9) | 26.2% / 17.4% |
| Huginn×ShareGPT | 32.7% | 25.5% | 24.4% (C8) | 34.6% / 27.5% |

Most of the gain comes from replacing 64-d random projections with **full-dimensional states**: the current-token embedding, the previous token's final recurrent state and, for Huginn, the Prelude output, each through a learned low-rank projection. Token-ID encodings add a little. Validation and test agree within about 1–2 pp. The grid choice mattered little (spread ≤ 1 pp).

## Physical sealed test: decode-wall saving vs FIFO (median of 3; * = R wins the cell)

| Lane / cohort | B=8 | B=16 | B=32 | B=64 |
| --- | --- | --- | --- | --- |
| **Ouro×Alpaca c0** hist / WP3 / **R** / oracle | 2.4 / 9.7 / **12.5\*** / 34.3% | 1.8 / 7.7 / **12.7\*** / 34.7 | 0.8 / 4.5 / **8.6\*** / 29.9 | −0.1 / 0.6 / **3.8** / 23.2 |
| Ouro×Alpaca c1 | 4.8 / 11.3 / **14.8\*** / 34.9 | 3.1 / 8.2 / **11.9\*** / 34.9 | −1.7 / 2.7 / **6.9\*** / 30.8 | −0.6 / −0.2 / **1.7** / 23.5 |
| **Ouro×ShareGPT c0** | 2.6 / 6.5 / **11.0\*** / 31.9 | 2.5 / 6.7 / **10.4\*** / 33.2 | 0.5 / 4.4 / **7.2\*** / 30.1 | −0.4 / 0.3 / **2.1** / 21.8 |
| Ouro×ShareGPT c1 | 3.0 / 6.7 / **11.0\*** / 32.0 | 1.8 / 5.9 / **10.0\*** / 32.4 | 1.1 / 3.5 / **6.5\*** / 29.5 | 0.5 / 1.6 / **3.1** / 21.0 |
| Huginn×Alpaca c0 | 1.4 / 1.8 / **3.1** / 14.4 | 1.8 / 3.1 / **4.7** / 15.4 | 1.9 / 3.2 / **4.6** / 14.3 | 1.1 / 2.5 / **3.4** / 11.2 |
| Huginn×Alpaca c1 | 1.6 / 1.9 / **3.0** / 13.5 | 2.3 / 3.0 / **4.6** / 15.1 | 1.8 / 3.5 / **4.6** / 14.4 | 1.0 / 2.5 / **3.5** / 11.0 |
| Huginn×ShareGPT c0 | 1.5 / 2.1 / **3.1** / 14.5 | 1.8 / 2.6 / **3.8** / 15.7 | 2.1 / 2.9 / **4.3** / 15.3 | 2.0 / 3.5 / **4.1** / 12.5 |
| Huginn×ShareGPT c1 | 1.7 / 1.9 / **2.9** / 14.0 | 2.2 / 2.3 / **3.6** / 15.3 | 2.4 / 2.4 / **3.7** / 14.3 | 2.1 / 2.4 / **3.3** / 11.8 |

Repeat spreads are tight; for example, Ouro×Alpaca c0 B16 FIFO took 88.5–88.8 s and R 76.6–77.8 s. One Huginn×Alpaca c0 repeat of every policy ran about 10% slow. It is retained, and it does not change any median or the rule outcome. Live work differs from FIFO by < 0.07% everywhere, so the comparison is equal-work.

## Translation and remaining gap

| Model | Wall share of oracle headroom: WP3 / **R** | RIC→wall translation: WP3 / **R** / oracle |
| --- | --- | --- |
| Ouro | 0.15 / **0.27** | 0.76 / **0.85** / 0.99 |
| Huginn | 0.19 / **0.27** | 0.43 / **0.51** / 0.75 |

- **Ouro:** with GPU serving, R converts RIC savings to wall almost one-for-one, so ranking quality is what limits it. R captures ~27% of the oracle wall headroom. Its gain decreases with B: at B=64 there are only 4 cohorts per step, and the oracle's advantage comes from separating a few deep tokens.
- **Huginn:** two independent limits remain. First, a smaller oracle headroom (11–16%). Second, weaker translation (0.51), because Huginn's 16 recurrences per step make the per-recurrence trajectory hook a larger fixed cost. Reaching 5% would need roughly R's ranking quality plus near-oracle translation, or a ranker around 40–45% recovery.

## Five-axis review

| Axis | Status |
| --- | --- |
| Measurement | VALID: 484/484, two fresh sealed cohorts per lane, tight spreads, all runs retained |
| Implementation | Gate-0 wiring PASS 4/4; hard checks 484/484; labels reproduce bit-identically |
| Simulator | None for the decision; offline recovery is post hoc only |
| Comparison | Same inputs, B, staging, serving path for both predictors; equal live work |
| Hypothesis | Stronger ranker **supported**; **Ouro GO** (B=8–32), **Huginn NO-GO** |

## Limits

- One fresh test split per lane: 2 × 256 requests, synthetic-length prompts capped at 64 tokens, T=64, no EOS.
- One threshold per model, eager runtime without CUDA graphs, lockstep scheduling.
- The headroom applies only when B is constrained. B=N=256 remains faster in absolute terms (WP2).
- Training used 1,024 requests per dataset. The ranker is model- and dataset-specific; cross-dataset transfer was not tested.
