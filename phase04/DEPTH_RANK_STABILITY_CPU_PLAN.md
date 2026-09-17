# Phase 04R — CPU-only depth-rank stability analysis plan

Date: 2026-09-17
Base evidence commit: `a68d0902172a36b08e17bbacb4667e4558d6a515`
Status: **analysis plan only; no GPU execution authorized or required by this plan**.

## Why this analysis exists

Phase 04R showed that BF16 batch shape can change a small number of live adaptive-exit decisions. The open research question is narrower than exact depth parity:

> Even when exact exit depth changes by one or more recurrence steps, is the **relative recurrence-demand ordering** of requests preserved well enough to support depth-aware microbatch formation?

This analysis must be completed from the already committed Phase-04R per-token receipts before adopting the proposed B2 contract or restarting predictor collection. It does **not** retroactively pass the original BF16 Gate A, and it does **not** by itself approve Branch B2.

## Data contract

Use only already committed receipts from:

- `phase04/results/20260917T060824Z_50910e6_bf16_resolution/ouro_fresh_validation/per_token_validation.csv`
- `phase04/results/20260917T060824Z_50910e6_bf16_resolution/huginn_fresh_validation/per_token_validation.csv`
- their matching `batch_shape_validation.csv`, `validation_manifest.json`, and review files.

The CSV schema already contains `B`, `order`, `token_idx`, `request_id`, `reference_depth`, `live_depth`, depth/next-token flip flags, and exit-margin information. Treat `reference_depth` as the canonical B1 profiling depth from this diagnostic, **not** as a unique batch-invariant true depth.

Do not generate new model traces, alter thresholds, infer missing rows, or merge nonmatching request/token identities. Verify expected row counts and uniqueness of `(model, B, order, token_idx, request_id)` before analysis. If receipt integrity fails, publish BLOCKED rather than repairing data silently.

## A. Exact-depth perturbation magnitude

For each model and each `(B, order)` condition, compute

`delta = live_depth - reference_depth`.

Report:
- total positions and exact-match rate;
- histogram of signed delta;
- histogram of absolute delta;
- among flipped positions only: mean/median/max `abs(delta)`;
- counts/fractions with `abs(delta) == 1`, `== 2`, and `>= 3`;
- directionality: shallower live exit (`delta < 0`) versus deeper live exit (`delta > 0`);
- the same summaries stratified by reference exit-margin bin already recorded in the receipts.

Do not summarize the observed evidence as “±1 noise” unless the complete committed receipts support that statement. Preserve every larger perturbation if present.

## B. Rank preservation inside the actual scheduling unit

The scheduler compares requests that are ready at the **same decode token step**, so rank metrics must be computed within each `token_idx` ready pool, never by flattening all token positions into one global ranking.

For every `(model, B, order, token_idx)` pool:
- Kendall tau-b between `reference_depth` and `live_depth` as the primary tie-aware correlation;
- Spearman correlation as a secondary summary when defined;
- number of distinct depth levels in reference and live labels;
- exact per-request depth agreement.

Aggregate across token steps with median, interquartile range, min/max, and a request-pool bootstrap confidence interval. Do not treat all pairwise comparisons as statistically independent samples.

If a pool is constant-depth and correlation is undefined, report it as `NA_constant`, not zero. Also report the fraction of pools for which tau is defined.

## C. Pairwise scheduling-order stability

Within each same-token-step pool, consider only request pairs whose canonical B1 depths differ. For pair `(i,j)` with `D_ref_i < D_ref_j`, classify the live result as:

- **preserved**: `D_live_i < D_live_j`;
- **collapsed_to_tie**: `D_live_i == D_live_j`;
- **inverted**: `D_live_i > D_live_j`.

Report preserved/tie/inversion rates overall and by canonical depth gap:

- gap = 1
- gap = 2
- gap = 3
- gap >= 4

Empty bins remain explicit `NA`. Because Ouro has Rmax=4, large-gap bins may naturally be empty.

This gap-conditioned view is a primary output. The key scheduling question is whether clearly separated shallow/deep requests invert, not whether neighboring depths occasionally swap or tie.

Also compute a tie-aware pairwise concordance score:

`concordance = (preserved + 0.5 * collapsed_to_tie) / comparable_pairs`

as a descriptive metric only; do not present it as a universal quality threshold.

## D. Stability across legal observed batch conditions

For each canonical `(token_idx, request_id)` present in all receipts, summarize observed live depth across:

- B = 2, 4, 8;
- FIFO and the committed row permutations available for each B.

Report:
- per-position min/max live depth and range;
- fraction with range 0, 1, 2, and >=3;
- same-B order sensitivity: max depth minus min depth across order variants;
- cross-B sensitivity using a fixed order when matching observations exist.

This answers whether the small exact-depth changes are isolated threshold flips or whether a request's observed recurrence demand can move substantially across the already executed legal shapes.

Do not extrapolate these results to B16/B32/B64; no new GPU trace is authorized by this plan.

## E. Relationship to the batching objective

From existing receipts only, report structural diagnostics that do not invent an unexecuted live counterfactual:

- actual live total recurrence work and actual live RIC already recorded for each executed `(B, order)` condition;
- canonical B1 work and frozen-work grouping RIC from the committed summaries;
- absolute and percentage change in actual live work relative to the canonical B1 work;
- correlation between condition-level depth-flip count and live-work delta, labeled descriptive because the number of conditions is small.

**Do not** sort requests by canonical depth and then call RIC computed from another grouping's observed live depths a “live Oracle.” Changing composition can itself change live depth, so that counterfactual is not observed. The Phase-04R proposed B2 contract remains the governing warning here.

## F. Required figures/tables

Produce compact research-facing outputs:

1. `depth_delta_histogram.csv` — signed/absolute depth perturbations by model/B/order.
2. `rank_stability_by_step.csv` — tau-b, Spearman, distinct levels, agreement per token step.
3. `pairwise_gap_stability.csv` — preserved/tie/inverted rates by depth-gap bin.
4. `position_depth_range.csv` — per request-token min/max/range across observed conditions.
5. `rank_stability_summary.csv` — model-level and condition-level aggregates.
6. One depth-delta histogram figure per model.
7. One gap-vs-pairwise-preservation figure per model.
8. One concise `RESULT_REVIEW.md` explaining whether current evidence supports the statement “exact depth is slightly batch-sensitive but coarse recurrence-demand ordering remains stable.”

Figures are descriptive; raw CSVs are authoritative.

## G. Interpretation rules

This CPU analysis has three possible outcomes:

### G1 — ordering is strongly preserved
If flips are small and concentrated near neighboring depth levels, with very few/no large-gap inversions, record that the current evidence supports **recurrence-demand ranking** as a more stable target than exact-depth regression. This is evidence for considering an amended B2/ranking contract, not automatic approval.

### G2 — exact depths are noisy but coarse ordering is mixed
If nearby ranks are unstable and some larger-gap inversions occur, report the gap-dependent stability honestly. A later predictor may need coarse buckets/distributions rather than a single scalar rank, but no method change is authorized by this analysis alone.

### G3 — ordering is materially unstable
If large-gap inversions or multi-step depth ranges are substantial, do not resume the planned ranking predictor on the assumption of stable recurrence propensity. Return to the research-contract decision before any GPU collection.

Do not invent a post-hoc numeric pass threshold. Report the complete metrics, then make the research decision explicitly.

## H. GPU rule and next action

**No GPU is needed or authorized for this analysis.** Use CPU/Python only on committed CSV/JSON/Markdown artifacts. Do not touch GPU 2, 3, 5, or 7 and do not inspect/interrupt unrelated GPU jobs unless a later task explicitly requests GPU execution.

After publishing the CPU result/review in a new additive result root, stop. Do not collect predictor data, train predictors, rerun speed benchmarks, run B16/B32/B64, or activate the B2 contract automatically. The next step is an explicit research decision based on this analysis.

## Provenance and publication

Implement the analysis in a small deterministic script under `phase04_analysis/` (or another clearly named additive Phase-04 path), with tests for ties, constant-depth pools, pair classifications, missing cells and duplicate identities.

Publish source before derived results. Result root format:

`phase04/results/<timestamp>_<source_sha>_rank_stability_cpu/`

Record input file SHAs, analysis source SHA, Python/package versions, exact command, and row counts. Preserve all Phase-04R raw receipts unchanged.
