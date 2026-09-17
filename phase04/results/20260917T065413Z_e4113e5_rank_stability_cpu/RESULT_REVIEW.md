# Phase-04R CPU depth-rank stability review — 2026-09-17

**Decision: `supports ranking-contract discussion` for both model families, within this N32/T8 synthetic BF16 diagnostic only.** Exact live exit depth is not batch-invariant, but the observed one-step perturbations seldom reverse relative shallow/deep order. This is evidence to discuss a ranking/propensity target, **not** evidence that a practical predictor works, that physical speed improves, or that draft B2 is approved. The original BF16 Gate A remains **FAIL**.

## Scope and receipt integrity

This is `saved-data-analysis`: no model, CUDA, GPU, or new trace execution. The [analysis source](../../../phase04_analysis/depth_rank_stability.py) at `e4113e5` was committed before analysis; the exact command, Python/package versions, input SHA-256s, and output SHA-256s are in [`analysis_manifest.json`](analysis_manifest.json). The analyzed evidence was the previously committed fresh BF16 validation: each model has 32 distinct synthetic teacher-forced requests, eight decode steps, and nine repeated legal B/order conditions (B=2,4,8 × FIFO/seed-17/seed-29). These are **2,304 condition-position observations**, but only **256 distinct request-token positions** per model. The nine conditions reuse the same requests; they are not independent cohorts.

Both 2,304-row receipts passed required-schema, exact condition/request-token intersection, key uniqueness, reference consistency, saved flip-flag, margin, work, and actual/frozen RIC checks. There were 72 ready pools/model. `reference_depth` is the canonical **B1 profiling depth**, not a uniquely batch-invariant truth.

## Depth perturbation and cross-condition range

| Model | Changed observations | Signed deltas | `|delta|` among changes | Distinct changed positions | Live-depth range across all nine conditions |
| --- | ---: | --- | --- | ---: | --- |
| Ouro-1.4B, q=0.2 | 25/2,304 = **1.085%** | 3 shallower, 22 deeper | mean/median/max **1/1/1**; no `|delta|>=2` | 4/256 | 254 range 0, 2 range 1; max **1** |
| Huginn-3.5B, threshold=0.12 | 38/2,304 = **1.649%** | 29 shallower, 9 deeper | mean/median/max **1/1/1**; no `|delta|>=2` | 9/256 | 248 range 0, 8 range 1; max **1** |

The complete signed and absolute distributions are in [`depth_delta_histogram.csv`](depth_delta_histogram.csv); condition and reference-margin-bin denominators are in [`depth_delta_summary.csv`](depth_delta_summary.csv). In Ouro, **9/25** changed observations fall in the reference margin `[0.005,0.02)`, so a `<0.005` exception would miss them. All **38/38** Huginn depth changes fall below `0.005`. Those counts repeat some positions across conditions. No larger depth perturbation occurred in this cohort; that is an observation, not a bound for other batches or prompts.

The per-position [`range table`](position_depth_range.csv) includes same-B row-order ranges and fixed-order cross-B ranges. All of these axes had maximum range **1**; some rows were order-sensitive, especially Huginn (5 positions at B2, 1 at B4, 1 at B8), and cross-B changes reached 8 Huginn positions under seed-29 order. Range 0 does **not** guarantee B1 agreement: a position can differ from B1 in every batched condition yet be stable across those conditions.

## Within-token-step ordering, including ties

Each correlation was calculated separately over the 32 requests ready at one `token_idx`, never across decode steps. [`rank_stability_by_step.csv`](rank_stability_by_step.csv) gives every pool. The median Kendall tau-b is **1.0** for both models (Ouro: defined 72/72 pools, Q1 0.988, minimum 0.730; Huginn: defined 63/72, Q1 0.947, minimum 0.883). The nine undefined Huginn pools are constant-depth token-0 pools and are marked `NA_constant`, not zero. Median Spearman is also 1.0 for both. Request-ID-cluster bootstrap 95% percentile intervals for the model-level median tau-b are [1.0, 1.0] in both cases; this near-degenerate interval reflects many unchanged/tied pools and is **not** a population guarantee. Condition-level IQRs, extrema, defined fractions and intervals are in [`rank_stability_summary.csv`](rank_stability_summary.csv).

Comparable pairs are only those with *different* canonical B1 depths **within the same token step**. Across the nine repeated conditions, [`pairwise_gap_stability.csv`](pairwise_gap_stability.csv) records:

| Model / canonical gap | Preserved | Collapsed to tie | Inverted | Comparable |
| --- | ---: | ---: | ---: | ---: |
| Ouro, 1 | 10,876 | 209 | **3** | 11,088 |
| Ouro, 2 / 3 / ≥4 | 567 / 144 / NA | 0 / 0 / NA | **0 / 0 / NA** | 567 / 144 / 0 |
| Huginn, 1 | 12,130 | 404 | **3** | 12,537 |
| Huginn, 2 / 3 / ≥4 | 1,980 / 423 / 54 | 0 / 0 / 0 | **0 / 0 / 0** | 1,980 / 423 / 54 |

Thus the gap-1 inversion rates are **0.0271%** (Ouro) and **0.0239%** (Huginn) per repeated pair observation; gap≥2 had no observed inversions (Ouro 711; Huginn 2,457 repeated comparisons). All three Ouro inversions are the *same* request pair 9/10 at token 1 under B8's three orders. All three Huginn inversions are the *same* pair 10/31 at token 6 under B8's three orders. These are not six independent incidents. Tie collapse is more common than inversion and is retained explicitly. Ouro's gap≥4 bin is genuinely empty, not a zero-rate success. The figure files visualize the complete signed deltas and gap-conditioned pair outcomes for each model.

## Executed work and batching objective

The [`condition summary`](rank_stability_summary.csv) keeps canonical B1 work, actual live work, **actual executed-group RIC**, and frozen-B1-depth grouping RIC separate. Ouro B1 work is 636; actual live work is 637–639 (**+0.157% to +0.472%**) across conditions. Huginn B1 work is 2,118; live work is 2,114–2,118 (**−0.189% to 0%**). Actual live RIC ranges by executed width/order are Ouro B2 **341–343**, B4 **181–182**, B8 **91–94**; Huginn B2 **1,089–1,090**, B4 **557–559**, B8 **287–289**. The condition-level Pearson correlation of depth-flip count with live-work delta is 0.068 (Ouro) and −0.199 (Huginn), merely descriptive for nine non-independent conditions/model. There is no scored wall-time comparison here.

No live Oracle was constructed. Applying the observed depths from one executed grouping to a newly sorted grouping would create an **unexecuted** counterfactual, because changing membership may change live depths. The separate frozen B1-depth RIC is a fixed-work packing calculation only, not live adaptive speed or equal-work proof.

## Five-axis review and research decision

| Axis | Status | Reason |
| --- | --- | --- |
| Measurement | **VALID in bounded saved-data scope** | Complete 32×8×9 receipts/model, identities and work/RIC verified; no fresh wall times or non-synthetic cohort. |
| Implementation | **CHECKED_IN_SCOPE for CPU analysis; runtime gate still FAIL** | Deterministic metric tests cover ties, constant pools, pair outcomes, empty bins, missing and duplicate identities; no new runtime correctness run. |
| Simulator | **NOT_RUN** | Only executed live RIC and separately saved frozen-depth RIC are reported; no invented live regrouping. |
| Comparison | **COMPARABLE for same-input B1 profiling versus observed live conditions; not a speed comparison** | Same request/token IDs within model; conditions have potentially different actual work and reuse one cohort. |
| Hypothesis | **EXPLORATORY_SIGNAL: supports ranking-contract discussion** | Every observed flip is one step, coarse gaps never invert; adjacent pairs can tie or invert, and scope is N32/T8 B≤8. |

The evidence supports the narrower proposition that **relative recurrence-demand ordering is more stable than exact-depth equality in these already observed conditions**. It does not establish that ranking is intrinsically easier to predict than exact depth, that an unseen N256/B32 cohort behaves likewise, or that the physical Oracle headroom can be recovered. The next action is an explicit research-contract decision on whether and how to amend the draft B2/ranking direction. Until then, predictor collection/training, a new Gate A, and speed/Oracle experiments remain blocked; no GPU was used for this review.
