# Phase 04R — CPU rank-stability executor task

This task uses only already committed Phase-04R receipts. It must not launch a GPU job.

Read first:
1. `phase04/DEPTH_RANK_STABILITY_CPU_PLAN.md`
2. `phase04/results/20260917T060824Z_50910e6_bf16_resolution/CAUSE_REVIEW.md`
3. `phase04/results/20260917T060824Z_50910e6_bf16_resolution/PROPOSED_B2_CONTRACT.md`
4. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md`

## Work package 1 — receipt integrity

Load Ouro and Huginn `per_token_validation.csv` plus matching manifests. Verify schema, row counts, uniqueness of `(model,B,order,token_idx,request_id)`, expected B/order coverage, and request-token identity intersections used for cross-condition comparisons. Fail closed on missing/duplicate identities.

## Work package 2 — deterministic CPU analysis

Implement and unit-test the metrics preregistered in `DEPTH_RANK_STABILITY_CPU_PLAN.md`:
- signed and absolute depth-delta distributions;
- per-token-step Kendall tau-b and Spearman rank correlation;
- preserved/tie/inverted pair classification by canonical depth gap;
- request-token live-depth range across observed B/order conditions;
- executed-condition live-work/RIC summaries without inventing a live counterfactual Oracle.

Tests must cover ties, constant-depth ready pools, all pair outcomes, empty gap bins, missing-condition intersections, and duplicate-key rejection.

## Work package 3 — publish, then stop

Commit the analysis source before using it to generate result artifacts. Run it on CPU only. Publish a new result root:

`phase04/results/<timestamp>_<source_sha>_rank_stability_cpu/`

Required outputs are the CSVs, two descriptive figures per model family as specified in the plan, a manifest, exact command, and `RESULT_REVIEW.md`.

The review must answer, separately for Ouro and Huginn:
1. What fraction of all positions changes exact depth?
2. Among changed positions, how large is `|delta depth|`?
3. How stable is within-token-step ordering?
4. How often do pairs invert, and how does that depend on canonical depth gap?
5. Across B/order variants already executed, what is the maximum per-position live-depth range?
6. Do the current traces support a ranking/propensity target more strongly than exact-depth regression?

Do not use an arbitrary pass threshold. Do not activate the B2 contract or resume predictor collection automatically. End with one of: `supports ranking-contract discussion`, `mixed evidence`, or `ordering materially unstable`, each backed by the recorded metrics.

## Hard prohibitions

- no CUDA/model loading;
- no new B16/B32/B64 trace;
- no predictor training/selection;
- no FIFO/Oracle speed rerun;
- no threshold/q/checkpoint/KV-policy changes;
- no rewriting or deleting Phase-04R evidence;
- no describing B1 depth as unique ground truth.
