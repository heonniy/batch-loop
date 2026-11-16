# Phase 05 WP1 — B2 decision and preregistered semantic Gate A

Committed **before** any Phase-05 GPU execution.

## Research-contract decision (2026-10-17)

The user explicitly **approved `phase04/results/20260917T060824Z_50910e6_bf16_resolution/PROPOSED_B2_CONTRACT.md` without amendment** and authorized **WP1 only** on physical H100 GPUs 0, 1, 2, 3 (`phase05/GPU_ASSIGNMENT_20261017.json`).

Consequences:
- The B1 BF16 depth trace is a **canonical frozen workload**, not a depth every legal batch must reproduce. Depth and next-token flips versus B1 are **measured outcomes**, not correctness failures.
- The original BF16 Gate A (`phase04/GATE_A_PARITY_RULE.md`) remains **FAIL**; this document does not reinterpret it.
- WP2 (physical headroom), WP3 (predictor) and any speed/Oracle claim stay blocked until a separate decision after the WP1 review.

## Cohort and fixed policy

| Item | Ouro-1.4B | Huginn-3.5B |
| --- | --- | --- |
| Revision | `37bc7e8ff85a90d813dc096383405f521b6a67b0` | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` |
| Imported model source SHA-256 | `59e7523e…c447b` | `a1d447da…f84aa` |
| Precision | BF16 | BF16 |
| Threshold / Rmax | q=0.2 / 4 | 0.12 / 16 |
| KV policy | last-exited depth-indexed | shared physical layer |
| h0 | n/a | request-keyed truncated normal, seed 1 |

- N=32 fresh synthetic prompts `Phase05 confirmation item 2000..2031` (`phase05/wp1_confirmation.py:fresh_prompts`), prompt cap 64. Effective-input hashes must be 32 unique and **disjoint** from the old Gate-A cohort and the Phase-04R `Validation request 1000..1031` cohort.
- T=8 decode positions, greedy argmax, no EOS. B1 per-request reference generates the token IDs; every live condition is teacher-forced with those IDs.
- Live conditions, all B=8, strict no-refill (membership fixed per interval, exited rows removed only):
  `fifo`, `permutation_seed_41`, `permutation_seed_53`, and `fifo_repeat` (same-process duplicate of `fifo`).
- Lanes: A = Ouro GPU0, B = Huginn GPU1, C = Ouro GPU2 (replica), D = Huginn GPU3 (replica). Independent processes; no TP/DP/EP.

## Semantic Gate A (B2) — hard checks; any failure = lane FAIL, lane stops

Applied to the B1 reference and to every live condition:

1. **Provenance**: physical index + UUID match the assignment with exactly one visible ordinal; BF16 weights; model revision and imported source SHA match the table.
2. **Input identity**: 32 unique effective-input hashes, disjoint from both earlier cohorts.
3. **Row identity**: every (token, request) executed exactly once per condition; each cohort's members equal the declared order slice; no missing/duplicate rows.
4. **No-refill widths**: per cohort, width[0]=B, widths non-increasing, `len(widths)=max depth`, and width at recurrence r equals #members with depth ≥ r; KV rows equal request rows.
5. **Finite**: all exit scores, logits and captured states finite.
6. **Exit comparator/cap**: depth independently recomputed from recorded scores (Ouro first `cumulative ≥ q`; Huginn first `score < threshold`; else Rmax) equals recorded depth; 1 ≤ depth ≤ Rmax.
7. **KV bookkeeping**: zero KV slot-set/shape/valid-mask structural mismatches versus B1 for each request.
8. **h0 identity (Huginn)**: every (request, position) h0 SHA-256 equals the B1 reference.
9. **Duplicate control**: `fifo_repeat` bit-identical to `fifo` in depths, next tokens, scores and logits.

## Measured outcomes (reported, never used to pass/fail)

Depth flips and signed delta histogram vs B1; next-token flips; live work vs B1 work; actual live RIC; frozen-B1-depth RIC of the same grouping; hidden/logit/KV relative-L2 and max gate-score difference; per-position reference margins.

Preregistered **ordering-stability flags** (inform the WP2 decision, not correctness): any `|live − B1| ≥ 2`, or any within-token-step inversion of a request pair whose B1 depth gap ≥ 2 → `ordering materially unstable`; otherwise `ordering stable at observed scope`.

**Replica agreement** (A vs C, B vs D): fraction of bit-identical depths/next tokens/scores across GPUs, reported as a measured outcome. A replica that fails its own hard gate is a FAIL of that lane.

## Outputs

`phase05/results/<UTC timestamp>_<source_sha>_wp1/lane_{A,B,C,D}_*/` with `gate_report.json`, `per_token.csv`, `condition_summary.csv`, `run_manifest.json`, then `replica_agreement.json`, `RESULT_REVIEW.md`, `review_manifest.json`. All trials, including failures, are retained.
