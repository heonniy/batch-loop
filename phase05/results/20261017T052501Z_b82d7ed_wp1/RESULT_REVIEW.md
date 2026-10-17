# Phase 05 WP1 review — fresh N32/B8/T8 BF16 confirmation under approved B2

**Decision: semantic Gate A (B2) = PASS on all four lanes (Ouro A/C, Huginn B/D). Ordering-stability flag = `ordering stable at observed scope` for both models.** This authorizes *requesting* a WP2 decision; it does not start WP2 and is not a speed result. The original BF16 Gate A (`phase04/GATE_A_PARITY_RULE.md`) remains FAIL.

## Scope

- Contract: `PROPOSED_B2_CONTRACT.md` approved by the user 2026-10-17 without amendment; gate preregistered in [`WP1_PREREGISTRATION.md`](../../WP1_PREREGISTRATION.md) at `b82d7ed` before execution.
- Hardware: physical H100 GPU0 (A, Ouro), GPU1 (B, Huginn), GPU2 (C, Ouro replica), GPU3 (D, Huginn replica); UUIDs verified per lane in each `run_manifest.json`. Independent processes, no TP/DP/EP. GPUs 4–7 untouched.
- Cohort: 32 fresh synthetic prompts (`Phase05 confirmation item 2000..2031`), effective-input hashes unique and disjoint from the old Gate-A and Phase-04R cohorts. T=8, greedy, no EOS; B1 tokens teacher-forced into every live run.
- Conditions: B=8 strict no-refill `fifo`, `permutation_seed_41`, `permutation_seed_53`, `fifo_repeat`. 256 request-token positions each; the four conditions reuse one cohort and are not independent samples.

## Hard checks (all lanes, all conditions): 0 errors

Provenance (UUID, BF16, revision, imported source SHA, clean source), input identity/disjointness, row identity, no-refill widths and KV-row = active-request equality, finite scores/logits/states, independent exit-comparator/Rmax recheck, zero KV structural mismatches, Huginn h0 SHA-256 identity, and bit-identical `fifo_repeat` duplicate control. Per-lane detail: `lane_*/gate_report.json`.

## Measured outcomes (not gate criteria)

| Model | Condition | Depth flips /256 | Next-token flips /256 | B1 work → live work | Live RIC | Frozen-B1 RIC same grouping |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Ouro-1.4B q=0.2 | fifo | 0 | 4 | 621 → 621 | 105 | 105 |
| | perm 41 | 0 | 4 | 621 → 621 | 104 | 104 |
| | perm 53 | 0 | 3 | 621 → 621 | 105 | 105 |
| | fifo_repeat | 0 | 4 | 621 → 621 | 105 | 105 |
| Huginn-3.5B 0.12 | fifo | 3 (+1×2, −1×1) | 2 | 2119 → 2120 | 313 | 312 |
| | perm 41 | 5 (+1×4, −1×1) | 2 | 2119 → 2122 | 313 | 313 |
| | perm 53 | 5 (+1×4, −1×1) | 2 | 2119 → 2122 | 314 | 313 |
| | fifo_repeat | 3 | 2 | 2119 → 2120 | 313 | 312 |

- All depth deltas are ±1; zero within-step inversions of pairs with B1 depth gap ≥ 2.
- **Ouro next-token flips without depth flips** (4/256 in fifo, same depth both sides): the logit argmax changed under batched BF16. One has reference gate margin 0.009; the others have margins 0.09–0.15, so these are logit near-ties, not exit-gate effects. Max logit relative L2 0.030, hidden 0.14, KV 0.096 — larger than Huginn (0.031–0.041 / 0.039–0.053 / 0.028–0.034). These numerical differences are retained as measured outcomes under B2.
- **Cross-GPU replicas**: A≡C and B≡D bit-identical on 100% of 1,024 positions for depths, next tokens, scores and logit hashes.

## Structural context only (frozen B1 depths, not live, not a speed claim)

Re-grouping each token step's 32 requests by sorted frozen B1 depth into four B=8 cohorts gives RIC 105 → 91 (−13.3%) for Ouro and 312 → 279 (−10.6%) for Huginn. This is an **unexecuted frozen-work packing bound** on one synthetic cohort; live regroupings may change actual depth, and RIC is not wall time.

## Five-axis review

| Axis | Status |
| --- | --- |
| Measurement | VALID for bounded correctness; no wall-time scoring |
| Implementation | B2 semantic Gate A PASS (4/4 lanes); original BF16 Gate A still FAIL |
| Simulator | Not used for the gate; frozen RIC above is labeled structural |
| Comparison | Same input, same policy, B1 vs B8 live; replicas identical |
| Hypothesis | Ranking contract survives a fresh cohort at N32/B8/T8; ≥10% frozen packing headroom exists structurally |

## Limitations

Synthetic prompts only; one B; T=8; four conditions share one cohort; no non-synthetic corpus; no wall time; a frozen headroom of 10–13% RIC may shrink after Prelude/Coda and per-step fixed costs.

## Next decision required (not taken here)

Approve/amend WP2 (physical no-refill headroom: FIFO/random/static/history/live-executed Oracle, predeclared N256 split and B/threshold grid, 3 repeats) or stop.
