# Phase 05 — Offline predictive cohort scheduling (proposal / CPU prototype)

Status: **CPU source committed; no GPU benchmark performed; Phase-04 BF16 release gate remains FAIL**.

## Research question

Under STRICT no-refill, can request/next-token cohort selection at the decode boundary reduce actual physical recurrent work and offline makespan relative to CDB no-refill FIFO? Predict a relative remaining-depth ordering, not an exact batch-invariant BF16 depth. Do not change the model exit policy.

This phase is additive: do not modify or reinterpret Phase-01–04 artifacts.

## Semantics

- One cohort is chosen before a recurrent interval.
- Surviving tokens shrink inside this cohort. **No refilling, inter-cohort survivor merging, or intra-recurrence regrouping.**
- Prelude -> shared recurrent core (shrink-only) -> Coda, with one request-token per member.
- The same request cannot start its next token until the preceding cohort has finished Coda.
- Regroup at **decode boundaries / cohort creation**, never within an active cohort.
- Dynamic ranking can change as newly completed request-tokens update their history.
- Grouping must not alter gate thresholds, checkpoint, KV semantics, Rmax, output sampling policy or model weights.

## Already completed work — do not repeat it

Read repository AGENTS.md and:
- phase04/results/20260917T065413Z_e4113e5_rank_stability_cpu/RESULT_REVIEW.md
- phase04/results/20260917T060824Z_50910e6_bf16_resolution/CAUSE_REVIEW.md
- phase04/results/20260917T060824Z_50910e6_bf16_resolution/PROPOSED_B2_CONTRACT.md

The CPU rank review found limited one-step BF16 exit perturbations and mostly preserved shallow/deep ordering on N32/T8 synthetic evidence. This does **not** authorize a new predictor, GPU trace collection or claimed wall-time gain. The original Gate A is still FAIL; the proposed B2 contract is only a draft.

## CPU prototype

phase05/frozen_cohort_sim.py is an **abstract serial cost / frozen-label scheduling test**, NOT an integrated inference runtime. Supported policies:

- fifo: oldest ready request-token first.
- static_first: sort by the first previously observed depth, fixed thereafter.
- history_last: sort by preceding generated token's observed depth.
- history_mean: sort by the completed depth history mean.
- history_long_first: descending preceding depth for critical-chain prioritization.
- oracle_depth: **clairvoyant current-token depth**, available only in frozen replay.

All policies recompute groups when a cohort completes; only oracle_depth can use current-token future information. They are evaluated on a common **frozen** work matrix, so RIC differences are structural diagnostics, not physically measured performance. The oracle_depth heuristic is not necessarily the globally optimal schedule because request dependencies constrain readiness.

Run CPU unit tests:

    python -m unittest -v phase05.test_frozen_cohort_sim

Example with the prior Phase-04R per-token *B1 reference* depths, filtered to one repeated condition:

    python -m phase05.frozen_cohort_sim \
      --input phase04/results/20260917T060824Z_50910e6_bf16_resolution/ouro_fresh_validation/per_token_validation.csv \
      --model ouro --source-B 2 --order fifo \
      --batch-size 8 \
      --output phase05/local_outputs/ouro_B8_frozen.json

Do not commit arbitrary output from this 32-request synthetic demo as a new speed result. The simulator rejects the batch-dependent live_depth column as a transferable Oracle label. It does not simulate BF16 numerical changes induced by different live batch groupings, GPU parallel execution, cache traffic, CUDA Graph shape effects or scheduling overhead. These must be measured separately.

## H100 x4 experimental blueprint (future work, after gate approval)

The four available H100s are **four independent validation lanes**, NOT assumed to be a four-GPU model-parallel serving configuration. Confirm the exact physical GPU indices, UUIDs, memory occupancy and exclusive ownership at launch. Previous Phase-04 assignments are historical: never reuse them without a new authorization. Use an explicit per-job CUDA_VISIBLE_DEVICES mask; never kill or modify other users' processes.

Logical lanes once permitted:

| Lane | Workload | Purpose |
| --- | --- | --- |
| A | Ouro-1.4B | BF16 exit/KV correctness gate, FIFO vs sorting pilot |
| B | Huginn-3.5B | shared physical-layer KV correctness gate, no-refill pilot |
| C | Ouro-1.4B separate GPU | same-model validation/replication, row-order and source-shape sensitivity |
| D | Huginn-3.5B separate GPU | same-model validation/replication, row-order and source-shape sensitivity |

If memory or checkpoint access does not allow four simultaneous independent model instances, use fewer lanes and document the limitation; do not reassign without verification.

**Hard stop before any new GPU run or predictor training:** a human research decision must explicitly approve/amend/reject the prospective B2 (batch-dependent-depth) contract. Then preregister validation rules and run a fresh request-disjoint N32/B8/T8 confirmation. If it fails, report FAIL and stop the lane. Merely having H100 GPUs available is not sufficient to treat Gate A as passed.

After the gate:
1. Freeze model revision, dtype BF16, Rmax, threshold, KV policy, seed and exact tokenized split before the speed runs.
2. Measure workload depth distribution, within-request next-token depth correlation, and by-token-step coarse ordering.
3. Compare CDB No-refill FIFO, random, history_last, history_mean, static_first and frozen-depth Oracle at fixed B and identical generation limits. For physical tests compare **actual live** RIC, work, output differences, decode/E2E wall, scheduler overhead, HBM and OOM separately. Oracle must execute its own proposed grouping to be called live.
4. Sweep initially N=256, B=4/8/16/32/64, Alpaca and ShareGPT, fixed output length (64 initially), representative **preselected** exit thresholds. Include B=1 and N=B no-composition controls where feasible.
5. Record at least three whole-model repetitions per scored cell once correctness passes. Report every trial, including failures. Never present native Ouro HF post-hoc-gate timing as physical early exit.
6. Only if these checks establish genuine speed headroom: develop lightweight **causal pre-recurrence** depth-propensity predictor, compare ranking methods and sealed test cohorts. No after-loop re-grouping.

## Required evidence and publication

Commit prospectively frozen source/config and bounded plan **before GPU execution**. Write immutable results under phase05/results/<timestamp>_<source_sha>/, with:
- MODEL/KV/EXIT/DEVICE manifest, input hashes, seed and environment versions;
- raw per-token actual live depths, exit margins, next-token differences, work and actual live RIC;
- native/integrated/replay labels in distinct fields;
- CPU schedule and predictor overhead, measured core/Prelude/Coda/copy timing, whole decode and E2E timing;
- OOM/error/correctness outcomes, no selective trial filtering;
- RESULT_REVIEW.md and review_manifest.json according to phase01/EXPERIMENT_REVIEW_PROTOCOL.md.

Interpret frozen replay strictly as structural potential, and measured BF16 live schedules as potentially **different actual work**. No theorem or speed claim follows from the CPU prototype alone.

## GO / STOP decision

- Proceed to predictor development only after a validated practical FIFO-versus-Oracle physical headroom measurement, and successful exact trace identity/KV bookkeeping checks under the newly approved contract.
- If the system's full-size batch already saturates memory/compute efficiently, if depth ordering is too unstable, or if physical grouping cannot beat CDB No-refill, record the negative result and stop.
