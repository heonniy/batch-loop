# Phase 02 — Oracle Upper-Bound Characterization Plan

## Why this block exists

Phase 2 has already crossed the key implementation gate: the recurrent tensor batch physically shrinks after exits, Ouro uses last-exited KV, Huginn uses shared KV, and multi-token physical FIFO/Oracle runs complete.

The first smoke results were intentionally small (`N=32`). They showed an important hardware fact: Oracle gains appeared mainly when grouping removed **whole recurrent-core invocations**, while merely reducing active width often changed H100 time only modestly.

Examples from the published smoke:

- Ouro q=0.4, one-token, `N=32,B=16`: FIFO executed 8 recurrent-core invocations and Oracle 7, matching about a 12.5% call-count reduction and about a 12.6% measured recurrent-time reduction.
- Huginn threshold 0.5, `B=8`: FIFO executed 17 invocations and Oracle 16, matching about a 5.9% call-count reduction and about a 5.9% measured recurrent-time reduction.
- `N=B=32` is almost invariant to sorting because there is only one microbatch; the measured Oracle gain was approximately zero.

Therefore the next question is not yet “how good is our predictor?” It is:

> **With a realistic large offline ready pool, how many recurrent-core invocations can perfect depth-compatible batching actually eliminate, especially on Huginn where depth can span a wider range?**

## Primary metric: recurrent invocation count (RIC)

For one token boundary, after forming fixed-size microbatches `m`, define:

```text
RIC = sum_m max(depth_i for i in m)
```

This counts how many times the recurrent core must be invoked across all microbatches in pure no-refill execution.

Example:

```text
Depths: [2,8,2,8], B=2

FIFO:
  [2,8] -> 8 invocations
  [2,8] -> 8 invocations
  RIC = 16

Oracle:
  [2,2] -> 2 invocations
  [8,8] -> 8 invocations
  RIC = 10

RIC reduction = 37.5%
```

RIC is not a replacement for physical timing, but the Phase-2 smoke indicates it is a much better first-order go/no-go metric than slot utilization alone for the current runtime.

## Hypothesis to test, not assume

Huginn supports a much wider recurrence-depth range than Ouro, but the existing physical Huginn smoke used a small ready pool and mostly threshold `0.5`. A wider raw depth range only helps if a sufficiently large ready pool lets the scheduler cluster shallow and deep requests into different microbatches.

Thus:

```text
larger depth diversity
        x
larger N/B (more microbatches to rearrange)
        -> potentially larger RIC reduction
```

This is a hypothesis. The experiment must be allowed to show that the expected 20–30% regime does not exist.

## Stage A — collect/freeze large B=1 exit traces

### Huginn

Use the existing Phase-2 shared-KV adaptive semantics and collect/freeze B=1 exit depths for:

```text
Rmax = 16
threshold = 0.50, 0.35, 0.26, 0.12
N = at least 256 requests in the trace pool
decode = at least 64 tokens; 128 preferred when practical
```

B=1 is the exit-depth reference. Do not use the upstream dense B>1 adaptive helper as the depth oracle.

For each threshold, publish:

- depth histogram;
- mean/std;
- p10/p50/p90 depth;
- fraction exiting at each depth;
- per-token-boundary number of available requests.

### Ouro

Prefer reusing existing Phase-1 traces rather than recollecting equivalent data. Use q=`0.2,0.4,0.5,0.7` with at least 256 requests where available.

## Stage B — trace-only N/B/RIC sweep

For each model/threshold trace, evaluate ready-pool prefixes or fixed fair cohorts:

```text
N = 64, 128, 256
(optional 512 only if trace pool permits cleanly)

B = 4, 8, 16, 32
```

Only use cells where `N` is divisible by `B` unless a common frozen remainder policy is explicitly implemented.

At every token boundary:

1. freeze the exact same logical requests for FIFO and Oracle;
2. FIFO order = stable request order;
3. Oracle order = exact current-token depth, then stable request id;
4. partition into B-sized microbatches;
5. compute each microbatch maximum depth;
6. accumulate RIC and secondary metrics.

Required output columns include:

```text
model
threshold
trace source/hash
N
B
num_tokens
num_microbatches
same_work_checksum
mean_depth
depth_std
fifo_ric
oracle_ric
oracle_ric_reduction_pct
fifo_slot_utilization
oracle_slot_utilization
fifo_mean_microbatch_max_depth
oracle_mean_microbatch_max_depth
fifo_num_full_rmax_microbatches
oracle_num_full_rmax_microbatches
```

Also save the distribution of per-token-boundary RIC reduction, not only the aggregate mean.

## Stage C — rank regimes before using GPUs

Create a compact ranking table.

Primary ordering:

```text
oracle_ric_reduction_pct descending
```

Select a small physical matrix containing:

- strongest Huginn cell(s), ideally at a threshold with visibly broad depth distribution;
- one moderate Huginn cell;
- one zero/near-zero control such as `N=B`;
- strongest comparable Ouro cell(s).

Do not select cells by slot-utilization gain alone.

A 20–30%+ RIC reduction is an interesting target for a stronger physical result, not a required or assumed outcome.

## Stage D — physical confirmation

Use the already implemented survivor runtime.

### GPU ownership

```text
GPU 6 -> Ouro only
GPU 7 -> Huginn only
```

### Runtime contracts

```text
Ouro  -> last-exited depth-indexed KV
Huginn -> shared physical-layer KV
pure no-refill
```

Do not change KV semantics between FIFO and Oracle.

Use preferably:

```text
N = 128 or 256
B selected from trace sweep
num_tokens = 32 or 64+
>= 2 warmups per strategy
>= 5 measured repeats for final characterization cells when cost permits
```

Record:

- total actual recurrent-core invocation count;
- sum of active request-steps;
- recurrent CUDA-event time;
- prelude/coda time;
- compaction time;
- KV update / exit-copy time;
- prefill-excluded decode wall time;
- generated tokens/s for the fixed decode contract;
- active-width sequence per microbatch;
- peak HBM.

The physical result must assert that FIFO and Oracle have the same selected requests and same total per-request recurrence work.

## Expected diagnostic interpretation

### Case 1 — RIC reduction tracks wall time

If trace Oracle reduces RIC by ~25% and physical recurrent time falls similarly, the research mechanism becomes much sharper:

> Depth-compatible batching improves no-refill execution primarily by making shallow microbatches terminate earlier, removing complete recurrent-core invocations.

Then predictor design should optimize expected microbatch maximum depth / RIC rather than only per-request depth MAE.

### Case 2 — RIC drops strongly but wall time does not

Then profile why:

- prelude/coda dominance;
- KV/compaction overhead;
- kernel launch/synchronization overhead;
- very small/large B efficiency curve;
- current Huginn SDPA math backend;
- attention/KV implementation costs.

Do not immediately blame the scheduling idea.

### Case 3 — real Huginn traces do not produce large RIC reduction

Then wider nominal depth range is not enough. Measure whether depths are concentrated within each token boundary or correlated in a way that prevents microbatch separation. This is a valid negative result and should be reflected in the problem framing.

## Important scheduling implication

The smoke suggests the scheduler objective should not necessarily be “maximize slot utilization.” A more hardware-relevant objective may be closer to:

```text
minimize sum_m max(predicted_depths_in_microbatch_m)
```

subject to batch-size/capacity constraints.

This observation may later change the practical predictor/scheduler design. Do not implement that optimizer yet; first establish the real Oracle RIC landscape.

## Scope exclusions

This block does not include:

- native/full-depth quality comparison;
- GSM8K/perplexity;
- KV-quality recovery;
- refill scheduling;
- learned predictor training;
- production serving claims.

The purpose is to establish the strongest honest physical Oracle opportunity before spending effort on prediction.

## Deliverables

1. Large B=1 Huginn depth traces for all four thresholds.
2. `oracle_ric_sweep.csv` or equivalent for both models.
3. Per-threshold depth-distribution tables/plots.
4. Ranked top physical-confirmation cells.
5. GPU6/GPU7 physical results for selected cells.
6. `RESULT_REVIEW.md` explaining whether larger N and Huginn diversity actually increase the physical Oracle bound.
7. A decision at the end: proceed to practical predictor design, adjust runtime/kernel implementation, or revise the problem setting.
