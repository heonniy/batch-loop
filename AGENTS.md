# AGENTS.md

## Current priority — characterize physical Oracle upper bound (2026-09-16)

Read in order:
1. `phase02/ORACLE_UPPER_BOUND_PLAN.md` — next experiment/analysis contract; do this before predictor work.
2. `phase02/physical_results_20260916/RESULT_REVIEW.md` — physical survivor-compaction evidence motivating the new priority.
3. `phase02/PHYSICAL_EARLY_EXIT_IMPLEMENTATION.md` — canonical runtime contract.
4. `phase02/AGENT_TASK.md` — completed Phase-2 implementation checklist and reproduction details.
5. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — result/provenance rules that still apply.

### Research target

The physical no-refill survivor runtime now works for both model families. The immediate question is no longer whether survivor compaction is implementable; it is **how much batching headroom can exist on real hardware before building a practical predictor**.

Do **not** move directly to a learned/practical depth predictor yet. First characterize the Oracle upper bound over a sufficiently large offline ready pool and depth distributions that actually expose scheduling freedom.

The current physical smoke used `N=32` requests and often `B=16`, leaving only two microbatches. That setting is intentionally small and can suppress the benefit of depth-compatible grouping, especially for Huginn. `N=B=32` has no reordering freedom across microbatches and is a required near-zero control, not an Oracle ceiling.

### Primary physical-cost metric

Add and report **recurrent invocation count (RIC)** alongside slot utilization:

```text
RIC = sum over microbatches m of max_depth(m)
```

At a token boundary, a microbatch incurs one recurrent-core invocation for every loop step up to its maximum member depth. Physical smoke showed that wall/core reduction tracked removal of whole recurrent-core invocations much more strongly than reduced active width alone.

For FIFO and Oracle on the exact same selected requests/work, report:

```text
fifo_ric
oracle_ric
oracle_ric_reduction_pct = 100 * (fifo_ric - oracle_ric) / fifo_ric
```

Keep slot utilization and active-width curves as secondary diagnostics.

### Next characterization — trace first, GPU second

#### Huginn priority

Use the shared-KV, B=1 adaptive trace path to obtain per-token depths under:

- `Rmax=16`
- convergence thresholds `0.50, 0.35, 0.26, 0.12`
- ready-pool sizes `N=64,128,256` (add `512` only if trace availability/cost is reasonable)
- microbatch sizes `B=4,8,16,32`
- decode horizon at least `64` tokens; use `128` where already available/cheap

The goal is specifically to expose Huginn's wider recurrence-depth diversity. Do not infer that the existing threshold-0.5/N32 smoke is Huginn's upper bound.

#### Ouro control/comparison

Use existing/native-reference Phase-1 depth traces where possible:

- thresholds `q=0.2,0.4,0.5,0.7`
- ready-pool sizes `N=64,128,256`
- `B=4,8,16,32` where divisible/meaningful

Ouro remains useful as a positive/control family, but the main motivation for this characterization is to test whether Huginn's wider depth support yields more removable recurrent invocations once the ready pool is large enough.

### Trace-only output required before GPU runs

For every `(model, dataset/trace source, threshold, N, B)` cell, compute at each token boundary and aggregate:

- depth histogram, mean/std, and fraction at each depth;
- number of microbatches;
- FIFO `RIC`;
- Oracle `RIC`;
- Oracle RIC reduction percentage;
- FIFO/Oracle slot utilization;
- number/fraction of microbatches that terminate before `Rmax`;
- distribution of microbatch maximum depths;
- same selected request/token work checksum for FIFO and Oracle.

Write a compact ranked table sorted by Oracle RIC reduction. This is the go/no-go filter for expensive physical runs.

### Physical confirmation policy

Do not brute-force the whole grid on GPUs. First rank trace cells, then physically confirm a small set containing:

1. the strongest Huginn Oracle-RIC cells;
2. a moderate-gain cell;
3. a near-zero control such as `N=B`;
4. comparable Ouro cells.

Target cells with clearly larger RIC reduction (ideally 20–30%+ if the real trace supports it), but **do not manufacture or assume such a regime**. If the real trace does not provide it, report that negative result.

For physical confirmation:

- GPU 6 remains Ouro only.
- GPU 7 remains Huginn only.
- Use the existing physical survivor runtime and model-specific KV backends unchanged unless a bug is found.
- Prefer `N>=128` and especially `N=256` for the offline setting.
- Use enough decode tokens (32/64+) and warmed repeats to suppress one-token noise.
- Record recurrent-core GPU time, decode wall time, RIC, active-width curves, compaction/KV overhead, and number of actual recurrent-core calls.

### Model-specific KV/runtime contract remains unchanged

- **Ouro-1.4B:** physical survivor compaction + last-exited depth-indexed KV.
- **Huginn-3.5B:** physical survivor compaction + shared physical-layer KV.
- Pure no-refill only; no slot refill inside a recurrent interval.
- Stable logical request↔KV-row IDs must be preserved while active tensors shrink.
- Huginn head-dim-96/padding path must continue passing the existing systems checks.

### Scope boundary

Continue to exclude native/full-depth quality comparison, GSM8K, perplexity, and accuracy-recovery work from this phase. The objective is systems opportunity characterization.

Do not start practical predictor optimization until the Oracle physical opportunity has been characterized under large ready pools and multiple Huginn thresholds. Once the strongest physical Oracle regime is known, predictor work should target recovering that measured upper bound rather than the small N32 smoke result.

### Evidence rules

Keep these evidence classes separate:

- trace-only RIC/headroom characterization;
- forced-depth physical survivor timing;
- live B=1 exit-trace collection;
- physical FIFO vs Oracle timing;
- future practical-predictor timing.

Never call trace-only RIC reduction a speedup. Never extrapolate a 1.2×/1.3× physical gain unless measured.

### Commit/publication protocol

1. Make additive changes; preserve Phase-1 and existing Phase-2 results.
2. Bind source SHA, model revision, KV backend, exit trace hash/source, N, B, threshold, GPU identity, and workload in every physical run manifest.
3. Preserve negative and zero-gain cells.
4. Publish compact characterization tables and a result review before changing the research priority to predictor design.
5. Do not invent completed runs, speedups, cache semantics, or missing measurements.

Preferred subjects: `phase02: characterize ...`, `phase02: run oracle ...`, `phase02: publish oracle ...`, `phase02: review oracle ...`.
