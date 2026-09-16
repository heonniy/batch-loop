# Phase 02 — Physical Early-Exit Runtime

Date: 2026-09-16
Parent evidence: `phase01/results/20260916_four_gpu_validation/RESULT_REVIEW.md`

## Goal

Phase 1 established a large **logical** batching opportunity, but the repository still lacks the mechanism needed for a physical speed claim: after a token exits at recurrent depth `r`, later recurrent steps must actually execute on a smaller tensor batch.

This phase implements that missing runtime.  The goal is not to redesign the exit rule or study model quality.  The goal is to answer:

> If exited tokens are physically removed from recurrent compute, does depth-compatible micro-batch formation reduce measured H100 recurrent time / BCT beyond ordinary FIFO no-refill execution?

The key comparison is:

`FIFO + physical survivor compaction` vs `Oracle depth grouping + the same physical survivor compaction`.

`Native full-depth` may be reported as a reference, but it is not the scheduling baseline because it mixes the benefit of early exit itself with the benefit of batch formation.

## Global runtime invariant

Pure no-refill only:

```text
token boundary
    -> form microbatch
    -> prelude
    -> recurrent step 1
    -> decide exits
    -> compact survivors
    -> recurrent step 2
    -> ...
    -> recurrent interval drains
    -> coda
    -> next token boundary
```

A slot freed inside the recurrent interval is never filled by a different token until the next token boundary.

The implementation must make the recurrent-core input shape genuinely shrink, e.g. `B=16 -> 11 -> 6 -> 2`.  Merely masking rows or recording an exit flag while still executing the full `B=16` tensor is not accepted as physical early exit.

## Runtime abstraction

Implement a small common runtime interface with model-specific adapters.  Names may differ, but keep the conceptual split:

```text
NoRefillRuntime
  - form_microbatch(...)
  - run_prelude(...)
  - run_recurrent_step(active_ids, ...)
  - decide_exit(...)
  - update_exit_kv(...)
  - compact_survivors(...)
  - run_coda(...)

KVBackend
  - OuroLastExitedKV
  - HuginnSharedKV
```

Keep a stable logical `request_id` / global cache-row identity.  Survivor compaction changes the **active compute view**, not the logical identity of a request.  Avoid copying the whole persistent KV cache just to compact a recurrent batch; use active row indices / gather views / page-style indirection where practical.

## Ouro lane — GPU 6

### Model

Start with Ouro-1.4B only.  Preserve its learned gate and current threshold semantics.  Rmax remains 4.

Initial thresholds: `q = 0.2, 0.4`.
Initial physical batch sizes: `B = 8,16,32`.

### Physical exit

The public Ouro HF path currently computes every recurrence and selects the gate-chosen state post hoc.  The new adapter must instead expose one recurrence at a time:

```text
active hidden
  -> one full recurrent-core pass
  -> gate for this recurrence
  -> save exited state
  -> remove exited rows from active hidden / input state
  -> next recurrence only for survivors
```

Do not execute a later recurrent pass for an exited request.

### Ouro KV backend: last-exited

Use a depth-indexed recurrent KV layout.  If request `i` exits after recurrence `r`, its actually computed KV exists through `r`.  For every skipped recurrent depth `s in [r+1, Rmax]` and every physical recurrent layer, materialize the logical slot for `(i,s)` from the last actually computed recurrent KV `(i,r)`.

Conceptually:

```text
exit depth = 2
r1: actual
r2: actual
r3: copy r2
r4: copy r2
```

The purpose is to make later-token attention well-defined while still skipping recurrent compute.  This phase does not ask whether this policy matches native/full-depth quality.

Do not silently replace this with shared KV on Ouro.

## Huginn lane — GPU 7

### Model

Huginn-3.5B, Rmax=16.  Use the latent-diff convergence rule for live-exit mode.

Initial thresholds: `0.5, 0.35` (add `0.26` after the runtime is stable).
Initial physical batch sizes: `B = 8,16,32`.

### Physical exit

The upstream adaptive path records per-sequence exits but continues to call `iterate_one_step` on the full batch until all rows have exited.  Replace that behavior in the Phase-2 adapter:

```text
active_ids = all rows
repeat:
    recurrent step(active rows only)
    convergence check(active rows only)
    save exits
    active_ids = survivors
until empty or Rmax
```

The tensor passed to the recurrent core on the next step must contain survivors only.

### Huginn KV backend: shared

Use one recurrent KV slot per physical recurrent layer and logical request.  Every recurrent pass overwrites that request's existing slot for that physical layer.  On exit, the last written value remains the request's recurrent KV; do not create missing deeper-depth slots and do not copy to depth-indexed slots.

Conceptually:

```text
request A exits at r2 -> shared slot contains r2 KV
request B exits at r7 -> shared slot contains r7 KV
```

Implement this backend independently of the earlier `latest-m4` memory probe so the physical runtime has a clear one-slot-per-physical-layer contract.

### Current Huginn kernel issue

Phase 1 hit the batched attention error:

`NYI: Currently non power of 2 embedding dimension are not supported. Got E=96 and Ev=96.`

The Phase-2 Huginn lane must isolate and repair/bypass this execution path before timing.  Use an attention backend/path that supports head dimension 96 on H100.  Do not interpret this kernel failure as a model limitation.

Also ensure padded multi-prompt execution is not silently attending to padding.  A smoke test must exercise heterogeneous prompt lengths.

## Exit source modes

Implement both modes because they answer different questions.

### 1. `trace` / forced-depth mode — required first

Use pre-recorded per-token exit depths (or a deterministic synthetic depth vector for the smallest smoke).  This is the first physical proof because it isolates survivor compaction and KV bookkeeping from gate timing/noise.

This mode is also the clean physical FIFO-vs-Oracle upper-bound comparison: every strategy executes the same token-loop work; only micro-batch composition differs.

### 2. `live` mode — required before Phase 2 closes

Run the model's actual exit rule after each recurrence and compact from the resulting per-row decisions.

- Ouro: learned gate after each recurrent pass.
- Huginn: latent-diff convergence check.

Do not claim the live mode is implemented if the code only replays depths.

## Implementation milestones

### M0 — deterministic compaction smoke

Before full generation, hard-code or load a tiny depth vector such as `[1,2,4,2]` and prove:

```text
r1 active ids: 0,1,2,3
r2 active ids:   1,2,3
r3 active ids:     2
r4 active ids:     2
```

Required assertions:
- active set is monotonically non-increasing;
- each request exits exactly once;
- an exited request is never passed to a later recurrent block;
- output scatter restores logical request order;
- KV writes/copies/overwrites target the intended logical request rows.

### M1 — one-token physical timing

Use native/model-specific prefill, then execute exactly one decode recurrent interval with the Phase-2 runtime.

Compare:
- full recurrent execution;
- FIFO + survivor compaction;
- Oracle + survivor compaction.

This milestone establishes whether the logical headroom becomes actual H100 recurrent-time headroom before investing in long generation.

### M2 — multi-token no-refill

Run fixed 16/32 generated-token workloads with the model-specific KV backend across token boundaries.

Record BCT, generated tokens/s, and recurrence-level active widths.  Do not add refill.

### M3 — live exit

Repeat a bounded subset in live-exit mode.  The first publication-quality scheduling comparison may still use trace replay for identical-work fairness, but live mode must demonstrate that the same runtime can consume actual model exits.

## Scheduling comparisons

For initial physical measurements implement only:

- `fifo`: request-id/order baseline;
- `oracle`: sort/group by the current token's recorded exit depth in trace mode.

Do not spend Phase-2 implementation time on a new predictor until physical FIFO-vs-Oracle timing proves that the hardware gap is worth recovering.

Later add predictor grouping only after this gate.

## Required measurements

For every recurrent interval, log at minimum:

- original microbatch size;
- `active_batch_size` for each recurrence;
- logical request IDs in each active set (debug receipt; may be disabled for large runs after validation);
- recurrent-core GPU time per recurrence;
- gate/check time;
- compaction/index-select/scatter time;
- KV copy time (Ouro) or shared-KV overwrite/update time (Huginn);
- prelude time;
- coda time;
- total token-step time;
- peak allocated and reserved HBM.

For the full run report:
- offline BCT;
- generated tokens/s;
- requests/s;
- scheduler/runtime overhead as a fraction of BCT;
- FIFO-vs-Oracle speedup under identical selected requests and recorded exit work.

Synchronize CUDA around benchmark boundaries.  Use warmup and repeated measurements for the final timing packet; the first M0/M1 smoke can be single-pass.

## GPU / worktree contract

Only these physical devices may be used:

```text
GPU 6 -> Ouro lane
GPU 7 -> Huginn lane
```

Every execution script must bind exactly one device and record physical identity with `nvidia-smi` in its manifest.  Fail if the intended physical GPU is not the one exposed.

Use separate worktrees/output roots, for example:

```text
../batch-loop-phase02-ouro-gpu6
../batch-loop-phase02-huginn-gpu7

results/phase02_physical_exit_<timestamp>/gpu6_ouro/
results/phase02_physical_exit_<timestamp>/gpu7_huginn/
```

Never delete, overwrite, or repurpose `results/four_gpu_model_validation_20260916_114421`, `phase01/results`, or the GPU0 Huginn diagnosis.

## Out of scope

Do not spend this phase on:
- GSM8K / perplexity / quality recovery;
- native/full-depth quality equivalence;
- refill scheduling;
- new exit-head training;
- sophisticated depth predictors;
- multi-GPU model parallelism.

The implementation question comes first: **does a physically shrinking recurrent batch convert the Phase-1 oracle headroom into measured H100 time savings?**

## Phase-2 completion gate

Phase 2 is successful when both lanes have a working physical survivor path and the result packet can truthfully show recurrence traces such as:

```text
FIFO:   16 -> 13 -> 7 -> 3
Oracle: 16 -> 16 -> 15 -> 11
```

alongside the measured per-step/total H100 times produced by those actual shapes.

If the physical FIFO-vs-Oracle gap is negligible, document that negative result before investing in predictor work.  If it is material, the next research block is to recover the Oracle gap with a practical causal depth predictor.
