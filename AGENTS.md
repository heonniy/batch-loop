# AGENTS.md

## Current priority — physical early-exit runtime (2026-09-16)

Read in order:
1. `phase02/PHYSICAL_EARLY_EXIT_IMPLEMENTATION.md` — canonical implementation contract.
2. `phase02/AGENT_TASK.md` — concrete coding/execution checklist for GPUs 6 and 7.
3. `phase01/results/20260916_four_gpu_validation/RESULT_REVIEW.md` — evidence that motivates this phase.
4. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — result/provenance rules that still apply.

### Research target

Turn the existing trace/headroom evidence into a **physical no-refill runtime**.  Inside one token-generation recurrent interval, tokens that exit must be removed from subsequent recurrent compute so the actual recurrent tensor batch shrinks.  No freed slot is refilled until the next token boundary.

The scheduling variable remains micro-batch formation at token boundaries.  The runtime mechanism is survivor compaction after each recurrent step.

### Model-specific KV contract

Do **not** force Ouro and Huginn to share one KV design.

- **Ouro-1.4B first:** physical survivor compaction + **last-exited KV**.  Keep depth-indexed recurrent KV slots.  When a request exits at depth `r`, fill its skipped recurrent-depth KV slots `r+1..Rmax` from the last actually computed recurrent KV for that request.  Do not compute the skipped recurrent blocks merely to create KV.
- **Huginn-3.5B:** physical survivor compaction + **shared KV**.  Keep one recurrent KV slot per physical recurrent layer/request and overwrite it on every recurrence.  Do not allocate recurrence-indexed KV slots for the shared backend.

Prelude/coda remain model-specific boundary stages.  For pure no-refill, run prelude once, repeatedly run the recurrent core while shrinking the survivor set, then run coda after the current recurrent interval is drained.

### Scope boundary for this phase

The user explicitly wants to focus on the systems implementation, **not native/full-depth quality comparison**.  Do not spend this phase on GSM8K, perplexity, accuracy recovery, or proving that last-exited/shared KV reproduces native full-depth model quality.

Still perform implementation correctness checks that are necessary to trust timing: stable request↔KV-row bookkeeping, deterministic/reproducible smoke tests, valid shapes, every request exits exactly once, exited requests receive zero later recurrent-core calls, and cache updates go to the intended request/layer/slot.

### GPU ownership

Use **physical GPU 6 for Ouro** and **physical GPU 7 for Huginn**.  Do not use GPUs 0–5 for Phase 2.  Every runner must fail closed unless exactly its assigned GPU is visible.  Use separate worktrees and result roots for the two lanes; never hot-patch or overwrite Phase-1 result roots.

Recommended bindings:
- GPU 6: Ouro-1.4B survivor runtime, last-exited KV, FIFO/Oracle physical timing.
- GPU 7: Huginn survivor runtime, shared KV, kernel/path repair as needed, FIFO/Oracle physical timing.

### Required evidence separation

Keep these labels separate:
- trace/logical headroom;
- forced-depth physical compaction smoke;
- live-exit physical runtime;
- physical FIFO vs Oracle no-refill timing;
- later practical-predictor timing.

Do not call a trace replay an integrated runtime.  A physical result requires the recurrent core to actually receive the reduced survivor batch on later recurrences.

### Commit/publication protocol

1. Make additive changes; preserve all Phase-1 code/results.
2. Bind source SHA, model revision, KV backend, exit source (`trace` or `live`), GPU identity, and workload in every run manifest.
3. Record active batch size and recurrent-core time per recurrence, plus compaction/KV/gate overhead.
4. Preserve failures and unsupported cells.
5. Publish compact results + `RESULT_REVIEW.md` + `review_manifest.json` before making speed claims.

Preferred subjects: `phase02: implement ...`, `phase02: run ...`, `phase02: publish ...`, `phase02: review ...`.

Do not invent completed runs, speedups, cache semantics, or missing measurements.
