# Phase 02 — Coding Agent Task

Implement the physical early-exit runtime described in `PHYSICAL_EARLY_EXIT_IMPLEMENTATION.md`.

This is an implementation-first phase.  Do not divert into model-quality experiments.

## Non-negotiable constraints

- Preserve all Phase-1 files and result roots.
- Pure no-refill only; no slot refill inside a recurrent interval.
- GPU 6 is reserved for Ouro work.
- GPU 7 is reserved for Huginn work.
- Do not use GPUs 0–5.
- Ouro KV backend = last-exited depth-indexed KV.
- Huginn KV backend = one-slot-per-physical-layer shared KV.
- A row marked exited must not be executed by later recurrent-core passes.
- Do not report simulated/replayed latency as integrated runtime latency.

## Recommended code layout

Keep implementation additive.  A suggested layout is:

```text
phase02_runtime/
  __init__.py
  common.py
  scheduler.py
  benchmark.py
  ouro_adapter.py
  ouro_last_exited_kv.py
  huginn_adapter.py
  huginn_shared_kv.py

run_phase02_gpu6_ouro.sh
run_phase02_gpu7_huginn.sh
```

Names may differ if the design is cleaner, but retain explicit separation between common scheduler/runtime code and model-specific KV/model adapters.

Do not edit cached Hugging Face model source in place.  Prefer local adapters/subclasses/wrappers in this repository so the exact behavior is source-controlled.

## Step 1 — implement common survivor bookkeeping

Create a reusable active-set abstraction that tracks at least:

```text
logical request id
stable KV row id
current active tensor row
exit depth / exited flag
```

After each recurrent step:

1. receive a per-active-row exit mask;
2. scatter final recurrent state for newly exited requests into logical-request order;
3. apply model-specific exit-KV handling;
4. build survivor indices;
5. `index_select` / gather recurrent state and any recurrence-local inputs for survivors;
6. update active logical IDs;
7. run the next recurrence on the smaller tensors only.

Add assertions that no exited logical request appears in a later active set.

## Step 2 — forced-depth smoke before live gates

Implement `--exit-source trace` and a minimal synthetic mode.  The first test should not depend on the gate.

Example forced depths:

```text
[1, 2, 4, 2]
```

Expected active sizes:

```text
r1: 4
r2: 3
r3: 1
r4: 1
```

Record an event receipt with active logical IDs, not only counts.

This test must run on each model adapter before moving to multi-token timing.

## Step 3 — Ouro adapter on GPU 6

Target Ouro-1.4B first.

Expose the recurrent core as a one-recurrence callable instead of calling the public whole-model path that computes all UT steps.

For each recurrence:

```text
active recurrent state
 -> one recurrence over the shared recurrent layers
 -> learned gate for active rows
 -> exits / survivors
```

### Last-exited KV

Maintain depth-indexed recurrent slots.  When request `i` exits at `r`, copy/reference its last actually computed K/V for each recurrent physical layer into logical recurrence slots `r+1..Rmax` for request `i`.

Keep stable logical KV row IDs.  Do not compact/copy the entire persistent cache just because the active tensor shrinks.

Add counters/receipts:
- actual recurrent KV writes;
- last-exited fill operations;
- target request id / recurrence slot / physical layer for debug mode.

### Ouro initial benchmark cells

Use GPU 6 only.

Start with:
- q=`0.2,0.4`
- B=`8,16,32`
- one-token M1 first
- then fixed 16/32-token M2
- FIFO and Oracle in trace mode

Use existing Phase-1 traces when possible rather than recollecting equivalent Ouro traces.

## Step 4 — Huginn adapter on GPU 7

Target Huginn-3.5B, Rmax=16.

Do not reuse the upstream batched adaptive implementation as the final runtime: it marks exits but keeps the full tensor batch.

Implement one recurrence over the current active rows, run the convergence check, save exits, and gather survivors before the next recurrence.

### Shared KV

Implement a clean shared recurrent KV backend:

```text
[physical recurrent layer, logical request, token position, ...]
```

There is no recurrence-depth dimension in the recurrent shared backend.  Each recurrence overwrites the same physical-layer slot for the active logical request.

On exit, the most recent value simply remains in place.

### Kernel/path repair

The earlier Huginn B>1 path failed at head dimension 96.  Isolate the failing attention backend and select/implement an H100-compatible path for head_dim=96.  Add a small B=2/B=4 smoke before large timing.

Also test heterogeneous prompt lengths so padding/masking cannot silently corrupt batch execution.

### Huginn initial benchmark cells

Use GPU 7 only.

Start with:
- thresholds=`0.5,0.35`
- B=`8,16,32`
- one-token M1 first
- then fixed 16/32-token M2

If no suitable Phase-1 Huginn trace exists, collect a small B=1 exit trace with the Phase-2 shared-KV contract, then replay those exits for the fair physical FIFO/Oracle timing.  This trace collection is for exit-work control, not a quality experiment.

## Step 5 — live-exit mode

After forced-depth/trace mode is stable, add `--exit-source live`.

- Ouro: learned gate after each actual recurrent pass.
- Huginn: latent-diff convergence check after each actual recurrent pass.

Keep live-exit timing separate from trace-driven fair FIFO/Oracle timing.

For Huginn reproducibility, control RNG explicitly when the model initializes recurrent state stochastically.  A same-setting repeat must be reproducible before comparing runtimes.

## Step 6 — runners and GPU safety

Create `run_phase02_gpu6_ouro.sh` and `run_phase02_gpu7_huginn.sh`.

Each runner must:
- set `CUDA_VISIBLE_DEVICES` to exactly its physical assignment;
- fail closed if multiple devices are exposed;
- log `nvidia-smi --query-gpu=index,uuid,name,memory.total`;
- record `git rev-parse HEAD` and model revision;
- create a new timestamped result root;
- never write into Phase-1 result roots.

If physical-vs-visible numbering becomes ambiguous after `CUDA_VISIBLE_DEVICES`, capture host GPU identity before launching the isolated child process or otherwise record enough UUID information to prove the physical binding.

## Step 7 — benchmark instrumentation

For every physical recurrent interval save machine-readable rows containing:

```text
request cohort / token index / microbatch id
strategy
exit source
initial B
recurrence index
active B
active logical request ids (debug runs)
recurrent GPU ms
gate ms
compaction ms
KV update/copy ms
prelude ms
coda ms
peak allocated HBM
```

Use CUDA events for GPU timing where appropriate and synchronize benchmark boundaries.

The final summary must distinguish:

```text
NATIVE_FULL_REFERENCE
PHYSICAL_SURVIVOR_FIFO
PHYSICAL_SURVIVOR_ORACLE
PHYSICAL_SURVIVOR_LIVE
```

Never label trace replay as one of these physical categories.

## Correctness gates (not quality gates)

Before accepting timing:

- same forced-depth vector produces the expected survivor sequence;
- every logical request has exactly one final recurrent state;
- no request executes after its exit depth;
- output scatter order equals original logical request order;
- KV row identity is stable across compaction;
- Ouro skipped slots are filled only from that request's last actual KV;
- Huginn shared recurrent backend does not grow a recurrence-depth dimension;
- repeated deterministic smoke runs reproduce active-set traces;
- no NaN/Inf in final logits for smoke inputs.

Do **not** run GSM8K/perplexity/native-quality comparison in this phase unless the user explicitly changes the scope.

## Result gate

Before launching a broad sweep, produce a compact M1 report for each GPU showing actual active tensor shapes and measured recurrence time.

A useful report should make it visually obvious that physical compute changed, e.g.:

```text
GPU6 Ouro, B16, FIFO
r1 B16  x.xx ms
r2 B13  x.xx ms
r3 B7   x.xx ms
r4 B3   x.xx ms

GPU6 Ouro, B16, Oracle
r1 B16  x.xx ms
r2 B16  x.xx ms
r3 B15  x.xx ms
r4 B11  x.xx ms
```

Only after this is real should M2 multi-token runs proceed.

## Commit sequence

Suggested commits:

1. `phase02: implement common survivor runtime`
2. `phase02: implement Ouro last-exited KV runtime`
3. `phase02: implement Huginn shared-KV runtime`
4. `phase02: add GPU6 GPU7 physical timing runners`
5. `phase02: publish physical early-exit smoke results`

Do not squash away implementation failures that materially explain design changes; preserve them in the result review.
