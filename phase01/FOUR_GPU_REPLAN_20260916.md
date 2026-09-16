# Four-GPU replan — model-specific LoopLM validation

Updated 2026-09-16.

## Why the previous matrix needs to be reset

The latest committed full sweep is still a **plan/source checkpoint**, not a completed result publication. The only bound experimental evidence in the repository remains the first PoC result (`4ec78c1`), which showed exploratory Ouro-1.4B oracle headroom, weak history-only predictors, and little Huginn headroom at convergence threshold 0.12.

The previous one-GPU sweep also used one generic runtime contract for models with different recurrent/KV semantics. In particular, the current Huginn trace/E2E/calibration paths default to `cache_lookup_strategy=full`; this can make depth-indexed KV the capacity bottleneck and produced OOMs in the reported B=64/context=512/deep-step region. That capacity result should be preserved, but it is not the canonical adaptive-serving configuration for the next study.

The new study therefore freezes **model-specific depth, exit, and KV contracts first**, then evaluates the same no-refill batching question on four H100s.

## Research invariant

The research variable is micro-batch formation at token boundaries. No refill occurs inside one recurrent interval.

For each model family, keep these three axes fixed while comparing FIFO / history grouping / oracle grouping:

1. maximum recurrent depth;
2. exit rule and threshold;
3. KV policy.

Do not change KV policy or maximum depth between batching strategies within a comparison.

## Canonical model settings

### Ouro-1.4B

- Architecture: 24-layer fully looped core.
- Maximum recurrence: `Rmax=4`.
- Exit rule: learned Ouro gate.
- Threshold sweep: `q={0.2,0.4,0.5,0.7}`.
- Main KV policy: **native/reference Ouro cache semantics** from the pinned checkpoint/runtime. Do not force Huginn-style shared/compressed KV into the main Ouro comparison.
- Paper-facing B: `4,8,16,32,64`.
- Important evidence boundary: the public HF path evaluates all recurrent steps before selecting the gated output, so native threshold-dependent E2E is a real whole-model baseline but not physical early-exit speedup. The integrated no-refill runtime must physically skip recurrent work for exited tokens before an `integrated_*` speed claim is allowed.

Rationale: Ouro is sensitive to aggressive KV sharing in existing evidence, while `Rmax=4` keeps native recurrence-wise KV tractable enough to characterize separately from the batching method.

### Ouro-2.6B

- Same policy as Ouro-1.4B.
- `Rmax=4`.
- Learned gate, `q={0.2,0.4,0.5,0.7}`.
- Native/reference Ouro KV semantics.
- B=`4,8,16,32,64`.

This is a same-family scale check rather than a different scheduling design.

### Huginn-3.5B

- Architecture: 2-layer prelude, 4-layer recurrent core, 2-layer coda.
- Main maximum recurrence: `Rmax=16`.
- Main exit rule: `latent-diff`, to stay comparable with the existing CDB-style threshold study.
- Threshold sweep: `{0.5,0.35,0.26,0.12}`.
- B=`4,8,16,32,64`.

Huginn gets an explicit KV gate before the main matrix:

1. `full` — **capacity/reference only**, not the default serving configuration.
2. `latest-m4` — official adaptive-compute lookup semantics.
3. `latest-m4-compress-s4` — memory-efficient candidate aligned with the current recurrent-pretraining code path.

The main Huginn throughput/headroom matrix uses one KV policy selected by the following fail-closed rule:

- Prefer `latest-m4` if it passes deterministic parity/correctness checks and fits the target B/context cells.
- If `latest-m4` does not fit the target long-context cells, use `latest-m4-compress-s4` only after B=1 output checks and a bounded GSM8K quality sanity show no material collapse relative to the chosen reference.
- Keep `full` OOMs as a memory-capacity characterization; do not silently retry them as if they were missing data.

Do not reduce Huginn to `Rmax<16` just to avoid OOM. A lower `Rmax` changes the model-compute budget and would confound the scheduling study.

## Workloads

### Throughput/headroom workloads

Use both datasets for every model family:

| Dataset | Role | Trace requests | Native/integrated E2E requests | Prompt cap | Fixed decode tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Alpaca | short / low-KV | 256 | 256 | 512 | 64 |
| ShareGPT | long / high-KV | 256 | 256 | 2048 | 128 |

Why fixed decode lengths: keep the amount of generation work identical across scheduling policies and provide enough token positions to measure temporal depth behavior. EOS/caps must still be logged. ShareGPT B=64 OOM under a model's canonical KV policy is a valid capacity result, not a reason to change the policy mid-comparison.

### KV/quality sanity workload

Use a bounded GSM8K cohort (`N=128`, B=1) only to validate Huginn KV-policy changes. This is not the throughput workload and should not be mixed into BCT tables.

At minimum compare the selected Huginn memory-efficient policy against `latest-m4`/reference at fixed depth 16 and one representative adaptive threshold (default `0.26`). Report accuracy plus exact generated-output agreement where meaningful.

## H100 calibration

Calibrate model-specific latency surfaces at active batch sizes `1,2,4,8,16,32,64` and contexts `128,512,2048`.

- Ouro: depths `1,2,3,4`, native/reference cache contract.
- Huginn main: depths `4,8,12,16`, selected canonical adaptive KV policy.
- Huginn `full`: only a bounded memory/capacity characterization; no requirement to complete every large-B cell.

For every cell record peak allocated/reserved HBM as well as time so OOM boundaries can be explained rather than only listed.

## Four-GPU assignment

Logical lanes are used; do not assume physical GPU numbers until launch-time ownership is checked.

- **Lane A / GPU0:** Ouro-1.4B — Alpaca + ShareGPT full threshold/B/headroom/native-E2E matrix, then integrated no-refill validation.
- **Lane B / GPU1:** Ouro-2.6B — same matrix and integrated no-refill validation.
- **Lane C / GPU2:** Huginn-3.5B canonical adaptive path — threshold/B/workload matrix with the selected KV policy.
- **Lane D / GPU3:** Huginn KV/capacity lane — `full` vs `latest-m4` vs `latest-m4-compress-s4`, HBM/OOM surface, parity and GSM8K sanity; after the KV decision it may take confirmation/repeat cells from Lane C.

Model loading and result roots are isolated per lane. Do not share one mutable source/output tree across running jobs.

## Measurements

For every main model x workload x threshold x B cell:

1. per-token exit-depth distribution;
2. fair FIFO/random/history/oracle slot utilization and idle-slot ratio;
3. actual native whole-model BCT, generated tokens/s, requests/s, p50/p95 batch time and peak HBM;
4. hardware-calibrated no-refill BCT/throughput replay;
5. OOM/parity/error status.

For integrated no-refill runtime measurements, additionally report:

- actual survivor batch size at every recurrence;
- recurrent-core wall-clock;
- prelude/coda time;
- scheduler/compaction overhead;
- total E2E BCT and generated tokens/s;
- FIFO vs oracle vs current practical grouping.

### Integrated runtime scope

Do not wait for every threshold cell before implementing the physical no-refill path. After the model-specific contracts above are validated, use the full trace sweep to choose two representative thresholds per model (one high-headroom/aggressive and one moderate). Measure the integrated runtime at those thresholds across B=4..64 and both workloads.

Ouro requires a model adapter that exits the recurrent loop physically; its native post-hoc gate is not sufficient. Huginn may reuse the official adaptive-compute path, but survivor compaction, KV semantics and B-vs-B1 parity must be verified before timing is called integrated.

## Result publication

Publish one compact result handoff after the four lanes finish or reach a documented capacity/correctness stop. Preserve:

- source SHA and exact model revisions;
- exact KV policy per result row;
- dataset/prompt/decode contract;
- HBM peak/reserved memory and OOM cells;
- measurement / implementation / replay / comparison / hypothesis statuses;
- `RESULT_REVIEW.md` and `review_manifest.json`.

Do not merge `native_e2e`, `hardware_replay`, and `integrated` evidence into one speedup claim.