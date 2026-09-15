# Phase01 full experiment — headroom + native E2E + H100 cost

Updated 2026-09-16.

## Research question

Can offline micro-batch formation reduce the recurrent-depth valley of **pure no-refill** looped-LM inference, and is the opportunity large enough to matter in real H100 end-to-end execution?

Refill is intentionally out of scope.  Scheduling occurs only at token boundaries; a token that exits a recurrent interval is not replaced inside that interval.

## Fixed matrix

### Models
- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`
- `tomg-group-umd/huginn-0125` (Huginn-3.5B)

Use the pinned revisions recorded by `run_h100_full_sweep.sh` unless a compatibility failure is documented and a replacement revision is explicitly committed.

### Batch sizes
Paper-facing B = `4, 8, 16, 32, 64` only.

### Thresholds
- Ouro-1.4B / Ouro-2.6B: `0.2, 0.4, 0.5, 0.7`
- Huginn-3.5B latent-diff: `0.5, 0.35, 0.26, 0.12`, max recurrent depth 16

### Workload
Primary: Alpaca, fixed seed and prompt cohort within each model.  Default trace pool is 256 requests/model; default native-E2E pool is 128 requests/model so the full 3-model x threshold x B block fits one H100 session.  Increase only by explicit environment override, not by silently changing the plan.

## What is measured

### A. Real exit traces and fair scheduling headroom

Collect per-request/per-token exit depths from the official model code.  For each threshold and B, compare the exact same selected token/request set under:

- FIFO
- random
- previous-depth
- recent-depth
- sequence-mean
- oracle current-token depth

Report slot utilization, idle-slot ratio, depth variance, and oracle-minus-FIFO headroom.  Oracle is an upper bound, not an implementable policy.

### B. Actual whole-model E2E / throughput / BCT

Use the official/native generation implementation for each model, loaded once per model, and sweep every threshold and B.  Measure actual H100 wall-clock:

- whole offline batch-completion time (BCT)
- generated tokens/s
- requests/s
- model-generation wall time
- tokenization and H2D time
- p50/p95 micro-batch generation time
- peak allocated HBM

One-time model load is recorded separately and excluded from BCT.  Performance runs force the same output-token cap/length contract to avoid comparing different amounts of generation work.

For Huginn, run a small deterministic batched-vs-B1 output parity check at each `(threshold, B)`.  Batched adaptive timing is not promoted to trusted evidence if parity fails.

For Ouro, the current public HF implementation computes all recurrent steps and applies the exit gate post-hoc.  Its native E2E measurement is therefore a **real whole-model baseline**, but threshold changes must not be described as physical early-exit speedup.

### C. Real H100 recurrent-step latency surface

Measure fixed-depth generation at internal active batch sizes `1,2,4,8,16,32,64` and representative context lengths.  These measurements price the cost of a no-refill batch shrinking from B to smaller survivor counts.

### D. Hardware-calibrated no-refill scheduling replay

Replay FIFO/oracle/causal-grouping traces against the measured H100 latency surface and report estimated no-refill BCT and tokens/s.  Label this evidence exactly as hardware-calibrated trace replay; it is not an integrated compaction runtime.

## Why both B and D are needed

`native_e2e_all.csv` answers **how expensive the complete model is on the real H100** across model/threshold/B.

`runtime_replay_all.csv` answers **how much of that cost structure could change if the observed no-refill survivor pattern were physically compacted/grouped**.

They must not be merged into a single unsupported speedup claim.  A future integrated compaction engine is needed before claiming measured scheduler E2E speedup for Ouro.

## One-command execution

```bash
bash run_h100_full_sweep.sh
```

The runner records source commit, exact model revisions, environment, launch command and a unique output root.

Primary outputs:

```text
results/full_sweep_*/analysis/fair/oracle_vs_fifo.csv
results/full_sweep_*/analysis/depth_distribution.csv
results/full_sweep_*/e2e/native_e2e_all.csv
results/full_sweep_*/calibration/*_fit.csv
results/full_sweep_*/analysis/runtime_replay_all.csv
```

## Result publication

The executing agent must commit a compact result packet after completion:

1. `RESULT_REVIEW.md` — what actually ran, key tables, failures, limitations.
2. `review_manifest.json` — source SHA, run root, model revisions, execution kinds, measurement/comparison/hypothesis status.
3. compact CSVs/plots required to reproduce the review.
4. hashes/paths for any large raw traces that remain server-side.

Do not publish only favorable cells.  OOM/unsupported/parity-failed cells remain in the result table.

## Decision criterion

The idea has system-level room when a realistic threshold simultaneously shows:

- substantial fair FIFO -> oracle headroom over B=4..64;
- a corresponding material reduction in hardware-calibrated no-refill BCT / increase in tokens/s;
- native whole-model BCT demonstrating that recurrent decode cost is a meaningful part of total execution.

A positive Ouro result with weak Huginn room is acceptable, but the conclusion must be architecture-specific.  Weak simple predictors are a method problem, not evidence against oracle headroom.
