# Current priority — run the complete three-model no-refill study in one block

Updated 2026-09-16.

Read [`FULL_E2E_EXPERIMENT_PLAN.md`](FULL_E2E_EXPERIMENT_PLAN.md) for the exact matrix and evidence labels.  The owner requested one combined experiment rather than separate characterization/runtime approval stages.

## Fixed study

### Models
- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`
- `tomg-group-umd/huginn-0125` (Huginn-3.5B)

### Batch sizes
`4, 8, 16, 32, 64`

### Thresholds
- Ouro-1.4B / Ouro-2.6B: `0.2, 0.4, 0.5, 0.7`
- Huginn-3.5B latent-diff: `0.5, 0.35, 0.26, 0.12`

### Current workload
Alpaca, same seeded prompt cohort within a model.  Default trace pool: 256 requests/model.  Default native-E2E pool: 128 requests/model.

## Required measurements in the same run

1. **Fair headroom** — exact same items across FIFO/random/history/oracle; report slot utilization, idle slots and oracle headroom.
2. **Depth distribution** — every model x threshold.
3. **Actual native whole-model H100 E2E** — BCT, generated tokens/s, requests/s, p50/p95 micro-batch time, tokenization/H2D, HBM peak for every model x threshold x B.
4. **Real H100 recurrent timing surface** — active B=1,2,4,8,16,32,64 internally so no-refill shrinkage can be priced.
5. **Hardware-calibrated no-refill replay** — FIFO/oracle/current causal policies; report BCT/tokens/s opportunity with an explicit replay label.
6. **Parity / failure evidence** — Huginn batched adaptive execution requires deterministic B-vs-B1 output parity. OOM/unsupported cells are recorded, not dropped.

Refill is not implemented or evaluated in this block.

## Critical evidence distinction

- `native_e2e_all.csv` is **actual whole-model wall-clock** on the H100.
- Ouro's public HF code still computes every recurrent step and uses the gate post-hoc, so Ouro native threshold timing is a baseline, not physical early-exit speedup.
- `runtime_replay_all.csv` uses real H100 timing plus real exit traces to estimate no-refill grouping cost; it is not an integrated dynamic compaction engine.
- Do not convert slot-utilization gain or hardware replay into a claim of measured scheduler E2E speedup.

## One-command run

```bash
bash run_h100_full_sweep.sh
```

The runner performs trace collection, fair analysis, native E2E benchmarking, timing calibration and hardware replay for all three models.

## Result publication

After the run, the executing coding/model agent publishes one result-bound commit containing compact CSVs/plots, `RESULT_REVIEW.md`, and `review_manifest.json`.  It must state what finished, what failed, actual source/model revisions and the evidence kind of every throughput/BCT table.

Do not wait for another owner approval between headroom and native E2E measurements.  Stop only for a correctness/resource failure that would make the remaining measurements misleading or unsafe.
