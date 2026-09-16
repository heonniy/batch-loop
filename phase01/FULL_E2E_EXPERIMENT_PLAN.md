# Phase01 full experiment — headroom + native E2E + H100 cost

Updated 2026-09-16.

## Research question

Can offline micro-batch formation reduce the recurrent-depth valley of **pure no-refill** looped-LM inference, and is the opportunity large enough to matter in real H100 end-to-end execution?

Refill is intentionally out of scope. Scheduling occurs only at token boundaries; a token that exits a recurrent interval is not replaced inside that interval.

## Fixed matrix

### Models
- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`
- `tomg-group-umd/huginn-0125` (Huginn-3.5B)

### Batch sizes
Paper-facing B = `4, 8, 16, 32, 64` only.

### Thresholds
- Ouro-1.4B / Ouro-2.6B: `0.2, 0.4, 0.5, 0.7`
- Huginn-3.5B latent-diff: `0.5, 0.35, 0.26, 0.12`, max recurrent depth 16

### Workloads: Alpaca + ShareGPT

Both workloads are required so the scheduling claim is not tied only to short instructions.

- **Alpaca**: short/tightly concentrated workload. Default bounded run: 256 trace requests, 128 native-E2E requests, prompt cap 512, generation cap 24.
- **ShareGPT**: longer/heavier-tailed chatbot workload. Default bounded run: 128 trace requests, 64 native-E2E requests, prompt cap 2048, generation cap 64.

The ShareGPT collector reuses the existing `ewof/sharegpt-instruct-unfiltered-deduped` loader and first-user-message extraction in `poc.py`. The caps are explicit PoC resource bounds, not claims about the full original ShareGPT output-length distribution. They can be overridden with `SHAREGPT_*` environment variables, but every result publication must record the actual values.

Alpaca and ShareGPT artifacts stay in separate dataset subdirectories. Cross-workload CSVs are derived only after each dataset-specific run is complete.

## What is measured for every model x threshold x B x workload

1. **Depth characterization**: histogram, mean/std depth, request/token denominators.
2. **Fair trace headroom**: FIFO/random/history/oracle slot utilization, idle-slot ratio, oracle-minus-FIFO headroom. Strategies must share the exact same selected items and total token-loop work.
3. **Actual native whole-model E2E**: H100 offline BCT, generated tokens/s, requests/s, model generation time, batch p50/p95, tokenization/H2D, and peak HBM.
4. **Real H100 latency surface**: fixed-depth timing at active B=1,2,4,8,16,32,64 and contexts 32/128/512/2048. OOM/error cells are evidence and must remain visible.
5. **Hardware-calibrated no-refill replay**: FIFO/oracle/causal grouping BCT and tokens/s using the observed exit traces and measured latency surface.

## Evidence boundaries

- `slot_utilization` is trace geometry, not physical GPU utilization.
- `native_e2e_*` is actual whole-model wall-clock execution.
- Ouro native code still computes all recurrent steps and exposes exit depth post-hoc, so threshold-dependent Ouro E2E is a real baseline but not physical early-exit speedup.
- Huginn adaptive E2E is trusted only where deterministic batched-vs-B1 parity passes.
- `hardware_replay_*` uses real H100 calibration plus real exit traces, but is not an integrated dynamic compaction engine.

## Why two workloads matter

Alpaca tests the short-context regime where weight reload and recurrent batch shape can dominate. ShareGPT tests whether the same valley/headroom survives longer context and heavier KV traffic. A method that only helps one regime must be reported as workload-dependent rather than generalized silently.

## One-command execution

```bash
bash run_h100_full_sweep.sh
```

Useful explicit overrides:

```bash
SHAREGPT_N_REQUESTS=256 SHAREGPT_E2E_N_REQUESTS=128 \
SHAREGPT_PROMPT_TOKENS=2048 SHAREGPT_NEW_TOKENS=96 \
bash run_h100_full_sweep.sh
```

The runner records source SHA, exact launch command, model revisions, environment, and the dataset-specific cap contract into the result root.

## Result publication

The executing model/coding agent must publish compact dataset-specific and cross-workload CSV/JSON/plots, a human-readable `RESULT_REVIEW.md`, and `review_manifest.json`. Keep positive and negative model/threshold/workload results. Do not promote replay estimates into integrated-runtime claims.
