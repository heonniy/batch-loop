# Current priority — complete the three-model, two-workload no-refill study

Updated 2026-09-16.

The current bounded block combines fair headroom validation, model/threshold generalization, real H100 whole-model E2E measurement, and hardware-calibrated no-refill BCT/throughput replay. Refill is out of scope.

## Fixed experiment matrix

- Models: Ouro-1.4B, Ouro-2.6B, Huginn-3.5B.
- Paper-facing batch sizes: `4, 8, 16, 32, 64`.
- Ouro thresholds: `0.2, 0.4, 0.5, 0.7`.
- Huginn latent-diff thresholds: `0.5, 0.35, 0.26, 0.12`, max depth 16.
- Workloads: **Alpaca + ShareGPT**.

Default bounded workload contracts:

| Workload | Trace requests | Native-E2E requests | Prompt cap | Generation cap |
| --- | ---: | ---: | ---: | ---: |
| Alpaca | 256 | 128 | 512 | 24 |
| ShareGPT | 128 | 64 | 2048 | 64 |

ShareGPT is included specifically to test the longer-context/heavier-KV regime rather than validating only short Alpaca instructions. Dataset-specific results must stay separate before any combined table is produced.

## Required outputs

For every model x threshold x B x workload:

1. depth distribution and denominators;
2. fair FIFO/oracle/history no-refill headroom;
3. actual native whole-model H100 BCT and throughput;
4. physical H100 latency calibration, including longer contexts for ShareGPT;
5. hardware-calibrated FIFO/oracle/history no-refill BCT and throughput replay.

The combined runner is `run_h100_full_sweep.sh`. It runs Alpaca and ShareGPT sequentially on one assigned H100 and reuses the model-level calibration surface across workloads.

## Claim boundary

- Trace headroom is not throughput.
- Native Ouro E2E is real whole-model timing but not physical threshold-dependent early exit because the public implementation evaluates all recurrent steps before the post-hoc gate.
- Huginn adaptive timing is trusted only where parity passes.
- Hardware replay is not an integrated compaction runtime.

The result publication must preserve these labels and report OOM/error cells rather than silently dropping them. Positive Ouro and negative Huginn or workload-specific results are all valid evidence.
