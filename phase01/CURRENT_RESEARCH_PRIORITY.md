# Current priority — complete the three-model headroom + runtime study in one bounded block

Updated 2026-09-16.

The owner asked to stop splitting characterization and runtime work into many approval stages. The current block therefore combines fair headroom validation, model/threshold generalization, real H100 timing calibration, hardware-calibrated BCT/throughput replay, and integrated no-refill runtime engineering when parity can be established.

## Fixed experiment matrix

### Models
- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`
- `tomg-group-umd/huginn-0125` (Huginn-3.5B)

### Paper-facing batch sizes
`4, 8, 16, 32, 64`

No B=128/256/512 results are required for the current study. Internal latency calibration may additionally use B=1,2 so shrunk active batches can be priced rather than clamped blindly.

### Thresholds
- Ouro-1.4B and Ouro-2.6B: q = `0.2, 0.4, 0.5, 0.7`
- Huginn-3.5B latent-diff: `0.5, 0.35, 0.26, 0.12`, max depth 16

### Workload
Start with Alpaca under the existing tokenizer/prompt cap contract. Keep the same prompt cohort across thresholds within each model whenever the collector permits it. Do not silently mix datasets or model revisions inside one comparison table.

## Required outputs

For every model x threshold x B:

1. **Depth characterization**: histogram, mean/std depth, request/token denominators.
2. **Fair trace headroom**: FIFO/random/history/oracle slot utilization, idle-slot ratio, oracle-minus-FIFO headroom. Strategies must share the exact same selected items and total token-loop work.
3. **Real H100 latency calibration**: fixed-depth first-token and incremental decode latency at representative context lengths; report failed/OOM cells rather than dropping them.
4. **Hardware-calibrated scheduling outcome**: throughput and offline BCT replay for FIFO, oracle and already-implemented causal grouping. Label these results as hardware-calibrated trace replay, not integrated engine timing.
5. **Integrated runtime when validated**: physically compact survivors after exits, preserve the selected KV semantics, and report actual tokens/s, BCT, recurrent/boundary/scheduler breakdown. This evidence is publishable as physical speed only after deterministic parity checks pass.

## Runtime validation gates

Before an integrated timing number is trusted:

- fixed-depth mode matches the native model on a deterministic small cohort;
- mixed exit-depth execution preserves the intended exit-selected logits/tokens under the chosen KV layout;
- logged active batch sizes exactly match the runtime's compacted tensors;
- no exited request continues recurrent arithmetic;
- prefill/decode token counts, EOS/caps and missing outputs are explicit;
- timing uses CUDA synchronization/events consistently and excludes one-time model load unless separately reported.

If integrated runtime implementation cannot pass these gates in the current environment, publish the fair headroom + physical calibration + hardware replay honestly and mark integrated runtime `IMPLEMENTATION_INCOMPLETE`; do not manufacture an E2E claim.

## One-command run

`run_h100_full_sweep.sh` is the combined source-bound runner for the three-model trace sweep, B=4..64 fair analysis, H100 latency calibration, and hardware replay. It records source/model/environment receipts into a unique result root.

## Scientific decision

The scheduling direction remains promising only if a realistic threshold has both:

- material fair oracle headroom, and
- a material H100 wall-clock opportunity after pricing the actual active-batch latency curve.

A positive result on Ouro but not Huginn is acceptable as architecture-dependent evidence, but the paper claim must say so. A predictor that recovers little of the oracle is a method gap, not evidence that the oracle opportunity is false.
