# Current priority — four-GPU model-specific reset

Updated 2026-09-16.

Read [`FOUR_GPU_REPLAN_20260916.md`](FOUR_GPU_REPLAN_20260916.md) before launching new GPU work.

The previous one-H100 combined sweep is superseded for **new runs** because Huginn was being evaluated with the generic `full` recurrence-indexed KV path, which creates a large HBM confound and can prevent the intended `Rmax=16` cells from running. Existing OOMs/results remain evidence in their measured scope and must not be deleted.

## Fixed main matrix

- Models: Ouro-1.4B, Ouro-2.6B, Huginn-3.5B.
- B: `4,8,16,32,64`.
- Workloads: Alpaca + ShareGPT.
- Ouro: `Rmax=4`, learned gate, thresholds `0.2,0.4,0.5,0.7`, native/reference KV semantics.
- Huginn: `Rmax=16`, `latent-diff`, thresholds `0.5,0.35,0.26,0.12`; choose one canonical adaptive KV policy after the bounded `full` / `latest-m4` / `latest-m4-compress-s4` gate.
- Main trace/E2E cohort: 256 requests per workload; fixed decode 64 tokens for Alpaca and 128 for ShareGPT.

## Four H100 lanes

- Lane A: Ouro-1.4B.
- Lane B: Ouro-2.6B.
- Lane C: Huginn main adaptive path.
- Lane D: Huginn KV/HBM/parity/quality characterization, then confirmation cells.

Do not reduce Huginn depth to make a memory cell pass. Fix/select the KV contract instead, and retain `full` OOMs as capacity evidence.

## Required evidence

1. fair headroom/depth distributions;
2. native whole-model E2E/BCT/throughput;
3. model-specific H100 latency + HBM surfaces;
4. hardware-calibrated no-refill replay;
5. integrated physical no-refill timing at two representative thresholds/model after parity gates.

Refill remains out of scope. Every result row must name the model revision, KV policy, Rmax, threshold, B, workload and evidence kind.