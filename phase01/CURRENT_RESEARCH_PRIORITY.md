# Current priority — validate the oracle headroom fairly before building the runtime

Updated 2026-09-16.

The first H100 PoC produced a positive Ouro oracle signal but also exposed a comparison defect and weak practical predictors. The next work is therefore characterization, not method tuning.

## Next bounded block: R1

1. **Fair subset.** For each token index and micro-batch size, choose the eligible complete-batch subset once, then apply FIFO/random/oracle/history reorderings only within that identical subset. Assert identical `total_token_loop_work` and item identity across strategies.
2. **Scale.** Ouro-1.4B: increase the offline request pool to at least 1024 when runtime permits. Measure B=16,32,64,128,256 and optionally 512. Large-B points require enough independent complete micro-batches to make reordering meaningful.
3. **Thresholds.** Sweep Ouro q={0.2,0.4,0.5,0.7} and Huginn convergence threshold={0.5,0.35,0.26,0.12}. Record the depth histogram, FIFO valley/utilization, oracle utilization, and oracle-minus-FIFO headroom for every setting.

Do not improve/train a depth predictor in R1. Do not implement refill. Do not call a trace-simulator improvement a throughput or BCT improvement.

## Decision after R1

- **Promote to R2 runtime:** fair oracle headroom remains material over a useful B range and at least one realistic accuracy/threshold operating point.
- **Narrow scope:** signal is model/threshold-specific but repeatable; focus the paper/system claim on the supported architecture/regime and add a second architecture only as characterization.
- **Stop/reframe:** fair oracle headroom collapses or exists only at impractical batch/workload conditions.

R2, if approved, is the first phase allowed to claim physical speed evidence: implement actual no-refill early-exit recurrent execution and measure throughput, recurrent wall-clock, and offline BCT on the single H100.
