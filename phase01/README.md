# Phase 01 — offline no-refill depth-compatible batching

## Research question

Depth-adaptive looped LMs create a recurrent-depth valley under pure no-refill execution: tokens exit at different loop depths, so a micro-batch shrinks within one token-generation interval. The Phase 01 question is:

> Can an offline scheduler form recurrence-compatible micro-batches at token boundaries and recover much of the lost utilization **without intra-loop refill**?

The intended execution semantics remain strict no-refill inside the recurrent interval. Scheduling/re-grouping happens only at a token boundary.

## Completed R0 PoC — what actually ran

Result commit: `4ec78c1f5c0df7925f9888c7db3d75769e494ee7`.

Execution kind:

1. Real single-H100 model generation / gate tracing for Ouro-1.4B and Huginn-3.5B.
2. Saved exit-depth traces.
3. Offline trace simulation of FIFO/random/oracle/history-based micro-batch formation.

This was **not** a real early-exit no-refill runtime benchmark. Ouro's public implementation computes all recurrent steps and exposes the gate post hoc; the simulator estimates batch-shape opportunity from the resulting per-token exit depths.

Main measured signal:

- Ouro-1.4B: oracle grouping showed about +22 to +28 percentage points slot-utilization headroom at B=4..32, with meaningful room still at B=64.
- Huginn, threshold 0.12: FIFO was already roughly 93-95% utilized, so oracle room was small.
- Previous-depth / recent-mean / request-mean predictors recovered only a small fraction of the Ouro oracle room.

Known comparison defect discovered after R0: the original simulator ordered first and then dropped an incomplete remainder, so different strategies could evaluate slightly different token subsets. R1 must fix this before any stronger claim.

See `results/20260915_poc/RESULT_REVIEW.md` for the bound review.

## Current bounded block: R1 characterization

Source checkpoints already on main:

- `a1e8ced`: fair-subset analyzer.
- `0bb63b7`: H100 Phase 1-3 sweep runner.

These commits are **not result commits**. The next run must publish a separate result-bound commit.

R1 contains only:

- R1-A: fair-subset reanalysis.
- R1-B: Ouro pool scaling (target >=1024 requests) and B=16..256/512 sweep.
- R1-C: exit/convergence threshold sweep for Ouro and Huginn.

Stop after R1 for owner/research review.

## Gate to R2 real runtime

Proceed to a custom no-refill runtime only if fair R1 still shows a substantial, stable headroom region. R2 must report actual H100 measurements:

- recurrent-core wall-clock time,
- generated tokens/s,
- offline batch-completion time (BCT),
- end-to-end latency/time breakdown where practical,
- slot utilization as a diagnostic, not a speed claim.

The scientific target is not to prove refill is bad. The question is whether a simpler token-boundary grouping policy can recover a useful fraction of refill's benefit for offline workloads.
