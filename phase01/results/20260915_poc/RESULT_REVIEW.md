# R0 result review — first H100 exit-trace + no-refill simulator PoC

Bound result commit: `4ec78c1f5c0df7925f9888c7db3d75769e494ee7`

Source used by the published GPU run: `830eb5a0259cff9305d55a0cddd23bebb0825faa` (per the committed result receipt).

## What actually completed

- Single H100 80GB GPU1 run.
- Ouro-1.4B Alpaca exit-gate traces: 256 requests, max 24 generated tokens.
- Huginn-3.5B Alpaca adaptive-depth traces: 128 requests, max 24 generated tokens.
- Offline trace simulation for FIFO/random/oracle/previous/recent/sequence-mean grouping over multiple simulated micro-batch sizes.

Execution kind: `exit-trace-collection + trace-sim`.

This is not a real no-refill early-exit runtime result and does not measure physical throughput or BCT.

## Supported observations

Ouro-1.4B showed a large logical batching opportunity in the original simulator: oracle-minus-FIFO slot-utilization headroom was roughly +22.1pp (B=4), +26.0pp (B=8), +27.5pp (B=16), +26.5pp (B=32), and +18.3pp (B=64).

Huginn at convergence threshold 0.12 did not show the same opportunity: FIFO was already about 93-95% utilized and oracle headroom was small.

The simple causal predictors were weak relative to the Ouro oracle bound. The practical prediction/scheduling problem therefore remains open.

## Important comparison defect

After R0 publication, source review found that the simulator sorted by strategy and only then truncated the incomplete remainder. Therefore FIFO/oracle/etc. could use slightly different token subsets and total token-loop work. The effect sizes above remain exploratory until the fair-subset R1 reanalysis is published.

Status by axis:

- Measurement: `VALID_IN_SCOPE` for recorded exit-depth traces; late-token coverage shrinks as requests terminate.
- Implementation: `CHECKED_IN_SCOPE` for trace collection; Ouro gate is post-hoc and not physical early exit.
- Simulator: `SIMULATOR_DEFECT` for strategy-specific remainder/subset selection in R0 comparison.
- Comparison: `COMPARISON_INCOMPLETE` until fair-subset reanalysis.
- Hypothesis: `EXPLORATORY_SIGNAL` for Ouro; `NEGATIVE_IN_SCOPE` for Huginn threshold 0.12 scheduling headroom.

## Next bounded check

R1 only: fair-subset reanalysis, larger Ouro pool, and threshold sweeps. Do not promote the R0 percentage-point headroom into a runtime-speed claim. Do not start predictor training or custom runtime work until R1 is reviewed.
