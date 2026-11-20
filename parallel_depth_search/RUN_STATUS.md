# Experimental execution status — 2026-11-21

**NOT_RUN (new campaign).** This branch contains an extended research protocol, machine-readable adaptive DAG, offline tests, and opt-in CUDA diagnostic runner. There is NO new H200 measurement or server-side execution claim associated with these commits. Prior PoC GPU results remain at [batch-loop@829076bb](https://github.com/heonniy/batch-loop/blob/829076bb/parallel_depth_poc/RESULTS.md).

- Current run: none initialized on an H200 host by this assistant.
- Physical hardware access in the current session: unavailable.
- Next safely executable step: coding agent on allocated H200 server follows [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md), initializes run manifest, runs G0 native parity and cache gate, ingests verified receipts; repeats adaptive selection until user-authorized time budget or safety stop.
- Implemented diagnostic arms: Constant/Linear/Geometric/Secant/Prev-Shift × Jacobi/Damped/Anderson/Identity-Newton/Scaled-Newton × Full/Suffix/Prefix/Alternating/Residual-TopK; not all combinations have sound theoretical meaning and **none yet demonstrate serving gains**.
- Deferred implementation gates: genuine ParaTAA triangular formulation, low-rank/diagonal JVP-corrector, physical Parareal with rich coarse G, native full B>1 E2E cache-safe decoder, online calibration, final held-out tests.

**Never replace `NOT_RUN` with `completed` based solely on successful CPU CI or generated job manifests.**
