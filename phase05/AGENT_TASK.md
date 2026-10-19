# Phase 05 — No-refill depth-grouping regime and architecture study

Status: **EXPERIMENT SPECIFICATION COMMITTED; NOT EXECUTED**. This task is for a coding agent with access to the H100 host. Do not fabricate results or claim a benchmark was run by adding this file.

## Goal

Determine when depth-aware grouping improves **physical no-refill decode time** on Ouro-1.4B, Ouro-2.6B, and Huginn-3.5B. Run three independent but linked experiments:

1. **E1 Retrospective regime analysis:** reanalyze existing phase01–04 measured calibration, recurrence events, depth traces, and physical Oracle/FIFO comparisons.
2. **E2 H100 architecture microbenchmark:** measure actual survivor-compacted recurrent Core and Prelude/Coda as functions of active batch, context, model and KV policy.
3. **E3 Latency-aware Oracle grouping:** compare FIFO, depth-sorted/RIC Oracle, and measured-cost Oracle at equal *frozen* token work, first by replay then in a physical no-refill runtime.

No refill; no predictor training in this phase. Use existing `phase02_runtime` implementation rather than rewriting the engine. Read root `AGENTS.md` and phase01–04 reviews before running anything. The phase02 Oracle speedups are **trace-forced** (not native live-exit or quality-equivalent generation); preserve that attribution.

## Required artifact layout

```
phase05/
  README.md
  E1_REGIME_REANALYSIS.md
  E2_STAGE_MICROBENCH.md
  E3_LATENCY_AWARE_ORACLE.md
  scripts/...
  tests/...
  results/<date>/
    RUN_MANIFEST.json
    RESULTS.md
    REVIEW.md
    e1_regime_cells.csv
    e2_stage_latency.csv
    e3_oracle_comparison.csv
    ... raw event receipts ...
```

Code should have an inexpensive CPU-only `--smoke`/unit-test path and explicit GPU commands documented in `phase05/README.md`. If using a different layout, document deviations. Save all raw data; do not overwrite existing phase01–04 results.

## Common experimental contract

- Hardware: H100 80 GB; use **one GPU per independent benchmark**, isolate device via `CUDA_VISIBLE_DEVICES`, verify currently available devices/ownership with host operator. Prior historical physical GPU assignments **do not imply current permission**. Never touch other users' GPUs. No multi-GPU communication in this phase.
- Models pinned to the previously used revisions, unless unavailable (record change explicitly): Ouro-1.4B `37bc7e8ff85a90d813dc096383405f521b6a67b0`; Ouro-2.6B `f8e38372d72a8aa7f4ea9f14b98739a311589088`; Huginn-3.5B `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
- Precision BF16 primary; track SDPA/FlashAttention backend, CUDA, PyTorch, GPU UUID, model revision, data snapshot/prompt hash, source SHA and seed. Do not silently switch to FP32 to make a correctness gate pass.
- KV: Ouro reference last-exited depth-indexed semantics; Huginn shared physical-layer KV. Exact cache semantics must be identical across compared strategies. `Rmax=4` for Ouro and `Rmax=16` for Huginn.
- Workloads: use frozen Alpaca/ShareGPT prompts and existing trace manifests when possible. Trace replay uses identical request IDs, token positions, context lengths, forced depths, and total per-token work across strategies. Compare **within each cohort**, not across differing prompt truncations.
- Time both GPU CUDA events and synchronized decode wall, after >=2 warmups and >=5 repetitions for final scored cells. Report median and spread; randomize policy order. Track outliers without silently dropping them. Separate model load, prefill, recurrent, Prelude/Coda, KV, compaction, host scheduler and prediction overhead.
- Mark evidence as `TRACE_ONLY`, `LATENCY_MODEL`, `PHYSICAL_FORCED_DEPTH` or `PHYSICAL_LIVE_EXIT`; never conflate these. Record OOM/failed parity as results. Missing cells are **missing**, never zero.

## E1 — Retrospective regime analysis (CPU only)

Inputs: phase01 Ouro `ouro_calibration.csv`, `ouro_calibration_fit.csv`, `ouro_headroom_fair.csv`, `ouro_hardware_replay_b4_b8.csv`; phase02 `physical_matrix.csv`, `trace_sim_64tok/oracle_ric_sweep.csv`, recurrence-event receipts; phase03 predictor tournament results; phase04 BF16 divergence and rank-stability receipts.

1. Audit column schemas and provenance; map comparable rows by model, threshold, N, B, context, KV, trace hash and runtime kind. Exclude OOM-partial native E2E cells from throughput comparisons. Native Ouro HF computes full recurrence post hoc; it is **not** a physical early-exit baseline.
2. Compute per-cell `N/B`, depth mean/std/CV/p10/p50/p90, per-microbatch max-depth distribution, `RIC=sum_m max_i d_i`, token-loop work `sum_i d_i`, slot utilization, FIFO→Oracle RIC reduction, actual physical core/wall reduction and overhead ratio. Distinguish normalized `reduction=(T_fifo-T_policy)/T_fifo` from `speedup=T_fifo/T_policy`.
3. Plot or tabulate RIC reduction versus N/B, B, depth dispersion; and physical core/wall reduction versus RIC reduction. Stratify by model, threshold, prompt/context, and KV policy. Use matched cohorts only; annotate phase02 differences from phase01 prompts.
4. Summarize what is **measured**, what is a latency model, and which exact cells lack comparable evidence. Do not infer a hardware compute-bound regime solely from a flat time-versus-B curve.
5. Acceptance: all source rows/checksums reconciled, missing/OOM tracked, independent tests for RIC/slot arithmetic, no rows treating model replay as GPU-measured latency.

## E2 — Stage-specific H100 latency and architecture study

Primary question: where does an additional surviving token increase recurrent-core time, and when are weight, KV bandwidth, compute or launch overhead dominant?

1. Instrument existing *survivor-compacted* recurrent runtime to measure **one full recurrent-core invocation** with active `B in {1,2,4,8,16,32,64,128}` and context `L in {128,512,1024,2048}`, subject to actual HBM constraints. Try short and long contexts separately; OOMs retained. Use representative valid KV/state; no synthetic GEMM substituted for model Core.
2. Measure `Prelude(B,L)`, `Coda(B,L)` (including / separately excluding LM head), `Core(B,L)`, KV updates, compaction, launch/synchronization overhead, and peak allocated/reserved HBM. Ouro has zero Transformer Prelude/Coda blocks; record embedding/LM-head stages rather than reporting zero boundary wall time.
3. Use CUDA events with synchronization, >=2 warmups and >=5 repetitions for exploratory; >=10 measurements for shortlisted cells. Prefer Nsight Compute/Systems counters for DRAM bytes, achieved FLOPs and occupancy on a small, predeclared subset; document missing profiler permissions. Estimate arithmetic intensity and plot latency against B; label **weight/compute/KV/launch bottleneck only where supported**.
4. Run architecture ablation by comparing Core vs Prelude/Coda fractions under matched B/L per model. Clearly state cross-model confounds: different layer width, attention implementation, KV semantics, and depth distribution. Do not claim layer-count alone causes measured timing difference.
5. Correctness gate: for each tested shape assert equal model/KV input contract and stable state outputs within predeclared numerical tolerances. Phase04 shows BF16 B-dependent depth changes; stage timing can proceed at fixed state but live-exit quality claims are blocked until appropriate gate.
6. Save every raw trial with stage, B, L, dtype, GPU, model, policy, status, latency and memory; define operational knee `B*` from measured marginal latency (explicit objective, tolerance and uncertainty), not a guessed memory-to-compute threshold.

## E3 — Architecture-aware Oracle grouping

Compare **FIFO**, **depth-sorted/RIC Oracle**, and **latency-aware Oracle**. Optional random equal-work control. An Oracle uses *future true depth* and is NOT deployable; explicit objective and exact solver/heuristic limits are mandatory.

For `N in {32,64,128,256}`, `B in {4,8,16,32,64}` where HBM permits, use 32 and 64 fixed decode-token horizons, frozen matching trace and model-specific KV semantics.

For any microbatch `m`, depth list `d_i` and per-token context `L_i`, record:

```
RIC(m) = max_i d_i
cost(m) = sum_{r=1}^{max_i d_i} T_core(B_active(m,r), context_characterization(m,r))
          + stage/KV/compaction costs when the execution order actually changes them
```

Measure `T_core` from **E2 real H100** table, with specified interpolation only **within** observed B/L support; no silent extrapolation. If mixed context cannot be modeled reliably, use exact context buckets and validate replay residuals on FIFO first. The simple RIC objective is a baseline, not automatically hardware-optimal.

1. First validate `FIFO` predicted time vs physically measured FIFO with residuals and confidence/spread. Reject `LATENCY_MODEL` claims when replay calibration is poor or unavailable.
2. Build latency-aware grouping from the frozen per-token depths. For small N/B, compare exact/exhaustive or bounded combinatorial optimum when computationally feasible; otherwise state heuristic and certify non-regression against depth-sorted Oracle in the predicted objective. The same requests must appear exactly once per decode boundary, with no token dependency violation and no within-Core refill.
3. Score trace `RIC`, measured-calibrated predicted Core time and independent *physical survivor* Core CUDA-event and synchronized decode wall for top predeclared cells; control `N=B` (grouping cannot change RIC).
4. **Never use B=1 depth labels to claim exact live-exit parity.** Frozen forced-depth physical cells are valid scheduling mechanism tests, while actual live-exit comparison requires new BF16 same-policy parity/quality validation; phase04 Gate A previously failed.
5. Compare physical reductions across model and B/L regimes; quantify whether latency-aware cost grouping improves on simple depth sorting *at equal recorded work*. Report negative/zero effects. If assumptions fail, publish a bounded negative result rather than modifying work silently.

## Reporting and go/no-go

Publish a single summary table per model/regime with: N, B, L, threshold, mean/max depth, N/B, Core/Prelude/Coda share, measured marginal batch-scaling, FIFO RIC, depth-Oracle RIC, latency-Oracle RIC, calibrated prediction, physical Core and decode wall median/spread, and evidence class.

Answer three research questions explicitly:
- RQ1: Is physical gain explained by removal of **whole recurrent Core invocations** rather than merely improved logical slot utilization?
- RQ2: Which B/L ranges are empirically weight-memory-, compute-, KV- or launch-dominated and how does this change by architecture?
- RQ3: Does measured-cost Oracle materially outperform simple depth-sorted Oracle in matched physical no-refill runs?

Final review must include blockers (OOM, numerical parity, profiler absence, mixed-length model mismatch), exact reproduction commands and commits, and follow the root AGENTS.md review workflow. Do **not** perform or report any GPU runs until device ownership is verified.
