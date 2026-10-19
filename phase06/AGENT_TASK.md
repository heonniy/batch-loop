# Phase 06 — Optimized CDB no-refill regime study (execution handoff)

**Status: plan committed, not yet implemented or run.** Agent: implement, run bounded H100 experiments, review, and push results. Do not claim success until receipts exist.

## Pinned starting points
- This repository: branch `research/phase06-optimized-cdb-20261019`, parent Phase 05 commit `2257aa4fa3460fb6d6a2a2866f8f771a6c47bd4c`. Do not overwrite prior experiments.
- Upstream CDB: `https://github.com/kschwethelm/continuous-depth-batching`, **pin commit `abdfae0adbdd12be8c6123ccb34eee98a38afb39`**, verify commit and lockfile locally. Respect MIT/third-party notices; prefer a reproducible clone/submodule in ignored working area or a fork with an explicit upstream SHA; do NOT vendor anonymously.
- Read root `AGENTS.md`, `phase05/results/20261019/{RESULTS,REVIEW}.md`, upstream README, scheduler/model_runner/config/benchmarks and CUDA tests before changing code.
- Upstream critical paths: `src/looped_cdb/continuous_depth_batching/{scheduler,model_runner,config,cache}.py`, `scripts/{benchmark_throughput,benchmark_decode_step_latency,export_schedule_trace}.py`.
- Upstream explicitly supports `refill=False`, `use_cuda_graph=True`, `use_async_batching=True`, paged FlashAttention, and synthetic recorded exit replay. Preserve exactly these choices when comparing grouping policies.

## Research questions
**RQ1, optimized-regime study:** With paged attention/CUDA Graph, what is the measured recurrent Core latency as B and context L vary? Distinguish launch/host, HBM-weight, KV-memory and compute limits *with counters where available*. Flat B curve alone is not sufficient evidence of memory-bound.

**RQ2, architecture:** How do Ouro-1.4B, Ouro-2.6B (0–Core–0) versus Huginn-3.5B (2–4–2) differ in Core/Prelude/Coda/KV/LM-head cost? Avoid causal claims based only on model identity because widths, depth distributions, kernels and KV policies differ.

**RQ3, grouping:** On this SAME optimized CDB no-refill runtime, compare FIFO cohort formation, depth-sorted Oracle cohort formation, and a random control. Measure whether the Phase 05 forced-depth grouping headroom persists. Optional latency-cost Oracle only after E2 curves are validated; Phase 05 found no material improvement and E2→E3 absolute calibration was rejected.

## Execution sequence and gates
### E0 — upstream reproduction and correctness gate (MUST pass before E1–E3)
1. Install upstream pinned environment `uv sync --frozen` (optionally `--extra fa2` when required). Record host/GPU UUID, driver, CUDA, PyTorch, FlashAttention version/backend, upstream and local SHAs, cache policy. Check available GPU ownership before every job; earlier GPU IDs do not authorize current usage. No work on other users' GPUs.
2. Run upstream no-refill CUDA tests, Ouro/Huginn model parity tests, and a small throughput smoke with `--backend cdb --no-refill`, CUDA Graph ON, async ON, paged KV. Compare CB and original CDB FIFO at identical workload/exit policy. Keep published reproduction separate from novel policy.
3. Verify for both models: deterministic frozen-work request IDs, output-token counts and prompt hashes; no incorrect cross-token dependencies; stage boundaries; no-refill cohort semantics (new token cannot enter during the current recurrent wave). Validate dynamic exit replay and KV-policy correctness. Phase 04 BF16 batch-sensitive live gate remains a prior blocker. Equal frozen work is not live quality parity.
4. If E0 fails, save log and stop downstream performance claims; repair/test in a separate, reviewable commit.

### E1 — optimized Core-regime microbenchmark
- On real upstream model runner, use 1 GPU per run, B={1,2,4,8,16,32,64,128} (OOM retained), L={128,512,1024,2048}; also L=64/96 to match prior Phase 05. Models: both Ouro and Huginn. Same dtype, paged attention, per-model legal KV policy as upstream. Pin Rmax/threshold when using traces.
- Measure **one full Core invocation**, whole recurrent wave, CUDA-event GPU elapsed, synchronized host wall, graph capture/hit rate, graph recapture/padding buckets, KV traffic/update, compaction, GPU HBM peak; separated Prelude/Coda/LM-head. >=2 warmups, >=5 scored repeats (>=10 selected); randomize B/L execution order where safe.
- Where profiling permitted: Nsight Systems GPU-active and launch gaps, Nsight Compute DRAM bytes, FLOPs, occupancy (no invented counters). Mark bottlenecks only when supported. Determine operational B* by predeclared +10% Core latency threshold relative to B1 within fixed L; never claim B* itself is the rigorous roofline transition.
- Compare to Phase 05 eager as **cross-runtime descriptive** only, after matching models, B, L and KV semantics or identifying confounds.

### E2 — architecture & stage-breakdown study
- Measure optimized Core / Prelude / Coda / head / attention-KV / metadata staging under identical fixed B,L settings, and effective share in actual decode round using recorded depth distributions.
- Ouro has zero *Transformer* Prelude/Coda layers, but embedding, LM head and scheduling are nonzero. Huginn has 2+4+2. Report stage time % of synchronized decode wall without double-counting overlapping timers.
- Test short (L<=128) and long (L>=1024) context, identify when Core calls are flat versus B-dependent, map KV-policy/backend dependencies. Attribute only with within-model controls; do not infer causal architecture effect from mismatched model widths alone.

### E3 — physical FIFO versus Oracle no-refill grouping
- Create an explicit **cohort-selection extension point** before the no-refill recurrent wave (not simply sorting the recurrent `ready_queue` after launch). Do not modify the Core kernels or attention implementation between policies.
- Use fixed offline ready pool, N={32,64,128,256} and B={4,8,16,32,64} where HBM permits; 32 and 64 decode steps; Alpaca/ShareGPT frozen inputs. Require N>B positive controls, N=B zero-headroom controls, and random grouping.
- FIFO is stable request ID/arrival order; Oracle sorts by true *current-token* recurrent exit depth using an identical frozen replay trace, then stable request ID. At each decode boundary apply selection without violating per-request token causality. Exactly same request-token IDs, positions, depths, `sum_i d_i`, KV policy, Rmax, EOS/generation length and model parameters across methods. Record per-wave membership and Core active-width sequence, RIC=sum over waves of max depth, tokens/s, TTFT as appropriate, GPU core/wall time and all stage overheads.
- **Evidence taxonomy**: TRACE_ONLY; PHYSICAL_FORCED_DEPTH; PHYSICAL_LIVE_EXIT; LATENCY_MODEL. Never label replay as native quality or live speed. Compare real exit gate separately only after BF16 numerical/quality agreement is acceptable. No new predictor in this phase.
- Use >=2 warmups, >=5 independent scored repeats, interleaved/randomized order, median/spread and raw per-trial receipts. Exclude no measurements silently, keep OOM/failures. Matched datasets and exact token/work hashes are mandatory.

## Reporting artifacts
Commit `phase06/README.md` (commands and exact pin), executable scripts/tests in `phase06/`, and `phase06/results/<run_id>/` with `RUN_MANIFEST.json`, `RESULTS.md`, `REVIEW.md`, raw trial CSV/JSONL, plots and hashes. All code and results should be pushed to this branch; report SHA. Plot Core latency vs B at L, bottleneck-by-regime and model stage shares, Oracle gain vs N/B and depth CV, optimized CDB vs Phase05 matched descriptive comparison. Report speedup separately from % time reduction and label uncertainty.

**Decision table required**: RQ1 bottleneck evidence; RQ2 measured architecture costs; RQ3 FIFO→Oracle physical decode change; whether optimized CDB preserves headroom; next practical-predictor GO/NO-GO. If bottleneck counters cannot be collected, label unresolved and retain traces. Stop at the relevant gate rather than manufacturing a positive result.
