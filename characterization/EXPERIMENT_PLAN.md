# Looped Transformer Inference Characterization — Experiment Plan

**Status:** Experiment plan only. No benchmark code or H100 runs are claimed by this commit.
**Repository:** heonniy/batch-loop
**Branch:** research/looped-inference-characterization-20261022
**Scope:** Q1–Q6 below using *only* CDB and vllm-rlt as runtime codebases. Phase07 depth prediction is independent and out of scope. No new depth-aware grouping or scheduling policy in this study.

## 0. Goal and research methodology

Characterize practical looped-transformer inference bottlenecks, determine whether dynamic early exit, refill, KV management and asynchronous execution meaningfully improve *wall-clock* serving and offline processing performance. Do not infer benefits from FLOPs, logical active slots, or trace-only timing.

**Reference runtimes**
1. Continuous Depth Batching (CDB): https://github.com/kschwethelm/continuous-depth-batching — recurrent decode, fixed/dynamic depth, no-refill/refill, recorded exit replay, stage timings.
2. vllm-rlt: https://github.com/ThinkFlowLab/vllm-rlt — serving-oriented runtime comparison, optimized execution, async scheduling where supported.

Pin each repo to a recorded exact commit SHA, CUDA/Torch versions, weights revisions, tokenizer revision, GPU UUID, models and configs in RUN_MANIFEST before execution. Prefer the CDB SHA already pinned by Phase06 (abdfae0adbdd12be8c6123ccb34eee98a38afb39) if still reproducible; do not pretend a different commit is identical. Review vllm-rlt supports/limitations prior to comparison (Huginn dynamic depth, CUDA Graph, KV policy, async correctness). Unsupported combinations are N/A, not failed wins.

**Equipment:** 4 x H100, assigned only from GPUs explicitly authorized and currently idle; launch independent single-GPU lanes (no GPU sharing, no accidental use of GPUs allocated to other people). Record UUIDs and topology. Cross-over a key subset of conditions across devices to control GPU differences. Use one serialized run per GPU; log thermals, clock, memory peaks, unexpected competing processes. If only subset of GPUs is available, preserve correct measurements instead of overcommitting.

**Models:** Ouro-1.4B and Huginn-3.5B primarily; Ouro-2.6B optional. Both precision BF16 when supported; pin exit thresholds and min/max recurrent depths separately for each model and show why. Dataset: Alpaca / ShareGPT / long-context ArXiv-style workload using recorded source revisions and reproducible subsets. Fix prompt tokenization and generated output token target (e.g. 64 and 128); distinguish forced same token trajectories from true generation. Report task accuracy / next-token parity for live exits.

## 1. Common instrumentation and controls

- Record TTFT, TPOT, end-to-end request drain time, aggregate output tokens/s, latency percentiles, per-stage CUDA event timings, CPU scheduler time, GPU idle/launch gaps, peak HBM and KV block counts, token- and loop-level active widths, OOM/failures.
- Stage attribution: Prelude / per-loop Core / Coda plus gate/convergence decision, token sampling, host/device transfer, KV cache updates, scheduler, inter-stage waits. Quantify invocation counts, latency medians/distributions, and percentages of *non-overlapping* GPU busy time versus overlapped CPU work; don't add overlapping stage wall times as if disjoint.
- Profile representative cells with torch profiler / Nsight Systems when available; CUDA events with synchronization discipline for isolated stage measurements; benchmark warmed CUDA graphs separately from capture/startup.
- Stable benchmarking: >=2 warmups and >=5 timed repeats per primary cell, randomized policy execution order, raw per-trial data, median with dispersion (also means/std where sample count supports), report uncertainty and negative cells. Avoid aggressive matrix explosion: pilot first, preselect representative cells, then expand.
- Fair comparisons require identical model, prompt set, output token budget, precision, KV semantics, batch-slot cap Bmax, GPU, exit thresholds and code-path compatibility. For fixed-depth vs dynamic-depth, report quality and actual generated token divergence; replay comparisons must use matched recorded depth + trajectory.
- Official engine performance != vLLM production baseline; CDB and vllm-rlt have architectural differences. Comparisons across runtimes are systems comparisons, not controlled causal attribution of one scheduler component.
- **Known caveat:** Earlier batch-loop Phase04 BF16 B1 vs batched live exits were not strictly identical; Phase05 forced-depth timing used an eager path. Revalidate parity before calling any frozen B1 depth trace a batch-independent ground truth. Verify appropriate model-specific exit policy and offset/delayed-gate consumption.

## Q1 — Prelude / Core / Coda latency breakdown

**Question:** Where is inference time spent on actual Ouro/Huginn, and how does stage share change with batch size and context?

- On CDB measure stage latency for Bmax=2,4,8,16,32,64 (128 if feasible) and context L=128,512,2048,8192 (skip infeasible combinations with reasons); use matched output length.
- Stage decomposition by model and policy: aggregate cumulative per-stage GPU time, per-call distribution, invocations/token, Core live width b per recurrent iteration, weights/kernel launch time, attention/KV update path where possible.
- Report stages separately for prefill vs decode. Ouro has no Transformer Prelude/Coda stack, but non-Core operations are not literally free; do not equate “0 layers” with “0 latency.”
- Output breakdown stacked charts and absolute values; explain underfilled small-batch execution vs true idle gap.

## Q2 — Fixed-depth vs Live Exit vs Recorded Replay

**Question:** When does dynamic early exit actually improve output throughput and where are costs?

- Fixed-depth: fixed max recurrence; Live Exit: actual trained Ouro hazard / Huginn latent-convergence; Recorded Replay: pre-recorded per-token step decision applied on an otherwise matched runtime.
- Pin/verify decision horizon (same-step vs delayed consumption), thresholds, max/min depths, token/sequence mapping. Replay same trajectory, same wave grouping if isolating exit-decision overhead; strict next-token/quality parity checks on live paths.
- Report executed token-loops, GPU Core launches, host sync count/time, exit-signal D2H transfer, idle, stage work and overall drain; replay is NOT deployed unknown-depth oracle and can omit different overhead.
- Include no-dynamic-depth control; if speed worsens despite token-loop reductions, report why. Treat semantic quality as an essential axis.

## Q3 — Refill vs No-refill

**Question:** Which policy wins in each model/context/batch regime and what explains differences?

- On CDB run both policies with identical Bmax and memory budget, model, exit threshold and workload. Align recorded replay to isolate scheduler effects, then inspect live policies where correctness passes.
- Sweep Bmax=2,4,8,16,32,64 and context L=128,512,2048,8192 within memory capacity; practical pilot can use B=4,16,64 and L=512,2048.
- For Refill, record minimum Coda batch threshold, prelude/coda batch sizes, stage call counts, frequent interleaving, recurrent active width, queued/re-entered tokens and refill events. No-refill wave-level admissions distinct from request-level completion.
- Report best valid policy on each cell, not assume Ouro refill or Huginn no-refill a priori.

## Q4 — KV cache constraints

**Question:** How does recurrent KV policy and context length constrain slots, traffic and runtime?

- Record KV layout for each model/runtime, whether history is shared or indexed by recurrence step, block size, measured bytes/token/sequence and peak HBM with other residency accounted for.
- Sweep context length, max slots, prompt/output token mix; record OOM boundary, KV utilization, fragmentation, allocations, eviction/offload/preemption (where truly implemented), KV update and attention latency.
- Avoid claiming C (resident sequences) and Bmax (active Core batch width) are independently configurable if not supported in current code; when coupled, explicitly state this.
- Isolate prefill vs decode memory pressure; KV cache effective capacity not equivalent to nominal H100 80 GB.

## Q5 — CDB vs vllm-rlt

**Question:** How do different looped runtimes perform on **common supported** configurations?

- First create a feature/parity matrix per repository SHA: models, fixed/dynamic depth, refill/no-refill, CUDA Graph, KV semantics, async, preemption, availability of stage timer.
- Primary head-to-head: common Ouro-1.4B BF16 fixed-depth, matched token lengths, Bmax, context, GPU, warmup, output budget and KV correctness. Add other shared cases only after verifying support.
- Report absolute metrics, CPU overhead, GPU busy/idle timeline, cache footprint, kernel counts, CUDA Graph replay/capture, performance vs context and batch. Disentangle capability limitations and cross-engine differences; do not attribute all differences to scheduling or dynamic depth.
- If vllm-rlt Huginn only supports synchronous fixed-depth, do not invent refill/live-exit cells.

## Q6 — Async scheduling opportunities

**Question:** Does asynchronous host scheduling / exit-signal transport meaningfully remove bubbles, and what overhead remains?

- Within each runtime where valid switches are supported, compare async on/off under matched conditions, and separately CUDA Graph on/off if support and correctness permit.
- Record CPU request/scheduler wall, GPU idle gap from trace, kernel/launch and D2H sync timing, overlap of host work + GPU compute + copy, GPU Core underutilization distinct from idle.
- Separate true concurrent GPU kernels from async host prep/side-stream D2H; don't claim Core, Prelude, Coda compute overlap if serialized on same compute stream.
- Include correctness tests under async paths and known issues with async preemption/exit timing; supported paths only. Identify where additional optimization might actually help, without implementing speculative optimizations.

## Prioritized execution order

**Stage 0 — Environment/parity:** pin SHAs, install both runtimes independently, validate model output, exit policy and live-depth batching parity; collect feature compatibility matrix.
**Stage 1 — Pilot:** Q1/Q2/Q3 on CDB for Ouro and Huginn at L=512, B=4/16/64, output=64. Verify metrics and trace capture before sweeping.
**Stage 2 — Memory & regimes:** Q1/Q3/Q4 context and width sweeps, selected quality cells.
**Stage 3 — Engine comparison:** Q5 on common fixed-depth Ouro first; introduce dynamic depth only if supported and validated.
**Stage 4 — Async diagnosis:** Q6 ablations, Nsight representative cells.
**Stage 5 — Synthesis:** bottleneck map by model/context/batch/policy; rank next research opportunities (scheduling, KV, runtime), including negative findings.

### Independent 4-GPU lanes (tentative, NOT fixed device IDs)

- Lane A: CDB / Ouro-1.4B
- Lane B: CDB / Huginn-3.5B
- Lane C: vllm-rlt / Ouro-1.4B
- Lane D: vllm-rlt feature/smoke + async and cross-check replicates; Huginn on vllm-rlt only if actual support verified.
- Each lane may switch assignments after pilots. Avoid concurrent profiling/system interference that compromises measurement.

## Output contract

Proposed structure:
- `characterization/README.md`: summary, reproduction commands, model / environment manifest.
- `characterization/EXPERIMENT_PLAN.md`: this document.
- `characterization/scripts/`: measured scripts only after implementation.
- `characterization/results/<run_id>/RUN_MANIFEST.json`: SHAs, GPUs, config and data hashes.
- `characterization/results/<run_id>/raw/`: per-trial structured raw rows.
- `characterization/results/<run_id>/RESULTS.md`: Q1–Q6 tables, plots and interpretation.
- `characterization/results/<run_id>/REVIEW.md`: parity, correctness, measurement QA, caveats, failed/OOM cases.
- Figures: stage breakdown, depth-vs-performance, refill/no-refill regime map, KV-memory boundary, cross-runtime throughput, async CPU/GPU timeline.

No invented numbers. Label result cells `PASS`, `FAIL`, `OOM`, `N/A unsupported`, `NOT RUN` as appropriate. Commit measurements separately from plans. Explicitly preserve all prior Phase01–07 files.

**Research decision:** This is an exploratory characterization, **not** a commitment to depth-aware grouping or predictors. Use evidence to choose the next system problem.
