# AGENTS.md

## Current priority — full three-model no-refill validation on one H100 (2026-09-16)

Read in order:
1. `phase01/FULL_E2E_EXPERIMENT_PLAN.md` — exact model/threshold/B matrix and required E2E evidence.
2. `phase01/CURRENT_RESEARCH_PRIORITY.md` — current execution scope.
3. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — evidence, publication, and review contract.
4. `phase01/results/20260915_poc/RESULT_REVIEW.md` and `review_manifest.json` — first PoC evidence and known limitations.

### Research target

Study **offline pure no-refill micro-batch formation** for depth-adaptive looped LMs. Scheduling happens at token boundaries; never refill an exited recurrent slot inside the current recurrent interval. The question is whether depth-compatible batch formation can remove recurrent valleys and whether that opportunity matters in H100 wall-clock execution.

### Current experiment is one combined block

Do not split this work into repeated owner-approval stages. On the assigned single H100, run the committed full sweep for:

- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`
- `tomg-group-umd/huginn-0125` (Huginn-3.5B)
- paper-facing B = `4,8,16,32,64`
- Ouro thresholds = `0.2,0.4,0.5,0.7`
- Huginn latent-diff thresholds = `0.5,0.35,0.26,0.12`

The same run must collect:

1. fair FIFO/random/history/oracle no-refill headroom;
2. per-threshold exit-depth distributions;
3. **actual native whole-model E2E/BCT/throughput** on the H100;
4. fixed-depth recurrent latency surfaces including internal active B=1,2;
5. hardware-calibrated no-refill BCT/throughput replay;
6. deterministic parity/failure receipts for batched adaptive execution.

Use `bash run_h100_full_sweep.sh` from the committed source. `e2e_benchmark.py` performs native whole-model timing and `runtime_calibrate.py` handles recurrent timing/replay.

### Evidence labels are mandatory

Keep these categories separate:

- **exit-trace collection:** real model forward/generation used to record token exit depth.
- **trace-sim/headroom:** reordering observed depths; not a speed measurement.
- **native whole-model E2E:** actual H100 `model.generate` wall-clock/BCT/throughput.
- **hardware-calibrated trace replay:** observed trace priced by real H100 timing; not an integrated compaction engine.
- **integrated no-refill runtime:** only if active survivor compaction and KV semantics are actually implemented and parity checked.

Ouro's currently pinned public HF implementation computes all recurrent steps and exposes its gate post-hoc. Therefore Ouro native E2E is a real whole-model baseline, but threshold timing is NOT physical early-exit/no-refill speedup. Never state otherwise.

Huginn batched adaptive timing is trusted only for cells whose deterministic batch-vs-B1 output parity passes. Preserve parity failures and unsupported/OOM cells.

### What is already evidence

The result commit `4ec78c1f5c0df7925f9888c7db3d75769e494ee7` showed exploratory Ouro oracle slot-utilization headroom and little headroom for Huginn at threshold 0.12. It is trace-based opportunity evidence only. A first-simulator defect was later identified: strategies independently truncated post-sort remainders. `analyze_fair.py` fixes this by freezing the subset before reordering.

Simple previous/recent/sequence-mean predictors captured only a small fraction of the initial Ouro oracle gap. Preserve that adverse evidence.

### GPU / repository safety

- Use only the explicitly assigned H100. Do not kill unknown processes or reset devices.
- Bind every result to source SHA, exact model revision, environment and launch command.
- Do not hot-patch an in-flight source tree. If source changes are required, stop, commit the fix, and restart into a new output root.
- Do not hide failures, OOMs, parity mismatches or negative threshold/model results.
- Do not invent a completed run, timing number, vLLM support, or integrated runtime.

### Commit / publication protocol

Follow the result-bound style used in `sub-moe`:

1. source/plan commit before the run;
2. execute the bounded committed runner;
3. executing coding/model agent commits compact results plus `RESULT_REVIEW.md` and `review_manifest.json`;
4. separate measurement / implementation / comparison / hypothesis status in the review;
5. only then plan the next method change.

Preferred commit subjects:

- `phase01: plan <experiment>`
- `phase01: implement <runner/instrumentation>`
- `phase01: publish <result block>`
- `phase01: review <result and decision>`

Large raw traces may remain server-side only when the result commit records verified path/hash/size; compact tables and review artifacts must be committed.
