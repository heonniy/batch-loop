# AGENTS.md

## Current priority — one combined three-model headroom + runtime block (2026-09-16)

Read in order:
1. `phase01/README.md` — research question and existing evidence.
2. `phase01/CURRENT_RESEARCH_PRIORITY.md` — exact combined experiment matrix and claim boundary.
3. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — evidence/publication contract.
4. `phase01/results/20260915_poc/RESULT_REVIEW.md` and `review_manifest.json` — first H100 PoC.

### Existing evidence

The first H100 PoC collected real model exit-depth traces and then simulated offline no-refill micro-batch formation. It supports an exploratory scheduling-opportunity claim only.

- Ouro-1.4B showed substantial oracle slot-utilization headroom over FIFO at B=4..64.
- Huginn at convergence threshold 0.12 started near full slot utilization and showed little oracle room.
- Previous/recent/sequence-mean causal predictors recovered only a small fraction of Ouro's oracle headroom.
- The first simulator truncated the remainder after strategy-specific sorting, so strategy totals were not always based on exactly the same token subset. `analyze_fair.py` fixes this by freezing the common subset first.

### Owner-directed combined block

Do NOT stop after a characterization-only phase. Use one assigned H100 and complete as much of the following block as can be validated from the same source checkpoint:

1. **Models:** Ouro-1.4B, Ouro-2.6B, Huginn-3.5B.
2. **Paper-facing micro-batch sizes:** B = 4, 8, 16, 32, 64 only.
3. **Threshold sweep:**
   - Ouro q = {0.2, 0.4, 0.5, 0.7} for both 1.4B and 2.6B.
   - Huginn latent-diff convergence threshold = {0.5, 0.35, 0.26, 0.12} with max depth 16.
4. **Fair headroom:** FIFO/random/history/oracle must see the identical eligible token/request subset. Report depth distribution, slot utilization, idle-slot ratio and oracle-minus-FIFO headroom for every model/threshold/B.
5. **Physical timing in the same run:** measure real H100 fixed-depth first-token and incremental decode latency over B=1,2,4,8,16,32,64 (1/2 are internal calibration points) and representative contexts. Fit boundary and recurrent-step latency surfaces and combine them with the observed depth traces to produce hardware-calibrated throughput/BCT replay for every threshold and batching policy.
6. **Integrated real no-refill runtime:** the final paper claim still requires a custom runtime that physically compacts the active batch after exits while preserving KV semantics. Implement and validate it in this same development cycle when feasible; do not call the hardware-calibrated replay an integrated runtime. Before trusting integrated timing, require fixed-depth parity, token/output parity on a small deterministic cohort, correct KV behavior under mixed exit depths, and agreement between logged active batch sizes and the trace.
7. **Runtime metrics:** generated tokens/s, total offline BCT, recurrent wall-clock, prefill/boundary time, scheduler/compaction overhead, and speedup versus FIFO no-refill. Compare oracle grouping and any practical causal grouping that is already implemented. CDB refill is a reference baseline when an executable compatible implementation is available; do not fabricate it from paper numbers.

`run_h100_full_sweep.sh` is the one-command source checkpoint for items 1–5. `runtime_calibrate.py` performs real H100 timing plus hardware-calibrated replay. Those replay numbers MUST carry the evidence label `HARDWARE_CALIBRATED_TRACE_REPLAY_NOT_INTEGRATED_RUNTIME` until item 6 is actually implemented and passes parity.

### No artificial stage inflation

The owner explicitly asked not to split easily combinable checks into many approval stages. Within this bounded matrix, agents may proceed from fair traces -> physical calibration -> hardware replay -> integrated runtime engineering/validation without waiting for another owner message, provided failures are retained and claim boundaries are respected. Stop only on resource/correctness failure, completion of the matrix, or a result that makes further runtime work scientifically meaningless.

### Repository / GPU safety

- Use one explicitly assigned H100 only. Never assume a GPU number without checking the launch environment.
- Do not kill unknown PIDs, reset GPUs, force-push, rewrite historical results, or silently change model revisions/datasets.
- Bind each run to a clean source commit and separate output root.
- Preserve adverse results. A model/threshold with no headroom is evidence, not a failed experiment to hide.
- Never invent a launch receipt, completed GPU run, physical speedup, BCT, or missing artifact.

### Commit and publication protocol

Follow the same result-bound discipline as `sub-moe`:

1. Commit source/plan before launching.
2. Save source SHA, exact command, model revisions, environment and GPU identity in the run root.
3. The agent that executes the experiment publishes compact CSV/JSON/plots plus a human-readable result review and `review_manifest.json`.
4. Separate measurement, implementation, simulator/replay, comparison and hypothesis status. A software PASS is not a scientific GO.
5. Large raw traces may remain server-side only if their path/size/hash are recorded; compact derived evidence must be committed.

Preferred commit subjects:
- `phase01: plan <bounded experiment>`
- `phase01: implement <runner/instrumentation>`
- `phase01: publish <result block>`
- `phase01: review <result and decision>`

### Claim boundary

`slot_utilization` = trace geometry.
`hardware_replay_*` = real H100 latency calibration combined with trace replay.
`integrated_*` = only a real dynamic no-refill engine that physically executes the compacted batches and passes parity.

Never promote one evidence level into the next in prose, plots, or commit messages.
