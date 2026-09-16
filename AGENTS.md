# AGENTS.md

## Current priority — four-H100 model-specific no-refill study (2026-09-16)

Read in order:
1. `phase01/FOUR_GPU_REPLAN_20260916.md` — canonical model/KV/depth/workload contracts and four-GPU lanes.
2. `phase01/CURRENT_RESEARCH_PRIORITY.md` — compact current scope.
3. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — evidence/publication contract.
4. `phase01/results/20260915_poc/RESULT_REVIEW.md` and `review_manifest.json` — first PoC evidence.

### Research target

Study offline **pure no-refill** micro-batch formation for depth-adaptive looped LMs. Scheduling occurs at token boundaries; an exited recurrent slot is never refilled inside the current recurrent interval.

### Important reset

Do not launch the old one-GPU monolithic matrix unchanged. Huginn's previous generic paths defaulted to `full` KV and exposed a memory-capacity confound at deep/B64 cells. Preserve those OOMs, but select and freeze a model-appropriate adaptive KV contract before the main Huginn scheduling matrix.

Ouro and Huginn must not be forced into one identical KV policy:

- Ouro-1.4B / 2.6B: Rmax=4, learned exit thresholds, native/reference Ouro KV semantics for the main comparison.
- Huginn-3.5B: Rmax=16, latent-diff thresholds, bounded KV gate over `full`, `latest-m4`, and `latest-m4-compress-s4`; full is capacity/reference only.

Never reduce Huginn Rmax just to avoid OOM.

### Four logical H100 lanes

- A: Ouro-1.4B full study.
- B: Ouro-2.6B full study.
- C: Huginn canonical adaptive study.
- D: Huginn KV/HBM/parity/GSM8K gate and confirmation/repeat cells.

Verify live GPU ownership before binding logical lanes to physical IDs. Use separate worktree/output roots per running lane; do not hot-patch an in-flight tree.

### Main measurements

B=`4,8,16,32,64`; workloads Alpaca + ShareGPT. Collect fair headroom, depth distributions, native whole-model H100 BCT/throughput, peak HBM, model-specific timing surfaces, hardware-calibrated replay, and integrated physical no-refill timings after parity gates.

Evidence labels remain separate:

- trace/headroom;
- native whole-model E2E;
- hardware-calibrated trace replay;
- integrated no-refill runtime.

Ouro's native HF gate is post-hoc and cannot be called physical early exit. An integrated Ouro speed claim requires a recurrent-loop adapter that actually skips exited-token work.

### Commit/publication protocol

Follow the result-bound protocol used in `sub-moe`:
1. plan/source commit before launch;
2. bind exact source SHA, model revisions, KV policy, dataset contract and GPU identity;
3. executing agents publish compact results + `RESULT_REVIEW.md` + `review_manifest.json`;
4. retain OOMs, parity failures and negative cells;
5. separate software readiness from scientific support.

Preferred subjects: `phase01: plan ...`, `phase01: implement ...`, `phase01: publish ...`, `phase01: review ...`.

Do not invent completed runs, speedups, cache semantics or missing measurements.