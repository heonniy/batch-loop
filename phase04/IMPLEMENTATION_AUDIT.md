# Phase 04 — Implementation audit before depth-predictor research

Reviewed baseline: `a4a089811f53d4acf300074cfb56d27f5c4f1f8b` (2026-09-17 review).

## 1. Verdict and evidence scope

**The survivor-runtime skeleton is present and its principal exit/KV formulas are internally plausible. It is NOT yet certified as a schedule-invariant, live-adaptive, end-to-end benchmark. Do not equate forced-depth GPU success with that certification.**

This document records a source review, not a new H100 run. Repository files were read through the GitHub connector. No model weights were loaded and no server GPU process was started. A local CPU-only mechanism check (PyTorch 2.10.0+cpu) checked the cumulative-gate identity and constructed RNG/sparse-loss counterexamples. Those are independent reference calculations, NOT a run of the repository's integration tests. Fetches of the exact Hugging Face revisions were unsuccessful; accessible public model source supports the high-level comparisons below, but comparison with the executor's pinned cached source remains mandatory.

Reviewed files:
- `phase02_runtime/{common,kv,ouro_adapter,huginn_adapter,test_runtime,oracle_ric}.py`
- `phase03_predictor/{collect_huginn,collect_ouro,tournament,physical}.py`
- `phase03/predictor_tournament_results_20260916/{RESULT_REVIEW.md,physical_matrix.csv}`
- `AGENTS.md`, `phase01/EXPERIMENT_REVIEW_PROTOCOL.md`

Selected baseline blob identifiers:
- common.py: `1ed88b75d2fd777a0030299492b31fba763c4f13`
- kv.py: `96608ba2a7c26ebcdfadd8af6ca8c35e23e7930c`
- ouro_adapter.py: `9ba67bec161c53b689b43987360231fd709ae13e`
- huginn_adapter.py: `1dd8a743897f56efb3b2d08f5d6e27727b4e5678`
- collect_huginn.py: `0a563282aaf38ac028f170ca24f26d319eb90fdd`
- collect_ouro.py: `6afaea6a16b6bb085a977a4454968a9d2dc1a4de`
- tournament.py: `4f6a188727c5ec7cbc9251fc9d3d3ae6f6975ca2`
- physical.py: `983963e195b2c59e4e8bb1c68d404c6db9064cad`

## 2. What appears correct in the reviewed source

| Component | Source observation | Limit of this finding |
| --- | --- | --- |
| Survivor removal | `run_interval` scatters exited states to their original local positions and index-selects surviving state, embedding, request IDs and persistent KV row IDs together. | Static review and existing tests do not replace a real-model batched live-exit test. |
| No-refill | A started microbatch only shrinks. Coda follows that microbatch's complete recurrent interval. No new row is inserted into the interval. | Current physical runner also imposes a global token boundary across all N requests; this is an experimental scheduling choice. |
| Ouro gate | Uses lambda=sigmoid(gate(after)), remaining*=1-lambda, exits on 1-remaining >= q, forces completion at Rmax=4. `remaining` is indexed by persistent row and reset at a new token. | Rmin is effectively 1 in this adapter. Do not describe it as CDB's Rmin=2 or silently change it. Threshold rounding and pinned-model parity remain to be checked. |
| Ouro recurrent state | Final model norm is applied after every core traversal, then reused as the next recurrent state. | Accessible public Ouro source uses this placement too; exact cached revision must be verified. |
| Ouro last-exited KV | Maps flattened index to (recurrence, physical layer); on exit copies only the exiting rows' current-position KV to later recurrence slots. | This is the chosen approximate KV policy, not native full-depth equivalence. |
| Huginn exit | Relative L2 change per row, evaluated in FP32: norm(after-before)/clamp(norm(after)); strict `< threshold`; forced Rmax=16. | Validate each row independently, especially threshold ties and nonfinite values. |
| Huginn shared KV | Core slots wrap by physical layer; exited rows retain their last current-position write; history grows once per completed token, not once per recurrence. | For the current 2/4/2 architecture. Assert source block-index conventions and config rather than assuming other architectures work. |
| Position bookkeeping | Prompt RoPE uses per-row valid-token positions; a new decode position is computed after adding one valid KV column. | Needs heterogeneous padding, permutations and multi-token KV-read tests. |

The existing CPU test file exercises forced compaction, last-exited row fills, shared slots and history growth. Its fake adapter explicitly rejects live-gate calls. Therefore those tests cannot certify the real live gate or native model execution path.

## 3. Findings that must be resolved or explicitly bounded

### F1 — Huginn collector/replay does not bind random latent state identity [HIGH; source-confirmed gap]

`collect_huginn.py` seeds and consumes CUDA RNG to create a Gaussian feature projection, then prefills the entire collection cohort and initializes decode states one request at a time. `physical.py::_run_once` reseeds before a different-size held-out prefill, loads rather than generates the projection, and initializes states in B-sized chunks. Neither pipeline records/replays per-(request, token) initial latent tensors or an identity-keyed RNG contract. A shared seed number does not make these random trajectories the same.

The public Huginn initialization uses random draws, and its current core can also inject test-time noise depending on config. Verify the pinned source and actual noise config. A CPU counterexample with native-style randn+trunc_normal showed different tensors for one (4,1,32) call versus four (1,1,32) calls at the same seed; consuming projection RNG first also changed the tensor. This demonstrates the reproducibility mechanism, not its numerical contribution to this checkpoint's depth error.

Required: bind stochastic state to canonical request identity, absolute token position and independent RNG stream purpose; include prefill initialization and any per-recurrence noise. Reuse the SAME sampled states for reference/compact/FIFO/Oracle/Predictor comparisons, independent of execution order, padding, cohort subset, projection/training RNG, and microbatch width. Keep the intended initialization distribution; do not silently switch to zeros or previous-hidden warm starts. Record h0 checksums and config. Measure label sensitivity to seed separately; it is not evidence that weak predictors are caused by RNG until tested.

### F2 — Main physical comparisons force depths and input token IDs [HIGH; intentional scope, not itself a bug]

`physical.py` loads frozen token IDs each token boundary and calls `run_interval(exit_source="trace", forced_depths=...)`. A successful run proves that prescribed work was physically packed differently, not that each row's live gate chose that depth under the current batched state.

The primary committed matrix reports Huginn FIFO prefill token match 0.97265625 and next-token match 0.973034274..., versus the collector. Ouro reports 1.0 prefill and 0.991809475... next-token match. These are recorded discrepancies, not measured task accuracy and not proof of a specific bug. Match ratios alone are insufficient to diagnose RNG, numerical ties, masks, KV semantics or row mapping.

Required: compare compact execution against an independent row-wise reference with the SAME chosen KV/exit/init policy; record first divergent layer/recurrence, gate margins, logits and KV read/write identities. Preserve forced replay as a systems benchmark, but add batched live-exit verification before collecting new predictor targets. Do not enforce equal work on genuinely divergent live runs and then hide the divergence by forcing their exits.

### F3 — Diagnostic instrumentation contaminates timing and accumulates [HIGH for performance interpretation; source-confirmed]

`common.py::timed` synchronizes before and after small operations. `run_interval` repeatedly calls it for core, on_exit and compaction. `_StableRowKV` creates two CUDA events per update unconditionally on CUDA; collectors call adapter methods directly and do not drain `_update_events`, allowing the event list to grow with total writes. `finish_prefill` does not clear it, so the first `pop_update_gpu_ms` can attribute earlier prefill/Prelude events to a recurrent pass. Ouro `fill_receipts` also appends CPU row lists at each copied slot. Trajectory extraction performs CPU transfers in predictor strategies.

Required: separate `debug/profile` and low-instrumentation `benchmark` modes. Disable or boundedly drain events/receipts for collection and normal benchmarks, reset phase-specific measurements at boundaries, and preserve optional detailed traces outside scored timing. Include all feature/trajectory maintenance needed by the predictor in its real overhead. Re-measure FIFO AND Oracle with the same clean backend; removing extra launches/synchronizations may change the old Oracle headroom. The previous wall times were real prototype wall times, not fabricated, but are not optimized-engine ceilings.

### F4 — Candidate selection uses test results [HIGH; confirmed evaluation-protocol defect]

`physical.py::_top_three` filters `split == "test"`; old AGENTS also instructed test-RIC selection. This is partly a plan defect, not merely an executor mistake. Training/early stopping in `tournament.py` uses validation; the problem is using test outcomes again to select methods and physical winners, not proof that all training examples leaked.

Required: select architectures, features, thresholds, hyperparameters, batch/chunk settings and physical finalists using development/validation only. Freeze before opening a NEW confirmation cohort. Existing Phase-3 test results are exploratory selection evidence and cannot serve as untouched confirmation of subsequent choices.

### F5 — Checksums and split audits are incomplete [MEDIUM; source-confirmed omissions, impact unmeasured]

Raw prompt hashes are checked, but truncation at 64 tokens can make different raw prompts identical model inputs. No tokenized-prompt cross-split audit is present in the inspected collector/tournament path. Physical replay records hashes but does not validate every consumed feature/checkpoint artifact against the expected stored hash. Feature shape/projection and model-policy compatibility need explicit checks.

Required: group duplicates at the tokenized, templated, truncated input level; split conversation/document families together. Preserve canonical IDs across subset/chunk loading. Validate labels/tokens/projection/head checksums and policy signatures before replay. Report actual duplicate counts; do not claim duplicates occurred without inspecting the data.

### F6 — Shape/config/measurement boundaries need explicit contracts [MEDIUM]

Current physical code rejects N not divisible by B, forces fixed token counts and does not exercise EOS removal. Huginn core indexing hard-codes the current 2-layer Prelude offset. Its local attention wrapper has no explicit GQA fallback; assert actual Q/KV-head compatibility rather than claiming it is broken for the tested checkpoint. Ouro should assert layer-mask configuration rather than silently treating any future sliding-window config as full attention. Both adapters need explicit finite-value, valid-threshold and Rmin/Rmax checks.

Required: same-policy tests for incomplete batches, padding, request permutations, and repeated boundaries, plus an explicit choice between fixed-horizon and EOS-driven completion. Unsupported configurations must fail closed. Label prefill-excluded decode time as such; full workload BCT includes prefill, scheduling and all selected requests, excluding only one-time model load/warmup under a stated convention.

### F7 — Predictor limitations are not predictability ceilings [RESEARCH LIMITATION]

The existing study uses fixed Gaussian projections, small heads and a bounded training search. Huginn does not provide previous exit hidden or h0 to the predictor; Ouro's previous hidden is projected with token-0 zero fallback. C7 already learns cumulative survival outputs but collapses them to expected depth for sorting. C9's single adjacent-group swap term can have zero immediate regret when multiple deep items contaminate a shallow group. CPU example: [3,3,12,12] + [3,3,12,12] has RIC 24; one swap leaves 24; two swaps reach 15. This is a limitation of the sparse extra objective, not proof the entire C9 loss is zero.

Required: the plan's pattern analysis and full-hidden/prior/residual comparisons, not another claim that all pre-loop prediction fails. Calibrating a scalar monotonically cannot improve its ranking by itself.

### F8 — GPU ownership and execution provenance conflict [OPERATIONS]

Old AGENTS specifies GPU7/Huginn and GPU8/Ouro. Reviewed Phase-3 source hard-codes GPU7/Huginn and GPU6/Ouro and asserts a user correction. This review cannot resolve authorization from those contradictory files. Do not rewrite old results or silently select GPU6/GPU8. New GPU launches require an explicit executor-side authorized model-to-physical-GPU/UUID mapping; absent confirmation, stop only GPU execution and continue CPU/source work. This commit launches nothing and changes no running jobs.

## 4. Release gate

Do not start expensive recollection until Gate A in `DEPTH_PATTERN_PREDICTOR_PLAN.md` passes. Fixes must be new source commits and new result roots. Keep all Phase-1/2/3 results intact and publish a delta table naming which prior findings remain meaningful.

Required dispositions: F1 stochastic identity tests; F2 row-reference/live compaction evidence; F3 profile-free baseline and bounded instrumentation; F4 new selection protocol; F5 artifact/tokenized split audit; F6 config tests; F8 explicit launch authorization. F7 is addressed by research, not a mandatory positive result.

## 5. Reference sources

- Ouro public source: https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/modeling_ouro.py (`OuroModel.forward`, cumulative exit PDF/CDF).
- Huginn public source: https://huggingface.co/tomg-group-umd/huginn-0125/blob/main/raven_modeling_minimal.py (`core_block_forward`, `initialize_state`, `LatentDiffExitEvaluator`, `predict_from_latents`).
- Required local pinned checks: Ouro `37bc7e8ff85a90d813dc096383405f521b6a67b0`; Huginn `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. Save actual imported source hashes, config, versions and relevant callable source. Public main is NOT a substitute for those files.
- User-supplied EasyBalance, Algorithm 1 and section 4.2: select ready workloads and defer others; do not import its cross-layer execution or no-overhead claim into our strict no-refill design.
