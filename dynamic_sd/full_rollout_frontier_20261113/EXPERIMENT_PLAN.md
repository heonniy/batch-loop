# Full-rollout AR/SD crossover and K-frontier characterization in Looped Transformers

**Date:** 2026-11-13 | **Status:** PLANNED — NO NEW GPU RUNS OR MEASURED RESULTS
**Branch base:** `research/dynamic-sd-decision-characterization-20261113@4f6e9c50` (Stage A/B: retrospective CPU reanalysis).
**Objective:** Reproduce fixed-depth author settings on author benchmark distributions, then test native token-level dynamic exits where supported, without assuming oracle information is deployable.

## Research questions and anticipated null results

- **RQ1 (AR vs SD)**: Under matched native AR target semantics, are there operating points / token regimes where AR is faster than SD, and vice versa? Does that boundary change with *real* target depth heterogeneity after accounting for draft acceptance, prompt/task, context length and implementation costs?
- **RQ2 (SD optimum)**: When SD is beneficial, does the **best physical K** vary with exit policy and task, and with observed token-level depth D? Test both fixed best K and per-round action choice.
- **RQ3 (specificity)**: Are any AR/K decision gains specific to recurrent exit or already explained by generic acceptance+cost? A positive descriptive crossover alone is insufficient.
- **RQ4 (architecture replication)**: Are results consistent across Huginn, Ouro and Dory under explicitly comparable target semantics? Dory is a **separate elastic-budget, not yet validated native token-level halting, lane**. Never label changing Dory's globally selected loop budget as Huginn/Ouro-style live exit.

**Nearest records:** `F-079` live-vs-fixed matched AR; `F-080` post-hoc D/acceptance; `F-081` draft-depth negative/limited; `F-082` lockstep batching; `F-083` BF16 strict parity fails; `F-085` B1 discarded verifier share small; `F-087` bounded FP32 live FULL parity; `F-088` same-prefix oracle already fixed-depth. Latest source-only `4f6e9c50` Stage A/B disaggregates oracle: Ouro adaptive d near zero, Ouro K/A 1.017 (fixed) to 1.134 (tau=.1) and AR/SD depth-associated crossover; Huginn d gains strongest at fixed depth. This work must improve *evidence type*, not rerun the same K-conditioned one-round oracle.

## Paper / official code reproduction scope (read before model runs)

| Track | Official source | Original protocol to preserve | Not equivalent to our principal intervention |
| --- | --- | --- | --- |
| WaveFront / WFD | `summerbro-hhj/wavefront-decoding`, `experiments/exp2_spec_bench/run.sh` | Spec-Bench question IDs 81..560 (480 prompts, six task categories), greedy, default max prompt=512 and output=512, B=1, natural fixed R=4 Ouro / R=32 Huginn, d1 / d4; WFD vs DtV and matched AR. `exp3_accuracy/run.sh`: GSM8K and MATH-500 task accuracy, source config 8-shot / 4-shot default. | WFD's geometry is a native diagonal wave; **WFD ignores DRAFT_LENGTH** in official exp2. Cannot pretend a DtV K sweep is a WFD K sweep. |
| LoopSpec | `kaist-flexml-lab/loopspec`, `experiments/{110-speed-math,112-speed-coding,113-speed-general,tasks}.yaml` | Ouro-1.4B/2.6B, Raven (not Huginn), fixed R=4/32, SGLang, single concurrent request, greedy; base tasks GSM8K-CoT 3-shot, MATH-500 4-shot, BBH 3-shot, HumanEval+ and MBPP+ 0-shot. Uses one/two proposal-depth configurations, **not necessarily DtV K**. | Do not claim official LoopSpec on Huginn or Dory. A custom port is a different implementation and needs separate validation. |
| Dory1.0-4B | NVIDIA model/paper, `P-2611-03998` and model release | Elastic **global** loop budgets 1,2,4, BF16 release; paper's sequential self-spec baseline uses draft d=1, target R4, K4, **per-loop KV with activation reuse**, not default last-loop KV. | We have **no verified native per-token live-exit gate** or checked official standalone speculative benchmark code. Use R1/R2/R4 fixed baselines and label any proposed controller-defined depth as a **new target**, not native live. Never compare last-loop and per-loop as exact outputs without parity. |

### Benchmarks and source fidelity

Use the **same prompts and exact prompt templates** within a model/exit comparison:
- Primary overlap: GSM8K-CoT (LoopSpec 3-shot) + MATH-500 (LoopSpec 4-shot); preserve WaveFront 8-shot GSM8K and 4-shot MATH500 in a separate **reproduction lane**, never silently blend. Looped checkpoint chat templates and reasoning settings pinned.
- Across task families: **Spec-Bench** six categories (480 specified question IDs); **BBH**, **HumanEval+**, **MBPP+** (LoopSpec base configurations). Limit 32 smoke, then 100/200 per model-task where available; full Spec-Bench is 480 and MATH-500 is 500, no invented sample count.
- Do not run generated code outside a sandbox; HumanEval+/MBPP+ quality evaluation requires isolated approved execution or report generation-only statistics and mark pass@1 **not evaluated**.
- Quality: task EM/math_verify/pass@1 only when scorer is actually used, plus sequence parity versus matched AR. Fixed and live may produce different correct outputs; compare accuracy per matched target separately, not assert cross-target exactness.

## Stages and gating

### S0 — setup / implement exact comparator (no interpretation until parity)
1. Pin model revisions, upstream Git SHAs, CUDA/torch/attention backend, H200 host and GPU IDs; check memory/precision. Native Huginn exit may use different rule than WaveFront relative-L2 `epsilon`: explicitly record the **selected rule**, threshold and calibration histogram. Ouro learned hazard uses `tau`.
2. Separate output-accurate **FP32 FULL per-loop KV** primary correctness lane from matched BF16 performance lane; for Dory add `per_loop` / `last_loop` policies in **distinct** experiments. Do not mix unverified stochastic Huginn initialization seeds or BF16 strict failures.
3. Implement / integrate named actions `AR`, `DtV(d,K)` and `WFD(d)`; report WFD separately for its asynchronous wavefront geometry. For Dory, do not start until a true per-loop reference/SD implementation is verified on the same checkpoint; if unavailable, mark `blocked`, not `0% gain`.
4. Golden correctness: **same prompt, same dtype, same KV policy, same native exit rule, same seed, same target** with targeted fixed/live strict parity at smoke n32. Record exact output/exit-depth mismatch details. No claims of lossless BF16 until demonstrated; if fails, carry paired accuracy and mismatch fraction as an explicitly non-exact lane.

### S1 — **new** full-rollout fixed action frontier (main deliverable)
1. For each `(model, dataset, exit-policy, dtype, backend, B=1)`, run complete teacher-free greedy generation for **AR** and **fixed SD** with each K, matched prompts and output budget. Count only verified committed tokens. Use K in `{1,2,4,8,16}` where implementation supports, mark unsupported K16 instead of synthesizing.
2. Fix `d` as prespecified initial setting: Ouro1.4/2.6 d1; Huginn d4; Dory sequential self-spec d1 with R4, K4 as paper anchor, then K sweep *only when parity and ability to vary K are verified*. No adaptive d under test in main grid; d sensitivity (Ouro d2, Huginn d2/d8) separate and secondary.
3. Native live policies **only** Ouro `tau∈{0.4,0.1}` and Huginn `epsilon∈{0.1,0.2}` after matching exit rules/quality; fixed depth anchor R4 and R32. Calibrate threshold distribution before interpreting mixed exit. Dory R1/R2/R4 fixed global budgets are a separate elastic-depth comparison; no per-token gate claimed.
4. Record end-to-end wall and prefill-vs-decode wall on **every rollout**, with 5 timing repeats after warmup where affordable, randomized policy order, same GPU/workload, prompt-level bootstrap 95% CI. Throughput `Σ generated committed tokens / Σ wall`, not mean of per-prompt ratios. Report `Speedup(K)=T_AR/T_SD(K)` for same target; `K_opt` by held-out calibration prompts, with an uncertainty band and throughput difference to runner-up.
5. Required primary figures: (a) **AR/SD crossover** versus K, (b) **K → end-to-end throughput** per exit regime, (c) **fixed/live optimum K** at matched quality, (d) same result across datasets and models. Distinguish a *different best grid point* from statistically reliable optimum change. Pre-register minimum effect `>2%` with prompt-bootstrap CI excluding one for claims of a reproducible gap; report close ties as ties.

### S2 — matched-prefix, depth-conditioned physical cost decomposition (diagnostic)
1. Use periodic matched native AR KV checkpoints (e.g. every 16 generated tokens) **and** real full rollouts; record next-token `D_ref`, all verifier source depths `D_i`, K, draft accepted prefix `A`, target_core_calls/rows, wall by draft/verify/prelude/coda/host/gate, active width, wavefront width, KV traffic if measured, GPU sync and graph bucket.
2. Post-hoc `D_ref` and `Σ D_i` strata: AR/SD and K_opt changes with prompt/bootstrap CIs, n per bin. Never treat D_ref or realised acceptance as available pre-action.
3. Separate *acceptance contribution* from *target compute cost*: compare actual `acceptance A`, `total verifier work`, measured `verifier wall`, `draft wall`, `A + cost` explanatory fits. In stratified analyses, explicitly note conditioning on outcomes can confound causality. A forced D budget changes target semantics; label diagnostic and do not use for quality-matched throughput.
4. Required **overlap table** where acceptance matched but D/verification cost differs; include common-support n, position/context controls, and direct measured stage time. Do not say D causes SD benefit unless an intervention isolates it.

### S3 — affordable deployable check (after S1/S2)
- Freeze global best K from prompt-disjoint calibration. Compare `Static`, `Past-acceptance+Cost`, `+past Exit D`, `+recurrent state` for AR/K action set. Predictor features must be **available before action**, and overhead must be measured. Validate on unseen prompts (≥100 test prompts/model-dataset), real full rollout and output fidelity. An oracle is diagnostic only.
- Main falsifier: after acceptance/cost baseline, depth/state adds no held-out net throughput improvement or optimum K is stable across regimes/tasks. Then **report no dynamic-depth-specific action room**, even if descriptive D-conditioned crossover exists.

### S4 — optional systems translation, only after S1 gate
- Evaluate B∈{1,8,32}, optimized graph backend, fixed vs live target, no-refill cohorts, fair AR/DtV/WFD (and LoopSpec only when ported). Compare full-cohort optimized baseline to avoid `F-035` subbatch confound; no GPU/loss claim from eager B1 extrapolation.

## Exact fields / outputs / stop conditions

Trial key `(source_commit, model_revision, dataset_version, prompt_hash, prompt_template, seed, backend, dtype, KV_policy, target_exit_rule, target_R_or_tau_eps, scheduler, d, K, batch_size, generation_budget, repeat)`.
Outputs `prompt_id, generated_ids_hash, exact_match_to_AR, correctness, measured_wall_ms, prefill_ms, decode_ms, tokens_committed, sample_count, accepted_count, draft_tokens, rejected_suffix_count, mean_target_D, target_D_hist, verifier_D_array, core_rows, draft_ms, verifier_ms, coda_ms, exit_gate_ms, host_sync_ms, max_active_rows, GPU_memory_peak, errors` with **missing** for uninstrumented fields, not fabricated zeros.

Source and derived data under `dynamic_sd/full_rollout_frontier_20261113/results_<date>` (new directory only), a SHA manifest, `RESULTS.md` with fixed/live × K speedup/accuracy and CI plus raw parity failures, `SCOPE.md` (which official protocol reproduced), and a clear `RUN_STATUS` table with `not_run / blocked / passed / failed` per arm. Update wiki only under a separately authorized checkpoint save. **No new GPU run is claimed by this PR.**
