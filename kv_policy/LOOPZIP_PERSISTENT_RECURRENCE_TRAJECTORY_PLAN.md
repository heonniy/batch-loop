# LoopZip v6 — Persistent KV Eviction and Recurrent Trajectory Characterization

Date: 2026-10-31
Branch: research/loopzip-persistent-trajectory-characterization-20261031
Parent commit (completed v5 evidence-controlled pilot): c917ba60b9ea3b55da90f544dabe4b3bb9b5e353

**RESULTS (2026-10-31, H200x8):** G0, Stage A, Stage B and Stage C executed. Pilot: results/20261031T070815Z_loopzip_v6_trajectory_pilot (TRAJECTORY_OBSERVATIONS.md, STAGEA_STAGEC_FINDINGS.md, V6_PILOT_ASSESSMENT.md); Stage C controls / Ouro readout / CR auxiliary: results/20261031T084101Z_loopzip_v6_stagec_controls. Stage D not run (see V6_PILOT_ASSESSMENT.md). The plan text below is unchanged.

**STATUS: EXPERIMENT PLAN ONLY — NO NEW EXPERIMENT RUNS, SOURCE IMPLEMENTATION, OR PERFORMANCE RESULTS IN THIS COMMIT.**
This branch preserves previous KV-sharing, failure-first, evidence-coverage, and v5 evidence-lock results unmodified. This plan **replaces v5's immediate policy-design priority**, not the old data. It deliberately postpones token×loop importance policies, refill, live exit, and HBM savings claims until an observable recurrent mechanism is validated.

## 0. Central question and hypothesis discipline

**Research motivation:** A long-decode no-refill system must decide at prompt-prefill completion which historical KV to retain. Attention-only token ranking may be inadequate *if* the past token's recurrent-state behavior predicts how its persistent eviction changes later query-token recurrence, above and beyond attention, recency, content role, and semantic evidence.

**Main question:** For a fixed FULL-loop KV cache, does persistently evicting different past tokens cause distinct, reproducible deviations of **within-query-token recurrent trajectories**, and do those deviations predict incremental degradation over a long, fixed decode continuation?

Research questions:
- **RQ1 / baseline trajectory:** How do recurrent update magnitudes, directions, endpoint distances, and variability differ across *prompt and generated tokens* without eviction?
- **RQ2 / causal eviction:** With a single fixed source token i evicted from all stored loop/layer histories after a clean prefill, how do future query token trajectories change at every recurrent step r?
- **RQ3 / decode accumulation:** For the SAME target token sequence, how much of the late decode deviation comes from persistent source masking itself versus compounding changes to previously generated tokens' historical KV?
- **RQ4 / prefill observability (conditional):** Can clean-prefill features of source token i forecast its future harm more accurately than strong nonrecurrent controls (attention, role, position, and source age)?

**Do not presuppose** there are stable trajectory types, convergence, growing damage, or a loop-specific effect. A persistent mask introduces a fresh perturbation on each loop, so an increasing deviation curve alone **cannot** prove intrinsic recurrent amplification. If evidence/role/position explain the result, explicitly label the outcome general KV importance rather than "recurrent sensitivity."

### Prior work and non-duplication

Existing code:
- kvp/role_sensitivity.py: token-role alignment and source interventions; one-shot/persistent prefill and post-prefill cases; optional per-loop state capture.
- kvp/depth_eviction_probe.py: Ouro/Huginn per-token execution, recurrence states, native/fixed depth, historical KV read semantics.
- kvp/loopzip_cached_eval.py: reuse a clean full-depth prompt, mask selected past positions on future query reads and verify cached reference parity.
- kvp/engine.py: full-loop KV banks and distinct Ouro vs Huginn architectures.
Existing report: results/20261026T080106Z_huginn_ouro_full_last_convert/analysis/e2_localisation.md includes per-layer cross-depth KV substitution and matched noise controls; **this is NOT the same as persistent, whole-token KV removal and exhaustive long-decode trajectories**.
Existing v5 report: results/20261031T034209Z_loopzip_v5_stage1_pilot20/analysis/EVIDENCE_MATCHED_FINDINGS.md indicates evidence-locked success at 30/50%, warning against a naive answer-evidence narrative.

Novel observational value must come from exhaustive source-token-by-query-token-by-loop measurements with controlled decode history, not rediscovering that first-loop perturbation matters or that removing the answer evidence hurts QA.

## 1. Frozen reference semantics (do not cross factors in the first study)

| Dimension | First-stage choice | Reason |
|---|---|---|
| Models | Ouro-2.6B, fixed R=4; Huginn-0125, fixed R=32 | architecture contrast |
| Model and dtype | same pinned model revisions as the v5 pilot; BF16 | comparable to frozen prior data |
| Prompt | full-depth clean prefill of every prompt token | eliminates changed prefill computation |
| Historical KV | FULL-loop, same-depth bank reads | excludes Last/First/Average substitution |
| Main eviction | one source prompt token, ALL stored recurrent loop banks and physical layers; include ordinary prelude/coda history in the all-layer arm | actual whole-token eviction semantics |
| Timing | at the end of prefill, before any new generated-token query | matches proposed offline serving decision |
| Query lifetime | **persistent** mask on every subsequent query token and every eligible attention read; no restoration/refill | no-refill setting |
| Decode | fixed depth and identical input token IDs | isolates recurrent state/read effects from exit and text branching |
| Decode horizon | up to 128 continuation tokens, preselected and frozen | examine short vs long decode changes |
| Storage | full-sized reference cache + read masks | measures causal impact, NOT actual HBM savings |
| Scoring | all future token loops + logit distribution on a shared continuation | avoids privileged evidence selection |

**Eviction granularity:** Evicting prompt position i means that when a query token t after the prompt attends to historical position i, the attention mask forbids it **at every recurrent loop, every recurrent attention layer, and every plain-attention layer that possesses that historical position**. Do not delete only the bank at loop 1 or substitute a Last-loop bank. The still-computed original prompt KV bytes can remain in the reference buffer but are never readable post-prefill.

A companion **recurrent-only mask control** can be run later, holding all plain-layer reads intact, to attribute differences to recurrent versus prelude/coda pathways. This must be explicitly separated from the all-layer primary eviction, not presented as the same policy.

**Time-index alignment:** The next token immediately following prefill is predicted by the final untouched prompt computation, so its probability MUST be identical under all policies. The first observable eviction-conditioned logits arise once a generated token itself has been processed. Freeze and record the tokenizer, EOS/stop protocol, and whether the 128-token reference continuation actually exists; do not silently generate beyond EOS as if normal decoding.

## 2. Workload and deterministic study population

Use two MAIN classes without needing a specific correct-answer evidence span:

**W1. Natural-language continuation**
- short coherent natural-language prompt/context; request or allow a continuation that is not a one-fact lookup;
- explicit source, original item identifier, prompt role labels when available;
- avoid exclusively random strings, which create atypical model states.

**W2. Instruction-conditioned generation**
- Context → Instruction → optional Question → continuation, with natural language operations such as explanation, synthesis, paraphrasing, and summarization;
- preserve exact tokenizer-derived roles: content, instruction, question, other/special;
- roles are diagnostic/stratification features, NOT a proxy for true importance; no answer annotations.

**Follow-up robustness, not a replacement for W1/W2:** GSM8K/MATH continuation and the older Controlled Retrieval corpus. A later 512/1K-token prompt is a separate long-context validation, not conflated with initial exhaustive feasibility.

Freeze a **disjoint** master manifest before looking at outcomes:
- **Stage A:** 40 prompts per (model, workload): 2 × 2 × 40 = 160 model-prompt units; track all prompt tokens and all executed continuation tokens, no source eviction.
- **Stage B/C:** an *independently designated* 4 prompts per (model, workload): 2 × 2 × 4 = 16 exhaustive model-prompt units. Fixed, balanced prompt choices are made **before model outputs are inspected**, stratifying only on workload/template/length, never failure severity or answer content.
- Prompt length in Stage B/C: target 128–192 actual tokenizer tokens. Record effective length and eligible source count separately. If an item falls outside that range, fail the prespecified eligibility check and replace from a fixed preregistered reserve list before intervention results are seen.
- **Holdout:** reserve independently sourced unseen prompts for Stage D; never calibrate a predictive model and report its fit on the same 16 exhausted prompts as an out-of-sample score.
- Continue FULL baseline decoding with greedy to 128 tokens, if permitted by EOS/stop conditions. Do not silently drop early-EOS items. Record completion length, EOS rate, fallback/eligibility policy, and analyzed horizon per request. For fixed-prefix comparisons, use the SAME clean baseline token IDs. Additional full 128-token natural continuations must be selected or obtained via a frozen, documented protocol.

A pilot preflight on real checkpoints determines whether 128-token fixed continuations and the selected prompt lengths are computationally feasible; it may cause a **new, separately versioned protocol**, not a silent edit to the frozen run.

## 3. Stage A — Complete NORMAL recurrent trajectory census (no eviction)

**Question:** Are normal recurrent trajectories homogeneous, or do they vary with token position, role, and generation phase even in one fixed model?

For every token t in the 160-prompt cohort, capture normalized latent states after each *complete recurrent-block pass*:

    h_(t,0), h_(t,1), ... h_(t,R).

- Ouro: after each full UT recurrence and its model-native normalization (matching the existing reference runner).
- Huginn: after each core-block recurrence, before final coda normalization; also retain initial latent and prelude summary. Do not compare raw magnitudes between the architectures.
- In both cases the same state-capture checkpoint must be used in every Full vs Evicted paired execution; avoid reading different layers or placing an extra normalization in the observer.
- Huginn requires the exact deterministic per-(UID, position) latent initialization already used in kvp/engine.py.

Per-token, per-recurrence metrics (do NOT clip non-monotonic cases):

    U_t(r) = ||h_(t,r) - h_(t,r-1)||_2 / (||h_(t,r-1)||_2 + eps)

    F_t(r) = ||h_(t,r) - h_(t,R)||_2 / (||h_(t,R)||_2 + eps)

    C_t(r) = cos(h_(t,r)-h_(t,r-1), h_(t,r+1)-h_(t,r))  [r < R]

Add norm of h, cosine to initial/terminal state, pairwise loop update angle, variance across tokens, and, where model-native and numerically supported, per-loop intermediate readout change. Intermediate head/coda readouts are NOT automatically comparable across Ouro and Huginn and must be labeled native versus diagnostic.

**Selection-free analysis first:** plot empirical distributions and full per-loop curves; only AFTER observing them propose descriptive pattern labels (e.g., rapidly settling, gradual, nonmonotonic). Freeze any subsequent thresholds on a development subset, validate labels on holdout; do not manufacture 3 trajectory classes because a figure would look nice.

Stratify normal trajectories by content/instruction/question/other role, prompt vs generated token, token position percentile, and source age. No claim that these categories are inherently causal. Save all-loop summaries; Huginn cannot be subsampled to only r=1/4/8/16/32 in this baseline census.

**Primary outputs:** normal_trajectory_long.csv or parquet (UID, model, dataset, t, r, h_norm, U, F, C, role, position, token_id), normal_trajectory_distribution.json, NORMAL_TRAJECTORY_OBSERVATIONS.md.

## 4. Stage B — Exhaustive ONE-SOURCE PERSISTENT prompt-KV eviction

**Question:** Does the identity of the evicted source position i alter the entire query-token recurrence trajectory in ways not already explained by role, position, or clean attention?

For EACH of the 16 designated short prompts:
1. Run exactly one untouched FULL prompt prefill, freeze its KV and clean per-loop attention/feature observations.
2. Enumerate **ALL eligible aligned prompt positions** i in Context, Instruction, and Question. Do NOT restrict to top-attention, randomly sample just 6–12, choose answer-bearing tokens, or exclude edge positions without reporting denominators. Special/prompt-format tokens are counted in a separate role; the primary arm may exclude them only under a frozen, disclosed eligibility rule.
3. For every i, perform a new paired **all-loop/all-layer source-i persistent read mask** from that same clean prefill. Keep the selected mask for the complete continuation; do not evict future generated tokens, do not change token IDs.
4. Record every executed query token t and every executed r ∈ [1,R], not just r=1/R.
5. Isolate/restore the reference KV between runs; the source prompt buffers and clean first-token logits must be bitwise/strict-parity equal whenever the reference engine is deterministic.

**Main observation array:** E_i(t,r) with all i × t × r:

    E_i(t,r) =
      ||h_(t,r)^[evict i] - h_(t,r)^[FULL]||_2 / (||h_(t,r)^[FULL]||_2 + eps).

Also store:
- U_t^[evict i](r) and U_t^[FULL](r): has the UPDATE curve changed, not just state position?
- normalized update direction cosine between Full and Evicted at each r;
- final-state error E_i(t,R);
- same-prefix next-token KL(p_FULL,t || p_evict_i,t), in fp32 log-prob accumulation;
- per-token maximum logit difference, top-1 changed indicator, entropy shift, token-NLL on SAME reference continuation where applicable;
- normal-source-prefill features of token i: all-loop U_i(r), F_i(r), attention profile across r and layers/heads if available, role, age, absolute position, position percentile. No gold or future attention is allowed as a prospective predictor.

DO NOT classify a source as important based on a particular ground-truth answer. FULL generated continuation is an observable reference trajectory, not a gold answer selection oracle.

### B1. Two paired history semantics (critical)

**Accumulated / real no-refill effect:**
- In the evicted branch, process the **entire same continuation** sequentially while masking source i at all future query reads.
- Newly generated tokens' K/V are therefore *different from Full* and persist for later generated-token queries.
- No prefill recomputation and no source refill.
- This is the PRIMARY long-decode outcome.

**Clean-history / isolated query-token effect:**
- At each observation token t, begin with FULL reference KV for all prior generated-token positions plus unchanged prompt KV, then apply the SAME source-i mask persistently across every recurrent loop of ONLY the query token t.
- Discard its newly computed KV after the probe; the next observation token begins from a separate copy of the FULL history.
- This is a COUNTERFACTUAL diagnostic, not a realizable continuous eviction policy. It isolates effects inside one token from accumulated changed generated-token history.

The source mask is persistent *within the queried token's all R loops* in BOTH arms. The difference is prior generated-token history, not whether the original source token was restored within that token.

For **every eligible source**, run Accumulated observations at every available t=1..128. Run Clean-history probes at preregistered t∈{1,2,4,8,16,32,64,128} (if the continuation reaches t); selected later cases may receive all-t clean-history follow-up. The first next-token prediction is from the untouched prefill; the earliest mask-affected query state and subsequent next-token prediction must be indexed unambiguously.

### B2. Complexity and execution safeguards

If a short prompt has N eligible sources, B1 requires N counterfactual continuations of up to 128 tokens and up to R recurrence steps, plus 8N isolated-token probes and the baseline. This is EXHAUSTIVE **over eligible source tokens**, NOT every possible multi-token subset (2^N) or every source-bank choice.

Do a 1-GPU REAL model preflight: N=5 source examples, 16 and 128 target token windows, profile per-source wall time, peak GPU bytes, reference KV cloning/reset cost, state collection overhead, and deterministic replay. Record throughput/load profile before launching all 8 H200 workers. Do not promise an unmeasured runtime. If infeasible, predeclare a revised finite cohort or fixed budget, preserve the census principle on the smaller prompt, and keep failures/omissions visible.

Optimize observability without corrupting the trajectory:
- run metrics and vector norms on GPU; transfer compact numeric summaries rather than all h vectors for every i,t,r;
- do not store full-vocabulary logits persistently per source/token: compute KL/top-1 online against a frozen Full reference, release intermediate tensors promptly;
- store selected raw state vectors only for preregistered debug/illustration cases and specify precision/compression;
- use one model per independent H200 worker, one request at a time in first reference pilot; add batching only after exact parity on a paired subset;
- never pool cross-model absolute hidden errors unnormalized.

**Outputs:** source_metadata.csv, exhaustive_eligibility_audit.json, eviction_trajectory_long.parquet, per_source_128step_summary.csv, longdecode_kl_by_position.csv, history_mode_difference.csv, first_token_and_cache_parity.json, execution_profile.json, errors_and_completeness.json.

## 5. Stage C — Describe and test trajectory *mechanisms*, do not over-label

**Primary comparisons**
- For each evicted i, compare paired U^[FULL] vs U^[evicted] and E_i(t,r) at FULL R with identical current token ID and identical deterministic initial latent.
- Within each query token, compare early/middle/late loop errors and update directions; report the signed/nonmonotonic curve instead of reducing to a single scalar.
- Across t, compare E_i(t,R) and logit KL in early (t≤16), medium (17..64), late (65..128) fixed decode bins with missing EOS denominators explicit.
- Compare Accumulated vs Clean at the same t,r. Define e.g.:

    H_i(t,r) = E_i^[Accum](t,r) - E_i^[Clean](t,r).

  H can be negative. It is NOT a pure additive "propagation component" because histories interact nonlinearly; interpret as a paired history-state counterfactual difference.
- Examine whether normal Full U_t(r) becomes small while Evicted U_t(r) also becomes small **but the final states differ**, versus genuine re-alignment of both trajectories; label those descriptive patterns, NOT mathematical convergence or distinct fixed points without additional tests.

**Persistently evicted token vs recurrence**
Even if E_i(t,r) grows with r, EACH loop still cannot attend to source i: later differences reflect BOTH new source information loss and propagation of earlier altered hidden state. To distinguish these, reserve *small, explicitly labeled* one-loop mask or matched-amplitude perturbation controls for representative stable i,t cases after the full persistent census. One-loop controls are mechanism diagnostics only: do not substitute them for PRIMARY persistent eviction.

**Important controls and negative cases**
- Mask none / retain 100% reference; repeat FULL baseline and compare cached prefix versus independent reference.
- Paired same role, close position/age, similar mean/last/max clean attention score; include BOTH high and low-attention source examples, and report how many matched pairs exist.
- Content vs Instruction vs Question masks should be compared within reasonable age/position/attention overlap; if there is no common support, no role-causality claim.
- Different prompts must have independently frozen source IDs, not multiple budget cells masquerading as independent observations.
- Normal-model R=4 vs R=32: compare relative depth r/R and remaining iterations qualitatively, not raw absolute hidden errors or KL as if architecture were controlled.
- Keep old v5 Controlled Retrieval as an **auxiliary evidence-loss control**, never cherry-pick it as the only favorable case.
- Compare against existing earlier single-call cross-depth-bank substitution and random-noise recovery only as DIFFERENT interventions, not replicated values.

**Descriptive examples selection protocol:** Determine largest, smallest, and median paired late-KL effects, plus attention-matched discordant source pairs on a DEVELOPMENT subset. Hold out independent prompts for confirmation; don't choose paper figures from a post-hoc maximum and call it typical.

### Statistical unit and uncertainty

Source tokens from the same prompt and 128 time steps are **NOT independent replicates**. Report per-prompt distributions, source-token within-prompt correlations, and prompt-level bootstrap intervals across independent prompts. Treat the 4-prompt/model/workload exhaustive census as a pilot that characterizes observed mechanisms, not a powered generalization test. Replicate on heldout prompts if a mechanism seems repeatable. Report outliers, nulls, finite denominators, cutoff effects, and model-specific results.

## 6. Stage D — Only if a reproducible effect exists: prefill-only predictability

**Target (diagnostic, not a deployed model):** future harm of removing source i, e.g. mean/late same-prefix logit KL across generated tokens, worst-bin KL, and final-state error. Never use Gold answer to pick i.

**Predictors must be computed BEFORE eviction at clean prefill:**
1. Control features: position/age, role, context length, source type, attention mean/last/max, attention entropy/variance across loops.
2. Added source recurrence features: U_i(r), F_i(r), relative loop-by-loop change concentration, update direction consistency, within-prompt normalized state/trajectory statistics.
3. Optionally combined role×loop attention interactions; keep each ablation explicit and comparable.

Evaluate:
- heldout prompts, not heldout source tokens within the same training prompts;
- rank correlation with late-KL harm, Recall@K of truly high-risk sources under equal token budget, and paired reduction of future harm from protection sets;
- feature ablations (control-only vs recurrence-only vs control+recurrence), bootstrap by prompt;
- no future attention, no replayed decoder state, no dynamic exit signal, no oracle swap or evidence ground truth supplied to the predictor;
- do not convert this into an online eviction policy until predictive lift is nontrivial and replicated.

**Go/no-go**
1. If normal trajectories show negligible or nonrepeatable token-to-token structure, mark RQ1 inconclusive and revise mechanism hypothesis before scaling.
2. If all eviction harm is explained by source age, role, attention, or evidence and recurrent features add no heldout predictive lift, do **NOT** claim recurrence-specific KV importance.
3. If distinct trajectory/historical accumulation patterns persist after matched control and prefill recurrence features predict late-KL beyond attention-only features, then justify a later LoopZip retention method.
4. Even if predictive lift exists, do not claim real HBM/TTFT/TPOT improvements until compact storage and serving-system measurements are implemented.
5. Dynamic exit, Last-loop shared KV, first/last protection, and source-bank-level masking are **separate subsequent studies**, not mixed into the main experiment.

## 7. Engineering implementation/validation order

Reference modules already present (read and reuse, preserve prior working code):
- kvp/role_sensitivity.py (RoleProbeKV: per-loop state, mask all/role; postprefill reads);
- kvp/depth_eviction_probe.py (_ouro_one, _huginn_one and fixed-depth);
- kvp/loopzip_cached_eval.py (cached clean prefill, reset, first-target parity);
- kvp/role_data.py (token-role provenance);
- kvp/engine.py (Full KV and separate plain/recurrent layers).

**Proposed implementation modules (NOT created by this plan commit):**
- kvp/loopzip_trajectory_recorder.py: memory-bounded GPU-side state/KL observations at every t,r;
- kvp/loopzip_persistent_eviction.py: all-layer source mask, clean-history reference restore, independent accumulated rollout;
- scripts/loopzip_trajectory_sweep.py: prepare/execute/analyze, frozen UIDs and source manifests, strict completeness;
- scripts/loopzip_trajectory_analysis.py: paired statistical aggregation and source-feature comparisons;
- tests/test_loopzip_trajectory.py: tiny fake backbone + both checkpoint parity smoke.

**Mandatory validation before H200 sweep:**
1. No-mask and 100%-retention branches reproduce FULL tokenwise logits/NLL, recurrent states within the specified BF16/FP32 tolerances and deterministic token IDs.
2. First generated prediction is unchanged; future target-token index alignment is verified with a tiny causal example.
3. Single source mask blocks exactly source i in every recurrent query loop AND plain layer in the primary arm, but does not affect unmasked prompt calculation/current token's own KV.
4. Switching source i→j restores the prior source histories; no stale cache, overwritten generated buffers or token leakage.
5. Clean-history probe is equivalent to a reference recompute with a pristine FULL generated history and only current query source mask; accumulated rollout retains altered generated-token KV.
6. Huginn initial latent/UID noise are identical between branches; Ouro per-loop normalization and Huginn core/prelude/coda observer checkpoints are audited independently.
7. Every eligible source appears in the census exactly once per prompt; no "interesting" source sampling in Stage B.
8. Fixed source selection happens before inspecting future KL; manifest records source SHA, model revision, tokenizer IDs, prompt data, seeds, all candidate positions, EOS/cap policies, hardware and dtype.
9. Per-prompt and per-worker completion fails closed on missing/duplicate/corrupted/NaN rows, unrecorded early EOS, source/loop mismatches, or stale manifest SHA.
10. New feature implementation must be isolated from existing v4/v5 production results. No modifications to FERRY or unrelated repositories.

### Stage gates / operational order

| Gate | Operation | Scale | Pass condition |
|---|---|---|---|
| G0 | CPU unit tests + real checkpoint smoke, profiling | each model, 1 GPU | reference/first-token/loop+layer mask/cache reset parity |
| A | FULL normal trajectory census | 160 model-prompt units; every available token and all loops | valid observed normal patterns and role distribution |
| B | exhaustive persistent source-token census | 16 designated short prompts, all eligible i; 128-token accumulated history | all i×t×r rows or explicit short-EOS, correct provenance |
| C | clean-history probes and paired mechanistic analysis | same sources; t in 1,2,4,8,16,32,64,128 | paired delta, bootstrapped uncertainty, no hidden cherry picking |
| D | heldout prediction test, conditional | independently sourced unseen prompts | nonrecurrent controls beaten robustly, otherwise NO GO |

**Report both successes and nulls.** A phase with no recurrent-specific predictive signal is still a completed characterization, but it is NOT an excuse to invent a method. No GPU execution or measured performance is represented by this Markdown document.
