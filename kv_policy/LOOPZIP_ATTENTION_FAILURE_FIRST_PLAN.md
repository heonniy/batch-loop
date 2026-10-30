# LoopZip Failure-first Characterization: Experiment Plan (v4)

Date: 2026-10-31
Branch: research/loopzip-attention-failure-first-20261031
Parent: research/recurrent-sensitivity-main-ablation-v3-20261031
Target hardware: H200 x8, independent BF16 model worker per GPU, B=1 reference
Status: Phase 1/2 reference runner, exact-budget policies, local swaps and CPU tests IMPLEMENTED. Phase 3 decode-loop intervention, actual H200 tests, failure observations, savings/throughput remain NOT DONE.

## 0. Research decision and change from v3

The current work is at hypothesis discovery, before a recurrent KV importance signal has been validated. The previous v3 starts by sampling eight Context tokens and testing Context -> Question recurrent sensitivity. This may miss the practical failure mode or prematurely lock the paper to a particular feature.

V4 prioritizes a falsifiable, architecture-aware diagnosis:

1. Find where strong **attention-only** KV retention fails at the SAME KV budget.
2. Determine whether an alternative same-budget retained KV set improves the same request (local-swap oracle, not a deployable method).
3. Ask whether failures surviving better attention aggregation are explained by attention-output effects, recurrent error persistence/amplification, loop-bank dependence, or dynamic exit changes.
4. Only after the mechanism is observed, search for a cheap clean-prefill feature and build LoopZip.

The old v3 work remains valuable, but is moved to **downstream Phase 3/4 mechanistic ablation**, not the first experiment. Do not delete or overwrite v3 data or branches.

**Fixed user-requested Context KV retention ratios: 10%, 30%, 50%.** No 70% arm in initial screening. If all methods fail at 10%, label it a severe-compression floor, not evidence against attention. If all methods also fail at 50%, check baseline answerability, mask correctness, source identity, and context length before any post-hoc budget change; additional ratios require a separately preregistered run.

## 1. Core question and operational failure definitions

Research question:
When and why does attention-based KV eviction lose answer-critical information in looped transformers, despite retaining highly attended Context tokens, and does recurrence explain residual loss beyond strong attention baselines?

Definitions for question/request u and Context-retention ratio b in {0.10,0.30,0.50}:

- L_full(u) = future teacher-forced gold NLL under untouched Full-depth prefill and Full KV during generation.
- L_p(u,b) = same gold future NLL for policy p retaining exactly the allowed Context KV budget, applied AFTER an untouched Full-depth prefill.
- Damage D_p(u,b) = L_p(u,b) - L_full(u). Signed (negative values are valid).
- Accuracy flip = Full-KV greedy answer correct, policy-p greedy answer incorrect. Do not use this as the sole metric.
- Swap gain G_swap(u,b) = L_attention(u,b) - min_{S in evaluated same-budget one-swap neighbors} L_S(u,b), where the evaluated neighborhood INCLUDES the original retained set. Positive G_swap means a tested equal-budget improvement exists; NOT a global oracle optimum.

Distinguish:
- **Capacity failure:** All admissible policies, including tested local swaps, fail. Could be insufficient KV budget.
- **Attention aggregation failure:** Mean fails but Last/Max/Recent+Attention succeeds at same budget. Not yet proof of non-attention recurrent signal.
- **Attention selection failure:** Several strong attention aggregations fail and a tested same-budget swap improves NLL/accuracy. Shows a *ranking gap*, not that recurrence is its cause.
- **Recurrent-specific candidate:** The residual damage is associated with recurrent error propagation / loop-conditional sensitivity beyond Attention and immediate Attention Output perturbation.
- **Baseline unable to answer:** Full KV is wrong. Keep future NLL in overall distributions; EXCLUDE from the special full-correct-to-wrong flip denominator and report count.

Screening and inference must never use gold answer labels in policy selection. Gold is permitted only to score quality and to construct the OFFLINE swap oracle. Supporting-fact labels are also prohibited from any initial attention policy or choice of screening cases (except analysis stratification).

## 2. Frozen inference and fair budget semantics

MODELS
- Initial smoke: Ouro-2.6B, pinned model revision in kvp/specs.py, Ouro max depth R=4.
- Initial pilot: Ouro-2.6B and Huginn-3.5B (huginn-0125), pinned Huginn R=32. Ouro-1.4B is a later replication, not a blocker for pilot.
- BF16, Full-depth token-major Prompt Prefill, fixed-depth Decode FIRST. No-refill. Baseline and compression branches share the SAME completely processed prompt, target text, tokenizer, exit schedule, and initial random state/seed.
- Dynamic decode with Recorded Replay vs Live Exit comes only after robust fixed-depth failures, because a changed exit decision can confound the first diagnosis. Record deviations of Huginn FP32 cosine evaluator from official native model-dtype exit.
- Run identical question UIDs for all policies and budgets, grouped by model/dataset. Respect per-model token offsets.

DATA
- Controlled Retrieval (existing seeded generator, 16 facts by default): one relevant fact and 15 other facts, controlled target positions. This is NIAH/RULER-inspired, NOT official RULER. Optionally freeze NEW runs with 32 or 64 facts if the 16-fact setting is too easy / produces no discriminative failures.
- HotpotQA-Distractor: first use the existing evidence_excerpt (all supporting-fact sentences + eight sampled distracting sentences); this is a CUSTOM reduced context, NOT official full 10-document HotpotQA. A separate full_context run checks ecological validity only after prompt-length/HBM preflight.
- Both Context and downstream Question are present; the actual gold completion is the **short answer**, as in v3. Test the clean model's ability to follow the response template before interpreting policy-induced damage.
- Fixed 20 questions per model x dataset in the initial pilot (80 model-question units if two models x two datasets); main expansion only if failures are attributable and controls pass.

KV BUDGET
- Context consists of all tokenizer-aligned Context positions (relevant and irrelevant). If N_c is this count, retain exactly K_b = max(1, floor(b * N_c)) Context token positions for b in {0.1,0.3,0.5}. Report N_c, K_b, K_b/N_c, number of retained tokens and mask list for every question; no implicit rounding-dependent mismatches.
- All R Loop banks and all physical attention layers for a retained Context token stay readable; for an evicted Context token, all historical Loop/layer KV attention reads are masked **only for future generated queries**. Do NOT remove source text, re-prefill differently, or partially preserve selected loop banks in this first experiment.
- Instruction, Question and all generated tokens are ALWAYS accessible and are outside the compressed Context count. Prompt delimiters and other non-Context positions remain accessible; report their fixed capacity separately. Do NOT present b as total resident bytes or as an HBM saving. If a policy pins Context sinks or recent Context tokens, they COUNT against K_b. If compulsory pins exceed K_b, the policy fails its feasibility check or reduces the pinned subset deterministically; never allow extra hidden budget.
- All policies operate on the same token-level eligibility and identical allowed Context position count K_b. Head-specific cache retention is outside Phase 1 unless it can be given an equivalent bank/byte budget.
- Store **full-size reference KV** but apply multi-position read masks. It is a correctness and mechanism baseline, NOT real physical memory reclamation, TTFT/TPOT/TPS or throughput evidence.
- Require first generated/gold token prediction to match unmodified prefill (off-by-one invariant). For NLL use future target positions >=2 only. Check deterministic Full vs policy at 100% retention as a separate parity gate, although 100% is NOT a screening budget.

## 3. Phase 0 — Implemented instrumentation and required model checks

**Status updated 2026-10-31:** Phase 1 AND Phase 2 reference code is now implemented on this branch. See [LOOPZIP_FAILURE_RUNBOOK.md](LOOPZIP_FAILURE_RUNBOOK.md) for runnable GPU commands and output interpretation.

- kvp/loopzip_policy.py: mean/last/max/recent-attention and seeded random policy selection over the entire aligned Context; exact floor-rounded K_b at 10/30/50%; local-swap neighborhood with high, low and hash-diverse evicted candidates (at most 6x6, including original).
- kvp/loopzip_cached_eval.py: normal full-depth Prompt computed once, reuses full prompt R-plane KV across policy decodes; future-only simultaneous multi-token masks; exact first-token invariant; independent role_forward parity check.
- scripts/loopzip_failure_screen.py: CPU --prepare/--analyze/--prepare-swap/--analyze-swap, explicit GPU --execute/--execute-swap; source SHA and model revision pinning; worker claims, atomic raw JSON and fail-closed completeness; per-request NLL/optional Greedy Accuracy and request-grouped stats; Phase 2 failure/control cohorts and swap evaluation.
- tests/test_loopzip_policy.py, tests/test_loopzip_failure_runner.py and tests/test_loopzip_cached_eval.py: budget/selection/mask/cached-prefix regression tests. GitHub Actions workflow is installed for CPU-only tests and a fake tiny-backbone PyTorch test.
- **No H200 or full checkpoint-backed execution verified yet.** The GitHub CI fake model cannot replace a real model/tokenizer parity smoke. Do not interpret synthetic test data as empirical evidence of Attention failure.

Still NOT implemented: true physical compact memory eviction, equal-byte HBM measurement, faithful H2O/SnapKV ports, shared-KV-bank-specific policy comparison, decode-loop-selective interventions, and the Phase-3 recurrent mechanism tracing. The current Phase 1/2 experiment measures candidate retained-set quality with FULL-sized KV storage, not saved memory or serving throughput.

**Required H200 GO gates:** 5-request smoke with actual pinned Ouro checkpoint; clean Full replay parity <=1e-4, unique exact selection per budget, first target and first greedy token unchanged under eviction, no evidence/answer leakage, deterministic repeat within tolerance, and no errors or missing cells. Pilot two architecture families only after smoke confirms runtime and memory capacity.

## 4. Phase 1 — Failure Screening with budget sweep 10%, 30%, 50%

All policies see exactly the SAME clean Full-depth prefill and instruction/question-only attention observations.

Baseline policies (proxies explicitly named; do not claim exact official H2O/SnapKV until faithful versions exist):
- Full KV (100% Context retained): reference quality, not part of constrained ranking comparison.
- Mean-Attn: aggregate clean Instruction/Question -> Context attention across all loops/heads/layers, select top K_b.
- Last-Attn: final recurrent-loop attention aggregate only, select top K_b.
- Max-Attn: maximum over loop-specific aggregated attention per Context token, select top K_b.
- Recent+Attn: fixed recent-Context allocation inside K_b, remaining slots filled from Mean-Attn rank. Pre-register e.g. n_recent=min(floor(K_b/4), N_c) and count all sinks/recent slots in K_b.
- Random: seeded, same token budget, at least 3 independent frozen seeds for mean/variance; each seed is a separate policy realization rather than a new request.

Observation protocol: compute source Attention ONLY from completed Context -> downstream Instruction/Question processing; exclude all teacher-forced/generated Answer attention. Normalize sums consistently across numbers of heads, physical layers, loops, and query positions; freeze mean/last/max calculation and tie-breaks BEFORE inspection. Distinguish ordinary Huginn prelude/coda from recurrent core — initial comparison uses aligned CORE Attention only, while any baseline including plain layers must be a separately named condition. If possible, compare later with position-balanced recency and stronger attention policy without relabeling the original baseline.

Screening matrix:
- Smoke: Ouro-2.6B x Controlled Retrieval x 5 requests x [10,30,50]% x 5 policy families (+ randomized seed replicates, Full).
- Pilot: Ouro-2.6B and Huginn-3.5B x Controlled Retrieval and HotpotQA excerpt x 20 requests/cell x [10,30,50]% x baseline policies. Primary fixed-depth.
- This is 2 x 2 x 20 = 80 model-question units; x 3 budgets x 5 policy families = 1200 policy-budget request evaluations, PLUS reference Full and random-seed replications. These are not 1200 independent problems and not a runtime estimate.
- Main expansion: only after smoke integrity and pilot discrimination, 100 requests/model/dataset, with Ouro-1.4B optional replication.

Metrics and plots:
- Signed future teacher-forced DeltaNLL vs Full; median/p90 across REQUESTS and bootstrap 95% request intervals; do not clamp negatives.
- Greedy Exact Match accuracy, full-correct -> policy-wrong flip rate and conditional denominator; show total Full accuracy alongside.
- Fraction of requests with any attention baseline loss; per-budget failure count; source question/prompt length; Context K_b, actual retained ratio, target position (annotation used here only AFTER policies selected).
- Pairwise per-request comparisons Mean vs Last vs Max vs Recent+Attn vs Random at exactly K_b; heavy collapse at 10% must be reported as capacity stress, not silently removed.
- A heatmap of requests x budgets for whether each policy preserves correctness, plus NLL damage distribution; separately show which requests fail ALL attention policies.
- Run a mask parity test before examining the first quality plot. Do NOT imply benchmark quality from excerpt-only HotpotQA.

Candidate cases for investigation:
A) Full correct, Mean wrong, Last/Max/Recent correct: attention aggregation failure.
B) Full correct, all attention policies wrong but one-swap neighbor recovers: promising attention selection failure.
C) All policies including tested swaps wrong: likely capacity-limited or higher-order interactions; report separately.
D) Full wrong: never count as a "correct -> wrong" failure, retain in signed NLL summary.

**Phase-1 GO gate:** at least a meaningful subset of Full-correct requests shows reproducible same-budget variation between attention policies, OR a tested same-budget local swap gives positive utility. If not, investigate 10% saturation, clean QA correctness, 16-fact workload easiness, and policy parity before Phase 3. Do not promise a failure will exist.

## 5. Phase 2 — Local-swap Oracle: can the failure be repaired at exactly K_b?

Apply ONLY to predeclared top 10-20 failed requests selected using Full-correct and high Mean-Attn loss, plus a matched small control cohort of nonfailure requests. Avoid scanning every possible huge swap.

For retained set S_attn of size K_b, choose up to six LOWEST-scoring retained Context positions a in S_attn and up to six restoration candidates e spanning high-scored evicted, low-scored evicted, and deterministic hash-diverse evicted positions. Evaluate swap neighbors:

    S_(a->e) = (S_attn \ {a}) U {e}.

Every candidate retains EXACTLY K_b Context token positions, and same prompt/target baseline. Compute future-NLL improvement relative to original S_attn. Evaluate up to 6 x 6 = 36 swap candidates per failed request/budget (or fewer if limited by K_b or N_c-K_b).

Score all swaps with gold NLL offline only; the oracle's chosen (a,e) is NOT a deployable predictor. The minimum across the TESTED neighborhood gives a **local search upper bound** on how much improvement is available among the tested choices, not a globally optimal K-subset. An unhelpful single swap does NOT prove no better multi-token retained set exists.

For promising swaps, optionally check greedy generation: did corrected retained set recover the final answer and does the same improvement replicate in other requests? Report improvement over the strongest attention baseline as well as Mean. Do not claim attention-limited failure if only the weakest policy could be improved.

Preserve full list of evaluated a/e/source positions, Attention ranks, gold NLL, Exact Match if evaluated, and best tested swap. Check fairness (K_b exact, no supplementary pins).

## 6. Phase 3 — Mechanism analysis on recovered failures (not on arbitrary successes only)

Only after screening and swaps are trustworthy, select 10-20 good failures and equally sized matched successful controls for analysis.

A. Characterize Loop-wise Attention and whether a better loop aggregation already explains source selection.
B. Measure *immediate Attention Output change* from selectively blocking lost/restored KV, not merely Attention weights. Distinguish value-vector effects and softmax renormalization from later recurrence.
C. Measure per-loop hidden-state perturbation, one-shot propagation vs persistent reinjection, and output-logit KL; compare same initial perturbation magnitude to isolate amplification/recovery, with small-denominator safeguards.
D. Use an additional **Decode-loop-specific** intervention: block a source KV at exactly one recurrent Decode loop, then restore its reads for subsequent loops. Compare r=1..R separately (Ouro 1..4, Huginn 1..32). This feature is NOT implemented by the v3 prefill-only one-shot mask and must be added/tested.
E. Once fixed-depth behavior is understood, compare dynamic Recorded Replay and live exit on matched budgets, and track exit flips, signed depth shifts, future NLL, generation length/accuracy. Exit flips are not necessarily harmful.

A proposed propagation statistic for a single-loop perturbation injected at r:

    E_i(k) = || h_i^masked(k) - h^clean(k) ||_2
             / max(|| h^clean(k) ||_2, epsilon)

    G_(i,r) = E_i(R) / max(E_i(r), epsilon).

Here h is the **current generated query** hidden state; r is the injection loop, R the final loop. Interpret cautiously when E_i(r) is near zero. Track signed final NLL alongside G; a large norm change alone is not a quality loss.

**Recurrent-specificity gate:** Recurrent behavior must explain residual failures AFTER matching clean attention, position/age and immediate Attention Output perturbation, rather than only outperforming one weak Attention-Mean aggregation.

## 7. Phase 4 — Optional predictor + LoopZip method (conditional)

Only if Phase 3 finds repeatable mechanism:

- Fit low-cost clean-prefill recurrent features, with no gold answer, future Answer Attention, or counterfactual oracle in inputs.
- Predict future policy damage or rescue candidates on **request-disjoint** held-out questions. Compare strongest Attention aggregation + Age + Size against additions involving source loop-state change, attention-weighted Query recurrence, and phase-specific convergence.
- A masked hidden-state oracle or local-swap gold oracle is an analysis target, not an online feature.
- Demonstrate benefit at equal K_b on truly simultaneous retained-sets, then implement physical compact KV and evaluate actual HBM/TTFT/TPOT/TPS if promising.
- Keep relevance/Supporting Fact labels for a named **evidence ablation**, not main source ranking.

## 8. Runnable commands and frozen protocol

Phase 1 and Phase 2 are now implemented (FIXED-depth Decode, Full-depth Prefill). Use the committed [H200 executable runbook](LOOPZIP_FAILURE_RUNBOOK.md) for the exact CPU tests, 5-request Ouro smoke, two-model/two-workload pilot, Stage-2 --prepare-swap/--execute-swap/--analyze-swap and audit checklist. The command line name is scripts/loopzip_failure_screen.py.

The two phases are deliberately separate. NEVER run the original role_sensitivity_sweep.py in place of the budget-screening runner, and never interpret its eight-token singleton NLL as the joint quality of an entire retention set. Prepare, execute, and analyze Phase 1 before producing the frozen Phase 2 cohort. The same Source SHA and checkpoint revisions guard both phases; changing code or inputs requires a NEW directory.

H200 runtime, observed Attention ranking failures, local-swap gains and real-answer accuracy are **not available** from authoring this commit. The GitHub Actions test covers CPU pure-policy/manifest checks and tiny fake-backbone KV-cache reuse; actual pinned model smoke remains mandatory.

## 9. Expected inspectable outputs and audit gates

- manifest.json: model/dataset/checkpoint revisions, source hashes, prompt templates, question UIDs, policy formulas, RNG seeds, 0.10/0.30/0.50 budget protocol, active code SHA.
- selected_kv.jsonl: per request, budget and policy => exact Context token retained/evicted lists and their computed clean attention scores.
- screening_quality.csv: request, model, dataset, budget, policy, future teacher-forced NLL, Full-NLL, DeltaNLL, greedy Exact Match when enabled, exit mode, clean-correct flip, actual K_b/N_c.
- baseline_qa.csv: every Full-KV request's response and gold answer, to audit format ability and denominator.
- failure_cases.csv: reproducible classified failure types; does NOT cherry-pick away negative/control cases.
- swap_candidates.csv: each evaluated (retained a -> evicted e) local neighborhood, unchanged K_b and signed NLL gain.
- mechanism_trace.jsonl (Phase 3): selected attention/value/output and hidden-state per-loop traces, decode-loop injection index.
- analysis/QUALITY_SUMMARY.md + SUMMARY.json: paired request-bootstrap intervals, policy budget fairness, failure counts, controls, error/missing logs.
- completeness.json: fail closed for any missing policy-budget cells, budget mismatch, baseline-parity failure, corrupt source provenance, model mismatch, or partial claimed success.

Before interpreting results, ensure: (i) policy-selected masks applied **only** in future generated queries; (ii) prefix and first gold prediction identical; (iii) all policies use exactly K_b Context positions, including any pinned sinks; (iv) gold and Supporting Fact tags not exposed to the policy selector; (v) fixed-depth paired decode parity; (vi) damaged requests paired against Full clean correctness; (vii) local-swap Oracle reported explicitly as oracle; (viii) no false speedup/HBM claim from full-size masks.

## 10. Decision criteria and paper contribution candidates

- If Mean-Attn loses but Last/Max improves: pursue a **loop-aware attention aggregation** result, not a strong non-attention claim.
- If all reasonable attention policies lose and equal-budget swaps improve: promising **Attention selection gap**, worth analyzing.
- If propagation/loop-conditional sensitivity explains and predicts this gap better than Attention, recency, and immediate Attention Output: promising **recurrent-specific KV importance** signal for LoopZip.
- If attention-only policies are already near the tested local oracle: report the negative finding rather than fabricating a recurrent failure.
- If no robust failures at 10/30/50 with tiny context: add a separately frozen longer-context or alternative task. Do not redefine budget or filter after seeing losses.
- If FULL model often fails baseline QA: adapt the task prompt in an independently preregistered run, never post-hoc repair heldout cases.

This document changes RESEARCH PRIORITY and planned protocol. It does not claim the requested failure-screening implementation or GPU results already exist.
