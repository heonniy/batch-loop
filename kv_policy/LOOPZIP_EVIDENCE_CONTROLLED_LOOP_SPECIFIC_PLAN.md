# LoopZip v5 — Evidence-Controlled KV Failure and Token-by-Loop Importance

Date: 2026-10-31
Branch: research/loopzip-evidence-coverage-loopwise-plan-20261031
Parent result branch: research/loopzip-attention-failure-first-20261031
Frozen parent commit: ee972218a78b7cb5e3e246fdbf900e4c8dc21090

**STATUS: RESEARCH/EXPERIMENT PLAN ONLY.** This commit does not implement Stage 0–3 analysis runners, does not execute any H200 experiment, and does not claim new NLL/accuracy results. Existing Phase-1/2 raw results and v3 code remain unchanged. GPU experiments must be performed by an execution environment with H200 access.

## 0. Decision, research question, and why we need this follow-up

**Main question:** Does KV importance in looped transformers depend on token–loop interactions **beyond answer-evidence preservation and fixed first/last-loop protection**?

The latest failure-first pilot (see the frozen result directory below) used:
- Ouro-2.6B and Huginn-0125; Controlled Retrieval and a CUSTOM HotpotQA evidence excerpt;
- 20 examples per model/dataset = 80 model-question units;
- 10% / 30% / 50% CONTEXT-TOKEN retention; full-depth prefill and fixed-depth decode;
- Mean/Last/Max/Recent+Attention and three seeded Random policies;
- Phase-1 completeness 80/80, Phase-2 local swaps 43/43 jobs and 100 swap budget targets.

Descriptive results already obtained: Full-KV greedy correct 59/80 model-question units. All four tested attention policies are wrong in 51/33/20 of those 59 units at 10/30/50%; 85/86 tested failure budget-cells have positive *local-swap oracle* future-NLL gain, but zero tested **best-NLL-swap** greedy answer rescues. These counts are repeated budget cells, not independent questions. A single swap minimizing Gold NLL does NOT establish a deployable method, and greedy accuracy was not evaluated for all swap neighbors.

**Critical competing explanations:**
1. **Evidence loss/position bias:** a high-score policy removed a Record ID, answer code, or a necessary Hotpot supporting-fact span. Recent+Attention success on last-position Controlled Retrieval records is consistent with a location-specific effect.
2. **Generic attention-importance mismatch:** the key/value content, normalization, or non-evidence context is important despite low attention, even in non-looped transformers.
3. **Loop-specific interaction:** after controlling for actual evidence positions and strong first/last-loop protection, the importance of historical KV differs by source token *and* query recurrence step.

The program is **gated**: Stage 0 is the mandatory first analysis on existing results. Stage 1 tests interventions controlling evidence at fixed token budget. Stage 2 maps token-by-loop causal effects on a small cohort, and Stage 3 compares competing read/bank policies at matched memory, only if the prior stages leave a meaningful residual mechanism. Do not jump to a new LoopZip feature before diagnosing evidence loss.

### Existing evidence and source locations

- Failure-first results: kv_policy/results/20261030T164908Z_loopzip_failure_pilot20/
  - run_manifest.json, jobs.json, raw/*.json, swap_raw/*.json
  - analysis/screening_quality.csv, failure_cases.csv, swap_candidates.csv, OBSERVATIONS.md
  - completeness.json and swap_completeness.json (both complete).
- Existing prior Full/Last study: kv_policy/HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md
  and results/20261026T080106Z_huginn_ouro_full_last_convert/analysis/
- Existing loop-bank experiment plan: kv_policy/DYNAMIC_LOOP_BANK_IMPORTANCE_PLAN.md

Earlier E2 results showed large Ouro early-loop sensitivity under *terminal-bank substitution* (Ouro-2.6B first-step all-layer substitution final logit KL ~0.611 vs ~0.0012 for step 2/3 in its prior workload). This is NOT the same intervention as token-slot masking and does NOT establish anything about retrieval evidence; treat it as a **strong prior/negative-novelty check**, not as a promised v5 result. Huginn's R=32 and Ouro's R=4 make unnormalized cross-model loop comparisons invalid.

## 1. Definitions that MUST remain separate

Let u denote a model-question unit, C_u the set of tokenizer-aligned Context positions eligible for compression, E_u a subset annotated as answer-supporting evidence, R the FIXED number of recurrent loops, and L_rec the physical recurrent layers.

### Stage 0/1: token-global selection

For retention b in {0.10,0.30,0.50}:

    N_u = |C_u|
    K_u(b) = max(1, floor(b * N_u))
    S_u ⊆ C_u, with |S_u| = K_u(b).

The original failure-first implementation computes S_u ONCE from clean Instruction/Question → Context attention and applies the SAME source-token mask to every recurrent loop and ordinary historical attention layer **only after untouched prefill**, during generated-token queries. Each retained Context token keeps every loop-specific recurrent KV bank. The reference engine masks reads; it does not free GPU HBM.

### Stage 2/3: token-by-query-loop read selection

Let S_{u,r} be the Context positions readable by a future query at recurrent loop r. Under fixed-depth FULL history, the query loop r normally reads the stored source bank KV_i^(r,l) at physical recurrent layer l. Define

    M_{u,i,r} = 1 if i ∈ S_{u,r}, else 0.

This read-mask matrix is NOT equivalent to sharing/averaging physical KV banks. It is also NOT a new per-decode-step online decision; each S_{u,r} may be chosen ONCE at prompt completion, then applied statically at the corresponding query loop during all decoding. No-refill is maintained.

At fixed physical recurrent layers, nominal token-loop slot budget:

    B_u = Σ_{r=1}^R |S_{u,r}|.

- Token-global top-K: S_{u,r} = S_u for all r, B_u = R K_u.
- Loop-specific top-K: each |S_{u,r}| = K_u, B_u = R K_u, but S_{u,r} may differ.
- Phase 3 can evaluate VARIABLE per-loop allocation only under the SAME sum B_u, with deterministic feasible bounds.

The reference backend keeps FULL-sized KV buffers: B_u is a *logical retained recurrent-bank-slot* count, not measured allocated bytes. Before claiming HBM savings or real equal-byte performance, implement/verifiably measure compact storage, count actual per-layer/head K/V bytes, and include unchanged Huginn prelude/coda and prompt/transient peak separately.

**First+Last KV SHARING is a different baseline:** it retains the source token's first and terminal bank and lets query loops reuse those banks instead of denying a token's attention. E.g., in a fixed-R first/last variant, query r=1 reads bank 1, r>1 reads bank R. Every Context token may still be attended at all query loops. Do not equate this with protecting only query loops 1 and R under a MASK. A first+last baseline must use its defined fallback/read mapping and be reported separately, with realized bytes and any unused capacity. It must not receive uncounted extra KV.

## 2. Stage 0 — Evidence coverage audit of ALREADY-COMMITTED Phase 1/2 results (NO GPU)

**Purpose:** Establish the degree to which the apparent attention-only failures are explained by losing answer-bearing spans or by context-position selection, before any extra model execution.

### Inputs and provenance

Read the immutable model/dataset UIDs and exact tokenizer role indices from the EXISTING jobs.json; read each per-policy retained/evicted position set from raw/*.json and attention values from clean prefill; read original and winning swap ids, drop/restore positions, future NLL and greedy from swap_raw/*.json. Cross-check analysis/screening_quality.csv and swap_candidates.csv. Do not replace original source files, reselect requests, or silently change the existing 10/30/50 budgets. Require both existing completeness flags TRUE.

The GitHub connector may not inline the very large jobs.json; run the offline analysis from a checked-out repository/local file bytes, not a truncated API preview.

### Evidence definition and offset caveats

- **Controlled Retrieval:** E_u is the fully aligned token span of the ONE target Record sentence, irrespective of target position front/middle/end. Also compute separate subspans: target Record-ID/key and answer Code/value. The current role_data.context_spans knows the evidence sentence but does NOT already offer authoritative key/code token indices. Implement exact tokenizer-offset-based subspan derivation, verify code/record text against the generated document, and fail closed if boundaries are ambiguous. NEVER infer positions by proportional string length or guess.
- **HotpotQA:** E_u is the union of *annotated* Supporting Fact sentence spans. Per supporting sentence, retain exact title/sentence id, numerator/denominator and full-span preservation. Unannotated sentences are **not guaranteed semantically irrelevant**; label them non-annotated/distractor rather than "provably irrelevant".
- **Alignment:** track tokens crossing label boundaries and excluded as Other. Define evidence coverage on the aligned/eligible E_u, report any missing/boundary token count, and flag cases that cannot be audited reliably. Keep this diagnostic metadata OUT of online attention ranking and main predictors.

### Required metrics for each (u, model, dataset, budget, policy)

    coverage_E(S) = |S ∩ E_u| / |E_u|
    complete_E(S) = 1[E_u ⊆ S]
    evidence_missed = |E_u \ S|

Plus (a) Controlled Retrieval ID coverage and code-value coverage/complete-code flags; (b) Hotpot minimum/mean coverage across annotated supporting sentences and fraction fully preserved; (c) non-annotated Context coverage; (d) target location front/middle/end, position-percentile bins; (e) evidence Attention rank-percentiles (Mean/Last/Max), retention overlaps, lost evidence vs lost non-evidence. Report models separately.

Analyze across 10/30/50 separately:
- Full-KV correct → compressed correct/wrong; NLL deterioration **conditional on coverage bins** and exact full-evidence indicator;
- high-evidence-coverage + wrong cases, low-evidence-coverage + right cases, and explicit evidence-alignment ambiguity;
- within matched (model, dataset, budget, target-position) groups, how strongly coverage and policy choice predict accuracy/NLL;
- the previous Controlled Retrieval last-record Recent+Attention advantage: test whether it is explained by last-position evidence coverage, not by prior loop-selection assumptions;
- Phase-2 best swap's original retained set, dropped/restored tokens: same evidence span? key vs value vs non-annotated? change in E coverage? change in evidence position coverage? Interpret gains with and without evidence coverage change.

Use *request-grouped* bootstrap (at least 1,000 resamples of complete model-question units), confidence intervals or honest descriptive uncertainty. The same question across 3 budgets is repeated, NOT independent. Only report meaningful within-model strata if the denominator supports it.

### Stage 0 outputs and decision

Proposed file outputs (NOT created by this plan commit):
- analysis/evidence_coverage_per_policy.csv
- analysis/evidence_coverage_per_swap.csv
- analysis/evidence_position_strata.csv
- analysis/evidence_audit_integrity.json
- analysis/EVIDENCE_COVERAGE_FINDINGS.md

Primary question: **How many attention failures remain after controlling exact answer-evidence preservation, not merely average token coverage?**

Go to Stage 1 even if only limited high-coverage residual failures remain, but label them exploratory and budget infeasibility honestly. If essentially every failure coincides with missing evidence, do NOT jump directly to a recurrent mechanism claim.

## 3. Stage 1 — Causally evidence-matched KV selection at fixed token budget

**Purpose:** Hold the retained evidence tokens EXACTLY FIXED and compare how non-evidence Context KV choices affect future predictions. Separates "more answer evidence protected" from other history/recency/loop-related effects, without assuming these residual effects are recurrent-specific.

### Two distinct arms

**Arm 1A: observational matched cohort**
Find naturally occurring retained sets from different attention baselines with

    S_A ∩ E_u = S_B ∩ E_u,  |S_A| = |S_B| = K_u.

Only compare when the EXACT retained evidence position sets match; equal percentage coverage alone does not suffice. Predeclare minimal pair coverage, include pair counts and report unmatched requests rather than silently loosening requirements.

**Arm 1B: evidence-locked diagnostic interventions (OFFLINE ORACLE)**
For requests with |E_u| <= K_u, lock E_u into every candidate set, then fill the K_u - |E_u| remaining slots using independently frozen non-evidence policies:
- Mean-Attn, Last-Attn, Max-Attn;
- Recent+Attn with pinned recent positions counting inside the remaining **non-evidence** capacity;
- seeded Random on non-evidence;
- optional position-matched random subset to control recency/coverage differences.

All policies preserve exactly the same E_u positions and the same number K_u of total Context tokens. This is an intentionally answer-annotation-informed *oracle control*, NOT a deployable KV compression method and never a candidate for the main LoopZip serving policy.

For |E_u| > K_u, mark "full-evidence lock infeasible" and preserve the complete feasibility denominator. Optional separate partial-lock arm freezes the SAME predetermined E0 ⊂ E_u across policies and uses K_u-|E0| remaining slots; distinguish it from full-evidence protection and do not select E0 using future Gold NLL.

### Outcome metrics and controls

- Future gold teacher-forced NLL (exclude first gold target predicted from unaffected final prompt query);
- Greedy exact-match and full-correct-to-policy-wrong flips, with the original Full correctness denominator;
- **Gold-independent distribution-shift diagnostic:** teacher-force the SAME clean Full-KV-generated continuation in both FULL and candidate policy, and calculate mean stepwise KL(p_FULL || p_policy) in fp32 probability arithmetic. The actual greedy rollouts have diverging prefixes and must not be compared tokenwise by simple KL.
- Preserved evidence positions, non-evidence position distribution, actual exact K, request-level paired deltas and bootstrap; never use Annotation or Gold for an online predictor.
- No-refill/full-depth prefill, fixed-depth decode; each new intervention starts from the same untouched prefill; source cache is isolated/replay-verified. No "first generated token changed" allowed.

**Interpretation gate:** If quality varies at identical evidence retained sets and K, there is residual Context selection sensitivity. It is NOT by itself evidence for recurrence: compare position/age, attention, value information, and all four attention families first.

### Stage 1 Pilot

Use already frozen model/dataset UIDs whenever feasible; first 5 requests/model for a smoke, then all 20/model/dataset at b={.1,.3,.5}, reporting infeasible full-evidence-lock cells explicitly. If no feasible 10% cells, DO NOT silently change b; an additional 30/50-only mechanism study may be frozen as a separate, clearly labeled analysis. Full HotpotQA Context and additional 32/64-record synthetic workloads are separate robustness runs, not substitutes for the pilot.

## 4. Stage 2 — Token × query-loop importance probe (small mechanistic study)

**Purpose:** Check whether the causal importance of a historical KV depends on the *pair* (source token i, future query loop r), rather than merely being a universal first-loop or terminal-loop effect.

### Intervention must be explicit

Within FULL fixed-depth reference history after completely untouched Prompt Prefill:
- A future generated-token query at loop r and recurrent layer l normally reads Context token i's KV_i^(r,l).
- To measure token/loop importance, MASK exactly that source token i from attention reads at the selected **future-query loop r** in every recurrent physical layer, throughout the fixed target continuation; leave the source token at other query loops, all other Context tokens and all plain prelude/coda historical attention readable.
- Compute future teacher-forced NLL for the SAME target suffix and fixed depth. Confirm that current-token own KV and clean first target prediction are unaffected.
- No physical KV bank deletion or substitution; this is **per-query-loop read removal**. The earlier first/last-sharing experiment substitutes a different bank and is a different intervention.

Denote by L_full the untouched future NLL; for a single source i and query loop r:

    D_{i,r} = L_future(mask source i only at future query loop r) - L_full.

Keep SIGNED D; negative means the intervention improved Gold NLL. Do not equate Σ_r D_{i,r} with whole-token eviction damage; interactions are nonadditive.

### Candidate coverage and sampling

- No answer/evidence labels in MAIN candidate selection. Stratify source Context positions by position quartile, clean attention rank, and near-boundary vs low-attention; freeze 8–12 nonoverlapping candidate tokens per request BEFORE inspecting future NLL.
- Provide a distinct, explicitly labeled ORACLE DIAGNOSTIC stratum sampling supporting-fact and non-annotated Context positions (same candidate count, width, and position controls where possible). No label-dependent oversampling in MAIN.
- Ouro R=4: probe every loop 1,2,3,4.
- Huginn R=32: **exploratory** loops 1,2,4,8,16,24,32; never claim these seven exhaust the 32-step dependence. Interpolate/full sweep only after a positive and reproducible pilot.
- Select 10 balanced requests/model/dataset at most for first pilot, including evidence-matched failure and matched nonfailure controls, not only the largest failures. Each (i,r) probe has a paired unchanged prefix and separate full replay.
- Existing v3 Instruction/Question ONE-SHOT prefill intervention and old terminal-bank substitution cannot be relabeled as this decode-time per-loop source read masking.

### Measures and baselines

- Per (u,i,r): D_{i,r}, observed clean A_i^(r), token position/age, evidence coverage/type (diagnostic only), first/last vs middle-loop label, request and target lengths.
- For each token: signed max/min D, concentration of |D| at first/last vs middle loops, top-impact loop and stability across requests; identify near-zero denominators and no-effect cells.
- Alongside per-loop Attention, assess immediate attention output perturbation and downstream logits for selected candidates (a separate, more expensive trace); do not call a value-induced or softmax effect recurrent-specific.
- Include negative controls: source/token role-preserving position-matched random candidates, loop-label shuffling in analysis (without changing any original scores), no-mask parity and unchanged all-loop read.
- Compare initial-loop dominance against PRIOR E2 evidence, within each model and matched task/semantics. If first/last explain nearly everything, report null result: the adaptive token-by-loop idea adds little beyond simple protection.

**Novelty gate:** a meaningful residual token-by-loop interaction should survive comparison against an explanation based only on first/last loop and per-loop Attention, evidence coverage and position. Do NOT claim novelty from "Ouro loop 1 is sensitive" alone.

## 5. Stage 3 — Equal-budget Token-global vs Loop-specific vs First+Last

This stage is CONDITIONAL on Stage 1 residual effects and Stage 2 token-by-loop heterogeneity. It is not justified merely by Phase-1 attention errors.

### Policies to evaluate

A. **Token-global Mean/Last Attention:** same set S across all query loops, retaining all R historical recurrent banks for those tokens. Exactly K tokens * R banks per recurrent layer.

B. **Loop-specific Attention:** static once-per-prefill selection S_r based on clean loop-r attention, each |S_r|=K, exactly R K source-token×loop read slots. The set of retained tokens may differ across loops. This is NOT per-decode-step re-ranking.

C. **Loop-specific clean-feature candidate (conditional):** same budget, features derived ONLY from one clean prefill, no answer annotation, future generated-token attention, oracle D_{i,r}, or swapped Gold NLL. Fit/choose using disjoint development questions; report on held-out questions.

D. **Fixed First/Last Priority MASK baseline:** fixed architecture-wide preference for loop 1 and terminal loop with a deterministic allocation of the SAME B=R K slots across eligible token×loop positions, then deterministic middle-loop fill if first/last cannot absorb the full budget. This is a READ-MASK baseline, not a replacement-bank sharing policy; report detailed allocation and degenerate capacity cases.

E. **First+Last KV SHARING baseline (strong prior method):** keep source bank KV^(1) and terminal KV^(R), and define fallback read mapping for intermediate query loops exactly. This is cross-bank substitution, not a token-specific mask. Since it normally retains 2 banks per source token, its fraction of FULL R-bank storage is 2/R, not uniformly "30%" across Ouro R4 and Huginn R32. Compare at verified actual retained K/V bytes; if it underuses/overuses a given budget, report the gap and use a predeclared added-bank or restricted-token variant rather than claiming an invalid equal-byte comparison. Cite earlier FULL/LAST/FirstLast work. Ordinary layers remain unchanged in first isolated recurrent study.

F. **Loop-shuffled control:** take EXACT per-loop sets selected by B (or C) and permute recurrence labels of the sets. This preserves each token's number of readable loops AND each loop's K when all |S_r|=K; only token–loop pairing changes. Use several frozen non-identity loop permutations. If observed benefit vanishes after shuffling, that is evidence pairing matters beyond generic token diversity.

G. **Random token/loop slots, same B** and unchanged Full KV upper reference.

### Strict fairness and effect metrics

- Require same Context token/loop eligibility, fixed Prompt Prefill, fixed Decode and fixed target text; compare per-request and by budget 10/30/50 without mixing total-Context-token fraction with fraction of historical recurrent bank slots.
- For A/B/D/F/G: exact nominal B=R K masked read slots and comparable recurrent layers; report selected count per loop, unique Context tokens reached (union of S_r), duplicate coverage across loops, preserved evidence positions and evidence loop-read coverage.
- For E: report actual bank-sharing storage **AND** read fallback semantics, actual K/V bytes, additional bookkeeping, and any budget slack. A fairness claim requires an equivalent actual-byte implementation, not simply a per-loop mask over FULL-sized buffers.
- Report signed future Gold NLL, greedy QA accuracy, Gold-independent fixed-prefix KL, paired request bootstrap and failure-control strata. No observed quality improvement is a valid negative result.
- Validate full/no-mask parity, all-loop agreement with v4 Global Selection, never mask own/current generated tokens, and deterministic selection/return to clean branches.
- Avoid pretending the FULL-size reference mask measured HBM or serving TPOT/TPS. Physical compact KV, true peak memory and throughput tests are separate implementation gates.

**GO only if:** Loop-specific selection has a reproducible held-out advantage over strong Global Attention, Fixed First/Last PRIORITY, First+Last SHARING under valid byte accounting, and Loop-shuffled controls at matched budget. If simpler first/last protection or loop Attention suffices, report that as the result and STOP proposing a complex LoopZip predictor.

## 6. Models, outputs, audit checks, and work schedule

### Frozen target platform/workloads

- H200 x8, BF16, one independent B=1 reference worker per GPU, no tensor parallelism in first studies.
- Ouro-2.6B (R4) and Huginn-0125 (R32); pinned revisions inherited from Phase 1/2.
- Controlled Retrieval: 16 facts with target positions cycled among front/middle/end. HotpotQA: labeled multi-hop supporting facts with the CUSTOM evidence_excerpt first; separate full_context validation. No accidental answer annotation in actual policy.
- Initial No-refill, Full-depth untouched Prompt Prefill, fixed-depth Decode. Dynamic-depth/Recorded Replay/Live Exit is a separate follow-up after the fixed-depth mechanism is reproducible.
- Main metrics are paired future NLL and greedy accuracy; per-loop oracle D and KL are diagnostics, not deployment features.

### Milestones

| Gate | Action | GPU | Suggested scope | Must pass before promotion |
| --- | --- | --- | --- | --- |
| 0 | Audit existing evidence retention + swaps, report failures and controls | NO | All 80 prior model-question units × three budgets | Exact offset mapping; no missing files; no label leakage into Phase 1 policy; audit counts vs frozen results |
| 1a | Smoke evidence-locked selection | 1 | 5 questions/model, 10/30/50 with infeasible arm counted | No-mask/cached parity, exact shared E set and K, first token invariant |
| 1b | Evidence-matched pilot | Up to 8 | 20/model/dataset, fixed depth | Report feasible denominator, paired residual damage, controls and uncertainty |
| 2a | One source × one query-loop reference validation | 1 | Tiny model and real-checkpoint smoke | All non-target loops untouched, full mask/injection semantics, fixed-prefix parity |
| 2b | Token×Loop sensitivity | Up to 8 | 10 balanced requests/model/dataset; 8–12 token samples each | Separation from universal Loop1/Last effect, proper negative controls |
| 3 | Equal-budget policy comparison | Up to 8 | Frozen pilot and then independent heldout requests | Valid byte or clearly labeled logical-slot matching; strong First+Last; loop-shuffled control |

The H200 model run is authorized ONLY after CPU synthetic fixtures and one-checkpoint parity tests pass. No runtime estimate, automatically scheduled job, or new GPU result is implied by committing this document.

### Required artifacts

- Stage 0: evidence_coverage_per_policy.csv; evidence_coverage_per_swap.csv; evidence_position_strata.csv; evidence_audit_integrity.json; EVIDENCE_COVERAGE_FINDINGS.md.
- Stage 1: evidence_matched_policy_quality.csv; evidence_lock_feasibility.csv; fixed_prefix_kl.csv; paired_effects.json.
- Stage 2: token_query_loop_oracle.csv; loop_sensitivity_profiles.csv; loop_coverage_audit.json; provenance trace/paired intervention checks.
- Stage 3: policy_quality_equal_budget.csv; retained_bank_budget.json; loop_shuffle_controls.csv; baseline_recovery.csv; per-request heldout decision report.
- For EACH stage: source SHA, frozen model/dataset revision and UID, exact tokenizer offsets, policy rules and seed, original per-request rows, error receipts, completeness fail-closed flag, request/bootstrap uncertainty, and honest null/negative findings.

### Suggested (NON-RUNNABLE) implementation modules

The following are requirements for a future implementation, NOT existing scripts in this branch:
- kvp/loopzip_evidence_audit.py + scripts/loopzip_evidence_audit.py (Stage 0 CPU-only);
- kvp/loopzip_evidence_lock.py + scripts/loopzip_evidence_matched.py (Stage 1);
- kvp/loopzip_loop_read_mask.py + scripts/loopzip_token_loop_probe.py (Stage 2);
- kvp/loopzip_bank_budget.py + scripts/loopzip_loop_policy_compare.py (Stage 3);
- dedicated tests for exact evidence/key/value offsets; equal evidence sets and K; infeasible budgets; first-target invariance; ordinary layer preservation; exact query-loop read mask; firstlast bank substitution semantics; valid byte accounting; randomized loop-set permutation and non-leaking heldout predictors.

## 7. Go/No-Go interpretation for a paper

**A. Evidence-only explanation:** If the apparent attention failures almost entirely track missing evidence, and once E is kept identical other policies have no repeatable residual difference, report this as a limitation of the current screening workload. Do NOT claim a recurrence-specific weakness.

**B. General non-evidence dependence:** If equal E and K still produce failure variation but per-loop sensitivity is captured by strong Attention/Value controls or fixed first/last protection, this motivates a better general Context policy, not necessarily LoopZip.

**C. Actual token–loop interaction:** If equal E, age/position and attention-output perturbation fail to explain the observed response, Token×Loop damage differs beyond universal Loop1/Last concentration, and matched-bank Loop-specific policy improves heldout quality while loop shuffling removes that gain, LoopZip has a stronger architecture-specific claim.

**D. Null but valuable:** If first/last sharing or loop-wise Attention is sufficient, the correct research conclusion is that complex token×loop eviction is not justified under these checkpoints/tasks. Do not hide negative cases.

**Not established by this plan:** that recurrence is necessarily the cause of current Phase-1 failures; that First+Last is inferior; that a learned online predictor exists; that physical KV storage/HBM or latency has been reduced.
