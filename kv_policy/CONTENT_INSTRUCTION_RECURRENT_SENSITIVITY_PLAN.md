> **PRIORITY UPDATE (2026-10-31):** This v3 MAIN+ABLATION plan remains an archival/secondary mechanism study. The current *first* experiment is the [LoopZip Attention Failure-first Plan](LOOPZIP_ATTENTION_FAILURE_FIRST_PLAN.md), with frozen **10%, 30%, 50% Context KV retention**; attention budget screening and same-budget local swap precede the v3 recurrent sensitivity study. The Phase-1 budget-screening runner and local-swap implementation **are not in this branch yet**. Do not interpret the v3 eight-token runner as implementing the new failure-first study.

# Final v3 Plan: Context Recurrent Sensitivity (MAIN + Evidence ABLATION)

**Research question:** Does Context-to-Question recurrent sensitivity predict actual future Context-KV eviction damage beyond conventional attention and token age?

**Date:** 2026-10-31
**Repository:** heonniy/batch-loop
**Branch:** research/recurrent-sensitivity-main-ablation-v3-20261031
**Based on:** research/evidence-recurrent-sensitivity-v2-20261030
**Hardware target:** 8 independent H200 GPUs; one BF16 reference worker per GPU; no TP.
**Actual status:** Code, tests, experiment plan committed. GPU evaluation and full pytest suite are NOT verified in this session.
**Core principle:** MAIN must not use answer/evidence labels to select source KV. ABLATION may use labels only for an explicitly labeled *oracle analysis*.

## 0. Change log and why v3 is necessary

v1 (GSM8K / MATH-500) used random problem-token probes but did not reliably distinguish answer evidence from unrelated facts.
v2 introduced Controlled Retrieval + HotpotQA Supporting Facts, but made ground-truth Relevant/Distractor selection the PRIMARY study. This is an oracle-based mechanism comparison, not evidence that recurrence finds important Context tokens by itself.
v3 restores the original research question:

| Protocol | Candidate selection | What it tests | May use ground-truth tags? |
|---|---|---|---|
| **MAIN** | Sample from *all* aligned Context token positions, positional-stratified without oracle information | Attention–Recurrent mismatch; future signed NLL damage; offline predictive value of cheap recurrent features | **NO**, not for sampling, features or labels |
| **ABLATION** | Sample matched-width windows inside supporting/distractor sentences | Why sensitive tokens differ; how relevance relates to recurrence and actual future damage | **YES**, strictly for offline probe selection and interpretation |
| Future actual policy stage | Score/rank *all* Context tokens on a single normal prefill, physically evict at equal KV bytes | Whether a usable method improves NLL/accuracy/HBM/TPOT vs real baselines | **NO** |

MoEZip-like part: Context -> downstream Instruction/Question -> future Answer, using internally observed Instruction-derived signals to prioritize earlier Context. v3 does NOT claim that MoEZip performed Ground-truth Evidence/Distractor selection, and does NOT copy its router-fragility formula. MAIN is closer in candidate eligibility to MoEZip; it is still a diagnostic sample of Context positions, not an exhaustive all-token policy.

IMPORTANT: v3 makes two *separate* run directories and reports. Never pool MAIN and ABLATION intervention rows, fit one predictor across both, or compare raw damages without equalizing masked-token budget.

## 1. Frozen model and dataset protocol

Models: Huginn-3.5B (huginn-0125), Ouro-1.4B, Ouro-2.6B using pinned repository revisions from kvp/specs.py. Full-depth, token-major prompt prefill (all Context / Instruction / Question tokens get R loops). Decode either fixed-depth R or native dynamic exit. For dynamic experiments distinguish recorded replay at matched depth from separately re-evaluated live exit. Huginn's FP32 cosine-exit diagnostic changes exit evaluation arithmetic; it is *not* identical to the native BF16 evaluator.

Datasets:
- **Controlled Retrieval:** NIAH / RULER-inspired synthetic retrieval generated with a fixed seed, default 16 record-to-code facts; exactly one ground-truth target fact, 15 other facts. Target position cycles among beginning / middle / end; question asks for one record's code. Record UID, seed, fact count, target location, prompt hash. This is *not* an official RULER run.
- **HotpotQA-Distractor:** revision-pinned HF hotpotqa/hotpot_qa validation set; source = distractor/validation-00000-of-00001.parquet; pinned revision 1908d6afbbead072334abe2965f91bd2709910ab, SHA256 c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6. Original (title, sent_id) supporting_facts annotations are retained in dataset metadata.
- **HotpotQA excerpt (initial mechanistic study):** all supporting sentences plus 8 deterministic distractor sentences, chosen preferentially from other documents. This is deliberately NOT the official full-document HotpotQA input. Use --hotpot-context-mode full_context in a separate run for complete provided context, preflight length and memory; custom rendering is not directly comparable to the benchmark leaderboard.
- The *same request UIDs* are used for models and fixed/dynamic modes. The exact token positions can differ by model tokenizer/chat template, so do not pretend source-token pairs are identical across model architectures.

Rendered body (Huginn with chat template, Ouro as plain continuation):

    Context:
    [Document A] Relevant sentence, not visibly marked to model.
    [Document B] Distractor sentence, not visibly marked to model.
    ...
    Instruction:
    Use the context to answer the question. Reply with 'Answer: <short answer>'.
    Question:
    Per-example downstream question
    Response:
    ... generated continuation ...

Only earlier Context can causally affect later Instruction/Question in an autoregressive model. We fail closed if max(Context position) >= min(Instruction/Question position). Token roles come from exact tokenizer offsets: relevant_content / irrelevant_content / instruction / question / other; boundary pieces and chat delimiters are other. The labels are metadata, not text shown to the model. A candidate cannot remove a question token.

Gold teacher-forcing continuation is the new short-answer target 'Answer: <gold>\n', independent of older GSM8K/MATH few-shot solutions. **First gold target token** is predicted at the final unmasked prompt position. It is excluded from post-prefill future-NLL utility. This matters for short outputs: require >=2 scored target tokens and inspect non-degenerate future NLL variance.

## 2. MAIN — ground-truth-blind Context sampling

Implementation: kvp/role_main_candidates.py.

- Eligible pool: **all** tokenizer-aligned Context positions across the entire Context, regardless of supporting-fact annotation. No instruction or question positions.
- Default unit: **one KV token** per intervention, closer to MoEZip's per-token anchor selection. Use --main-window-tokens 4 or 8 as independent multi-token sensitivity sweeps (fresh run directory for each width). A wider window is not an apples-to-apples NLL comparison with a single-token mask.
- Default diagnostic sample: 8 candidates per request (--main-candidates 8). Compute all eligible contiguous windows of required width. Divide by **Context position**, not relevance, into 4 strata. Within each stratum, deterministic salted SHA order; cycle strata and select nonoverlapping windows. Fail if requested width/count cannot fit.
- Critical **leakage check**: the selector receives only Context token-position indices, integer seed and width/count. It never receives ground-truth role labels, supporting_facts, gold answer text, baseline attention, masked prefill errors, or future decode losses. Altering answer or swapping supporting/distractor annotations while keeping Context positions unchanged MUST yield exactly the same selected candidates. Add this as a unit test.
- Main source_role is neutral 'context', and all source candidate identifiers are position-based. Ground-truth fields may be present in frozen dataset provenance but are not used to sample or fit features. In MAIN analysis, the evidence-specific attention breakdown is suppressed and Relevant/Distractor strata are not used.
- Sampled positions are a diagnostic approximation of the full pool, **not** a complete Context ranking. A negative result may be due to sparse useful-token sampling; do not silently oversample gold evidence to fix it. Increase n_candidates in a predeclared NEW run if needed.

**Main experiment A (mechanism, during normal prompt prefill):**
1. Clean FULL-depth Context -> Instruction -> Question.
2. Persistent masked-Instruction: hide a sampled Context source token set i only while later Instruction/Question recurrent *core* attention executes.
3. One-shot masked-Question: hide i only at the final Question query's **loop 1**, restore normal KV reads for all later recurrences. This separates initial disturbance propagation from repeated injection.
4. Measure per-loop error and recovery; compare to per-loop clean Instruction/Question -> Context Attention.

Define i = selected KV-token index/window, j = final Question token, r = loop, and h_{j,r} = Question hidden state after loop r:

    E_i(r) = || h_{j,r}^{masked_i} - h_{j,r}^{full} ||_2
             / (||h_{j,r}^{full}||_2 + epsilon),    r = 1,...,R

    Persistence_i = E_i(R) / max(E_i(1), epsilon)

A high E_i(r) is a counterfactual recurrent-state perturbation, **NOT** itself proof of future importance. One-shot recovery and persistent-mask error have different interpretations. E_i(R) requires a masked re-prefill and is thus an expensive diagnostic oracle, not a free serving feature.

**Main experiment B (actual post-prefill KV READ eviction):**
1. Compute a completely untouched FULL-depth prefill.
2. Record the clean fixed/dynamic generation depth trace.
3. Starting only at generated-token query positions t >= prompt length P, mask candidate i from the attention reads of all recurrent and ordinary layers.
4. Score the same gold-token sequence with *identical forced depth* to isolate the KV effect.
5. Measure future generated-token hidden divergence under matched depth (first four processed target queries), future teacher-forced NLL difference, and optional free greedy QA Exact Match.
6. Separate **Live Exit** run allows actual exit depth changes; report exit flip, signed depth shift and quality delta; do not interpret any exit flip as failure by itself.

Let T be the gold continuation token count, ell_t = negative log probability of gold token t, and P the clean prompt length:

    Utility_i := DeltaFutureNLL_i
      = mean_{t=2..T} (ell_t^{evict_i, recorded} - ell_t^{clean, recorded})

The first target token's logit is pre-mask and must be unchanged; delta may be negative (eviction improves NLL). Teacher forcing and free generation are different measurements. No source text deletion occurs; the allocated Full-loop KV buffers remain in HBM.

**Main experiment C (attention-matched mismatch and prediction):**

At the SAME prompt, attempt unlabeled pairs with:
- equal masked-token width,
- clean Attention relative gap <=20%, and
- relative source-age gap <=20%.
Choose at most one closest pair per request to avoid inflating the independent sample count. If pair coverage is poor, report N and inconclusive; never loosen thresholds post-hoc. Analyze whether differences in recurrent error relate to differences in signed future NLL within these pairs.

Fit on a request-disjoint SHA-based 70/30 split per (model, dataset, decode mode), training-only standardization; reported negative heldout R-squared is a real result.

| Predictor | Inputs | Pre-eviction observable? |
|---|---|---|
| Attention+Age+W (control) | Clean Instruction/Question-to-source Attention mean/max, token age, mask width W | Yes |
| Attention+Age+W+Recurrence | Previous controls + source's own clean loop-to-loop change + attention-weighted alignment with the clean final Question's recurrent change | Yes |
| Attention+Age+W+masked-Question oracle | Previous controls + costly masked Question final hidden error E_i(R) | **NO** |

The clean observable recurrent alignment is:

    A_i(r) = sum of clean Question/Instruction-to-window-i attention at loop r
    C_j(r) = || h_{j,r}^{clean} - h_{j,r-1}^{clean} ||_2
             / (||h_{j,r-1}^{clean}||_2 + epsilon),  r >= 2
    Alignment_i = sum_r A_i(r) C_j(r) / max(sum_r A_i(r), epsilon)

Only future DeltaFutureNLL is the supervised label. The gold answer's location is NEVER an input feature. Report heldout R-squared and incremental difference between baseline and online recurrence; request-level bootstrap / multi-seed splits are required before inferential paper claims. A positive oracle-only result is *not* a viable runtime method.

## 3. ABLATION — evidence/distractor causal explanation (offline oracle)

Implementation: kvp/role_data.py choose_evidence_candidates() and the same role_span_oracle engine, executed with --study ablation.

- Choose up to 2 supporting-fact sentences and up to 2 distracting sentences per question from ground-truth annotation. Controlled Retrieval may have only **one** relevant fact; the code chooses available facts rather than inventing a second.
- Use contiguous KV-token windows of identical width **within the same request**, default up to 8-12 tokens (CLI --window-tokens 8 is recommended for pilot). Width may fall below the cap if a selected sentence is shorter; actual n_masked_tokens is recorded. This is NOT entire-sentence eviction.
- In Ablation only, if the gold string appears in a Relevant sentence, a window may be centered on it; otherwise center the sentence. This gold-answer anchoring is deliberately forbidden in MAIN. Record whether it occurred.
- Apply the **same A, B, C instrumentation** as MAIN at identical W for Relevant vs Distractor windows. Collect persistent/one-shot error trajectories, clean attention, future signed NLL, matched-depth generated state error, optional greedy accuracy and dynamic exit.
- Report separate role means/distributions and within-request paired signed difference:

    d_u = mean(DeltaFutureNLL_i | relevant candidates in request u)
          - mean(DeltaFutureNLL_i | distractor candidates in request u)

    d_bar = (1/N) sum_{u=1..N} d_u

Use 1,000 deterministic resamples of complete requests for a descriptive 95% paired interval; samples from one request are correlated and must not be treated as independent. Also compare the closest Relevant vs Distractor windows within each request when attention differs by <=20%; report remaining age mismatch and pair count. A Ground-truth Relevant window is *not guaranteed* to cause more damage; signed negative/null differences are valid and informative.

- Important comparability limit: default MAIN masks W=1, default ABLATION masks W=8. Do NOT compare their raw NLL deltas or inferred sensitivity magnitudes. If such a direct comparison is desired, rerun Ablation with --window-tokens 1 and explicitly label it a matched-budget check. The principal ABLATION comparison is **Relevant vs Distractor within the same request**.
- HotpotQA unannotated sentences can be useful despite not being marked Supporting Fact; the "irrelevant_content" role means "not annotated as supporting" rather than "logically guaranteed to be irrelevant."

## 4. Run schedule (realistic H200 x 8 gates)

| Phase | Study | Data | Scale | Decode | Required pass gate |
| --- | --- | --- | --- | --- | --- |
| 0 | Code/fixture tests | synthetic and tiny Huginn/Ouro models | unit + parity | fixed | exact role attribution, label-blind main selection, grouped masks, replay equality, first-token invariant |
| 1 | MAIN Smoke | Controlled Retrieval | 5/model | fixed | correct unlabeled candidates, prompt and target, finite NLL |
| 2 | ABLATION Smoke | Controlled Retrieval | 5/model | fixed | evidence/distractor matched widths, annotation provenance |
| 3 | MAIN Pilot | Controlled Retrieval + HotpotQA excerpt | 20/model/dataset | fixed | meaningful NLL variance, attention matching coverage, baseline QA |
| 4 | ABLATION Pilot | same | 20/model/dataset | fixed | paired signs, no candidate leakage, optional accuracy |
| 5 | MAIN Main | same | 100/model/dataset | fixed + dynamic | 1200 model-question-mode jobs, heldout predictor and matched pairs |
| 6 | ABLATION Main | same | 100/model/dataset | fixed + dynamic | 1200 model-question-mode jobs, request-paired bootstrap |
| 7 | Controlled variants | HotpotQA full_context, widths 1/4/8, additional controlled lengths | separate frozen runs | fixed first | robustness to context/width, no silent protocol changes |
| 8 | Conditional policy system stage | actual budget-constrained compact KV | separate implementation | fixed/dynamic | equal KV bytes vs H2O/SnapKV/StreamingLLM/random, HBM/TPOT/TPS |

**Totals if BOTH full studies run:** 2 x (3 models x 2 datasets x 100 request UIDs x 2 decode modes) = **2400 model-question-mode jobs**, but only **200 unique questions** if datasets supply 100 each. Each job includes many model re-prefill and decode interventions; do not treat these jobs as individual forward passes. No H200 runtime estimates or results are claimed.

### Commands (run in a fresh checkout)

    git fetch origin
    git checkout research/recurrent-sensitivity-main-ablation-v3-20261031
    cd kv_policy

    # Never skip CPU/fixture gates:
    python -m pytest -q \
      tests/test_role_sensitivity.py tests/test_role_model_parity.py \
      tests/test_role_sweep.py tests/test_role_evidence.py \
      tests/test_role_main_candidates.py

    # MAIN smoke: position-only candidate sampler, W=1
    python scripts/role_sensitivity_sweep.py --prepare \
      --study main --datasets controlled_retrieval \
      --run-dir results/v3_main_smoke5 \
      --n-examples 5 --main-candidates 8 --main-window-tokens 1 \
      --decode-modes fixed --free-generate

    # ABLATION smoke: labeled equal-budget windows, W cap=8
    python scripts/role_sensitivity_sweep.py --prepare \
      --study ablation --datasets controlled_retrieval \
      --run-dir results/v3_ablation_smoke5 \
      --n-examples 5 --n-relevant 1 --n-irrelevant 2 \
      --window-tokens 8 --decode-modes fixed --free-generate

    # Only when tests pass and H200 ownership/model compatibility are confirmed:
    python scripts/role_sensitivity_sweep.py --execute \
      --run-dir results/v3_main_smoke5 --gpu-ids 0
    python scripts/role_sensitivity_sweep.py --analyze \
      --run-dir results/v3_main_smoke5
    python scripts/role_sensitivity_sweep.py --execute \
      --run-dir results/v3_ablation_smoke5 --gpu-ids 0
    python scripts/role_sensitivity_sweep.py --analyze \
      --run-dir results/v3_ablation_smoke5

    # Frozen fixed-depth pilot (20 requests per model/dataset)
    python scripts/role_sensitivity_sweep.py --prepare \
      --study main --run-dir results/v3_main_pilot20 \
      --n-examples 20 --main-candidates 8 \
      --main-window-tokens 1 --decode-modes fixed

    python scripts/role_sensitivity_sweep.py --prepare \
      --study ablation --run-dir results/v3_ablation_pilot20 \
      --n-examples 20 --n-relevant 2 --n-irrelevant 2 \
      --window-tokens 8 --decode-modes fixed --free-generate

    # MAIN 100, only after pilot approval; 1200 jobs with both decode modes.
    python scripts/role_sensitivity_sweep.py --prepare \
      --study main --run-dir results/v3_main100 \
      --n-examples 100 --main-candidates 8 --main-window-tokens 1 \
      --decode-modes fixed,dynamic --max-target-tokens 64
    python scripts/role_sensitivity_sweep.py --execute \
      --run-dir results/v3_main100 --gpu-ids 0,1,2,3,4,5,6,7
    python scripts/role_sensitivity_sweep.py --analyze \
      --run-dir results/v3_main100

    # ABLATION 100; separate dataset manifest, raw outputs, and statistics.
    python scripts/role_sensitivity_sweep.py --prepare \
      --study ablation --run-dir results/v3_ablation100 \
      --n-examples 100 --n-relevant 2 --n-irrelevant 2 \
      --window-tokens 8 --decode-modes fixed,dynamic --max-target-tokens 64
    python scripts/role_sensitivity_sweep.py --execute \
      --run-dir results/v3_ablation100 --gpu-ids 0,1,2,3,4,5,6,7
    python scripts/role_sensitivity_sweep.py --analyze \
      --run-dir results/v3_ablation100

    # Optional separate matched-unit ABLATION (not the original ablation run):
    python scripts/role_sensitivity_sweep.py --prepare \
      --study ablation --run-dir results/v3_ablation_W1 \
      --n-examples 20 --window-tokens 1 --decode-modes fixed

The default model set is all three models; the default datasets are Controlled Retrieval and HotpotQA excerpt. Explicit --max-prompt-tokens (default 1536) FAILS preparation on overlong prompts, never truncates. Hotpot full_context should be a separate run with explicitly raised limit and H200 memory preflight. --free-generate adds large costs; when disabled, never report free-generation answer accuracy as if measured.

### Machine-readable artifacts per run

Frozen: jobs.json, run_manifest.json (including study, source SHA, model revisions, UIDs, token roles/positions, candidate list, prompt hash, original source/checksum).
Per worker: raw/<jobhash>.json and failure traces.
MAIN: analysis/main_context_sensitivity.csv, question_controls.csv, instruction_attention_by_loop.csv, SUMMARY.json, completeness.json.
ABLATION: analysis/ablation_evidence_sensitivity.csv, question_controls.csv, instruction_attention_by_loop.csv, SUMMARY.json, completeness.json.

Complete means all expected jobs and candidates exist, replay max abs per-token NLL error <=1e-4, no error files, correct study/protocol identity. Do not claim successful full runs merely because --prepare wrote files.

## 5. Interpretation and decision rules (pre-register before seeing data)

**Mechanism only:** E_i(r) differs across sources, or loop-recovery patterns differ, but little relationship to future NLL. This is an observation, not an eviction policy.
**MAIN scientific support:** Similar-attention/age Context sources differ in recurrent effect, and the effect predicts actual signed future NLL across request-heldout cohorts.
**MAIN deployability support:** A model using *only a clean prefill's cheap recurrent features* gives reproducibly higher heldout predictive quality than Attention+Age+W. A masked-prefill oracle-only improvement does not establish serving viability.
**ABLATION explanation:** Ground-truth evidence status explains some source heterogeneity, preferably even after controlling width/attention/age; a weak/negative Relevant-Distractor contrast is valid.
**No-go:** Attention/age explains all, counterfactual sensitivity has no future association, main samples miss important sources, dynamic exits saturate, or clean QA accuracy is degenerate. Do not retune the heldout cohort or evidence windows based on measured results.
**Next systems step:** a compact KV policy evaluated at exactly equal HBM bytes and generation quality against fair published baselines; this code does NOT physically evict KV nor measure throughput/latency.

## 6. Scope of implemented change

- kvp/role_main_candidates.py (NEW): no-label all-Context stratified sampling; checked invariant under evidence relabeling.
- kvp/role_span_oracle.py: separate candidate validation and metadata for MAIN and ABLATION; A and B use matched source positions.
- scripts/role_sensitivity_sweep.py: CLI --study; immutable per-study manifest, study-specific CSV/SUMMARY and failure guards.
- tests/test_role_main_candidates.py (NEW): no-evidence/no-answer leakage, candidate disjointness, tiny model main oracle.
- tests/test_role_sweep.py: both study modes, heldout and matched-pair checks.
- kvp/role_data.py, kvp/role_evidence_data.py, kvp/role_generation.py retained: original offset-alignment, data provenance and optional free greedy QA.

All code changes are on the new v3 branch; the v2 branch and FERRY are untouched.

**Validation status:** GitHub contents/commits checked; standalone live H200 model tests, full pytest, and performance measurements NOT observed as part of this commit. Must verify before running MAIN or interpreting the paper.
