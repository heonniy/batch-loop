# Evidence-conditioned Recurrent Sensitivity: Experiment v2

Date: 2026-10-30
Repository: heonniy/batch-loop
Branch: research/evidence-recurrent-sensitivity-v2-20261030
Parent: research/content-instruction-recurrent-sensitivity-h200x8-20261030
Hardware target: H200 x 8, one BF16 model worker/GPU, no TP
Status: Code and tests committed. No local/server tests or H200 evaluations observed passing in this session.

## 0. Why the original experiment is superseded

| Design choice | Original v1 | Corrected v2 |
| --- | --- | --- |
| Primary tasks | GSM8K and MATH-500 | Controlled Retrieval and HotpotQA-Distractor |
| Context content | Math problem and static background | Ground-truth supporting facts plus irrelevant facts |
| Candidate | Random isolated Content token | Tokenizer-aligned, matched-width window within a labeled sentence |
| Roles | Content, Instruction, Background | Relevant Context, Irrelevant Context, Instruction, Question, Generated |
| Target | Old math reasoning/rationale | New task-specific short answer: Answer: gold_answer |
| Observables | Attention, prefill counterfactual, future teacher-forced NLL | Same plus generated-token recurrent error, paired controls, optional greedy Exact Match |
| Analysis | Pooled score correlations | Separate role strata, attention-matched pairs, request-paired bootstrap and 70/30 heldout predictor |
| Systems claim | FULL-sized reference KV | Still FULL-sized masked reference KV: NO physical saving/latency claim |

The original study was not wrong to use a Content-before-Instruction prompt for a basic mechanism check, but it did not provide the annotated relevant/irrelevant context needed for this research question. Retain the original branch; do not retroactively label the GSM8K/MATH-500 study as the main result.

Research question: Does the change induced in later Question/Instruction recurrent computation by an earlier Context KV window predict future post-prefill eviction damage beyond attention and source age? This MUST be tested, not assumed.

## 1. Dataset protocols and provenance

**Controlled Retrieval (primary mechanism control):** deterministic NIAH/RULER-inspired generation in kvp/role_evidence_data.py. Default 16 synthetic record-to-code facts, exactly one answer-bearing fact, 15 irrelevant facts, and target positions cycling among front/middle/end. Record generated UIDs, fact count, target position, seed and SHA256. This is NOT an official RULER benchmark or published NIAH score.

**HotpotQA-Distractor (multi-hop generalization):** pinned HF repo hotpotqa/hotpot_qa, revision 1908d6afbbead072334abe2965f91bd2709910ab, validation parquet distractor/validation-00000-of-00001.parquet. Expected SHA256 c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6. Read original supporting_facts (title, sent_id) and context (title, sentences), requiring at least two distinct supporting document titles and at least one irrelevant sentence. Store number filtered and exact sample UIDs.

HotpotQA has TWO explicit context modes. The default evidence_excerpt retains every supporting-fact sentence plus eight deterministic distracting sentences, prioritized from unrelated documents; this is a custom *mechanism* workload, NOT standard full-document HotpotQA. The separate full_context mode retains every supplied sentence, with its title, but still uses our custom prompt. Never mix the two settings in one run or call either an official leaderboard score.

Main sample: 100 different questions per dataset, same UIDs across Huginn-3.5B, Ouro-1.4B and Ouro-2.6B. Fixed and native dynamic decode yield 3 x 2 x 100 x 2 = 1,200 model-question-mode jobs. The experiments within each job make multiple full-prefill passes; a job is NOT a single inference.

## 2. Prompt contract and exact roles

Content precedes Instruction, and actual per-example Question follows the generic Instruction. Conceptual form:

    Context:
    [Doc A] Supporting evidence 1
    [Doc B] Supporting evidence 2
    [Doc C] Unrelated distracting fact
    Instruction:
    Use the context to answer. Reply with 'Answer: <short answer>'.
    Question:
    What is the requested factual answer?
    Response:

Huginn renders the body through its chat template; Ouro uses plain text. Tokenizer offset maps (not regex heuristics) assign the exact roles relevant_content, irrelevant_content, instruction, question, other. The generated sequence is a separate decode region. Cross-boundary token pieces are Other and are not candidates. Fail if a role cannot be aligned or if a prompt exceeds the explicit token limit; no last-96-token truncation.

The gold target is the new answer continuation 'Answer: <gold>\n', not the earlier mathematical few-shot solution. The leading "Answer:" ensures that answer-bearing tokens normally occur after the first generated token. The first target token is predicted at the last (clean) prompt query, before any post-prefill mask, and must NOT be credited to the eviction effect.

**Ground-truth probe windows:** choose up to two relevant and two irrelevant *sentences*. Hide equal-width contiguous tokenizer-token windows within each chosen sentence. Default W<=12, reduced to the minimum available width of any chosen sentence. Each chosen candidate for the same question hides exactly W KV positions. For relevant evidence, a text match of the gold answer can anchor the window (otherwise use center). This is an annotation-based *offline diagnostic*, not a practical online policy. Log sentence ID, source title/sentence, all masked token positions, answer anchor use, full sentence length, age, attention and model revision. Do NOT describe this as full-sentence eviction; only a matched window is evicted.

## 3. Separate causal experiments

Let P denote prompt token count; i a selected set of W Content token positions, j the last later Question token, r recurrent iteration (1 through R), and h(j,r) the hidden state for Question token j after loop r.

### A — Mechanism during prefill

Run all Content tokens at FULL recurrent depth with no source modification. When computing later Instruction/Question queries, block those queries' recurrent-core attention reads to Content window i:

- Clean FULL prefill: h_full(j,r)
- Persistent A: mask i for every later Instruction and Question query and every loop
- One-shot A: mask i only for final Question query at loop 1; restore Content KV reads in later loops

Define normalized hidden divergence:

    E_i(r) = || h_mask_i(j,r) - h_full(j,r) ||_2
             / max( ||h_full(j,r)||_2, epsilon )

and persistence ratio E_i(R) / max(E_i(1), epsilon). Persistent A includes repeated perturbation; one-shot A isolates propagation/recovery. An arbitrary state difference is NOT automatically helpful, harmful, or specific to looped models. Measure clean per-loop Question/Instruction→Context attention as a baseline signal.

### B — Actual future KV-read eviction AFTER clean full prefill

Compute the entire original prompt normally, including all Instruction and Question recurrent states. Only when later *generated-token* queries are processed (position >= P), hide the same Context KV window from all recurrent and ordinary attention layers. No previously computed prefill state is altered.

Use recorded replay to hold every generated token at the identical recurrent depth for paired NLL comparisons:

    Delta_future_NLL(i)
      = mean NLL(gold positions 2..T | clean full prefill, evict i)
      - mean NLL(gold positions 2..T | clean full prefill, no eviction)

T is capped at 64 gold tokens. The signed delta can be negative. Also measure matched-depth generated-token hidden-state divergence (up to first four processed generated queries). When using dynamic exit, compare frozen replay and separately *live* exit decisions, because live exit may change depth and NLL. An exit flip is not necessarily a quality loss.

Optional --free-generate (implemented): greedy decode from a clean prompt with no gold continuation fed into the model, comparing pre/post-eviction Exact Match and clean-correct→evicted-incorrect rate. The last prompt token predicts the same first generated token under both branches; check parity. Exact-match extraction is a short-answer scorer and must be validated on real samples. Without this option, do NOT claim answer accuracy, only teacher-forced NLL.

All these operations use an **allocated FULL-sized multi-loop KV reference plus read masks**. Physical HBM reduction and TPS/TPOT are NOT measured.

## 4. Importance predictor and statistical controls

The future eviction NLL delta from B is the supervised ground truth. Compare models fit only on 70% of request UIDs (fixed SHA-order split; 30% request-heldout), with train-only feature standardization.

| Predictor | Features | Available online? |
| --- | --- | --- |
| Attention+Age+W | Clean Instruction/Question→Content mean/max attention, source age, masked window width | Yes |
| + clean recurrent | Above plus Context source's own clean loop-state change and attention-weighted alignment with the Question's clean loop-state change | Yes |
| + masked-Question oracle | Baseline plus A's expensive counterfactual final state divergence | NO |

Define source-window attention mass A_i(r) and clean Question change C_j(r):

    C_j(r) = ||h_full(j,r) - h_full(j,r-1)||_2
             / max(||h_full(j,r-1)||_2, epsilon), r>=2
    Alignment_i = sum_r A_i(r)*C_j(r) / max(sum_r A_i(r), epsilon)

This correlation is a candidate ONLINE feature, not proven instability. Counterfactual E_i(R) is deliberately labeled an expensive oracle. Do not pass ground-truth relevance labels, gold answers, future generated attention, or future NLL into predictors.

Report separate Relevant/Irrelevant signed future-NLL mean, generated-state error, Spearman source attention and recurrence associations, plus paired within-request Relevant-minus-Irrelevant damage. Bootstrap this paired difference by request (1,000 resamples). Separately match the closest Relevant/Irrelevant source windows inside the *same request* when relative attention gap <=20%, preregistered; report the number of matched requests and remaining age difference. Matching attention does not exactly match age or semantics.

Output request-heldout R² for Attention+Age+W vs the augmented recurrence predictor and oracle control, including negative R². Before claiming a paper method, repeat split seeds and compute request-bootstrap intervals on the incremental benefit; current one-split R² is exploratory.

**Go/no-go:**
1. Does A show content-specific (not merely generic) hidden divergence at comparable attention?
2. Does A predict signed B future NLL damage after role/age/width controls?
3. Do affordable *clean prefill* recurrent features improve heldout prediction beyond attention+age+W?
4. Is clean greedy accuracy nondegenerate? A model that cannot answer the original workload cannot support strong claims about protecting answer accuracy.

If (1) only, this is mechanism characterization. If (1) and (2), this is a future utility association, but still an oracle. Only an affordable signal plus matched-budget eviction wins can motivate a systems method. Negative/zero results should be reported.

## 5. Staged runs and commands

Stage | Data | Size/model | Decode | Gate
--- | --- | --- | --- | ---
CPU tests | synthetic fixture and tiny random models | tests | fixed/dynamic | offset labels, annotations, grouped masking, first-token invariant, reference parity
Smoke | Controlled Retrieval | 5 | fixed | inspect five prompt roles, targets, masked positions, greedy outputs
Pilot | Controlled Retrieval + HotpotQA excerpt | 20/dataset | fixed first | nontrivial future NLL, relevance discrimination, measured runtime/HBM
Main | Both datasets | 100/dataset | fixed+dynamic | 1,200 jobs, matched comparisons, heldout prediction
Separate validation | HotpotQA full_context | 5 then 20+ | fixed initially | longer prompts, verify no truncation
Generalization | GSM8K/MATH-500 | new adapter later | conditional | no reuse of old prompt/target as equivalent benchmark

Revisions, all question UIDs, exact prompt token IDs/roles, gold answers, evidence sentence IDs, source hashes and Git HEAD are frozen in jobs.json and run_manifest.json. Every frozen setting requires a fresh run directory; --prepare performs CPU/data/tokenizer work but no GPU model execution. --execute alone launches model workers. Before running eight GPUs, verify server ownership and model revision / input lengths. Fail-closed on any overlength prompt or parity mismatch.

Commands:

    git fetch origin
    git checkout research/evidence-recurrent-sensitivity-v2-20261030
    cd kv_policy

    python -m pytest -q tests/test_role_sensitivity.py \
      tests/test_role_model_parity.py tests/test_role_sweep.py \
      tests/test_role_evidence.py

    # Smoke (5 questions per model, fixed)
    python scripts/role_sensitivity_sweep.py --prepare \
      --run-dir results/evidence_v2_smoke5 \
      --datasets controlled_retrieval --n-examples 5 \
      --n-relevant 1 --n-irrelevant 1 --window-tokens 8 \
      --decode-modes fixed --free-generate

    python scripts/role_sensitivity_sweep.py --execute \
      --run-dir results/evidence_v2_smoke5 --gpu-ids 0
    python scripts/role_sensitivity_sweep.py --analyze \
      --run-dir results/evidence_v2_smoke5

    # Pilot 20 per model/dataset, fixed
    python scripts/role_sensitivity_sweep.py --prepare \
      --run-dir results/evidence_v2_pilot20_fixed \
      --n-examples 20 --n-relevant 2 --n-irrelevant 2 \
      --window-tokens 12 --decode-modes fixed --free-generate

    # Main after pilot gates; model x dataset x question x mode = 1200 jobs
    python scripts/role_sensitivity_sweep.py --prepare \
      --run-dir results/evidence_v2_main100 --n-examples 100 \
      --n-relevant 2 --n-irrelevant 2 --window-tokens 12 \
      --decode-modes fixed,dynamic --free-generate
    python scripts/role_sensitivity_sweep.py --execute \
      --run-dir results/evidence_v2_main100 --gpu-ids 0,1,2,3,4,5,6,7
    python scripts/role_sensitivity_sweep.py --analyze \
      --run-dir results/evidence_v2_main100

    # Separate HotpotQA complete-context preflight (may exceed 4096)
    python scripts/role_sensitivity_sweep.py --prepare \
      --run-dir results/evidence_v2_hotpot_full10 \
      --datasets hotpotqa_distractor \
      --hotpot-context-mode full_context --max-prompt-tokens 4096 \
      --n-examples 5 --decode-modes fixed

Token-major FULL-depth prefill re-executes for every candidate and is EXPENSIVE. --free-generate adds additional full-prefill passes. Default target horizon is up to 64 gold tokens; default greedy answer generation horizon is 32 tokens. Consider an explicitly separate pilot without free generation for profiling but NEVER claim Accuracy from it. Huginn FP32 cosine threshold evaluation remains a documented modified evaluator relative to its BF16-native threshold behavior.

## 6. Verification checklist / required outputs

- Verify HotpotQA file SHA, original support sentence IDs, valid two-document support, and sample UID provenance. Audit 5 examples before expanding.
- Verify identical selected W token positions and masking semantics in A and B, true role and causal ordering, prompt/target token length and no truncation.
- Verify first gold-token NLL and first greedy generated token are unchanged by B.
- Verify clean baseline and recorded replay per-token NLL parity <=1e-4, full-depth prefill R for all prompt tokens, dynamic depth only in generated positions.
- Report evidence_sensitivity.csv, question_controls.csv, instruction_attention_by_loop.csv, analysis/SUMMARY.json, completeness.json. Label any failed/incomplete jobs; no silent omission.
- Report role-specific NLL, generation Exact Match and clean-correct loss (if evaluated), matched-attention pair count, request-bootstrap CI, heldout R² and its uncertainty.
- Do not claim physical KV memory savings, latency, TPS or TPOT. A later multi-token byte-budget policy stage must compare actual compressed eviction with fair H2O/SnapKV/StreamingLLM/random baselines.

Changed files:
- kvp/role_evidence_data.py: seeded retrieval, pinned HotpotQA and sentence provenance.
- kvp/role_data.py: tokenizer-offset evidence/question roles and matched source windows.
- kvp/role_sensitivity.py: grouped key masks and per-loop relevant/irrelevant attention.
- kvp/role_span_oracle.py: paired causal A/B, generated states, optional free accuracy.
- kvp/role_generation.py: greedy answer generation and Exact Match.
- scripts/role_sensitivity_sweep.py: v2 preparation, worker, structured analysis.
- tests/test_role_sweep.py, tests/test_role_evidence.py: regression gates.

Tests and H200 experiments are NOT claimed to have passed as part of this GitHub-only update.
