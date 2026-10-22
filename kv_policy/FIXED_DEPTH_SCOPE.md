# Fixed-depth KV policy reproduction — narrowed contract (2026-10-22)

This replaces the broader initial proposal for Phase 1–3. Goal: determine whether FULL has higher predictive quality than LAST at the checkpoint's native fixed loop depth, with AVERAGE as secondary comparator. No dynamic exit, no Ragged KV, no Ours, no depth sweep, no full test-set expansion in this phase.

## Models and native fixed loop depth
- ByteDance/Ouro-1.4B: total_ut_steps=4, early exit OFF.
- ByteDance/Ouro-2.6B: total_ut_steps=4, early exit OFF.
- tomg-group-umd/huginn-0125: num_steps=16, fixed execution. Document that 16 is a pragmatic reference setting; model supports other depths.
Pin model revisions and tokenizer code, and make sure loop terminology maps to the actual physical architecture. Never assume identical loop counts across checkpoints.

## Primary question and policy semantics
- FULL: previous token K/V remains distinct for each computed recurrent iteration.
- LAST: previous token's final-iteration K/V is reused by all current recurrent iterations.
- AVERAGE: cache the separately averaged K and V vectors over computed recurrent iterations; each current iteration uses these mean historical K/V values.
- KV sharing means cross-iteration reuse of past-token K/V, not GPU sharing. Validate prefill causality, write timing, token positional correctness, and whether the reference algorithm requires a second pass. Do not call vllm-rlt SHARED equivalent to LAST until proved.
- Main report: FULL vs LAST accuracy/NLL deltas, AVERAGE secondary; policy-specific KV bytes only as supporting context.

## Evaluation
- GSM8K official TEST: fixed seed subset n=200 of 1,319; record IDs and selection hash.
- MATH-500: fixed seed subset n=200 of 500; preferably level/subject stratification, record exact IDs and selection hash.
- Accuracy: greedy decoding; correct per-model tokenizer/chat template, stop IDs/strings, max_new_tokens and prompting conventions pinned before policy comparison; use the same prompts/seeds/decode settings within each model for all cache policies.
- GSM8K: robust final numeric answer extraction (####, commas, negatives, decimals), canonicalize to numeric answer, distinguish parser failures from incorrect answers.
- MATH-500: balanced-brace \\boxed extraction, robust LaTeX/math equivalence verifier (not naive regex/string compare), parser-failure tracking, manual audit sample of parsing failures.
- NLL: teacher-forced gold-solution log likelihood on same examples, mask prompt tokens, report token-weighted average NLL (and if useful per-problem NLL) separately from generated-answer accuracy; capture tokenizer/template and gold-target definition.
- Evaluator audit: test answer parser with positive/negative/adversarial fixtures including nested boxed fractions; save prompt, raw generated answer, parsed answer, ground truth, score, and failure reason for EVERY item. Full policy parity against native HF model required before benchmarking LAST/AVERAGE.
- Prompt selection: prefer each model's authoritative official eval setup, if verifiable; otherwise explicitly identify and freeze a reasonable model-specific template. Avoid calling an invented template 'official'. Never optimize templates per policy.
- Optional 5–10 examples strictly for functional smoke tests, followed by all 200+200 for actual reporting. No unsupported claim of official paper score reproducibility when prompts differ.

## Execution (only stages 1–3)
1. Correctness semantics/reference: HF implementation + policy read/write tests and native FULL logits parity.
2. Quality evaluation: GSM8K200 + MATH500200, Accuracy & NLL for FULL/LAST/AVERAGE; report per-item raw data and Full-vs-Last paired deltas.
3. Lightweight serving validation: vllm-rlt after HF parity; measure active KV bytes, one or two representative batch sizes for decode TPOT/throughput with >=3 warm runs. Avoid broad context/depth/batch sweeps, TTFT/kernel-level deep dives, and capacity stress tests.
No separate phase 4/full dataset assessment for now.

## H200 x8
Use eight independent single-GPU workers on the authorized H200 server, rotating task/model/policy assignments so model and policy aren't confounded with GPU ID. For 200-problem eval, distribute deterministic disjoint sample shards and aggregate. Keep all 8 busy where possible. Respect original AGENTS.md restriction on the other older/shared GPU machine; do not touch unrelated processes. This commit changes plan only and does not execute GPU experiments.

## Deliverables
Implementation (when execution becomes possible), pinned environment/manifests, quality summary (Full, Last, Average for each model), paired confidence intervals, parse errors, measured recurrent KV bytes, representative decode performance, reproducible commands, raw per-example artifacts. Never report planned work as measured.
