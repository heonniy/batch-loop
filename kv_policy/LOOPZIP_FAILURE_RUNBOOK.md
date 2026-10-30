# LoopZip Phase-1/2 Executable Runbook — 10/30/50% Context KV Retention

Date: 2026-10-31
Branch: research/loopzip-attention-failure-first-20261031
Read the [research hypothesis and observational definitions](LOOPZIP_ATTENTION_FAILURE_FIRST_PLAN.md) before interpreting results.

## Status and scope

The *implemented* reference experiment is **fixed-depth Decode** under a completely normal Full-depth, token-major Prefill, using a FULL-sized loop-KV store and future-only Attention read masks. It includes Mean/Last/Max/Recent+Attention/Random policies, retention 10/30/50%, reused clean prompt KV, teacher-forced future NLL, optional greedy QA accuracy, failure classification and equal-budget Local-swap Oracle. It is NOT real physical KV compaction or HBM/TPOT/TPS measurement.

The later explanation of **why** a candidate swap works (immediate Attention Output, loop-selective decode masks, Recurrent Error Propagation), dynamically recorded/live exits, faithful H2O/SnapKV ports and physical memory saving remain future work. No H200 results were produced during GitHub authoring. The presence of a code commit is not a model-run result.

## What EXACTLY runs

- Phase 0: Freeze request prompts, checkpoint revision, role offsets, candidate Context token positions, budgets and code SHA. All Context and Instruction/Question tokens are fully processed before deciding what to retain.
- Phase 1: On the **clean prompt only**, compute Instruction/Question-to-Context core Attention by loop. For Context length Nc and b in (0.1,0.3,0.5), reserve K=max(1,floor(b Nc)) tokens. Keep **all recurrent loop/physical layers** for selected positions, mask ALL other Context positions only in future generated-token reads. Keep Instruction, Question, markers and generated KV unmasked across policies. Every policy uses the same K.
- Baselines: Mean-Attn, Last-Attn, Max-Attn (over recurrent-loop profiles); Recent+Mean-Attn (floor(K/4) latest Context positions included *inside* K); Random seeds 11/29/47; clean Full KV.
- Performance optimization: Full-depth prompt prefilled once per request/model; same immutable prompt KV reused, with generated-token scratch overwritten across policies. Independent full re-prefill parity is checked before scoring. This is a reuse optimization, not speculative approximated attention.
- Output: Future NLL (gold tokens 2..T, first target predicted pre-eviction), greedy short-answer Exact Match when enabled. Record every actual retained/evicted token. No gold answer or evidence tag input into policy selection. Score NLL for all requests, separately report clean-correct->wrong flips.
- Phase 2: Select Full-correct requests where **all four Attention policies** fail at a budget, plus matched Full-correct/no-failure controls. Without greedy, choose large **best-attention signed NLL damage** using a frozen threshold, explicitly label NLL-only. Starting from the best of four Attention policies by offline future NLL, create <=36 one-token exchange neighbors with same K. Candidate restoration spans high-ranked evicted, low-ranked evicted, and deterministic diverse evicted positions. Evaluate every neighbor with the same clean prompt and gold NLL. Report chosen local swap as OFFLINE ORACLE; it does NOT infer a deployable ranking rule.

### Before any GPU run

- CPU tests for pure policies, exact budget, manifest, classifying failures and exchange invariants must pass.
- Torch CPU fake-backbone test for cached-prefix reuse and actual grouped-mask effects must pass.
- Run a **real tiny or checkpoint-backed parity smoke** on Ouro AND Huginn (including real chat templates/tokenizer offsets); do not infer large checkpoint correctness from fake-model tests.
- Verify model revisions, dataset downloads/sha256, GPU memory ownership and prompt length. HotpotQA excerpt is a custom workload; later full_context run separately.
- Check the no-eviction clone produces <=1e-4 max per-token NLL difference from independent role_forward, and repeated cached passes reproduce it.
- Measure actual throughput/duration/peak HBM on the 5-request smoke BEFORE committing a large pilot. If a run is too expensive, change the cohort/target length in a NEW run rather than silently dropping jobs or changing the frozen manifest.
- Ensure every failed/missing claim is resolved. The worker uses exclusive claims; a failed run requires manual inspection before deciding whether to re-run.

## Commands

Use a clean branch HEAD and run the CPU gates first:

~~~bash
git fetch origin
git checkout research/loopzip-attention-failure-first-20261031
cd kv_policy
python -m pytest -q \
  tests/test_loopzip_policy.py \
  tests/test_loopzip_failure_runner.py \
  tests/test_loopzip_cached_eval.py
~~~

PHASE 1, smoke (one pinned Ouro, 5 Controlled Retrieval questions; all three retention ratios):

~~~bash
python scripts/loopzip_failure_screen.py --prepare \
  --run-dir results/loopzip_failure_smoke5 \
  --models ouro-2.6b \
  --datasets controlled_retrieval \
  --n-examples 5 --retention 0.10,0.30,0.50 \
  --policies mean_attn,last_attn,max_attn,recent_attn,random \
  --free-generate
# Verify GPU ownership and capacity before execution.
python scripts/loopzip_failure_screen.py --execute \
  --run-dir results/loopzip_failure_smoke5 --gpu-ids 0
python scripts/loopzip_failure_screen.py --analyze \
  --run-dir results/loopzip_failure_smoke5
~~~

PHASE 1, pilot (H200x8, 2 models x 2 datasets x 20 questions = 80 unique model-question cases):

~~~bash
python scripts/loopzip_failure_screen.py --prepare \
  --run-dir results/loopzip_failure_pilot20 \
  --models ouro-2.6b,huginn-0125 \
  --datasets controlled_retrieval,hotpotqa_distractor \
  --n-examples 20 --retention 0.10,0.30,0.50 \
  --policies mean_attn,last_attn,max_attn,recent_attn,random \
  --max-prompt-tokens 1536 --max-target-tokens 64 \
  --free-generate
python scripts/loopzip_failure_screen.py --execute \
  --run-dir results/loopzip_failure_pilot20 \
  --gpu-ids 0,1,2,3,4,5,6,7
python scripts/loopzip_failure_screen.py --analyze \
  --run-dir results/loopzip_failure_pilot20
~~~

**Phase 2 follows only after COMPLETE Phase 1 and manual result inspection:**

~~~bash
python scripts/loopzip_failure_screen.py --prepare-swap \
  --run-dir results/loopzip_failure_pilot20 \
  --max-failures-per-cell 12 \
  --max-controls-per-cell 3 \
  --min-delta-nll 0.05
python scripts/loopzip_failure_screen.py --execute-swap \
  --run-dir results/loopzip_failure_pilot20 \
  --gpu-ids 0,1,2,3,4,5,6,7
python scripts/loopzip_failure_screen.py --analyze-swap \
  --run-dir results/loopzip_failure_pilot20
~~~

Each failure/control may be included at several budgets, but the clean prompt prefix is recomputed **once per swap task**, not per individual exchange candidate. The maximum neighborhood size per selected budget is 6 retained candidates x 6 evicted candidates = 36, plus the original selected set. Swap-Greedy checks only the best tested swap; most swap comparisons use signed NLL.

## Files to look at

- run_manifest.json, jobs.json: frozen SHA, checkpoint revision, prompt and role token positions, target IDs, budget rounding, data provenance.
- raw/<job>.json: all policy-budget candidate sets, clean per-loop attention, future NLL and optional Greedy QA.
- analysis/screening_quality.csv: one row per model/request/budget/policy, signed NLL damage and accuracy flips.
- analysis/baseline_qa.csv: clean full-KV ability and actual generated short answer.
- analysis/failure_cases.csv: full-wrong, all-attention-wrong, mean-only aggregation failure and NLL-only labels.
- analysis/SUMMARY.json and completeness.json: request-grouped metrics, error/invalid/missing guard.
- swap_jobs.json and swap_selection.json: frozen second-stage cohort and case-selection criteria.
- swap_raw/<job>.json, analysis/swap_candidates.csv, analysis/SWAP_SUMMARY.json, swap_completeness.json: all tested swaps and equal-budget NLL gains.

## Paper-claim decision rule

**Attention selection gap** requires Full-correct or meaningful Full-NLL reference; several strong attention baselines losing at the *same K*; and at least one **actual evaluated simultaneous KV retained set** at the same K with better future NLL or greedy accuracy. One local swap success is an existence proof for that request/budget, NOT a global best set or architecture-specific mechanism. If the only gain is against Mean while Last/Max succeeds, call it a **loop attention aggregation** issue. A failed 10% setting by itself may be just a severe capacity shortage. Keep negative/non-improving cases.

This work cannot yet prove that **recurrence caused** any residual Attention failure. That requires a separate Phase-3 decode-loop intervention and matched immediate Attention Output controls. The Phase-1/2 goal is to find and document genuine, reproducible same-budget *failures and recoveries*, then decide what mechanism to investigate.
