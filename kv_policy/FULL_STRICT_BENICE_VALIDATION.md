# Full vs Strict Shared: Huginn / Ouro reproducibility study

This is **executable evaluation code plus protocol**, not a completed GPU result.

## Scope and semantic choice

Models: pinned Huginn-0125 R32, Ouro-1.4B R4 and Ouro-2.6B R4.
Original checkpoints, fixed depth, early exit OFF, HF reference engine, H200 x8.
Datasets: existing frozen GSM8K-200 and MATH-500-200; pilot uses the pinned
32-per-dataset Stage-1 selection. Primary inference cache conditions:

1. `full/decode`: depth-indexed FULL history; loop-major parallel prompt.
2. `full/strict`: **same FULL read semantics** but token-major sequential
   prompt. This controls for numerical and batching changes introduced by
   token-major execution.
3. `first/strict`: token-major prompt; each processed token retains its **first recurrence** KV\n   for all future query recurrence steps. This is the same 1-plane budget as LAST,\n   but should not be confused with the L0R1 selective-protection policy.\n4. `last/strict`: token-major prompt; after each token finishes R recurrences,
   save its terminal KV. Each later token, at ANY recurrence, reads the
   previous token's terminal KV. The token being processed always computes
   its own loop-specific K/V; no causal future-state read.

These conditions must NOT be conflated with the older `last/convert`:
FULL parallel prefill then retain terminal history. Also do not silently
call `full/strict` the native fast production execution mode; it is the
matched numerical control for strict shared.

## BeNice! reference implementation

Official repository: https://github.com/SandyResearch/parcae
Evaluation file:
https://github.com/SandyResearch/parcae/blob/main/eval_configs/eval-val-loss-compress-s1.yaml
The config uses `strategies: [compress-s1]`, `device_batch_size: 1`
for KV-cache validation. The code and Appendix C describe a single shared
terminal cache with token-major causal evaluation. This is the semantic
reference only. **Parcae code is not directly reusable for Ouro/Huginn
checkpoint internals, attention layout, or RNG**; do not represent this
as a run of BeNice!'s engine. Pin the upstream commit when importing
or adapting any external files; preserve license and attribution.

## Required audit before measuring quality

- Read `AGENTS.md`, existing E0–E3 reports and `kvp/engine.py`.
- Write an independent toy A,B prompt -> C,D generation read/write trace
  for each model, layer and recurrence:
  FULL: Q_B^r reads K_A^r; STRICT: Q_B^r reads K_A^R.
- Verify self-token KV always current-depth and causal mask/RoPE are identical.
- Check a toy B1 independent slow reference against model outputs for
  `full/decode`, `full/strict`, `last/strict`.
- FP32 baseline parity to native FULL; BF16 FULL/strict vs FULL/decode
  difference is a numerical-path control and must be reported, not hidden.
- Verify `full/strict` and `full/decode` match in FP32, with near-zero
  NLL/logits difference and same token IDs where numerical margin allows.
- Trace actual prompt positions, pad masks and cache write timings.
- Report official Ouro paper Last policy as **unresolved** until the
  authors' implementation matches; `compress-s1` is BeNice!/Parcae,
  not proof of what Ouro Table 14 used.

## Execute on authorized H200 x8 host

```bash
git checkout research/full-strict-benice-validation-20261027
cd kv_policy
python scripts/strict_shared_validation.py \
  --run-dir results/full_first_last_strict_pilot --batching pilot
python scripts/strict_shared_validation.py \
  --run-dir results/full_strict_pilot --batching pilot --execute \
  --gpu-ids 0,1,2,3,4,5,6,7
```

The pilot uses 32 GSM8K and 32 MATH per model with two tasks (generation
and NLL), four cache settings. Token-major strict prompt can be very slow;
run the pilot first, inspect time and GPU memory, then only if valid:

```bash
python scripts/strict_shared_validation.py \
  --run-dir results/full_first_last_strict_main --batching main --execute \
  --gpu-ids 0,1,2,3,4,5,6,7
```

Execution is resumable by the existing worker claim convention, but stale
locks on failed jobs need deliberate manual audit. Do not rerun an existing
run under a different git SHA without recording the deviation.

## Analysis requirements

- Accuracy, gold-token NLL, first generated disagreement, repetition,
  truncated/no-answer fractions, identical prompt hashes, policy-paired
  outcomes and per-example output.
- Compare FULL/decode vs FULL/strict **first**; then LAST/strict vs the
  SAME full/strict matched path and also vs official-style FULL/decode.
- Record prompt lengths, output caps (GSM8K 512, MATH500 2048), and
  model-specific prompts (MATH-500 Ouro official 5-shot is unpublished).
- Evaluate the difference across models WITHIN each checkpoint;
  do not claim ACT vs stochastic training causality.
- Check 2.6B GSM8K vs MATH500 separately, including same-prefix
  first 32/64/128 positions if the accuracy anomaly survives strict mode.
- Note strict sharing may incur enormous sequential prefill time.
  No speedup claims without an optimized scheduler or serving implementation.
- Preserve historical receipts; push raw outputs, completeness manifest,
  test logs, paper-semantics audit and paired analysis on a new commit.

## Go / No-Go

GO to full sample only if matched FULL/strict parity and strict-KV toy
reference succeed without unexplained failure. If FULL/strict differs
substantially from FULL/decode in FP32, fix before interpreting LAST.
If STRICT shares also collapse on Ouro, document that this is
not solely a Last-Convert prompt conversion artifact. If they do not
collapse, revisit the original inference semantic mismatch.

## First-loop baseline added 2026-10-27

The matrix is now FULL/decode, FULL/strict, LAST/strict, FIRST/strict.
FIRST stores K/V from recurrence 1 (zero-based index 0) after each
prompt and decode token completes its full R recurrence computations;
all future recurrences read that retained history. Current token self-KV
continues to use its own recurrence. `kvp.reference` independently
implements the same rule. CPU float64 unit tests now include FIRST,
including no-future-leakage, left-padding, Full-vs-First and R=1 parity.
**Use new run directories**, since existing job manifests intentionally
reject adding policies to an already created experiment. No GPU results
or paper quality numbers are claimed by this code change.
