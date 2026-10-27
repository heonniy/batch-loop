# FULL vs strict shared (LAST / FIRST) — Huginn R32, Ouro-1.4B R4, Ouro-2.6B R4

Plan: `kv_policy/FULL_STRICT_BENICE_VALIDATION.md` (commit f4112d0, which adds first/strict).

## Runs

| Run | Path | Scope |
|---|---|---|
| Audit | `results/20261027T042338Z_full_strict_audit` | Toy trace, independent-reference checks, native parity, strict-vs-decode mode parity |
| Pilot | `results/20261027T042338Z_full_first_last_strict_pilot` | Stage-1 32 + 32 items, 288 jobs, 0 failures |
| Main | `results/20261027T042338Z_full_first_last_strict_main` | GSM8K-200 + MATH500-200, 1800 jobs, 0 failures |

All runs: BF16, greedy, caps 512 / 2048, fixed depth, early exit off, HF reference engine.

## Policies

- **full/decode**: depth-indexed history with a loop-major parallel prompt.
- **full/strict**: the same FULL reads with a token-major prompt. This is the matched numerical-path control.
- **last/strict**: every past token, prompt included, is read at its terminal recurrence by every query recurrence;
  the prompt is processed token-major. BeNice!/Parcae `compress-s1` semantics, re-implemented here; Parcae code
  was not run.
- **first/strict**: the same as last/strict, except past tokens are read at their first recurrence.

## Audit (gate)

**Toy trace** (`results/20261027T042338Z_full_strict_audit/trace_table.md`, `trace_reads_writes.csv`): prompt A,B and generations C,D on tiny Ouro and
Huginn models.
- FULL: Q^r reads K^r.
- last/strict: Q^r reads K^R, also inside the prompt (B reads A^R).
- first/strict: Q^r reads K^1.
- The current token always uses its own current-recurrence K/V.
- Engine vs independent dictionary reference: max |Δlogit| = 0 for all 5 policies × 2 families.

**Native parity, FP32** (`results/20261027T042338Z_full_strict_audit/parity/parity_*.json`): FULL decode and strict vs the official HF forward give
|Δlogit| ≤ 2.3e-3 with top-1 1.000 for all three models. last/strict vs the reference gives 0.

**FP32 full/strict vs full/decode** (`results/20261027T042338Z_full_strict_audit/parity/mode_parity_*.json`, 4 + 4 items per model):
|Δlogit| ≤ 6e-4, KL ≤ 1.4e-9, |ΔNLL| ≤ 8e-7, and 64-token greedy generations identical in 24/24 → **gate passed**.

**BF16 numerical-path control**:
- Teacher-forced: full/strict vs full/decode gives max |Δlogit| ≤ 7, top-1 ≥ 0.98 and |ΔNLL| ≤ 4e-3. Greedy
  generations diverge in 5/24.
- Full-sample effect: full/strict vs full/decode Δacc is within ±0.03 (all CIs include 0) and ΔNLL ≤ 3e-4.
- This is why last/strict is compared against the matched full/strict.

## Main results (GSM8K-200 / MATH500-200)

Δ is vs the matched full/strict, with paired 95% bootstrap CIs.

| Model | Data | full/decode | full/strict | last/strict (Δ) | first/strict | last/strict NLL vs full/strict |
|---|---|---:|---:|---|---:|---|
| Huginn R32 | GSM8K | 0.415 | 0.420 | 0.405 (−0.015 [−0.050, +0.015]) | 0.005 | +0.0017 |
| Huginn R32 | MATH500 | 0.155 | 0.145 | 0.160 (+0.015 [−0.005, +0.040]) | 0.015 | +0.0003 |
| Ouro-1.4B | GSM8K | 0.835 | 0.810 | 0.000 (−0.810 [−0.865, −0.755]) | 0.210 | +5.34 |
| Ouro-1.4B | MATH500 | 0.710 | 0.740 | 0.000 (−0.740 [−0.800, −0.680]) | 0.050 | +4.85 |
| Ouro-2.6B | GSM8K | 0.880 | 0.880 | 0.600 (−0.280 [−0.345, −0.215]) | 0.090 | +0.223 |
| Ouro-2.6B | MATH500 | 0.560 | 0.565 | 0.330 (−0.235 [−0.315, −0.160]) | 0.075 | +0.346 |

For comparison, our earlier last/convert results (E0) were: Ouro-1.4B 0.000 / 0.015; Ouro-2.6B 0.770 / 0.035;
Huginn 0.445 / 0.165.

## Findings

1. **Huginn: no change under strict terminal sharing.** All Δ CIs include 0 and ΔNLL ≤ 0.002. This replicates the
   last/convert finding under BeNice!-style semantics.
2. **Ouro collapse is not only a LAST-CONVERT prompt-conversion artifact.**
   - Ouro-1.4B collapses completely under last/strict: 0.000 accuracy, NLL +5, 96–99% of generations hit the cap.
     This is worse in NLL than last/convert (+2.7).
   - Ouro-2.6B loses 28 pp on GSM8K and 23 pp on MATH500.
3. **The Ouro-2.6B dataset asymmetry depends on the semantics.**
   - Under last/convert: GSM8K −11 pp, MATH500 −52.5 pp.
   - Under last/strict: GSM8K −28 pp, MATH500 −23.5 pp.
   - The anomaly does not survive in the same form under strict mode. Same-prefix per-position analysis was
     therefore not added.
4. **FIRST shows a family-dependent pattern.**
   - first/strict collapses for Huginn (0.005 / 0.015) and Ouro-2.6B (0.090 / 0.075), in the direction of Ouro
     Table 14's "first-step" row (18.73 GSM8K).
   - For Ouro-1.4B, first/strict beats last/strict: accuracy 0.210 vs 0.000 on GSM8K and 0.050 vs 0.000 on
     MATH500; NLL 1.21 vs 6.07 and 1.21 vs 5.39. Both are still far below FULL.
   - This matches the diagnosis branch, where (layer 0, recurrence 1) history carries most of Ouro-1.4B's
     sensitivity.
5. **The paper's near-lossless Last (Table 14) is not reproduced under either our last/convert or the
   BeNice!-style last/strict.**
   - The Ouro paper's implementation is unreleased, so its sharing semantics remain **unresolved**.
   - compress-s1 is BeNice!/Parcae's policy, not evidence of what Ouro Table 14 used.
   - The cross-model difference is an observation within these checkpoints. No ACT-vs-stochastic causality is
     claimed.

## Notes

- Strict mode prefills the prompt token by token (R recurrences per token). Pilot jobs took up to about 4 min each
  for Ouro-2.6B MATH. No speed claim.
- Per-item outputs: `analysis/items_gen.csv`, `items_nll.csv`. Paired tables: `summary.md`.
- Completeness: `strict_validation_completeness.json`.
- `scripts/strict/audit_trace.py` and `scripts/strict/mode_parity.py` are new audit code.
- Repository-root `AGENTS.md` blocks GPU use for the Phase-04R task only. This is a separate, user-requested
  kv_policy task.
