# FULL vs STRICT shared (LAST / FIRST) — Huginn R32, Ouro-1.4B R4, Ouro-2.6B R4

Plan: `kv_policy/FULL_STRICT_BENICE_VALIDATION.md` (commit f4112d0).
Main run: `results/20261027T042338Z_full_first_last_strict_main` (GSM8K-200 + MATH500-200, BF16, greedy, caps 512/2048).
Pilot: `results/20261027T042338Z_full_first_last_strict_pilot` (32 + 32 items).
Audit: `results/20261027T042338Z_full_strict_audit`. Full tables: `summary.md`, `paired_vs_full.csv`, `items_*.csv`.

## Audit (before measuring quality)

- **Toy trace** (`results/20261027T042338Z_full_strict_audit/trace_table.md`, `trace_reads_writes.csv`). The engine equals the independent dictionary
  reference with logit difference 0.0 for every policy, for both families.
  - full/decode and full/strict read the per-recurrence history.
  - last/strict reads the terminal recurrence (r4) of every earlier token, including prompt tokens, during the
    token-by-token prefill.
  - first/strict reads r1 at every query recurrence.
  - The current token always uses its own current-recurrence K/V.
  - Prompt schedule: one parallel call for decode; one token per call for strict.
- **Native FULL parity, FP32** (`results/20261027T042338Z_full_strict_audit/parity`). For all three checkpoints, both modes match the official HF forward:
  max |Δlogit| ≤ 2.3e-3 and top-1 agreement 1.0.
- **Matched-path control, full/strict vs full/decode** (the go/no-go gate):
  - FP32: max |Δlogit| ≤ 6e-4, KL ≤ 1.4e-9, |ΔNLL| ≤ 8e-7, and every 64-token greedy generation is identical in all
    three models. **PASS.**
  - BF16: the numerical-path difference is not hidden: max |Δlogit| up to 7 (Ouro-1.4B) and 5 of 24 greedy
    generations diverge.
  - On the full sample, full/strict vs full/decode accuracy differences are all within ±3 pp (CI includes 0) and
    gold-NLL differences are ≤ 3e-4.

## Results (accuracy; Δ vs the matched full/strict path [95% CI]; token-weighted gold NLL)

| model / dataset | full/decode | full/strict | last/strict | first/strict |
|---|---:|---:|---:|---:|
| Huginn GSM8K | 0.415 | 0.420 | 0.405 (−0.015 [−0.050, +0.015]); NLL 0.918 | 0.005; NLL 3.05 |
| Huginn MATH500 | 0.155 | 0.145 | 0.160 (+0.015 [−0.005, +0.040]); NLL 0.669 | 0.015; NLL 2.13 |
| Ouro-1.4B GSM8K | 0.835 | 0.810 | **0.000** (−0.810); NLL 6.03 | 0.210; NLL 1.21 |
| Ouro-1.4B MATH500 | 0.710 | 0.740 | **0.000** (−0.740); NLL 5.42 | 0.050; NLL 1.26 |
| Ouro-2.6B GSM8K | 0.880 | 0.880 | 0.600 (−0.280 [−0.345, −0.215]); NLL 0.922 | 0.090; NLL 1.63 |
| Ouro-2.6B MATH500 | 0.560 | 0.565 | 0.330 (−0.235 [−0.315, −0.160]); NLL 0.811 | 0.075; NLL 1.43 |

## Reading (within each checkpoint; no ACT-vs-stochastic-training causality is claimed)

1. **Strict terminal sharing also collapses Ouro.** The Ouro failure is therefore *not solely* an artifact of
   LAST-CONVERT's prompt conversion.
   - Ouro-1.4B: last/strict is even worse than last/convert (gold NLL 6.0 vs 3.5; accuracy 0 in both).
   - Ouro-2.6B: last/strict drops 28 pp on GSM8K and 23.5 pp on MATH500. The earlier 2.6B GSM8K-vs-MATH asymmetry
     (convert: −11 vs −52.5 pp) becomes roughly symmetric under strict.
2. **Huginn is unaffected by strict terminal sharing** (Δ within ±1.5 pp, ΔNLL ≤ 0.002). This is consistent with
   BeNice!'s compress-s1 semantics being near-lossless for this kind of model.
3. **FIRST-recurrence sharing (same 1-plane budget) collapses every model.** This matches the direction of Ouro
   Table 14's "first-step" row.
   - For Ouro-1.4B it is less damaging than LAST in NLL (1.2 vs 6.0 gold NLL, GSM8K 0.21 vs 0.00). This is
     consistent with the diagnosis finding that 1.4B depends most on first-recurrence history.
4. **The paper's near-lossless "Last-step" result is not reproduced under either reading** (convert or strict) on
   these checkpoints with our prompts. The paper's sharing implementation remains unresolved. compress-s1 is
   BeNice!/Parcae's semantics and is not evidence of what Ouro Table 14 used.

## Notes

- Strict prefill is sequential: Huginn R32 strict jobs took 1.5–4 min per batch. No speed claims.
- Run, pilot and audit used the committed code f4112d0 plus the audit scripts in `scripts/strict/`
  (`audit_trace.py`, `mode_parity.py`). The source SHA in the receipts is f4112d0, except the audit parity runs,
  which used 0e2cbd8; the FULL path is unchanged between the two.
