# Ouro LAST/strict + protected banks (layer 0, recurrence 1)

Run: `results/20261027T152845Z_ouro_last_strict_protect`. Ouro-1.4B and Ouro-2.6B, GSM8K-200 + MATH500-200, BF16, greedy, caps 512/2048.

- **New cells:** `bank*/strict` = LAST/strict, i.e. token-major prompt in which every earlier token is read at its
  terminal recurrence, EXCEPT the protected (layer, recurrence) banks. Those keep per-recurrence K/V for every
  historical token, prompt and generated (kvp.banks).
- **Reference check:** `tests/test_banks.py::test_bank_strict_matches_dict_reference` checks the engine against the
  independent dictionary reference at 1e-9, and no banks == last/strict exactly.
- **Banks** (`banks.json`, fixed before this run):

| Policy | Protected banks |
|---|---|
| bankP0 | (0,0) |
| bankR1all | (l,0) for every layer |
| bankcal (1.4B) | (0,0), (1,0): the calibration-selected set from disjoint prompts. For 2.6B the calibration selected (0,0) alone, i.e. bankP0. |
| bankrand0 | one random same-budget bank: (20,2) for 1.4B, (30,0) for 2.6B |

- **Reused receipts:** full/decode, full/strict, last/strict and first/strict were copied from `results/20261027T042338Z_full_first_last_strict_main` (same items,
  batching, engine, git tree). `summary.md` / `paired_vs_full.csv` were computed on the union.

Recovered = (x − last/strict) / (full/strict − last/strict). Retained recurrent KV is relative to FULL.

| Model / dataset | full/strict | last/strict | L0R1 (KV 0.26) | Calibrated (KV 0.27) | All recurrence-1 (KV 0.50) | Random bank |
|---|---:|---:|---:|---:|---:|---:|
| 1.4B GSM8K | 0.810 | 0.000 | 0.405 (acc 50%, NLL 98%) | 0.660 (81%) | 0.720 (89%) | 0.000 |
| 1.4B MATH500 | 0.740 | 0.000 | 0.170 (acc 23%, NLL 96%) | 0.485 (66%) | 0.610 (82%) | 0.000 |
| 2.6B GSM8K | 0.880 | 0.600 | 0.770 (acc 61%, NLL 89%) | = L0R1 | 0.845 (88%) | 0.555 |
| 2.6B MATH500 | 0.565 | 0.330 | 0.410 (acc 34%, NLL 76%) | = L0R1 | 0.500 (72%) | 0.330 |

All protected cells beat last/strict (paired McNemar p ≤ 0.026). The random bank does not.

## Reading

- **Protecting layer 0 / recurrence 1 alone removes most of the strict-sharing NLL damage:**
  - Ouro-1.4B: gold NLL 6.03 → 0.86 on GSM8K and 5.42 → 0.78 on MATH500.
  - It restores only part of the accuracy: 50% / 23% (1.4B) and 61% / 34% (2.6B).
  - The remaining accuracy gap to full/strict is significant in every cell.
- **The calibrated two-bank set for 1.4B** (adds layer 1, recurrence 1) recovers 81% / 66%.
- **Protecting every recurrence-1 bank** recovers 72–89% of the accuracy at half of FULL's retained recurrent KV.
- **Compared with LAST-CONVERT + L0R1 (diagnosis run):** strict + L0R1 recovers less accuracy on 1.4B MATH500
  (0.170 vs 0.345). Under strict, even the prompt history is terminal for every non-protected bank, so more banks
  matter.
- **Same-budget random bank:** no recovery (−16 pp to 0 pp), so the effect is specific to the recurrence-1 banks,
  above all (0,0).
