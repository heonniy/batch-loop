# Ouro KV-sharing collapse — reproduction and training-free diagnosis

Plan: `kv_policy/OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md`. Run: `results/20261026T123825Z_ouro_sharing_diagnosis`.

| Section | Report |
|---|---|
| Step 0 | `semantics_audit.md` |
| Step 1 pilot | `step1_report.md` |
| Step 1 full | `step1_full_report.md` |
| Step 2 | `step2_report.md` |
| Step 3 | `step3_memory.md` |

"LAST-CONVERT" is **our** policy throughout. The original paper's Last semantics remain unverified (Step 0).

## Answers to the hypotheses

### H1 — the (layer 0, recurrence 1) KV mismatch accounts for most of the collapse: supported (strongest in Ouro-1.4B)

On a frozen FULL prefix, swapping only that bank to terminal history (everything else FULL) reproduces LAST-CONVERT's
damage:

| Row | Ouro-1.4B: KL to FULL | Ouro-1.4B: top-1 | Ouro-2.6B: KL to FULL | Ouro-2.6B: top-1 |
|---|---:|---:|---:|---:|
| Swap (layer 0, recurrence 1) only | 2.55 | 0.43 | 0.46 | 0.87 |
| Full LAST-CONVERT | 1.91 | 0.54 | 0.64 | 0.85 |

The swap row equals or exceeds full LAST-CONVERT's divergence for Ouro-1.4B and is ~70% of it for Ouro-2.6B. FP32
repeats agree. Controls show no effect:
- Huginn R32 has KL ≤ 1.5e-3 for every row.
- A random single bank or the layer-1 recurrence-1 bank (negative control) does not recover accuracy.

### H2 — protecting that bank restores quality at much lower retained KV than FULL: partially supported

Full frozen GSM8K-200 / MATH500-200, accuracy (fraction of the LAST→FULL accuracy gap recovered).
Retained recurrent KV relative to FULL is 0.25 for LAST-CONVERT, 0.255–0.26 for one bank, 0.271 for two banks and
0.50 for all recurrence-1 banks.

| Model / dataset | FULL | LAST-C | L0R1 (1 bank) | Calibrated (1.4B: 2 banks; 2.6B: = L0R1) | All recurrence-1 (0.50) |
|---|---:|---:|---:|---:|---:|
| 1.4B GSM8K | 0.835 | 0.000 | 0.640 (77%) | 0.670 (80%) | 0.745 (89%) |
| 1.4B MATH500 | 0.710 | 0.015 | 0.345 (48%) | 0.535 (75%) | 0.635 (89%) |
| 2.6B GSM8K | 0.880 | 0.770 | 0.845 (68%) | 0.845 (68%) | 0.865 (86%) |
| 2.6B MATH500 | 0.560 | 0.035 | 0.450 (79%) | 0.450 (79%) | 0.495 (88%) |

- Gold-NLL recovery is 86–99% for the calibrated sets.
- The residual accuracy gap to FULL is still significant (95% CI excludes 0) for 1.4B GSM8K and MATH500 and for
  2.6B MATH500.
- The same-budget random bank and the negative-control bank recover nothing (≤ 0.09).

### H3 — part of the discrepancy with the paper is semantics: plausible, not established

- The diagnostic LAST-DECODE reading keeps the prompt K/V per recurrence and terminalises only generated tokens.
  It is near-lossless for Ouro-2.6B (GSM8K −2.0 pp, MATH500 −6.5 pp, both CIs include or nearly include 0) and for
  1.4B GSM8K up to −14 pp.
- It still collapses on 1.4B MATH500 (0.190 vs 0.710). Table 14's checkpoint is most likely 1.4B, so semantics alone
  do not explain the reported near-lossless MATH-500.
- The paper's prompt protocol is unpublished and its sharing code is unreleased.
- ACT is not shown to cause the failure. Architecture, depth and training all covary.

## Step 2 — recovery vs persistent mismatch

- Protecting (layer 0, recurrence 1) cuts the fixed-prefix KL by about 13×.
- It leaves a shifted recurrent endpoint: final-step hidden error vs FULL is 0.57 (1.4B) and 0.37 (2.6B), compared
  with 1.02 and 0.61 under LAST-CONVERT.
- Protecting every recurrence-1 bank lowers it to about 0.2.
- So the failure is a persistent-context mismatch concentrated in the first recurrence. It is not a one-off
  perturbation that later recurrences repair. Under an unchanged history, E2 showed one-call perturbations are
  repaired.

## Decision (plan rules)

**CONDITIONAL.**
- Targeted protection recovers most of the lost NLL (86–99%) and most of the accuracy (68–80%) at about 1.02–1.08×
  LAST-CONVERT's retained KV (0.255–0.271 of FULL).
- It passed the bank-vs-dictionary-reference unit tests and the B1 FP32 subset (L0R1 6/8 for both models).
- It is CONDITIONAL rather than GO for two reasons:
  1. The bank count is model-specific: Ouro-1.4B needs the calibrated 2-bank set to reach the "most" level on
     MATH500, where one bank recovers only 48%.
  2. A statistically significant accuracy gap to FULL remains in three of four cells.
- Not NO-GO: protection works, its cost is far below FULL, and correcting the native-parity checks did not remove
  the collapse (FP32 official forward = reference FULL).

## Step 3 — memory

See `step3_memory.md`.
- Retained recurrent KV per token is L + |protected| planes, versus L × R for FULL.
- Every policy still needs all R planes for the prompt during the parallel FULL prefill.
- No compact store or serving benchmark was run, and no speed-up is claimed.

## Deviations and incidents

- Full-evaluation random control: 1 seed (bank (20,2) for 1.4B, (30,0) for 2.6B) for cost reasons. The pilot used
  5 seeds, all collapsing.
- FULL / LAST-CONVERT full-evaluation receipts were reused from E0 (same main batching and engine path).
- The Ouro-2.6B calibration was accidentally killed by the agent while closing idle tmux sessions and was restarted
  from scratch. The 1.4B calibration is unaffected.
- `last-decode` rows are a diagnostic alternate reading only; they are not part of the primary comparison.
