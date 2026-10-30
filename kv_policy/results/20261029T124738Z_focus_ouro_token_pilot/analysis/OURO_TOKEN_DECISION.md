# OURO_TOKEN_DECISION — A0 pilot (32 GSM8K + 32 MATH500, BF16 greedy, token-major STRICT)

Run: `results/20261029T124738Z_focus_ouro_token_pilot`; 624 jobs, 0 errors.

- **KV accounting.** KV% is the logical FULL-recurrent-KV fraction: realised for token policies
  (`token_loop_logical_memory.csv`, generation task), static for layer masks. LAST = 25%, ALL-R1 (bankR1all) = 50%.
- **Paired comparison.** Δ is paired vs ALL-R1 by UID [95% bootstrap CI] (`sub50_paired_vs_loop1all.csv`).
- **No HBM saving is claimed.** The engine allocates full-sized broadcast buffers.

## Ouro-1.4B (ALL-R1: GSM8K 0.719, MATH500 0.688; FULL 0.781 / 0.812)

| Policy | KV% (GSM / MATH) | GSM8K acc (Δ vs ALL-R1) | MATH500 acc (Δ) |
|---|---|---|---|
| bankcal (2 banks, layer-only) | 27.1 | 0.719 (+0.000) | 0.656 (−0.031) |
| bankTopB35 | 34.4 | 0.656 (−0.062) | 0.688 (+0.000) |
| tokL1hash40 | 34.9 | 0.469 (−0.250 [−0.469, −0.031]) | 0.219 (−0.469) |
| tokL1hash60 | 40.0 | 0.656 (−0.062) | 0.438 (−0.250) |
| tokL1prompt (prompt tokens only) | 34.9 / 31.9 | 0.688 (−0.031) | 0.156 (−0.531) |
| tokL1decode (generated tokens only) | 40.2 / 43.3 | 0.000 | 0.000 |
| tokL1delta50 (current-token K/V delta) | 39.1 / 40.4 | 0.656 (−0.062) | 0.344 (−0.344) |

- **Token-selective first-loop retention fails on 1.4B.** No token policy is within 3 pp of ALL-R1 on both tasks at
  ≤ 35% (or ≤ 40%) KV.
- **Uniform hashing degrades smoothly with the protected fraction.** MATH500 needs nearly every token.
- **Generated tokens alone collapse; prompt tokens alone suffice only for GSM8K.**
- **The layer-only 2-bank recipe (bankcal, 27% KV) is already within 3 pp of ALL-R1 on both tasks** (point
  estimates; CIs ±0.13).
- **Decision for 1.4B: token-adaptive A0 NO-GO.** Keep the simple layer-only recipe
  ("A''-If exact same two banks beat all token variants").

## Ouro-2.6B (ALL-R1: GSM8K 0.781, MATH500 0.594; FULL 0.812 / 0.562)

| Policy | KV% (GSM / MATH) | GSM8K acc (Δ vs ALL-R1) | MATH500 acc (Δ) |
|---|---|---|---|
| bankcal (1 bank = L0R1) | 25.5 | 0.750 (−0.031) | 0.531 (−0.062) |
| bankTopB35 | 34.9 | 0.688 (−0.094) | 0.625 (+0.031) |
| tokL1hash40 | 34.9 | 0.688 (−0.094) | 0.656 (+0.062) |
| tokL1hash60 | 40.0 | 0.656 (−0.125) | 0.625 (+0.031) |
| tokL1prompt | 43.9 / 41.5 | 0.781 (+0.000) | 0.688 (+0.094) |
| tokL1decode | 30.7 / 31.5 | 0.594 (−0.188) | 0.500 (−0.094) |
| **tokL1delta50** | **35.1 / 33.9** | **0.781 (+0.000 [−0.094, +0.094])** | **0.656 (+0.062 [−0.094, +0.219])** |

- **tokL1delta50 meets the A0 point-estimate criterion on both tasks.** It is within 3 pp of ALL-R1 at about 34–35%
  logical KV, i.e. about 30% extra saving vs ALL-R1.
  - Its GSM8K KV (35.1%) sits on the 35% boundary.
  - Its MATH500 gold-NLL is slightly above ALL-R1 (0.640 vs 0.611).
  - The CIs span about ±0.1–0.15, so non-inferiority is **not** shown.
- **At a matched budget (about 35%), it beats layer-only bankTopB35 and uniform hash-40 on GSM8K** (0.781 vs
  0.688 / 0.688), and is comparable on MATH500.
- **The 1-bank calibrated set (25.5%)** loses 3–6 pp.
- **Decision for 2.6B: CONDITIONAL / weak A0-GO candidate.** It requires the 200 + 200 confirmation with the
  layer-only and random controls before any token-level claim.

## Recommended next step (not launched)

Run 200 GSM8K + 200 MATH500 on Ouro-2.6B (and 1.4B for completeness) with frozen hypotheses:
- full, last, bankR1all
- bankcal, bankTopB35, bankTopB40
- tokL1delta50, tokL1hash40, tokL1prompt
- a second delta threshold, chosen on held-out calibration and not on these pilot items

The causal delta selector's threshold was not calibrated here.
