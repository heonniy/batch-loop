# HUGINN_HARD_DECISION — B0 depth pilot (32 GSM8K + 32 MATH500, default prompt, BF16 greedy, STRICT)

Run: `results/20261029T124738Z_focus_huginn_depth_pilot`; 192 jobs, 0 errors. R = test-time recurrence of the same
Huginn-0125 checkpoint; R=32 is native.

| R | GSM8K FULL / LAST (Δ LAST−FULL [CI]) | MATH500 FULL / LAST (Δ [CI]) | gold NLL Δ (GSM / MATH) | FULL vs R32 FULL |
|---|---|---|---|---|
| 8 | 0.250 / 0.406 (+0.156 [−0.06, +0.41]) | 0.125 / 0.250 (+0.125 [0.00, +0.28]) | −0.025 / −0.082 | 0.50 / 0.33 (unusable) |
| 16 | 0.469 / 0.469 (0.000) | 0.281 / 0.312 (+0.031) | +0.037 [+0.018, +0.059] / −0.002 | 0.94 / 0.75 |
| 24 | 0.438 / 0.531 (+0.094 [0.00, +0.19]) | 0.375 / 0.344 (−0.031) | +0.009 / −0.003 | 0.88 / 1.00 |
| 32 | 0.500 / 0.500 (0.000) | 0.375 / 0.344 (−0.031 [−0.094, 0.00]) | +0.005 / −0.000 | 1 |

## Screening rule

GO requires both `A_full(R) ≥ 0.8 · A_full(R32)` and `A_full − A_last ≥ 0.05` with a supporting CI.

**No condition passes. B0 screening is NO-GO on GSM8K and MATH500 with the default prompt:**
- At every usable depth (R16/24/32, FULL ≥ 0.8 · R32 except MATH500 R16 at 0.75), LAST is within −3 to +9 pp of FULL.
- The only significant ΔNLL is GSM8K R16 (+0.037 nats). It does not translate into accuracy.
- At R8, FULL itself is unusable (0.25 / 0.125). LAST is *better* than FULL there (+16 / +12.5 pp; NLL −0.03 / −0.08).
  This is not compression headroom.

## Next step (per plan; not launched)

B1 prompt-stress at R16 / R24 / R32 with `reason` and `distract512` (the plan's commands), generation only, same
frozen items.
- If B1 also shows no FULL > LAST gap: Huginn on GSM8K / MATH500 is a negative-generalisation baseline.
- A hard-task adapter (AIME-like, RULER, code) would be the remaining conditional option. It is not implemented.
