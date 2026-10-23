Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full | decode | 200 | 0.415 | 0.415 | 0.010 | 3 / 0 | 0.340 | 0.8996 | 0.9162 | 105.0 | 0.015 |
| huginn-0125 | gsm8k | last | convert | 200 | 0.445 | 0.445 | 0.015 | 1 / 0 | 0.360 | 0.9010 | 0.9179 | 103.5 | 0.005 |
| huginn-0125 | gsm8k | average | convert | 200 | 0.420 | 0.400 | 0.085 | 2 / 0 | 0.375 | 0.8715 | 0.8836 | 103.4 | 0.010 |
| huginn-0125 | math500 | full | decode | 200 | 0.155 | 0.150 | 0.200 | 47 / 1 | 0.150 | 0.6317 | 0.6679 | 611.2 | 0.235 |
| huginn-0125 | math500 | last | convert | 200 | 0.165 | 0.155 | 0.185 | 52 / 1 | 0.160 | 0.6315 | 0.6677 | 650.2 | 0.260 |
| huginn-0125 | math500 | average | convert | 200 | 0.125 | 0.090 | 0.285 | 49 / 1 | 0.120 | 0.6398 | 0.6791 | 610.8 | 0.245 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full/decode | last | convert | +0.030 [+0.005, +0.060] | 1 / 7 | 0.0703 | +0.0014 [-0.0002, +0.0034] |
| huginn-0125 | gsm8k | full/decode | average | convert | +0.005 [-0.040, +0.050] | 11 / 12 | 1 | -0.0281 [-0.0339, -0.0224] |
| huginn-0125 | math500 | full/decode | last | convert | +0.010 [-0.015, +0.040] | 3 / 5 | 0.727 | -0.0003 [-0.0010, +0.0003] |
| huginn-0125 | math500 | full/decode | average | convert | -0.030 [-0.070, +0.010] | 11 / 5 | 0.21 | +0.0081 [+0.0066, +0.0096] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | average | convert | 0.000 | 0.010 | 0.420 | 0.420 | +0.005 [-0.040, +0.050] | 0 | 0 |
| huginn-0125 | gsm8k | full | decode | 0.000 | 0.015 | 0.415 | 0.415 |  | 0 | 0 |
| huginn-0125 | gsm8k | last | convert | 0.000 | 0.005 | 0.445 | 0.445 | +0.030 [+0.005, +0.060] | 0 | 0 |
| huginn-0125 | math500 | average | convert | 0.045 | 0.245 | 0.125 | 0.125 | -0.030 [-0.070, +0.010] | 0 | 9 |
| huginn-0125 | math500 | full | decode | 0.060 | 0.235 | 0.155 | 0.155 |  | 0 | 12 |
| huginn-0125 | math500 | last | convert | 0.065 | 0.260 | 0.165 | 0.165 | +0.010 [-0.015, +0.040] | 0 | 13 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | average | convert | 0-63 | 200 | -0.0365 [-0.0439, -0.0296] |
| huginn-0125 | gsm8k | average | convert | 64-127 | 130 | -0.0116 [-0.0188, -0.0046] |
| huginn-0125 | gsm8k | average | convert | 128-255 | 23 | -0.0003 [-0.0129, +0.0123] |
| huginn-0125 | gsm8k | average | convert | 256-511 | 1 | +0.0233 [+0.0233, +0.0233] |
| huginn-0125 | gsm8k | last | convert | 0-63 | 200 | +0.0020 [+0.0003, +0.0041] |
| huginn-0125 | gsm8k | last | convert | 64-127 | 130 | -0.0000 [-0.0032, +0.0033] |
| huginn-0125 | gsm8k | last | convert | 128-255 | 23 | +0.0015 [-0.0044, +0.0090] |
| huginn-0125 | gsm8k | last | convert | 256-511 | 1 | -0.0036 [-0.0036, -0.0036] |
| huginn-0125 | math500 | average | convert | 0-63 | 200 | +0.0071 [+0.0043, +0.0099] |
| huginn-0125 | math500 | average | convert | 64-127 | 192 | +0.0100 [+0.0069, +0.0131] |
| huginn-0125 | math500 | average | convert | 128-255 | 136 | +0.0075 [+0.0048, +0.0103] |
| huginn-0125 | math500 | average | convert | 256-511 | 67 | +0.0090 [+0.0060, +0.0122] |
| huginn-0125 | math500 | average | convert | 512-99999 | 18 | +0.0052 [-0.0025, +0.0129] |
| huginn-0125 | math500 | last | convert | 0-63 | 200 | -0.0001 [-0.0015, +0.0011] |
| huginn-0125 | math500 | last | convert | 64-127 | 192 | -0.0013 [-0.0026, -0.0002] |
| huginn-0125 | math500 | last | convert | 128-255 | 136 | +0.0001 [-0.0007, +0.0011] |
| huginn-0125 | math500 | last | convert | 256-511 | 67 | -0.0001 [-0.0007, +0.0006] |
| huginn-0125 | math500 | last | convert | 512-99999 | 18 | +0.0006 [-0.0018, +0.0039] |
