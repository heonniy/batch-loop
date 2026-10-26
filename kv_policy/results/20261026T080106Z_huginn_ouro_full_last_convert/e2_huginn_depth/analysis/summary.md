Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125-r16 | gsm8k | full | decode | 200 | 0.425 | 0.425 | 0.005 | 6 / 0 | 0.335 | 0.8812 | 0.8947 | 109.7 | 0.030 |
| huginn-0125-r16 | gsm8k | last | convert | 200 | 0.390 | 0.380 | 0.010 | 2 / 0 | 0.295 | 0.8892 | 0.9035 | 105.3 | 0.010 |
| huginn-0125-r16 | math500 | full | decode | 200 | 0.140 | 0.135 | 0.120 | 46 / 1 | 0.135 | 0.6374 | 0.6729 | 599.7 | 0.230 |
| huginn-0125-r16 | math500 | last | convert | 200 | 0.150 | 0.145 | 0.140 | 52 / 2 | 0.145 | 0.6317 | 0.6676 | 653.9 | 0.260 |
| huginn-0125-r4 | gsm8k | full | decode | 200 | 0.005 | 0.000 | 0.555 | 44 / 0 | 0.000 | 1.4543 | 1.4670 | 163.7 | 0.220 |
| huginn-0125-r4 | gsm8k | last | convert | 200 | 0.060 | 0.050 | 0.230 | 31 / 0 | 0.050 | 1.0784 | 1.0885 | 138.8 | 0.155 |
| huginn-0125-r4 | math500 | full | decode | 200 | 0.015 | 0.000 | 0.200 | 136 / 3 | 0.015 | 1.0698 | 1.1396 | 1437.1 | 0.680 |
| huginn-0125-r4 | math500 | last | convert | 200 | 0.020 | 0.010 | 0.225 | 108 / 0 | 0.020 | 0.8345 | 0.8904 | 1175.8 | 0.545 |
| huginn-0125-r8 | gsm8k | full | decode | 200 | 0.140 | 0.140 | 0.015 | 16 / 0 | 0.140 | 0.9516 | 0.9624 | 114.8 | 0.080 |
| huginn-0125-r8 | gsm8k | last | convert | 200 | 0.320 | 0.320 | 0.015 | 2 / 0 | 0.305 | 0.8861 | 0.8995 | 95.0 | 0.010 |
| huginn-0125-r8 | math500 | full | decode | 200 | 0.055 | 0.035 | 0.175 | 82 / 2 | 0.050 | 0.7318 | 0.7731 | 913.8 | 0.410 |
| huginn-0125-r8 | math500 | last | convert | 200 | 0.115 | 0.100 | 0.185 | 56 / 0 | 0.110 | 0.6596 | 0.7005 | 679.7 | 0.280 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125-r16 | gsm8k | full/decode | last | convert | -0.035 [-0.085, +0.015] | 16 / 9 | 0.23 | +0.0080 [+0.0039, +0.0123] |
| huginn-0125-r16 | math500 | full/decode | last | convert | +0.010 [-0.030, +0.050] | 7 / 9 | 0.804 | -0.0057 [-0.0073, -0.0041] |
| huginn-0125-r4 | gsm8k | full/decode | last | convert | +0.055 [+0.025, +0.090] | 1 / 12 | 0.00342 | -0.3759 [-0.3951, -0.3570] |
| huginn-0125-r4 | math500 | full/decode | last | convert | +0.005 [-0.020, +0.030] | 3 / 4 | 1 | -0.2353 [-0.2478, -0.2233] |
| huginn-0125-r8 | gsm8k | full/decode | last | convert | +0.180 [+0.115, +0.245] | 8 / 44 | 4.04e-07 | -0.0655 [-0.0743, -0.0569] |
| huginn-0125-r8 | math500 | full/decode | last | convert | +0.060 [+0.020, +0.100] | 3 / 15 | 0.00754 | -0.0723 [-0.0779, -0.0667] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125-r16 | gsm8k | full | decode | 0.000 | 0.030 | 0.425 | 0.425 |  | 0 | 0 |
| huginn-0125-r16 | gsm8k | last | convert | 0.000 | 0.010 | 0.390 | 0.390 | -0.035 [-0.085, +0.015] | 0 | 0 |
| huginn-0125-r16 | math500 | full | decode | 0.060 | 0.230 | 0.140 | 0.140 |  | 0 | 12 |
| huginn-0125-r16 | math500 | last | convert | 0.060 | 0.260 | 0.150 | 0.150 | +0.010 [-0.030, +0.050] | 0 | 12 |
| huginn-0125-r4 | gsm8k | full | decode | 0.005 | 0.220 | 0.005 | 0.005 |  | 0 | 1 |
| huginn-0125-r4 | gsm8k | last | convert | 0.000 | 0.155 | 0.060 | 0.060 | +0.055 [+0.020, +0.090] | 0 | 0 |
| huginn-0125-r4 | math500 | full | decode | 0.055 | 0.680 | 0.015 | 0.015 |  | 0 | 11 |
| huginn-0125-r4 | math500 | last | convert | 0.050 | 0.545 | 0.020 | 0.020 | +0.005 [-0.020, +0.030] | 0 | 10 |
| huginn-0125-r8 | gsm8k | full | decode | 0.000 | 0.080 | 0.140 | 0.140 |  | 0 | 0 |
| huginn-0125-r8 | gsm8k | last | convert | 0.000 | 0.010 | 0.320 | 0.320 | +0.180 [+0.115, +0.245] | 0 | 0 |
| huginn-0125-r8 | math500 | full | decode | 0.050 | 0.410 | 0.055 | 0.055 |  | 0 | 10 |
| huginn-0125-r8 | math500 | last | convert | 0.075 | 0.280 | 0.115 | 0.115 | +0.060 [+0.020, +0.105] | 0 | 15 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| huginn-0125-r16 | gsm8k | last | convert | 0-63 | 200 | +0.0128 [+0.0077, +0.0183] |
| huginn-0125-r16 | gsm8k | last | convert | 64-127 | 130 | -0.0029 [-0.0088, +0.0029] |
| huginn-0125-r16 | gsm8k | last | convert | 128-255 | 23 | +0.0005 [-0.0080, +0.0085] |
| huginn-0125-r16 | gsm8k | last | convert | 256-511 | 1 | -0.0237 [-0.0237, -0.0237] |
| huginn-0125-r16 | math500 | last | convert | 0-63 | 200 | -0.0048 [-0.0074, -0.0024] |
| huginn-0125-r16 | math500 | last | convert | 64-127 | 192 | -0.0081 [-0.0116, -0.0045] |
| huginn-0125-r16 | math500 | last | convert | 128-255 | 136 | -0.0050 [-0.0079, -0.0023] |
| huginn-0125-r16 | math500 | last | convert | 256-511 | 67 | -0.0061 [-0.0105, -0.0018] |
| huginn-0125-r16 | math500 | last | convert | 512-99999 | 18 | -0.0024 [-0.0084, +0.0045] |
| huginn-0125-r4 | gsm8k | last | convert | 0-63 | 200 | -0.3669 [-0.3861, -0.3470] |
| huginn-0125-r4 | gsm8k | last | convert | 64-127 | 130 | -0.3958 [-0.4269, -0.3660] |
| huginn-0125-r4 | gsm8k | last | convert | 128-255 | 23 | -0.3947 [-0.5293, -0.2581] |
| huginn-0125-r4 | gsm8k | last | convert | 256-511 | 1 | -0.2861 [-0.2861, -0.2861] |
| huginn-0125-r4 | math500 | last | convert | 0-63 | 200 | -0.2617 [-0.2792, -0.2450] |
| huginn-0125-r4 | math500 | last | convert | 64-127 | 192 | -0.2434 [-0.2630, -0.2240] |
| huginn-0125-r4 | math500 | last | convert | 128-255 | 136 | -0.2352 [-0.2538, -0.2168] |
| huginn-0125-r4 | math500 | last | convert | 256-511 | 67 | -0.1987 [-0.2189, -0.1792] |
| huginn-0125-r4 | math500 | last | convert | 512-99999 | 18 | -0.2028 [-0.2422, -0.1586] |
| huginn-0125-r8 | gsm8k | last | convert | 0-63 | 200 | -0.0569 [-0.0671, -0.0466] |
| huginn-0125-r8 | gsm8k | last | convert | 64-127 | 130 | -0.0793 [-0.0948, -0.0639] |
| huginn-0125-r8 | gsm8k | last | convert | 128-255 | 23 | -0.1188 [-0.2050, -0.0584] |
| huginn-0125-r8 | gsm8k | last | convert | 256-511 | 1 | -0.0181 [-0.0181, -0.0181] |
| huginn-0125-r8 | math500 | last | convert | 0-63 | 200 | -0.0764 [-0.0847, -0.0679] |
| huginn-0125-r8 | math500 | last | convert | 64-127 | 192 | -0.0852 [-0.0970, -0.0736] |
| huginn-0125-r8 | math500 | last | convert | 128-255 | 136 | -0.0631 [-0.0731, -0.0534] |
| huginn-0125-r8 | math500 | last | convert | 256-511 | 67 | -0.0688 [-0.0808, -0.0568] |
| huginn-0125-r8 | math500 | last | convert | 512-99999 | 18 | -0.0582 [-0.0780, -0.0418] |
