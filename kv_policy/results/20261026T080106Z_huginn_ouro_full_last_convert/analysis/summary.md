Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | full | decode | 200 | 0.835 | 0.830 | 0.015 | 1 / 0 | 0.600 | 0.7316 | 0.7445 | 111.3 | 0.005 |
| ouro-1.4b | gsm8k | last | convert | 200 | 0.000 | 0.000 | 0.010 | 177 / 18 | 0.000 | 3.4984 | 3.4630 | 473.4 | 0.895 |
| ouro-1.4b | math500 | full | decode | 200 | 0.710 | 0.710 | 0.000 | 4 / 0 | 0.675 | 0.5436 | 0.5910 | 538.7 | 0.020 |
| ouro-1.4b | math500 | last | convert | 200 | 0.015 | 0.015 | 0.040 | 127 / 49 | 0.015 | 2.6168 | 2.6649 | 1436.4 | 0.685 |
| ouro-2.6b | gsm8k | full | decode | 200 | 0.880 | 0.880 | 0.000 | 0 / 0 | 0.660 | 0.6958 | 0.7084 | 109.4 | 0.000 |
| ouro-2.6b | gsm8k | last | convert | 200 | 0.770 | 0.770 | 0.005 | 6 / 1 | 0.615 | 0.8645 | 0.8569 | 119.3 | 0.040 |
| ouro-2.6b | math500 | full | decode | 200 | 0.560 | 0.560 | 0.000 | 18 / 0 | 0.520 | 0.4649 | 0.5029 | 423.9 | 0.090 |
| ouro-2.6b | math500 | last | convert | 200 | 0.035 | 0.035 | 0.025 | 163 / 8 | 0.035 | 2.1549 | 2.0150 | 1788.2 | 0.865 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | full/decode | last | convert | -0.835 [-0.885, -0.780] | 167 / 0 | 1.07e-50 | +2.7668 [+2.6796, +2.8524] |
| ouro-1.4b | math500 | full/decode | last | convert | -0.695 [-0.760, -0.630] | 139 / 0 | 2.87e-42 | +2.0732 [+1.9751, +2.1760] |
| ouro-2.6b | gsm8k | full/decode | last | convert | -0.110 [-0.165, -0.055] | 29 / 7 | 0.000313 | +0.1687 [+0.1375, +0.2038] |
| ouro-2.6b | math500 | full/decode | last | convert | -0.525 [-0.595, -0.455] | 105 / 0 | 4.93e-32 | +1.6901 [+1.5386, +1.8430] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | full | decode | 0.000 | 0.005 | 0.835 | 0.835 |  | 0 | 0 |
| ouro-1.4b | gsm8k | last | convert | 0.145 | 0.895 | 0.000 | 0.000 | -0.835 [-0.885, -0.785] | 0 | 29 |
| ouro-1.4b | math500 | full | decode | 0.005 | 0.020 | 0.710 | 0.710 |  | 0 | 1 |
| ouro-1.4b | math500 | last | convert | 0.480 | 0.685 | 0.015 | 0.010 | -0.700 [-0.760, -0.635] | 0 | 96 |
| ouro-2.6b | gsm8k | full | decode | 0.000 | 0.000 | 0.880 | 0.880 |  | 0 | 0 |
| ouro-2.6b | gsm8k | last | convert | 0.000 | 0.040 | 0.770 | 0.770 | -0.110 [-0.170, -0.055] | 0 | 0 |
| ouro-2.6b | math500 | full | decode | 0.015 | 0.090 | 0.560 | 0.560 |  | 0 | 3 |
| ouro-2.6b | math500 | last | convert | 0.260 | 0.865 | 0.035 | 0.040 | -0.520 [-0.590, -0.450] | 1 | 51 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | last | convert | 0-63 | 200 | +2.6253 [+2.5377, +2.7121] |
| ouro-1.4b | gsm8k | last | convert | 64-127 | 143 | +2.9662 [+2.8479, +3.0832] |
| ouro-1.4b | gsm8k | last | convert | 128-255 | 37 | +3.2627 [+2.9980, +3.5370] |
| ouro-1.4b | gsm8k | last | convert | 256-511 | 1 | +3.4124 [+3.4124, +3.4124] |
| ouro-1.4b | math500 | last | convert | 0-63 | 200 | +2.1830 [+2.0620, +2.3051] |
| ouro-1.4b | math500 | last | convert | 64-127 | 193 | +2.0739 [+1.9555, +2.1953] |
| ouro-1.4b | math500 | last | convert | 128-255 | 144 | +2.0355 [+1.8956, +2.1731] |
| ouro-1.4b | math500 | last | convert | 256-511 | 70 | +2.0536 [+1.8721, +2.2431] |
| ouro-1.4b | math500 | last | convert | 512-99999 | 21 | +1.8886 [+1.7161, +2.1067] |
| ouro-2.6b | gsm8k | last | convert | 0-63 | 200 | +0.1557 [+0.1228, +0.1929] |
| ouro-2.6b | gsm8k | last | convert | 64-127 | 143 | +0.2012 [+0.1581, +0.2503] |
| ouro-2.6b | gsm8k | last | convert | 128-255 | 37 | +0.1428 [+0.0902, +0.2024] |
| ouro-2.6b | gsm8k | last | convert | 256-511 | 1 | -0.0020 [-0.0020, -0.0020] |
| ouro-2.6b | math500 | last | convert | 0-63 | 200 | +1.4873 [+1.3429, +1.6326] |
| ouro-2.6b | math500 | last | convert | 64-127 | 193 | +1.5750 [+1.4242, +1.7282] |
| ouro-2.6b | math500 | last | convert | 128-255 | 144 | +1.7956 [+1.5981, +2.0119] |
| ouro-2.6b | math500 | last | convert | 256-511 | 70 | +1.8955 [+1.6053, +2.1914] |
| ouro-2.6b | math500 | last | convert | 512-99999 | 21 | +1.8026 [+1.2841, +2.3673] |
