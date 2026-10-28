Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | full | decode | 200 | 0.835 | 0.830 | 0.015 | 1 / 0 | 0.600 | 0.7316 | 0.7445 | 111.3 | 0.005 |
| ouro-1.4b | gsm8k | last | strict | 200 | 0.000 | 0.000 | 0.005 | 199 / 0 | 0.000 | 6.0684 | 6.0277 | 510.7 | 0.995 |
| ouro-1.4b | gsm8k | first | strict | 200 | 0.210 | 0.195 | 0.095 | 12 / 1 | 0.185 | 1.2110 | 1.2086 | 124.5 | 0.070 |
| ouro-1.4b | gsm8k | full | strict | 200 | 0.810 | 0.805 | 0.015 | 0 / 0 | 0.565 | 0.7318 | 0.7448 | 108.8 | 0.000 |
| ouro-1.4b | gsm8k | bankP0 | strict | 200 | 0.405 | 0.405 | 0.040 | 2 / 0 | 0.315 | 0.8536 | 0.8635 | 100.2 | 0.010 |
| ouro-1.4b | gsm8k | bankR1all | strict | 200 | 0.720 | 0.720 | 0.015 | 2 / 0 | 0.530 | 0.7504 | 0.7636 | 113.0 | 0.010 |
| ouro-1.4b | gsm8k | bankcal | strict | 200 | 0.660 | 0.660 | 0.015 | 0 / 0 | 0.485 | 0.7731 | 0.7866 | 106.7 | 0.000 |
| ouro-1.4b | gsm8k | bankrand0 | strict | 200 | 0.000 | 0.000 | 0.010 | 198 / 0 | 0.000 | 6.1074 | 6.0750 | 511.2 | 0.990 |
| ouro-1.4b | math500 | full | decode | 200 | 0.710 | 0.710 | 0.000 | 4 / 0 | 0.675 | 0.5436 | 0.5910 | 538.7 | 0.020 |
| ouro-1.4b | math500 | last | strict | 200 | 0.000 | 0.000 | 0.000 | 192 / 8 | 0.000 | 5.3896 | 5.4156 | 1998.7 | 0.960 |
| ouro-1.4b | math500 | first | strict | 200 | 0.050 | 0.050 | 0.000 | 141 / 5 | 0.050 | 1.2056 | 1.2577 | 1646.9 | 0.735 |
| ouro-1.4b | math500 | full | strict | 200 | 0.740 | 0.740 | 0.000 | 4 / 0 | 0.700 | 0.5433 | 0.5907 | 545.1 | 0.020 |
| ouro-1.4b | math500 | bankP0 | strict | 200 | 0.170 | 0.170 | 0.000 | 72 / 0 | 0.160 | 0.7640 | 0.7821 | 844.7 | 0.360 |
| ouro-1.4b | math500 | bankR1all | strict | 200 | 0.610 | 0.610 | 0.000 | 4 / 0 | 0.580 | 0.5607 | 0.6077 | 569.1 | 0.020 |
| ouro-1.4b | math500 | bankcal | strict | 200 | 0.485 | 0.485 | 0.000 | 32 / 0 | 0.420 | 0.5693 | 0.6112 | 820.3 | 0.220 |
| ouro-1.4b | math500 | bankrand0 | strict | 200 | 0.000 | 0.000 | 0.000 | 189 / 11 | 0.000 | 5.5156 | 5.5719 | 1981.6 | 0.945 |
| ouro-2.6b | gsm8k | full | decode | 200 | 0.880 | 0.880 | 0.000 | 0 / 0 | 0.660 | 0.6958 | 0.7084 | 109.4 | 0.000 |
| ouro-2.6b | gsm8k | last | strict | 200 | 0.600 | 0.600 | 0.000 | 3 / 0 | 0.500 | 0.9188 | 0.9221 | 112.8 | 0.025 |
| ouro-2.6b | gsm8k | first | strict | 200 | 0.090 | 0.055 | 0.575 | 28 / 4 | 0.050 | 1.6057 | 1.6346 | 172.6 | 0.145 |
| ouro-2.6b | gsm8k | full | strict | 200 | 0.880 | 0.880 | 0.000 | 0 / 0 | 0.655 | 0.6959 | 0.7086 | 109.1 | 0.000 |
| ouro-2.6b | gsm8k | bankP0 | strict | 200 | 0.770 | 0.770 | 0.000 | 1 / 1 | 0.670 | 0.7256 | 0.7332 | 110.0 | 0.005 |
| ouro-2.6b | gsm8k | bankR1all | strict | 200 | 0.845 | 0.845 | 0.000 | 0 / 0 | 0.635 | 0.7002 | 0.7119 | 107.9 | 0.000 |
| ouro-2.6b | gsm8k | bankrand0 | strict | 200 | 0.555 | 0.555 | 0.005 | 4 / 1 | 0.470 | 0.9552 | 0.9593 | 114.1 | 0.025 |
| ouro-2.6b | math500 | full | decode | 200 | 0.560 | 0.560 | 0.000 | 18 / 0 | 0.520 | 0.4649 | 0.5029 | 423.9 | 0.090 |
| ouro-2.6b | math500 | last | strict | 200 | 0.330 | 0.330 | 0.000 | 32 / 1 | 0.305 | 0.8114 | 0.8105 | 468.1 | 0.165 |
| ouro-2.6b | math500 | first | strict | 200 | 0.075 | 0.075 | 0.005 | 82 / 1 | 0.075 | 1.3985 | 1.4273 | 1579.8 | 0.655 |
| ouro-2.6b | math500 | full | strict | 200 | 0.565 | 0.565 | 0.000 | 19 / 0 | 0.520 | 0.4650 | 0.5031 | 427.5 | 0.095 |
| ouro-2.6b | math500 | bankP0 | strict | 200 | 0.410 | 0.410 | 0.000 | 11 / 0 | 0.395 | 0.5514 | 0.5762 | 292.2 | 0.055 |
| ouro-2.6b | math500 | bankR1all | strict | 200 | 0.500 | 0.500 | 0.000 | 27 / 1 | 0.465 | 0.4751 | 0.5126 | 523.8 | 0.135 |
| ouro-2.6b | math500 | bankrand0 | strict | 200 | 0.330 | 0.330 | 0.000 | 19 / 1 | 0.315 | 0.7803 | 0.7811 | 373.2 | 0.110 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | full/decode | last | strict | -0.835 [-0.885, -0.780] | 167 / 0 | 1.07e-50 | +5.3368 [+5.2271, +5.4476] |
| ouro-1.4b | gsm8k | full/decode | first | strict | -0.625 [-0.695, -0.550] | 129 / 4 | 2.36e-33 | +0.4794 [+0.4564, +0.5028] |
| ouro-1.4b | gsm8k | full/decode | full | strict | -0.025 [-0.055, +0.005] | 7 / 2 | 0.18 | +0.0003 [-0.0002, +0.0007] |
| ouro-1.4b | gsm8k | full/decode | bankP0 | strict | -0.430 [-0.500, -0.360] | 88 / 2 | 6.62e-24 | +0.1220 [+0.1107, +0.1343] |
| ouro-1.4b | gsm8k | full/decode | bankR1all | strict | -0.115 [-0.170, -0.060] | 28 / 5 | 6.62e-05 | +0.0188 [+0.0149, +0.0230] |
| ouro-1.4b | gsm8k | full/decode | bankcal | strict | -0.175 [-0.235, -0.120] | 38 / 3 | 1.05e-08 | +0.0415 [+0.0352, +0.0480] |
| ouro-1.4b | gsm8k | full/decode | bankrand0 | strict | -0.835 [-0.885, -0.780] | 167 / 0 | 1.07e-50 | +5.3759 [+5.2818, +5.4709] |
| ouro-1.4b | gsm8k | full/strict | last | strict | -0.810 [-0.860, -0.755] | 162 / 0 | 3.42e-49 | +5.3365 [+5.2302, +5.4482] |
| ouro-1.4b | gsm8k | full/strict | first | strict | -0.600 [-0.670, -0.525] | 124 / 4 | 6.48e-32 | +0.4792 [+0.4557, +0.5022] |
| ouro-1.4b | gsm8k | full/strict | bankP0 | strict | -0.405 [-0.480, -0.335] | 84 / 3 | 1.42e-21 | +0.1217 [+0.1105, +0.1336] |
| ouro-1.4b | gsm8k | full/strict | bankR1all | strict | -0.090 [-0.150, -0.035] | 27 / 9 | 0.00393 | +0.0185 [+0.0144, +0.0228] |
| ouro-1.4b | gsm8k | full/strict | bankcal | strict | -0.150 [-0.210, -0.090] | 36 / 6 | 2.83e-06 | +0.0412 [+0.0349, +0.0477] |
| ouro-1.4b | gsm8k | full/strict | bankrand0 | strict | -0.810 [-0.865, -0.755] | 162 / 0 | 3.42e-49 | +5.3756 [+5.2824, +5.4694] |
| ouro-1.4b | gsm8k | last/strict | bankP0 | strict | +0.405 [+0.340, +0.475] | 0 / 81 | 8.27e-25 | -5.2148 [-5.3244, -5.1067] |
| ouro-1.4b | gsm8k | last/strict | bankR1all | strict | +0.720 [+0.660, +0.780] | 0 / 144 | 8.97e-44 | -5.3180 [-5.4291, -5.2113] |
| ouro-1.4b | gsm8k | last/strict | bankcal | strict | +0.660 [+0.595, +0.725] | 0 / 132 | 3.67e-40 | -5.2953 [-5.4057, -5.1857] |
| ouro-1.4b | gsm8k | last/strict | bankrand0 | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | +0.0391 [-0.0389, +0.1107] |
| ouro-1.4b | math500 | full/decode | last | strict | -0.710 [-0.770, -0.645] | 142 / 0 | 3.59e-43 | +4.8460 [+4.7178, +4.9704] |
| ouro-1.4b | math500 | full/decode | first | strict | -0.660 [-0.725, -0.595] | 132 / 0 | 3.67e-40 | +0.6620 [+0.6338, +0.6916] |
| ouro-1.4b | math500 | full/decode | full | strict | +0.030 [-0.010, +0.070] | 5 / 11 | 0.21 | -0.0003 [-0.0006, -0.0001] |
| ouro-1.4b | math500 | full/decode | bankP0 | strict | -0.540 [-0.615, -0.465] | 112 / 4 | 1.79e-28 | +0.2203 [+0.2033, +0.2368] |
| ouro-1.4b | math500 | full/decode | bankR1all | strict | -0.100 [-0.155, -0.045] | 27 / 7 | 0.000821 | +0.0171 [+0.0139, +0.0205] |
| ouro-1.4b | math500 | full/decode | bankcal | strict | -0.225 [-0.295, -0.155] | 53 / 8 | 2.99e-09 | +0.0257 [+0.0158, +0.0368] |
| ouro-1.4b | math500 | full/decode | bankrand0 | strict | -0.710 [-0.770, -0.650] | 142 / 0 | 3.59e-43 | +4.9720 [+4.8423, +5.1025] |
| ouro-1.4b | math500 | full/strict | last | strict | -0.740 [-0.800, -0.680] | 148 / 0 | 5.61e-45 | +4.8463 [+4.7187, +4.9722] |
| ouro-1.4b | math500 | full/strict | first | strict | -0.690 [-0.750, -0.625] | 138 / 0 | 5.74e-42 | +0.6623 [+0.6340, +0.6928] |
| ouro-1.4b | math500 | full/strict | bankP0 | strict | -0.570 [-0.645, -0.495] | 117 / 3 | 4.33e-31 | +0.2207 [+0.2039, +0.2371] |
| ouro-1.4b | math500 | full/strict | bankR1all | strict | -0.130 [-0.185, -0.075] | 31 / 5 | 1.29e-05 | +0.0174 [+0.0142, +0.0208] |
| ouro-1.4b | math500 | full/strict | bankcal | strict | -0.255 [-0.325, -0.185] | 57 / 6 | 1.64e-11 | +0.0260 [+0.0162, +0.0370] |
| ouro-1.4b | math500 | full/strict | bankrand0 | strict | -0.740 [-0.800, -0.680] | 148 / 0 | 5.61e-45 | +4.9724 [+4.8446, +5.0994] |
| ouro-1.4b | math500 | last/strict | bankP0 | strict | +0.170 [+0.120, +0.225] | 0 / 34 | 1.16e-10 | -4.6256 [-4.7508, -4.4995] |
| ouro-1.4b | math500 | last/strict | bankR1all | strict | +0.610 [+0.545, +0.675] | 0 / 122 | 3.76e-37 | -4.8289 [-4.9549, -4.7035] |
| ouro-1.4b | math500 | last/strict | bankcal | strict | +0.485 [+0.415, +0.555] | 0 / 97 | 1.26e-29 | -4.8203 [-4.9488, -4.6968] |
| ouro-1.4b | math500 | last/strict | bankrand0 | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | +0.1261 [+0.1049, +0.1477] |
| ouro-2.6b | gsm8k | full/decode | last | strict | -0.280 [-0.350, -0.215] | 59 / 3 | 1.72e-14 | +0.2230 [+0.1930, +0.2557] |
| ouro-2.6b | gsm8k | full/decode | first | strict | -0.790 [-0.850, -0.730] | 160 / 2 | 4.52e-45 | +0.9099 [+0.8714, +0.9487] |
| ouro-2.6b | gsm8k | full/decode | full | strict | +0.000 [-0.015, +0.015] | 1 / 1 | 1 | +0.0001 [-0.0003, +0.0005] |
| ouro-2.6b | gsm8k | full/decode | bankP0 | strict | -0.110 [-0.165, -0.055] | 28 / 6 | 0.000195 | +0.0298 [+0.0220, +0.0379] |
| ouro-2.6b | gsm8k | full/decode | bankR1all | strict | -0.035 [-0.075, +0.005] | 12 / 5 | 0.143 | +0.0044 [+0.0017, +0.0071] |
| ouro-2.6b | gsm8k | full/decode | bankrand0 | strict | -0.325 [-0.395, -0.260] | 67 / 2 | 8.19e-18 | +0.2593 [+0.2303, +0.2903] |
| ouro-2.6b | gsm8k | full/strict | last | strict | -0.280 [-0.350, -0.215] | 59 / 3 | 1.72e-14 | +0.2229 [+0.1920, +0.2555] |
| ouro-2.6b | gsm8k | full/strict | first | strict | -0.790 [-0.850, -0.730] | 160 / 2 | 4.52e-45 | +0.9098 [+0.8712, +0.9490] |
| ouro-2.6b | gsm8k | full/strict | bankP0 | strict | -0.110 [-0.165, -0.055] | 27 / 5 | 0.000113 | +0.0297 [+0.0217, +0.0379] |
| ouro-2.6b | gsm8k | full/strict | bankR1all | strict | -0.035 [-0.075, +0.005] | 12 / 5 | 0.143 | +0.0043 [+0.0015, +0.0071] |
| ouro-2.6b | gsm8k | full/strict | bankrand0 | strict | -0.325 [-0.395, -0.255] | 67 / 2 | 8.19e-18 | +0.2592 [+0.2304, +0.2904] |
| ouro-2.6b | gsm8k | last/strict | bankP0 | strict | +0.170 [+0.100, +0.245] | 13 / 47 | 1.21e-05 | -0.1932 [-0.2237, -0.1657] |
| ouro-2.6b | gsm8k | last/strict | bankR1all | strict | +0.245 [+0.180, +0.315] | 6 / 55 | 5.38e-11 | -0.2186 [-0.2512, -0.1887] |
| ouro-2.6b | gsm8k | last/strict | bankrand0 | strict | -0.045 [-0.110, +0.020] | 27 / 18 | 0.233 | +0.0363 [+0.0161, +0.0570] |
| ouro-2.6b | math500 | full/decode | last | strict | -0.230 [-0.305, -0.155] | 59 / 13 | 3.81e-08 | +0.3465 [+0.3102, +0.3849] |
| ouro-2.6b | math500 | full/decode | first | strict | -0.485 [-0.560, -0.410] | 100 / 3 | 3.59e-26 | +0.9336 [+0.9001, +0.9673] |
| ouro-2.6b | math500 | full/decode | full | strict | +0.005 [-0.030, +0.040] | 6 / 7 | 1 | +0.0001 [-0.0001, +0.0004] |
| ouro-2.6b | math500 | full/decode | bankP0 | strict | -0.150 [-0.220, -0.075] | 45 / 15 | 0.000135 | +0.0865 [+0.0760, +0.0982] |
| ouro-2.6b | math500 | full/decode | bankR1all | strict | -0.060 [-0.130, +0.010] | 32 / 20 | 0.126 | +0.0102 [+0.0084, +0.0121] |
| ouro-2.6b | math500 | full/decode | bankrand0 | strict | -0.230 [-0.310, -0.150] | 62 / 16 | 1.51e-07 | +0.3154 [+0.2818, +0.3521] |
| ouro-2.6b | math500 | full/strict | last | strict | -0.235 [-0.310, -0.160] | 60 / 13 | 2.31e-08 | +0.3464 [+0.3098, +0.3857] |
| ouro-2.6b | math500 | full/strict | first | strict | -0.490 [-0.560, -0.420] | 100 / 2 | 2.07e-27 | +0.9335 [+0.8997, +0.9676] |
| ouro-2.6b | math500 | full/strict | bankP0 | strict | -0.155 [-0.230, -0.080] | 45 / 14 | 6.53e-05 | +0.0864 [+0.0758, +0.0983] |
| ouro-2.6b | math500 | full/strict | bankR1all | strict | -0.065 [-0.135, +0.005] | 33 / 20 | 0.0984 | +0.0101 [+0.0083, +0.0120] |
| ouro-2.6b | math500 | full/strict | bankrand0 | strict | -0.235 [-0.315, -0.155] | 63 / 16 | 9.44e-08 | +0.3153 [+0.2810, +0.3515] |
| ouro-2.6b | math500 | last/strict | bankP0 | strict | +0.080 [+0.015, +0.145] | 15 / 31 | 0.0259 | -0.2600 [-0.2922, -0.2289] |
| ouro-2.6b | math500 | last/strict | bankR1all | strict | +0.170 [+0.100, +0.240] | 11 / 45 | 5.38e-06 | -0.3363 [-0.3752, -0.3002] |
| ouro-2.6b | math500 | last/strict | bankrand0 | strict | +0.000 [-0.060, +0.060] | 20 / 20 | 1 | -0.0311 [-0.0455, -0.0173] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | bankP0 | strict | 0.000 | 0.010 | 0.405 | 0.405 | -0.430 [-0.500, -0.360] | 0 | 0 |
| ouro-1.4b | gsm8k | bankR1all | strict | 0.000 | 0.010 | 0.720 | 0.720 | -0.115 [-0.170, -0.060] | 0 | 0 |
| ouro-1.4b | gsm8k | bankcal | strict | 0.000 | 0.000 | 0.660 | 0.660 | -0.175 [-0.235, -0.120] | 0 | 0 |
| ouro-1.4b | gsm8k | bankrand0 | strict | 0.135 | 0.990 | 0.000 | 0.005 | -0.830 [-0.880, -0.775] | 1 | 26 |
| ouro-1.4b | gsm8k | first | strict | 0.000 | 0.070 | 0.210 | 0.210 | -0.625 [-0.695, -0.550] | 0 | 0 |
| ouro-1.4b | gsm8k | full | decode | 0.000 | 0.005 | 0.835 | 0.835 |  | 0 | 0 |
| ouro-1.4b | gsm8k | full | strict | 0.000 | 0.000 | 0.810 | 0.810 | -0.025 [-0.055, +0.005] | 0 | 0 |
| ouro-1.4b | gsm8k | last | strict | 0.335 | 0.995 | 0.000 | 0.000 | -0.835 [-0.885, -0.780] | 0 | 67 |
| ouro-1.4b | math500 | bankP0 | strict | 0.120 | 0.360 | 0.170 | 0.170 | -0.540 [-0.615, -0.465] | 0 | 24 |
| ouro-1.4b | math500 | bankR1all | strict | 0.000 | 0.020 | 0.610 | 0.610 | -0.100 [-0.155, -0.045] | 0 | 0 |
| ouro-1.4b | math500 | bankcal | strict | 0.100 | 0.220 | 0.485 | 0.300 | -0.410 [-0.485, -0.335] | 7 | 13 |
| ouro-1.4b | math500 | bankrand0 | strict | 0.100 | 0.945 | 0.000 | 0.000 | -0.710 [-0.775, -0.645] | 0 | 20 |
| ouro-1.4b | math500 | first | strict | 0.045 | 0.735 | 0.050 | 0.045 | -0.665 [-0.730, -0.600] | 0 | 9 |
| ouro-1.4b | math500 | full | decode | 0.005 | 0.020 | 0.710 | 0.710 |  | 0 | 1 |
| ouro-1.4b | math500 | full | strict | 0.005 | 0.020 | 0.740 | 0.740 | +0.030 [-0.010, +0.070] | 0 | 1 |
| ouro-1.4b | math500 | last | strict | 0.175 | 0.960 | 0.000 | 0.000 | -0.710 [-0.775, -0.650] | 0 | 35 |
| ouro-2.6b | gsm8k | bankP0 | strict | 0.000 | 0.005 | 0.770 | 0.770 | -0.110 [-0.165, -0.055] | 0 | 0 |
| ouro-2.6b | gsm8k | bankR1all | strict | 0.000 | 0.000 | 0.845 | 0.845 | -0.035 [-0.075, +0.005] | 0 | 0 |
| ouro-2.6b | gsm8k | bankrand0 | strict | 0.000 | 0.025 | 0.555 | 0.555 | -0.325 [-0.390, -0.255] | 0 | 0 |
| ouro-2.6b | gsm8k | first | strict | 0.010 | 0.145 | 0.090 | 0.090 | -0.790 [-0.845, -0.730] | 0 | 2 |
| ouro-2.6b | gsm8k | full | decode | 0.000 | 0.000 | 0.880 | 0.880 |  | 0 | 0 |
| ouro-2.6b | gsm8k | full | strict | 0.000 | 0.000 | 0.880 | 0.880 | +0.000 [-0.015, +0.015] | 0 | 0 |
| ouro-2.6b | gsm8k | last | strict | 0.000 | 0.025 | 0.600 | 0.600 | -0.280 [-0.345, -0.215] | 0 | 0 |
| ouro-2.6b | math500 | bankP0 | strict | 0.010 | 0.055 | 0.410 | 0.400 | -0.160 [-0.235, -0.085] | 0 | 2 |
| ouro-2.6b | math500 | bankR1all | strict | 0.005 | 0.135 | 0.500 | 0.500 | -0.060 [-0.130, +0.010] | 0 | 1 |
| ouro-2.6b | math500 | bankrand0 | strict | 0.020 | 0.110 | 0.330 | 0.330 | -0.230 [-0.310, -0.150] | 0 | 4 |
| ouro-2.6b | math500 | first | strict | 0.000 | 0.655 | 0.075 | 0.075 | -0.485 [-0.560, -0.410] | 0 | 0 |
| ouro-2.6b | math500 | full | decode | 0.015 | 0.090 | 0.560 | 0.560 |  | 0 | 3 |
| ouro-2.6b | math500 | full | strict | 0.015 | 0.095 | 0.565 | 0.565 | +0.005 [-0.030, +0.040] | 0 | 3 |
| ouro-2.6b | math500 | last | strict | 0.035 | 0.165 | 0.330 | 0.325 | -0.235 [-0.310, -0.160] | 0 | 7 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | bankP0 | strict | 0-63 | 200 | +0.1219 [+0.1078, +0.1366] |
| ouro-1.4b | gsm8k | bankP0 | strict | 64-127 | 143 | +0.1249 [+0.1075, +0.1422] |
| ouro-1.4b | gsm8k | bankP0 | strict | 128-255 | 37 | +0.1101 [+0.0799, +0.1427] |
| ouro-1.4b | gsm8k | bankP0 | strict | 256-511 | 1 | +0.0067 [+0.0067, +0.0067] |
| ouro-1.4b | gsm8k | bankR1all | strict | 0-63 | 200 | +0.0186 [+0.0137, +0.0233] |
| ouro-1.4b | gsm8k | bankR1all | strict | 64-127 | 143 | +0.0215 [+0.0138, +0.0298] |
| ouro-1.4b | gsm8k | bankR1all | strict | 128-255 | 37 | +0.0079 [-0.0044, +0.0190] |
| ouro-1.4b | gsm8k | bankR1all | strict | 256-511 | 1 | -0.0359 [-0.0359, -0.0359] |
| ouro-1.4b | gsm8k | bankcal | strict | 0-63 | 200 | +0.0426 [+0.0358, +0.0497] |
| ouro-1.4b | gsm8k | bankcal | strict | 64-127 | 143 | +0.0402 [+0.0280, +0.0538] |
| ouro-1.4b | gsm8k | bankcal | strict | 128-255 | 37 | +0.0364 [+0.0064, +0.0635] |
| ouro-1.4b | gsm8k | bankcal | strict | 256-511 | 1 | +0.0400 [+0.0400, +0.0400] |
| ouro-1.4b | gsm8k | bankrand0 | strict | 0-63 | 200 | +5.3437 [+5.2480, +5.4369] |
| ouro-1.4b | gsm8k | bankrand0 | strict | 64-127 | 143 | +5.4357 [+5.2928, +5.5774] |
| ouro-1.4b | gsm8k | bankrand0 | strict | 128-255 | 37 | +5.4543 [+5.0989, +5.7796] |
| ouro-1.4b | gsm8k | bankrand0 | strict | 256-511 | 1 | +4.0058 [+4.0058, +4.0058] |
| ouro-1.4b | gsm8k | first | strict | 0-63 | 200 | +0.4791 [+0.4503, +0.5088] |
| ouro-1.4b | gsm8k | first | strict | 64-127 | 143 | +0.4897 [+0.4543, +0.5253] |
| ouro-1.4b | gsm8k | first | strict | 128-255 | 37 | +0.4282 [+0.3639, +0.4956] |
| ouro-1.4b | gsm8k | first | strict | 256-511 | 1 | +0.4387 [+0.4387, +0.4387] |
| ouro-1.4b | gsm8k | full | strict | 0-63 | 200 | +0.0003 [-0.0003, +0.0010] |
| ouro-1.4b | gsm8k | full | strict | 64-127 | 143 | +0.0000 [-0.0006, +0.0007] |
| ouro-1.4b | gsm8k | full | strict | 128-255 | 37 | +0.0008 [-0.0009, +0.0026] |
| ouro-1.4b | gsm8k | full | strict | 256-511 | 1 | -0.0070 [-0.0070, -0.0070] |
| ouro-1.4b | gsm8k | last | strict | 0-63 | 200 | +5.2937 [+5.1928, +5.3957] |
| ouro-1.4b | gsm8k | last | strict | 64-127 | 143 | +5.3761 [+5.2238, +5.5306] |
| ouro-1.4b | gsm8k | last | strict | 128-255 | 37 | +5.5926 [+5.0983, +6.0611] |
| ouro-1.4b | gsm8k | last | strict | 256-511 | 1 | +5.9722 [+5.9722, +5.9722] |
| ouro-1.4b | math500 | bankP0 | strict | 0-63 | 200 | +0.1403 [+0.1219, +0.1598] |
| ouro-1.4b | math500 | bankP0 | strict | 64-127 | 193 | +0.2429 [+0.2221, +0.2649] |
| ouro-1.4b | math500 | bankP0 | strict | 128-255 | 144 | +0.2393 [+0.2129, +0.2661] |
| ouro-1.4b | math500 | bankP0 | strict | 256-511 | 70 | +0.2675 [+0.2320, +0.3047] |
| ouro-1.4b | math500 | bankP0 | strict | 512-99999 | 21 | +0.2385 [+0.1919, +0.2940] |
| ouro-1.4b | math500 | bankR1all | strict | 0-63 | 200 | +0.0108 [+0.0051, +0.0165] |
| ouro-1.4b | math500 | bankR1all | strict | 64-127 | 193 | +0.0166 [+0.0111, +0.0220] |
| ouro-1.4b | math500 | bankR1all | strict | 128-255 | 144 | +0.0198 [+0.0146, +0.0252] |
| ouro-1.4b | math500 | bankR1all | strict | 256-511 | 70 | +0.0208 [+0.0131, +0.0296] |
| ouro-1.4b | math500 | bankR1all | strict | 512-99999 | 21 | +0.0209 [+0.0109, +0.0346] |
| ouro-1.4b | math500 | bankcal | strict | 0-63 | 200 | -0.0449 [-0.0577, -0.0321] |
| ouro-1.4b | math500 | bankcal | strict | 64-127 | 193 | +0.0363 [+0.0240, +0.0497] |
| ouro-1.4b | math500 | bankcal | strict | 128-255 | 144 | +0.0522 [+0.0385, +0.0678] |
| ouro-1.4b | math500 | bankcal | strict | 256-511 | 70 | +0.0627 [+0.0361, +0.0973] |
| ouro-1.4b | math500 | bankcal | strict | 512-99999 | 21 | +0.0442 [+0.0183, +0.0823] |
| ouro-1.4b | math500 | bankrand0 | strict | 0-63 | 200 | +5.0922 [+4.9680, +5.2173] |
| ouro-1.4b | math500 | bankrand0 | strict | 64-127 | 193 | +5.0442 [+4.9015, +5.1829] |
| ouro-1.4b | math500 | bankrand0 | strict | 128-255 | 144 | +5.0237 [+4.8714, +5.1769] |
| ouro-1.4b | math500 | bankrand0 | strict | 256-511 | 70 | +4.8656 [+4.6327, +5.1081] |
| ouro-1.4b | math500 | bankrand0 | strict | 512-99999 | 21 | +4.4517 [+4.0084, +4.9658] |
| ouro-1.4b | math500 | first | strict | 0-63 | 200 | +0.5232 [+0.4937, +0.5530] |
| ouro-1.4b | math500 | first | strict | 64-127 | 193 | +0.6676 [+0.6297, +0.7075] |
| ouro-1.4b | math500 | first | strict | 128-255 | 144 | +0.6922 [+0.6507, +0.7362] |
| ouro-1.4b | math500 | first | strict | 256-511 | 70 | +0.7485 [+0.6838, +0.8236] |
| ouro-1.4b | math500 | first | strict | 512-99999 | 21 | +0.7850 [+0.6700, +0.9263] |
| ouro-1.4b | math500 | full | strict | 0-63 | 200 | -0.0007 [-0.0014, -0.0001] |
| ouro-1.4b | math500 | full | strict | 64-127 | 193 | -0.0002 [-0.0009, +0.0004] |
| ouro-1.4b | math500 | full | strict | 128-255 | 144 | +0.0001 [-0.0004, +0.0006] |
| ouro-1.4b | math500 | full | strict | 256-511 | 70 | -0.0005 [-0.0011, +0.0001] |
| ouro-1.4b | math500 | full | strict | 512-99999 | 21 | -0.0004 [-0.0013, +0.0006] |
| ouro-1.4b | math500 | last | strict | 0-63 | 200 | +4.9029 [+4.7882, +5.0202] |
| ouro-1.4b | math500 | last | strict | 64-127 | 193 | +4.8980 [+4.7628, +5.0290] |
| ouro-1.4b | math500 | last | strict | 128-255 | 144 | +4.9069 [+4.7541, +5.0561] |
| ouro-1.4b | math500 | last | strict | 256-511 | 70 | +4.7798 [+4.5410, +5.0250] |
| ouro-1.4b | math500 | last | strict | 512-99999 | 21 | +4.4593 [+4.0071, +4.9868] |
| ouro-2.6b | gsm8k | bankP0 | strict | 0-63 | 200 | +0.0252 [+0.0163, +0.0344] |
| ouro-2.6b | gsm8k | bankP0 | strict | 64-127 | 143 | +0.0416 [+0.0281, +0.0568] |
| ouro-2.6b | gsm8k | bankP0 | strict | 128-255 | 37 | +0.0175 [-0.0067, +0.0459] |
| ouro-2.6b | gsm8k | bankP0 | strict | 256-511 | 1 | +0.0220 [+0.0220, +0.0220] |
| ouro-2.6b | gsm8k | bankR1all | strict | 0-63 | 200 | +0.0021 [-0.0015, +0.0056] |
| ouro-2.6b | gsm8k | bankR1all | strict | 64-127 | 143 | +0.0100 [+0.0059, +0.0143] |
| ouro-2.6b | gsm8k | bankR1all | strict | 128-255 | 37 | -0.0003 [-0.0104, +0.0081] |
| ouro-2.6b | gsm8k | bankR1all | strict | 256-511 | 1 | -0.0012 [-0.0012, -0.0012] |
| ouro-2.6b | gsm8k | bankrand0 | strict | 0-63 | 200 | +0.2671 [+0.2368, +0.2987] |
| ouro-2.6b | gsm8k | bankrand0 | strict | 64-127 | 143 | +0.2645 [+0.2164, +0.3158] |
| ouro-2.6b | gsm8k | bankrand0 | strict | 128-255 | 37 | +0.1514 [+0.0483, +0.2593] |
| ouro-2.6b | gsm8k | bankrand0 | strict | 256-511 | 1 | -0.0225 [-0.0225, -0.0225] |
| ouro-2.6b | gsm8k | first | strict | 0-63 | 200 | +0.9071 [+0.8638, +0.9518] |
| ouro-2.6b | gsm8k | first | strict | 64-127 | 143 | +0.9132 [+0.8543, +0.9739] |
| ouro-2.6b | gsm8k | first | strict | 128-255 | 37 | +0.9350 [+0.7996, +1.1029] |
| ouro-2.6b | gsm8k | first | strict | 256-511 | 1 | +0.5248 [+0.5248, +0.5248] |
| ouro-2.6b | gsm8k | full | strict | 0-63 | 200 | +0.0002 [-0.0003, +0.0008] |
| ouro-2.6b | gsm8k | full | strict | 64-127 | 143 | -0.0002 [-0.0010, +0.0005] |
| ouro-2.6b | gsm8k | full | strict | 128-255 | 37 | +0.0006 [-0.0009, +0.0020] |
| ouro-2.6b | gsm8k | full | strict | 256-511 | 1 | -0.0032 [-0.0032, -0.0032] |
| ouro-2.6b | gsm8k | last | strict | 0-63 | 200 | +0.2265 [+0.1964, +0.2585] |
| ouro-2.6b | gsm8k | last | strict | 64-127 | 143 | +0.2310 [+0.1839, +0.2845] |
| ouro-2.6b | gsm8k | last | strict | 128-255 | 37 | +0.1476 [+0.0662, +0.2263] |
| ouro-2.6b | gsm8k | last | strict | 256-511 | 1 | -0.0368 [-0.0368, -0.0368] |
| ouro-2.6b | math500 | bankP0 | strict | 0-63 | 200 | +0.0897 [+0.0756, +0.1038] |
| ouro-2.6b | math500 | bankP0 | strict | 64-127 | 193 | +0.0927 [+0.0784, +0.1078] |
| ouro-2.6b | math500 | bankP0 | strict | 128-255 | 144 | +0.0919 [+0.0703, +0.1177] |
| ouro-2.6b | math500 | bankP0 | strict | 256-511 | 70 | +0.0803 [+0.0566, +0.1140] |
| ouro-2.6b | math500 | bankP0 | strict | 512-99999 | 21 | +0.0548 [+0.0362, +0.0730] |
| ouro-2.6b | math500 | bankR1all | strict | 0-63 | 200 | +0.0090 [+0.0056, +0.0124] |
| ouro-2.6b | math500 | bankR1all | strict | 64-127 | 193 | +0.0124 [+0.0086, +0.0164] |
| ouro-2.6b | math500 | bankR1all | strict | 128-255 | 144 | +0.0116 [+0.0084, +0.0150] |
| ouro-2.6b | math500 | bankR1all | strict | 256-511 | 70 | +0.0095 [+0.0060, +0.0131] |
| ouro-2.6b | math500 | bankR1all | strict | 512-99999 | 21 | +0.0051 [-0.0012, +0.0125] |
| ouro-2.6b | math500 | bankrand0 | strict | 0-63 | 200 | +0.3222 [+0.2844, +0.3616] |
| ouro-2.6b | math500 | bankrand0 | strict | 64-127 | 193 | +0.2762 [+0.2439, +0.3112] |
| ouro-2.6b | math500 | bankrand0 | strict | 128-255 | 144 | +0.3153 [+0.2642, +0.3740] |
| ouro-2.6b | math500 | bankrand0 | strict | 256-511 | 70 | +0.3288 [+0.2702, +0.3920] |
| ouro-2.6b | math500 | bankrand0 | strict | 512-99999 | 21 | +0.3701 [+0.2015, +0.6168] |
| ouro-2.6b | math500 | first | strict | 0-63 | 200 | +0.8986 [+0.8618, +0.9355] |
| ouro-2.6b | math500 | first | strict | 64-127 | 193 | +0.9032 [+0.8606, +0.9467] |
| ouro-2.6b | math500 | first | strict | 128-255 | 144 | +0.9394 [+0.8903, +0.9923] |
| ouro-2.6b | math500 | first | strict | 256-511 | 70 | +0.9713 [+0.8981, +1.0461] |
| ouro-2.6b | math500 | first | strict | 512-99999 | 21 | +1.0216 [+0.9426, +1.1096] |
| ouro-2.6b | math500 | full | strict | 0-63 | 200 | +0.0005 [-0.0001, +0.0011] |
| ouro-2.6b | math500 | full | strict | 64-127 | 193 | -0.0000 [-0.0005, +0.0004] |
| ouro-2.6b | math500 | full | strict | 128-255 | 144 | -0.0000 [-0.0005, +0.0004] |
| ouro-2.6b | math500 | full | strict | 256-511 | 70 | +0.0000 [-0.0005, +0.0005] |
| ouro-2.6b | math500 | full | strict | 512-99999 | 21 | +0.0003 [-0.0010, +0.0014] |
| ouro-2.6b | math500 | last | strict | 0-63 | 200 | +0.3489 [+0.3096, +0.3895] |
| ouro-2.6b | math500 | last | strict | 64-127 | 193 | +0.3119 [+0.2745, +0.3497] |
| ouro-2.6b | math500 | last | strict | 128-255 | 144 | +0.3480 [+0.2930, +0.4109] |
| ouro-2.6b | math500 | last | strict | 256-511 | 70 | +0.3541 [+0.2854, +0.4294] |
| ouro-2.6b | math500 | last | strict | 512-99999 | 21 | +0.4117 [+0.2535, +0.6197] |
