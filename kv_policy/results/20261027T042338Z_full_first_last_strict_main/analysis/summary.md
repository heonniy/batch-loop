Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full | decode | 200 | 0.415 | 0.415 | 0.010 | 3 / 0 | 0.340 | 0.8996 | 0.9162 | 105.0 | 0.015 |
| huginn-0125 | gsm8k | last | strict | 200 | 0.405 | 0.405 | 0.010 | 3 / 0 | 0.320 | 0.9012 | 0.9177 | 104.5 | 0.015 |
| huginn-0125 | gsm8k | first | strict | 200 | 0.005 | 0.005 | 0.135 | 32 / 113 | 0.005 | 2.9974 | 3.0470 | 101.5 | 0.160 |
| huginn-0125 | gsm8k | full | strict | 200 | 0.420 | 0.415 | 0.015 | 2 / 0 | 0.335 | 0.8994 | 0.9162 | 102.8 | 0.010 |
| huginn-0125 | math500 | full | decode | 200 | 0.155 | 0.150 | 0.200 | 47 / 1 | 0.150 | 0.6317 | 0.6679 | 611.2 | 0.235 |
| huginn-0125 | math500 | last | strict | 200 | 0.160 | 0.145 | 0.210 | 50 / 1 | 0.155 | 0.6321 | 0.6685 | 641.4 | 0.250 |
| huginn-0125 | math500 | first | strict | 200 | 0.015 | 0.015 | 0.010 | 33 / 3 | 0.025 | 2.0549 | 2.1317 | 455.7 | 0.195 |
| huginn-0125 | math500 | full | strict | 200 | 0.145 | 0.140 | 0.195 | 51 / 2 | 0.140 | 0.6318 | 0.6681 | 655.9 | 0.255 |
| ouro-1.4b | gsm8k | full | decode | 200 | 0.835 | 0.830 | 0.015 | 1 / 0 | 0.600 | 0.7316 | 0.7445 | 111.3 | 0.005 |
| ouro-1.4b | gsm8k | last | strict | 200 | 0.000 | 0.000 | 0.005 | 199 / 0 | 0.000 | 6.0684 | 6.0277 | 510.7 | 0.995 |
| ouro-1.4b | gsm8k | first | strict | 200 | 0.210 | 0.195 | 0.095 | 12 / 1 | 0.185 | 1.2110 | 1.2086 | 124.5 | 0.070 |
| ouro-1.4b | gsm8k | full | strict | 200 | 0.810 | 0.805 | 0.015 | 0 / 0 | 0.565 | 0.7318 | 0.7448 | 108.8 | 0.000 |
| ouro-1.4b | math500 | full | decode | 200 | 0.710 | 0.710 | 0.000 | 4 / 0 | 0.675 | 0.5436 | 0.5910 | 538.7 | 0.020 |
| ouro-1.4b | math500 | last | strict | 200 | 0.000 | 0.000 | 0.000 | 192 / 8 | 0.000 | 5.3896 | 5.4156 | 1998.7 | 0.960 |
| ouro-1.4b | math500 | first | strict | 200 | 0.050 | 0.050 | 0.000 | 141 / 5 | 0.050 | 1.2056 | 1.2577 | 1646.9 | 0.735 |
| ouro-1.4b | math500 | full | strict | 200 | 0.740 | 0.740 | 0.000 | 4 / 0 | 0.700 | 0.5433 | 0.5907 | 545.1 | 0.020 |
| ouro-2.6b | gsm8k | full | decode | 200 | 0.880 | 0.880 | 0.000 | 0 / 0 | 0.660 | 0.6958 | 0.7084 | 109.4 | 0.000 |
| ouro-2.6b | gsm8k | last | strict | 200 | 0.600 | 0.600 | 0.000 | 3 / 0 | 0.500 | 0.9188 | 0.9221 | 112.8 | 0.025 |
| ouro-2.6b | gsm8k | first | strict | 200 | 0.090 | 0.055 | 0.575 | 28 / 4 | 0.050 | 1.6057 | 1.6346 | 172.6 | 0.145 |
| ouro-2.6b | gsm8k | full | strict | 200 | 0.880 | 0.880 | 0.000 | 0 / 0 | 0.655 | 0.6959 | 0.7086 | 109.1 | 0.000 |
| ouro-2.6b | math500 | full | decode | 200 | 0.560 | 0.560 | 0.000 | 18 / 0 | 0.520 | 0.4649 | 0.5029 | 423.9 | 0.090 |
| ouro-2.6b | math500 | last | strict | 200 | 0.330 | 0.330 | 0.000 | 32 / 1 | 0.305 | 0.8114 | 0.8105 | 468.1 | 0.165 |
| ouro-2.6b | math500 | first | strict | 200 | 0.075 | 0.075 | 0.005 | 82 / 1 | 0.075 | 1.3985 | 1.4273 | 1579.8 | 0.655 |
| ouro-2.6b | math500 | full | strict | 200 | 0.565 | 0.565 | 0.000 | 19 / 0 | 0.520 | 0.4650 | 0.5031 | 427.5 | 0.095 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full/decode | last | strict | -0.010 [-0.040, +0.015] | 5 / 3 | 0.727 | +0.0016 [-0.0002, +0.0035] |
| huginn-0125 | gsm8k | full/decode | first | strict | -0.410 [-0.480, -0.345] | 82 / 0 | 4.14e-25 | +2.0978 [+2.0419, +2.1545] |
| huginn-0125 | gsm8k | full/decode | full | strict | +0.005 [-0.025, +0.035] | 4 / 5 | 1 | -0.0001 [-0.0006, +0.0003] |
| huginn-0125 | gsm8k | full/strict | last | strict | -0.015 [-0.050, +0.015] | 7 / 4 | 0.549 | +0.0017 [+0.0000, +0.0037] |
| huginn-0125 | math500 | full/decode | last | strict | +0.005 [-0.020, +0.030] | 3 / 4 | 1 | +0.0003 [-0.0003, +0.0011] |
| huginn-0125 | math500 | full/decode | first | strict | -0.140 [-0.190, -0.090] | 30 / 2 | 2.46e-07 | +1.4231 [+1.3880, +1.4595] |
| huginn-0125 | math500 | full/decode | full | strict | -0.010 [-0.030, +0.010] | 3 / 1 | 0.625 | +0.0001 [-0.0001, +0.0003] |
| huginn-0125 | math500 | full/strict | last | strict | +0.015 [-0.005, +0.040] | 1 / 4 | 0.375 | +0.0003 [-0.0005, +0.0011] |
| ouro-1.4b | gsm8k | full/decode | last | strict | -0.835 [-0.885, -0.780] | 167 / 0 | 1.07e-50 | +5.3368 [+5.2272, +5.4493] |
| ouro-1.4b | gsm8k | full/decode | first | strict | -0.625 [-0.695, -0.550] | 129 / 4 | 2.36e-33 | +0.4794 [+0.4558, +0.5024] |
| ouro-1.4b | gsm8k | full/decode | full | strict | -0.025 [-0.055, +0.005] | 7 / 2 | 0.18 | +0.0003 [-0.0002, +0.0007] |
| ouro-1.4b | gsm8k | full/strict | last | strict | -0.810 [-0.865, -0.755] | 162 / 0 | 3.42e-49 | +5.3365 [+5.2260, +5.4464] |
| ouro-1.4b | math500 | full/decode | last | strict | -0.710 [-0.775, -0.645] | 142 / 0 | 3.59e-43 | +4.8460 [+4.7187, +4.9701] |
| ouro-1.4b | math500 | full/decode | first | strict | -0.660 [-0.725, -0.595] | 132 / 0 | 3.67e-40 | +0.6620 [+0.6333, +0.6921] |
| ouro-1.4b | math500 | full/decode | full | strict | +0.030 [-0.010, +0.070] | 5 / 11 | 0.21 | -0.0003 [-0.0006, -0.0001] |
| ouro-1.4b | math500 | full/strict | last | strict | -0.740 [-0.800, -0.680] | 148 / 0 | 5.61e-45 | +4.8463 [+4.7207, +4.9705] |
| ouro-2.6b | gsm8k | full/decode | last | strict | -0.280 [-0.345, -0.215] | 59 / 3 | 1.72e-14 | +0.2230 [+0.1936, +0.2557] |
| ouro-2.6b | gsm8k | full/decode | first | strict | -0.790 [-0.845, -0.730] | 160 / 2 | 4.52e-45 | +0.9099 [+0.8733, +0.9481] |
| ouro-2.6b | gsm8k | full/decode | full | strict | +0.000 [-0.015, +0.015] | 1 / 1 | 1 | +0.0001 [-0.0003, +0.0005] |
| ouro-2.6b | gsm8k | full/strict | last | strict | -0.280 [-0.345, -0.215] | 59 / 3 | 1.72e-14 | +0.2229 [+0.1927, +0.2565] |
| ouro-2.6b | math500 | full/decode | last | strict | -0.230 [-0.305, -0.155] | 59 / 13 | 3.81e-08 | +0.3465 [+0.3090, +0.3855] |
| ouro-2.6b | math500 | full/decode | first | strict | -0.485 [-0.555, -0.410] | 100 / 3 | 3.59e-26 | +0.9336 [+0.8994, +0.9677] |
| ouro-2.6b | math500 | full/decode | full | strict | +0.005 [-0.030, +0.040] | 6 / 7 | 1 | +0.0001 [-0.0001, +0.0004] |
| ouro-2.6b | math500 | full/strict | last | strict | -0.235 [-0.315, -0.160] | 60 / 13 | 2.31e-08 | +0.3464 [+0.3091, +0.3860] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | first | strict | 0.015 | 0.160 | 0.005 | 0.005 | -0.410 [-0.480, -0.345] | 0 | 3 |
| huginn-0125 | gsm8k | full | decode | 0.000 | 0.015 | 0.415 | 0.415 |  | 0 | 0 |
| huginn-0125 | gsm8k | full | strict | 0.000 | 0.010 | 0.420 | 0.420 | +0.005 [-0.025, +0.035] | 0 | 0 |
| huginn-0125 | gsm8k | last | strict | 0.000 | 0.015 | 0.405 | 0.405 | -0.010 [-0.040, +0.015] | 0 | 0 |
| huginn-0125 | math500 | first | strict | 0.010 | 0.195 | 0.015 | 0.015 | -0.140 [-0.190, -0.090] | 0 | 2 |
| huginn-0125 | math500 | full | decode | 0.060 | 0.235 | 0.155 | 0.155 |  | 0 | 12 |
| huginn-0125 | math500 | full | strict | 0.040 | 0.255 | 0.145 | 0.145 | -0.010 [-0.030, +0.010] | 0 | 8 |
| huginn-0125 | math500 | last | strict | 0.050 | 0.250 | 0.160 | 0.160 | +0.005 [-0.020, +0.030] | 0 | 10 |
| ouro-1.4b | gsm8k | first | strict | 0.000 | 0.070 | 0.210 | 0.210 | -0.625 [-0.695, -0.550] | 0 | 0 |
| ouro-1.4b | gsm8k | full | decode | 0.000 | 0.005 | 0.835 | 0.835 |  | 0 | 0 |
| ouro-1.4b | gsm8k | full | strict | 0.000 | 0.000 | 0.810 | 0.810 | -0.025 [-0.055, +0.005] | 0 | 0 |
| ouro-1.4b | gsm8k | last | strict | 0.335 | 0.995 | 0.000 | 0.000 | -0.835 [-0.885, -0.780] | 0 | 67 |
| ouro-1.4b | math500 | first | strict | 0.045 | 0.735 | 0.050 | 0.045 | -0.665 [-0.730, -0.600] | 0 | 9 |
| ouro-1.4b | math500 | full | decode | 0.005 | 0.020 | 0.710 | 0.710 |  | 0 | 1 |
| ouro-1.4b | math500 | full | strict | 0.005 | 0.020 | 0.740 | 0.740 | +0.030 [-0.010, +0.070] | 0 | 1 |
| ouro-1.4b | math500 | last | strict | 0.175 | 0.960 | 0.000 | 0.000 | -0.710 [-0.770, -0.645] | 0 | 35 |
| ouro-2.6b | gsm8k | first | strict | 0.010 | 0.145 | 0.090 | 0.090 | -0.790 [-0.850, -0.725] | 0 | 2 |
| ouro-2.6b | gsm8k | full | decode | 0.000 | 0.000 | 0.880 | 0.880 |  | 0 | 0 |
| ouro-2.6b | gsm8k | full | strict | 0.000 | 0.000 | 0.880 | 0.880 | +0.000 [-0.015, +0.015] | 0 | 0 |
| ouro-2.6b | gsm8k | last | strict | 0.000 | 0.025 | 0.600 | 0.600 | -0.280 [-0.350, -0.215] | 0 | 0 |
| ouro-2.6b | math500 | first | strict | 0.000 | 0.655 | 0.075 | 0.075 | -0.485 [-0.560, -0.410] | 0 | 0 |
| ouro-2.6b | math500 | full | decode | 0.015 | 0.090 | 0.560 | 0.560 |  | 0 | 3 |
| ouro-2.6b | math500 | full | strict | 0.015 | 0.095 | 0.565 | 0.565 | +0.005 [-0.030, +0.040] | 0 | 3 |
| ouro-2.6b | math500 | last | strict | 0.035 | 0.165 | 0.330 | 0.325 | -0.235 [-0.315, -0.155] | 0 | 7 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | first | strict | 0-63 | 200 | +2.1698 [+2.1061, +2.2350] |
| huginn-0125 | gsm8k | first | strict | 64-127 | 130 | +1.9588 [+1.8787, +2.0393] |
| huginn-0125 | gsm8k | first | strict | 128-255 | 23 | +1.8433 [+1.6334, +2.0407] |
| huginn-0125 | gsm8k | first | strict | 256-511 | 1 | +1.5310 [+1.5310, +1.5310] |
| huginn-0125 | gsm8k | full | strict | 0-63 | 200 | -0.0003 [-0.0009, +0.0002] |
| huginn-0125 | gsm8k | full | strict | 64-127 | 130 | +0.0003 [-0.0004, +0.0011] |
| huginn-0125 | gsm8k | full | strict | 128-255 | 23 | +0.0003 [-0.0021, +0.0019] |
| huginn-0125 | gsm8k | full | strict | 256-511 | 1 | -0.0013 [-0.0013, -0.0013] |
| huginn-0125 | gsm8k | last | strict | 0-63 | 200 | +0.0022 [+0.0003, +0.0044] |
| huginn-0125 | gsm8k | last | strict | 64-127 | 130 | -0.0000 [-0.0032, +0.0034] |
| huginn-0125 | gsm8k | last | strict | 128-255 | 23 | +0.0014 [-0.0035, +0.0084] |
| huginn-0125 | gsm8k | last | strict | 256-511 | 1 | -0.0061 [-0.0061, -0.0061] |
| huginn-0125 | math500 | first | strict | 0-63 | 200 | +1.5330 [+1.4829, +1.5847] |
| huginn-0125 | math500 | first | strict | 64-127 | 192 | +1.3778 [+1.3316, +1.4246] |
| huginn-0125 | math500 | first | strict | 128-255 | 136 | +1.3992 [+1.3497, +1.4495] |
| huginn-0125 | math500 | first | strict | 256-511 | 67 | +1.3975 [+1.3147, +1.4822] |
| huginn-0125 | math500 | first | strict | 512-99999 | 18 | +1.2910 [+1.2008, +1.3717] |
| huginn-0125 | math500 | full | strict | 0-63 | 200 | +0.0001 [-0.0004, +0.0006] |
| huginn-0125 | math500 | full | strict | 64-127 | 192 | +0.0001 [-0.0004, +0.0006] |
| huginn-0125 | math500 | full | strict | 128-255 | 136 | +0.0000 [-0.0004, +0.0004] |
| huginn-0125 | math500 | full | strict | 256-511 | 67 | +0.0001 [-0.0003, +0.0005] |
| huginn-0125 | math500 | full | strict | 512-99999 | 18 | -0.0001 [-0.0006, +0.0003] |
| huginn-0125 | math500 | last | strict | 0-63 | 200 | +0.0007 [-0.0005, +0.0018] |
| huginn-0125 | math500 | last | strict | 64-127 | 192 | -0.0002 [-0.0019, +0.0016] |
| huginn-0125 | math500 | last | strict | 128-255 | 136 | +0.0004 [-0.0004, +0.0014] |
| huginn-0125 | math500 | last | strict | 256-511 | 67 | +0.0001 [-0.0007, +0.0008] |
| huginn-0125 | math500 | last | strict | 512-99999 | 18 | +0.0015 [-0.0007, +0.0045] |
| ouro-1.4b | gsm8k | first | strict | 0-63 | 200 | +0.4791 [+0.4507, +0.5082] |
| ouro-1.4b | gsm8k | first | strict | 64-127 | 143 | +0.4897 [+0.4544, +0.5246] |
| ouro-1.4b | gsm8k | first | strict | 128-255 | 37 | +0.4282 [+0.3644, +0.4946] |
| ouro-1.4b | gsm8k | first | strict | 256-511 | 1 | +0.4387 [+0.4387, +0.4387] |
| ouro-1.4b | gsm8k | full | strict | 0-63 | 200 | +0.0003 [-0.0002, +0.0010] |
| ouro-1.4b | gsm8k | full | strict | 64-127 | 143 | +0.0000 [-0.0006, +0.0007] |
| ouro-1.4b | gsm8k | full | strict | 128-255 | 37 | +0.0008 [-0.0008, +0.0026] |
| ouro-1.4b | gsm8k | full | strict | 256-511 | 1 | -0.0070 [-0.0070, -0.0070] |
| ouro-1.4b | gsm8k | last | strict | 0-63 | 200 | +5.2937 [+5.1951, +5.3974] |
| ouro-1.4b | gsm8k | last | strict | 64-127 | 143 | +5.3761 [+5.2264, +5.5320] |
| ouro-1.4b | gsm8k | last | strict | 128-255 | 37 | +5.5926 [+5.0878, +6.0619] |
| ouro-1.4b | gsm8k | last | strict | 256-511 | 1 | +5.9722 [+5.9722, +5.9722] |
| ouro-1.4b | math500 | first | strict | 0-63 | 200 | +0.5232 [+0.4938, +0.5542] |
| ouro-1.4b | math500 | first | strict | 64-127 | 193 | +0.6676 [+0.6302, +0.7072] |
| ouro-1.4b | math500 | first | strict | 128-255 | 144 | +0.6922 [+0.6509, +0.7364] |
| ouro-1.4b | math500 | first | strict | 256-511 | 70 | +0.7485 [+0.6816, +0.8229] |
| ouro-1.4b | math500 | first | strict | 512-99999 | 21 | +0.7850 [+0.6706, +0.9254] |
| ouro-1.4b | math500 | full | strict | 0-63 | 200 | -0.0007 [-0.0014, -0.0001] |
| ouro-1.4b | math500 | full | strict | 64-127 | 193 | -0.0002 [-0.0008, +0.0004] |
| ouro-1.4b | math500 | full | strict | 128-255 | 144 | +0.0001 [-0.0004, +0.0006] |
| ouro-1.4b | math500 | full | strict | 256-511 | 70 | -0.0005 [-0.0011, +0.0001] |
| ouro-1.4b | math500 | full | strict | 512-99999 | 21 | -0.0004 [-0.0013, +0.0006] |
| ouro-1.4b | math500 | last | strict | 0-63 | 200 | +4.9029 [+4.7898, +5.0165] |
| ouro-1.4b | math500 | last | strict | 64-127 | 193 | +4.8980 [+4.7655, +5.0330] |
| ouro-1.4b | math500 | last | strict | 128-255 | 144 | +4.9069 [+4.7550, +5.0565] |
| ouro-1.4b | math500 | last | strict | 256-511 | 70 | +4.7798 [+4.5384, +5.0328] |
| ouro-1.4b | math500 | last | strict | 512-99999 | 21 | +4.4593 [+3.9991, +4.9883] |
| ouro-2.6b | gsm8k | first | strict | 0-63 | 200 | +0.9071 [+0.8640, +0.9516] |
| ouro-2.6b | gsm8k | first | strict | 64-127 | 143 | +0.9132 [+0.8535, +0.9742] |
| ouro-2.6b | gsm8k | first | strict | 128-255 | 37 | +0.9350 [+0.7983, +1.1042] |
| ouro-2.6b | gsm8k | first | strict | 256-511 | 1 | +0.5248 [+0.5248, +0.5248] |
| ouro-2.6b | gsm8k | full | strict | 0-63 | 200 | +0.0002 [-0.0003, +0.0008] |
| ouro-2.6b | gsm8k | full | strict | 64-127 | 143 | -0.0002 [-0.0010, +0.0005] |
| ouro-2.6b | gsm8k | full | strict | 128-255 | 37 | +0.0006 [-0.0009, +0.0020] |
| ouro-2.6b | gsm8k | full | strict | 256-511 | 1 | -0.0032 [-0.0032, -0.0032] |
| ouro-2.6b | gsm8k | last | strict | 0-63 | 200 | +0.2265 [+0.1967, +0.2584] |
| ouro-2.6b | gsm8k | last | strict | 64-127 | 143 | +0.2310 [+0.1823, +0.2832] |
| ouro-2.6b | gsm8k | last | strict | 128-255 | 37 | +0.1476 [+0.0666, +0.2264] |
| ouro-2.6b | gsm8k | last | strict | 256-511 | 1 | -0.0368 [-0.0368, -0.0368] |
| ouro-2.6b | math500 | first | strict | 0-63 | 200 | +0.8986 [+0.8627, +0.9360] |
| ouro-2.6b | math500 | first | strict | 64-127 | 193 | +0.9032 [+0.8618, +0.9457] |
| ouro-2.6b | math500 | first | strict | 128-255 | 144 | +0.9394 [+0.8894, +0.9919] |
| ouro-2.6b | math500 | first | strict | 256-511 | 70 | +0.9713 [+0.8983, +1.0464] |
| ouro-2.6b | math500 | first | strict | 512-99999 | 21 | +1.0216 [+0.9422, +1.1114] |
| ouro-2.6b | math500 | full | strict | 0-63 | 200 | +0.0005 [-0.0002, +0.0011] |
| ouro-2.6b | math500 | full | strict | 64-127 | 193 | -0.0000 [-0.0005, +0.0004] |
| ouro-2.6b | math500 | full | strict | 128-255 | 144 | -0.0000 [-0.0005, +0.0004] |
| ouro-2.6b | math500 | full | strict | 256-511 | 70 | +0.0000 [-0.0005, +0.0005] |
| ouro-2.6b | math500 | full | strict | 512-99999 | 21 | +0.0003 [-0.0010, +0.0014] |
| ouro-2.6b | math500 | last | strict | 0-63 | 200 | +0.3489 [+0.3091, +0.3906] |
| ouro-2.6b | math500 | last | strict | 64-127 | 193 | +0.3119 [+0.2754, +0.3519] |
| ouro-2.6b | math500 | last | strict | 128-255 | 144 | +0.3480 [+0.2917, +0.4098] |
| ouro-2.6b | math500 | last | strict | 256-511 | 70 | +0.3541 [+0.2864, +0.4290] |
| ouro-2.6b | math500 | last | strict | 512-99999 | 21 | +0.4117 [+0.2536, +0.6253] |
