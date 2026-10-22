Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full | decode | 200 | 0.410 | 0.410 | 0.000 | 7 / 0 | 0.310 | 0.8812 | 0.8949 | 103.1 | 0.035 |
| huginn-0125 | gsm8k | last | convert | 200 | 0.395 | 0.390 | 0.005 | 6 / 0 | 0.305 | 0.8890 | 0.9033 | 103.8 | 0.030 |
| huginn-0125 | gsm8k | average | convert | 200 | 0.345 | 0.265 | 0.220 | 10 / 0 | 0.265 | 0.8655 | 0.8740 | 100.3 | 0.050 |
| huginn-0125 | gsm8k | last | decode | 200 | 0.395 | 0.385 | 0.010 | 5 / 0 | 0.305 | 0.8825 | 0.8963 | 104.4 | 0.025 |
| huginn-0125 | gsm8k | average | decode | 200 | 0.340 | 0.340 | 0.025 | 10 / 0 | 0.325 | 0.8833 | 0.8963 | 103.8 | 0.050 |
| huginn-0125 | gsm8k | last | strict | 200 | 0.410 | 0.405 | 0.010 | 4 / 0 | 0.325 | 0.9022 | 0.9196 | 100.8 | 0.020 |
| huginn-0125 | gsm8k | average | strict | 200 | 0.395 | 0.390 | 0.045 | 2 / 0 | 0.380 | 0.8712 | 0.8811 | 95.7 | 0.010 |
| huginn-0125 | gsm8k | full | strict | 200 | 0.415 | 0.415 | 0.000 | 7 / 0 | 0.315 | 0.8811 | 0.8947 | 103.3 | 0.035 |
| huginn-0125 | gsm8k | full_altbatch | decode | 200 | 0.415 | 0.415 | 0.005 | 5 / 0 | 0.320 | 0.8814 | 0.8948 | 102.4 | 0.025 |
| huginn-0125 | math500 | full | decode | 200 | 0.140 | 0.135 | 0.135 | 41 / 1 | 0.135 | 0.6376 | 0.6731 | 343.6 | 0.205 |
| huginn-0125 | math500 | last | convert | 200 | 0.160 | 0.155 | 0.155 | 49 / 3 | 0.150 | 0.6319 | 0.6677 | 381.7 | 0.245 |
| huginn-0125 | math500 | average | convert | 200 | 0.125 | 0.085 | 0.320 | 54 / 1 | 0.125 | 0.6610 | 0.7040 | 380.1 | 0.270 |
| huginn-0125 | math500 | last | decode | 200 | 0.155 | 0.155 | 0.155 | 47 / 1 | 0.150 | 0.6323 | 0.6682 | 372.0 | 0.235 |
| huginn-0125 | math500 | average | decode | 200 | 0.135 | 0.100 | 0.205 | 54 / 2 | 0.130 | 0.6515 | 0.6895 | 384.1 | 0.270 |
| huginn-0125 | math500 | last | strict | 200 | 0.155 | 0.150 | 0.225 | 54 / 2 | 0.150 | 0.6311 | 0.6671 | 394.9 | 0.270 |
| huginn-0125 | math500 | average | strict | 200 | 0.140 | 0.090 | 0.355 | 53 / 1 | 0.140 | 0.6547 | 0.6955 | 373.3 | 0.265 |
| huginn-0125 | math500 | full | strict | 200 | 0.140 | 0.135 | 0.135 | 41 / 1 | 0.135 | 0.6376 | 0.6731 | 343.4 | 0.205 |
| huginn-0125 | math500 | full_altbatch | decode | 200 | 0.150 | 0.145 | 0.150 | 45 / 1 | 0.145 | 0.6375 | 0.6730 | 356.5 | 0.225 |
| ouro-1.4b | gsm8k | full | decode | 200 | 0.820 | 0.815 | 0.015 | 2 / 0 | 0.580 | 0.7319 | 0.7449 | 107.5 | 0.010 |
| ouro-1.4b | gsm8k | last | convert | 200 | 0.000 | 0.000 | 0.010 | 186 / 12 | 0.000 | 3.4998 | 3.4625 | 242.1 | 0.930 |
| ouro-1.4b | gsm8k | average | convert | 200 | 0.230 | 0.115 | 0.510 | 24 / 5 | 0.110 | 1.0203 | 1.0100 | 101.0 | 0.120 |
| ouro-1.4b | gsm8k | last | decode | 200 | 0.685 | 0.685 | 0.005 | 3 / 0 | 0.510 | 0.7528 | 0.7661 | 236.6 | 0.865 |
| ouro-1.4b | gsm8k | average | decode | 200 | 0.735 | 0.735 | 0.005 | 4 / 0 | 0.540 | 0.7414 | 0.7517 | 251.8 | 0.975 |
| ouro-1.4b | gsm8k | last | strict | 200 | 0.000 | 0.000 | 0.000 | 200 / 0 | 0.000 | 6.2638 | 6.2300 | 256.0 | 1.000 |
| ouro-1.4b | gsm8k | average | strict | 200 | 0.025 | 0.000 | 0.800 | 26 / 4 | 0.000 | 1.5375 | 1.5377 | 89.9 | 0.150 |
| ouro-1.4b | gsm8k | full | strict | 200 | 0.815 | 0.810 | 0.020 | 2 / 0 | 0.565 | 0.7313 | 0.7443 | 108.7 | 0.010 |
| ouro-1.4b | gsm8k | full_altbatch | decode | 200 | 0.810 | 0.805 | 0.015 | 1 / 0 | 0.575 | 0.7318 | 0.7448 | 108.8 | 0.005 |
| ouro-1.4b | math500 | full | decode | 200 | 0.710 | 0.710 | 0.000 | 13 / 0 | 0.675 | 0.5435 | 0.5909 | 500.2 | 0.075 |
| ouro-1.4b | math500 | last | convert | 200 | 0.000 | 0.000 | 0.040 | 139 / 41 | 0.000 | 2.6125 | 2.6518 | 771.1 | 0.735 |
| ouro-1.4b | math500 | average | convert | 200 | 0.025 | 0.025 | 0.005 | 145 / 5 | 0.025 | 1.2609 | 1.2528 | 890.8 | 0.835 |
| ouro-1.4b | math500 | last | decode | 200 | 0.160 | 0.150 | 0.020 | 144 / 4 | 0.150 | 0.6177 | 0.6501 | 919.9 | 0.800 |
| ouro-1.4b | math500 | average | decode | 200 | 0.320 | 0.320 | 0.000 | 106 / 0 | 0.310 | 0.5908 | 0.6312 | 1024.0 | 1.000 |
| ouro-1.4b | math500 | last | strict | 200 | 0.000 | 0.000 | 0.005 | 182 / 17 | 0.000 | 5.4176 | 5.4458 | 973.4 | 0.910 |
| ouro-1.4b | math500 | average | strict | 200 | 0.000 | 0.000 | 0.000 | 200 / 0 | 0.000 | 2.3193 | 2.2298 | 1024.0 | 1.000 |
| ouro-1.4b | math500 | full | strict | 200 | 0.695 | 0.695 | 0.000 | 13 / 0 | 0.660 | 0.5436 | 0.5909 | 510.1 | 0.070 |
| ouro-1.4b | math500 | full_altbatch | decode | 200 | 0.720 | 0.720 | 0.000 | 16 / 0 | 0.690 | 0.5434 | 0.5907 | 513.9 | 0.085 |
| ouro-2.6b | gsm8k | full | decode | 200 | 0.875 | 0.875 | 0.000 | 0 / 0 | 0.655 | 0.6959 | 0.7084 | 109.2 | 0.000 |
| ouro-2.6b | gsm8k | last | convert | 200 | 0.760 | 0.760 | 0.005 | 5 / 1 | 0.610 | 0.8610 | 0.8540 | 108.3 | 0.030 |
| ouro-2.6b | gsm8k | average | convert | 200 | 0.195 | 0.195 | 0.040 | 47 / 0 | 0.145 | 1.0196 | 1.0230 | 147.5 | 0.345 |
| ouro-2.6b | gsm8k | last | decode | 200 | 0.840 | 0.840 | 0.000 | 0 / 0 | 0.670 | 0.7045 | 0.7168 | 107.5 | 0.005 |
| ouro-2.6b | gsm8k | average | decode | 200 | 0.840 | 0.840 | 0.000 | 1 / 0 | 0.645 | 0.6966 | 0.7083 | 108.0 | 0.010 |
| ouro-2.6b | gsm8k | last | strict | 200 | 0.545 | 0.545 | 0.010 | 7 / 0 | 0.445 | 0.9777 | 0.9786 | 104.9 | 0.035 |
| ouro-2.6b | gsm8k | average | strict | 200 | 0.010 | 0.000 | 0.230 | 154 / 0 | 0.000 | 3.0756 | 3.0140 | 203.7 | 0.770 |
| ouro-2.6b | gsm8k | full | strict | 200 | 0.875 | 0.875 | 0.005 | 1 / 0 | 0.655 | 0.6959 | 0.7085 | 109.2 | 0.005 |
| ouro-2.6b | gsm8k | full_altbatch | decode | 200 | 0.875 | 0.875 | 0.000 | 1 / 0 | 0.655 | 0.6958 | 0.7083 | 110.5 | 0.005 |
| ouro-2.6b | math500 | full | decode | 200 | 0.565 | 0.565 | 0.000 | 21 / 0 | 0.525 | 0.4646 | 0.5027 | 328.4 | 0.105 |
| ouro-2.6b | math500 | last | convert | 200 | 0.030 | 0.030 | 0.010 | 164 / 7 | 0.030 | 2.1717 | 2.0262 | 907.2 | 0.875 |
| ouro-2.6b | math500 | average | convert | 200 | 0.040 | 0.040 | 0.010 | 155 / 7 | 0.040 | 0.9343 | 0.9354 | 902.5 | 0.865 |
| ouro-2.6b | math500 | last | decode | 200 | 0.505 | 0.505 | 0.000 | 22 / 0 | 0.490 | 0.4943 | 0.5286 | 316.5 | 0.110 |
| ouro-2.6b | math500 | average | decode | 200 | 0.500 | 0.500 | 0.000 | 37 / 0 | 0.475 | 0.4933 | 0.5268 | 370.5 | 0.185 |
| ouro-2.6b | math500 | last | strict | 200 | 0.325 | 0.325 | 0.000 | 33 / 0 | 0.310 | 0.8201 | 0.8180 | 308.7 | 0.175 |
| ouro-2.6b | math500 | average | strict | 200 | 0.000 | 0.000 | 0.000 | 16 / 178 | 0.000 | 4.4869 | 4.2590 | 116.2 | 0.110 |
| ouro-2.6b | math500 | full | strict | 200 | 0.540 | 0.540 | 0.000 | 26 / 0 | 0.510 | 0.4649 | 0.5030 | 341.6 | 0.130 |
| ouro-2.6b | math500 | full_altbatch | decode | 200 | 0.550 | 0.550 | 0.000 | 21 / 0 | 0.530 | 0.4647 | 0.5027 | 323.6 | 0.105 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full/decode | last | convert | -0.015 [-0.065, +0.035] | 15 / 12 | 0.701 | +0.0077 [+0.0037, +0.0120] |
| huginn-0125 | gsm8k | full/decode | average | convert | -0.065 [-0.135, +0.005] | 33 / 20 | 0.0984 | -0.0157 [-0.0225, -0.0089] |
| huginn-0125 | gsm8k | full/decode | last | decode | -0.015 [-0.065, +0.035] | 15 / 12 | 0.701 | +0.0012 [-0.0016, +0.0042] |
| huginn-0125 | gsm8k | full/decode | average | decode | -0.070 [-0.130, -0.010] | 26 / 12 | 0.0336 | +0.0021 [-0.0023, +0.0068] |
| huginn-0125 | gsm8k | full/decode | last | strict | +0.000 [-0.060, +0.060] | 18 / 18 | 1 | +0.0210 [+0.0158, +0.0268] |
| huginn-0125 | gsm8k | full/decode | average | strict | -0.015 [-0.080, +0.050] | 23 / 20 | 0.761 | -0.0100 [-0.0164, -0.0038] |
| huginn-0125 | gsm8k | full/decode | full | strict | +0.005 [+0.000, +0.015] | 0 / 1 | 1 | -0.0001 [-0.0002, +0.0000] |
| huginn-0125 | gsm8k | full/decode | full_altbatch | decode | +0.005 [-0.020, +0.030] | 3 / 4 | 1 | +0.0002 [-0.0002, +0.0006] |
| huginn-0125 | gsm8k | full/strict | last | strict | -0.005 [-0.065, +0.055] | 19 / 18 | 1 | +0.0211 [+0.0158, +0.0268] |
| huginn-0125 | gsm8k | full/strict | average | strict | -0.020 [-0.080, +0.045] | 23 / 19 | 0.644 | -0.0099 [-0.0162, -0.0037] |
| huginn-0125 | math500 | full/decode | last | convert | +0.020 [-0.020, +0.060] | 6 / 10 | 0.454 | -0.0057 [-0.0073, -0.0041] |
| huginn-0125 | math500 | full/decode | average | convert | -0.015 [-0.055, +0.025] | 10 / 7 | 0.629 | +0.0234 [+0.0207, +0.0262] |
| huginn-0125 | math500 | full/decode | last | decode | +0.015 [-0.020, +0.055] | 6 / 9 | 0.607 | -0.0053 [-0.0070, -0.0036] |
| huginn-0125 | math500 | full/decode | average | decode | -0.005 [-0.045, +0.035] | 9 / 8 | 1 | +0.0139 [+0.0117, +0.0161] |
| huginn-0125 | math500 | full/decode | last | strict | +0.015 [-0.025, +0.055] | 7 / 10 | 0.629 | -0.0065 [-0.0081, -0.0049] |
| huginn-0125 | math500 | full/decode | average | strict | +0.000 [-0.035, +0.035] | 7 / 7 | 1 | +0.0171 [+0.0147, +0.0195] |
| huginn-0125 | math500 | full/decode | full | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | -0.0000 [-0.0001, +0.0000] |
| huginn-0125 | math500 | full/decode | full_altbatch | decode | +0.010 [-0.015, +0.035] | 2 / 4 | 0.688 | -0.0001 [-0.0003, +0.0001] |
| huginn-0125 | math500 | full/strict | last | strict | +0.015 [-0.025, +0.055] | 7 / 10 | 0.629 | -0.0064 [-0.0080, -0.0048] |
| huginn-0125 | math500 | full/strict | average | strict | +0.000 [-0.035, +0.035] | 7 / 7 | 1 | +0.0171 [+0.0148, +0.0195] |
| ouro-1.4b | gsm8k | full/decode | last | convert | -0.820 [-0.870, -0.765] | 164 / 0 | 8.55e-50 | +2.7678 [+2.6792, +2.8560] |
| ouro-1.4b | gsm8k | full/decode | average | convert | -0.590 [-0.660, -0.520] | 120 / 2 | 2.82e-33 | +0.2884 [+0.2693, +0.3080] |
| ouro-1.4b | gsm8k | full/decode | last | decode | -0.135 [-0.195, -0.075] | 34 / 7 | 2.53e-05 | +0.0209 [+0.0159, +0.0260] |
| ouro-1.4b | gsm8k | full/decode | average | decode | -0.085 [-0.140, -0.030] | 24 / 7 | 0.00333 | +0.0095 [+0.0064, +0.0126] |
| ouro-1.4b | gsm8k | full/decode | last | strict | -0.820 [-0.870, -0.765] | 164 / 0 | 8.55e-50 | +5.5319 [+5.4326, +5.6319] |
| ouro-1.4b | gsm8k | full/decode | average | strict | -0.795 [-0.850, -0.740] | 159 / 0 | 2.74e-48 | +0.8056 [+0.7774, +0.8334] |
| ouro-1.4b | gsm8k | full/decode | full | strict | -0.005 [-0.030, +0.020] | 4 / 3 | 1 | -0.0007 [-0.0011, -0.0002] |
| ouro-1.4b | gsm8k | full/decode | full_altbatch | decode | -0.010 [-0.040, +0.020] | 5 / 3 | 0.727 | -0.0001 [-0.0005, +0.0003] |
| ouro-1.4b | gsm8k | full/strict | last | strict | -0.815 [-0.870, -0.760] | 163 / 0 | 1.71e-49 | +5.5325 [+5.4311, +5.6358] |
| ouro-1.4b | gsm8k | full/strict | average | strict | -0.790 [-0.845, -0.730] | 158 / 0 | 5.47e-48 | +0.8063 [+0.7792, +0.8343] |
| ouro-1.4b | math500 | full/decode | last | convert | -0.710 [-0.770, -0.650] | 142 / 0 | 3.59e-43 | +2.0690 [+1.9673, +2.1733] |
| ouro-1.4b | math500 | full/decode | average | convert | -0.685 [-0.750, -0.620] | 137 / 0 | 1.15e-41 | +0.7174 [+0.6789, +0.7565] |
| ouro-1.4b | math500 | full/decode | last | decode | -0.550 [-0.620, -0.475] | 112 / 2 | 6.31e-31 | +0.0741 [+0.0608, +0.0889] |
| ouro-1.4b | math500 | full/decode | average | decode | -0.390 [-0.465, -0.315] | 83 / 5 | 2.69e-19 | +0.0473 [+0.0394, +0.0555] |
| ouro-1.4b | math500 | full/decode | last | strict | -0.710 [-0.775, -0.645] | 142 / 0 | 3.59e-43 | +4.8741 [+4.7449, +5.0025] |
| ouro-1.4b | math500 | full/decode | average | strict | -0.710 [-0.770, -0.645] | 142 / 0 | 3.59e-43 | +1.7758 [+1.7057, +1.8420] |
| ouro-1.4b | math500 | full/decode | full | strict | -0.015 [-0.045, +0.015] | 6 / 3 | 0.508 | +0.0001 [-0.0001, +0.0004] |
| ouro-1.4b | math500 | full/decode | full_altbatch | decode | +0.010 [-0.025, +0.045] | 5 / 7 | 0.774 | -0.0002 [-0.0004, +0.0001] |
| ouro-1.4b | math500 | full/strict | last | strict | -0.695 [-0.755, -0.630] | 139 / 0 | 2.87e-42 | +4.8740 [+4.7466, +4.9964] |
| ouro-1.4b | math500 | full/strict | average | strict | -0.695 [-0.760, -0.630] | 139 / 0 | 2.87e-42 | +1.7757 [+1.7062, +1.8420] |
| ouro-2.6b | gsm8k | full/decode | last | convert | -0.115 [-0.170, -0.060] | 28 / 5 | 6.62e-05 | +0.1651 [+0.1349, +0.1987] |
| ouro-2.6b | gsm8k | full/decode | average | convert | -0.680 [-0.745, -0.615] | 137 / 1 | 7.98e-40 | +0.3237 [+0.3052, +0.3424] |
| ouro-2.6b | gsm8k | full/decode | last | decode | -0.035 [-0.080, +0.005] | 13 / 6 | 0.167 | +0.0086 [+0.0052, +0.0124] |
| ouro-2.6b | gsm8k | full/decode | average | decode | -0.035 [-0.075, +0.005] | 12 / 5 | 0.143 | +0.0007 [-0.0018, +0.0031] |
| ouro-2.6b | gsm8k | full/decode | last | strict | -0.330 [-0.400, -0.260] | 69 / 3 | 2.64e-17 | +0.2819 [+0.2463, +0.3212] |
| ouro-2.6b | gsm8k | full/decode | average | strict | -0.865 [-0.910, -0.815] | 173 / 0 | 1.67e-52 | +2.3797 [+2.3081, +2.4538] |
| ouro-2.6b | gsm8k | full/decode | full | strict | +0.000 [-0.015, +0.015] | 1 / 1 | 1 | +0.0000 [-0.0004, +0.0005] |
| ouro-2.6b | gsm8k | full/decode | full_altbatch | decode | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | -0.0001 [-0.0005, +0.0003] |
| ouro-2.6b | gsm8k | full/strict | last | strict | -0.330 [-0.400, -0.260] | 69 / 3 | 2.64e-17 | +0.2818 [+0.2464, +0.3207] |
| ouro-2.6b | gsm8k | full/strict | average | strict | -0.865 [-0.910, -0.815] | 173 / 0 | 1.67e-52 | +2.3797 [+2.3077, +2.4527] |
| ouro-2.6b | math500 | full/decode | last | convert | -0.535 [-0.600, -0.465] | 107 / 0 | 1.23e-32 | +1.7071 [+1.5515, +1.8627] |
| ouro-2.6b | math500 | full/decode | average | convert | -0.525 [-0.595, -0.450] | 107 / 2 | 1.85e-29 | +0.4697 [+0.4375, +0.5044] |
| ouro-2.6b | math500 | full/decode | last | decode | -0.060 [-0.130, +0.010] | 31 / 19 | 0.119 | +0.0297 [+0.0254, +0.0345] |
| ouro-2.6b | math500 | full/decode | average | decode | -0.065 [-0.130, +0.000] | 29 / 16 | 0.0725 | +0.0288 [+0.0246, +0.0333] |
| ouro-2.6b | math500 | full/decode | last | strict | -0.240 [-0.320, -0.160] | 63 / 15 | 3.75e-08 | +0.3555 [+0.3161, +0.3975] |
| ouro-2.6b | math500 | full/decode | average | strict | -0.565 [-0.635, -0.495] | 113 / 0 | 1.93e-34 | +4.0224 [+3.8850, +4.1577] |
| ouro-2.6b | math500 | full/decode | full | strict | -0.025 [-0.070, +0.020] | 13 / 8 | 0.383 | +0.0003 [+0.0001, +0.0006] |
| ouro-2.6b | math500 | full/decode | full_altbatch | decode | -0.015 [-0.045, +0.015] | 7 / 4 | 0.549 | +0.0001 [-0.0001, +0.0004] |
| ouro-2.6b | math500 | full/strict | last | strict | -0.215 [-0.290, -0.135] | 58 / 15 | 4.09e-07 | +0.3552 [+0.3156, +0.3969] |
| ouro-2.6b | math500 | full/strict | average | strict | -0.540 [-0.610, -0.470] | 108 / 0 | 6.16e-33 | +4.0220 [+3.8863, +4.1534] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | average | convert | 0.005 | 0.050 | 0.345 | 0.345 | -0.065 [-0.135, +0.005] | 0 | 1 |
| huginn-0125 | gsm8k | average | decode | 0.000 | 0.050 | 0.340 | 0.340 | -0.070 [-0.130, -0.010] | 0 | 0 |
| huginn-0125 | gsm8k | average | strict | 0.000 | 0.010 | 0.395 | 0.395 | -0.015 [-0.080, +0.050] | 0 | 0 |
| huginn-0125 | gsm8k | full | decode | 0.000 | 0.035 | 0.410 | 0.410 |  | 0 | 0 |
| huginn-0125 | gsm8k | full | strict | 0.000 | 0.035 | 0.415 | 0.415 | +0.005 [+0.000, +0.015] | 0 | 0 |
| huginn-0125 | gsm8k | full_altbatch | decode | 0.000 | 0.025 | 0.415 | 0.415 | +0.005 [-0.020, +0.030] | 0 | 0 |
| huginn-0125 | gsm8k | last | convert | 0.000 | 0.030 | 0.395 | 0.395 | -0.015 [-0.065, +0.035] | 0 | 0 |
| huginn-0125 | gsm8k | last | decode | 0.000 | 0.025 | 0.395 | 0.395 | -0.015 [-0.065, +0.035] | 0 | 0 |
| huginn-0125 | gsm8k | last | strict | 0.000 | 0.020 | 0.410 | 0.410 | +0.000 [-0.060, +0.060] | 0 | 0 |
| huginn-0125 | math500 | average | convert | 0.040 | 0.270 | 0.125 | 0.125 | -0.015 [-0.055, +0.025] | 0 | 8 |
| huginn-0125 | math500 | average | decode | 0.060 | 0.270 | 0.135 | 0.135 | -0.005 [-0.045, +0.035] | 0 | 12 |
| huginn-0125 | math500 | average | strict | 0.045 | 0.265 | 0.140 | 0.140 | +0.000 [-0.035, +0.035] | 0 | 9 |
| huginn-0125 | math500 | full | decode | 0.055 | 0.205 | 0.140 | 0.140 |  | 0 | 11 |
| huginn-0125 | math500 | full | strict | 0.045 | 0.205 | 0.140 | 0.140 | +0.000 [+0.000, +0.000] | 0 | 9 |
| huginn-0125 | math500 | full_altbatch | decode | 0.060 | 0.225 | 0.150 | 0.150 | +0.010 [-0.015, +0.035] | 0 | 12 |
| huginn-0125 | math500 | last | convert | 0.060 | 0.245 | 0.160 | 0.160 | +0.020 [-0.020, +0.060] | 0 | 12 |
| huginn-0125 | math500 | last | decode | 0.050 | 0.235 | 0.155 | 0.155 | +0.015 [-0.020, +0.055] | 0 | 10 |
| huginn-0125 | math500 | last | strict | 0.045 | 0.270 | 0.155 | 0.155 | +0.015 [-0.025, +0.055] | 0 | 9 |
| ouro-1.4b | gsm8k | average | convert | 0.000 | 0.120 | 0.230 | 0.230 | -0.590 [-0.660, -0.520] | 0 | 0 |
| ouro-1.4b | gsm8k | average | decode | 0.960 | 0.975 | 0.735 | 0.735 | -0.085 [-0.140, -0.035] | 144 | 48 |
| ouro-1.4b | gsm8k | average | strict | 0.120 | 0.150 | 0.025 | 0.035 | -0.785 [-0.840, -0.725] | 2 | 22 |
| ouro-1.4b | gsm8k | full | decode | 0.000 | 0.010 | 0.820 | 0.820 |  | 0 | 0 |
| ouro-1.4b | gsm8k | full | strict | 0.000 | 0.010 | 0.815 | 0.815 | -0.005 [-0.030, +0.020] | 0 | 0 |
| ouro-1.4b | gsm8k | full_altbatch | decode | 0.000 | 0.005 | 0.810 | 0.810 | -0.010 [-0.040, +0.020] | 0 | 0 |
| ouro-1.4b | gsm8k | last | convert | 0.120 | 0.930 | 0.000 | 0.000 | -0.820 [-0.870, -0.765] | 0 | 24 |
| ouro-1.4b | gsm8k | last | decode | 0.845 | 0.865 | 0.685 | 0.685 | -0.135 [-0.195, -0.075] | 117 | 52 |
| ouro-1.4b | gsm8k | last | strict | 0.190 | 1.000 | 0.000 | 0.000 | -0.820 [-0.870, -0.765] | 0 | 38 |
| ouro-1.4b | math500 | average | convert | 0.270 | 0.835 | 0.025 | 0.020 | -0.690 [-0.755, -0.625] | 0 | 54 |
| ouro-1.4b | math500 | average | decode | 0.980 | 1.000 | 0.320 | 0.120 | -0.590 [-0.660, -0.515] | 23 | 173 |
| ouro-1.4b | math500 | average | strict | 0.990 | 1.000 | 0.000 | 0.000 | -0.710 [-0.770, -0.645] | 0 | 198 |
| ouro-1.4b | math500 | full | decode | 0.000 | 0.075 | 0.710 | 0.710 |  | 0 | 0 |
| ouro-1.4b | math500 | full | strict | 0.000 | 0.070 | 0.695 | 0.695 | -0.015 [-0.045, +0.015] | 0 | 0 |
| ouro-1.4b | math500 | full_altbatch | decode | 0.000 | 0.085 | 0.720 | 0.720 | +0.010 [-0.025, +0.045] | 0 | 0 |
| ouro-1.4b | math500 | last | convert | 0.410 | 0.735 | 0.000 | 0.000 | -0.710 [-0.770, -0.645] | 0 | 82 |
| ouro-1.4b | math500 | last | decode | 0.860 | 0.800 | 0.160 | 0.080 | -0.630 [-0.700, -0.560] | 12 | 160 |
| ouro-1.4b | math500 | last | strict | 0.090 | 0.910 | 0.000 | 0.000 | -0.710 [-0.770, -0.645] | 0 | 18 |
| ouro-2.6b | gsm8k | average | convert | 0.025 | 0.345 | 0.195 | 0.200 | -0.675 [-0.740, -0.605] | 0 | 5 |
| ouro-2.6b | gsm8k | average | decode | 0.000 | 0.010 | 0.840 | 0.840 | -0.035 [-0.075, +0.005] | 0 | 0 |
| ouro-2.6b | gsm8k | average | strict | 0.110 | 0.770 | 0.010 | 0.015 | -0.860 [-0.905, -0.810] | 0 | 22 |
| ouro-2.6b | gsm8k | full | decode | 0.000 | 0.000 | 0.875 | 0.875 |  | 0 | 0 |
| ouro-2.6b | gsm8k | full | strict | 0.000 | 0.005 | 0.875 | 0.875 | +0.000 [-0.015, +0.015] | 0 | 0 |
| ouro-2.6b | gsm8k | full_altbatch | decode | 0.000 | 0.005 | 0.875 | 0.875 | +0.000 [+0.000, +0.000] | 0 | 0 |
| ouro-2.6b | gsm8k | last | convert | 0.005 | 0.030 | 0.760 | 0.760 | -0.115 [-0.170, -0.060] | 0 | 1 |
| ouro-2.6b | gsm8k | last | decode | 0.000 | 0.005 | 0.840 | 0.840 | -0.035 [-0.080, +0.010] | 0 | 0 |
| ouro-2.6b | gsm8k | last | strict | 0.000 | 0.035 | 0.545 | 0.545 | -0.330 [-0.400, -0.260] | 0 | 0 |
| ouro-2.6b | math500 | average | convert | 0.185 | 0.865 | 0.040 | 0.045 | -0.520 [-0.595, -0.445] | 2 | 35 |
| ouro-2.6b | math500 | average | decode | 0.030 | 0.185 | 0.500 | 0.500 | -0.065 [-0.130, +0.000] | 0 | 6 |
| ouro-2.6b | math500 | average | strict | 0.085 | 0.110 | 0.000 | 0.000 | -0.565 [-0.630, -0.495] | 0 | 17 |
| ouro-2.6b | math500 | full | decode | 0.015 | 0.105 | 0.565 | 0.565 |  | 0 | 3 |
| ouro-2.6b | math500 | full | strict | 0.020 | 0.130 | 0.540 | 0.540 | -0.025 [-0.070, +0.020] | 0 | 4 |
| ouro-2.6b | math500 | full_altbatch | decode | 0.020 | 0.105 | 0.550 | 0.550 | -0.015 [-0.050, +0.015] | 0 | 4 |
| ouro-2.6b | math500 | last | convert | 0.240 | 0.875 | 0.030 | 0.030 | -0.535 [-0.605, -0.465] | 0 | 48 |
| ouro-2.6b | math500 | last | decode | 0.010 | 0.110 | 0.505 | 0.495 | -0.070 [-0.135, -0.005] | 0 | 2 |
| ouro-2.6b | math500 | last | strict | 0.040 | 0.175 | 0.325 | 0.325 | -0.240 [-0.320, -0.160] | 0 | 8 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | average | convert | 0-63 | 200 | -0.0235 [-0.0319, -0.0151] |
| huginn-0125 | gsm8k | average | convert | 64-127 | 130 | +0.0011 [-0.0100, +0.0122] |
| huginn-0125 | gsm8k | average | convert | 128-255 | 23 | +0.0018 [-0.0181, +0.0254] |
| huginn-0125 | gsm8k | average | convert | 256-511 | 1 | -0.0083 [-0.0083, -0.0083] |
| huginn-0125 | gsm8k | average | decode | 0-63 | 200 | +0.0024 [-0.0026, +0.0075] |
| huginn-0125 | gsm8k | average | decode | 64-127 | 130 | +0.0013 [-0.0078, +0.0100] |
| huginn-0125 | gsm8k | average | decode | 128-255 | 23 | +0.0037 [-0.0134, +0.0216] |
| huginn-0125 | gsm8k | average | decode | 256-511 | 1 | -0.0345 [-0.0345, -0.0345] |
| huginn-0125 | gsm8k | average | strict | 0-63 | 200 | -0.0132 [-0.0209, -0.0055] |
| huginn-0125 | gsm8k | average | strict | 64-127 | 130 | -0.0038 [-0.0133, +0.0059] |
| huginn-0125 | gsm8k | average | strict | 128-255 | 23 | +0.0014 [-0.0226, +0.0286] |
| huginn-0125 | gsm8k | average | strict | 256-511 | 1 | -0.0402 [-0.0402, -0.0402] |
| huginn-0125 | gsm8k | full | strict | 0-63 | 200 | -0.0001 [-0.0002, +0.0001] |
| huginn-0125 | gsm8k | full | strict | 64-127 | 130 | -0.0001 [-0.0004, +0.0001] |
| huginn-0125 | gsm8k | full | strict | 128-255 | 23 | -0.0002 [-0.0008, +0.0002] |
| huginn-0125 | gsm8k | full | strict | 256-511 | 1 | +0.0000 [+0.0000, +0.0000] |
| huginn-0125 | gsm8k | full_altbatch | decode | 0-63 | 200 | +0.0000 [-0.0005, +0.0006] |
| huginn-0125 | gsm8k | full_altbatch | decode | 64-127 | 130 | +0.0006 [-0.0001, +0.0012] |
| huginn-0125 | gsm8k | full_altbatch | decode | 128-255 | 23 | -0.0002 [-0.0015, +0.0009] |
| huginn-0125 | gsm8k | full_altbatch | decode | 256-511 | 1 | +0.0067 [+0.0067, +0.0067] |
| huginn-0125 | gsm8k | last | convert | 0-63 | 200 | +0.0125 [+0.0074, +0.0180] |
| huginn-0125 | gsm8k | last | convert | 64-127 | 130 | -0.0029 [-0.0087, +0.0028] |
| huginn-0125 | gsm8k | last | convert | 128-255 | 23 | -0.0001 [-0.0091, +0.0084] |
| huginn-0125 | gsm8k | last | convert | 256-511 | 1 | -0.0173 [-0.0173, -0.0173] |
| huginn-0125 | gsm8k | last | decode | 0-63 | 200 | +0.0037 [+0.0004, +0.0072] |
| huginn-0125 | gsm8k | last | decode | 64-127 | 130 | -0.0049 [-0.0099, +0.0001] |
| huginn-0125 | gsm8k | last | decode | 128-255 | 23 | +0.0014 [-0.0051, +0.0089] |
| huginn-0125 | gsm8k | last | decode | 256-511 | 1 | -0.0078 [-0.0078, -0.0078] |
| huginn-0125 | gsm8k | last | strict | 0-63 | 200 | +0.0301 [+0.0236, +0.0371] |
| huginn-0125 | gsm8k | last | strict | 64-127 | 130 | +0.0016 [-0.0050, +0.0085] |
| huginn-0125 | gsm8k | last | strict | 128-255 | 23 | +0.0010 [-0.0068, +0.0091] |
| huginn-0125 | gsm8k | last | strict | 256-511 | 1 | -0.0277 [-0.0277, -0.0277] |
| huginn-0125 | math500 | average | convert | 0-63 | 200 | +0.0219 [+0.0176, +0.0264] |
| huginn-0125 | math500 | average | convert | 64-127 | 192 | +0.0266 [+0.0212, +0.0322] |
| huginn-0125 | math500 | average | convert | 128-255 | 136 | +0.0231 [+0.0176, +0.0287] |
| huginn-0125 | math500 | average | convert | 256-511 | 67 | +0.0232 [+0.0182, +0.0290] |
| huginn-0125 | math500 | average | convert | 512-99999 | 18 | +0.0198 [+0.0083, +0.0320] |
| huginn-0125 | math500 | average | decode | 0-63 | 200 | +0.0155 [+0.0118, +0.0193] |
| huginn-0125 | math500 | average | decode | 64-127 | 192 | +0.0157 [+0.0117, +0.0197] |
| huginn-0125 | math500 | average | decode | 128-255 | 136 | +0.0127 [+0.0082, +0.0172] |
| huginn-0125 | math500 | average | decode | 256-511 | 67 | +0.0125 [+0.0088, +0.0163] |
| huginn-0125 | math500 | average | decode | 512-99999 | 18 | +0.0098 [+0.0001, +0.0185] |
| huginn-0125 | math500 | average | strict | 0-63 | 200 | +0.0174 [+0.0133, +0.0215] |
| huginn-0125 | math500 | average | strict | 64-127 | 192 | +0.0195 [+0.0152, +0.0240] |
| huginn-0125 | math500 | average | strict | 128-255 | 136 | +0.0162 [+0.0117, +0.0211] |
| huginn-0125 | math500 | average | strict | 256-511 | 67 | +0.0157 [+0.0116, +0.0205] |
| huginn-0125 | math500 | average | strict | 512-99999 | 18 | +0.0145 [+0.0048, +0.0245] |
| huginn-0125 | math500 | full | strict | 0-63 | 200 | -0.0001 [-0.0002, +0.0001] |
| huginn-0125 | math500 | full | strict | 64-127 | 192 | +0.0000 [-0.0001, +0.0001] |
| huginn-0125 | math500 | full | strict | 128-255 | 136 | -0.0000 [-0.0002, +0.0001] |
| huginn-0125 | math500 | full | strict | 256-511 | 67 | -0.0000 [-0.0001, +0.0000] |
| huginn-0125 | math500 | full | strict | 512-99999 | 18 | -0.0001 [-0.0003, +0.0000] |
| huginn-0125 | math500 | full_altbatch | decode | 0-63 | 200 | -0.0001 [-0.0005, +0.0004] |
| huginn-0125 | math500 | full_altbatch | decode | 64-127 | 192 | -0.0002 [-0.0007, +0.0003] |
| huginn-0125 | math500 | full_altbatch | decode | 128-255 | 136 | -0.0004 [-0.0008, +0.0001] |
| huginn-0125 | math500 | full_altbatch | decode | 256-511 | 67 | +0.0003 [-0.0001, +0.0007] |
| huginn-0125 | math500 | full_altbatch | decode | 512-99999 | 18 | +0.0002 [-0.0006, +0.0008] |
| huginn-0125 | math500 | last | convert | 0-63 | 200 | -0.0046 [-0.0071, -0.0021] |
| huginn-0125 | math500 | last | convert | 64-127 | 192 | -0.0084 [-0.0119, -0.0049] |
| huginn-0125 | math500 | last | convert | 128-255 | 136 | -0.0050 [-0.0078, -0.0023] |
| huginn-0125 | math500 | last | convert | 256-511 | 67 | -0.0062 [-0.0106, -0.0019] |
| huginn-0125 | math500 | last | convert | 512-99999 | 18 | -0.0019 [-0.0077, +0.0048] |
| huginn-0125 | math500 | last | decode | 0-63 | 200 | -0.0035 [-0.0062, -0.0006] |
| huginn-0125 | math500 | last | decode | 64-127 | 192 | -0.0070 [-0.0107, -0.0033] |
| huginn-0125 | math500 | last | decode | 128-255 | 136 | -0.0055 [-0.0083, -0.0027] |
| huginn-0125 | math500 | last | decode | 256-511 | 67 | -0.0062 [-0.0104, -0.0021] |
| huginn-0125 | math500 | last | decode | 512-99999 | 18 | -0.0041 [-0.0101, +0.0022] |
| huginn-0125 | math500 | last | strict | 0-63 | 200 | -0.0076 [-0.0104, -0.0051] |
| huginn-0125 | math500 | last | strict | 64-127 | 192 | -0.0084 [-0.0121, -0.0049] |
| huginn-0125 | math500 | last | strict | 128-255 | 136 | -0.0050 [-0.0078, -0.0022] |
| huginn-0125 | math500 | last | strict | 256-511 | 67 | -0.0063 [-0.0107, -0.0020] |
| huginn-0125 | math500 | last | strict | 512-99999 | 18 | -0.0014 [-0.0070, +0.0051] |
| ouro-1.4b | gsm8k | average | convert | 0-63 | 200 | +0.2169 [+0.2011, +0.2325] |
| ouro-1.4b | gsm8k | average | convert | 64-127 | 143 | +0.4021 [+0.3698, +0.4350] |
| ouro-1.4b | gsm8k | average | convert | 128-255 | 37 | +0.4693 [+0.3925, +0.5388] |
| ouro-1.4b | gsm8k | average | convert | 256-511 | 1 | +0.6006 [+0.6006, +0.6006] |
| ouro-1.4b | gsm8k | average | decode | 0-63 | 200 | +0.0052 [+0.0015, +0.0089] |
| ouro-1.4b | gsm8k | average | decode | 64-127 | 143 | +0.0168 [+0.0115, +0.0223] |
| ouro-1.4b | gsm8k | average | decode | 128-255 | 37 | +0.0185 [+0.0066, +0.0287] |
| ouro-1.4b | gsm8k | average | decode | 256-511 | 1 | -0.0164 [-0.0164, -0.0164] |
| ouro-1.4b | gsm8k | average | strict | 0-63 | 200 | +0.7933 [+0.7617, +0.8249] |
| ouro-1.4b | gsm8k | average | strict | 64-127 | 143 | +0.8347 [+0.7895, +0.8797] |
| ouro-1.4b | gsm8k | average | strict | 128-255 | 37 | +0.7908 [+0.6815, +0.8826] |
| ouro-1.4b | gsm8k | average | strict | 256-511 | 1 | +0.6370 [+0.6370, +0.6370] |
| ouro-1.4b | gsm8k | full | strict | 0-63 | 200 | -0.0009 [-0.0015, -0.0003] |
| ouro-1.4b | gsm8k | full | strict | 64-127 | 143 | -0.0005 [-0.0011, +0.0001] |
| ouro-1.4b | gsm8k | full | strict | 128-255 | 37 | +0.0017 [+0.0000, +0.0034] |
| ouro-1.4b | gsm8k | full | strict | 256-511 | 1 | -0.0024 [-0.0024, -0.0024] |
| ouro-1.4b | gsm8k | full_altbatch | decode | 0-63 | 200 | -0.0002 [-0.0008, +0.0005] |
| ouro-1.4b | gsm8k | full_altbatch | decode | 64-127 | 143 | -0.0000 [-0.0006, +0.0006] |
| ouro-1.4b | gsm8k | full_altbatch | decode | 128-255 | 37 | +0.0001 [-0.0016, +0.0018] |
| ouro-1.4b | gsm8k | full_altbatch | decode | 256-511 | 1 | -0.0004 [-0.0004, -0.0004] |
| ouro-1.4b | gsm8k | last | convert | 0-63 | 200 | +2.6245 [+2.5353, +2.7142] |
| ouro-1.4b | gsm8k | last | convert | 64-127 | 143 | +2.9709 [+2.8542, +3.0846] |
| ouro-1.4b | gsm8k | last | convert | 128-255 | 37 | +3.2581 [+2.9999, +3.5319] |
| ouro-1.4b | gsm8k | last | convert | 256-511 | 1 | +3.6327 [+3.6327, +3.6327] |
| ouro-1.4b | gsm8k | last | decode | 0-63 | 200 | +0.0158 [+0.0108, +0.0210] |
| ouro-1.4b | gsm8k | last | decode | 64-127 | 143 | +0.0306 [+0.0195, +0.0438] |
| ouro-1.4b | gsm8k | last | decode | 128-255 | 37 | +0.0267 [+0.0089, +0.0451] |
| ouro-1.4b | gsm8k | last | decode | 256-511 | 1 | -0.0250 [-0.0250, -0.0250] |
| ouro-1.4b | gsm8k | last | strict | 0-63 | 200 | +5.4890 [+5.3874, +5.5893] |
| ouro-1.4b | gsm8k | last | strict | 64-127 | 143 | +5.5920 [+5.4395, +5.7410] |
| ouro-1.4b | gsm8k | last | strict | 128-255 | 37 | +5.7187 [+5.3205, +6.1294] |
| ouro-1.4b | gsm8k | last | strict | 256-511 | 1 | +4.5675 [+4.5675, +4.5675] |
| ouro-1.4b | math500 | average | convert | 0-63 | 200 | +0.4626 [+0.4346, +0.4904] |
| ouro-1.4b | math500 | average | convert | 64-127 | 193 | +0.6814 [+0.6428, +0.7217] |
| ouro-1.4b | math500 | average | convert | 128-255 | 144 | +0.7652 [+0.7198, +0.8154] |
| ouro-1.4b | math500 | average | convert | 256-511 | 70 | +0.8683 [+0.7935, +0.9454] |
| ouro-1.4b | math500 | average | convert | 512-99999 | 21 | +1.1214 [+0.9835, +1.2666] |
| ouro-1.4b | math500 | average | decode | 0-63 | 200 | +0.0251 [+0.0179, +0.0326] |
| ouro-1.4b | math500 | average | decode | 64-127 | 193 | +0.0430 [+0.0325, +0.0549] |
| ouro-1.4b | math500 | average | decode | 128-255 | 144 | +0.0548 [+0.0422, +0.0685] |
| ouro-1.4b | math500 | average | decode | 256-511 | 70 | +0.0679 [+0.0481, +0.0915] |
| ouro-1.4b | math500 | average | decode | 512-99999 | 21 | +0.0552 [+0.0383, +0.0763] |
| ouro-1.4b | math500 | average | strict | 0-63 | 200 | +1.4233 [+1.3647, +1.4864] |
| ouro-1.4b | math500 | average | strict | 64-127 | 193 | +1.7240 [+1.6481, +1.8036] |
| ouro-1.4b | math500 | average | strict | 128-255 | 144 | +1.8552 [+1.7696, +1.9450] |
| ouro-1.4b | math500 | average | strict | 256-511 | 70 | +2.0040 [+1.8839, +2.1228] |
| ouro-1.4b | math500 | average | strict | 512-99999 | 21 | +2.2440 [+2.0744, +2.4182] |
| ouro-1.4b | math500 | full | strict | 0-63 | 200 | +0.0003 [-0.0003, +0.0010] |
| ouro-1.4b | math500 | full | strict | 64-127 | 193 | -0.0002 [-0.0006, +0.0003] |
| ouro-1.4b | math500 | full | strict | 128-255 | 144 | +0.0003 [-0.0002, +0.0008] |
| ouro-1.4b | math500 | full | strict | 256-511 | 70 | -0.0001 [-0.0007, +0.0004] |
| ouro-1.4b | math500 | full | strict | 512-99999 | 21 | +0.0002 [-0.0006, +0.0009] |
| ouro-1.4b | math500 | full_altbatch | decode | 0-63 | 200 | +0.0001 [-0.0005, +0.0008] |
| ouro-1.4b | math500 | full_altbatch | decode | 64-127 | 193 | -0.0002 [-0.0008, +0.0004] |
| ouro-1.4b | math500 | full_altbatch | decode | 128-255 | 144 | -0.0006 [-0.0011, -0.0001] |
| ouro-1.4b | math500 | full_altbatch | decode | 256-511 | 70 | -0.0001 [-0.0008, +0.0005] |
| ouro-1.4b | math500 | full_altbatch | decode | 512-99999 | 21 | +0.0003 [-0.0005, +0.0009] |
| ouro-1.4b | math500 | last | convert | 0-63 | 200 | +2.1693 [+2.0467, +2.2902] |
| ouro-1.4b | math500 | last | convert | 64-127 | 193 | +2.0582 [+1.9362, +2.1832] |
| ouro-1.4b | math500 | last | convert | 128-255 | 144 | +2.0302 [+1.8920, +2.1659] |
| ouro-1.4b | math500 | last | convert | 256-511 | 70 | +2.0738 [+1.8871, +2.2606] |
| ouro-1.4b | math500 | last | convert | 512-99999 | 21 | +1.8905 [+1.7229, +2.1151] |
| ouro-1.4b | math500 | last | decode | 0-63 | 200 | +0.0348 [+0.0252, +0.0449] |
| ouro-1.4b | math500 | last | decode | 64-127 | 193 | +0.0683 [+0.0552, +0.0822] |
| ouro-1.4b | math500 | last | decode | 128-255 | 144 | +0.0840 [+0.0643, +0.1079] |
| ouro-1.4b | math500 | last | decode | 256-511 | 70 | +0.1079 [+0.0725, +0.1526] |
| ouro-1.4b | math500 | last | decode | 512-99999 | 21 | +0.1024 [+0.0700, +0.1467] |
| ouro-1.4b | math500 | last | strict | 0-63 | 200 | +4.9442 [+4.8288, +5.0614] |
| ouro-1.4b | math500 | last | strict | 64-127 | 193 | +4.9315 [+4.8012, +5.0616] |
| ouro-1.4b | math500 | last | strict | 128-255 | 144 | +4.9355 [+4.7817, +5.0854] |
| ouro-1.4b | math500 | last | strict | 256-511 | 70 | +4.7977 [+4.5702, +5.0368] |
| ouro-1.4b | math500 | last | strict | 512-99999 | 21 | +4.4521 [+3.9853, +4.9784] |
| ouro-2.6b | gsm8k | average | convert | 0-63 | 200 | +0.3245 [+0.3018, +0.3473] |
| ouro-2.6b | gsm8k | average | convert | 64-127 | 143 | +0.3238 [+0.2923, +0.3552] |
| ouro-2.6b | gsm8k | average | convert | 128-255 | 37 | +0.3159 [+0.2652, +0.3757] |
| ouro-2.6b | gsm8k | average | convert | 256-511 | 1 | +0.2967 [+0.2967, +0.2967] |
| ouro-2.6b | gsm8k | average | decode | 0-63 | 200 | -0.0021 [-0.0054, +0.0009] |
| ouro-2.6b | gsm8k | average | decode | 64-127 | 143 | +0.0054 [+0.0015, +0.0091] |
| ouro-2.6b | gsm8k | average | decode | 128-255 | 37 | +0.0074 [-0.0009, +0.0171] |
| ouro-2.6b | gsm8k | average | decode | 256-511 | 1 | -0.0124 [-0.0124, -0.0124] |
| ouro-2.6b | gsm8k | average | strict | 0-63 | 200 | +2.2995 [+2.2406, +2.3602] |
| ouro-2.6b | gsm8k | average | strict | 64-127 | 143 | +2.4924 [+2.3890, +2.5957] |
| ouro-2.6b | gsm8k | average | strict | 128-255 | 37 | +2.7055 [+2.3592, +3.0727] |
| ouro-2.6b | gsm8k | average | strict | 256-511 | 1 | +1.3059 [+1.3059, +1.3059] |
| ouro-2.6b | gsm8k | full | strict | 0-63 | 200 | +0.0001 [-0.0005, +0.0007] |
| ouro-2.6b | gsm8k | full | strict | 64-127 | 143 | +0.0001 [-0.0007, +0.0008] |
| ouro-2.6b | gsm8k | full | strict | 128-255 | 37 | -0.0010 [-0.0018, -0.0000] |
| ouro-2.6b | gsm8k | full | strict | 256-511 | 1 | +0.0026 [+0.0026, +0.0026] |
| ouro-2.6b | gsm8k | full_altbatch | decode | 0-63 | 200 | -0.0000 [-0.0006, +0.0005] |
| ouro-2.6b | gsm8k | full_altbatch | decode | 64-127 | 143 | -0.0001 [-0.0009, +0.0006] |
| ouro-2.6b | gsm8k | full_altbatch | decode | 128-255 | 37 | -0.0000 [-0.0016, +0.0016] |
| ouro-2.6b | gsm8k | full_altbatch | decode | 256-511 | 1 | +0.0027 [+0.0027, +0.0027] |
| ouro-2.6b | gsm8k | last | convert | 0-63 | 200 | +0.1530 [+0.1213, +0.1888] |
| ouro-2.6b | gsm8k | last | convert | 64-127 | 143 | +0.1963 [+0.1541, +0.2434] |
| ouro-2.6b | gsm8k | last | convert | 128-255 | 37 | +0.1363 [+0.0808, +0.1951] |
| ouro-2.6b | gsm8k | last | convert | 256-511 | 1 | -0.0010 [-0.0010, -0.0010] |
| ouro-2.6b | gsm8k | last | decode | 0-63 | 200 | +0.0035 [-0.0003, +0.0071] |
| ouro-2.6b | gsm8k | last | decode | 64-127 | 143 | +0.0194 [+0.0113, +0.0281] |
| ouro-2.6b | gsm8k | last | decode | 128-255 | 37 | +0.0091 [-0.0087, +0.0302] |
| ouro-2.6b | gsm8k | last | decode | 256-511 | 1 | -0.0048 [-0.0048, -0.0048] |
| ouro-2.6b | gsm8k | last | strict | 0-63 | 200 | +0.2891 [+0.2526, +0.3275] |
| ouro-2.6b | gsm8k | last | strict | 64-127 | 143 | +0.2852 [+0.2315, +0.3421] |
| ouro-2.6b | gsm8k | last | strict | 128-255 | 37 | +0.1912 [+0.0846, +0.3022] |
| ouro-2.6b | gsm8k | last | strict | 256-511 | 1 | -0.0429 [-0.0429, -0.0429] |
| ouro-2.6b | math500 | average | convert | 0-63 | 200 | +0.4312 [+0.4019, +0.4623] |
| ouro-2.6b | math500 | average | convert | 64-127 | 193 | +0.4262 [+0.3927, +0.4613] |
| ouro-2.6b | math500 | average | convert | 128-255 | 144 | +0.4852 [+0.4287, +0.5519] |
| ouro-2.6b | math500 | average | convert | 256-511 | 70 | +0.5357 [+0.4604, +0.6183] |
| ouro-2.6b | math500 | average | convert | 512-99999 | 21 | +0.5007 [+0.4393, +0.5763] |
| ouro-2.6b | math500 | average | decode | 0-63 | 200 | +0.0198 [+0.0156, +0.0241] |
| ouro-2.6b | math500 | average | decode | 64-127 | 193 | +0.0255 [+0.0199, +0.0316] |
| ouro-2.6b | math500 | average | decode | 128-255 | 144 | +0.0334 [+0.0256, +0.0417] |
| ouro-2.6b | math500 | average | decode | 256-511 | 70 | +0.0368 [+0.0275, +0.0484] |
| ouro-2.6b | math500 | average | decode | 512-99999 | 21 | +0.0313 [+0.0226, +0.0397] |
| ouro-2.6b | math500 | average | strict | 0-63 | 200 | +3.6610 [+3.5472, +3.7748] |
| ouro-2.6b | math500 | average | strict | 64-127 | 193 | +3.8010 [+3.6670, +3.9356] |
| ouro-2.6b | math500 | average | strict | 128-255 | 144 | +4.0440 [+3.8861, +4.2057] |
| ouro-2.6b | math500 | average | strict | 256-511 | 70 | +4.4762 [+4.2476, +4.6972] |
| ouro-2.6b | math500 | average | strict | 512-99999 | 21 | +4.6380 [+4.2950, +4.9884] |
| ouro-2.6b | math500 | full | strict | 0-63 | 200 | +0.0007 [+0.0000, +0.0014] |
| ouro-2.6b | math500 | full | strict | 64-127 | 193 | +0.0004 [-0.0001, +0.0009] |
| ouro-2.6b | math500 | full | strict | 128-255 | 144 | +0.0003 [-0.0001, +0.0007] |
| ouro-2.6b | math500 | full | strict | 256-511 | 70 | -0.0003 [-0.0008, +0.0002] |
| ouro-2.6b | math500 | full | strict | 512-99999 | 21 | +0.0003 [-0.0005, +0.0013] |
| ouro-2.6b | math500 | full_altbatch | decode | 0-63 | 200 | +0.0003 [-0.0003, +0.0009] |
| ouro-2.6b | math500 | full_altbatch | decode | 64-127 | 193 | +0.0004 [-0.0001, +0.0008] |
| ouro-2.6b | math500 | full_altbatch | decode | 128-255 | 144 | -0.0000 [-0.0005, +0.0004] |
| ouro-2.6b | math500 | full_altbatch | decode | 256-511 | 70 | -0.0002 [-0.0007, +0.0004] |
| ouro-2.6b | math500 | full_altbatch | decode | 512-99999 | 21 | +0.0005 [-0.0007, +0.0022] |
| ouro-2.6b | math500 | last | convert | 0-63 | 200 | +1.4932 [+1.3560, +1.6409] |
| ouro-2.6b | math500 | last | convert | 64-127 | 193 | +1.5636 [+1.4161, +1.7095] |
| ouro-2.6b | math500 | last | convert | 128-255 | 144 | +1.8086 [+1.6061, +2.0219] |
| ouro-2.6b | math500 | last | convert | 256-511 | 70 | +1.9357 [+1.6465, +2.2355] |
| ouro-2.6b | math500 | last | convert | 512-99999 | 21 | +1.8924 [+1.3705, +2.4654] |
| ouro-2.6b | math500 | last | decode | 0-63 | 200 | +0.0206 [+0.0153, +0.0265] |
| ouro-2.6b | math500 | last | decode | 64-127 | 193 | +0.0317 [+0.0243, +0.0403] |
| ouro-2.6b | math500 | last | decode | 128-255 | 144 | +0.0352 [+0.0267, +0.0456] |
| ouro-2.6b | math500 | last | decode | 256-511 | 70 | +0.0329 [+0.0240, +0.0430] |
| ouro-2.6b | math500 | last | decode | 512-99999 | 21 | +0.0271 [+0.0176, +0.0373] |
| ouro-2.6b | math500 | last | strict | 0-63 | 200 | +0.3534 [+0.3137, +0.3951] |
| ouro-2.6b | math500 | last | strict | 64-127 | 193 | +0.3178 [+0.2788, +0.3597] |
| ouro-2.6b | math500 | last | strict | 128-255 | 144 | +0.3611 [+0.2995, +0.4354] |
| ouro-2.6b | math500 | last | strict | 256-511 | 70 | +0.3750 [+0.2953, +0.4601] |
| ouro-2.6b | math500 | last | strict | 512-99999 | 21 | +0.4002 [+0.2567, +0.5651] |
