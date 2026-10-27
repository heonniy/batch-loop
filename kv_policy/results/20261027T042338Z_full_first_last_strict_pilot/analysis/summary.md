Grader: kvp/grading.py sha256 c188871fe74c (all generations re-scored at analysis time)

## Per-cell summary

| model | dataset | policy | mode | n | acc | acc (format only) | fallback | no answer: truncated / finished | ref scorer | NLL tok-w | NLL/problem | mean gen tok | hit max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full | decode | 32 | 0.500 | 0.500 | 0.000 | 1 / 0 | 0.438 | 0.9200 | 0.9468 | 111.3 | 0.031 |
| huginn-0125 | gsm8k | last | strict | 32 | 0.500 | 0.500 | 0.000 | 1 / 0 | 0.438 | 0.9254 | 0.9521 | 112.5 | 0.031 |
| huginn-0125 | gsm8k | first | strict | 32 | 0.031 | 0.000 | 0.125 | 3 / 18 | 0.000 | 2.9661 | 3.0300 | 72.6 | 0.094 |
| huginn-0125 | gsm8k | full | strict | 32 | 0.500 | 0.500 | 0.000 | 1 / 0 | 0.438 | 0.9200 | 0.9468 | 108.9 | 0.031 |
| huginn-0125 | math500 | full | decode | 32 | 0.406 | 0.406 | 0.094 | 7 / 0 | 0.406 | 0.7590 | 0.7528 | 565.6 | 0.219 |
| huginn-0125 | math500 | last | strict | 32 | 0.344 | 0.344 | 0.062 | 7 / 0 | 0.344 | 0.7577 | 0.7521 | 574.8 | 0.219 |
| huginn-0125 | math500 | first | strict | 32 | 0.062 | 0.062 | 0.000 | 3 / 0 | 0.062 | 2.2375 | 2.2255 | 311.9 | 0.125 |
| huginn-0125 | math500 | full | strict | 32 | 0.375 | 0.375 | 0.094 | 7 / 0 | 0.375 | 0.7588 | 0.7525 | 555.9 | 0.219 |
| ouro-1.4b | gsm8k | full | decode | 32 | 0.812 | 0.781 | 0.031 | 0 / 0 | 0.531 | 0.7495 | 0.7616 | 104.3 | 0.000 |
| ouro-1.4b | gsm8k | last | strict | 32 | 0.000 | 0.000 | 0.031 | 31 / 0 | 0.000 | 6.2267 | 6.2241 | 505.5 | 0.969 |
| ouro-1.4b | gsm8k | first | strict | 32 | 0.281 | 0.250 | 0.062 | 1 / 0 | 0.250 | 1.1823 | 1.1872 | 105.7 | 0.031 |
| ouro-1.4b | gsm8k | full | strict | 32 | 0.781 | 0.750 | 0.031 | 0 / 0 | 0.531 | 0.7498 | 0.7618 | 109.3 | 0.000 |
| ouro-1.4b | math500 | full | decode | 32 | 0.844 | 0.844 | 0.000 | 0 / 0 | 0.844 | 0.6787 | 0.6845 | 418.2 | 0.000 |
| ouro-1.4b | math500 | last | strict | 32 | 0.000 | 0.000 | 0.000 | 30 / 2 | 0.000 | 5.6639 | 5.6297 | 1996.7 | 0.938 |
| ouro-1.4b | math500 | first | strict | 32 | 0.094 | 0.094 | 0.000 | 20 / 1 | 0.094 | 1.3972 | 1.3612 | 1629.7 | 0.719 |
| ouro-1.4b | math500 | full | strict | 32 | 0.812 | 0.812 | 0.000 | 0 / 0 | 0.812 | 0.6773 | 0.6829 | 423.3 | 0.000 |
| ouro-2.6b | gsm8k | full | decode | 32 | 0.812 | 0.812 | 0.000 | 0 / 0 | 0.625 | 0.7122 | 0.7229 | 107.6 | 0.000 |
| ouro-2.6b | gsm8k | last | strict | 32 | 0.500 | 0.500 | 0.000 | 1 / 0 | 0.406 | 0.9934 | 0.9996 | 111.8 | 0.031 |
| ouro-2.6b | gsm8k | first | strict | 32 | 0.031 | 0.000 | 0.625 | 2 / 0 | 0.000 | 1.5600 | 1.5677 | 126.4 | 0.062 |
| ouro-2.6b | gsm8k | full | strict | 32 | 0.812 | 0.812 | 0.000 | 0 / 0 | 0.656 | 0.7115 | 0.7221 | 107.2 | 0.000 |
| ouro-2.6b | math500 | full | decode | 32 | 0.625 | 0.625 | 0.000 | 2 / 0 | 0.625 | 0.6006 | 0.5893 | 353.9 | 0.062 |
| ouro-2.6b | math500 | last | strict | 32 | 0.500 | 0.500 | 0.000 | 1 / 0 | 0.469 | 1.0223 | 0.9246 | 193.1 | 0.031 |
| ouro-2.6b | math500 | first | strict | 32 | 0.156 | 0.156 | 0.000 | 10 / 0 | 0.156 | 1.5927 | 1.5295 | 1764.3 | 0.812 |
| ouro-2.6b | math500 | full | strict | 32 | 0.562 | 0.562 | 0.000 | 2 / 0 | 0.562 | 0.6006 | 0.5899 | 332.4 | 0.062 |

## Paired deltas vs baseline

| model | dataset | baseline | policy | mode | Δacc [95% CI] | base-only / policy-only | McNemar p | ΔNLL tok-w [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | full/decode | last | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | +0.0054 [+0.0001, +0.0132] |
| huginn-0125 | gsm8k | full/decode | first | strict | -0.469 [-0.656, -0.281] | 16 / 1 | 0.000275 | +2.0460 [+1.9170, +2.1778] |
| huginn-0125 | gsm8k | full/decode | full | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | -0.0000 [-0.0010, +0.0010] |
| huginn-0125 | gsm8k | full/strict | last | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | +0.0054 [+0.0003, +0.0131] |
| huginn-0125 | math500 | full/decode | last | strict | -0.062 [-0.156, +0.000] | 2 / 0 | 0.5 | -0.0012 [-0.0044, +0.0007] |
| huginn-0125 | math500 | full/decode | first | strict | -0.344 [-0.531, -0.156] | 12 / 1 | 0.00342 | +1.4785 [+1.3927, +1.5621] |
| huginn-0125 | math500 | full/decode | full | strict | -0.031 [-0.125, +0.062] | 2 / 1 | 1 | -0.0002 [-0.0006, +0.0003] |
| huginn-0125 | math500 | full/strict | last | strict | -0.031 [-0.094, +0.000] | 1 / 0 | 1 | -0.0011 [-0.0042, +0.0009] |
| ouro-1.4b | gsm8k | full/decode | last | strict | -0.812 [-0.938, -0.656] | 26 / 0 | 2.98e-08 | +5.4771 [+5.1625, +5.8138] |
| ouro-1.4b | gsm8k | full/decode | first | strict | -0.531 [-0.719, -0.312] | 19 / 2 | 0.000221 | +0.4328 [+0.3825, +0.4826] |
| ouro-1.4b | gsm8k | full/decode | full | strict | -0.031 [-0.125, +0.062] | 2 / 1 | 1 | +0.0003 [-0.0011, +0.0016] |
| ouro-1.4b | gsm8k | full/strict | last | strict | -0.781 [-0.906, -0.625] | 25 / 0 | 5.96e-08 | +5.4768 [+5.1604, +5.8232] |
| ouro-1.4b | math500 | full/decode | last | strict | -0.844 [-0.969, -0.719] | 27 / 0 | 1.49e-08 | +4.9851 [+4.7354, +5.2284] |
| ouro-1.4b | math500 | full/decode | first | strict | -0.750 [-0.906, -0.594] | 24 / 0 | 1.19e-07 | +0.7185 [+0.6259, +0.8143] |
| ouro-1.4b | math500 | full/decode | full | strict | -0.031 [-0.125, +0.062] | 2 / 1 | 1 | -0.0015 [-0.0025, -0.0006] |
| ouro-1.4b | math500 | full/strict | last | strict | -0.812 [-0.938, -0.656] | 26 / 0 | 2.98e-08 | +4.9866 [+4.7382, +5.2271] |
| ouro-2.6b | gsm8k | full/decode | last | strict | -0.312 [-0.469, -0.156] | 10 / 0 | 0.00195 | +0.2813 [+0.2107, +0.3599] |
| ouro-2.6b | gsm8k | full/decode | first | strict | -0.781 [-0.906, -0.625] | 25 / 0 | 5.96e-08 | +0.8479 [+0.7631, +0.9364] |
| ouro-2.6b | gsm8k | full/decode | full | strict | +0.000 [+0.000, +0.000] | 0 / 0 | 1 | -0.0006 [-0.0019, +0.0007] |
| ouro-2.6b | gsm8k | full/strict | last | strict | -0.312 [-0.469, -0.156] | 10 / 0 | 0.00195 | +0.2819 [+0.2110, +0.3599] |
| ouro-2.6b | math500 | full/decode | last | strict | -0.125 [-0.312, +0.062] | 7 / 3 | 0.344 | +0.4217 [+0.3008, +0.5395] |
| ouro-2.6b | math500 | full/decode | first | strict | -0.469 [-0.656, -0.281] | 16 / 1 | 0.000275 | +0.9920 [+0.8905, +1.0884] |
| ouro-2.6b | math500 | full/decode | full | strict | -0.062 [-0.188, +0.062] | 3 / 1 | 0.625 | +0.0000 [-0.0006, +0.0008] |
| ouro-2.6b | math500 | full/strict | last | strict | -0.062 [-0.281, +0.156] | 8 / 6 | 0.791 | +0.4217 [+0.2972, +0.5430] |

## Degeneration and a policy-independent stop rule (secondary)

Stop rule: cut every generation at the first '\n\n\n' and re-score (identical for all cells; greedy decoding makes this equal to stopping online). Main accuracy is unchanged.

| model | dataset | policy | mode | degenerate | hit max | acc main | acc stop-rule | Δ stop-rule vs FULL [95% CI] | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | first | strict | 0.000 | 0.094 | 0.031 | 0.031 | -0.469 [-0.656, -0.281] | 0 | 0 |
| huginn-0125 | gsm8k | full | decode | 0.000 | 0.031 | 0.500 | 0.500 |  | 0 | 0 |
| huginn-0125 | gsm8k | full | strict | 0.000 | 0.031 | 0.500 | 0.500 | +0.000 [+0.000, +0.000] | 0 | 0 |
| huginn-0125 | gsm8k | last | strict | 0.000 | 0.031 | 0.500 | 0.500 | +0.000 [+0.000, +0.000] | 0 | 0 |
| huginn-0125 | math500 | first | strict | 0.000 | 0.125 | 0.062 | 0.062 | -0.344 [-0.531, -0.156] | 0 | 0 |
| huginn-0125 | math500 | full | decode | 0.062 | 0.219 | 0.406 | 0.406 |  | 0 | 2 |
| huginn-0125 | math500 | full | strict | 0.031 | 0.219 | 0.375 | 0.375 | -0.031 [-0.125, +0.062] | 0 | 1 |
| huginn-0125 | math500 | last | strict | 0.031 | 0.219 | 0.344 | 0.344 | -0.062 [-0.156, +0.000] | 0 | 1 |
| ouro-1.4b | gsm8k | first | strict | 0.000 | 0.031 | 0.281 | 0.281 | -0.531 [-0.719, -0.312] | 0 | 0 |
| ouro-1.4b | gsm8k | full | decode | 0.000 | 0.000 | 0.812 | 0.812 |  | 0 | 0 |
| ouro-1.4b | gsm8k | full | strict | 0.000 | 0.000 | 0.781 | 0.781 | -0.031 [-0.125, +0.062] | 0 | 0 |
| ouro-1.4b | gsm8k | last | strict | 0.406 | 0.969 | 0.000 | 0.000 | -0.812 [-0.938, -0.656] | 0 | 13 |
| ouro-1.4b | math500 | first | strict | 0.062 | 0.719 | 0.094 | 0.094 | -0.750 [-0.875, -0.594] | 0 | 2 |
| ouro-1.4b | math500 | full | decode | 0.000 | 0.000 | 0.844 | 0.844 |  | 0 | 0 |
| ouro-1.4b | math500 | full | strict | 0.000 | 0.000 | 0.812 | 0.812 | -0.031 [-0.156, +0.062] | 0 | 0 |
| ouro-1.4b | math500 | last | strict | 0.156 | 0.938 | 0.000 | 0.000 | -0.844 [-0.969, -0.719] | 0 | 5 |
| ouro-2.6b | gsm8k | first | strict | 0.000 | 0.062 | 0.031 | 0.031 | -0.781 [-0.906, -0.625] | 0 | 0 |
| ouro-2.6b | gsm8k | full | decode | 0.000 | 0.000 | 0.812 | 0.812 |  | 0 | 0 |
| ouro-2.6b | gsm8k | full | strict | 0.000 | 0.000 | 0.812 | 0.812 | +0.000 [+0.000, +0.000] | 0 | 0 |
| ouro-2.6b | gsm8k | last | strict | 0.000 | 0.031 | 0.500 | 0.500 | -0.312 [-0.469, -0.156] | 0 | 0 |
| ouro-2.6b | math500 | first | strict | 0.000 | 0.812 | 0.156 | 0.156 | -0.469 [-0.656, -0.281] | 0 | 0 |
| ouro-2.6b | math500 | full | decode | 0.031 | 0.062 | 0.625 | 0.625 |  | 0 | 1 |
| ouro-2.6b | math500 | full | strict | 0.000 | 0.062 | 0.562 | 0.562 | -0.062 [-0.188, +0.062] | 0 | 0 |
| ouro-2.6b | math500 | last | strict | 0.000 | 0.031 | 0.500 | 0.500 | -0.125 [-0.312, +0.062] | 0 | 0 |

## Teacher-forced ΔNLL per token vs FULL/decode by gold-target position

| model | dataset | policy | mode | positions | problems | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|
| huginn-0125 | gsm8k | first | strict | 0-63 | 32 | +2.0919 [+1.9425, +2.2483] |
| huginn-0125 | gsm8k | first | strict | 64-127 | 21 | +1.9204 [+1.7249, +2.1010] |
| huginn-0125 | gsm8k | first | strict | 128-255 | 5 | +2.0452 [+1.7541, +2.2875] |
| huginn-0125 | gsm8k | full | strict | 0-63 | 32 | +0.0001 [-0.0010, +0.0013] |
| huginn-0125 | gsm8k | full | strict | 64-127 | 21 | +0.0000 [-0.0018, +0.0021] |
| huginn-0125 | gsm8k | full | strict | 128-255 | 5 | -0.0015 [-0.0034, -0.0001] |
| huginn-0125 | gsm8k | last | strict | 0-63 | 32 | +0.0042 [+0.0007, +0.0090] |
| huginn-0125 | gsm8k | last | strict | 64-127 | 21 | +0.0086 [-0.0009, +0.0290] |
| huginn-0125 | gsm8k | last | strict | 128-255 | 5 | +0.0044 [-0.0083, +0.0310] |
| huginn-0125 | math500 | first | strict | 0-63 | 32 | +1.5618 [+1.4417, +1.6840] |
| huginn-0125 | math500 | first | strict | 64-127 | 31 | +1.3691 [+1.2580, +1.4861] |
| huginn-0125 | math500 | first | strict | 128-255 | 21 | +1.4138 [+1.2732, +1.5449] |
| huginn-0125 | math500 | first | strict | 256-511 | 11 | +1.5949 [+1.4040, +1.7576] |
| huginn-0125 | math500 | first | strict | 512-99999 | 3 | +1.4207 [+1.3879, +1.4906] |
| huginn-0125 | math500 | full | strict | 0-63 | 32 | -0.0004 [-0.0014, +0.0006] |
| huginn-0125 | math500 | full | strict | 64-127 | 31 | -0.0001 [-0.0013, +0.0011] |
| huginn-0125 | math500 | full | strict | 128-255 | 21 | -0.0003 [-0.0010, +0.0004] |
| huginn-0125 | math500 | full | strict | 256-511 | 11 | +0.0003 [-0.0013, +0.0022] |
| huginn-0125 | math500 | full | strict | 512-99999 | 3 | +0.0002 [-0.0004, +0.0010] |
| huginn-0125 | math500 | last | strict | 0-63 | 32 | -0.0017 [-0.0074, +0.0020] |
| huginn-0125 | math500 | last | strict | 64-127 | 31 | -0.0015 [-0.0072, +0.0020] |
| huginn-0125 | math500 | last | strict | 128-255 | 21 | -0.0018 [-0.0031, -0.0003] |
| huginn-0125 | math500 | last | strict | 256-511 | 11 | +0.0000 [-0.0017, +0.0026] |
| huginn-0125 | math500 | last | strict | 512-99999 | 3 | +0.0009 [-0.0014, +0.0020] |
| ouro-1.4b | gsm8k | first | strict | 0-63 | 32 | +0.4007 [+0.3392, +0.4690] |
| ouro-1.4b | gsm8k | first | strict | 64-127 | 24 | +0.4667 [+0.3757, +0.5466] |
| ouro-1.4b | gsm8k | first | strict | 128-255 | 6 | +0.5526 [+0.3727, +0.6922] |
| ouro-1.4b | gsm8k | full | strict | 0-63 | 32 | -0.0000 [-0.0018, +0.0017] |
| ouro-1.4b | gsm8k | full | strict | 64-127 | 24 | +0.0012 [-0.0013, +0.0037] |
| ouro-1.4b | gsm8k | full | strict | 128-255 | 6 | -0.0000 [-0.0041, +0.0036] |
| ouro-1.4b | gsm8k | last | strict | 0-63 | 32 | +5.4788 [+5.1648, +5.8094] |
| ouro-1.4b | gsm8k | last | strict | 64-127 | 24 | +5.4373 [+4.9819, +5.8923] |
| ouro-1.4b | gsm8k | last | strict | 128-255 | 6 | +5.5798 [+5.0025, +6.4156] |
| ouro-1.4b | math500 | first | strict | 0-63 | 32 | +0.5261 [+0.4460, +0.6089] |
| ouro-1.4b | math500 | first | strict | 64-127 | 31 | +0.7193 [+0.6207, +0.8256] |
| ouro-1.4b | math500 | first | strict | 128-255 | 23 | +0.7449 [+0.6441, +0.8519] |
| ouro-1.4b | math500 | first | strict | 256-511 | 11 | +0.9036 [+0.6782, +1.1679] |
| ouro-1.4b | math500 | first | strict | 512-99999 | 3 | +0.9021 [+0.6353, +1.3450] |
| ouro-1.4b | math500 | full | strict | 0-63 | 32 | -0.0017 [-0.0035, -0.0001] |
| ouro-1.4b | math500 | full | strict | 64-127 | 31 | -0.0016 [-0.0030, -0.0003] |
| ouro-1.4b | math500 | full | strict | 128-255 | 23 | -0.0013 [-0.0031, +0.0005] |
| ouro-1.4b | math500 | full | strict | 256-511 | 11 | -0.0014 [-0.0037, +0.0005] |
| ouro-1.4b | math500 | full | strict | 512-99999 | 3 | -0.0011 [-0.0048, +0.0014] |
| ouro-1.4b | math500 | last | strict | 0-63 | 32 | +5.0729 [+4.7891, +5.3352] |
| ouro-1.4b | math500 | last | strict | 64-127 | 31 | +5.0174 [+4.7305, +5.2971] |
| ouro-1.4b | math500 | last | strict | 128-255 | 23 | +4.9122 [+4.4619, +5.3310] |
| ouro-1.4b | math500 | last | strict | 256-511 | 11 | +4.9304 [+4.6141, +5.3006] |
| ouro-1.4b | math500 | last | strict | 512-99999 | 3 | +4.9642 [+4.6697, +5.3423] |
| ouro-2.6b | gsm8k | first | strict | 0-63 | 32 | +0.8156 [+0.7244, +0.9159] |
| ouro-2.6b | gsm8k | first | strict | 64-127 | 24 | +0.8516 [+0.6956, +1.0032] |
| ouro-2.6b | gsm8k | first | strict | 128-255 | 6 | +1.0556 [+0.6673, +1.3874] |
| ouro-2.6b | gsm8k | full | strict | 0-63 | 32 | -0.0007 [-0.0024, +0.0009] |
| ouro-2.6b | gsm8k | full | strict | 64-127 | 24 | -0.0007 [-0.0032, +0.0020] |
| ouro-2.6b | gsm8k | full | strict | 128-255 | 6 | +0.0005 [-0.0038, +0.0067] |
| ouro-2.6b | gsm8k | last | strict | 0-63 | 32 | +0.2766 [+0.2072, +0.3529] |
| ouro-2.6b | gsm8k | last | strict | 64-127 | 24 | +0.3096 [+0.1633, +0.4624] |
| ouro-2.6b | gsm8k | last | strict | 128-255 | 6 | +0.2318 [+0.1275, +0.3733] |
| ouro-2.6b | math500 | first | strict | 0-63 | 32 | +0.9264 [+0.8466, +1.0098] |
| ouro-2.6b | math500 | first | strict | 64-127 | 31 | +0.9750 [+0.8334, +1.1228] |
| ouro-2.6b | math500 | first | strict | 128-255 | 23 | +0.9640 [+0.8223, +1.1237] |
| ouro-2.6b | math500 | first | strict | 256-511 | 11 | +1.1095 [+0.9202, +1.2775] |
| ouro-2.6b | math500 | first | strict | 512-99999 | 3 | +1.1219 [+0.8897, +1.5329] |
| ouro-2.6b | math500 | full | strict | 0-63 | 32 | +0.0012 [-0.0001, +0.0026] |
| ouro-2.6b | math500 | full | strict | 64-127 | 31 | -0.0003 [-0.0018, +0.0012] |
| ouro-2.6b | math500 | full | strict | 128-255 | 23 | -0.0004 [-0.0018, +0.0011] |
| ouro-2.6b | math500 | full | strict | 256-511 | 11 | +0.0005 [-0.0010, +0.0023] |
| ouro-2.6b | math500 | full | strict | 512-99999 | 3 | -0.0042 [-0.0087, +0.0008] |
| ouro-2.6b | math500 | last | strict | 0-63 | 32 | +0.3781 [+0.2674, +0.5060] |
| ouro-2.6b | math500 | last | strict | 64-127 | 31 | +0.3263 [+0.2217, +0.4478] |
| ouro-2.6b | math500 | last | strict | 128-255 | 23 | +0.3397 [+0.2362, +0.4564] |
| ouro-2.6b | math500 | last | strict | 256-511 | 11 | +0.6807 [+0.3289, +1.0367] |
| ouro-2.6b | math500 | last | strict | 512-99999 | 3 | +0.5422 [+0.3021, +1.0035] |
