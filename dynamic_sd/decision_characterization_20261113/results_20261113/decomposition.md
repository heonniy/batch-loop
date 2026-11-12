# Stage A — nested decision decomposition (DERIVED FROM RECEIPTS)

Source: committed `followup_20261112/results_20261112/cf/*.jsonl.gz` (SOURCE MEASURED at 5c6bbea3, B=1 eager one-round). S/d0/K0 calibrated on the other prompt fold; per-prefix choice inside a class is a same-prefix descriptive oracle (2 timing repeats pick, held-out repeat scores); the pick maximises the class aggregate Σtok/Σms (Dinkelbach). Rates = Σ committed tokens / Σ measured ms. CIs: prompt-cluster bootstrap.

## Audit

| cell | rows | prompts | prefixes kept / total | excluded (inexact) | inexact trials | selection bias kept vs excluded (mean ref D, mean k) |
|---|---:|---:|---:|---:|---:|---|
| ouro_fixed | 13824 | 32 | 512 / 512 | 0 | 0 | - |
| ouro_tau0.4 | 13824 | 32 | 512 / 512 | 0 | 0 | - |
| ouro_tau0.1 | 13824 | 32 | 512 / 512 | 0 | 0 | - |
| ouro_tau0.4_bf16 | 13824 | 32 | 491 / 512 | 21 | 183 | D 2.57 vs 2.81; k 125 vs 109 |
| huginn_fixed | 11520 | 24 | 384 / 384 | 0 | 0 | - |
| huginn_eps0.1 | 11520 | 24 | 384 / 384 | 0 | 0 | - |

## Calibrated fixed actions (per prompt fold)

| cell | fold | S | d0,K0 (best SD) |
|---|---:|---|---|
| ouro_fixed | 0 | DtV:d1:K8 | d1,K8 |
| ouro_fixed | 1 | DtV:d1:K8 | d1,K8 |
| ouro_tau0.4 | 0 | DtV:d1:K4 | d1,K4 |
| ouro_tau0.4 | 1 | DtV:d1:K8 | d1,K8 |
| ouro_tau0.1 | 0 | DtV:d1:K2 | d1,K2 |
| ouro_tau0.1 | 1 | DtV:d1:K2 | d1,K2 |
| ouro_tau0.4_bf16 | 0 | DtV:d1:K4 | d1,K4 |
| ouro_tau0.4_bf16 | 1 | DtV:d1:K8 | d1,K8 |
| huginn_fixed | 0 | DtV:d8:K8 | d8,K8 |
| huginn_fixed | 1 | DtV:d8:K8 | d8,K8 |
| huginn_eps0.1 | 0 | DtV:d4:K8 | d4,K8 |
| huginn_eps0.1 | 1 | DtV:d4:K8 | d4,K8 |

## Nested class rates and incremental ratios

| cell | S vs AR | A/S | K/A | d/A | Joint/K | Joint/d | Joint/S |
|---|---:|---:|---:|---:|---:|---:|---:|
| ouro_fixed | 1.850 | 1.194 [1.162, 1.224] | 1.017 [1.012, 1.022] | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] | 1.016 [1.011, 1.022] | 1.213 [1.178, 1.247] |
| ouro_tau0.4 | 1.340 | 1.209 [1.167, 1.249] | 1.056 [1.037, 1.078] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.056 [1.037, 1.078] | 1.277 [1.237, 1.315] |
| ouro_tau0.1 | 1.038 | 1.179 [1.151, 1.207] | 1.134 [1.114, 1.154] | 1.003 [1.000, 1.007] | 1.000 [0.998, 1.003] | 1.130 [1.111, 1.148] | 1.336 [1.294, 1.376] |
| ouro_tau0.4_bf16 | 1.443 | 1.191 [1.150, 1.237] | 1.065 [1.047, 1.087] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.065 [1.047, 1.087] | 1.268 [1.225, 1.321] |
| huginn_fixed | 1.860 | 1.097 [1.073, 1.130] | 1.005 [1.002, 1.008] | 1.138 [1.128, 1.150] | 1.149 [1.138, 1.160] | 1.015 [1.011, 1.018] | 1.267 [1.237, 1.304] |
| huginn_eps0.1 | 1.470 | 1.260 [1.227, 1.294] | 1.008 [1.005, 1.012] | 1.015 [1.010, 1.020] | 1.019 [1.015, 1.023] | 1.012 [1.008, 1.017] | 1.295 [1.259, 1.333] |

## Screen (plan's decision rule, per cell)

- **ouro_fixed**: Joint/K ≤1% or CI∋1 → adaptive d low priority; K/A material; d/A negligible (≤1% or CI∋1)
- **ouro_tau0.4**: Joint/K ≤1% or CI∋1 → adaptive d low priority; K/A material; d/A negligible (≤1% or CI∋1)
- **ouro_tau0.1**: Joint/K ≤1% or CI∋1 → adaptive d low priority; K/A material; d/A negligible (≤1% or CI∋1)
- **ouro_tau0.4_bf16**: Joint/K ≤1% or CI∋1 → adaptive d low priority; K/A material; d/A negligible (≤1% or CI∋1)
- **huginn_fixed**: Joint/K >1% with CI>1 → adaptive d adds over adaptive K; K/A negligible (≤1% or CI∋1); d/A material
- **huginn_eps0.1**: Joint/K >1% with CI>1 → adaptive d adds over adaptive K; K/A negligible (≤1% or CI∋1); d/A material

## Per-prompt heterogeneity (ratio per evaluation prompt)

| cell | Joint/K median [min, max], frac>1.01 | K/A median [min, max], frac>1.01 | d/A median [min, max], frac>1.01 |
|---|---|---|---|
| ouro_fixed | 1.000 [0.992, 1.000], 0.00 | 1.015 [0.983, 1.118], 0.66 | 1.000 [1.000, 1.000], 0.00 |
| ouro_tau0.4 | 1.000 [1.000, 1.000], 0.00 | 1.062 [0.997, 1.158], 0.75 | 1.000 [1.000, 1.000], 0.00 |
| ouro_tau0.1 | 1.000 [0.990, 1.026], 0.06 | 1.123 [1.003, 1.269], 0.97 | 1.001 [0.988, 1.039], 0.16 |
| ouro_tau0.4_bf16 | 1.000 [1.000, 1.000], 0.00 | 1.091 [0.999, 1.167], 0.88 | 1.000 [1.000, 1.000], 0.00 |
| huginn_fixed | 1.149 [1.102, 1.211], 1.00 | 1.004 [0.995, 1.038], 0.17 | 1.139 [1.083, 1.196], 1.00 |
| huginn_eps0.1 | 1.016 [1.000, 1.044], 0.88 | 1.007 [0.999, 1.060], 0.38 | 1.015 [1.000, 1.033], 0.67 |

## Class behaviour (evaluation prompts)

| cell | class | frac AR | mean K (SD rounds) | mean accepted | frac rejected rounds | core rows/round | top picks |
|---|---|---:|---:|---:|---:|---:|---|
| ouro_fixed | S | 0.00 | 8.00 | 5.86 | 0.42 | 36.0 | DtV:d1:K8:1536 |
| ouro_fixed | A | 0.36 | 8.00 | 7.87 | 0.09 | 24.4 | DtV:d1:K8:979, AR:557 |
| ouro_fixed | K | 0.16 | 6.64 | 6.61 | 0.03 | 26.3 | DtV:d1:K8:929, AR:244, DtV:d1:K4:210, DtV:d1:K2:150 |
| ouro_fixed | d | 0.36 | 8.00 | 7.87 | 0.09 | 24.4 | DtV:d1:K8:979, AR:557 |
| ouro_fixed | Joint | 0.16 | 6.64 | 6.61 | 0.03 | 26.3 | DtV:d1:K8:929, AR:244, DtV:d1:K4:210, DtV:d1:K2:149 |
| ouro_tau0.4 | S | 0.00 | 6.00 | 4.56 | 0.34 | 18.3 | DtV:d1:K4:768, DtV:d1:K8:768 |
| ouro_tau0.4 | A | 0.35 | 5.81 | 5.80 | 0.01 | 11.6 | AR:544, DtV:d1:K4:544, DtV:d1:K8:448 |
| ouro_tau0.4 | K | 0.25 | 6.78 | 6.78 | 0.00 | 14.6 | DtV:d1:K8:870, AR:390, DtV:d1:K4:153, DtV:d1:K2:75 |
| ouro_tau0.4 | d | 0.35 | 5.81 | 5.80 | 0.01 | 11.6 | AR:544, DtV:d1:K4:544, DtV:d1:K8:448 |
| ouro_tau0.4 | Joint | 0.25 | 6.78 | 6.78 | 0.00 | 14.6 | DtV:d1:K8:870, AR:390, DtV:d1:K4:153, DtV:d1:K2:75 |
| ouro_tau0.1 | S | 0.00 | 2.00 | 1.78 | 0.14 | 5.3 | DtV:d1:K2:1536 |
| ouro_tau0.1 | A | 0.49 | 2.00 | 2.00 | 0.00 | 3.0 | DtV:d1:K2:791, AR:745 |
| ouro_tau0.1 | K | 0.37 | 4.89 | 4.89 | 0.00 | 5.4 | AR:567, DtV:d1:K8:437, DtV:d1:K2:214, DtV:d1:K4:167 |
| ouro_tau0.1 | d | 0.47 | 2.00 | 2.00 | 0.00 | 3.0 | AR:727, DtV:d1:K2:471, DtV:d2:K2:338 |
| ouro_tau0.1 | Joint | 0.36 | 4.87 | 4.87 | 0.00 | 5.4 | AR:550, DtV:d1:K8:289, DtV:d2:K8:149, DtV:d1:K2:130 |
| ouro_tau0.4_bf16 | S | 0.00 | 5.98 | 4.52 | 0.34 | 18.1 | DtV:d1:K4:744, DtV:d1:K8:729 |
| ouro_tau0.4_bf16 | A | 0.34 | 5.74 | 5.71 | 0.03 | 11.7 | DtV:d1:K4:548, AR:502, DtV:d1:K8:423 |
| ouro_tau0.4_bf16 | K | 0.14 | 6.07 | 6.07 | 0.00 | 15.0 | DtV:d1:K8:833, AR:206, DtV:d1:K2:158, DtV:d1:K4:144 |
| ouro_tau0.4_bf16 | d | 0.34 | 5.74 | 5.71 | 0.03 | 11.7 | DtV:d1:K4:548, AR:502, DtV:d1:K8:423 |
| ouro_tau0.4_bf16 | Joint | 0.14 | 6.07 | 6.07 | 0.00 | 15.0 | DtV:d1:K8:833, AR:206, DtV:d1:K2:158, DtV:d1:K4:144 |
| huginn_fixed | S | 0.00 | 8.00 | 6.92 | 0.23 | 288.0 | DtV:d8:K8:1152 |
| huginn_fixed | A | 0.21 | 8.00 | 7.97 | 0.03 | 234.2 | DtV:d8:K8:910, AR:242 |
| huginn_fixed | K | 0.12 | 7.49 | 7.49 | 0.00 | 242.0 | DtV:d8:K8:881, AR:143, DtV:d8:K4:128 |
| huginn_fixed | d | 0.18 | 8.00 | 7.86 | 0.09 | 240.9 | DtV:d4:K8:593, DtV:d8:K8:298, AR:212, DtV:d2:K8:49 |
| huginn_fixed | Joint | 0.13 | 7.11 | 7.05 | 0.06 | 230.3 | DtV:d4:K8:561, DtV:d8:K8:185, AR:148, DtV:d4:K4:131 |
| huginn_eps0.1 | S | 0.00 | 8.00 | 5.74 | 0.40 | 146.7 | DtV:d4:K8:1152 |
| huginn_eps0.1 | A | 0.39 | 8.00 | 7.97 | 0.03 | 94.6 | DtV:d4:K8:703, AR:449 |
| huginn_eps0.1 | K | 0.29 | 7.30 | 7.30 | 0.00 | 100.0 | DtV:d4:K8:679, AR:329, DtV:d4:K4:144 |
| huginn_eps0.1 | d | 0.38 | 8.00 | 7.95 | 0.05 | 95.6 | DtV:d4:K8:633, AR:441, DtV:d2:K8:78 |
| huginn_eps0.1 | Joint | 0.26 | 7.19 | 7.16 | 0.03 | 102.1 | DtV:d4:K8:607, AR:296, DtV:d4:K4:110, DtV:d2:K8:78 |

## Cross-regime (live − fixed), paired over shared prompt ids — descriptive, distinct targets

| pair | metric | live | fixed | diff [95% CI] |
|---|---|---:|---:|---:|
| ouro_tau0.4-ouro_fixed (n=32) | Joint/S | 1.277 | 1.213 | +0.064 [+0.025, +0.106] |
| ouro_tau0.4-ouro_fixed (n=32) | Joint/K | 1.000 | 0.999 | +0.001 [+0.000, +0.001] |
| ouro_tau0.4-ouro_fixed (n=32) | K/A | 1.056 | 1.017 | +0.039 [+0.020, +0.062] |
| ouro_tau0.4-ouro_fixed (n=32) | d/A | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| ouro_tau0.1-ouro_fixed (n=32) | Joint/S | 1.336 | 1.213 | +0.123 [+0.071, +0.172] |
| ouro_tau0.1-ouro_fixed (n=32) | Joint/K | 1.000 | 0.999 | +0.000 [-0.002, +0.003] |
| ouro_tau0.1-ouro_fixed (n=32) | K/A | 1.134 | 1.017 | +0.117 [+0.097, +0.138] |
| ouro_tau0.1-ouro_fixed (n=32) | d/A | 1.003 | 1.000 | +0.003 [-0.000, +0.007] |
| huginn_eps0.1-huginn_fixed (n=24) | Joint/S | 1.295 | 1.267 | +0.028 [-0.024, +0.074] |
| huginn_eps0.1-huginn_fixed (n=24) | Joint/K | 1.019 | 1.149 | -0.130 [-0.141, -0.119] |
| huginn_eps0.1-huginn_fixed (n=24) | K/A | 1.008 | 1.005 | +0.003 [-0.001, +0.008] |
| huginn_eps0.1-huginn_fixed (n=24) | d/A | 1.015 | 1.138 | -0.123 [-0.134, -0.112] |

## Sensitivity: per-prefix ratio-argmax selection (rule used by oracle_frontier.py at 5c6bbea3)

Not aggregate-optimal: a superset class can score below its subset. Shown only for comparability.

| cell | A/S | K/A | d/A | Joint/K | Joint/d | Joint/S |
|---|---:|---:|---:|---:|---:|---:|
| ouro_fixed | 1.129 [1.101, 1.159] | 1.070 [1.056, 1.086] | 0.990 [0.985, 0.995] | 0.973 [0.969, 0.978] | 1.052 [1.041, 1.063] | 1.176 [1.146, 1.207] |
| ouro_tau0.4 | 1.182 [1.145, 1.218] | 1.070 [1.051, 1.090] | 0.974 [0.965, 0.982] | 0.976 [0.969, 0.982] | 1.072 [1.053, 1.091] | 1.234 [1.199, 1.268] |
| ouro_tau0.1 | 1.144 [1.117, 1.173] | 1.125 [1.103, 1.146] | 0.993 [0.984, 1.002] | 0.977 [0.967, 0.986] | 1.107 [1.088, 1.124] | 1.257 [1.225, 1.292] |
| ouro_tau0.4_bf16 | 1.170 [1.133, 1.212] | 1.077 [1.061, 1.097] | 0.963 [0.953, 0.974] | 0.979 [0.970, 0.987] | 1.095 [1.077, 1.114] | 1.233 [1.195, 1.281] |
| huginn_fixed | 1.078 [1.053, 1.112] | 1.019 [1.011, 1.029] | 1.130 [1.118, 1.142] | 1.145 [1.135, 1.157] | 1.034 [1.025, 1.043] | 1.259 [1.229, 1.296] |
| huginn_eps0.1 | 1.223 [1.193, 1.253] | 1.032 [1.019, 1.046] | 0.985 [0.976, 0.995] | 0.992 [0.987, 0.996] | 1.039 [1.027, 1.050] | 1.251 [1.218, 1.287] |

*Caveat:* one-round reward with different K advances different token counts; these ratios are not multi-round speedups. No controller cost. Eager B=1 (FP32 uses SDPA fallback attention).
