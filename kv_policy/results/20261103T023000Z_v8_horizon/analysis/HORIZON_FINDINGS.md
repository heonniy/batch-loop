# Decode-horizon growth of KV damage: looped (Huginn R=32, Ouro R=4) vs standard (Qwen2.5-3B)

Runs: 20261103T023000Z_v8_horizon, 20261103T025000Z_v8_horizon_poolA; standard: 20261103T022500Z_std_qwen_v6horizon, 20261103T025000Z_std_qwen_v6horizon_poolA

Outcome: KL(p_full || p_policy) at generated step t of the model's OWN frozen greedy 128-token continuation (v6 W1 continuation / W2 summarise-explain prompts), teacher-forced; binned over t. 'late/early' = mean KL at t 33–128 over mean KL at t 1–8 (per prompt, median; CI on log10). Growth > 1 means damage accumulates through the generated history; < 1 means the model recovers.

## Looped models: plane compression (prompt-only vs prompt+generated), token eviction (30% kept), mixed

| model | dataset | arm | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | share of prompts with late > early |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | v6_W1 | mixed:last@prompt+generated+fut_oracle_rR | 14 | 0.138 | 0.0735 | 0.13 | 0.736 [10^-0.23,10^0.07] | 0.33 (12 prompts) |
| huginn-0125 | v6_W1 | mixed:last@prompt+generated+la2_rR | 14 | 0.0437 | 0.23 | 0.349 | 1.68 [10^0.25,10^1.33] | 0.67 (12 prompts) |
| huginn-0125 | v6_W1 | plane:even@prompt | 14 | 0.000312 | 0.000327 | 0.000316 | 0.955 [10^-0.26,10^0.14] | 0.5 (12 prompts) |
| huginn-0125 | v6_W1 | plane:even@prompt+generated | 14 | 0.000333 | 0.000336 | 0.000338 | 0.922 [10^-0.24,10^0.12] | 0.42 (12 prompts) |
| huginn-0125 | v6_W1 | plane:exit@prompt | 14 | 0.000887 | 0.00062 | 0.000447 | 0.877 [10^-0.58,10^0.00] | 0.42 (12 prompts) |
| huginn-0125 | v6_W1 | plane:exit@prompt+generated | 14 | 0.00124 | 0.00146 | 0.00112 | 1.11 [10^-0.33,10^0.17] | 0.5 (12 prompts) |
| huginn-0125 | v6_W1 | plane:first@prompt | 14 | 0.257 | 0.322 | 0.29 | 0.408 [10^-0.48,10^0.20] | 0.42 (12 prompts) |
| huginn-0125 | v6_W1 | plane:first@prompt+generated | 14 | 0.523 | 1.01 | 1.13 | 1.87 [10^0.11,10^0.48] | 0.75 (12 prompts) |
| huginn-0125 | v6_W1 | plane:last@prompt | 14 | 0.00027 | 0.00042 | 0.000324 | 1.29 [10^-0.07,10^0.18] | 0.58 (12 prompts) |
| huginn-0125 | v6_W1 | plane:last@prompt+generated | 14 | 0.000423 | 0.000382 | 0.000361 | 0.877 [10^-0.43,10^0.11] | 0.5 (12 prompts) |
| huginn-0125 | v6_W1 | token:full@prompt+def4_rR | 14 | 0.0906 | 0.534 | 0.519 | 2.86 [10^0.14,10^1.10] | 0.67 (12 prompts) |
| huginn-0125 | v6_W1 | token:full@prompt+fut_oracle_rR | 14 | 0.272 | 0.238 | 0.259 | 0.791 [10^-0.29,10^-0.02] | 0.25 (12 prompts) |
| huginn-0125 | v6_W1 | token:full@prompt+la2_rR | 14 | 0.13 | 0.479 | 0.496 | 1.87 [10^0.04,10^0.92] | 0.58 (12 prompts) |
| huginn-0125 | v6_W1 | token:full@prompt+qa_rR | 14 | 0.175 | 0.524 | 0.508 | 1.9 [10^-0.27,10^0.56] | 0.67 (12 prompts) |
| huginn-0125 | v6_W1 | token:full@prompt+random_11 | 14 | 0.623 | 0.63 | 0.479 | 0.476 [10^-0.72,10^-0.03] | 0.25 (12 prompts) |
| huginn-0125 | v6_W2 | mixed:last@prompt+generated+fut_oracle_rR | 14 | 0.145 | 0.0362 | 0.117 | 1.97 [10^0.07,10^0.74] | 0.57 (14 prompts) |
| huginn-0125 | v6_W2 | mixed:last@prompt+generated+la2_rR | 14 | 0.00587 | 0.287 | 0.671 | 372 [10^2.01,10^2.82] | 1 (14 prompts) |
| huginn-0125 | v6_W2 | plane:even@prompt | 14 | 0.00023 | 0.00023 | 0.000257 | 1.42 [10^0.02,10^0.62] | 0.64 (14 prompts) |
| huginn-0125 | v6_W2 | plane:even@prompt+generated | 14 | 0.000207 | 0.000322 | 0.00034 | 2.51 [10^0.16,10^0.76] | 0.79 (14 prompts) |
| huginn-0125 | v6_W2 | plane:exit@prompt | 14 | 0.000724 | 0.000567 | 0.000725 | 1 [10^-0.12,10^0.26] | 0.5 (14 prompts) |
| huginn-0125 | v6_W2 | plane:exit@prompt+generated | 14 | 0.00105 | 0.00109 | 0.00136 | 1.18 [10^0.01,10^0.46] | 0.57 (14 prompts) |
| huginn-0125 | v6_W2 | plane:first@prompt | 14 | 0.141 | 0.298 | 0.395 | 3.5 [10^0.34,10^0.75] | 1 (14 prompts) |
| huginn-0125 | v6_W2 | plane:first@prompt+generated | 14 | 0.413 | 0.96 | 1.5 | 3.84 [10^0.48,10^0.70] | 1 (14 prompts) |
| huginn-0125 | v6_W2 | plane:last@prompt | 14 | 0.000277 | 0.000293 | 0.000327 | 1.52 [10^-0.04,10^0.42] | 0.57 (14 prompts) |
| huginn-0125 | v6_W2 | plane:last@prompt+generated | 14 | 0.000278 | 0.000351 | 0.000458 | 2.33 [10^0.15,10^0.77] | 0.71 (14 prompts) |
| huginn-0125 | v6_W2 | token:full@prompt+def4_rR | 14 | 0.093 | 0.644 | 1.05 | 77.6 [10^1.37,10^2.08] | 1 (14 prompts) |
| huginn-0125 | v6_W2 | token:full@prompt+fut_oracle_rR | 14 | 0.603 | 0.414 | 0.403 | 0.94 [10^-0.25,10^0.57] | 0.43 (14 prompts) |
| huginn-0125 | v6_W2 | token:full@prompt+la2_rR | 14 | 0.0595 | 0.654 | 1.05 | 40.4 [10^1.24,10^1.99] | 1 (14 prompts) |
| huginn-0125 | v6_W2 | token:full@prompt+qa_rR | 14 | 0.419 | 0.817 | 0.92 | 5.51 [10^0.50,10^1.37] | 0.86 (14 prompts) |
| huginn-0125 | v6_W2 | token:full@prompt+random_11 | 14 | 0.926 | 0.742 | 0.837 | 0.997 [10^-0.17,10^0.45] | 0.5 (14 prompts) |
| ouro-2.6b | v6_W1 | mixed:last@prompt+generated+fut_oracle_rR | 13 | 0.113 | 0.214 | 0.143 | 2.4 [10^0.04,10^0.47] | 0.69 (13 prompts) |
| ouro-2.6b | v6_W1 | mixed:last@prompt+generated+la2_rR | 13 | 0.272 | 0.279 | 0.39 | 2.15 [10^-0.16,10^0.84] | 0.69 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:even@prompt | 13 | 0.0101 | 0.0129 | 0.0196 | 1.75 [10^-0.14,10^0.45] | 0.77 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:even@prompt+generated | 13 | 0.0137 | 0.0156 | 0.0198 | 1.61 [10^-0.17,10^0.38] | 0.77 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:exit@prompt | 13 | 0.000693 | 0.000433 | 0.000348 | 0.558 [10^-0.72,10^-0.13] | 0.15 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:exit@prompt+generated | 13 | 0.000747 | 0.00112 | 0.00136 | 1.54 [10^0.01,10^0.41] | 0.77 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:first@prompt | 13 | 0.418 | 0.502 | 0.413 | 0.815 [10^-0.38,10^0.23] | 0.38 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:first@prompt+generated | 13 | 0.413 | 0.453 | 0.482 | 1.01 [10^-0.08,10^0.35] | 0.54 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:last@prompt | 13 | 0.00622 | 0.00789 | 0.00881 | 1.04 [10^-0.08,10^0.36] | 0.54 (13 prompts) |
| ouro-2.6b | v6_W1 | plane:last@prompt+generated | 13 | 0.0315 | 0.053 | 0.0982 | 3.23 [10^0.30,10^0.78] | 0.85 (13 prompts) |
| ouro-2.6b | v6_W1 | token:full@prompt+def4_rR | 13 | 0.132 | 0.481 | 0.523 | 3.49 [10^-0.00,10^1.14] | 0.69 (13 prompts) |
| ouro-2.6b | v6_W1 | token:full@prompt+fut_oracle_rR | 13 | 0.321 | 0.338 | 0.291 | 0.803 [10^-0.47,10^0.23] | 0.31 (13 prompts) |
| ouro-2.6b | v6_W1 | token:full@prompt+la2_rR | 13 | 0.314 | 0.455 | 0.518 | 0.814 [10^-0.36,10^0.78] | 0.46 (13 prompts) |
| ouro-2.6b | v6_W1 | token:full@prompt+qa_rR | 13 | 0.493 | 0.559 | 0.502 | 0.757 [10^-0.49,10^0.63] | 0.46 (13 prompts) |
| ouro-2.6b | v6_W1 | token:full@prompt+random_11 | 13 | 0.543 | 0.591 | 0.494 | 0.719 [10^-0.43,10^0.02] | 0.31 (13 prompts) |
| ouro-2.6b | v6_W2 | mixed:last@prompt+generated+fut_oracle_rR | 4 | 0.0792 | 0.207 | 0.179 | 1.84 [10^0.14,10^0.47] | 1 (4 prompts) |
| ouro-2.6b | v6_W2 | mixed:last@prompt+generated+la2_rR | 4 | 0.0449 | 0.437 | 0.685 | 16 [10^0.90,10^1.58] | 1 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:even@prompt | 4 | 0.00896 | 0.0168 | 0.0157 | 2.76 [10^-0.44,10^0.72] | 0.5 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:even@prompt+generated | 4 | 0.0126 | 0.0183 | 0.0152 | 1.91 [10^-0.48,10^0.55] | 0.5 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:exit@prompt | 4 | 0.000585 | 0.000306 | 0.000221 | 0.931 [10^-0.81,10^0.24] | 0.5 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:exit@prompt+generated | 4 | 0.00134 | 0.00115 | 0.000871 | 1.59 [10^-0.76,10^0.62] | 0.75 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:first@prompt | 4 | 0.283 | 0.222 | 0.303 | 0.898 [10^-0.67,10^0.36] | 0.5 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:first@prompt+generated | 4 | 0.546 | 0.236 | 0.378 | 0.713 [10^-0.66,10^0.25] | 0.5 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:last@prompt | 4 | 0.0196 | 0.0202 | 0.013 | 1.4 [10^-0.29,10^1.01] | 0.5 (4 prompts) |
| ouro-2.6b | v6_W2 | plane:last@prompt+generated | 4 | 0.0533 | 0.0456 | 0.089 | 1.91 [10^-0.05,10^0.95] | 0.75 (4 prompts) |
| ouro-2.6b | v6_W2 | token:full@prompt+def4_rR | 4 | 0.183 | 0.419 | 0.923 | 73.6 [10^0.56,10^2.54] | 1 (4 prompts) |
| ouro-2.6b | v6_W2 | token:full@prompt+fut_oracle_rR | 4 | 0.197 | 0.351 | 0.509 | 1.29 [10^0.06,10^0.79] | 1 (4 prompts) |
| ouro-2.6b | v6_W2 | token:full@prompt+la2_rR | 4 | 0.0298 | 0.528 | 0.958 | 137 [10^1.28,10^2.65] | 1 (4 prompts) |
| ouro-2.6b | v6_W2 | token:full@prompt+qa_rR | 4 | 0.298 | 0.892 | 0.792 | 3.84 [10^0.23,10^1.29] | 1 (4 prompts) |
| ouro-2.6b | v6_W2 | token:full@prompt+random_11 | 4 | 0.957 | 0.865 | 0.781 | 2.29 [10^-0.33,10^0.75] | 0.75 (4 prompts) |

## Looped models: single Context-token eviction (frozen v6 census, mean over sources)

| model | dataset | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | share of prompts with late > early |
|---|---|---|---|---|---|---|---|
| huginn-0125 | v6_W1 | 518 | 0.00957 | 0.00783 | 0.00375 | 0.619 [10^-0.68,10^0.01] | 0.5 (2 prompts) |
| huginn-0125 | v6_W2 | 447 | 0.0126 | 0.0135 | 0.0129 | 1.45 [10^-0.08,10^0.34] | 0.75 (4 prompts) |
| ouro-2.6b | v6_W1 | 508 | 0.00667 | 0.00933 | 0.00956 | 0.926 [10^-0.64,10^1.23] | 0.25 (4 prompts) |
| ouro-2.6b | v6_W2 | 437 | 0.0116 | 0.0153 | 0.0137 | 4.24 [10^-0.16,10^1.14] | 0.75 (4 prompts) |

## Looped models: single-token eviction, top-5% sources by early damage

| model | dataset | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | share of prompts with late > early |
|---|---|---|---|---|---|---|---|
| huginn-0125 | v6_W1 | 27 | 0.172 | 0.0246 | 0.00821 | 0.0627 [10^-1.44,10^-1.05] | 0 (2 prompts) |
| huginn-0125 | v6_W2 | 25 | 0.216 | 0.0348 | 0.00719 | 0.017 [10^-2.31,10^-0.67] | 0 (4 prompts) |
| ouro-2.6b | v6_W1 | 27 | 0.114 | 0.00878 | 0.0059 | 0.132 [10^-2.09,10^-0.48] | 0 (4 prompts) |
| ouro-2.6b | v6_W2 | 23 | 0.211 | 0.026 | 0.00348 | 0.0542 [10^-1.93,10^-0.38] | 0 (4 prompts) |

## Standard model: single Context-token eviction (mean over sources)

| model | dataset | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | share of prompts with late > early |
|---|---|---|---|---|---|---|---|
| qwen2.5-3b | v6_W1 | 1598 | 0.0064 | 0.00494 | 0.00439 | 0.797 [10^-0.41,10^0.02] | 0.36 (14 prompts) |
| qwen2.5-3b | v6_W2 | 1520 | 0.0135 | 0.0118 | 0.0179 | 2.48 [10^0.05,10^0.72] | 0.69 (13 prompts) |

## Standard model: single-token eviction, top-5% sources by early damage

| model | dataset | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | share of prompts with late > early |
|---|---|---|---|---|---|---|---|
| qwen2.5-3b | v6_W1 | 88 | 0.0931 | 0.0112 | 0.00368 | 0.0946 [10^-1.52,10^-0.82] | 0.071 (14 prompts) |
| qwen2.5-3b | v6_W2 | 81 | 0.235 | 0.0244 | 0.00398 | 0.0151 [10^-2.10,10^-1.18] | 0.077 (13 prompts) |

## Standard model: token policies at 30% kept (kl_late = second half of the continuation)

| model | dataset | policy | n | t 1–8 | t 9–32 | t 33–128 | late/early ratio [CI over prompts] | share of prompts with late > early |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-3b | v6_W1 | def4_early | 14 | 0.313 | — | 0.296 | 0.929 [10^-0.50,10^0.41] | 0.5 (14 prompts) |
| qwen2.5-3b | v6_W1 | def4_late | 14 | 0.223 | — | 0.298 | 1.21 [10^-0.39,10^0.49] | 0.57 (14 prompts) |
| qwen2.5-3b | v6_W1 | def4_mean | 14 | 0.272 | — | 0.297 | 1.32 [10^-0.45,10^0.33] | 0.57 (14 prompts) |
| qwen2.5-3b | v6_W1 | fut_oracle_early | 14 | 0.413 | — | 0.206 | 0.499 [10^-0.58,10^-0.23] | 0.071 (14 prompts) |
| qwen2.5-3b | v6_W1 | fut_oracle_late | 14 | 0.295 | — | 0.149 | 0.366 [10^-0.68,10^-0.19] | 0.071 (14 prompts) |
| qwen2.5-3b | v6_W1 | fut_oracle_mean | 14 | 0.313 | — | 0.18 | 0.474 [10^-0.50,10^-0.19] | 0.071 (14 prompts) |
| qwen2.5-3b | v6_W1 | la2_early | 14 | 0.548 | — | 0.27 | 0.496 [10^-0.82,10^-0.03] | 0.36 (14 prompts) |
| qwen2.5-3b | v6_W1 | la2_late | 14 | 0.436 | — | 0.291 | 0.666 [10^-0.68,10^-0.03] | 0.29 (14 prompts) |
| qwen2.5-3b | v6_W1 | la2_mean | 14 | 0.415 | — | 0.287 | 0.915 [10^-0.62,10^0.05] | 0.36 (14 prompts) |
| qwen2.5-3b | v6_W1 | qa_early | 14 | 0.691 | — | 0.249 | 0.288 [10^-0.84,10^-0.21] | 0.14 (14 prompts) |
| qwen2.5-3b | v6_W1 | qa_last_layer | 14 | 0.303 | — | 0.253 | 0.421 [10^-0.62,10^-0.18] | 0.071 (14 prompts) |
| qwen2.5-3b | v6_W1 | qa_late | 14 | 0.337 | — | 0.32 | 0.533 [10^-0.56,10^-0.08] | 0.21 (14 prompts) |
| qwen2.5-3b | v6_W1 | qa_max_layer | 14 | 0.578 | — | 0.251 | 0.433 [10^-0.75,10^-0.16] | 0.21 (14 prompts) |
| qwen2.5-3b | v6_W1 | qa_mean | 14 | 0.546 | — | 0.264 | 0.501 [10^-0.78,10^-0.18] | 0.14 (14 prompts) |
| qwen2.5-3b | v6_W1 | random_11 | 14 | 0.706 | — | 0.268 | 0.343 [10^-0.84,10^-0.27] | 0.14 (14 prompts) |
| qwen2.5-3b | v6_W1 | random_29 | 14 | 0.681 | — | 0.264 | 0.259 [10^-0.88,10^-0.17] | 0.14 (14 prompts) |
| qwen2.5-3b | v6_W2 | def4_early | 14 | 0.372 | — | 1.09 | 8.26 [10^0.66,10^1.36] | 0.93 (14 prompts) |
| qwen2.5-3b | v6_W2 | def4_late | 14 | 0.0708 | — | 1.31 | 143 [10^1.72,10^2.60] | 1 (14 prompts) |
| qwen2.5-3b | v6_W2 | def4_mean | 14 | 0.172 | — | 1.27 | 190 [10^1.50,10^2.63] | 0.93 (14 prompts) |
| qwen2.5-3b | v6_W2 | fut_oracle_early | 14 | 0.802 | — | 0.542 | 0.742 [10^-0.26,10^0.53] | 0.43 (14 prompts) |
| qwen2.5-3b | v6_W2 | fut_oracle_late | 14 | 0.87 | — | 0.492 | 0.619 [10^-0.29,10^0.31] | 0.43 (14 prompts) |
| qwen2.5-3b | v6_W2 | fut_oracle_mean | 14 | 1.2 | — | 0.504 | 0.726 [10^-0.45,10^0.25] | 0.43 (14 prompts) |
| qwen2.5-3b | v6_W2 | la2_early | 14 | 0.857 | — | 1.09 | 5.62 [10^0.25,10^1.15] | 0.79 (14 prompts) |
| qwen2.5-3b | v6_W2 | la2_late | 14 | 0.171 | — | 1.28 | 67.3 [10^1.40,10^2.40] | 0.93 (14 prompts) |
| qwen2.5-3b | v6_W2 | la2_mean | 14 | 0.173 | — | 1.26 | 94.5 [10^1.42,10^2.34] | 0.93 (14 prompts) |
| qwen2.5-3b | v6_W2 | qa_early | 14 | 1.43 | — | 0.84 | 0.882 [10^-0.37,10^0.28] | 0.5 (14 prompts) |
| qwen2.5-3b | v6_W2 | qa_last_layer | 14 | 0.523 | — | 0.991 | 2.8 [10^0.23,10^1.15] | 0.71 (14 prompts) |
| qwen2.5-3b | v6_W2 | qa_late | 14 | 0.491 | — | 1.18 | 10 [10^0.46,10^1.43] | 0.79 (14 prompts) |
| qwen2.5-3b | v6_W2 | qa_max_layer | 14 | 0.968 | — | 0.902 | 1.87 [10^-0.14,10^0.50] | 0.64 (14 prompts) |
| qwen2.5-3b | v6_W2 | qa_mean | 14 | 0.673 | — | 0.999 | 3.83 [10^0.18,10^0.99] | 0.79 (14 prompts) |
| qwen2.5-3b | v6_W2 | random_11 | 14 | 0.856 | — | 1.06 | 1.97 [10^0.04,10^0.95] | 0.79 (14 prompts) |
| qwen2.5-3b | v6_W2 | random_29 | 14 | 0.971 | — | 1.14 | 4.13 [10^0.13,10^1.10] | 0.64 (14 prompts) |
