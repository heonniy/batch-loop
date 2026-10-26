# E0 — FULL vs LAST-CONVERT (Huginn R32, Ouro-1.4B R4, Ouro-2.6B R4)

Ouro: this run (`results/20261026T080106Z_huginn_ouro_full_last_convert`). Huginn R32: reused `results/20261023T060125Z_huginn_r32_h200x8` (same frozen items, prompts, evaluator, caps 512/2048, FULL = full/decode, LAST-CONVERT = last/convert; same engine semantics). Greedy, fixed depth, BF16, early exit disabled. 'LAST-CONVERT' is **our** policy: loop-major FULL prefill, then every past token keeps only its terminal-step K/V; the official Ouro sharing semantics are not established, so degradation is attributed to this policy only.

| model | dataset | n | acc FULL | acc LAST-C | Δ (LAST−FULL) [95% CI] | FULL-only / LAST-only | McNemar p | gold NLL FULL / LAST (tok-weighted) | ΔNLL per problem [CI] | cap hit FULL / LAST | parse fail FULL / LAST | mean gen len FULL / LAST | identical generations | first divergence median [IQR] |
|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|---|---:|---|
| huginn-0125 | gsm8k | 200 | 0.415 | 0.445 | +0.030 [+0.005, +0.060] | 1 / 7 | 7.0e-02 | 0.8996 / 0.9010 | +0.0016 [-0.0001, +0.0035] | 3 / 1 | 3 / 1 | 105 / 103 | 127 | 35 [17, 70] |
| huginn-0125 | math500 | 200 | 0.155 | 0.165 | +0.010 [-0.020, +0.040] | 3 / 5 | 7.3e-01 | 0.6317 / 0.6315 | -0.0003 [-0.0011, +0.0004] | 47 / 52 | 48 / 53 | 611 / 650 | 102 | 47 [25, 102] |
| ouro-1.4b | gsm8k | 200 | 0.835 | 0.000 | -0.835 [-0.885, -0.785] | 167 / 0 | 1.1e-50 | 0.7316 / 3.4984 | +2.7185 [+2.6355, +2.7996] | 1 / 179 | 1 / 195 | 111 / 473 | 0 | 3 [2, 4] |
| ouro-1.4b | math500 | 200 | 0.710 | 0.015 | -0.695 [-0.755, -0.630] | 139 / 0 | 2.9e-42 | 0.5436 / 2.6168 | +2.0739 [+1.9747, +2.1748] | 4 / 137 | 4 / 176 | 539 / 1436 | 0 | 5 [3, 8] |
| ouro-2.6b | gsm8k | 200 | 0.880 | 0.770 | -0.110 [-0.170, -0.050] | 29 / 7 | 3.1e-04 | 0.6958 / 0.8645 | +0.1486 [+0.1239, +0.1756] | 0 / 8 | 0 / 7 | 109 / 119 | 8 | 13 [6, 25] |
| ouro-2.6b | math500 | 200 | 0.560 | 0.035 | -0.525 [-0.595, -0.460] | 105 / 0 | 4.9e-32 | 0.4649 / 2.1549 | +1.5121 [+1.3896, +1.6389] | 18 / 173 | 18 / 171 | 424 / 1788 | 1 | 4 [2, 7] |

Scorer variants (same generations):

- huginn-0125 gsm8k: lm_eval_strict_match FULL 0.340 / LAST 0.360, lm_eval_flexible_extract FULL 0.410 / LAST 0.440
- huginn-0125 math500: math_verify FULL 0.155 / LAST 0.165, minerva_is_equiv FULL 0.150 / LAST 0.160, normalized_string_equal FULL 0.145 / LAST 0.155
- ouro-1.4b gsm8k: lm_eval_strict_match FULL 0.600 / LAST 0.000, lm_eval_flexible_extract FULL 0.805 / LAST 0.000
- ouro-1.4b math500: math_verify FULL 0.710 / LAST 0.015, minerva_is_equiv FULL 0.675 / LAST 0.015, normalized_string_equal FULL 0.600 / LAST 0.010
- ouro-2.6b gsm8k: lm_eval_strict_match FULL 0.660 / LAST 0.615, lm_eval_flexible_extract FULL 0.865 / LAST 0.755
- ouro-2.6b math500: math_verify FULL 0.560 / LAST 0.035, minerva_is_equiv FULL 0.520 / LAST 0.035, normalized_string_equal FULL 0.090 / LAST 0.035

## KV footprint (analytic, BF16, per token)

| model | recurrent KV FULL | recurrent KV LAST-CONVERT (retained during decode) | prelude/coda KV | retained total LAST/FULL | prefill transient (both policies) |
|---|---:|---:|---:|---:|---|
| huginn-0125 | 2640 KiB | 82.5 KiB | 82.5 KiB | 0.061 | prompt needs all 32 planes during the parallel FULL prefill |
| ouro-1.4b | 768 KiB | 192.0 KiB | 0.0 KiB | 0.250 | prompt needs all 4 planes during the parallel FULL prefill |
| ouro-2.6b | 1536 KiB | 384.0 KiB | 0.0 KiB | 0.250 | prompt needs all 4 planes during the parallel FULL prefill |

Retained = after Convert, during decode. The prefill peak is NOT reduced by Convert (the prompt is prefilled loop-major with every per-step plane). The reference engine stores a FULL-sized broadcast layout for every policy, so its measured peaks are not policy footprints.

