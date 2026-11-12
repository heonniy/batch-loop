# Stage B — reference-D-conditioned crossover (POST-HOC, DERIVED FROM RECEIPTS)

D = native AR reference exit depth of the token processed at the checkpoint (post-verification; not a pre-decision feature). Ratio = aggregate U(hi)/U(lo), U = Σ committed tokens / Σ measured ms, paired on the identical committed-prefix snapshot; 95% CI = prompt-cluster bootstrap. `sparse` = n<30 prefixes or <5 prompts. Δlat = mean per-round latency hi − lo (ms). acc = mean accepted drafts; rows = core rows/round (derived).

**Missing fields (not reconstructed, so no stage-time attribution):** stage wall: Core / Prelude / Coda (ms); CUDA-event timings; CPU stall / host sync time; graph bucket; per-call prelude/coda counts beyond K+1 implied by the action

## ouro_fixed

reference-D distribution of checkpoints: D=4:512

### draft depth d at fixed K

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d1:K1 → DtV:d2:K1 | 4 | 512 (32) | 0.927 [0.918, 0.938] | +5.5 | 0.92→0.94 | 8.0→8.0 |  |
| **DtV:d1:K1 → DtV:d2:K1** | **no crossover — lower option dominates in every bin** | | | | | | |
| DtV:d1:K2 → DtV:d2:K2 | 4 | 512 (32) | 0.850 [0.838, 0.864] | +14.6 | 1.78→1.84 | 12.0→12.0 |  |
| **DtV:d1:K2 → DtV:d2:K2** | **no crossover — lower option dominates in every bin** | | | | | | |
| DtV:d1:K4 → DtV:d2:K4 | 4 | 512 (32) | 0.775 [0.761, 0.790] | +33.1 | 3.35→3.53 | 20.0→20.0 |  |
| **DtV:d1:K4 → DtV:d2:K4** | **no crossover — lower option dominates in every bin** | | | | | | |
| DtV:d1:K8 → DtV:d2:K8 | 4 | 512 (32) | 0.747 [0.727, 0.769] | +68.9 | 5.86→6.57 | 36.0→36.0 |  |
| **DtV:d1:K8 → DtV:d2:K8** | **no crossover — lower option dominates in every bin** | | | | | | |

### length K at fixed d

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d1:K1 → DtV:d1:K2 | 4 | 512 (32) | 1.220 [1.209, 1.231] | +11.6 | 0.92→1.78 | 8.0→12.0 |  |
| **DtV:d1:K1 → DtV:d1:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d1:K2 → DtV:d1:K4 | 4 | 512 (32) | 1.193 [1.181, 1.205] | +22.7 | 1.78→3.35 | 12.0→20.0 |  |
| **DtV:d1:K2 → DtV:d1:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d1:K4 → DtV:d1:K8 | 4 | 512 (32) | 1.048 [1.027, 1.070] | +48.5 | 3.35→5.86 | 20.0→36.0 |  |
| **DtV:d1:K4 → DtV:d1:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d2:K1 → DtV:d2:K2 | 4 | 512 (32) | 1.118 [1.110, 1.127] | +20.6 | 0.94→1.84 | 8.0→12.0 |  |
| **DtV:d2:K1 → DtV:d2:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d2:K2 → DtV:d2:K4 | 4 | 512 (32) | 1.087 [1.078, 1.097] | +41.2 | 1.84→3.53 | 12.0→20.0 |  |
| **DtV:d2:K2 → DtV:d2:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d2:K4 → DtV:d2:K8 | 4 | 512 (32) | 1.012 [0.997, 1.026] | +84.3 | 3.53→6.57 | 20.0→36.0 |  |
| **DtV:d2:K4 → DtV:d2:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### SD vs AR

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| AR → DtV:d1:K1 | 4 | 512 (32) | 1.212 [1.193, 1.230] | +22.8 | -→0.92 | 4.0→8.0 |  |
| **AR → DtV:d1:K1** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d1:K2 | 4 | 512 (32) | 1.478 [1.447, 1.509] | +34.4 | -→1.78 | 4.0→12.0 |  |
| **AR → DtV:d1:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d1:K4 | 4 | 512 (32) | 1.764 [1.717, 1.812] | +57.1 | -→3.35 | 4.0→20.0 |  |
| **AR → DtV:d1:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d1:K8 | 4 | 512 (32) | 1.850 [1.779, 1.925] | +105.6 | -→5.86 | 4.0→36.0 |  |
| **AR → DtV:d1:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d2:K1 | 4 | 512 (32) | 1.124 [1.110, 1.137] | +28.4 | -→0.94 | 4.0→8.0 |  |
| **AR → DtV:d2:K1** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d2:K2 | 4 | 512 (32) | 1.257 [1.237, 1.278] | +49.0 | -→1.84 | 4.0→12.0 |  |
| **AR → DtV:d2:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d2:K4 | 4 | 512 (32) | 1.367 [1.337, 1.396] | +90.2 | -→3.53 | 4.0→20.0 |  |
| **AR → DtV:d2:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d2:K8 | 4 | 512 (32) | 1.383 [1.340, 1.426] | +174.5 | -→6.57 | 4.0→36.0 |  |
| **AR → DtV:d2:K8** | **no crossover — higher option dominates in every bin** | | | | | | |

### confounding controls for DtV:d1:K4 → DtV:d2:K4 (post-hoc)

| stratum | n | U_hi/U_lo [95% CI] |
|---|---:|---:|
| accepted(lo)=0 | 40 | 1.841 [1.488, 2.232] |
| accepted(lo)=1 | 31 | 1.191 [1.026, 1.377] |
| accepted(lo)=2 | 21 | 1.050 [0.902, 1.190] |
| accepted(lo)=3 | 39 | 0.800 [0.757, 0.843] |
| accepted(lo)=4 | 381 | 0.728 [0.720, 0.735] |
| k<84 | 160 | 0.822 [0.787, 0.860] |
| 84≤k<164 | 160 | 0.765 [0.744, 0.789] |
| k≥164 | 192 | 0.747 [0.729, 0.766] |

## ouro_tau0.4

reference-D distribution of checkpoints: D=2:263, D=3:192, D=4:57

### draft depth d at fixed K

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d1:K1 → DtV:d2:K1 | 2 | 263 (32) | 0.876 [0.873, 0.880] | +5.8 | 1.00→1.00 | 4.4→4.4 |  |
| DtV:d1:K1 → DtV:d2:K1 | 3 | 192 (32) | 0.922 [0.905, 0.938] | +5.9 | 0.90→0.95 | 5.6→5.6 |  |
| DtV:d1:K1 → DtV:d2:K1 | 4 | 57 (31) | 0.982 [0.918, 1.058] | +5.4 | 0.58→0.68 | 7.5→7.5 |  |
| **DtV:d1:K1 → DtV:d2:K1** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d1:K2 → DtV:d2:K2 | 2 | 263 (32) | 0.786 [0.779, 0.793] | +15.6 | 1.97→1.98 | 6.8→6.8 |  |
| DtV:d1:K2 → DtV:d2:K2 | 3 | 192 (32) | 0.852 [0.827, 0.874] | +15.2 | 1.72→1.87 | 8.3→8.3 |  |
| DtV:d1:K2 → DtV:d2:K2 | 4 | 57 (31) | 0.922 [0.836, 1.030] | +15.8 | 0.98→1.21 | 10.8→10.9 |  |
| **DtV:d1:K2 → DtV:d2:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d1:K4 → DtV:d2:K4 | 2 | 263 (32) | 0.726 [0.711, 0.744] | +34.8 | 3.75→3.87 | 11.8→11.8 |  |
| DtV:d1:K4 → DtV:d2:K4 | 3 | 192 (32) | 0.792 [0.758, 0.827] | +34.9 | 3.19→3.62 | 13.5→13.5 |  |
| DtV:d1:K4 → DtV:d2:K4 | 4 | 57 (31) | 0.873 [0.756, 1.026] | +34.0 | 1.65→2.09 | 17.5→17.7 |  |
| **DtV:d1:K4 → DtV:d2:K4** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d1:K8 → DtV:d2:K8 | 2 | 263 (32) | 0.700 [0.677, 0.725] | +72.9 | 6.84→7.43 | 21.8→21.7 |  |
| DtV:d1:K8 → DtV:d2:K8 | 3 | 192 (32) | 0.785 [0.740, 0.834] | +72.5 | 5.58→6.84 | 23.8→23.7 |  |
| DtV:d1:K8 → DtV:d2:K8 | 4 | 57 (31) | 0.843 [0.691, 1.051] | +72.2 | 2.47→3.33 | 30.1→30.5 |  |
| **DtV:d1:K8 → DtV:d2:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### length K at fixed d

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d1:K1 → DtV:d1:K2 | 2 | 263 (32) | 1.086 [1.069, 1.103] | +15.2 | 1.00→1.97 | 4.4→6.8 |  |
| DtV:d1:K1 → DtV:d1:K2 | 3 | 192 (32) | 1.128 [1.110, 1.147] | +13.6 | 0.90→1.72 | 5.6→8.3 |  |
| DtV:d1:K1 → DtV:d1:K2 | 4 | 57 (31) | 1.048 [0.992, 1.094] | +12.5 | 0.58→0.98 | 7.5→10.8 |  |
| **DtV:d1:K1 → DtV:d1:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d1:K2 → DtV:d1:K4 | 2 | 263 (32) | 1.071 [1.053, 1.089] | +27.8 | 1.97→3.75 | 6.8→11.8 |  |
| DtV:d1:K2 → DtV:d1:K4 | 3 | 192 (32) | 1.105 [1.077, 1.131] | +25.2 | 1.72→3.19 | 8.3→13.5 |  |
| DtV:d1:K2 → DtV:d1:K4 | 4 | 57 (31) | 0.999 [0.923, 1.060] | +25.6 | 0.98→1.65 | 10.8→17.5 |  |
| **DtV:d1:K2 → DtV:d1:K4** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d1:K4 → DtV:d1:K8 | 2 | 263 (32) | 1.022 [1.004, 1.040] | +51.9 | 3.75→6.84 | 11.8→21.8 |  |
| DtV:d1:K4 → DtV:d1:K8 | 3 | 192 (32) | 0.997 [0.960, 1.033] | +50.9 | 3.19→5.58 | 13.5→23.8 |  |
| DtV:d1:K4 → DtV:d1:K8 | 4 | 57 (31) | 0.881 [0.796, 0.961] | +49.4 | 1.65→2.47 | 17.5→30.1 |  |
| **DtV:d1:K4 → DtV:d1:K8** | **reproducible crossover** | | | | | | |
| DtV:d2:K1 → DtV:d2:K2 | 2 | 263 (32) | 0.973 [0.961, 0.984] | +24.9 | 1.00→1.98 | 4.4→6.8 |  |
| DtV:d2:K1 → DtV:d2:K2 | 3 | 192 (32) | 1.042 [1.031, 1.053] | +22.9 | 0.95→1.87 | 5.6→8.3 |  |
| DtV:d2:K1 → DtV:d2:K2 | 4 | 57 (31) | 0.984 [0.927, 1.030] | +22.9 | 0.68→1.21 | 7.5→10.9 |  |
| **DtV:d2:K1 → DtV:d2:K2** | **reproducible crossover** | | | | | | |
| DtV:d2:K2 → DtV:d2:K4 | 2 | 263 (32) | 0.989 [0.980, 1.000] | +47.0 | 1.98→3.87 | 6.8→11.8 |  |
| DtV:d2:K2 → DtV:d2:K4 | 3 | 192 (32) | 1.028 [1.015, 1.040] | +44.8 | 1.87→3.62 | 8.3→13.5 |  |
| DtV:d2:K2 → DtV:d2:K4 | 4 | 57 (31) | 0.945 [0.873, 1.000] | +43.8 | 1.21→2.09 | 10.9→17.7 |  |
| **DtV:d2:K2 → DtV:d2:K4** | **reproducible crossover** | | | | | | |
| DtV:d2:K4 → DtV:d2:K8 | 2 | 263 (32) | 0.986 [0.975, 0.996] | +89.9 | 3.87→7.43 | 11.8→21.7 |  |
| DtV:d2:K4 → DtV:d2:K8 | 3 | 192 (32) | 0.988 [0.965, 1.008] | +88.6 | 3.62→6.84 | 13.5→23.7 |  |
| DtV:d2:K4 → DtV:d2:K8 | 4 | 57 (31) | 0.852 [0.783, 0.906] | +87.6 | 2.09→3.33 | 17.7→30.5 |  |
| **DtV:d2:K4 → DtV:d2:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### SD vs AR

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| AR → DtV:d1:K1 | 2 | 263 (32) | 1.047 [1.029, 1.065] | +19.7 | -→1.00 | 2.0→4.4 |  |
| AR → DtV:d1:K1 | 3 | 192 (32) | 1.184 [1.160, 1.211] | +18.8 | -→0.90 | 3.0→5.6 |  |
| AR → DtV:d1:K1 | 4 | 57 (31) | 1.023 [0.940, 1.108] | +22.2 | -→0.58 | 4.0→7.5 |  |
| **AR → DtV:d1:K1** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d1:K2 | 2 | 263 (32) | 1.137 [1.116, 1.158] | +34.8 | -→1.97 | 2.0→6.8 |  |
| AR → DtV:d1:K2 | 3 | 192 (32) | 1.336 [1.291, 1.386] | +32.4 | -→1.72 | 3.0→8.3 |  |
| AR → DtV:d1:K2 | 4 | 57 (31) | 1.073 [0.952, 1.190] | +34.7 | -→0.98 | 4.0→10.8 |  |
| **AR → DtV:d1:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d1:K4 | 2 | 263 (32) | 1.217 [1.184, 1.250] | +62.6 | -→3.75 | 2.0→11.8 |  |
| AR → DtV:d1:K4 | 3 | 192 (32) | 1.477 [1.399, 1.557] | +57.6 | -→3.19 | 3.0→13.5 |  |
| AR → DtV:d1:K4 | 4 | 57 (31) | 1.071 [0.896, 1.235] | +60.3 | -→1.65 | 4.0→17.5 |  |
| **AR → DtV:d1:K4** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d1:K8 | 2 | 263 (32) | 1.244 [1.201, 1.289] | +114.5 | -→6.84 | 2.0→21.8 |  |
| AR → DtV:d1:K8 | 3 | 192 (32) | 1.473 [1.356, 1.595] | +108.5 | -→5.58 | 3.0→23.8 |  |
| AR → DtV:d1:K8 | 4 | 57 (31) | 0.944 [0.745, 1.134] | +109.7 | -→2.47 | 4.0→30.1 |  |
| **AR → DtV:d1:K8** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| AR → DtV:d2:K1 | 2 | 263 (32) | 0.917 [0.903, 0.931] | +25.5 | -→1.00 | 2.0→4.4 |  |
| AR → DtV:d2:K1 | 3 | 192 (32) | 1.092 [1.075, 1.110] | +24.7 | -→0.95 | 3.0→5.6 |  |
| AR → DtV:d2:K1 | 4 | 57 (31) | 1.005 [0.936, 1.066] | +27.6 | -→0.68 | 4.0→7.5 |  |
| **AR → DtV:d2:K1** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K2 | 2 | 263 (32) | 0.893 [0.879, 0.906] | +50.4 | -→1.98 | 2.0→6.8 |  |
| AR → DtV:d2:K2 | 3 | 192 (32) | 1.138 [1.111, 1.165] | +47.6 | -→1.87 | 3.0→8.3 |  |
| AR → DtV:d2:K2 | 4 | 57 (31) | 0.989 [0.890, 1.083] | +50.5 | -→1.21 | 4.0→10.9 |  |
| **AR → DtV:d2:K2** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K4 | 2 | 263 (32) | 0.883 [0.868, 0.898] | +97.4 | -→3.87 | 2.0→11.8 |  |
| AR → DtV:d2:K4 | 3 | 192 (32) | 1.170 [1.135, 1.204] | +92.5 | -→3.62 | 3.0→13.5 |  |
| AR → DtV:d2:K4 | 4 | 57 (31) | 0.935 [0.790, 1.064] | +94.3 | -→2.09 | 4.0→17.7 |  |
| **AR → DtV:d2:K4** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K8 | 2 | 263 (32) | 0.871 [0.850, 0.890] | +187.4 | -→7.43 | 2.0→21.7 |  |
| AR → DtV:d2:K8 | 3 | 192 (32) | 1.156 [1.103, 1.208] | +181.1 | -→6.84 | 3.0→23.7 |  |
| AR → DtV:d2:K8 | 4 | 57 (31) | 0.796 [0.633, 0.939] | +181.9 | -→3.33 | 4.0→30.5 |  |
| **AR → DtV:d2:K8** | **reproducible crossover** | | | | | | |

### confounding controls for DtV:d1:K4 → DtV:d2:K4 (post-hoc)

| stratum | n | U_hi/U_lo [95% CI] |
|---|---:|---:|
| accepted(lo)=0 | 44 | 2.069 [1.746, 2.391] |
| accepted(lo)=1 | 32 | 1.284 [1.088, 1.490] |
| accepted(lo)=2 | 28 | 0.968 [0.863, 1.060] |
| accepted(lo)=3 | 28 | 0.854 [0.824, 0.884] |
| accepted(lo)=4 | 380 | 0.698 [0.689, 0.705] |
| k<84 | 160 | 0.779 [0.748, 0.813] |
| 84≤k<164 | 160 | 0.786 [0.750, 0.825] |
| k≥164 | 192 | 0.731 [0.711, 0.752] |

## ouro_tau0.1

reference-D distribution of checkpoints: D=1:299, D=2:94, D=3:69, D=4:50

### draft depth d at fixed K

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d1:K1 → DtV:d2:K1 | 1 | 299 (32) | 1.001 [0.996, 1.007] | -0.0 | 1.00→1.00 | 2.5→2.5 |  |
| DtV:d1:K1 → DtV:d2:K1 | 2 | 94 (30) | 0.974 [0.937, 1.015] | +3.2 | 0.89→1.00 | 3.9→3.9 |  |
| DtV:d1:K1 → DtV:d2:K1 | 3 | 69 (30) | 0.967 [0.929, 1.012] | +3.7 | 0.83→0.90 | 5.5→5.5 |  |
| DtV:d1:K1 → DtV:d2:K1 | 4 | 50 (26) | 1.008 [0.938, 1.094] | +4.0 | 0.60→0.72 | 6.8→6.7 |  |
| **DtV:d1:K1 → DtV:d2:K1** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d1:K2 → DtV:d2:K2 | 1 | 299 (32) | 0.988 [0.979, 0.997] | +0.8 | 1.96→1.98 | 4.0→4.0 |  |
| DtV:d1:K2 → DtV:d2:K2 | 2 | 94 (30) | 0.966 [0.918, 1.027] | +6.8 | 1.72→1.97 | 5.8→5.7 |  |
| DtV:d1:K2 → DtV:d2:K2 | 3 | 69 (30) | 0.943 [0.884, 1.009] | +8.6 | 1.55→1.74 | 7.5→7.5 |  |
| DtV:d1:K2 → DtV:d2:K2 | 4 | 50 (26) | 0.988 [0.885, 1.119] | +10.2 | 1.08→1.34 | 9.4→9.3 |  |
| **DtV:d1:K2 → DtV:d2:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d1:K4 → DtV:d2:K4 | 1 | 299 (32) | 0.979 [0.960, 0.996] | +4.2 | 3.70→3.86 | 7.2→7.2 |  |
| DtV:d1:K4 → DtV:d2:K4 | 2 | 94 (30) | 0.986 [0.930, 1.051] | +14.8 | 3.15→3.84 | 9.3→9.1 |  |
| DtV:d1:K4 → DtV:d2:K4 | 3 | 69 (30) | 0.922 [0.846, 1.002] | +17.8 | 2.84→3.25 | 11.6→11.4 |  |
| DtV:d1:K4 → DtV:d2:K4 | 4 | 50 (26) | 1.005 [0.847, 1.197] | +20.3 | 1.84→2.44 | 14.1→13.8 |  |
| **DtV:d1:K4 → DtV:d2:K4** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d1:K8 → DtV:d2:K8 | 1 | 299 (32) | 0.982 [0.957, 1.008] | +14.0 | 6.65→7.34 | 13.8→13.5 |  |
| DtV:d1:K8 → DtV:d2:K8 | 2 | 94 (30) | 1.067 [0.987, 1.172] | +31.8 | 5.35→7.41 | 16.4→15.9 |  |
| DtV:d1:K8 → DtV:d2:K8 | 3 | 69 (30) | 0.914 [0.815, 1.027] | +34.9 | 4.97→5.84 | 19.4→19.0 |  |
| DtV:d1:K8 → DtV:d2:K8 | 4 | 50 (26) | 1.002 [0.811, 1.236] | +42.7 | 2.94→4.08 | 23.8→23.4 |  |
| **DtV:d1:K8 → DtV:d2:K8** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |

### length K at fixed d

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d1:K1 → DtV:d1:K2 | 1 | 299 (32) | 0.975 [0.959, 0.992] | +14.8 | 1.00→1.96 | 2.5→4.0 |  |
| DtV:d1:K1 → DtV:d1:K2 | 2 | 94 (30) | 1.037 [1.002, 1.071] | +14.9 | 0.89→1.72 | 3.9→5.8 |  |
| DtV:d1:K1 → DtV:d1:K2 | 3 | 69 (30) | 1.121 [1.082, 1.152] | +12.3 | 0.83→1.55 | 5.5→7.5 |  |
| DtV:d1:K1 → DtV:d1:K2 | 4 | 50 (26) | 1.066 [1.008, 1.123] | +13.2 | 0.60→1.08 | 6.8→9.4 |  |
| **DtV:d1:K1 → DtV:d1:K2** | **reproducible crossover** | | | | | | |
| DtV:d1:K2 → DtV:d1:K4 | 1 | 299 (32) | 0.946 [0.921, 0.971] | +29.4 | 1.96→3.70 | 4.0→7.2 |  |
| DtV:d1:K2 → DtV:d1:K4 | 2 | 94 (30) | 1.002 [0.960, 1.038] | +27.8 | 1.72→3.15 | 5.8→9.3 |  |
| DtV:d1:K2 → DtV:d1:K4 | 3 | 69 (30) | 1.045 [1.010, 1.078] | +27.4 | 1.55→2.84 | 7.5→11.6 |  |
| DtV:d1:K2 → DtV:d1:K4 | 4 | 50 (26) | 1.015 [0.946, 1.081] | +25.3 | 1.08→1.84 | 9.4→14.1 |  |
| **DtV:d1:K2 → DtV:d1:K4** | **reproducible crossover** | | | | | | |
| DtV:d1:K4 → DtV:d1:K8 | 1 | 299 (32) | 0.941 [0.920, 0.962] | +53.0 | 3.70→6.65 | 7.2→13.8 |  |
| DtV:d1:K4 → DtV:d1:K8 | 2 | 94 (30) | 0.943 [0.904, 0.987] | +50.6 | 3.15→5.35 | 9.3→16.4 |  |
| DtV:d1:K4 → DtV:d1:K8 | 3 | 69 (30) | 1.010 [0.962, 1.054] | +48.2 | 2.84→4.97 | 11.6→19.4 |  |
| DtV:d1:K4 → DtV:d1:K8 | 4 | 50 (26) | 0.921 [0.841, 0.999] | +50.0 | 1.84→2.94 | 14.1→23.8 |  |
| **DtV:d1:K4 → DtV:d1:K8** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d2:K1 → DtV:d2:K2 | 1 | 299 (32) | 0.962 [0.942, 0.983] | +15.7 | 1.00→1.98 | 2.5→4.0 |  |
| DtV:d2:K1 → DtV:d2:K2 | 2 | 94 (30) | 1.029 [0.992, 1.062] | +18.5 | 1.00→1.97 | 3.9→5.7 |  |
| DtV:d2:K1 → DtV:d2:K2 | 3 | 69 (30) | 1.092 [1.064, 1.121] | +17.2 | 0.90→1.74 | 5.5→7.5 |  |
| DtV:d2:K1 → DtV:d2:K2 | 4 | 50 (26) | 1.044 [0.986, 1.094] | +19.4 | 0.72→1.34 | 6.7→9.3 |  |
| **DtV:d2:K1 → DtV:d2:K2** | **reproducible crossover** | | | | | | |
| DtV:d2:K2 → DtV:d2:K4 | 1 | 299 (32) | 0.937 [0.918, 0.957] | +32.7 | 1.98→3.86 | 4.0→7.2 |  |
| DtV:d2:K2 → DtV:d2:K4 | 2 | 94 (30) | 1.023 [0.993, 1.056] | +35.8 | 1.97→3.84 | 5.7→9.1 |  |
| DtV:d2:K2 → DtV:d2:K4 | 3 | 69 (30) | 1.021 [0.977, 1.061] | +36.6 | 1.74→3.25 | 7.5→11.4 |  |
| DtV:d2:K2 → DtV:d2:K4 | 4 | 50 (26) | 1.032 [0.963, 1.092] | +35.4 | 1.34→2.44 | 9.3→13.8 |  |
| **DtV:d2:K2 → DtV:d2:K4** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d2:K4 → DtV:d2:K8 | 1 | 299 (32) | 0.944 [0.922, 0.966] | +62.8 | 3.86→7.34 | 7.2→13.5 |  |
| DtV:d2:K4 → DtV:d2:K8 | 2 | 94 (30) | 1.020 [0.987, 1.056] | +67.6 | 3.84→7.41 | 9.1→15.9 |  |
| DtV:d2:K4 → DtV:d2:K8 | 3 | 69 (30) | 1.002 [0.959, 1.042] | +65.3 | 3.25→5.84 | 11.4→19.0 |  |
| DtV:d2:K4 → DtV:d2:K8 | 4 | 50 (26) | 0.918 [0.850, 0.974] | +72.4 | 2.44→4.08 | 13.8→23.4 |  |
| **DtV:d2:K4 → DtV:d2:K8** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |

### SD vs AR

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| AR → DtV:d1:K1 | 1 | 299 (32) | 0.818 [0.791, 0.847] | +16.9 | -→1.00 | 1.0→2.5 |  |
| AR → DtV:d1:K1 | 2 | 94 (30) | 1.053 [1.001, 1.106] | +17.1 | -→0.89 | 2.0→3.9 |  |
| AR → DtV:d1:K1 | 3 | 69 (30) | 1.140 [1.072, 1.203] | +18.7 | -→0.83 | 3.0→5.5 |  |
| AR → DtV:d1:K1 | 4 | 50 (26) | 1.089 [0.988, 1.193] | +19.2 | -→0.60 | 4.0→6.8 |  |
| **AR → DtV:d1:K1** | **reproducible crossover** | | | | | | |
| AR → DtV:d1:K2 | 1 | 299 (32) | 0.798 [0.767, 0.830] | +31.7 | -→1.96 | 1.0→4.0 |  |
| AR → DtV:d1:K2 | 2 | 94 (30) | 1.092 [1.017, 1.166] | +32.0 | -→1.72 | 2.0→5.8 |  |
| AR → DtV:d1:K2 | 3 | 69 (30) | 1.278 [1.177, 1.368] | +31.0 | -→1.55 | 3.0→7.5 |  |
| AR → DtV:d1:K2 | 4 | 50 (26) | 1.161 [1.017, 1.313] | +32.4 | -→1.08 | 4.0→9.4 |  |
| **AR → DtV:d1:K2** | **reproducible crossover** | | | | | | |
| AR → DtV:d1:K4 | 1 | 299 (32) | 0.754 [0.719, 0.790] | +61.1 | -→3.70 | 1.0→7.2 |  |
| AR → DtV:d1:K4 | 2 | 94 (30) | 1.094 [0.999, 1.187] | +59.8 | -→3.15 | 2.0→9.3 |  |
| AR → DtV:d1:K4 | 3 | 69 (30) | 1.335 [1.207, 1.459] | +58.4 | -→2.84 | 3.0→11.6 |  |
| AR → DtV:d1:K4 | 4 | 50 (26) | 1.178 [0.986, 1.387] | +57.7 | -→1.84 | 4.0→14.1 |  |
| **AR → DtV:d1:K4** | **reproducible crossover** | | | | | | |
| AR → DtV:d1:K8 | 1 | 299 (32) | 0.710 [0.671, 0.752] | +114.0 | -→6.65 | 1.0→13.8 |  |
| AR → DtV:d1:K8 | 2 | 94 (30) | 1.031 [0.924, 1.144] | +110.3 | -→5.35 | 2.0→16.4 |  |
| AR → DtV:d1:K8 | 3 | 69 (30) | 1.349 [1.185, 1.503] | +106.5 | -→4.97 | 3.0→19.4 |  |
| AR → DtV:d1:K8 | 4 | 50 (26) | 1.085 [0.854, 1.342] | +107.7 | -→2.94 | 4.0→23.8 |  |
| **AR → DtV:d1:K8** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K1 | 1 | 299 (32) | 0.819 [0.791, 0.849] | +16.8 | -→1.00 | 1.0→2.5 |  |
| AR → DtV:d2:K1 | 2 | 94 (30) | 1.025 [0.991, 1.061] | +20.3 | -→1.00 | 2.0→3.9 |  |
| AR → DtV:d2:K1 | 3 | 69 (30) | 1.103 [1.044, 1.164] | +22.4 | -→0.90 | 3.0→5.5 |  |
| AR → DtV:d2:K1 | 4 | 50 (26) | 1.098 [1.007, 1.189] | +23.2 | -→0.72 | 4.0→6.7 |  |
| **AR → DtV:d2:K1** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K2 | 1 | 299 (32) | 0.788 [0.756, 0.822] | +32.5 | -→1.98 | 1.0→4.0 |  |
| AR → DtV:d2:K2 | 2 | 94 (30) | 1.055 [1.007, 1.104] | +38.8 | -→1.97 | 2.0→5.7 |  |
| AR → DtV:d2:K2 | 3 | 69 (30) | 1.205 [1.127, 1.285] | +39.6 | -→1.74 | 3.0→7.5 |  |
| AR → DtV:d2:K2 | 4 | 50 (26) | 1.147 [1.003, 1.286] | +42.6 | -→1.34 | 4.0→9.3 |  |
| **AR → DtV:d2:K2** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K4 | 1 | 299 (32) | 0.738 [0.704, 0.776] | +65.2 | -→3.86 | 1.0→7.2 |  |
| AR → DtV:d2:K4 | 2 | 94 (30) | 1.079 [1.025, 1.134] | +74.6 | -→3.84 | 2.0→9.1 |  |
| AR → DtV:d2:K4 | 3 | 69 (30) | 1.230 [1.127, 1.328] | +76.2 | -→3.25 | 3.0→11.4 |  |
| AR → DtV:d2:K4 | 4 | 50 (26) | 1.184 [0.980, 1.388] | +78.0 | -→2.44 | 4.0→13.8 |  |
| **AR → DtV:d2:K4** | **reproducible crossover** | | | | | | |
| AR → DtV:d2:K8 | 1 | 299 (32) | 0.697 [0.659, 0.737] | +128.0 | -→7.34 | 1.0→13.5 |  |
| AR → DtV:d2:K8 | 2 | 94 (30) | 1.101 [1.030, 1.167] | +142.1 | -→7.41 | 2.0→15.9 |  |
| AR → DtV:d2:K8 | 3 | 69 (30) | 1.232 [1.095, 1.367] | +141.5 | -→5.84 | 3.0→19.0 |  |
| AR → DtV:d2:K8 | 4 | 50 (26) | 1.087 [0.846, 1.339] | +150.4 | -→4.08 | 4.0→23.4 |  |
| **AR → DtV:d2:K8** | **reproducible crossover** | | | | | | |

### confounding controls for DtV:d1:K4 → DtV:d2:K4 (post-hoc)

| stratum | n | U_hi/U_lo [95% CI] |
|---|---:|---:|
| accepted(lo)=0 | 42 | 2.599 [2.023, 3.171] |
| accepted(lo)=1 | 31 | 1.474 [1.288, 1.700] |
| accepted(lo)=2 | 34 | 1.180 [1.051, 1.280] |
| accepted(lo)=3 | 29 | 0.988 [0.932, 1.047] |
| accepted(lo)=4 | 376 | 0.908 [0.889, 0.925] |
| k<84 | 160 | 0.963 [0.922, 1.009] |
| 84≤k<164 | 160 | 0.964 [0.916, 1.025] |
| k≥164 | 192 | 0.962 [0.941, 0.986] |

## huginn_fixed

reference-D distribution of checkpoints: D=32:384

### draft depth d at fixed K

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d2:K2 → DtV:d4:K2 | 32 (no exit) | 384 (24) | 1.275 [1.220, 1.328] | +5.4 | 1.03→1.67 | 96.0→96.0 |  |
| **DtV:d2:K2 → DtV:d4:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K2 → DtV:d8:K2 | 32 (no exit) | 384 (24) | 1.020 [0.998, 1.045] | +10.8 | 1.67→1.89 | 96.0→96.0 |  |
| **DtV:d4:K2 → DtV:d8:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d2:K4 → DtV:d4:K4 | 32 (no exit) | 384 (24) | 1.512 [1.415, 1.606] | +15.2 | 1.47→3.06 | 160.0→160.0 |  |
| **DtV:d2:K4 → DtV:d4:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K4 → DtV:d8:K4 | 32 (no exit) | 384 (24) | 0.990 [0.963, 1.022] | +31.1 | 3.06→3.68 | 160.0→160.0 |  |
| **DtV:d4:K4 → DtV:d8:K4** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d2:K8 → DtV:d4:K8 | 32 (no exit) | 384 (24) | 2.025 [1.876, 2.171] | +32.5 | 1.72→5.24 | 288.0→288.0 |  |
| **DtV:d2:K8 → DtV:d4:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K8 → DtV:d8:K8 | 32 (no exit) | 384 (24) | 1.028 [0.994, 1.065] | +65.2 | 5.24→6.92 | 288.0→288.0 |  |
| **DtV:d4:K8 → DtV:d8:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### length K at fixed d

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d2:K2 → DtV:d2:K4 | 32 (no exit) | 384 (24) | 1.157 [1.115, 1.198] | +8.8 | 1.03→1.47 | 96.0→160.0 |  |
| **DtV:d2:K2 → DtV:d2:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d2:K4 → DtV:d2:K8 | 32 (no exit) | 384 (24) | 0.781 [0.759, 0.805] | +71.6 | 1.47→1.72 | 160.0→288.0 |  |
| **DtV:d2:K4 → DtV:d2:K8** | **no crossover — lower option dominates in every bin** | | | | | | |
| DtV:d4:K2 → DtV:d4:K4 | 32 (no exit) | 384 (24) | 1.372 [1.348, 1.392] | +18.7 | 1.67→3.06 | 96.0→160.0 |  |
| **DtV:d4:K2 → DtV:d4:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K4 → DtV:d4:K8 | 32 (no exit) | 384 (24) | 1.046 [1.025, 1.067] | +88.9 | 3.06→5.24 | 160.0→288.0 |  |
| **DtV:d4:K4 → DtV:d4:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d8:K2 → DtV:d8:K4 | 32 (no exit) | 384 (24) | 1.333 [1.318, 1.344] | +39.1 | 1.89→3.68 | 96.0→160.0 |  |
| **DtV:d8:K2 → DtV:d8:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d8:K4 → DtV:d8:K8 | 32 (no exit) | 384 (24) | 1.086 [1.068, 1.102] | +123.0 | 3.68→6.92 | 160.0→288.0 |  |
| **DtV:d8:K4 → DtV:d8:K8** | **no crossover — higher option dominates in every bin** | | | | | | |

### SD vs AR

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| AR → DtV:d2:K2 | 32 (no exit) | 384 (24) | 0.989 [0.954, 1.027] | +84.8 | -→1.03 | 32.0→96.0 |  |
| **AR → DtV:d2:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d2:K4 | 32 (no exit) | 384 (24) | 1.144 [1.079, 1.214] | +93.6 | -→1.47 | 32.0→160.0 |  |
| **AR → DtV:d2:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d2:K8 | 32 (no exit) | 384 (24) | 0.894 [0.828, 0.963] | +165.3 | -→1.72 | 32.0→288.0 |  |
| **AR → DtV:d2:K8** | **no crossover — lower option dominates in every bin** | | | | | | |
| AR → DtV:d4:K2 | 32 (no exit) | 384 (24) | 1.261 [1.227, 1.293] | +90.1 | -→1.67 | 32.0→96.0 |  |
| **AR → DtV:d4:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d4:K4 | 32 (no exit) | 384 (24) | 1.730 [1.662, 1.793] | +108.9 | -→3.06 | 32.0→160.0 |  |
| **AR → DtV:d4:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d4:K8 | 32 (no exit) | 384 (24) | 1.810 [1.716, 1.897] | +197.8 | -→5.24 | 32.0→288.0 |  |
| **AR → DtV:d4:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d8:K2 | 32 (no exit) | 384 (24) | 1.286 [1.256, 1.309] | +100.9 | -→1.89 | 32.0→96.0 |  |
| **AR → DtV:d8:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d8:K4 | 32 (no exit) | 384 (24) | 1.713 [1.659, 1.755] | +140.0 | -→3.68 | 32.0→160.0 |  |
| **AR → DtV:d8:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d8:K8 | 32 (no exit) | 384 (24) | 1.860 [1.793, 1.916] | +263.0 | -→6.92 | 32.0→288.0 |  |
| **AR → DtV:d8:K8** | **no crossover — higher option dominates in every bin** | | | | | | |

### confounding controls for DtV:d4:K8 → DtV:d8:K8 (post-hoc)

| stratum | n | U_hi/U_lo [95% CI] |
|---|---:|---:|
| accepted(lo)=0 | 49 | 4.643 [3.687, 5.609] |
| accepted(lo)=1 | 29 | 2.836 [2.390, 3.199] |
| accepted(lo)=2 | 26 | 2.080 [1.866, 2.319] |
| accepted(lo)=3 | 25 | 1.284 [1.091, 1.463] |
| accepted(lo)=4 | 21 | 1.158 [0.981, 1.304] |
| accepted(lo)=5 | 19 | 1.172 [1.119, 1.215] |
| accepted(lo)=6 | 13 | 0.931 [0.833, 1.001] |
| accepted(lo)=7 | 15 | 0.888 [0.856, 0.916] |
| accepted(lo)=8 | 187 | 0.794 [0.781, 0.804] |
| k<84 | 120 | 1.090 [0.995, 1.195] |
| 84≤k<164 | 120 | 0.992 [0.914, 1.108] |
| k≥164 | 144 | 1.019 [0.958, 1.080] |

## huginn_eps0.1

reference-D distribution of checkpoints: D=13:12, D=14:28, D=15:94, D=16:104, D=17:79, D=18:34, D=19:16, D=20:7, D=21:3, D=22:1, D=23:1, D=25:3, D=26:1, D=32:1

### draft depth d at fixed K

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d2:K2 → DtV:d4:K2 | ≤14 | 40 (16) | 1.201 [1.062, 1.384] | +5.4 | 1.20→1.80 | 44.1→43.8 |  |
| DtV:d2:K2 → DtV:d4:K2 | 15 | 94 (24) | 1.197 [1.126, 1.286] | +4.8 | 1.15→1.70 | 46.9→46.3 |  |
| DtV:d2:K2 → DtV:d4:K2 | 16 | 104 (23) | 1.207 [1.143, 1.269] | +4.5 | 1.13→1.69 | 48.9→48.3 |  |
| DtV:d2:K2 → DtV:d4:K2 | 17 | 79 (24) | 1.275 [1.137, 1.423] | +4.7 | 1.09→1.78 | 50.9→50.1 |  |
| DtV:d2:K2 → DtV:d4:K2 | 18–20 | 57 (23) | 1.246 [1.090, 1.403] | +5.7 | 0.96→1.58 | 53.8→53.1 |  |
| DtV:d2:K2 → DtV:d4:K2 | 21–31 (tail) | 9 (8) | 1.986 [1.423, 2.879] | +0.9 | 0.33→1.67 | 63.2→58.2 | sparse |
| DtV:d2:K2 → DtV:d4:K2 | 32 (no exit) | 1 (1) | 0.959 [nan, nan] | +6.0 | 2.00→2.00 | 66.0→66.0 | sparse |
| **DtV:d2:K2 → DtV:d4:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K2 → DtV:d8:K2 | ≤14 | 40 (16) | 0.903 [0.844, 0.961] | +11.4 | 1.80→1.82 | 43.8→43.6 |  |
| DtV:d4:K2 → DtV:d8:K2 | 15 | 94 (24) | 0.967 [0.918, 1.022] | +10.9 | 1.70→1.89 | 46.3→45.9 |  |
| DtV:d4:K2 → DtV:d8:K2 | 16 | 104 (23) | 0.969 [0.933, 1.004] | +11.5 | 1.69→1.89 | 48.3→48.2 |  |
| DtV:d4:K2 → DtV:d8:K2 | 17 | 79 (24) | 0.952 [0.912, 0.994] | +11.4 | 1.78→1.94 | 50.1→49.9 |  |
| DtV:d4:K2 → DtV:d8:K2 | 18–20 | 57 (23) | 1.006 [0.948, 1.080] | +10.8 | 1.58→1.84 | 53.1→53.2 |  |
| DtV:d4:K2 → DtV:d8:K2 | 21–31 (tail) | 9 (8) | 0.997 [0.910, 1.256] | +10.8 | 1.67→1.89 | 58.2→57.4 | sparse |
| DtV:d4:K2 → DtV:d8:K2 | 32 (no exit) | 1 (1) | 0.922 [nan, nan] | +12.4 | 2.00→2.00 | 66.0→66.0 | sparse |
| **DtV:d4:K2 → DtV:d8:K2** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d2:K4 → DtV:d4:K4 | ≤14 | 40 (16) | 1.317 [1.153, 1.529] | +15.6 | 1.90→3.35 | 76.2→75.2 |  |
| DtV:d2:K4 → DtV:d4:K4 | 15 | 94 (24) | 1.315 [1.216, 1.444] | +15.4 | 1.80→3.17 | 79.8→78.6 |  |
| DtV:d2:K4 → DtV:d4:K4 | 16 | 104 (23) | 1.408 [1.310, 1.506] | +13.5 | 1.67→3.19 | 82.9→81.2 |  |
| DtV:d2:K4 → DtV:d4:K4 | 17 | 79 (24) | 1.592 [1.388, 1.833] | +13.7 | 1.46→3.35 | 85.8→83.3 |  |
| DtV:d2:K4 → DtV:d4:K4 | 18–20 | 57 (23) | 1.481 [1.233, 1.724] | +16.0 | 1.35→2.93 | 88.9→87.6 |  |
| DtV:d2:K4 → DtV:d4:K4 | 21–31 (tail) | 9 (8) | 2.810 [1.935, 4.177] | +9.6 | 0.33→3.00 | 105.1→89.9 | sparse |
| DtV:d2:K4 → DtV:d4:K4 | 32 (no exit) | 1 (1) | 0.904 [nan, nan] | +16.7 | 4.00→4.00 | 99.0→99.0 | sparse |
| **DtV:d2:K4 → DtV:d4:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K4 → DtV:d8:K4 | ≤14 | 40 (16) | 0.838 [0.759, 0.911] | +32.7 | 3.35→3.58 | 75.2→74.9 |  |
| DtV:d4:K4 → DtV:d8:K4 | 15 | 94 (24) | 0.895 [0.837, 0.962] | +31.8 | 3.17→3.64 | 78.6→77.7 |  |
| DtV:d4:K4 → DtV:d8:K4 | 16 | 104 (23) | 0.903 [0.868, 0.942] | +32.7 | 3.19→3.72 | 81.2→80.8 |  |
| DtV:d4:K4 → DtV:d8:K4 | 17 | 79 (24) | 0.883 [0.841, 0.928] | +32.4 | 3.35→3.77 | 83.3→82.5 |  |
| DtV:d4:K4 → DtV:d8:K4 | 18–20 | 57 (23) | 0.933 [0.855, 1.027] | +32.8 | 2.93→3.53 | 87.6→87.6 |  |
| DtV:d4:K4 → DtV:d8:K4 | 21–31 (tail) | 9 (8) | 0.965 [0.840, 1.288] | +31.6 | 3.00→3.67 | 89.9→90.0 | sparse |
| DtV:d4:K4 → DtV:d8:K4 | 32 (no exit) | 1 (1) | 0.823 [nan, nan] | +37.4 | 4.00→4.00 | 99.0→99.0 | sparse |
| **DtV:d4:K4 → DtV:d8:K4** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| DtV:d2:K8 → DtV:d4:K8 | ≤14 | 40 (16) | 1.625 [1.423, 1.891] | +33.3 | 2.55→5.88 | 142.2→139.7 |  |
| DtV:d2:K8 → DtV:d4:K8 | 15 | 94 (24) | 1.712 [1.514, 1.973] | +30.9 | 2.30→5.64 | 146.6→143.0 |  |
| DtV:d2:K8 → DtV:d4:K8 | 16 | 104 (23) | 1.870 [1.682, 2.068] | +31.1 | 2.13→5.88 | 151.5→146.3 |  |
| DtV:d2:K8 → DtV:d4:K8 | 17 | 79 (24) | 2.326 [1.985, 2.739] | +27.9 | 1.67→6.16 | 157.5→149.0 |  |
| DtV:d2:K8 → DtV:d4:K8 | 18–20 | 57 (23) | 2.114 [1.685, 2.555] | +31.6 | 1.44→5.04 | 161.3→154.3 |  |
| DtV:d2:K8 → DtV:d4:K8 | 21–31 (tail) | 9 (8) | 4.228 [2.849, 6.496] | +21.3 | 0.33→5.22 | 188.6→150.7 | sparse |
| DtV:d2:K8 → DtV:d4:K8 | 32 (no exit) | 1 (1) | 1.551 [nan, nan] | +34.5 | 4.00→8.00 | 161.0→161.0 | sparse |
| **DtV:d2:K8 → DtV:d4:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K8 → DtV:d8:K8 | ≤14 | 40 (16) | 0.873 [0.802, 0.946] | +67.9 | 5.88→6.97 | 139.7→139.2 |  |
| DtV:d4:K8 → DtV:d8:K8 | 15 | 94 (24) | 0.888 [0.814, 0.965] | +67.8 | 5.64→6.83 | 143.0→141.9 |  |
| DtV:d4:K8 → DtV:d8:K8 | 16 | 104 (23) | 0.903 [0.859, 0.953] | +68.6 | 5.88→7.25 | 146.3→145.9 |  |
| DtV:d4:K8 → DtV:d8:K8 | 17 | 79 (24) | 0.858 [0.804, 0.915] | +67.3 | 6.16→7.13 | 149.0→147.8 |  |
| DtV:d4:K8 → DtV:d8:K8 | 18–20 | 57 (23) | 0.940 [0.851, 1.058] | +68.1 | 5.04→6.46 | 154.3→154.3 |  |
| DtV:d4:K8 → DtV:d8:K8 | 21–31 (tail) | 9 (8) | 1.009 [0.808, 1.485] | +70.1 | 5.22→7.22 | 150.7→152.9 | sparse |
| DtV:d4:K8 → DtV:d8:K8 | 32 (no exit) | 1 (1) | 0.781 [nan, nan] | +69.8 | 8.00→8.00 | 161.0→161.0 | sparse |
| **DtV:d4:K8 → DtV:d8:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### length K at fixed d

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| DtV:d2:K2 → DtV:d2:K4 | ≤14 | 40 (16) | 1.072 [0.996, 1.140] | +21.0 | 1.20→1.90 | 44.1→76.2 |  |
| DtV:d2:K2 → DtV:d2:K4 | 15 | 94 (24) | 1.081 [1.011, 1.148] | +19.6 | 1.15→1.80 | 46.9→79.8 |  |
| DtV:d2:K2 → DtV:d2:K4 | 16 | 104 (23) | 1.053 [1.008, 1.097] | +18.9 | 1.13→1.67 | 48.9→82.9 |  |
| DtV:d2:K2 → DtV:d2:K4 | 17 | 79 (24) | 0.994 [0.934, 1.056] | +18.6 | 1.09→1.46 | 50.9→85.8 |  |
| DtV:d2:K2 → DtV:d2:K4 | 18–20 | 57 (23) | 1.034 [0.960, 1.119] | +16.9 | 0.96→1.35 | 53.8→88.9 |  |
| DtV:d2:K2 → DtV:d2:K4 | 21–31 (tail) | 9 (8) | 0.865 [0.851, 0.877] | +19.2 | 0.33→0.33 | 63.2→105.1 | sparse |
| DtV:d2:K2 → DtV:d2:K4 | 32 (no exit) | 1 (1) | 1.488 [nan, nan] | +16.9 | 2.00→4.00 | 66.0→99.0 | sparse |
| **DtV:d2:K2 → DtV:d2:K4** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d2:K4 → DtV:d2:K8 | ≤14 | 40 (16) | 0.795 [0.730, 0.848] | +60.9 | 1.90→2.55 | 76.2→142.2 |  |
| DtV:d2:K4 → DtV:d2:K8 | 15 | 94 (24) | 0.774 [0.725, 0.821] | +60.4 | 1.80→2.30 | 79.8→146.6 |  |
| DtV:d2:K4 → DtV:d2:K8 | 16 | 104 (23) | 0.783 [0.732, 0.839] | +59.2 | 1.67→2.13 | 82.9→151.5 |  |
| DtV:d2:K4 → DtV:d2:K8 | 17 | 79 (24) | 0.720 [0.685, 0.764] | +61.3 | 1.46→1.67 | 85.8→157.5 |  |
| DtV:d2:K4 → DtV:d2:K8 | 18–20 | 57 (23) | 0.696 [0.670, 0.741] | +61.0 | 1.35→1.44 | 88.9→161.3 |  |
| DtV:d2:K4 → DtV:d2:K8 | 21–31 (tail) | 9 (8) | 0.693 [0.679, 0.706] | +62.9 | 0.33→0.33 | 105.1→188.6 | sparse |
| DtV:d2:K4 → DtV:d2:K8 | 32 (no exit) | 1 (1) | 0.736 [nan, nan] | +56.6 | 4.00→4.00 | 99.0→161.0 | sparse |
| **DtV:d2:K4 → DtV:d2:K8** | **no crossover — lower option dominates in every bin** | | | | | | |
| DtV:d4:K2 → DtV:d4:K4 | ≤14 | 40 (16) | 1.175 [1.124, 1.222] | +31.3 | 1.80→3.35 | 43.8→75.2 |  |
| DtV:d4:K2 → DtV:d4:K4 | 15 | 94 (24) | 1.188 [1.160, 1.212] | +30.1 | 1.70→3.17 | 46.3→78.6 |  |
| DtV:d4:K2 → DtV:d4:K4 | 16 | 104 (23) | 1.228 [1.201, 1.253] | +27.9 | 1.69→3.19 | 48.3→81.2 |  |
| DtV:d4:K2 → DtV:d4:K4 | 17 | 79 (24) | 1.241 [1.206, 1.275] | +27.7 | 1.78→3.35 | 50.1→83.3 |  |
| DtV:d4:K2 → DtV:d4:K4 | 18–20 | 57 (23) | 1.229 [1.177, 1.275] | +27.1 | 1.58→2.93 | 53.1→87.6 |  |
| DtV:d4:K2 → DtV:d4:K4 | 21–31 (tail) | 9 (8) | 1.224 [1.137, 1.309] | +28.0 | 1.67→3.00 | 58.2→89.9 | sparse |
| DtV:d4:K2 → DtV:d4:K4 | 32 (no exit) | 1 (1) | 1.403 [nan, nan] | +27.6 | 2.00→4.00 | 66.0→99.0 | sparse |
| **DtV:d4:K2 → DtV:d4:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d4:K4 → DtV:d4:K8 | ≤14 | 40 (16) | 0.981 [0.908, 1.039] | +78.5 | 3.35→5.88 | 75.2→139.7 |  |
| DtV:d4:K4 → DtV:d4:K8 | 15 | 94 (24) | 1.008 [0.960, 1.053] | +75.9 | 3.17→5.64 | 78.6→143.0 |  |
| DtV:d4:K4 → DtV:d4:K8 | 16 | 104 (23) | 1.039 [1.003, 1.072] | +76.8 | 3.19→5.88 | 81.2→146.3 |  |
| DtV:d4:K4 → DtV:d4:K8 | 17 | 79 (24) | 1.052 [1.010, 1.095] | +75.5 | 3.35→6.16 | 83.3→149.0 |  |
| DtV:d4:K4 → DtV:d4:K8 | 18–20 | 57 (23) | 0.993 [0.921, 1.060] | +76.5 | 2.93→5.04 | 87.6→154.3 |  |
| DtV:d4:K4 → DtV:d4:K8 | 21–31 (tail) | 9 (8) | 1.043 [0.876, 1.173] | +74.6 | 3.00→5.22 | 89.9→150.7 | sparse |
| DtV:d4:K4 → DtV:d4:K8 | 32 (no exit) | 1 (1) | 1.262 [nan, nan] | +74.4 | 4.00→8.00 | 99.0→161.0 | sparse |
| **DtV:d4:K4 → DtV:d4:K8** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| DtV:d8:K2 → DtV:d8:K4 | ≤14 | 40 (16) | 1.091 [1.047, 1.124] | +52.6 | 1.82→3.58 | 43.6→74.9 |  |
| DtV:d8:K2 → DtV:d8:K4 | 15 | 94 (24) | 1.100 [1.068, 1.127] | +51.1 | 1.89→3.64 | 45.9→77.7 |  |
| DtV:d8:K2 → DtV:d8:K4 | 16 | 104 (23) | 1.146 [1.124, 1.165] | +49.2 | 1.89→3.72 | 48.2→80.8 |  |
| DtV:d8:K2 → DtV:d8:K4 | 17 | 79 (24) | 1.150 [1.127, 1.169] | +48.6 | 1.94→3.77 | 49.9→82.5 |  |
| DtV:d8:K2 → DtV:d8:K4 | 18–20 | 57 (23) | 1.140 [1.095, 1.175] | +49.1 | 1.84→3.53 | 53.2→87.6 |  |
| DtV:d8:K2 → DtV:d8:K4 | 21–31 (tail) | 9 (8) | 1.185 [1.133, 1.219] | +48.8 | 1.89→3.67 | 57.4→90.0 | sparse |
| DtV:d8:K2 → DtV:d8:K4 | 32 (no exit) | 1 (1) | 1.252 [nan, nan] | +52.7 | 2.00→4.00 | 66.0→99.0 | sparse |
| **DtV:d8:K2 → DtV:d8:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| DtV:d8:K4 → DtV:d8:K8 | ≤14 | 40 (16) | 1.021 [0.989, 1.045] | +113.8 | 3.58→6.97 | 74.9→139.2 |  |
| DtV:d8:K4 → DtV:d8:K8 | 15 | 94 (24) | 1.000 [0.970, 1.027] | +111.9 | 3.64→6.83 | 77.7→141.9 |  |
| DtV:d8:K4 → DtV:d8:K8 | 16 | 104 (23) | 1.039 [1.018, 1.056] | +112.7 | 3.72→7.25 | 80.8→145.9 |  |
| DtV:d8:K4 → DtV:d8:K8 | 17 | 79 (24) | 1.023 [0.983, 1.056] | +110.4 | 3.77→7.13 | 82.5→147.8 |  |
| DtV:d8:K4 → DtV:d8:K8 | 18–20 | 57 (23) | 1.000 [0.960, 1.040] | +111.8 | 3.53→6.46 | 87.6→154.3 |  |
| DtV:d8:K4 → DtV:d8:K8 | 21–31 (tail) | 9 (8) | 1.090 [1.034, 1.134] | +113.1 | 3.67→7.22 | 90.0→152.9 | sparse |
| DtV:d8:K4 → DtV:d8:K8 | 32 (no exit) | 1 (1) | 1.197 [nan, nan] | +106.7 | 4.00→8.00 | 99.0→161.0 | sparse |
| **DtV:d8:K4 → DtV:d8:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### SD vs AR

| lo → hi | D bin | n (prompts) | U_hi/U_lo [95% CI] | Δlat ms | acc lo→hi | rows lo→hi | |
|---|---|---:|---:|---:|---|---|---|
| AR → DtV:d2:K2 | ≤14 | 40 (16) | 0.939 [0.796, 1.079] | +52.6 | -→1.20 | 13.7→44.1 |  |
| AR → DtV:d2:K2 | 15 | 94 (24) | 0.951 [0.886, 1.015] | +53.5 | -→1.15 | 15.0→46.9 |  |
| AR → DtV:d2:K2 | 16 | 104 (23) | 0.970 [0.903, 1.050] | +54.5 | -→1.13 | 16.0→48.9 |  |
| AR → DtV:d2:K2 | 17 | 79 (24) | 0.973 [0.876, 1.081] | +54.3 | -→1.09 | 17.0→50.9 |  |
| AR → DtV:d2:K2 | 18–20 | 57 (23) | 0.936 [0.853, 1.025] | +56.2 | -→0.96 | 18.5→53.8 |  |
| AR → DtV:d2:K2 | 21–31 (tail) | 9 (8) | 0.679 [0.521, 0.836] | +60.3 | -→0.33 | 23.2→63.2 | sparse |
| AR → DtV:d2:K2 | 32 (no exit) | 1 (1) | 1.838 [nan, nan] | +54.6 | -→2.00 | 32.0→66.0 | sparse |
| **AR → DtV:d2:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d2:K4 | ≤14 | 40 (16) | 1.006 [0.833, 1.190] | +73.6 | -→1.90 | 13.7→76.2 |  |
| AR → DtV:d2:K4 | 15 | 94 (24) | 1.029 [0.913, 1.142] | +73.0 | -→1.80 | 15.0→79.8 |  |
| AR → DtV:d2:K4 | 16 | 104 (23) | 1.021 [0.950, 1.108] | +73.5 | -→1.67 | 16.0→82.9 |  |
| AR → DtV:d2:K4 | 17 | 79 (24) | 0.967 [0.835, 1.120] | +72.9 | -→1.46 | 17.0→85.8 |  |
| AR → DtV:d2:K4 | 18–20 | 57 (23) | 0.968 [0.846, 1.112] | +73.0 | -→1.35 | 18.5→88.9 |  |
| AR → DtV:d2:K4 | 21–31 (tail) | 9 (8) | 0.587 [0.456, 0.720] | +79.6 | -→0.33 | 23.2→105.1 | sparse |
| AR → DtV:d2:K4 | 32 (no exit) | 1 (1) | 2.735 [nan, nan] | +71.5 | -→4.00 | 32.0→99.0 | sparse |
| **AR → DtV:d2:K4** | **noisy crossover (point estimates cross 1, CIs do not separate)** | | | | | | |
| AR → DtV:d2:K8 | ≤14 | 40 (16) | 0.800 [0.644, 0.970] | +134.5 | -→2.55 | 13.7→142.2 |  |
| AR → DtV:d2:K8 | 15 | 94 (24) | 0.796 [0.678, 0.904] | +133.4 | -→2.30 | 15.0→146.6 |  |
| AR → DtV:d2:K8 | 16 | 104 (23) | 0.799 [0.732, 0.876] | +132.7 | -→2.13 | 16.0→151.5 |  |
| AR → DtV:d2:K8 | 17 | 79 (24) | 0.697 [0.593, 0.815] | +134.3 | -→1.67 | 17.0→157.5 |  |
| AR → DtV:d2:K8 | 18–20 | 57 (23) | 0.673 [0.577, 0.805] | +134.0 | -→1.44 | 18.5→161.3 |  |
| AR → DtV:d2:K8 | 21–31 (tail) | 9 (8) | 0.407 [0.312, 0.498] | +142.5 | -→0.33 | 23.2→188.6 | sparse |
| AR → DtV:d2:K8 | 32 (no exit) | 1 (1) | 2.013 [nan, nan] | +128.1 | -→4.00 | 32.0→161.0 | sparse |
| **AR → DtV:d2:K8** | **no crossover — lower option dominates in every bin** | | | | | | |
| AR → DtV:d4:K2 | ≤14 | 40 (16) | 1.128 [1.053, 1.192] | +58.0 | -→1.80 | 13.7→43.8 |  |
| AR → DtV:d4:K2 | 15 | 94 (24) | 1.139 [1.069, 1.206] | +58.3 | -→1.70 | 15.0→46.3 |  |
| AR → DtV:d4:K2 | 16 | 104 (23) | 1.171 [1.120, 1.223] | +59.0 | -→1.69 | 16.0→48.3 |  |
| AR → DtV:d4:K2 | 17 | 79 (24) | 1.241 [1.192, 1.286] | +59.0 | -→1.78 | 17.0→50.1 |  |
| AR → DtV:d4:K2 | 18–20 | 57 (23) | 1.166 [1.074, 1.248] | +61.9 | -→1.58 | 18.5→53.1 |  |
| AR → DtV:d4:K2 | 21–31 (tail) | 9 (8) | 1.348 [1.078, 1.617] | +61.2 | -→1.67 | 23.2→58.2 | sparse |
| AR → DtV:d4:K2 | 32 (no exit) | 1 (1) | 1.763 [nan, nan] | +60.6 | -→2.00 | 32.0→66.0 | sparse |
| **AR → DtV:d4:K2** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d4:K4 | ≤14 | 40 (16) | 1.326 [1.217, 1.425] | +89.3 | -→3.35 | 13.7→75.2 |  |
| AR → DtV:d4:K4 | 15 | 94 (24) | 1.353 [1.247, 1.451] | +88.4 | -→3.17 | 15.0→78.6 |  |
| AR → DtV:d4:K4 | 16 | 104 (23) | 1.438 [1.364, 1.514] | +87.0 | -→3.19 | 16.0→81.2 |  |
| AR → DtV:d4:K4 | 17 | 79 (24) | 1.539 [1.462, 1.615] | +86.6 | -→3.35 | 17.0→83.3 |  |
| AR → DtV:d4:K4 | 18–20 | 57 (23) | 1.433 [1.289, 1.563] | +89.0 | -→2.93 | 18.5→87.6 |  |
| AR → DtV:d4:K4 | 21–31 (tail) | 9 (8) | 1.649 [1.270, 2.018] | +89.2 | -→3.00 | 23.2→89.9 | sparse |
| AR → DtV:d4:K4 | 32 (no exit) | 1 (1) | 2.473 [nan, nan] | +88.2 | -→4.00 | 32.0→99.0 | sparse |
| **AR → DtV:d4:K4** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d4:K8 | ≤14 | 40 (16) | 1.300 [1.148, 1.440] | +167.7 | -→5.88 | 13.7→139.7 |  |
| AR → DtV:d4:K8 | 15 | 94 (24) | 1.363 [1.227, 1.503] | +164.3 | -→5.64 | 15.0→143.0 |  |
| AR → DtV:d4:K8 | 16 | 104 (23) | 1.494 [1.396, 1.599] | +163.8 | -→5.88 | 16.0→146.3 |  |
| AR → DtV:d4:K8 | 17 | 79 (24) | 1.620 [1.511, 1.747] | +162.1 | -→6.16 | 17.0→149.0 |  |
| AR → DtV:d4:K8 | 18–20 | 57 (23) | 1.423 [1.218, 1.624] | +165.6 | -→5.04 | 18.5→154.3 |  |
| AR → DtV:d4:K8 | 21–31 (tail) | 9 (8) | 1.720 [1.220, 2.247] | +163.8 | -→5.22 | 23.2→150.7 | sparse |
| AR → DtV:d4:K8 | 32 (no exit) | 1 (1) | 3.121 [nan, nan] | +162.6 | -→8.00 | 32.0→161.0 | sparse |
| **AR → DtV:d4:K8** | **no crossover — higher option dominates in every bin** | | | | | | |
| AR → DtV:d8:K2 | ≤14 | 40 (16) | 1.019 [0.937, 1.078] | +69.4 | -→1.82 | 13.7→43.6 |  |
| AR → DtV:d8:K2 | 15 | 94 (24) | 1.101 [1.063, 1.133] | +69.2 | -→1.89 | 15.0→45.9 |  |
| AR → DtV:d8:K2 | 16 | 104 (23) | 1.134 [1.098, 1.165] | +70.5 | -→1.89 | 16.0→48.2 |  |
| AR → DtV:d8:K2 | 17 | 79 (24) | 1.181 [1.150, 1.206] | +70.4 | -→1.94 | 17.0→49.9 |  |
| AR → DtV:d8:K2 | 18–20 | 57 (23) | 1.173 [1.116, 1.219] | +72.8 | -→1.84 | 18.5→53.2 |  |
| AR → DtV:d8:K2 | 21–31 (tail) | 9 (8) | 1.343 [1.183, 1.507] | +72.0 | -→1.89 | 23.2→57.4 | sparse |
| AR → DtV:d8:K2 | 32 (no exit) | 1 (1) | 1.626 [nan, nan] | +73.0 | -→2.00 | 32.0→66.0 | sparse |
| **AR → DtV:d8:K2** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d8:K4 | ≤14 | 40 (16) | 1.111 [0.987, 1.204] | +121.9 | -→3.58 | 13.7→74.9 |  |
| AR → DtV:d8:K4 | 15 | 94 (24) | 1.211 [1.145, 1.263] | +120.2 | -→3.64 | 15.0→77.7 |  |
| AR → DtV:d8:K4 | 16 | 104 (23) | 1.299 [1.238, 1.354] | +119.7 | -→3.72 | 16.0→80.8 |  |
| AR → DtV:d8:K4 | 17 | 79 (24) | 1.359 [1.311, 1.399] | +119.0 | -→3.77 | 17.0→82.5 |  |
| AR → DtV:d8:K4 | 18–20 | 57 (23) | 1.337 [1.225, 1.426] | +121.9 | -→3.53 | 18.5→87.6 |  |
| AR → DtV:d8:K4 | 21–31 (tail) | 9 (8) | 1.592 [1.339, 1.836] | +120.8 | -→3.67 | 23.2→90.0 | sparse |
| AR → DtV:d8:K4 | 32 (no exit) | 1 (1) | 2.036 [nan, nan] | +125.7 | -→4.00 | 32.0→99.0 | sparse |
| **AR → DtV:d8:K4** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |
| AR → DtV:d8:K8 | ≤14 | 40 (16) | 1.135 [0.982, 1.253] | +235.7 | -→6.97 | 13.7→139.2 |  |
| AR → DtV:d8:K8 | 15 | 94 (24) | 1.211 [1.118, 1.286] | +232.1 | -→6.83 | 15.0→141.9 |  |
| AR → DtV:d8:K8 | 16 | 104 (23) | 1.349 [1.272, 1.415] | +232.3 | -→7.25 | 16.0→145.9 |  |
| AR → DtV:d8:K8 | 17 | 79 (24) | 1.391 [1.307, 1.464] | +229.4 | -→7.13 | 17.0→147.8 |  |
| AR → DtV:d8:K8 | 18–20 | 57 (23) | 1.338 [1.198, 1.457] | +233.7 | -→6.46 | 18.5→154.3 |  |
| AR → DtV:d8:K8 | 21–31 (tail) | 9 (8) | 1.735 [1.392, 2.050] | +233.9 | -→7.22 | 23.2→152.9 | sparse |
| AR → DtV:d8:K8 | 32 (no exit) | 1 (1) | 2.438 [nan, nan] | +232.4 | -→8.00 | 32.0→161.0 | sparse |
| **AR → DtV:d8:K8** | **no crossover (same sign of point estimate, some CIs include 1)** | | | | | | |

### confounding controls for DtV:d4:K8 → DtV:d8:K8 (post-hoc)

| stratum | n | U_hi/U_lo [95% CI] |
|---|---:|---:|
| accepted(lo)=0 | 42 | 4.650 [3.829, 5.416] |
| accepted(lo)=1 | 28 | 2.188 [1.737, 2.637] |
| accepted(lo)=2 | 21 | 1.768 [1.383, 2.112] |
| accepted(lo)=3 | 15 | 1.170 [0.894, 1.481] |
| accepted(lo)=4 | 19 | 0.874 [0.665, 1.088] |
| accepted(lo)=5 | 11 | 1.054 [0.940, 1.159] |
| accepted(lo)=6 | 7 | 0.998 [0.965, 1.075] |
| accepted(lo)=7 | 10 | 0.848 [0.821, 0.870] |
| accepted(lo)=8 | 231 | 0.742 [0.734, 0.749] |
| k<84 | 120 | 1.134 [1.036, 1.244] |
| 84≤k<164 | 120 | 0.898 [0.845, 0.966] |
| k≥164 | 144 | 0.787 [0.769, 0.807] |

*Not done (needs GPU, diagnostic only):* forced counterfactual verification budget / matched replay depth at a fixed committed prefix. Any forced-depth arm has a different target; not run.
