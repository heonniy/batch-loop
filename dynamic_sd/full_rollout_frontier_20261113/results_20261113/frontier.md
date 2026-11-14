# Full-rollout frontier — derived tables

SOURCE MEASURED rollouts → DERIVED tables. Speedup = Σtok/Σwall of arm ÷ same-cell AR (same target), paired prompt bootstrap 95% CI. `plateau` = best vs runner-up gap ≤2% or CI ∋ 1.

## RUN_STATUS (validator)

| cell | status | n prompts | notes |
|---|---|---:|---|
| parity/huginn_eps0.1_gsm8k3 | passed | 16 |  parity inexact 0/64 |
| parity/huginn_eps0.1_math500 | passed | 8 |  parity inexact 0/32 |
| parity/huginn_eps0.1_spec | passed | 8 |  parity inexact 0/32 |
| parity/huginn_eps0.2_gsm8k3 | passed | 16 |  parity inexact 0/64 |
| parity/huginn_eps0.2_math500 | passed | 8 |  parity inexact 0/32 |
| parity/huginn_eps0.2_spec | passed | 8 |  parity inexact 0/32 |
| parity/huginn_fixed_gsm8k3 | passed | 16 |  parity inexact 0/64 |
| parity/huginn_fixed_math500 | passed | 8 |  parity inexact 0/32 |
| parity/huginn_fixed_spec | passed | 8 |  parity inexact 0/32 |
| parity/ouro_fixed_gsm8k3 | passed | 16 |  parity inexact 0/64 |
| parity/ouro_fixed_math500 | passed | 8 |  parity inexact 0/32 |
| parity/ouro_fixed_spec | passed | 8 |  parity inexact 0/32 |
| parity/ouro_tau0.1_gsm8k3 | passed | 16 |  parity inexact 0/64 |
| parity/ouro_tau0.1_math500 | passed | 8 |  parity inexact 0/32 |
| parity/ouro_tau0.1_spec | passed | 8 |  parity inexact 0/32 |
| parity/ouro_tau0.4_gsm8k3 | passed | 16 |  parity inexact 0/64 |
| parity/ouro_tau0.4_math500 | passed | 8 |  parity inexact 0/32 |
| parity/ouro_tau0.4_spec | passed | 8 |  parity inexact 0/32 |
| rep/huginn_eps0.1_gsm8k3 | passed | 16 |  |
| rep/ouro_tau0.4_gsm8k3 | passed | 16 |  |
| s1/huginn_eps0.1_gsm8k3 | passed | 300 |  |
| s1/huginn_eps0.1_gsm8k8 | passed | 50 |  |
| s1/huginn_eps0.1_math500 | passed | 50 |  |
| s1/huginn_eps0.1_spec | passed | 48 |  |
| s1/huginn_eps0.2_gsm8k3 | passed | 300 |  |
| s1/huginn_eps0.2_math500 | passed | 50 |  |
| s1/huginn_fixed_gsm8k3 | passed | 300 |  |
| s1/huginn_fixed_gsm8k8 | passed | 50 |  |
| s1/huginn_fixed_math500 | passed | 50 |  |
| s1/huginn_fixed_spec | passed | 48 |  |
| s1/ouro_fixed_gsm8k3 | passed | 300 |  |
| s1/ouro_fixed_gsm8k8 | passed | 50 |  |
| s1/ouro_fixed_math500 | passed | 50 |  |
| s1/ouro_fixed_spec | passed | 48 |  |
| s1/ouro_tau0.1_gsm8k3 | passed | 300 |  |
| s1/ouro_tau0.1_math500 | passed | 50 |  |
| s1/ouro_tau0.4_gsm8k3 | passed | 300 |  |
| s1/ouro_tau0.4_gsm8k8 | passed | 50 |  |
| s1/ouro_tau0.4_math500 | passed | 50 |  |
| s1/ouro_tau0.4_spec | passed | 48 |  |

## parity/huginn_eps0.1_gsm8k3 (n=16, passed)

AR: 19.6 tok/s, acc 0.25, mean len 128, mean exit depth 16.78926637124569, prefill share 12.7%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.041 [0.996, 1.085] | 1.048 [0.995, 1.097] | 20.4 | 0.737 | 0.250 | 1.00 | None |
| DtV:d4:K4 | 1.111 [1.038, 1.183] | 1.129 [1.044, 1.213] | 21.7 | 0.599 | 0.250 | 1.00 | None |
| DtV:d4:K8 | 0.947 [0.847, 1.050] | 0.940 [0.825, 1.058] | 18.5 | 0.427 | 0.250 | 1.00 | None |
| WFD:d4 | 1.242 [1.168, 1.314] | 1.287 [1.205, 1.371] | 24.3 | 0.818 | 0.250 | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.067 [1.034, 1.097] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.12; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<16.4) | 5 | DtV:d4:K4 | 1.136 | 1.136 | 1.284 |
| mid | 5 | DtV:d4:K4 | 1.058 | 1.058 | 1.190 |
| high (>=16.6) | 6 | DtV:d4:K4 | 1.113 | 1.113 | 1.231 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.542) | 5 | DtV:d4:K4 | 1.000 | 1.000 | 1.134 |
| mid | 5 | DtV:d4:K4 | 1.065 | 1.065 | 1.219 |
| high (>=0.614) | 6 | DtV:d4:K4 | 1.267 | 1.267 | 1.362 |

## parity/huginn_eps0.1_math500 (n=8, passed)

AR: 18.4 tok/s, acc 0.125, mean len 162, mean exit depth 15.108359133126935, prefill share 21.8%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.058 [1.007, 1.106] | 1.076 [1.011, 1.134] | 19.5 | 0.761 | 0.125 | 1.00 | None |
| DtV:d4:K4 | 1.136 [1.050, 1.229] | 1.181 [1.071, 1.298] | 21.0 | 0.643 | 0.125 | 1.00 | None |
| DtV:d4:K8 | 1.000 [0.886, 1.142] | 1.000 [0.846, 1.182] | 18.4 | 0.473 | 0.125 | 1.00 | None |
| WFD:d4 | 1.166 [1.089, 1.237] | 1.223 [1.130, 1.311] | 21.5 | 0.848 | 0.125 | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.074 [1.042, 1.111] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.25; AR beats every SD arm: 0.12.

## parity/huginn_eps0.1_spec (n=8, passed)

AR: 24.7 tok/s, acc None, mean len 236, mean exit depth 14.470682302771856, prefill share 4.5%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.078 [0.978, 1.167] | 1.082 [0.977, 1.176] | 26.6 | 0.793 | - | 1.00 | None |
| DtV:d4:K4 | 1.187 [0.984, 1.404] | 1.198 [0.983, 1.426] | 29.3 | 0.673 | - | 1.00 | None |
| DtV:d4:K8 | 1.019 [0.754, 1.373] | 1.020 [0.745, 1.396] | 25.2 | 0.492 | - | 1.00 | None |
| WFD:d4 | 1.350 [1.135, 1.595] | 1.373 [1.143, 1.628] | 33.3 | 0.865 | - | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.101 [1.005, 1.204] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.25; AR beats every SD arm: 0.12.

## parity/huginn_eps0.2_gsm8k3 (n=16, passed)

AR: 24.4 tok/s, acc 0.0, mean len 108, mean exit depth 12.094483568075118, prefill share 18.8%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 0.984 [0.957, 1.013] | 0.981 [0.947, 1.015] | 24.0 | 0.729 | 0.000 | 1.00 | None |
| DtV:d4:K4 | 0.996 [0.940, 1.049] | 0.995 [0.926, 1.059] | 24.3 | 0.588 | 0.000 | 1.00 | None |
| DtV:d4:K8 | 0.829 [0.756, 0.917] | 0.798 [0.711, 0.902] | 20.2 | 0.417 | 0.000 | 1.00 | None |
| WFD:d4 | 1.088 [1.058, 1.116] | 1.110 [1.074, 1.143] | 26.5 | 0.816 | 0.000 | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.012 [0.978, 1.039] → **plateau**; tied set ['DtV:d4:K2', 'DtV:d4:K4']. Prompts where AR beats K_opt: 0.62; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<11.9) | 5 | DtV:d4:K4 | 1.077 | 1.077 | 1.122 |
| mid | 5 | DtV:d4:K2 | 0.968 | 0.947 | 1.068 |
| high (>=12.1) | 6 | DtV:d4:K4 | 0.974 | 0.974 | 1.076 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.545) | 5 | DtV:d4:K2 | 0.942 | 0.908 | 1.047 |
| mid | 5 | DtV:d4:K2 | 0.977 | 0.968 | 1.069 |
| high (>=0.592) | 6 | DtV:d4:K4 | 1.089 | 1.089 | 1.130 |

## parity/huginn_eps0.2_math500 (n=8, passed)

AR: 22.1 tok/s, acc 0.125, mean len 156, mean exit depth 11.010483870967741, prefill share 27.0%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.008 [0.965, 1.054] | 1.011 [0.946, 1.072] | 22.3 | 0.764 | 0.125 | 1.00 | None |
| DtV:d4:K4 | 1.030 [0.968, 1.107] | 1.041 [0.951, 1.144] | 22.8 | 0.640 | 0.125 | 1.00 | None |
| DtV:d4:K8 | 0.880 [0.801, 0.995] | 0.842 [0.732, 0.993] | 19.4 | 0.461 | 0.125 | 1.00 | None |
| WFD:d4 | 1.046 [1.009, 1.087] | 1.064 [1.013, 1.118] | 23.1 | 0.841 | 0.125 | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.021 [1.000, 1.051] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.50; AR beats every SD arm: 0.25.

## parity/huginn_eps0.2_spec (n=8, passed)

AR: 32.1 tok/s, acc None, mean len 239, mean exit depth 10.617121848739496, prefill share 5.7%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.064 [0.985, 1.124] | 1.068 [0.984, 1.134] | 34.1 | 0.863 | - | 1.00 | None |
| DtV:d4:K4 | 1.114 [0.964, 1.260] | 1.122 [0.962, 1.280] | 35.7 | 0.724 | - | 1.00 | None |
| DtV:d4:K8 | 0.980 [0.769, 1.231] | 0.979 [0.758, 1.247] | 31.4 | 0.570 | - | 1.00 | None |
| WFD:d4 | 1.156 [1.065, 1.241] | 1.167 [1.069, 1.260] | 37.1 | 0.890 | - | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.048 [0.969, 1.126] → **plateau**; tied set ['DtV:d4:K2', 'DtV:d4:K4']. Prompts where AR beats K_opt: 0.38; AR beats every SD arm: 0.25.

## parity/huginn_fixed_gsm8k3 (n=16, passed)

AR: 12.1 tok/s, acc 0.125, mean len 102, mean exit depth 32, prefill share 9.8%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.065 [1.031, 1.093] | 1.073 [1.035, 1.104] | 12.9 | 0.688 | 0.125 | 1.00 | None |
| DtV:d4:K4 | 1.262 [1.202, 1.324] | 1.299 [1.231, 1.376] | 15.3 | 0.549 | 0.125 | 1.00 | None |
| DtV:d4:K8 | 1.106 [1.025, 1.191] | 1.119 [1.028, 1.217] | 13.4 | 0.375 | 0.125 | 1.00 | None |
| WFD:d4 | 1.231 [1.171, 1.290] | 1.263 [1.195, 1.331] | 14.9 | 0.786 | 0.125 | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K8, gap 1.141 [1.106, 1.179] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.538) | 5 | DtV:d4:K4 | 1.138 | 1.138 | 1.119 |
| mid | 5 | DtV:d4:K4 | 1.292 | 1.292 | 1.259 |
| high (>=0.583) | 6 | DtV:d4:K4 | 1.378 | 1.378 | 1.331 |

## parity/huginn_fixed_math500 (n=8, passed)

AR: 10.8 tok/s, acc 0.125, mean len 157, mean exit depth 32, prefill share 13.1%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.180 [1.094, 1.241] | 1.212 [1.119, 1.276] | 12.8 | 0.785 | 0.125 | 1.00 | None |
| DtV:d4:K4 | 1.431 [1.252, 1.582] | 1.530 [1.331, 1.693] | 15.5 | 0.648 | 0.125 | 1.00 | None |
| DtV:d4:K8 | 1.336 [1.096, 1.584] | 1.407 [1.121, 1.693] | 14.5 | 0.483 | 0.125 | 1.00 | None |
| WFD:d4 | 1.283 [1.087, 1.472] | 1.340 [1.108, 1.556] | 13.9 | 0.851 | 0.125 | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K8, gap 1.071 [1.000, 1.146] → distinct optimum; tied set ['DtV:d4:K4', 'DtV:d4:K8']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

## parity/huginn_fixed_spec (n=8, passed)

AR: 13.5 tok/s, acc None, mean len 256, mean exit depth 32, prefill share 2.2%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K2 | 1.121 [1.042, 1.213] | 1.124 [1.043, 1.218] | 15.1 | 0.757 | - | 1.00 | None |
| DtV:d4:K4 | 1.412 [1.243, 1.627] | 1.425 [1.250, 1.649] | 19.0 | 0.629 | - | 1.00 | None |
| DtV:d4:K8 | 1.253 [1.017, 1.619] | 1.260 [1.017, 1.639] | 16.9 | 0.442 | - | 1.00 | None |
| WFD:d4 | 1.437 [1.197, 1.815] | 1.452 [1.202, 1.842] | 19.3 | 0.833 | - | 1.00 | None |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K8, gap 1.127 [1.004, 1.222] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

## parity/ouro_fixed_gsm8k3 (n=16, passed)

AR: 24.3 tok/s, acc 0.375, mean len 118, mean exit depth 4, prefill share 2.2%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 1.409 [1.361, 1.455] | 1.422 [1.371, 1.472] | 34.3 | 0.856 | 0.375 | 1.00 | None |
| DtV:d1:K4 | 1.605 [1.521, 1.694] | 1.627 [1.539, 1.723] | 39.0 | 0.764 | 0.375 | 1.00 | None |
| DtV:d1:K8 | 1.574 [1.447, 1.714] | 1.595 [1.462, 1.742] | 38.3 | 0.620 | 0.375 | 1.00 | None |
| WFD:d1 | 1.727 [1.646, 1.804] | 1.756 [1.671, 1.839] | 42.0 | 0.908 | 0.375 | 1.00 | None |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K8, gap 1.020 [0.977, 1.060] → **plateau**; tied set ['DtV:d1:K4', 'DtV:d1:K8']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.681) | 5 | DtV:d1:K4 | 1.426 | 1.426 | 1.595 |
| mid | 5 | DtV:d1:K4 | 1.589 | 1.589 | 1.667 |
| high (>=0.846) | 6 | DtV:d1:K8 | 1.892 | 1.798 | 1.900 |

## parity/ouro_fixed_math500 (n=8, passed)

AR: 25.1 tok/s, acc 0.0, mean len 250, mean exit depth 4, prefill share 2.2%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 1.476 [1.442, 1.506] | 1.491 [1.457, 1.522] | 37.0 | 0.900 | 0.000 | 1.00 | None |
| DtV:d1:K4 | 1.693 [1.635, 1.765] | 1.719 [1.658, 1.796] | 42.5 | 0.819 | 0.000 | 1.00 | None |
| DtV:d1:K8 | 1.748 [1.660, 1.870] | 1.778 [1.685, 1.906] | 43.9 | 0.712 | 0.000 | 1.00 | None |
| WFD:d1 | 1.689 [1.601, 1.781] | 1.715 [1.624, 1.810] | 42.4 | 0.931 | 0.000 | 1.00 | None |

K_opt **DtV:d1:K8**, runner-up DtV:d1:K4, gap 1.033 [0.994, 1.071] → **plateau**; tied set ['DtV:d1:K4', 'DtV:d1:K8']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

## parity/ouro_fixed_spec (n=8, passed)

AR: 27.8 tok/s, acc None, mean len 234, mean exit depth 4, prefill share 0.7%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 1.224 [1.131, 1.351] | 1.226 [1.132, 1.354] | 34.0 | 0.694 | - | 1.00 | None |
| DtV:d1:K4 | 1.237 [1.046, 1.531] | 1.239 [1.047, 1.535] | 34.3 | 0.536 | - | 1.00 | None |
| DtV:d1:K8 | 1.038 [0.810, 1.493] | 1.039 [0.809, 1.500] | 28.8 | 0.362 | - | 1.00 | None |
| WFD:d1 | 1.419 [1.241, 1.679] | 1.423 [1.243, 1.688] | 39.4 | 0.789 | - | 1.00 | None |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.011 [0.914, 1.163] → **plateau**; tied set ['DtV:d1:K2', 'DtV:d1:K4']. Prompts where AR beats K_opt: 0.25; AR beats every SD arm: 0.00.

## parity/ouro_tau0.1_gsm8k3 (n=16, passed)

AR: 49.0 tok/s, acc 0.375, mean len 118, mean exit depth 1.6823027718550108, prefill share 4.5%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 0.956 [0.885, 1.024] | 0.954 [0.880, 1.025] | 46.8 | 0.851 | 0.375 | 1.00 | None |
| DtV:d1:K4 | 0.974 [0.931, 1.025] | 0.973 [0.928, 1.027] | 47.7 | 0.789 | 0.375 | 1.00 | None |
| DtV:d1:K8 | 0.794 [0.731, 0.863] | 0.787 [0.722, 0.857] | 38.9 | 0.624 | 0.375 | 1.00 | None |
| WFD:d1 | 1.106 [1.064, 1.160] | 1.112 [1.067, 1.169] | 54.2 | 0.799 | 0.375 | 1.00 | None |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.019 [0.972, 1.080] → **plateau**; tied set ['DtV:d1:K2', 'DtV:d1:K4']. Prompts where AR beats K_opt: 0.69; AR beats every SD arm: 0.06.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<1.62) | 5 | DtV:d1:K4 | 1.016 | 1.016 | 1.135 |
| mid | 5 | DtV:d1:K4 | 0.928 | 0.928 | 1.077 |
| high (>=1.77) | 6 | DtV:d1:K2 | 0.981 | 0.974 | 1.103 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.75) | 5 | DtV:d1:K2 | 0.956 | 0.898 | 1.054 |
| mid | 5 | DtV:d1:K4 | 1.001 | 1.001 | 1.128 |
| high (>=0.804) | 6 | DtV:d1:K4 | 1.014 | 1.014 | 1.129 |

## parity/ouro_tau0.1_math500 (n=8, passed)

AR: 53.7 tok/s, acc 0.0, mean len 256, mean exit depth 1.6660127513487004, prefill share 4.5%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 0.990 [0.954, 1.034] | 0.989 [0.952, 1.035] | 53.1 | 0.866 | 0.000 | 1.00 | None |
| DtV:d1:K4 | 0.879 [0.786, 0.987] | 0.874 [0.778, 0.986] | 47.2 | 0.788 | 0.000 | 1.00 | None |
| DtV:d1:K8 | 0.855 [0.809, 0.909] | 0.849 [0.801, 0.905] | 45.9 | 0.668 | 0.000 | 1.00 | None |
| WFD:d1 | 1.002 [0.934, 1.068] | 1.002 [0.931, 1.071] | 53.8 | 0.769 | 0.000 | 1.00 | None |

K_opt **DtV:d1:K2**, runner-up DtV:d1:K4, gap 1.126 [1.034, 1.222] → distinct optimum; tied set ['DtV:d1:K2']. Prompts where AR beats K_opt: 0.62; AR beats every SD arm: 0.25.

## parity/ouro_tau0.1_spec (n=8, passed)

AR: 46.5 tok/s, acc None, mean len 236, mean exit depth 1.9201277955271565, prefill share 1.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 0.917 [0.871, 0.962] | 0.915 [0.869, 0.961] | 42.6 | 0.709 | - | 1.00 | None |
| DtV:d1:K4 | 0.791 [0.727, 0.866] | 0.789 [0.725, 0.864] | 36.8 | 0.571 | - | 1.00 | None |
| DtV:d1:K8 | 0.612 [0.531, 0.733] | 0.608 [0.528, 0.728] | 28.4 | 0.393 | - | 1.00 | None |
| WFD:d1 | 0.996 [0.897, 1.103] | 0.995 [0.895, 1.103] | 46.3 | 0.592 | - | 1.00 | None |

K_opt **DtV:d1:K2**, runner-up DtV:d1:K4, gap 1.159 [1.097, 1.210] → distinct optimum; tied set ['DtV:d1:K2']. Prompts where AR beats K_opt: 1.00; AR beats every SD arm: 0.12.

## parity/ouro_tau0.4_gsm8k3 (n=16, passed)

AR: 36.6 tok/s, acc 0.3125, mean len 121, mean exit depth 2.5749870264660095, prefill share 3.2%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 1.161 [1.083, 1.236] | 1.168 [1.086, 1.246] | 42.5 | 0.844 | 0.312 | 1.00 | None |
| DtV:d1:K4 | 1.258 [1.207, 1.307] | 1.269 [1.217, 1.320] | 46.1 | 0.762 | 0.312 | 1.00 | None |
| DtV:d1:K8 | 1.144 [1.078, 1.207] | 1.149 [1.081, 1.216] | 41.9 | 0.608 | 0.312 | 1.00 | None |
| WFD:d1 | 1.314 [1.266, 1.375] | 1.328 [1.277, 1.392] | 48.1 | 0.900 | 0.312 | 1.00 | None |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.083 [1.028, 1.162] → distinct optimum; tied set ['DtV:d1:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<2.48) | 5 | DtV:d1:K4 | 1.313 | 1.313 | 1.330 |
| mid | 5 | DtV:d1:K4 | 1.279 | 1.279 | 1.369 |
| high (>=2.62) | 6 | DtV:d1:K4 | 1.209 | 1.209 | 1.270 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.724) | 5 | DtV:d1:K4 | 1.189 | 1.189 | 1.363 |
| mid | 5 | DtV:d1:K4 | 1.274 | 1.274 | 1.269 |
| high (>=0.85) | 6 | DtV:d1:K4 | 1.309 | 1.309 | 1.324 |

## parity/ouro_tau0.4_math500 (n=8, passed)

AR: 38.6 tok/s, acc 0.0, mean len 252, mean exit depth 2.4361914257228316, prefill share 3.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 1.220 [1.195, 1.245] | 1.229 [1.203, 1.255] | 47.1 | 0.897 | 0.000 | 1.00 | None |
| DtV:d1:K4 | 1.278 [1.238, 1.322] | 1.290 [1.249, 1.336] | 49.3 | 0.822 | 0.000 | 1.00 | None |
| DtV:d1:K8 | 1.204 [1.123, 1.290] | 1.212 [1.128, 1.302] | 46.5 | 0.705 | 0.000 | 1.00 | None |
| WFD:d1 | 1.234 [1.160, 1.300] | 1.244 [1.167, 1.312] | 47.6 | 0.926 | 0.000 | 1.00 | None |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.048 [1.024, 1.069] → distinct optimum; tied set ['DtV:d1:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

## parity/ouro_tau0.4_spec (n=8, passed)

AR: 39.2 tok/s, acc None, mean len 219, mean exit depth 2.6092943201376935, prefill share 1.1%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K2 | 1.056 [1.001, 1.129] | 1.057 [1.001, 1.131] | 41.4 | 0.742 | - | 1.00 | None |
| DtV:d1:K4 | 0.974 [0.843, 1.159] | 0.975 [0.841, 1.163] | 38.2 | 0.601 | - | 1.00 | None |
| DtV:d1:K8 | 0.819 [0.639, 1.105] | 0.817 [0.636, 1.106] | 32.1 | 0.424 | - | 1.00 | None |
| WFD:d1 | 1.149 [1.057, 1.260] | 1.151 [1.058, 1.265] | 45.0 | 0.832 | - | 1.00 | None |

K_opt **DtV:d1:K2**, runner-up DtV:d1:K4, gap 1.084 [0.931, 1.217] → **plateau**; tied set ['DtV:d1:K2', 'DtV:d1:K4', 'DtV:d1:K8']. Prompts where AR beats K_opt: 0.50; AR beats every SD arm: 0.00.

## rep/huginn_eps0.1_gsm8k3 (n=16, passed)

AR: 23.3 tok/s, acc 0.25, mean len 166, mean exit depth 16.545179584120984, prefill share 1.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.225 [1.148, 1.276] | 1.231 [1.158, 1.279] | 28.6 | 0.850 | 0.250 | 0.75 | 59.0 |
| DtV:d4:K2 | 1.348 [1.231, 1.433] | 1.357 [1.247, 1.441] | 31.4 | 0.774 | 0.250 | 0.81 | 69 |
| DtV:d4:K4 | 1.362 [1.168, 1.512] | 1.371 [1.180, 1.521] | 31.7 | 0.642 | 0.250 | 0.81 | 68 |
| DtV:d4:K8 | 1.279 [1.034, 1.471] | 1.283 [1.032, 1.480] | 29.8 | 0.516 | 0.250 | 0.88 | 42.0 |
| DtV:d4:K16 | 0.828 [0.598, 1.053] | 0.827 [0.596, 1.055] | 19.3 | 0.289 | 0.250 | 0.81 | 49 |
| WFD:d4 | 1.616 [1.394, 1.781] | 1.633 [1.418, 1.797] | 37.7 | 0.845 | 0.250 | 0.75 | 51.5 |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.010 [0.943, 1.062] → **plateau**; tied set ['DtV:d4:K2', 'DtV:d4:K4', 'DtV:d4:K8']. Prompts where AR beats K_opt: 0.06; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<16.6) | 5 | DtV:d4:K8 | 1.266 | 1.165 | 1.422 |
| mid | 5 | DtV:d4:K4 | 1.428 | 1.428 | 1.703 |
| high (>=17) | 6 | DtV:d4:K4 | 1.470 | 1.470 | 1.698 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.544) | 5 | DtV:d4:K8 | 1.188 | 1.037 | 1.277 |
| mid | 5 | DtV:d4:K2 | 1.333 | 1.294 | 1.561 |
| high (>=0.614) | 6 | DtV:d4:K4 | 1.621 | 1.621 | 1.863 |

median per-prompt wall CV over repeats: AR: 0.004, DtV:d4:K1: 0.003, DtV:d4:K2: 0.004, DtV:d4:K4: 0.003, DtV:d4:K8: 0.004, DtV:d4:K16: 0.002, WFD:d4: 0.003

## rep/ouro_tau0.4_gsm8k3 (n=16, passed)

AR: 31.7 tok/s, acc 0.375, mean len 119, mean exit depth 2.567711099309612, prefill share 1.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.217 [1.203, 1.228] | 1.220 [1.207, 1.232] | 38.6 | 0.893 | 0.312 | 0.88 | 16.0 |
| DtV:d1:K2 | 1.307 [1.274, 1.336] | 1.311 [1.279, 1.339] | 41.5 | 0.842 | 0.312 | 0.62 | 35.0 |
| DtV:d1:K4 | 1.354 [1.295, 1.402] | 1.358 [1.300, 1.406] | 43.0 | 0.769 | 0.312 | 0.62 | 38.0 |
| DtV:d1:K8 | 1.197 [1.120, 1.278] | 1.199 [1.121, 1.280] | 38.0 | 0.602 | 0.312 | 0.56 | 64 |
| DtV:d1:K16 | 0.918 [0.823, 1.013] | 0.917 [0.821, 1.013] | 29.1 | 0.420 | 0.312 | 0.69 | 64 |
| WFD:d1 | 1.575 [1.545, 1.600] | 1.585 [1.555, 1.609] | 50.0 | 0.903 | 0.375 | 0.69 | 47 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.036 [1.013, 1.055] → distinct optimum; tied set ['DtV:d1:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<2.48) | 5 | DtV:d1:K4 | 1.425 | 1.425 | 1.603 |
| mid | 5 | DtV:d1:K4 | 1.288 | 1.288 | 1.514 |
| high (>=2.62) | 6 | DtV:d1:K4 | 1.363 | 1.363 | 1.610 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.775) | 5 | DtV:d1:K4 | 1.230 | 1.230 | 1.530 |
| mid | 5 | DtV:d1:K4 | 1.406 | 1.406 | 1.588 |
| high (>=0.804) | 6 | DtV:d1:K4 | 1.424 | 1.424 | 1.607 |

median per-prompt wall CV over repeats: AR: 0.004, DtV:d1:K1: 0.005, DtV:d1:K2: 0.004, DtV:d1:K4: 0.003, DtV:d1:K8: 0.004, DtV:d1:K16: 0.003, WFD:d1: 0.004

## s1/huginn_eps0.1_gsm8k3 (n=300, passed)

AR: 22.0 tok/s, acc 0.2633333333333333, mean len 114, mean exit depth 16.964227642276423, prefill share 2.0%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.233 [1.217, 1.248] | 1.238 [1.223, 1.253] | 27.1 | 0.826 | 0.240 | 0.69 | 30 |
| DtV:d4:K2 | 1.351 [1.326, 1.375] | 1.359 [1.335, 1.383] | 29.7 | 0.752 | 0.240 | 0.72 | 30 |
| DtV:d4:K4 | 1.332 [1.295, 1.373] | 1.341 [1.303, 1.381] | 29.3 | 0.608 | 0.250 | 0.70 | 36.0 |
| DtV:d4:K8 | 1.143 [1.086, 1.199] | 1.145 [1.088, 1.202] | 25.1 | 0.442 | 0.247 | 0.72 | 28 |
| DtV:d4:K16 | 0.775 [0.727, 0.827] | 0.772 [0.723, 0.823] | 17.1 | 0.261 | 0.257 | 0.71 | 31.5 |
| WFD:d4 | 1.603 [1.559, 1.646] | 1.620 [1.578, 1.664] | 35.2 | 0.832 | 0.257 | 0.68 | 36 |

K_opt **DtV:d4:K2**, runner-up DtV:d4:K4, gap 1.014 [0.995, 1.034] → **plateau**; tied set ['DtV:d4:K2', 'DtV:d4:K4']. Prompts where AR beats K_opt: 0.01; AR beats every SD arm: 0.00.
Held-out K: pilot (n=100) picks DtV:d4:K2 → on confirmation (n=200) speedup 1.362 [1.331, 1.392]; eval-best DtV:d4:K2, regret ×1.000.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<16.7) | 100 | DtV:d4:K2 | 1.374 | 1.374 | 1.642 |
| mid | 100 | DtV:d4:K2 | 1.333 | 1.333 | 1.611 |
| high (>=17.4) | 100 | DtV:d4:K2 | 1.349 | 1.349 | 1.551 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.675) | 100 | DtV:d4:K2 | 1.218 | 1.218 | 1.422 |
| mid | 94 | DtV:d4:K2 | 1.293 | 1.293 | 1.530 |
| high (>=0.75) | 106 | DtV:d4:K4 | 1.512 | 1.462 | 1.780 |

## s1/huginn_eps0.1_gsm8k8 (n=50, passed)

AR: 22.0 tok/s, acc 0.36, mean len 122, mean exit depth 16.818814176879762, prefill share 3.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.206 [1.179, 1.230] | 1.214 [1.186, 1.238] | 26.5 | 0.794 | 0.380 | 0.62 | 27 |
| DtV:d4:K2 | 1.292 [1.243, 1.343] | 1.309 [1.260, 1.358] | 28.4 | 0.711 | 0.360 | 0.66 | 27 |
| DtV:d4:K4 | 1.272 [1.192, 1.358] | 1.287 [1.205, 1.375] | 27.9 | 0.575 | 0.380 | 0.68 | 27.0 |
| DtV:d4:K8 | 1.090 [0.982, 1.203] | 1.094 [0.983, 1.207] | 24.0 | 0.420 | 0.360 | 0.70 | 29 |
| DtV:d4:K16 | 0.711 [0.627, 0.810] | 0.705 [0.620, 0.806] | 15.6 | 0.236 | 0.360 | 0.70 | 31 |
| WFD:d4 | 1.466 [1.404, 1.536] | 1.500 [1.442, 1.568] | 32.2 | 0.797 | 0.360 | 0.72 | 47.5 |

K_opt **DtV:d4:K2**, runner-up DtV:d4:K4, gap 1.016 [0.988, 1.046] → **plateau**; tied set ['DtV:d4:K2', 'DtV:d4:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<16.4) | 16 | DtV:d4:K4 | 1.339 | 1.323 | 1.426 |
| mid | 17 | DtV:d4:K2 | 1.257 | 1.257 | 1.452 |
| high (>=16.9) | 17 | DtV:d4:K2 | 1.295 | 1.295 | 1.506 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.639) | 16 | DtV:d4:K2 | 1.159 | 1.159 | 1.318 |
| mid | 17 | DtV:d4:K2 | 1.263 | 1.263 | 1.468 |
| high (>=0.727) | 17 | DtV:d4:K4 | 1.479 | 1.425 | 1.594 |

## s1/huginn_eps0.1_math500 (n=50, passed)

AR: 24.6 tok/s, acc 0.14, mean len 605, mean exit depth 14.943572422014704, prefill share 0.8%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.273 [1.230, 1.305] | 1.278 [1.236, 1.308] | 31.3 | 0.939 | 0.180 | 0.66 | 23 |
| DtV:d4:K2 | 1.462 [1.399, 1.505] | 1.469 [1.408, 1.511] | 35.9 | 0.907 | 0.140 | 0.64 | 24.5 |
| DtV:d4:K4 | 1.671 [1.593, 1.723] | 1.679 [1.602, 1.731] | 41.0 | 0.871 | 0.160 | 0.58 | 26 |
| DtV:d4:K8 | 1.659 [1.502, 1.766] | 1.666 [1.510, 1.773] | 40.7 | 0.765 | 0.160 | 0.62 | 26 |
| DtV:d4:K16 | 1.529 [1.317, 1.684] | 1.534 [1.322, 1.689] | 37.6 | 0.630 | 0.160 | 0.60 | 23.5 |
| WFD:d4 | 1.763 [1.692, 1.816] | 1.774 [1.706, 1.826] | 43.3 | 0.945 | 0.160 | 0.60 | 24.5 |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K8, gap 1.007 [0.962, 1.072] → **plateau**; tied set ['DtV:d4:K4', 'DtV:d4:K8']. Prompts where AR beats K_opt: 0.02; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<15) | 16 | DtV:d4:K8 | 1.788 | 1.739 | 1.791 |
| mid | 17 | DtV:d4:K4 | 1.579 | 1.579 | 1.739 |
| high (>=15.6) | 17 | DtV:d4:K4 | 1.631 | 1.631 | 1.758 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.602) | 16 | DtV:d4:K8 | 1.356 | 1.198 | 1.604 |
| mid | 17 | DtV:d4:K2 | 1.362 | 1.350 | 1.576 |
| high (>=0.731) | 17 | DtV:d4:K16 | 1.877 | 1.767 | 1.838 |

## s1/huginn_eps0.1_spec (n=48, passed)

AR: 25.5 tok/s, acc None, mean len 434, mean exit depth 14.841311885956463, prefill share 0.6%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.289 [1.263, 1.314] | 1.291 [1.265, 1.316] | 32.9 | 0.938 | - | 0.71 | 83.0 |
| DtV:d4:K2 | 1.459 [1.415, 1.504] | 1.463 [1.419, 1.508] | 37.2 | 0.901 | - | 0.67 | 66.5 |
| DtV:d4:K4 | 1.608 [1.535, 1.682] | 1.614 [1.541, 1.689] | 41.0 | 0.837 | - | 0.71 | 69.0 |
| DtV:d4:K8 | 1.596 [1.460, 1.736] | 1.602 [1.464, 1.744] | 40.7 | 0.714 | - | 0.65 | 57 |
| DtV:d4:K16 | 1.371 [1.202, 1.566] | 1.374 [1.204, 1.571] | 35.0 | 0.550 | - | 0.62 | 46.0 |
| WFD:d4 | 1.908 [1.832, 1.988] | 1.918 [1.841, 1.998] | 48.7 | 0.940 | - | 0.69 | 65 |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K8, gap 1.007 [0.958, 1.060] → **plateau**; tied set ['DtV:d4:K4', 'DtV:d4:K8']. Prompts where AR beats K_opt: 0.06; AR beats every SD arm: 0.02.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<14.3) | 15 | DtV:d4:K4 | 1.476 | 1.476 | 1.758 |
| mid | 16 | DtV:d4:K4 | 1.653 | 1.653 | 1.907 |
| high (>=15) | 16 | DtV:d4:K8 | 1.812 | 1.678 | 2.044 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.78) | 15 | DtV:d4:K2 | 1.338 | 1.291 | 1.581 |
| mid | 16 | DtV:d4:K8 | 1.621 | 1.609 | 1.914 |
| high (>=0.911) | 16 | DtV:d4:K16 | 2.110 | 1.827 | 2.131 |

## s1/huginn_eps0.2_gsm8k3 (n=300, passed)

AR: 29.3 tok/s, acc 0.18333333333333332, mean len 116, mean exit depth 12.018158047646716, prefill share 2.6%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.149 [1.136, 1.161] | 1.152 [1.139, 1.164] | 33.7 | 0.832 | 0.197 | 0.69 | 33.0 |
| DtV:d4:K2 | 1.213 [1.192, 1.231] | 1.217 [1.197, 1.236] | 35.5 | 0.755 | 0.177 | 0.71 | 36.5 |
| DtV:d4:K4 | 1.167 [1.133, 1.200] | 1.171 [1.136, 1.204] | 34.2 | 0.623 | 0.200 | 0.70 | 34 |
| DtV:d4:K8 | 0.960 [0.918, 1.002] | 0.958 [0.915, 1.001] | 28.1 | 0.452 | 0.187 | 0.70 | 33 |
| DtV:d4:K16 | 0.650 [0.611, 0.692] | 0.643 [0.604, 0.686] | 19.0 | 0.275 | 0.187 | 0.71 | 28.0 |
| WFD:d4 | 1.405 [1.380, 1.431] | 1.419 [1.395, 1.445] | 41.2 | 0.834 | 0.193 | 0.68 | 37 |

K_opt **DtV:d4:K2**, runner-up DtV:d4:K4, gap 1.039 [1.022, 1.056] → distinct optimum; tied set ['DtV:d4:K2']. Prompts where AR beats K_opt: 0.03; AR beats every SD arm: 0.00.
Held-out K: pilot (n=100) picks DtV:d4:K2 → on confirmation (n=200) speedup 1.220 [1.194, 1.244]; eval-best DtV:d4:K2, regret ×1.000.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<12) | 100 | DtV:d4:K4 | 1.261 | 1.257 | 1.469 |
| mid | 100 | DtV:d4:K2 | 1.181 | 1.181 | 1.371 |
| high (>=12.3) | 100 | DtV:d4:K2 | 1.182 | 1.182 | 1.356 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.678) | 100 | DtV:d4:K2 | 1.087 | 1.087 | 1.295 |
| mid | 100 | DtV:d4:K2 | 1.187 | 1.187 | 1.373 |
| high (>=0.758) | 100 | DtV:d4:K4 | 1.342 | 1.308 | 1.507 |

## s1/huginn_eps0.2_math500 (n=50, passed)

AR: 32.7 tok/s, acc 0.1, mean len 493, mean exit depth 10.748882931188561, prefill share 1.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.207 [1.184, 1.223] | 1.209 [1.187, 1.225] | 39.4 | 0.948 | 0.060 | 0.60 | 52.5 |
| DtV:d4:K2 | 1.341 [1.310, 1.360] | 1.340 [1.310, 1.360] | 43.8 | 0.934 | 0.060 | 0.50 | 61 |
| DtV:d4:K4 | 1.436 [1.363, 1.478] | 1.439 [1.369, 1.481] | 46.9 | 0.872 | 0.060 | 0.50 | 52 |
| DtV:d4:K8 | 1.385 [1.219, 1.488] | 1.391 [1.226, 1.493] | 45.3 | 0.757 | 0.040 | 0.50 | 52 |
| DtV:d4:K16 | 1.151 [0.941, 1.302] | 1.152 [0.940, 1.305] | 37.6 | 0.591 | 0.060 | 0.56 | 49.5 |
| WFD:d4 | 1.486 [1.450, 1.508] | 1.490 [1.458, 1.512] | 48.5 | 0.956 | 0.060 | 0.54 | 53 |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K8, gap 1.037 [0.987, 1.131] → **plateau**; tied set ['DtV:d4:K4', 'DtV:d4:K8']. Prompts where AR beats K_opt: 0.20; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<10.8) | 16 | DtV:d4:K8 | 1.585 | 1.502 | 1.496 |
| mid | 17 | DtV:d4:K4 | 1.436 | 1.436 | 1.489 |
| high (>=11.1) | 17 | DtV:d4:K2 | 1.332 | 1.127 | 1.478 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.595) | 16 | DtV:d4:K2 | 1.243 | 0.985 | 1.258 |
| mid | 17 | DtV:d4:K2 | 1.320 | 1.114 | 1.544 |
| high (>=0.712) | 17 | DtV:d4:K8 | 1.589 | 1.520 | 1.501 |

## s1/huginn_fixed_gsm8k3 (n=300, passed)

AR: 13.2 tok/s, acc 0.26666666666666666, mean len 110, mean exit depth 32, prefill share 1.2%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.369 [1.352, 1.386] | 1.375 [1.358, 1.392] | 18.1 | 0.808 | 0.270 | 0.78 | 34.5 |
| DtV:d4:K2 | 1.612 [1.582, 1.645] | 1.624 [1.594, 1.657] | 21.4 | 0.727 | 0.270 | 0.78 | 34.5 |
| DtV:d4:K4 | 1.745 [1.691, 1.802] | 1.761 [1.706, 1.817] | 23.1 | 0.593 | 0.267 | 0.77 | 32.0 |
| DtV:d4:K8 | 1.608 [1.534, 1.690] | 1.620 [1.545, 1.704] | 21.3 | 0.418 | 0.263 | 0.77 | 38 |
| DtV:d4:K16 | 1.169 [1.096, 1.255] | 1.172 [1.098, 1.259] | 15.5 | 0.246 | 0.270 | 0.76 | 31 |
| WFD:d4 | 1.854 [1.774, 1.946] | 1.875 [1.794, 1.968] | 24.6 | 0.815 | 0.267 | 0.74 | 35 |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.082 [1.063, 1.101] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.
Held-out K: pilot (n=100) picks DtV:d4:K4 → on confirmation (n=200) speedup 1.784 [1.714, 1.852]; eval-best DtV:d4:K4, regret ×1.000.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.5) | 92 | DtV:d4:K2 | 1.452 | 1.422 | 1.548 |
| mid | 108 | DtV:d4:K4 | 1.668 | 1.668 | 1.740 |
| high (>=0.609) | 100 | DtV:d4:K4 | 2.066 | 2.066 | 2.251 |

## s1/huginn_fixed_gsm8k8 (n=50, passed)

AR: 13.3 tok/s, acc 0.3, mean len 109, mean exit depth 32, prefill share 2.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.344 [1.313, 1.381] | 1.355 [1.324, 1.391] | 17.9 | 0.785 | 0.300 | 0.86 | 26 |
| DtV:d4:K2 | 1.578 [1.527, 1.645] | 1.601 [1.549, 1.667] | 21.0 | 0.707 | 0.300 | 0.86 | 26 |
| DtV:d4:K4 | 1.672 [1.577, 1.795] | 1.699 [1.600, 1.824] | 22.3 | 0.561 | 0.320 | 0.74 | 29 |
| DtV:d4:K8 | 1.537 [1.424, 1.689] | 1.556 [1.439, 1.712] | 20.5 | 0.394 | 0.300 | 0.82 | 39 |
| DtV:d4:K16 | 1.097 [0.996, 1.241] | 1.099 [0.996, 1.247] | 14.6 | 0.231 | 0.320 | 0.84 | 72.0 |
| WFD:d4 | 1.719 [1.608, 1.875] | 1.747 [1.632, 1.908] | 22.9 | 0.800 | 0.280 | 0.74 | 28 |

K_opt **DtV:d4:K4**, runner-up DtV:d4:K2, gap 1.060 [1.028, 1.091] → distinct optimum; tied set ['DtV:d4:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.504) | 16 | DtV:d4:K2 | 1.468 | 1.440 | 1.486 |
| mid | 15 | DtV:d4:K4 | 1.625 | 1.625 | 1.653 |
| high (>=0.562) | 19 | DtV:d4:K4 | 1.940 | 1.940 | 2.017 |

## s1/huginn_fixed_math500 (n=50, passed)

AR: 13.3 tok/s, acc 0.16, mean len 726, mean exit depth 32, prefill share 0.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.461 [1.436, 1.479] | 1.464 [1.441, 1.481] | 19.4 | 0.942 | 0.180 | 0.70 | 60 |
| DtV:d4:K2 | 1.861 [1.804, 1.901] | 1.867 [1.813, 1.906] | 24.7 | 0.915 | 0.180 | 0.70 | 60 |
| DtV:d4:K4 | 2.268 [2.127, 2.362] | 2.281 [2.144, 2.374] | 30.1 | 0.851 | 0.140 | 0.64 | 50.0 |
| DtV:d4:K8 | 2.547 [2.295, 2.739] | 2.563 [2.312, 2.755] | 33.9 | 0.739 | 0.200 | 0.64 | 86.0 |
| DtV:d4:K16 | 2.393 [2.036, 2.711] | 2.406 [2.047, 2.725] | 31.8 | 0.581 | 0.180 | 0.70 | 39 |
| WFD:d4 | 2.711 [2.494, 2.867] | 2.730 [2.516, 2.884] | 36.0 | 0.952 | 0.180 | 0.70 | 60 |

K_opt **DtV:d4:K8**, runner-up DtV:d4:K16, gap 1.064 [0.997, 1.148] → **plateau**; tied set ['DtV:d4:K8', 'DtV:d4:K16']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.38) | 15 | DtV:d4:K4 | 1.546 | 1.356 | 1.613 |
| mid | 18 | DtV:d4:K4 | 1.751 | 1.696 | 1.849 |
| high (>=0.569) | 17 | DtV:d4:K16 | 3.027 | 2.878 | 3.017 |

## s1/huginn_fixed_spec (n=48, passed)

AR: 13.8 tok/s, acc None, mean len 449, mean exit depth 32, prefill share 0.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d4:K1 | 1.482 [1.461, 1.503] | 1.484 [1.463, 1.506] | 20.4 | 0.945 | - | 0.67 | 80.5 |
| DtV:d4:K2 | 1.871 [1.832, 1.906] | 1.876 [1.837, 1.912] | 25.7 | 0.912 | - | 0.67 | 80.5 |
| DtV:d4:K4 | 2.333 [2.240, 2.424] | 2.343 [2.249, 2.434] | 32.1 | 0.850 | - | 0.71 | 33.5 |
| DtV:d4:K8 | 2.582 [2.428, 2.743] | 2.595 [2.438, 2.756] | 35.5 | 0.737 | - | 0.67 | 40.5 |
| DtV:d4:K16 | 2.409 [2.148, 2.699] | 2.420 [2.158, 2.712] | 33.1 | 0.572 | - | 0.62 | 47.0 |
| WFD:d4 | 2.963 [2.764, 3.194] | 2.980 [2.779, 3.213] | 40.8 | 0.945 | - | 0.60 | 41 |

K_opt **DtV:d4:K8**, runner-up DtV:d4:K16, gap 1.072 [1.009, 1.141] → distinct optimum; tied set ['DtV:d4:K8']. Prompts where AR beats K_opt: 0.04; AR beats every SD arm: 0.04.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.733) | 15 | DtV:d4:K8 | 2.027 | 2.027 | 2.284 |
| mid | 16 | DtV:d4:K8 | 2.721 | 2.721 | 3.015 |
| high (>=0.848) | 16 | DtV:d4:K16 | 3.529 | 3.139 | 3.742 |

## s1/ouro_fixed_gsm8k3 (n=300, passed)

AR: 20.7 tok/s, acc 0.6233333333333333, mean len 107, mean exit depth 4, prefill share 0.9%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.431 [1.422, 1.440] | 1.437 [1.428, 1.446] | 29.6 | 0.900 | 0.633 | 0.77 | 29.5 |
| DtV:d1:K2 | 1.668 [1.656, 1.680] | 1.679 [1.666, 1.691] | 34.5 | 0.851 | 0.633 | 0.77 | 29.5 |
| DtV:d1:K4 | 1.832 [1.811, 1.852] | 1.847 [1.825, 1.868] | 37.9 | 0.758 | 0.627 | 0.74 | 30.0 |
| DtV:d1:K8 | 1.773 [1.738, 1.806] | 1.786 [1.751, 1.820] | 36.7 | 0.618 | 0.627 | 0.74 | 31.5 |
| DtV:d1:K16 | 1.447 [1.403, 1.491] | 1.453 [1.408, 1.497] | 29.9 | 0.447 | 0.627 | 0.76 | 29 |
| WFD:d1 | 2.206 [2.180, 2.230] | 2.232 [2.206, 2.256] | 45.7 | 0.908 | 0.617 | 0.79 | 31.0 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K8, gap 1.033 [1.023, 1.044] → distinct optimum; tied set ['DtV:d1:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.
Held-out K: pilot (n=100) picks DtV:d1:K4 → on confirmation (n=200) speedup 1.834 [1.809, 1.858]; eval-best DtV:d1:K4, regret ×1.000.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.731) | 100 | DtV:d1:K4 | 1.655 | 1.655 | 2.042 |
| mid | 94 | DtV:d1:K4 | 1.857 | 1.857 | 2.226 |
| high (>=0.8) | 106 | DtV:d1:K8 | 2.041 | 1.993 | 2.349 |

## s1/ouro_fixed_gsm8k8 (n=50, passed)

AR: 20.6 tok/s, acc 0.62, mean len 116, mean exit depth 4, prefill share 0.9%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.430 [1.415, 1.444] | 1.437 [1.423, 1.450] | 29.5 | 0.899 | 0.660 | 0.78 | 31 |
| DtV:d1:K2 | 1.653 [1.623, 1.680] | 1.665 [1.635, 1.691] | 34.1 | 0.842 | 0.660 | 0.78 | 31 |
| DtV:d1:K4 | 1.815 [1.759, 1.866] | 1.831 [1.773, 1.881] | 37.5 | 0.752 | 0.660 | 0.70 | 21 |
| DtV:d1:K8 | 1.750 [1.635, 1.873] | 1.762 [1.646, 1.887] | 36.1 | 0.616 | 0.660 | 0.68 | 24.0 |
| DtV:d1:K16 | 1.389 [1.289, 1.488] | 1.395 [1.294, 1.496] | 28.7 | 0.427 | 0.660 | 0.68 | 24.0 |
| WFD:d1 | 2.208 [2.136, 2.288] | 2.233 [2.161, 2.311] | 45.6 | 0.909 | 0.640 | 0.66 | 31 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K8, gap 1.037 [0.978, 1.093] → **plateau**; tied set ['DtV:d1:K4', 'DtV:d1:K8']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.724) | 16 | DtV:d1:K4 | 1.620 | 1.620 | 1.991 |
| mid | 17 | DtV:d1:K8 | 1.825 | 1.821 | 2.259 |
| high (>=0.787) | 17 | DtV:d1:K8 | 2.017 | 1.991 | 2.347 |

## s1/ouro_fixed_math500 (n=50, passed)

AR: 20.6 tok/s, acc 0.04, mean len 528, mean exit depth 4, prefill share 0.2%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.455 [1.446, 1.465] | 1.456 [1.447, 1.466] | 29.9 | 0.918 | 0.060 | 0.36 | 148.0 |
| DtV:d1:K2 | 1.712 [1.695, 1.730] | 1.715 [1.697, 1.732] | 35.2 | 0.880 | 0.060 | 0.36 | 148.0 |
| DtV:d1:K4 | 1.935 [1.886, 1.984] | 1.938 [1.890, 1.987] | 39.8 | 0.805 | 0.040 | 0.36 | 112.5 |
| DtV:d1:K8 | 1.935 [1.847, 2.023] | 1.938 [1.851, 2.026] | 39.8 | 0.677 | 0.060 | 0.40 | 94.0 |
| DtV:d1:K16 | 1.568 [1.475, 1.666] | 1.570 [1.477, 1.668] | 32.3 | 0.486 | 0.020 | 0.38 | 144 |
| WFD:d1 | 2.324 [2.267, 2.380] | 2.329 [2.273, 2.385] | 47.8 | 0.925 | 0.040 | 0.30 | 106 |

K_opt **DtV:d1:K8**, runner-up DtV:d1:K4, gap 1.000 [0.975, 1.022] → **plateau**; tied set ['DtV:d1:K4', 'DtV:d1:K8']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.613) | 16 | DtV:d1:K4 | 1.792 | 1.657 | 2.210 |
| mid | 17 | DtV:d1:K4 | 1.933 | 1.923 | 2.306 |
| high (>=0.701) | 17 | DtV:d1:K8 | 2.177 | 2.177 | 2.434 |

## s1/ouro_fixed_spec (n=48, passed)

AR: 22.0 tok/s, acc None, mean len 376, mean exit depth 4, prefill share 0.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.442 [1.417, 1.464] | 1.443 [1.418, 1.465] | 31.8 | 0.913 | - | 0.62 | 76.5 |
| DtV:d1:K2 | 1.694 [1.634, 1.747] | 1.697 [1.637, 1.750] | 37.3 | 0.870 | - | 0.62 | 76.5 |
| DtV:d1:K4 | 1.913 [1.792, 2.018] | 1.917 [1.795, 2.023] | 42.1 | 0.799 | - | 0.67 | 59.5 |
| DtV:d1:K8 | 1.858 [1.644, 2.069] | 1.863 [1.647, 2.076] | 40.9 | 0.649 | - | 0.65 | 91 |
| DtV:d1:K16 | 1.528 [1.291, 1.813] | 1.529 [1.291, 1.817] | 33.7 | 0.474 | - | 0.56 | 61 |
| WFD:d1 | 2.363 [2.236, 2.483] | 2.371 [2.243, 2.492] | 52.1 | 0.931 | - | 0.69 | 33 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K8, gap 1.029 [0.974, 1.091] → **plateau**; tied set ['DtV:d1:K4', 'DtV:d1:K8']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.791) | 16 | DtV:d1:K4 | 1.557 | 1.557 | 1.968 |
| mid | 16 | DtV:d1:K4 | 1.976 | 1.976 | 2.434 |
| high (>=0.897) | 16 | DtV:d1:K8 | 2.480 | 2.199 | 2.685 |

## s1/ouro_tau0.1_gsm8k3 (n=300, passed)

AR: 47.5 tok/s, acc 0.4633333333333333, mean len 113, mean exit depth 1.635078174045619, prefill share 2.0%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.030 [1.013, 1.046] | 1.031 [1.014, 1.046] | 48.9 | 0.910 | 0.467 | 0.68 | 33 |
| DtV:d1:K2 | 1.017 [1.001, 1.033] | 1.018 [1.002, 1.034] | 48.3 | 0.860 | 0.447 | 0.69 | 43.0 |
| DtV:d1:K4 | 0.980 [0.961, 0.998] | 0.980 [0.961, 0.997] | 46.6 | 0.783 | 0.443 | 0.71 | 36 |
| DtV:d1:K8 | 0.868 [0.847, 0.888] | 0.866 [0.845, 0.886] | 41.2 | 0.642 | 0.447 | 0.71 | 40.0 |
| DtV:d1:K16 | 0.681 [0.658, 0.706] | 0.677 [0.653, 0.701] | 32.4 | 0.469 | 0.450 | 0.70 | 36 |
| WFD:d1 | 1.175 [1.152, 1.197] | 1.180 [1.156, 1.202] | 55.8 | 0.793 | 0.467 | 0.64 | 34 |

K_opt **DtV:d1:K1**, runner-up DtV:d1:K2, gap 1.012 [1.003, 1.023] → **plateau**; tied set ['DtV:d1:K1', 'DtV:d1:K2']. Prompts where AR beats K_opt: 0.32; AR beats every SD arm: 0.00.
Held-out K: pilot (n=100) picks DtV:d1:K1 → on confirmation (n=200) speedup 1.020 [1.001, 1.037]; eval-best DtV:d1:K1, regret ×1.000.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<1.56) | 100 | DtV:d1:K1 | 0.999 | 0.999 | 1.109 |
| mid | 100 | DtV:d1:K1 | 1.032 | 1.032 | 1.199 |
| high (>=1.74) | 100 | DtV:d1:K1 | 1.063 | 1.063 | 1.225 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.889) | 99 | DtV:d1:K2 | 1.023 | 1.020 | 1.192 |
| mid | 100 | DtV:d1:K1 | 1.010 | 1.010 | 1.169 |
| high (>=0.935) | 101 | DtV:d1:K1 | 1.055 | 1.055 | 1.163 |

## s1/ouro_tau0.1_math500 (n=50, passed)

AR: 50.1 tok/s, acc 0.0, mean len 671, mean exit depth 1.5813252112634, prefill share 0.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 0.993 [0.942, 1.042] | 0.993 [0.943, 1.043] | 49.8 | 0.923 | 0.000 | 0.28 | 97.0 |
| DtV:d1:K2 | 1.031 [0.960, 1.095] | 1.031 [0.960, 1.095] | 51.7 | 0.895 | 0.000 | 0.16 | 108.5 |
| DtV:d1:K4 | 1.007 [0.939, 1.079] | 1.007 [0.940, 1.079] | 50.5 | 0.824 | 0.000 | 0.18 | 80 |
| DtV:d1:K8 | 0.874 [0.809, 0.944] | 0.874 [0.809, 0.944] | 43.8 | 0.679 | 0.000 | 0.16 | 97.5 |
| DtV:d1:K16 | 0.717 [0.645, 0.794] | 0.716 [0.645, 0.794] | 36.0 | 0.516 | 0.000 | 0.22 | 95 |
| WFD:d1 | 1.103 [1.053, 1.155] | 1.103 [1.054, 1.155] | 55.3 | 0.767 | 0.000 | 0.12 | 85.5 |

K_opt **DtV:d1:K2**, runner-up DtV:d1:K4, gap 1.023 [0.955, 1.095] → **plateau**; tied set ['DtV:d1:K1', 'DtV:d1:K2', 'DtV:d1:K4']. Prompts where AR beats K_opt: 0.36; AR beats every SD arm: 0.04.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<1.57) | 16 | DtV:d1:K2 | 0.946 | 0.946 | 1.016 |
| mid | 17 | DtV:d1:K4 | 1.049 | 1.018 | 1.156 |
| high (>=1.74) | 17 | DtV:d1:K2 | 1.172 | 1.172 | 1.200 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.858) | 16 | DtV:d1:K1 | 0.937 | 0.874 | 1.075 |
| mid | 17 | DtV:d1:K4 | 1.064 | 1.020 | 1.147 |
| high (>=0.902) | 17 | DtV:d1:K2 | 1.126 | 1.126 | 1.086 |

## s1/ouro_tau0.4_gsm8k3 (n=300, passed)

AR: 32.1 tok/s, acc 0.59, mean len 107, mean exit depth 2.5186526050684717, prefill share 1.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.217 [1.208, 1.226] | 1.221 [1.212, 1.230] | 39.0 | 0.903 | 0.593 | 0.84 | 23 |
| DtV:d1:K2 | 1.310 [1.298, 1.320] | 1.316 [1.305, 1.326] | 42.0 | 0.856 | 0.593 | 0.81 | 25.0 |
| DtV:d1:K4 | 1.347 [1.334, 1.361] | 1.354 [1.341, 1.368] | 43.2 | 0.774 | 0.583 | 0.81 | 29 |
| DtV:d1:K8 | 1.234 [1.214, 1.254] | 1.239 [1.219, 1.259] | 39.6 | 0.633 | 0.590 | 0.81 | 24.5 |
| DtV:d1:K16 | 0.972 [0.948, 0.997] | 0.971 [0.948, 0.996] | 31.2 | 0.455 | 0.600 | 0.83 | 25.0 |
| WFD:d1 | 1.575 [1.560, 1.589] | 1.589 [1.574, 1.602] | 50.5 | 0.911 | 0.590 | 0.80 | 24.0 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.029 [1.019, 1.041] → distinct optimum; tied set ['DtV:d1:K4']. Prompts where AR beats K_opt: 0.01; AR beats every SD arm: 0.00.
Held-out K: pilot (n=100) picks DtV:d1:K4 → on confirmation (n=200) speedup 1.349 [1.335, 1.364]; eval-best DtV:d1:K4, regret ×1.000.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<2.45) | 100 | DtV:d1:K4 | 1.377 | 1.377 | 1.564 |
| mid | 100 | DtV:d1:K4 | 1.362 | 1.362 | 1.594 |
| high (>=2.57) | 100 | DtV:d1:K4 | 1.308 | 1.308 | 1.572 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.74) | 100 | DtV:d1:K2 | 1.271 | 1.243 | 1.533 |
| mid | 100 | DtV:d1:K4 | 1.359 | 1.359 | 1.590 |
| high (>=0.82) | 100 | DtV:d1:K4 | 1.455 | 1.455 | 1.607 |

## s1/ouro_tau0.4_gsm8k8 (n=50, passed)

AR: 31.9 tok/s, acc 0.54, mean len 109, mean exit depth 2.5264919129949805, prefill share 1.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.204 [1.169, 1.228] | 1.209 [1.175, 1.232] | 38.4 | 0.898 | 0.540 | 0.84 | 48.0 |
| DtV:d1:K2 | 1.317 [1.296, 1.336] | 1.324 [1.302, 1.343] | 42.0 | 0.860 | 0.580 | 0.84 | 38.0 |
| DtV:d1:K4 | 1.329 [1.287, 1.368] | 1.338 [1.296, 1.377] | 42.4 | 0.763 | 0.560 | 0.82 | 27 |
| DtV:d1:K8 | 1.246 [1.187, 1.305] | 1.251 [1.190, 1.311] | 39.7 | 0.641 | 0.580 | 0.78 | 21 |
| DtV:d1:K16 | 0.927 [0.856, 0.999] | 0.928 [0.856, 0.999] | 29.6 | 0.432 | 0.580 | 0.82 | 27 |
| WFD:d1 | 1.579 [1.532, 1.617] | 1.595 [1.548, 1.632] | 50.4 | 0.910 | 0.560 | 0.78 | 35 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.009 [0.969, 1.042] → **plateau**; tied set ['DtV:d1:K2', 'DtV:d1:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<2.53) | 16 | DtV:d1:K8 | 1.374 | 1.337 | 1.539 |
| mid | 17 | DtV:d1:K4 | 1.315 | 1.315 | 1.590 |
| high (>=2.63) | 17 | DtV:d1:K4 | 1.354 | 1.354 | 1.632 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.732) | 16 | DtV:d1:K2 | 1.300 | 1.219 | 1.505 |
| mid | 17 | DtV:d1:K4 | 1.351 | 1.351 | 1.621 |
| high (>=0.812) | 17 | DtV:d1:K4 | 1.440 | 1.440 | 1.615 |

## s1/ouro_tau0.4_math500 (n=50, passed)

AR: 33.7 tok/s, acc 0.0, mean len 529, mean exit depth 2.4198878278005154, prefill share 0.3%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.234 [1.213, 1.256] | 1.235 [1.214, 1.256] | 41.6 | 0.930 | 0.000 | 0.34 | 121 |
| DtV:d1:K2 | 1.323 [1.299, 1.350] | 1.325 [1.300, 1.351] | 44.6 | 0.890 | 0.000 | 0.32 | 108.0 |
| DtV:d1:K4 | 1.398 [1.352, 1.438] | 1.399 [1.354, 1.439] | 47.1 | 0.830 | 0.000 | 0.32 | 102.5 |
| DtV:d1:K8 | 1.267 [1.204, 1.322] | 1.269 [1.206, 1.323] | 42.7 | 0.687 | 0.000 | 0.40 | 116.0 |
| DtV:d1:K16 | 1.069 [0.965, 1.171] | 1.070 [0.966, 1.171] | 36.0 | 0.531 | 0.000 | 0.40 | 71.5 |
| WFD:d1 | 1.572 [1.552, 1.599] | 1.575 [1.555, 1.601] | 53.0 | 0.935 | 0.000 | 0.38 | 115 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.056 [1.030, 1.082] → distinct optimum; tied set ['DtV:d1:K4']. Prompts where AR beats K_opt: 0.00; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<2.38) | 16 | DtV:d1:K4 | 1.448 | 1.448 | 1.557 |
| mid | 17 | DtV:d1:K4 | 1.407 | 1.407 | 1.598 |
| high (>=2.48) | 17 | DtV:d1:K4 | 1.329 | 1.329 | 1.552 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.781) | 16 | DtV:d1:K4 | 1.290 | 1.290 | 1.558 |
| mid | 17 | DtV:d1:K4 | 1.373 | 1.373 | 1.557 |
| high (>=0.846) | 17 | DtV:d1:K4 | 1.488 | 1.488 | 1.580 |

## s1/ouro_tau0.4_spec (n=48, passed)

AR: 37.1 tok/s, acc None, mean len 412, mean exit depth 2.289352437851248, prefill share 0.4%

| arm | speedup e2e | speedup decode-only | tok/s | acceptance | accuracy | exact to AR | first mismatch (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DtV:d1:K1 | 1.214 [1.182, 1.242] | 1.215 [1.183, 1.244] | 45.0 | 0.923 | - | 0.56 | 64 |
| DtV:d1:K2 | 1.314 [1.278, 1.349] | 1.316 [1.280, 1.351] | 48.8 | 0.888 | - | 0.58 | 48.0 |
| DtV:d1:K4 | 1.346 [1.282, 1.405] | 1.348 [1.283, 1.407] | 49.9 | 0.807 | - | 0.52 | 59 |
| DtV:d1:K8 | 1.222 [1.104, 1.336] | 1.223 [1.105, 1.338] | 45.3 | 0.673 | - | 0.58 | 53.0 |
| DtV:d1:K16 | 0.953 [0.829, 1.093] | 0.953 [0.828, 1.094] | 35.4 | 0.490 | - | 0.62 | 42.5 |
| WFD:d1 | 1.492 [1.457, 1.524] | 1.496 [1.460, 1.528] | 55.4 | 0.930 | - | 0.62 | 59.0 |

K_opt **DtV:d1:K4**, runner-up DtV:d1:K2, gap 1.024 [0.997, 1.052] → **plateau**; tied set ['DtV:d1:K2', 'DtV:d1:K4']. Prompts where AR beats K_opt: 0.02; AR beats every SD arm: 0.00.

post-hoc strata by AR mean exit depth (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<2.19) | 15 | DtV:d1:K8 | 1.526 | 1.510 | 1.502 |
| mid | 17 | DtV:d1:K4 | 1.353 | 1.353 | 1.525 |
| high (>=2.38) | 16 | DtV:d1:K2 | 1.216 | 1.168 | 1.423 |

post-hoc strata by acceptance of K_opt (prompt-level, outcome-derived; not pre-action):

| stratum | n | best DtV | its speedup | K_opt speedup | WFD speedup |
|---|---:|---|---:|---:|---:|
| low (<0.783) | 16 | DtV:d1:K2 | 1.224 | 1.160 | 1.431 |
| mid | 16 | DtV:d1:K4 | 1.379 | 1.379 | 1.522 |
| high (>=0.911) | 16 | DtV:d1:K8 | 1.569 | 1.529 | 1.521 |
