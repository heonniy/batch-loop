### oracle `ouro_tau0.4.jsonl` (ouro:tau0.4, float32)
trials 13824, prefixes 512 (32 prompts); inexact trials by action 0; prefixes kept (all actions exact) 512; restore/run stats {'prompts': 32, 'checkpoints': 512, 'trials': 13824, 'inexact_trials': 0, 'restore_failures': 0}

| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |
|---|---:|---:|---:|---:|
| DtV:d1:K8 | 0.0495 | 1.36× | 6.88 | 139.1 |
| DtV:d1:K4 | 0.0490 | 1.34× | 4.30 | 87.8 |
| DtV:d1:K2 | 0.0451 | 1.24× | 2.77 | 61.3 |
| DtV:d1:K1 | 0.0407 | 1.12× | 1.91 | 47.0 |
| DtV:d2:K4 | 0.0374 | 1.02× | 4.58 | 122.6 |
| DtV:d2:K2 | 0.0371 | 1.02× | 2.85 | 76.8 |
| DtV:d2:K1 | 0.0369 | 1.01× | 1.95 | 52.8 |
| DtV:d2:K8 | 0.0366 | 1.00× | 7.75 | 211.8 |
| AR | 0.0365 | 1.00× | 1.00 | 27.4 |

best static: **DtV:d1:K8** (1.36× AR). in-sample one-decision oracle: 1.221× best static
cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): **1.220× best static** [95% CI over prompts 1.186, 1.254]
cross-fitted picks: {'DtV:d1:K8': 858, 'DtV:d1:K4': 256, 'DtV:d1:K2': 154, 'AR': 100, 'DtV:d2:K8': 75, 'DtV:d1:K1': 66, 'DtV:d2:K4': 18, 'DtV:d2:K1': 5, 'DtV:d2:K2': 4}

in-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):
- D=2: {'DtV:d1:K8': 172, 'DtV:d1:K4': 43, 'DtV:d1:K2': 26, 'AR': 11}
- D=3: {'DtV:d1:K8': 103, 'DtV:d1:K4': 35, 'DtV:d1:K2': 20, 'DtV:d2:K8': 17}
- D=4: {'AR': 16, 'DtV:d1:K8': 11, 'DtV:d1:K4': 8, 'DtV:d2:K8': 7}

### oracle `ouro_tau0.1.jsonl` (ouro:tau0.1, float32)
trials 13824, prefixes 512 (32 prompts); inexact trials by action 0; prefixes kept (all actions exact) 512; restore/run stats {'prompts': 32, 'checkpoints': 512, 'trials': 13824, 'inexact_trials': 0, 'restore_failures': 0}

| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |
|---|---:|---:|---:|---:|
| DtV:d1:K2 | 0.0548 | 1.04× | 2.78 | 50.6 |
| DtV:d1:K4 | 0.0544 | 1.03× | 4.30 | 79.1 |
| DtV:d2:K2 | 0.0529 | 1.00× | 2.88 | 54.5 |
| AR | 0.0528 | 1.00× | 1.00 | 18.9 |
| DtV:d1:K1 | 0.0528 | 1.00× | 1.92 | 36.3 |
| DtV:d2:K4 | 0.0523 | 0.99× | 4.64 | 88.6 |
| DtV:d1:K8 | 0.0522 | 0.99× | 6.82 | 130.7 |
| DtV:d2:K1 | 0.0519 | 0.98× | 1.96 | 37.8 |
| DtV:d2:K8 | 0.0510 | 0.97× | 7.83 | 153.5 |

best static: **DtV:d1:K2** (1.04× AR). in-sample one-decision oracle: 1.262× best static
cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): **1.257× best static** [95% CI over prompts 1.225, 1.292]
cross-fitted picks: {'AR': 396, 'DtV:d1:K8': 323, 'DtV:d1:K4': 150, 'DtV:d1:K2': 146, 'DtV:d2:K8': 138, 'DtV:d2:K2': 109, 'DtV:d2:K4': 97, 'DtV:d1:K1': 95, 'DtV:d2:K1': 82}

in-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):
- D=1: {'AR': 111, 'DtV:d1:K2': 32, 'DtV:d2:K2': 28, 'DtV:d1:K4': 24}
- D=2: {'DtV:d1:K8': 38, 'DtV:d2:K8': 13, 'DtV:d1:K4': 12, 'DtV:d1:K2': 10}
- D=3: {'DtV:d1:K8': 35, 'DtV:d2:K8': 9, 'DtV:d1:K4': 6, 'AR': 6}
- D=4: {'AR': 11, 'DtV:d1:K8': 11, 'DtV:d2:K8': 8, 'DtV:d1:K4': 6}

### oracle `ouro_fixed.jsonl` (fixed, float32)
trials 13824, prefixes 512 (32 prompts); inexact trials by action 0; prefixes kept (all actions exact) 512; restore/run stats {'prompts': 32, 'checkpoints': 512, 'trials': 13824, 'inexact_trials': 0, 'restore_failures': 0}

| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |
|---|---:|---:|---:|---:|
| DtV:d1:K8 | 0.0475 | 1.85× | 6.86 | 144.6 |
| DtV:d1:K4 | 0.0453 | 1.76× | 4.35 | 96.1 |
| DtV:d1:K2 | 0.0379 | 1.48× | 2.78 | 73.4 |
| DtV:d2:K8 | 0.0355 | 1.38× | 7.57 | 213.5 |
| DtV:d2:K4 | 0.0351 | 1.37× | 4.53 | 129.2 |
| DtV:d2:K2 | 0.0322 | 1.26× | 2.84 | 87.9 |
| DtV:d1:K1 | 0.0311 | 1.21× | 1.92 | 61.8 |
| DtV:d2:K1 | 0.0288 | 1.12× | 1.94 | 67.3 |
| AR | 0.0257 | 1.00× | 1.00 | 39.0 |

best static: **DtV:d1:K8** (1.85× AR). in-sample one-decision oracle: 1.177× best static
cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): **1.176× best static** [95% CI over prompts 1.146, 1.207]
cross-fitted picks: {'DtV:d1:K8': 933, 'DtV:d1:K4': 282, 'DtV:d2:K8': 148, 'AR': 69, 'DtV:d1:K1': 54, 'DtV:d1:K2': 27, 'DtV:d2:K4': 11, 'DtV:d2:K2': 6, 'DtV:d2:K1': 6}

in-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):
- D=4: {'DtV:d1:K8': 311, 'DtV:d1:K4': 95, 'DtV:d2:K8': 48, 'AR': 23}

### oracle `ouro_tau0.4_bf16.jsonl` (ouro:tau0.4, bfloat16)
trials 13824, prefixes 512 (32 prompts); inexact trials by action {'DtV:d1:K4': 27, 'DtV:d1:K1': 15, 'DtV:d1:K8': 27, 'DtV:d1:K2': 18, 'DtV:d2:K8': 48, 'DtV:d2:K4': 33, 'DtV:d2:K2': 9, 'DtV:d2:K1': 6}; prefixes kept (all actions exact) 491; restore/run stats {'prompts': 32, 'checkpoints': 512, 'trials': 13824, 'inexact_trials': 183, 'restore_failures': 0}

| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |
|---|---:|---:|---:|---:|
| DtV:d1:K4 | 0.0475 | 1.47× | 4.30 | 90.5 |
| DtV:d1:K8 | 0.0471 | 1.45× | 6.86 | 145.7 |
| DtV:d1:K2 | 0.0446 | 1.38× | 2.77 | 62.1 |
| DtV:d1:K1 | 0.0405 | 1.25× | 1.91 | 47.2 |
| DtV:d2:K4 | 0.0347 | 1.07× | 4.67 | 134.5 |
| DtV:d2:K2 | 0.0344 | 1.06× | 2.88 | 83.8 |
| DtV:d2:K8 | 0.0341 | 1.05× | 7.98 | 234.3 |
| DtV:d2:K1 | 0.0338 | 1.04× | 1.95 | 57.7 |
| AR | 0.0324 | 1.00× | 1.00 | 30.9 |

best static: **DtV:d1:K4** (1.47× AR). in-sample one-decision oracle: 1.217× best static
cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): **1.214× best static** [95% CI over prompts 1.190, 1.240]
cross-fitted picks: {'DtV:d1:K8': 749, 'DtV:d1:K4': 281, 'DtV:d1:K2': 187, 'DtV:d1:K1': 109, 'AR': 81, 'DtV:d2:K8': 51, 'DtV:d2:K2': 9, 'DtV:d2:K4': 6}

in-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):
- D=2: {'DtV:d1:K8': 146, 'DtV:d1:K4': 64, 'DtV:d1:K2': 33, 'DtV:d1:K1': 11}
- D=3: {'DtV:d1:K8': 94, 'DtV:d1:K4': 25, 'DtV:d1:K1': 23, 'DtV:d1:K2': 21}
- D=4: {'AR': 15, 'DtV:d1:K8': 10, 'DtV:d1:K2': 8, 'DtV:d2:K8': 5}

