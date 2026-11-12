### oracle `huginn_eps0.1.jsonl` (huginn:eps0.1, float32)
trials 11520, prefixes 384 (24 prompts); inexact trials by action 0; prefixes kept (all actions exact) 384; restore/run stats {'prompts': 24, 'checkpoints': 384, 'trials': 11520, 'inexact_trials': 0, 'restore_failures': 0}

| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |
|---|---:|---:|---:|---:|
| DtV:d4:K8 | 0.0321 | 1.47× | 6.74 | 210.0 |
| DtV:d4:K4 | 0.0314 | 1.44× | 4.20 | 133.7 |
| DtV:d8:K8 | 0.0287 | 1.31× | 7.98 | 278.1 |
| DtV:d8:K4 | 0.0281 | 1.29× | 4.67 | 166.1 |
| DtV:d4:K2 | 0.0258 | 1.18× | 2.71 | 105.0 |
| DtV:d8:K2 | 0.0249 | 1.14× | 2.89 | 116.2 |
| DtV:d2:K4 | 0.0219 | 1.00× | 2.61 | 119.1 |
| AR | 0.0218 | 1.00× | 1.00 | 45.8 |
| DtV:d2:K2 | 0.0209 | 0.96× | 2.09 | 100.2 |
| DtV:d2:K8 | 0.0166 | 0.76× | 2.98 | 179.6 |

best static: **DtV:d4:K8** (1.47× AR). in-sample one-decision oracle: 1.252× best static
cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): **1.251× best static** [95% CI over prompts 1.218, 1.287]
cross-fitted picks: {'DtV:d4:K8': 582, 'DtV:d8:K8': 135, 'DtV:d2:K4': 115, 'DtV:d4:K4': 107, 'DtV:d2:K8': 72, 'AR': 54, 'DtV:d8:K4': 39, 'DtV:d2:K2': 18, 'DtV:d4:K2': 18, 'DtV:d8:K2': 12}

in-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):
- D=13: {'DtV:d4:K8': 8, 'DtV:d4:K4': 2, 'AR': 1, 'DtV:d8:K8': 1}
- D=14: {'DtV:d2:K4': 8, 'DtV:d4:K8': 6, 'DtV:d4:K4': 4, 'DtV:d2:K8': 4}
- D=15: {'DtV:d4:K8': 39, 'DtV:d2:K4': 16, 'DtV:d8:K8': 14, 'DtV:d4:K4': 8}
- D=16: {'DtV:d4:K8': 55, 'DtV:d8:K8': 14, 'DtV:d4:K4': 10, 'DtV:d2:K8': 9}
- D=17: {'DtV:d4:K8': 52, 'DtV:d2:K4': 5, 'DtV:d4:K4': 5, 'DtV:d8:K8': 4}
- D=18: {'DtV:d4:K8': 15, 'DtV:d4:K4': 5, 'DtV:d8:K8': 4, 'DtV:d8:K4': 3}
- D=19: {'DtV:d4:K8': 9, 'AR': 2, 'DtV:d4:K2': 1, 'DtV:d2:K4': 1}
- D=20: {'DtV:d4:K8': 3, 'DtV:d8:K8': 2, 'DtV:d4:K4': 1, 'AR': 1}
- D=21: {'DtV:d4:K8': 2, 'DtV:d2:K2': 1}
- D=22: {'DtV:d4:K8': 1}
- D=23: {'DtV:d8:K8': 1}
- D=25: {'DtV:d4:K8': 2, 'DtV:d8:K8': 1}
- D=26: {'DtV:d8:K8': 1}
- D=32: {'DtV:d4:K8': 1}

### oracle `huginn_fixed.jsonl` (fixed, float32)
trials 11520, prefixes 384 (24 prompts); inexact trials by action 0; prefixes kept (all actions exact) 384; restore/run stats {'prompts': 24, 'checkpoints': 384, 'trials': 11520, 'inexact_trials': 0, 'restore_failures': 0}

| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |
|---|---:|---:|---:|---:|
| DtV:d8:K8 | 0.0231 | 1.86× | 7.92 | 343.7 |
| DtV:d4:K8 | 0.0224 | 1.81× | 6.24 | 278.5 |
| DtV:d4:K4 | 0.0214 | 1.73× | 4.06 | 189.6 |
| DtV:d8:K4 | 0.0212 | 1.71× | 4.68 | 220.7 |
| DtV:d8:K2 | 0.0159 | 1.29× | 2.89 | 181.6 |
| DtV:d4:K2 | 0.0156 | 1.26× | 2.67 | 170.9 |
| DtV:d2:K4 | 0.0142 | 1.14× | 2.47 | 174.3 |
| AR | 0.0124 | 1.00× | 1.00 | 80.7 |
| DtV:d2:K2 | 0.0123 | 0.99× | 2.03 | 165.5 |
| DtV:d2:K8 | 0.0111 | 0.89× | 2.72 | 246.0 |

best static: **DtV:d8:K8** (1.86× AR). in-sample one-decision oracle: 1.262× best static
cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): **1.259× best static** [95% CI over prompts 1.229, 1.296]
cross-fitted picks: {'DtV:d4:K8': 558, 'DtV:d8:K8': 221, 'DtV:d4:K4': 128, 'DtV:d2:K4': 71, 'AR': 54, 'DtV:d8:K4': 49, 'DtV:d2:K8': 44, 'DtV:d8:K2': 15, 'DtV:d4:K2': 9, 'DtV:d2:K2': 3}

in-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):
- D=32: {'DtV:d4:K8': 186, 'DtV:d8:K8': 71, 'DtV:d4:K4': 45, 'DtV:d2:K4': 25}

