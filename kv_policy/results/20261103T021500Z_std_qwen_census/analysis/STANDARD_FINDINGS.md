# Standard-transformer control arm — findings (20261103T021500Z_std_qwen_census)

Completed units 40/40, errors 0; sources 14659

Model: Qwen2.5-3B (36 layers, non-looped). Same frozen prompts/targets as the looped-model runs; eviction = masking the token for every later query at every layer; no loop planes exist.

## Outcome structure (single-token eviction over the gold continuation)

| dataset | split | n sources | top-5% share of total KL | mean KL evidence | mean KL other | evidence share in top-5% |
|---|---|---|---|---|---|---|
| controlled_retrieval | dev | 2560 | 0.987 | 0.18 | 0.00017 | 0.414 |
| controlled_retrieval | holdout | 3840 | 0.982 | 0.153 | 0.000191 | 0.411 |
| hotpotqa_distractor | dev | 3367 | 0.628 | 0.00464 | 0.00101 | 0.536 |
| hotpotqa_distractor | holdout | 4892 | 0.857 | 0.0133 | 0.000683 | 0.494 |

## Layer-resolved attention importance: within-prompt ρ(attention at layer l, future damage), mean over prompts

| dataset | split | query | l=1 | l=2 | l=4 | l=8 | l=12 | l=18 | l=24 | l=30 | l=36 | mean over layers | early quarter | late quarter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled_retrieval | dev | consuming query t* (oracle) | -0.00096 | -0.0293 | 0.0629 | 0.087 | 0.0695 | 0.0597 | 0.0811 | 0.0955 | 0.00894 | 0.126 | 0.0886 | 0.119 |
| controlled_retrieval | dev | 2-token lookahead | -0.028 | -0.0201 | -0.00982 | 0.0248 | 0.0389 | 0.0515 | 0.0522 | 0.0942 | 0.0106 | 0.0839 | — | 0.0865 |
| controlled_retrieval | dev | question anchor | -0.00363 | -0.0152 | 0.0433 | 0.0662 | 0.0604 | 0.0483 | 0.0639 | 0.0395 | 0.0371 | 0.0655 | 0.0483 | 0.0519 |
| controlled_retrieval | holdout | consuming query t* (oracle) | -0.0152 | 0.00146 | 0.0329 | 0.0593 | 0.0267 | 0.0213 | 0.0555 | 0.0783 | 0.0164 | 0.0851 | 0.0734 | 0.0876 |
| controlled_retrieval | holdout | 2-token lookahead | -0.0303 | 0.00235 | 0.0217 | 0.0369 | 0.0245 | 0.0354 | 0.0451 | 0.0768 | 0.035 | 0.0829 | — | 0.0912 |
| controlled_retrieval | holdout | question anchor | -0.0143 | 0.0106 | 0.0228 | 0.0255 | 0.0278 | 0.0356 | 0.052 | 0.0397 | 0.0341 | 0.0545 | 0.0226 | 0.0435 |
| hotpotqa_distractor | dev | consuming query t* (oracle) | 0.0171 | 0.0262 | 0.0689 | 0.0725 | 0.00986 | 0.0547 | 0.0844 | 0.129 | 0.103 | 0.157 | 0.0984 | 0.147 |
| hotpotqa_distractor | dev | 2-token lookahead | -0.0167 | 0.0388 | 0.0652 | 0.05 | -0.00554 | 0.035 | 0.0703 | 0.142 | 0.0925 | 0.164 | — | 0.156 |
| hotpotqa_distractor | dev | question anchor | -0.00301 | 0.0252 | 0.0271 | 0.0261 | 0.0499 | 0.051 | 0.0978 | 0.0832 | 0.0538 | 0.0878 | 0.0433 | 0.0703 |
| hotpotqa_distractor | holdout | consuming query t* (oracle) | -0.0133 | 0.0138 | 0.0462 | 0.0529 | -0.016 | 0.0182 | 0.0701 | 0.126 | 0.0769 | 0.139 | 0.0757 | 0.147 |
| hotpotqa_distractor | holdout | 2-token lookahead | -0.0329 | 0.0182 | 0.0448 | 0.0366 | -0.0183 | 0.0334 | 0.0643 | 0.123 | 0.0805 | 0.148 | — | 0.154 |
| hotpotqa_distractor | holdout | question anchor | -0.0338 | -0.00779 | 0.0109 | 0.028 | 0.0221 | 0.0296 | 0.0765 | 0.0667 | 0.0379 | 0.0667 | 0.00452 | 0.0755 |

### Full per-layer curve, consuming query t* and question anchor (all layers; pooled dev+holdout)

| dataset | query | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled_retrieval | t* | -0.0095 | -0.011 | -0.005 | 0.045 | 0.0073 | 0.079 | 0.057 | 0.07 | 0.06 | 0.043 | 0.05 | 0.044 | 0.041 | 0.036 | 0.0067 | 0.044 | 0.035 | 0.037 | 0.049 | 0.057 | 0.08 | 0.07 | 0.065 | 0.066 | 0.079 | 0.087 | 0.094 | 0.087 | 0.091 | 0.085 | 0.1 | 0.095 | 0.091 | 0.091 | 0.056 | 0.013 |
| controlled_retrieval | anchor | -0.01 | 0.00028 | -0.0029 | 0.031 | 0.012 | 0.0054 | 0.055 | 0.042 | 0.06 | 0.027 | 0.059 | 0.041 | 0.025 | 0.0076 | -0.022 | 0.032 | 0.047 | 0.041 | 0.047 | 0.021 | 0.037 | 0.058 | 0.047 | 0.057 | 0.063 | 0.027 | 0.04 | 0.036 | 0.053 | 0.04 | 0.036 | 0.052 | 0.042 | 0.034 | 0.046 | 0.035 |
| hotpotqa_distractor | t* | -0.0011 | 0.019 | 0.012 | 0.055 | 0.041 | 0.086 | 0.038 | 0.061 | -0.02 | 0.0017 | 0.0093 | -0.0056 | -0.022 | 0.0069 | -0.013 | 0.027 | 0.0084 | 0.033 | 0.01 | 0.026 | 0.069 | 0.059 | 0.056 | 0.076 | 0.069 | 0.11 | 0.11 | 0.12 | 0.13 | 0.13 | 0.15 | 0.14 | 0.12 | 0.12 | 0.079 | 0.087 |
| hotpotqa_distractor | anchor | -0.022 | 0.0054 | 0.0093 | 0.017 | 0.014 | 0.03 | 0.053 | 0.027 | 0.017 | 0.021 | 0.025 | 0.033 | 0.0017 | 0.028 | 0.017 | 0.024 | 0.028 | 0.038 | 0.043 | 0.058 | 0.085 | 0.089 | 0.076 | 0.085 | 0.1 | 0.085 | 0.086 | 0.071 | 0.073 | 0.073 | 0.076 | 0.063 | 0.038 | 0.07 | 0.026 | 0.044 |

## Leave-one-UID-out ranking of damage (log10 mean KL): prompt-time vs lookahead vs future-query oracle

| dataset | split | features | mean ρ [CI] | AUROC top-10% | LIFT over A [CI] |
|---|---|---|---|---|---|
| controlled_retrieval | dev | A prompt-anchor attention (layers) + position + lexical | 0.0842 [0.033,0.132] | 0.588 | 0 [0.000,0.000] |
| controlled_retrieval | dev | A + lookahead k=1 | 0.0732 [0.030,0.113] | 0.575 | -0.011 [-0.029,0.008] |
| controlled_retrieval | dev | A + lookahead k=2 | 0.105 [0.066,0.141] | 0.65 | 0.0208 [-0.014,0.063] |
| controlled_retrieval | dev | A + lookahead k=3 | 0.102 [0.070,0.133] | 0.635 | 0.0173 [-0.022,0.067] |
| controlled_retrieval | dev | A + lookahead k=4 | 0.11 [0.079,0.139] | 0.64 | 0.0253 [-0.012,0.073] |
| controlled_retrieval | dev | A + lookahead k=6 | 0.118 [0.089,0.149] | 0.639 | 0.0341 [-0.003,0.078] |
| controlled_retrieval | dev | A + lookahead k=8 | 0.126 [0.095,0.158] | 0.638 | 0.0419 [0.003,0.085] |
| controlled_retrieval | dev | A + future-query census (oracle) | 0.106 [0.057,0.155] | 0.632 | 0.0213 [-0.027,0.070] |
| controlled_retrieval | holdout | A prompt-anchor attention (layers) + position + lexical | 0.0666 [0.032,0.095] | 0.598 | 0 [0.000,0.000] |
| controlled_retrieval | holdout | A + lookahead k=1 | 0.0627 [0.028,0.090] | 0.595 | -0.00382 [-0.013,0.005] |
| controlled_retrieval | holdout | A + lookahead k=2 | 0.0889 [0.058,0.117] | 0.627 | 0.0224 [-0.005,0.052] |
| controlled_retrieval | holdout | A + lookahead k=3 | 0.091 [0.066,0.113] | 0.627 | 0.0245 [-0.004,0.056] |
| controlled_retrieval | holdout | A + lookahead k=4 | 0.0962 [0.069,0.120] | 0.631 | 0.0297 [-0.000,0.061] |
| controlled_retrieval | holdout | A + lookahead k=6 | 0.0861 [0.059,0.109] | 0.625 | 0.0195 [-0.008,0.051] |
| controlled_retrieval | holdout | A + lookahead k=8 | 0.0816 [0.054,0.107] | 0.622 | 0.015 [-0.014,0.046] |
| controlled_retrieval | holdout | A + future-query census (oracle) | 0.0906 [0.062,0.120] | 0.627 | 0.024 [-0.007,0.057] |
| hotpotqa_distractor | dev | A prompt-anchor attention (layers) + position + lexical | 0.033 [-0.002,0.072] | 0.545 | 0 [0.000,0.000] |
| hotpotqa_distractor | dev | A + lookahead k=1 | 0.0593 [0.022,0.096] | 0.583 | 0.0263 [0.007,0.050] |
| hotpotqa_distractor | dev | A + lookahead k=2 | 0.132 [0.089,0.184] | 0.668 | 0.0985 [0.052,0.143] |
| hotpotqa_distractor | dev | A + lookahead k=3 | 0.136 [0.096,0.185] | 0.681 | 0.103 [0.056,0.143] |
| hotpotqa_distractor | dev | A + lookahead k=4 | 0.131 [0.090,0.174] | 0.675 | 0.0976 [0.050,0.140] |
| hotpotqa_distractor | dev | A + lookahead k=6 | 0.129 [0.088,0.176] | 0.672 | 0.0957 [0.049,0.137] |
| hotpotqa_distractor | dev | A + lookahead k=8 | 0.128 [0.088,0.172] | 0.669 | 0.0947 [0.051,0.135] |
| hotpotqa_distractor | dev | A + future-query census (oracle) | 0.116 [0.078,0.158] | 0.652 | 0.0832 [0.038,0.124] |
| hotpotqa_distractor | holdout | A prompt-anchor attention (layers) + position + lexical | 0.078 [0.033,0.124] | 0.593 | 0 [0.000,0.000] |
| hotpotqa_distractor | holdout | A + lookahead k=1 | 0.103 [0.054,0.151] | 0.618 | 0.025 [0.007,0.044] |
| hotpotqa_distractor | holdout | A + lookahead k=2 | 0.143 [0.078,0.202] | 0.679 | 0.0649 [0.017,0.115] |
| hotpotqa_distractor | holdout | A + lookahead k=3 | 0.143 [0.080,0.201] | 0.683 | 0.0648 [0.022,0.110] |
| hotpotqa_distractor | holdout | A + lookahead k=4 | 0.136 [0.072,0.195] | 0.68 | 0.0584 [0.015,0.105] |
| hotpotqa_distractor | holdout | A + lookahead k=6 | 0.135 [0.070,0.192] | 0.682 | 0.0571 [0.014,0.102] |
| hotpotqa_distractor | holdout | A + lookahead k=8 | 0.136 [0.071,0.193] | 0.683 | 0.0583 [0.015,0.103] |
| hotpotqa_distractor | holdout | A + future-query census (oracle) | 0.112 [0.051,0.169] | 0.66 | 0.0339 [-0.008,0.076] |

## Query mismatch diagnostics (standard model): Spearman of the question anchor's Context attention vs the consuming generated query's, and evidence mass by layer quarter

| dataset | split | ρ(anchor mean-layer, t* mean-layer) [CI] | ρ(anchor late-quarter, t* late-quarter) | ρ(format t=1, t*) | anchor evidence mass (early/late quarter) | t* evidence mass (early/late quarter) |
|---|---|---|---|---|---|---|
| controlled_retrieval | dev | 0.578 [0.559,0.601] | 0.426 | 0.272 | 0.193 / 0.116 | 0.251 / 0.338 |
| controlled_retrieval | holdout | 0.581 [0.567,0.599] | 0.378 | 0.261 | 0.188 / 0.108 | 0.219 / 0.321 |
| hotpotqa_distractor | dev | 0.615 [0.482,0.744] | 0.514 | 0.643 | 0.254 / 0.342 | 0.338 / 0.49 |
| hotpotqa_distractor | holdout | 0.511 [0.387,0.627] | 0.461 | 0.568 | 0.275 / 0.44 | 0.332 / 0.583 |

## Equal-budget token retention policies (gold ΔNLL_future, greedy EM; paired vs random)

qa_* = question-anchor attention (last layer / mean / early quarter / late quarter / max layer); la2 = 2-token format lookahead; def4 = deferred after 4 generated tokens; fut_oracle = all generated queries (oracle); *_early/_late = layer quarter.

| dataset | split | keep | policy | ΔNLL mean [CI] | EM | ΔNLL − random [CI] | EM − random |
|---|---|---|---|---|---|---|---|
| controlled_retrieval | dev | 0.1 | def4_early | 1.94 [1.788,2.081] | 0 | -0.406 [-0.903,0.007] | 0 |
| controlled_retrieval | dev | 0.1 | def4_late | 0.402 [0.017,0.790] | 0.62 | -1.94 [-2.609,-1.348] | 0.62 |
| controlled_retrieval | dev | 0.1 | def4_mean | 0.373 [0.017,0.715] | 0.62 | -1.97 [-2.614,-1.413] | 0.62 |
| controlled_retrieval | dev | 0.1 | fut_oracle_early | 0.0129 [0.006,0.020] | 1 | -2.33 [-2.771,-1.957] | 1 |
| controlled_retrieval | dev | 0.1 | fut_oracle_late | 0.0171 [0.011,0.024] | 1 | -2.33 [-2.758,-1.955] | 1 |
| controlled_retrieval | dev | 0.1 | fut_oracle_mean | 0.00913 [0.003,0.015] | 1 | -2.34 [-2.768,-1.963] | 1 |
| controlled_retrieval | dev | 0.1 | la2_early | 2.18 [1.993,2.383] | 0 | -0.164 [-0.418,0.045] | 0 |
| controlled_retrieval | dev | 0.1 | la2_late | 0.48 [0.123,0.818] | 0.5 | -1.87 [-2.582,-1.223] | 0.5 |
| controlled_retrieval | dev | 0.1 | la2_mean | 0.556 [0.227,0.845] | 0.38 | -1.79 [-2.549,-1.132] | 0.38 |
| controlled_retrieval | dev | 0.1 | qa_early | 2.95 [2.404,3.437] | 0 | 0.606 [-0.166,1.298] | 0 |
| controlled_retrieval | dev | 0.1 | qa_last_layer | 1.67 [1.526,1.855] | 0 | -0.672 [-0.970,-0.404] | 0 |
| controlled_retrieval | dev | 0.1 | qa_late | 1.34 [1.125,1.536] | 0 | -1.01 [-1.638,-0.470] | 0 |
| controlled_retrieval | dev | 0.1 | qa_max_layer | 1.24 [0.923,1.623] | 0 | -1.11 [-1.391,-0.869] | 0 |
| controlled_retrieval | dev | 0.1 | qa_mean | 1.54 [1.246,1.827] | 0 | -0.802 [-1.077,-0.594] | 0 |
| controlled_retrieval | dev | 0.1 | random_11 | 2.56 [2.091,3.035] | 0 | 0.213 [0.031,0.408] | 0 |
| controlled_retrieval | dev | 0.1 | random_29 | 2.13 [1.757,2.609] | 0 | -0.213 [-0.408,-0.031] | 0 |
| controlled_retrieval | dev | 0.3 | def4_early | 1.67 [1.550,1.786] | 0 | 0.0829 [-0.156,0.317] | 0 |
| controlled_retrieval | dev | 0.3 | def4_late | 0.00453 [0.001,0.008] | 1 | -1.58 [-1.770,-1.367] | 1 |
| controlled_retrieval | dev | 0.3 | def4_mean | 0.00795 [0.001,0.014] | 1 | -1.58 [-1.769,-1.363] | 1 |
| controlled_retrieval | dev | 0.3 | fut_oracle_early | 0.00423 [-0.001,0.010] | 1 | -1.58 [-1.775,-1.364] | 1 |
| controlled_retrieval | dev | 0.3 | fut_oracle_late | 0.0113 [0.007,0.015] | 1 | -1.57 [-1.768,-1.357] | 1 |
| controlled_retrieval | dev | 0.3 | fut_oracle_mean | -0.000304 [-0.007,0.007] | 1 | -1.58 [-1.777,-1.370] | 1 |
| controlled_retrieval | dev | 0.3 | la2_early | 1.86 [1.583,2.129] | 0 | 0.277 [-0.076,0.636] | 0 |
| controlled_retrieval | dev | 0.3 | la2_late | 0.00978 [0.005,0.015] | 1 | -1.57 [-1.768,-1.361] | 1 |
| controlled_retrieval | dev | 0.3 | la2_mean | 0.378 [0.012,0.749] | 0.62 | -1.21 [-1.534,-0.885] | 0.62 |
| controlled_retrieval | dev | 0.3 | qa_early | 0.617 [0.140,1.234] | 0.5 | -0.968 [-1.488,-0.259] | 0.5 |
| controlled_retrieval | dev | 0.3 | qa_last_layer | 1.51 [1.230,1.725] | 0 | -0.0792 [-0.413,0.243] | 0 |
| controlled_retrieval | dev | 0.3 | qa_late | 1.11 [0.465,1.669] | 0.38 | -0.475 [-1.157,0.141] | 0.38 |
| controlled_retrieval | dev | 0.3 | qa_max_layer | 0.669 [0.276,1.057] | 0.38 | -0.916 [-1.279,-0.524] | 0.38 |
| controlled_retrieval | dev | 0.3 | qa_mean | 0.43 [0.112,0.738] | 0.5 | -1.15 [-1.431,-0.851] | 0.5 |
| controlled_retrieval | dev | 0.3 | random_11 | 1.49 [1.135,1.838] | 0 | -0.0936 [-0.311,0.191] | 0 |
| controlled_retrieval | dev | 0.3 | random_29 | 1.68 [1.376,1.960] | 0 | 0.0936 [-0.191,0.311] | 0 |
| controlled_retrieval | dev | 0.5 | def4_early | 1.06 [0.686,1.338] | 0.12 | -0.332 [-0.570,-0.052] | 0.12 |
| controlled_retrieval | dev | 0.5 | def4_late | 0.00242 [-0.003,0.010] | 1 | -1.39 [-1.608,-1.149] | 1 |
| controlled_retrieval | dev | 0.5 | def4_mean | 0.00518 [-0.002,0.013] | 1 | -1.38 [-1.604,-1.147] | 1 |
| controlled_retrieval | dev | 0.5 | fut_oracle_early | -0.00106 [-0.004,0.001] | 1 | -1.39 [-1.609,-1.152] | 1 |
| controlled_retrieval | dev | 0.5 | fut_oracle_late | 0.0073 [0.002,0.014] | 1 | -1.38 [-1.603,-1.146] | 1 |
| controlled_retrieval | dev | 0.5 | fut_oracle_mean | 0.00367 [-0.000,0.008] | 1 | -1.38 [-1.607,-1.148] | 1 |
| controlled_retrieval | dev | 0.5 | la2_early | 1.6 [0.834,2.179] | 0.25 | 0.214 [-0.396,0.743] | 0.25 |
| controlled_retrieval | dev | 0.5 | la2_late | 0.00554 [0.001,0.010] | 1 | -1.38 [-1.605,-1.147] | 1 |
| controlled_retrieval | dev | 0.5 | la2_mean | -6.98e-05 [-0.005,0.005] | 1 | -1.39 [-1.609,-1.152] | 1 |
| controlled_retrieval | dev | 0.5 | qa_early | 0.232 [-0.002,0.486] | 0.75 | -1.16 [-1.311,-1.003] | 0.75 |
| controlled_retrieval | dev | 0.5 | qa_last_layer | 1.21 [0.609,1.644] | 0.25 | -0.18 [-0.695,0.237] | 0.25 |
| controlled_retrieval | dev | 0.5 | qa_late | 0.891 [0.363,1.369] | 0.38 | -0.498 [-0.941,-0.096] | 0.38 |
| controlled_retrieval | dev | 0.5 | qa_max_layer | 0.23 [-0.001,0.551] | 0.75 | -1.16 [-1.427,-0.827] | 0.75 |
| controlled_retrieval | dev | 0.5 | qa_mean | 0.401 [0.008,0.777] | 0.62 | -0.987 [-1.224,-0.658] | 0.62 |
| controlled_retrieval | dev | 0.5 | random_11 | 1.3 [1.054,1.517] | 0 | -0.0898 [-0.267,0.094] | 0 |
| controlled_retrieval | dev | 0.5 | random_29 | 1.48 [1.124,1.813] | 0 | 0.0898 [-0.094,0.267] | 0 |
| controlled_retrieval | holdout | 0.1 | def4_early | 1.92 [1.802,2.050] | 0 | -0.405 [-0.627,-0.170] | 0 |
| controlled_retrieval | holdout | 0.1 | def4_late | 0.338 [0.075,0.628] | 0.67 | -1.99 [-2.332,-1.574] | 0.67 |
| controlled_retrieval | holdout | 0.1 | def4_mean | 0.539 [0.277,0.806] | 0.42 | -1.79 [-2.161,-1.407] | 0.42 |
| controlled_retrieval | holdout | 0.1 | fut_oracle_early | 0.0175 [0.007,0.030] | 1 | -2.31 [-2.512,-2.086] | 1 |
| controlled_retrieval | holdout | 0.1 | fut_oracle_late | 0.0121 [0.006,0.019] | 1 | -2.32 [-2.520,-2.091] | 1 |
| controlled_retrieval | holdout | 0.1 | fut_oracle_mean | 0.016 [0.008,0.024] | 1 | -2.31 [-2.519,-2.086] | 1 |
| controlled_retrieval | holdout | 0.1 | la2_early | 2.24 [2.089,2.388] | 0 | -0.0906 [-0.227,0.044] | 0 |
| controlled_retrieval | holdout | 0.1 | la2_late | 0.736 [0.433,1.052] | 0.33 | -1.59 [-2.004,-1.168] | 0.33 |
| controlled_retrieval | holdout | 0.1 | la2_mean | 0.67 [0.397,0.950] | 0.33 | -1.66 [-2.053,-1.243] | 0.33 |
| controlled_retrieval | holdout | 0.1 | qa_early | 2.98 [2.700,3.236] | 0 | 0.654 [0.290,1.027] | 0 |
| controlled_retrieval | holdout | 0.1 | qa_last_layer | 1.53 [1.337,1.697] | 0 | -0.795 [-1.005,-0.566] | 0 |
| controlled_retrieval | holdout | 0.1 | qa_late | 1.17 [0.880,1.421] | 0.083 | -1.16 [-1.500,-0.817] | 0.083 |
| controlled_retrieval | holdout | 0.1 | qa_max_layer | 1.43 [1.147,1.710] | 0 | -0.901 [-1.159,-0.656] | 0 |
| controlled_retrieval | holdout | 0.1 | qa_mean | 1.58 [1.250,1.876] | 0 | -0.749 [-1.003,-0.528] | 0 |
| controlled_retrieval | holdout | 0.1 | random_11 | 2.16 [1.865,2.455] | 0 | -0.168 [-0.412,0.089] | 0 |
| controlled_retrieval | holdout | 0.1 | random_29 | 2.5 [2.109,2.858] | 0 | 0.168 [-0.089,0.412] | 0 |
| controlled_retrieval | holdout | 0.3 | def4_early | 1.57 [1.453,1.690] | 0 | 0.0646 [-0.127,0.273] | 0 |
| controlled_retrieval | holdout | 0.3 | def4_late | 0.00737 [0.001,0.014] | 1 | -1.5 [-1.684,-1.323] | 1 |
| controlled_retrieval | holdout | 0.3 | def4_mean | 0.0979 [0.002,0.283] | 0.92 | -1.41 [-1.609,-1.213] | 0.92 |
| controlled_retrieval | holdout | 0.3 | fut_oracle_early | 0.0116 [0.006,0.018] | 1 | -1.49 [-1.678,-1.318] | 1 |
| controlled_retrieval | holdout | 0.3 | fut_oracle_late | 0.00912 [0.004,0.015] | 1 | -1.5 [-1.685,-1.319] | 1 |
| controlled_retrieval | holdout | 0.3 | fut_oracle_mean | 0.00947 [0.004,0.015] | 1 | -1.49 [-1.684,-1.320] | 1 |
| controlled_retrieval | holdout | 0.3 | la2_early | 1.9 [1.656,2.115] | 0 | 0.394 [0.100,0.688] | 0 |
| controlled_retrieval | holdout | 0.3 | la2_late | 0.00733 [0.000,0.014] | 1 | -1.5 [-1.685,-1.322] | 1 |
| controlled_retrieval | holdout | 0.3 | la2_mean | 0.318 [0.091,0.573] | 0.58 | -1.19 [-1.480,-0.879] | 0.58 |
| controlled_retrieval | holdout | 0.3 | qa_early | 0.867 [0.341,1.422] | 0.5 | -0.638 [-1.177,-0.072] | 0.5 |
| controlled_retrieval | holdout | 0.3 | qa_last_layer | 1.5 [1.255,1.758] | 0 | -0.00289 [-0.338,0.308] | 0 |
| controlled_retrieval | holdout | 0.3 | qa_late | 1.27 [0.795,1.761] | 0.33 | -0.231 [-0.750,0.261] | 0.33 |
| controlled_retrieval | holdout | 0.3 | qa_max_layer | 0.729 [0.452,0.983] | 0.25 | -0.775 [-1.133,-0.425] | 0.25 |
| controlled_retrieval | holdout | 0.3 | qa_mean | 0.492 [0.173,0.839] | 0.58 | -1.01 [-1.382,-0.611] | 0.58 |
| controlled_retrieval | holdout | 0.3 | random_11 | 1.5 [1.239,1.773] | 0 | -0.00012 [-0.133,0.134] | 0 |
| controlled_retrieval | holdout | 0.3 | random_29 | 1.5 [1.321,1.674] | 0 | 0.00012 [-0.134,0.133] | 0 |
| controlled_retrieval | holdout | 0.5 | def4_early | 0.974 [0.614,1.302] | 0.25 | -0.329 [-0.671,-0.011] | 0.25 |
| controlled_retrieval | holdout | 0.5 | def4_late | 0.00571 [0.001,0.010] | 1 | -1.3 [-1.454,-1.157] | 1 |
| controlled_retrieval | holdout | 0.5 | def4_mean | 0.00717 [0.003,0.012] | 1 | -1.3 [-1.456,-1.152] | 1 |
| controlled_retrieval | holdout | 0.5 | fut_oracle_early | 0.00577 [0.002,0.010] | 1 | -1.3 [-1.456,-1.156] | 1 |
| controlled_retrieval | holdout | 0.5 | fut_oracle_late | 0.00501 [-0.001,0.010] | 1 | -1.3 [-1.457,-1.154] | 1 |
| controlled_retrieval | holdout | 0.5 | fut_oracle_mean | 0.00777 [0.003,0.012] | 1 | -1.3 [-1.453,-1.156] | 1 |
| controlled_retrieval | holdout | 0.5 | la2_early | 1.3 [0.717,1.815] | 0.33 | -0.0011 [-0.564,0.493] | 0.33 |
| controlled_retrieval | holdout | 0.5 | la2_late | 0.00102 [-0.004,0.006] | 1 | -1.3 [-1.461,-1.161] | 1 |
| controlled_retrieval | holdout | 0.5 | la2_mean | 0.000589 [-0.004,0.005] | 1 | -1.3 [-1.461,-1.161] | 1 |
| controlled_retrieval | holdout | 0.5 | qa_early | 0.274 [-0.002,0.616] | 0.75 | -1.03 [-1.241,-0.763] | 0.75 |
| controlled_retrieval | holdout | 0.5 | qa_last_layer | 1.16 [0.822,1.492] | 0.17 | -0.138 [-0.552,0.251] | 0.17 |
| controlled_retrieval | holdout | 0.5 | qa_late | 0.973 [0.591,1.356] | 0.33 | -0.33 [-0.720,0.062] | 0.33 |
| controlled_retrieval | holdout | 0.5 | qa_max_layer | 0.0904 [-0.001,0.257] | 0.92 | -1.21 [-1.394,-1.037] | 0.92 |
| controlled_retrieval | holdout | 0.5 | qa_mean | 0.161 [0.013,0.362] | 0.83 | -1.14 [-1.360,-0.925] | 0.83 |
| controlled_retrieval | holdout | 0.5 | random_11 | 1.32 [1.093,1.548] | 0 | 0.0137 [-0.115,0.134] | 0 |
| controlled_retrieval | holdout | 0.5 | random_29 | 1.29 [1.119,1.456] | 0 | -0.0137 [-0.134,0.115] | 0 |
| hotpotqa_distractor | dev | 0.1 | def4_early | 0.0357 [-0.119,0.224] | 0.38 | -1.47 [-2.181,-0.812] | 0.25 |
| hotpotqa_distractor | dev | 0.1 | def4_late | -0.0101 [-0.039,0.016] | 0.5 | -1.51 [-2.131,-0.880] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | def4_mean | -0.0269 [-0.059,0.003] | 0.5 | -1.53 [-2.161,-0.882] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | fut_oracle_early | -0.0894 [-0.145,-0.039] | 0.5 | -1.59 [-2.252,-0.915] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | fut_oracle_late | 0.00137 [-0.032,0.030] | 0.5 | -1.5 [-2.125,-0.865] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | fut_oracle_mean | -0.0197 [-0.041,0.003] | 0.5 | -1.52 [-2.129,-0.896] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | la2_early | 1.81 [1.010,2.587] | 0.12 | 0.307 [-0.035,0.714] | 0 |
| hotpotqa_distractor | dev | 0.1 | la2_late | 0.163 [-0.016,0.475] | 0.5 | -1.34 [-1.994,-0.763] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | la2_mean | 0.355 [-0.017,0.738] | 0.5 | -1.15 [-1.721,-0.681] | 0.38 |
| hotpotqa_distractor | dev | 0.1 | qa_early | 0.896 [0.181,1.812] | 0.38 | -0.607 [-1.348,0.172] | 0.25 |
| hotpotqa_distractor | dev | 0.1 | qa_last_layer | 1.42 [0.647,2.255] | 0.12 | -0.0869 [-0.855,0.630] | 0 |
| hotpotqa_distractor | dev | 0.1 | qa_late | 1.16 [0.470,1.992] | 0.12 | -0.346 [-1.058,0.351] | 0 |
| hotpotqa_distractor | dev | 0.1 | qa_max_layer | 1.25 [0.318,2.253] | 0.38 | -0.252 [-0.953,0.443] | 0.25 |
| hotpotqa_distractor | dev | 0.1 | qa_mean | 1.59 [0.613,2.573] | 0.12 | 0.0852 [-0.715,0.820] | 0 |
| hotpotqa_distractor | dev | 0.1 | random_11 | 1.78 [1.084,2.466] | 0 | 0.28 [0.033,0.551] | -0.12 |
| hotpotqa_distractor | dev | 0.1 | random_29 | 1.22 [0.616,1.864] | 0.25 | -0.28 [-0.551,-0.033] | 0.12 |
| hotpotqa_distractor | dev | 0.3 | def4_early | 0.0741 [-0.046,0.241] | 0.38 | -0.919 [-1.288,-0.543] | 0.25 |
| hotpotqa_distractor | dev | 0.3 | def4_late | 0.0192 [-0.001,0.042] | 0.5 | -0.974 [-1.387,-0.583] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | def4_mean | -0.0106 [-0.037,0.018] | 0.5 | -1 [-1.411,-0.618] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | fut_oracle_early | -0.0374 [-0.064,-0.006] | 0.5 | -1.03 [-1.470,-0.611] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | fut_oracle_late | 0.0128 [-0.008,0.038] | 0.5 | -0.98 [-1.380,-0.601] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | fut_oracle_mean | -0.00304 [-0.033,0.027] | 0.5 | -0.996 [-1.408,-0.610] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | la2_early | 0.326 [-0.034,0.975] | 0.5 | -0.667 [-1.143,-0.231] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | la2_late | 0.02 [-0.001,0.043] | 0.5 | -0.973 [-1.381,-0.594] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | la2_mean | 0.03 [-0.001,0.067] | 0.5 | -0.963 [-1.368,-0.586] | 0.38 |
| hotpotqa_distractor | dev | 0.3 | qa_early | 0.843 [0.029,1.750] | 0.38 | -0.151 [-0.771,0.487] | 0.25 |
| hotpotqa_distractor | dev | 0.3 | qa_last_layer | 0.853 [0.417,1.356] | 0.12 | -0.141 [-0.635,0.323] | 0 |
| hotpotqa_distractor | dev | 0.3 | qa_late | 0.945 [0.224,1.867] | 0.25 | -0.0486 [-0.779,0.657] | 0.12 |
| hotpotqa_distractor | dev | 0.3 | qa_max_layer | 0.698 [0.046,1.608] | 0.38 | -0.295 [-0.879,0.362] | 0.25 |
| hotpotqa_distractor | dev | 0.3 | qa_mean | 0.684 [-0.003,1.640] | 0.38 | -0.309 [-0.925,0.394] | 0.25 |
| hotpotqa_distractor | dev | 0.3 | random_11 | 1.01 [0.538,1.531] | 0.12 | 0.0143 [-0.275,0.345] | 0 |
| hotpotqa_distractor | dev | 0.3 | random_29 | 0.979 [0.495,1.560] | 0.12 | -0.0143 [-0.345,0.275] | 0 |
| hotpotqa_distractor | dev | 0.5 | def4_early | -0.0282 [-0.052,-0.005] | 0.5 | -0.562 [-0.980,-0.170] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | def4_late | 0.00899 [-0.006,0.026] | 0.5 | -0.525 [-0.947,-0.128] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | def4_mean | -0.00488 [-0.036,0.023] | 0.5 | -0.539 [-0.949,-0.155] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | fut_oracle_early | -0.0301 [-0.047,-0.014] | 0.5 | -0.564 [-1.000,-0.156] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | fut_oracle_late | 0.00163 [-0.016,0.022] | 0.5 | -0.533 [-0.953,-0.131] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | fut_oracle_mean | -0.0105 [-0.038,0.012] | 0.5 | -0.545 [-0.970,-0.155] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | la2_early | 0.211 [-0.044,0.696] | 0.5 | -0.323 [-0.751,0.063] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | la2_late | 0.00443 [-0.012,0.023] | 0.5 | -0.53 [-0.947,-0.136] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | la2_mean | 0.00662 [-0.014,0.034] | 0.5 | -0.528 [-0.935,-0.142] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | qa_early | 0.909 [0.003,1.917] | 0.38 | 0.375 [-0.575,1.408] | 0 |
| hotpotqa_distractor | dev | 0.5 | qa_last_layer | 0.506 [0.097,0.924] | 0.25 | -0.0277 [-0.664,0.610] | -0.12 |
| hotpotqa_distractor | dev | 0.5 | qa_late | 0.695 [0.036,1.650] | 0.38 | 0.161 [-0.554,0.896] | 0 |
| hotpotqa_distractor | dev | 0.5 | qa_max_layer | 0.699 [0.002,1.669] | 0.5 | 0.164 [-0.657,1.031] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | qa_mean | 0.488 [0.000,1.412] | 0.5 | -0.0463 [-0.688,0.703] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | random_11 | 0.521 [0.017,1.046] | 0.5 | -0.013 [-0.228,0.190] | 0.12 |
| hotpotqa_distractor | dev | 0.5 | random_29 | 0.547 [0.134,0.992] | 0.25 | 0.013 [-0.190,0.228] | -0.12 |
| hotpotqa_distractor | holdout | 0.1 | def4_early | 0.149 [-0.081,0.478] | 0.58 | -1.1 [-1.785,-0.482] | 0.33 |
| hotpotqa_distractor | holdout | 0.1 | def4_late | 0.0333 [-0.005,0.079] | 0.58 | -1.21 [-1.855,-0.635] | 0.33 |
| hotpotqa_distractor | holdout | 0.1 | def4_mean | 0.00764 [-0.018,0.032] | 0.58 | -1.24 [-1.875,-0.665] | 0.33 |
| hotpotqa_distractor | holdout | 0.1 | fut_oracle_early | -0.078 [-0.147,-0.028] | 0.83 | -1.33 [-1.958,-0.747] | 0.58 |
| hotpotqa_distractor | holdout | 0.1 | fut_oracle_late | 0.0386 [-0.001,0.082] | 0.5 | -1.21 [-1.851,-0.626] | 0.25 |
| hotpotqa_distractor | holdout | 0.1 | fut_oracle_mean | 0.00351 [-0.022,0.027] | 0.58 | -1.24 [-1.882,-0.664] | 0.33 |
| hotpotqa_distractor | holdout | 0.1 | la2_early | 1.21 [0.679,1.759] | 0.17 | -0.0394 [-0.205,0.130] | -0.083 |
| hotpotqa_distractor | holdout | 0.1 | la2_late | 0.383 [0.019,0.825] | 0.42 | -0.865 [-1.551,-0.274] | 0.17 |
| hotpotqa_distractor | holdout | 0.1 | la2_mean | 0.389 [0.027,0.851] | 0.42 | -0.858 [-1.549,-0.247] | 0.17 |
| hotpotqa_distractor | holdout | 0.1 | qa_early | 1.23 [0.735,1.729] | 0.17 | -0.0178 [-0.276,0.215] | -0.083 |
| hotpotqa_distractor | holdout | 0.1 | qa_last_layer | 1.23 [0.721,1.778] | 0.25 | -0.0188 [-0.206,0.166] | 0 |
| hotpotqa_distractor | holdout | 0.1 | qa_late | 1.13 [0.625,1.676] | 0.25 | -0.121 [-0.432,0.138] | 0 |
| hotpotqa_distractor | holdout | 0.1 | qa_max_layer | 0.997 [0.502,1.558] | 0.25 | -0.251 [-0.529,-0.032] | 0 |
| hotpotqa_distractor | holdout | 0.1 | qa_mean | 1.08 [0.584,1.647] | 0.17 | -0.163 [-0.349,0.018] | -0.083 |
| hotpotqa_distractor | holdout | 0.1 | random_11 | 1.15 [0.566,1.790] | 0.25 | -0.0999 [-0.280,0.086] | 0 |
| hotpotqa_distractor | holdout | 0.1 | random_29 | 1.35 [0.727,2.011] | 0.25 | 0.0999 [-0.086,0.280] | 0 |
| hotpotqa_distractor | holdout | 0.3 | def4_early | 0.211 [-0.095,0.625] | 0.58 | -0.515 [-0.982,-0.064] | 0.21 |
| hotpotqa_distractor | holdout | 0.3 | def4_late | 0.0175 [0.000,0.033] | 0.58 | -0.708 [-1.170,-0.299] | 0.21 |
| hotpotqa_distractor | holdout | 0.3 | def4_mean | 0.0128 [-0.008,0.035] | 0.58 | -0.712 [-1.170,-0.306] | 0.21 |
| hotpotqa_distractor | holdout | 0.3 | fut_oracle_early | -0.0707 [-0.139,-0.017] | 0.75 | -0.796 [-1.240,-0.389] | 0.38 |
| hotpotqa_distractor | holdout | 0.3 | fut_oracle_late | 0.0141 [-0.004,0.032] | 0.58 | -0.711 [-1.173,-0.303] | 0.21 |
| hotpotqa_distractor | holdout | 0.3 | fut_oracle_mean | 0.0153 [-0.008,0.037] | 0.67 | -0.71 [-1.175,-0.298] | 0.29 |
| hotpotqa_distractor | holdout | 0.3 | la2_early | 0.671 [0.179,1.245] | 0.33 | -0.0542 [-0.509,0.336] | -0.042 |
| hotpotqa_distractor | holdout | 0.3 | la2_late | 0.0995 [0.012,0.242] | 0.67 | -0.626 [-1.098,-0.214] | 0.29 |
| hotpotqa_distractor | holdout | 0.3 | la2_mean | 0.2 [0.018,0.454] | 0.5 | -0.525 [-1.011,-0.125] | 0.12 |
| hotpotqa_distractor | holdout | 0.3 | qa_early | 1.23 [0.652,1.822] | 0.25 | 0.505 [0.221,0.814] | -0.12 |
| hotpotqa_distractor | holdout | 0.3 | qa_last_layer | 0.766 [0.319,1.234] | 0.33 | 0.0406 [-0.373,0.361] | -0.042 |
| hotpotqa_distractor | holdout | 0.3 | qa_late | 1.11 [0.563,1.684] | 0.25 | 0.381 [0.036,0.777] | -0.12 |
| hotpotqa_distractor | holdout | 0.3 | qa_max_layer | 1.13 [0.493,1.832] | 0.25 | 0.402 [0.089,0.738] | -0.12 |
| hotpotqa_distractor | holdout | 0.3 | qa_mean | 0.948 [0.395,1.563] | 0.17 | 0.223 [-0.056,0.547] | -0.21 |
| hotpotqa_distractor | holdout | 0.3 | random_11 | 0.952 [0.383,1.630] | 0.17 | 0.227 [-0.169,0.646] | -0.21 |
| hotpotqa_distractor | holdout | 0.3 | random_29 | 0.498 [-0.024,1.132] | 0.58 | -0.227 [-0.646,0.169] | 0.21 |
| hotpotqa_distractor | holdout | 0.5 | def4_early | -0.011 [-0.040,0.022] | 0.58 | -0.623 [-0.991,-0.279] | 0.17 |
| hotpotqa_distractor | holdout | 0.5 | def4_late | 0.000499 [-0.012,0.014] | 0.75 | -0.612 [-0.989,-0.261] | 0.33 |
| hotpotqa_distractor | holdout | 0.5 | def4_mean | 0.00584 [-0.003,0.016] | 0.67 | -0.606 [-0.987,-0.255] | 0.25 |
| hotpotqa_distractor | holdout | 0.5 | fut_oracle_early | -0.0426 [-0.086,-0.004] | 0.75 | -0.655 [-1.017,-0.306] | 0.33 |
| hotpotqa_distractor | holdout | 0.5 | fut_oracle_late | 0.00181 [-0.013,0.017] | 0.75 | -0.61 [-0.988,-0.260] | 0.33 |
| hotpotqa_distractor | holdout | 0.5 | fut_oracle_mean | 0.00533 [-0.004,0.015] | 0.67 | -0.607 [-0.989,-0.254] | 0.25 |
| hotpotqa_distractor | holdout | 0.5 | la2_early | 0.392 [-0.009,0.893] | 0.5 | -0.22 [-0.725,0.245] | 0.083 |
| hotpotqa_distractor | holdout | 0.5 | la2_late | 0.0679 [-0.011,0.211] | 0.58 | -0.544 [-0.933,-0.194] | 0.17 |
| hotpotqa_distractor | holdout | 0.5 | la2_mean | 0.0655 [-0.013,0.206] | 0.67 | -0.547 [-0.936,-0.197] | 0.25 |
| hotpotqa_distractor | holdout | 0.5 | qa_early | 1.02 [0.448,1.670] | 0.17 | 0.406 [0.108,0.732] | -0.25 |
| hotpotqa_distractor | holdout | 0.5 | qa_last_layer | 0.6 [0.110,1.134] | 0.42 | -0.0118 [-0.486,0.352] | 0 |
| hotpotqa_distractor | holdout | 0.5 | qa_late | 0.733 [0.216,1.362] | 0.33 | 0.12 [-0.301,0.507] | -0.083 |
| hotpotqa_distractor | holdout | 0.5 | qa_max_layer | 1.02 [0.397,1.696] | 0.25 | 0.409 [0.055,0.768] | -0.17 |
| hotpotqa_distractor | holdout | 0.5 | qa_mean | 0.936 [0.349,1.586] | 0.33 | 0.324 [0.017,0.665] | -0.083 |
| hotpotqa_distractor | holdout | 0.5 | random_11 | 0.693 [0.219,1.291] | 0.33 | 0.0807 [-0.333,0.494] | -0.083 |
| hotpotqa_distractor | holdout | 0.5 | random_29 | 0.531 [0.033,1.133] | 0.5 | -0.0807 [-0.494,0.333] | 0.083 |
