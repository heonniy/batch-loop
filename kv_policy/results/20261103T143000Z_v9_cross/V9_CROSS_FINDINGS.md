# v9 cross-cell findings: Context-KV eviction → recurrent FFN gate → exit / answer

Cells: huginn-0125/controlled_retrieval (5/5 prompts, thresholds 0.0001, 0.00025); huginn-0125/hotpotqa_distractor (5/5 prompts, thresholds 0.0001, 0.00025); ouro-1.4b/controlled_retrieval (5/5 prompts, thresholds 0.85, 0.7); ouro-1.4b/hotpotqa_distractor (5/5 prompts, thresholds 0.85, 0.7); ouro-2.6b/controlled_retrieval (5/5 prompts, thresholds 0.95, 0.85); ouro-2.6b/hotpotqa_distractor (5/5 prompts, thresholds 0.85, 0.7)

All arms share the same LAST-plane-sharing cache (every historical recurrent read uses the terminal plane); EVICT masks a contiguous 4-token Context window for every generated query. Restorations patch ONLY the named FFN activation of the generated queries with the FULL run's value (privileged oracle). Positive 'recovered' = the patch removes eviction damage. CIs: prompt bootstrap.

## 1. Eviction penalty and BF16 floor by stratum

| model | dataset | stratum | n windows | n prompts | mean ΔNLL evict [CI] | mean KL evict | frac windows ΔNLL > 0.01 | EM full → evict |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 20 | 5 | 0.002691 [-0.00204,0.00742] | 6.863e-05 | 0.1 | 1 → 1 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 5 | 5 | 0.003067 [0.000277,0.00586] | 7.192e-05 | 0 | 1 → 1 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 5 | 5 | 0.3366 [0.00671,0.689] | 0.4029 | 0.8 | 1 → 0.6 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 20 | 5 | -0.008555 [-0.0259,0.0118] | 0.002381 | 0.15 | 0.2 → 0.2 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 5 | 5 | -0.0123 [-0.0284,0.000286] | 0.0003357 | 0 | 0.2 → 0.2 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 5 | 5 | -0.02137 [-0.0473,0.00214] | 0.002607 | 0.2 | 0.2 → 0.2 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 20 | 5 | 0.1231 [-0.0412,0.331] | 0.1326 | 0.45 | 0 → 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 5 | 5 | 0.1262 [0.00803,0.276] | 0.02232 | 0.6 | 0 → 0 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 5 | 5 | 0.1253 [-0.0332,0.368] | 0.02068 | 0.8 | 0 → 0 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 20 | 5 | 0.02565 [-0.0229,0.11] | 0.05291 | 0.45 | 0 → 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 5 | 5 | -0.008351 [-0.0492,0.025] | 0.009408 | 0.6 | 0 → 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 5 | 5 | 0.04859 [-0.0386,0.139] | 0.01622 | 0.6 | 0 → 0 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 20 | 5 | -0.01718 [-0.05,-0.00023] | 0.004227 | 0.05 | 0.8 → 0.8 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 5 | 5 | -0.01034 [-0.0308,0.000945] | 0.0009305 | 0 | 0.8 → 0.8 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 5 | 5 | 0.08107 [-0.693,0.898] | 0.5228 | 0.4 | 0.8 → 0.6 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 20 | 5 | 0.03358 [-0.00147,0.0935] | 0.007284 | 0.35 | 0.4 → 0.4 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 5 | 5 | 0.02761 [-0.00762,0.0915] | 0.005687 | 0.2 | 0.4 → 0.4 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 5 | 5 | 0.01924 [-0.0145,0.0726] | 0.004772 | 0.2 | 0.4 → 0.2 |

## 2. FFN restoration on the evicted branch: NLL recovered (positive = helps) and fraction of the penalty

| model | dataset | stratum | penalty | gate [CI] | up | shifted-gate (direction control) | full FFN | gate/penalty | ffn/penalty | gate − up [CI] | gate − shift [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 0.002691 | 0.002221 [1.59e-05,0.00463] | -0.06882 | 0.001097 | 0.001948 | 0.83 | 0.72 | 0.07104 [-0.00491,0.181] | 0.001124 [-0.00347,0.00637] |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.003067 | 0.003192 [-0.000283,0.00625] | 0.004783 | -0.0007254 | 0.0007293 | 1 | 0.24 | -0.00159 [-0.00776,0.00424] | 0.003918 [-0.00426,0.0109] |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.3366 | -0.388 [-0.986,0.0111] | -0.3725 | -0.04778 | -0.05673 | -1.2 | -0.17 | -0.01553 [-0.442,0.317] | -0.3402 [-0.801,-0.00116] |
| huginn-0125 | hotpotqa_distractor | main_position_hash | -0.008555 | -0.002595 [-0.00922,0.00307] | -0.01063 | 0.002286 | -0.007311 | — | — | 0.008034 [-0.01,0.0261] | -0.004881 [-0.0166,0.00812] |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | -0.0123 | 0.003256 [-0.00114,0.00773] | -0.005259 | -0.0064 | 0.001798 | — | — | 0.008515 [-0.00876,0.0324] | 0.009656 [-0.00275,0.0221] |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | -0.02137 | -0.0127 [-0.0402,0.00935] | -0.2994 | 0.01574 | -0.01143 | — | — | 0.2867 [-0.0152,0.873] | -0.02844 [-0.0815,0.0036] |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.1231 | 0.09282 [-0.0773,0.322] | -8.282 | -1.538 | -5.425 | 0.75 | -44 | 8.375 [4.75,13.7] | 1.631 [0.444,3.33] |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.1262 | 0.1335 [0.00109,0.288] | -8.212 | -1.797 | -5.371 | 1.1 | -43 | 8.346 [4.91,13.5] | 1.931 [0.793,3.58] |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.1253 | 0.03155 [-0.0467,0.0953] | -8.16 | -0.9718 | -5.617 | 0.25 | -45 | 8.191 [4.45,13.6] | 1.003 [-1.1,3.88] |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.02565 | 0.03125 [-0.0458,0.131] | -11.39 | -2.969 | -8.846 | 1.2 | -3.4e+02 | 11.42 [7.42,16.1] | 3.001 [1.9,4.29] |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | -0.008351 | 0.008538 [-0.021,0.0416] | -11.22 | -1.911 | -8.707 | — | — | 11.23 [7.08,16.2] | 1.919 [0.494,4.09] |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.04859 | 0.04282 [-0.0534,0.141] | -11.6 | -3.868 | -8.75 | 0.88 | -1.8e+02 | 11.65 [7.63,16.3] | 3.911 [1.83,6.09] |
| ouro-2.6b | controlled_retrieval | main_position_hash | -0.01718 | -0.01914 [-0.0574,0.000907] | -22.02 | 0.03661 | -15.5 | — | — | 22 [20,23.7] | -0.05575 [-0.166,0.000794] |
| ouro-2.6b | controlled_retrieval | oracle_distractor | -0.01034 | -0.0119 [-0.032,0.00011] | -21.63 | -0.02944 | -15.37 | — | — | 21.62 [19.3,23.5] | 0.01754 [-0.0064,0.0613] |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.08107 | -0.3279 [-0.807,0.00249] | -21.63 | -0.2015 | -15.31 | -4 | -1.9e+02 | 21.31 [19.1,23.5] | -0.1264 [-0.661,0.227] |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.03358 | 0.02718 [-0.0106,0.0972] | -18.13 | 0.01862 | -18.55 | 0.81 | -5.5e+02 | 18.16 [15.5,21.2] | 0.008559 [-0.0152,0.0381] |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.02761 | 0.03035 [-0.00894,0.0982] | -18.36 | 0.02148 | -18.44 | 1.1 | -6.7e+02 | 18.39 [15.8,21.4] | 0.008862 [-0.0102,0.0352] |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.01924 | 0.02726 [-0.0155,0.0909] | -18.31 | 0.03681 | -18.52 | 1.4 | -9.6e+02 | 18.34 [15.8,21.2] | -0.009542 [-0.0334,0.018] |

## 3. Loop-local gate restoration (NLL recovered when only the gates of the named loops are patched)

| model | dataset | stratum | 1–8 / 1–2 | 9–16 / 3–4 | 17–24 | 25–32 | all loops |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 0.0006591 | 0.001517 | 0.0007703 | 0.002603 | 0.002221 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.004342 | 0.004211 | 0.003192 | 0.005416 | 0.003192 |
| huginn-0125 | controlled_retrieval | oracle_evidence | -0.002932 | -0.01581 | -0.01286 | -0.3812 | -0.388 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.001864 | 0.00402 | 0.002179 | -0.003522 | -0.002595 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | -0.001758 | 0.00267 | -0.007372 | -0.002239 | 0.003256 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.001147 | 0.0002019 | 1.664e-05 | -0.01838 | -0.0127 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.08762 | 0.1477 | — | — | 0.09282 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.1162 | 0.1748 | — | — | 0.1335 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.007423 | 0.03519 | — | — | 0.03155 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.02713 | -0.02171 | — | — | 0.03125 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.01636 | 0.01777 | — | — | 0.008538 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.08819 | 0.06459 | — | — | 0.04282 |
| ouro-2.6b | controlled_retrieval | main_position_hash | -0.02213 | -0.0274 | — | — | -0.01914 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | -0.01679 | -0.007762 | — | — | -0.0119 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | -0.1451 | -0.2675 | — | — | -0.3279 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.0369 | 0.006787 | — | — | 0.02718 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.02636 | 0.01351 | — | — | 0.03035 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.02321 | 0.01205 | — | — | 0.02726 |

## 4. How much does eviction change the recurrent FFN gate, by loop (relative L2 of pre-SiLU gate; sign-flip fraction)

| model | dataset | stratum | loop → gate relative change / sign-flip fraction |
|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | r1: 0.00512 / 0.00153, r2: 0.00834 / 0.00269, r3: 0.00875 / 0.0029, r4: 0.00968 / 0.00326, r8: 0.0101 / 0.00339, r16: 0.00999 / 0.00336, r24: 0.00976 / 0.00326, r32: 0.00987 / 0.00331 |
| huginn-0125 | controlled_retrieval | oracle_distractor | r1: 0.00496 / 0.00146, r2: 0.00786 / 0.00253, r3: 0.00828 / 0.00278, r4: 0.00924 / 0.00308, r8: 0.00967 / 0.00323, r16: 0.00969 / 0.00325, r24: 0.00945 / 0.00316, r32: 0.00951 / 0.00318 |
| huginn-0125 | controlled_retrieval | oracle_evidence | r1: 0.0509 / 0.0161, r2: 0.0976 / 0.0302, r3: 0.107 / 0.0351, r4: 0.132 / 0.0442, r8: 0.14 / 0.0473, r16: 0.141 / 0.0473, r24: 0.14 / 0.0472, r32: 0.141 / 0.0473 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | r1: 0.00627 / 0.00189, r2: 0.0107 / 0.00331, r3: 0.0123 / 0.00391, r4: 0.0138 / 0.00442, r8: 0.0152 / 0.00494, r16: 0.0154 / 0.00497, r24: 0.0152 / 0.00489, r32: 0.0153 / 0.00494 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | r1: 0.0058 / 0.00177, r2: 0.00887 / 0.00278, r3: 0.00919 / 0.00304, r4: 0.0101 / 0.00332, r8: 0.0101 / 0.00339, r16: 0.0102 / 0.00339, r24: 0.00997 / 0.00331, r32: 0.00996 / 0.00328 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | r1: 0.00916 / 0.00275, r2: 0.0157 / 0.00487, r3: 0.0199 / 0.00644, r4: 0.0249 / 0.00818, r8: 0.0297 / 0.00987, r16: 0.03 / 0.01, r24: 0.0299 / 0.00998, r32: 0.0301 / 0.01 |
| ouro-1.4b | controlled_retrieval | main_position_hash | r1: 0.0575 / 0.02, r2: 0.0817 / 0.0277, r3: 0.0908 / 0.0306, r4: 0.1 / 0.0335 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | r1: 0.0481 / 0.0166, r2: 0.0655 / 0.022, r3: 0.0729 / 0.0245, r4: 0.0799 / 0.0265 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | r1: 0.0476 / 0.0166, r2: 0.0664 / 0.0225, r3: 0.0756 / 0.0255, r4: 0.0839 / 0.0281 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | r1: 0.054 / 0.0188, r2: 0.0802 / 0.0271, r3: 0.0932 / 0.0311, r4: 0.0962 / 0.0321 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | r1: 0.0363 / 0.0126, r2: 0.0484 / 0.0164, r3: 0.0526 / 0.0175, r4: 0.0515 / 0.0171 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | r1: 0.0537 / 0.0186, r2: 0.0688 / 0.0233, r3: 0.0748 / 0.0249, r4: 0.0737 / 0.0244 |
| ouro-2.6b | controlled_retrieval | main_position_hash | r1: 0.0255 / 0.00841, r2: 0.036 / 0.0119, r3: 0.0414 / 0.0138, r4: 0.0476 / 0.0158 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | r1: 0.0227 / 0.00749, r2: 0.0283 / 0.00933, r3: 0.0336 / 0.0111, r4: 0.0355 / 0.0117 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | r1: 0.0597 / 0.0202, r2: 0.157 / 0.053, r3: 0.192 / 0.0646, r4: 0.218 / 0.0736 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | r1: 0.0329 / 0.0108, r2: 0.0401 / 0.013, r3: 0.0409 / 0.0133, r4: 0.0388 / 0.0127 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | r1: 0.0289 / 0.00946, r2: 0.0352 / 0.0114, r3: 0.0365 / 0.0118, r4: 0.035 / 0.0114 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | r1: 0.0311 / 0.0102, r2: 0.0341 / 0.011, r3: 0.0348 / 0.0112, r4: 0.0335 / 0.0109 |

## 5. Native dynamic exit under eviction (clean-only thresholds): depth change, live vs replay vs gate-restored replay, answers

| model | dataset | stratum | threshold | frac queries depth changed | mean Σdepth Δ | live ΔNLL | replay ΔNLL | gate-replay recovered | oracle gate-live: frac exits changed vs evict-live | frac answer changed (live full vs live evict) | EM full-live → evict-live | frac live≠fixed correctness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 0.0001 | 0.691 | 1.35 | -0.0003687 | 0.0003228 | 0.001479 | 0.641 | 0 | 1 → 1 | 0 |
| huginn-0125 | controlled_retrieval | main_position_hash | 0.00025 | 0.4 | 1.55 | 0.001271 | -0.001343 | -0.0009534 | 0.368 | 0 | 1 → 1 | 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.0001 | 0.655 | 0.4 | 0.003753 | 1.608e-05 | 0.003568 | 0.527 | 0 | 1 → 1 | 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.00025 | 0.309 | 2.6 | 0.0003408 | -0.0004217 | -0.003441 | 0.4 | 0 | 1 → 1 | 0 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.0001 | 0.709 | 12.8 | 0.3277 | 0.3358 | -0.3803 | 0.673 | 0.4 | 1 → 0.6 | 0 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.00025 | 0.564 | 11.8 | 0.3316 | 0.3304 | -0.3838 | 0.509 | 0.4 | 1 → 0.6 | 0 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.0001 | 0.534 | 2.2 | -0.0002202 | 0.001827 | 0.005292 | 0.497 | 0 | 0.2 → 0.2 | 0 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.00025 | 0.24 | -0.05 | -0.01025 | -0.003613 | -0.003709 | 0.227 | 0 | 0.2 → 0.2 | 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 0.0001 | 0.698 | 3.2 | -0.002628 | -0.005365 | -3.114e-05 | 0.56 | 0 | 0.2 → 0.2 | 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 0.00025 | 0.198 | -0.8 | -0.005469 | -0.006392 | 0.005928 | 0.133 | 0 | 0.2 → 0.2 | 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.0001 | 0.477 | 1 | -0.02109 | -0.01622 | -0.01358 | 0.568 | 0 | 0.2 → 0.2 | 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.00025 | 0.347 | -1.8 | -0.02144 | -0.02518 | -0.0299 | 0.4 | 0 | 0.2 → 0.2 | 0 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.85 | 0.0182 | 0.1 | 0.1068 | 0.1287 | 0.0959 | 0.0318 | 0.7 | 0 → 0 | 0 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.7 | 0.0409 | -0.45 | 0.174 | 0.1595 | 0.1019 | 0.123 | 0.3 | 0 → 0 | 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.85 | 0.0545 | -0.2 | 0.1885 | 0.1193 | 0.1248 | 0.0727 | 0.4 | 0 → 0 | 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.7 | 0.0545 | -0.6 | 0.1768 | 0.1629 | 0.155 | 0.145 | 0.4 | 0 → 0 | 0 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.85 | 0.0364 | 0 | 0.07231 | 0.1186 | 0.01397 | 0.0545 | 0.8 | 0 → 0 | 0 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.7 | 0.0727 | -0.8 | 0.3186 | 0.3193 | -0.2016 | 0.127 | 0.4 | 0 → 0 | 0 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.85 | 0.005 | 0.05 | 0.02599 | 0.02599 | 0.02848 | 0.005 | 0.05 | 0 → 0 | 0 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.7 | 0.0675 | 0.2 | 0.0458 | 0.02476 | 0.01178 | 0.111 | 0.2 | 0 → 0 | 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.85 | 0.02 | 0.2 | 0.006817 | 0.006817 | 0.02027 | 0.02 | 0 | 0 → 0 | 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.7 | 0.025 | 0.2 | -0.04228 | -0.04299 | -0.0386 | 0.0733 | 0.2 | 0 → 0 | 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.85 | 0 | 0 | 0.07 | 0.07 | 0.0486 | 0 | 0.2 | 0 → 0 | 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.7 | 0.0583 | 0 | 0.03975 | 0.04312 | 0.04407 | 0.127 | 0 | 0 → 0 | 0 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 0.95 | 0 | 0 | -0.01672 | -0.01672 | -0.01932 | 0.00455 | 0.05 | 0.8 → 0.8 | 0 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 0.85 | 0 | 0 | -0.05707 | -0.05707 | -0.07444 | 0.0364 | 0.35 | 0.2 → 0.2 | 0.9 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 0.95 | 0 | 0 | -0.009043 | -0.009043 | -0.01102 | 0 | 0 | 0.8 → 0.8 | 0 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 0.85 | 0 | 0 | -0.3658 | -0.3658 | -0.391 | 0.0364 | 0.4 | 0.2 → 0.4 | 0.8 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.95 | 0.0182 | 0.2 | 0.07985 | -0.03885 | -0.3686 | 0.0182 | 0.2 | 0.8 → 0.6 | 0 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.85 | 0.0364 | 0 | 0.574 | 0.2547 | 0.1147 | 0.0909 | 1 | 0.2 → 0 | 0.6 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.85 | 0.01 | 0.05 | 0.009036 | 0.008384 | 0.007665 | 0.01 | 0.2 | 0.2 → 0.2 | 0.2 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.7 | 0.01 | 0.05 | 0.009687 | 0.01025 | 0.007085 | 0.05 | 0 | 0.6 → 0.6 | 0.2 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.85 | 0 | 0 | -0.0006836 | -0.0006836 | 0.00178 | 0 | 0.2 | 0.2 → 0.2 | 0.2 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.7 | 0 | 0 | 0.01049 | 0.01049 | 0.008151 | 0.04 | 0 | 0.6 → 0.6 | 0.2 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.85 | 0 | 0 | 0.009939 | 0.009939 | 0.02089 | 0 | 0.2 | 0.2 → 0.2 | 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.7 | 0 | 0 | 0.006425 | 0.006425 | 0.014 | 0.04 | 0 | 0.6 → 0.6 | 0.4 |
