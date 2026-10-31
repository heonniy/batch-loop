# LoopZip v6 — persistent eviction and recurrent trajectories (observations)

complete=True; jobs 291; missing 0; errors 0; problems 0.

Full-sized reference KV + read masks; no HBM/latency claims. Intervals: bootstrap over prompts.

## Stage A — normal trajectories (no eviction)

Mean U_t(r) by model / workload / phase at relative depth (all loops kept in the parquet).

| model | workload | phase | n tok | U(r=1) | U(mid) | U(R) | F(r=1) | C(mid) |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | W1 | generated | 3182 | 1.8123 | 0.0618 | 0.0154 | 1.1878 | 0.160 |
| huginn-0125 | W1 | prompt | 6527 | 1.8158 | 0.0548 | 0.0141 | 1.1322 | 0.262 |
| huginn-0125 | W2 | generated | 3788 | 1.8098 | 0.0832 | 0.0235 | 1.2098 | 0.016 |
| huginn-0125 | W2 | prompt | 6752 | 1.8151 | 0.0571 | 0.0145 | 1.1324 | 0.314 |
| ouro-2.6b | W1 | generated | 5120 | 52.5395 | 0.6297 | 0.1534 | 0.6389 | 0.248 |
| ouro-2.6b | W1 | prompt | 5679 | 43.6169 | 0.8534 | 0.1707 | 0.6861 | 0.196 |
| ouro-2.6b | W2 | generated | 5120 | 52.4116 | 0.5950 | 0.1855 | 0.7094 | 0.247 |
| ouro-2.6b | W2 | prompt | 5977 | 41.6097 | 0.9056 | 0.1825 | 0.7112 | 0.176 |

| model | role | n tok | U(r=1) | U(R) | F(r=1) |
|---|---|---|---|---|---|
| huginn-0125 | content | 9628 | 1.8090 | 0.0143 | 1.1628 |
| huginn-0125 | generated | 6970 | 1.8110 | 0.0198 | 1.1997 |
| huginn-0125 | instruction | 856 | 1.8125 | 0.0184 | 1.1766 |
| huginn-0125 | other | 2672 | 1.8400 | 0.0129 | 1.0059 |
| huginn-0125 | question | 123 | 1.8086 | 0.0187 | 1.1888 |
| ouro-2.6b | content | 9564 | 42.6312 | 0.1591 | 0.6772 |
| ouro-2.6b | generated | 10240 | 52.4756 | 0.1695 | 0.6742 |
| ouro-2.6b | instruction | 890 | 31.0576 | 0.2879 | 0.7702 |
| ouro-2.6b | other | 1080 | 52.6424 | 0.2354 | 0.8190 |
| ouro-2.6b | question | 122 | 34.2786 | 0.2336 | 0.8296 |

## Stage B — exhaustive one-source persistent eviction (accumulated arm)

Per prompt: mean over ALL eligible sources; then mean [95% CI] over prompts.

| model | workload | prompts | sources | KL early | KL medium | KL late | E(t,R) late | top1 late |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | W1 | 4 | 558 | 0.006525 [0.00145,0.0116] | 0.002741 [0.000802,0.00468] | 0.001351 | 0.01993 | 0.008253 |
| huginn-0125 | W2 | 4 | 504 | 0.01361 [0.00916,0.0177] | 0.01089 [0.00829,0.014] | 0.01404 [0.00272,0.0259] | 0.03973 [0.0269,0.0556] | 0.006341 [0.00321,0.012] |
| ouro-2.6b | W1 | 4 | 548 | 0.008082 [0.00251,0.0139] | 0.01103 [0.00181,0.0243] | 0.008462 [0.00371,0.0145] | 0.05489 [0.0252,0.11] | 0.01194 [0.00463,0.0213] |
| ouro-2.6b | W2 | 4 | 495 | 0.009269 [0.00359,0.0189] | 0.01313 [0.00974,0.0182] | 0.01539 [0.00902,0.0218] | 0.08486 [0.0454,0.124] | 0.00597 [0.00256,0.0105] |

Within-prompt Spearman(source feature, mean KL), averaged over prompts (descriptive only):

| model | workload | position | attn recv | attn last-query | src U mean | CV of KL |
|---|---|---|---|---|---|---|
| huginn-0125 | W1 | -0.264 | 0.219 | 0.343 | -0.114 | 2.912 |
| huginn-0125 | W2 | -0.175 | 0.092 | 0.032 | -0.179 | 2.120 |
| ouro-2.6b | W1 | -0.054 | 0.135 | 0.167 | -0.186 | 2.501 |
| ouro-2.6b | W2 | 0.038 | 0.117 | 0.048 | -0.362 | 1.771 |

## Stage C — accumulated vs clean-history (H = E_accum − E_clean at r = R)

| model | t | mean E_accum | mean E_clean | mean H | KL accum | KL clean |
|---|---|---|---|---|---|---|
| huginn-0125 | 1 | 0.02084 | 0.02084 | 0 | 0.003723 | 0.003723 |
| huginn-0125 | 2 | 0.02277 | 0.02192 | 0.0008471 | 0.02623 | 0.02434 |
| huginn-0125 | 4 | 0.0237 | 0.0208 | 0.002907 | 0.004566 | 0.003966 |
| huginn-0125 | 8 | 0.02518 | 0.02092 | 0.004254 | 0.004114 | 0.003571 |
| huginn-0125 | 16 | 0.02687 | 0.01965 | 0.007223 | 0.002437 | 0.00164 |
| huginn-0125 | 32 | 0.03725 | 0.02315 | 0.0141 | 0.008682 | 0.005972 |
| huginn-0125 | 64 | 0.03915 | 0.02072 | 0.01843 | 0.01958 | 0.01527 |
| huginn-0125 | 128 | 0.01853 | 0.01302 | 0.005514 | 0.0005219 | 8.501e-05 |
| ouro-2.6b | 1 | 0.02205 | 0.02205 | 0 | 0.00308 | 0.00308 |
| ouro-2.6b | 2 | 0.02725 | 0.02577 | 0.001473 | 0.006248 | 0.00598 |
| ouro-2.6b | 4 | 0.02322 | 0.02061 | 0.002613 | 0.006327 | 0.006067 |
| ouro-2.6b | 8 | 0.02736 | 0.02261 | 0.004749 | 0.02777 | 0.02594 |
| ouro-2.6b | 16 | 0.06492 | 0.04047 | 0.02445 | 0.0005309 | 0.0002544 |
| ouro-2.6b | 32 | 0.04153 | 0.02092 | 0.02062 | 0.01731 | 0.01292 |
| ouro-2.6b | 64 | 0.0521 | 0.022 | 0.0301 | 0.006609 | 0.004223 |
| ouro-2.6b | 128 | 0.05477 | 0.01808 | 0.03669 | 0.02438 | 0.004949 |

Descriptive census of a 4-prompt/model/workload pilot; not a powered generalization test. Correlations with position/attention are NOT evidence of recurrence-specific importance.
