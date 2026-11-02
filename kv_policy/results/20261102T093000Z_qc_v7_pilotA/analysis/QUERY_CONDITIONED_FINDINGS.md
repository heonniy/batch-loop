# v7 query-conditioned sensitivity — findings (20261102T093000Z_qc_v7_pilotA)

Completeness: 466/466 jobs, errors 0, problems 0, complete=True

Sources in table: 14943 over 48 units; per model/dataset: huginn-0125/controlled_retrieval: 2816 sources / 8 prompts, huginn-0125/hotpotqa_distractor: 3677 sources / 8 prompts, huginn-0125/v6_W1: 554 sources / 4 prompts, huginn-0125/v6_W2: 500 sources / 4 prompts, ouro-2.6b/controlled_retrieval: 2816 sources / 8 prompts, ouro-2.6b/hotpotqa_distractor: 3545 sources / 8 prompts, ouro-2.6b/v6_W1: 544 sources / 4 prompts, ouro-2.6b/v6_W2: 491 sources / 4 prompts

## Outcome structure: heavy tail and evidence labels (post-hoc stratification only)

| model | dataset | n sources | share of total KL in top 5% | mean KL evidence | mean KL non-evidence | evidence share among top-5% | mean ΔNLL_future evidence |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 2816 | 0.988 | 0.115 | 0.000116 | 0.454 | 0.106 |
| huginn-0125 | hotpotqa_distractor | 3677 | 0.794 | 0.00462 | 0.000531 | 0.565 | -0.00507 |
| ouro-2.6b | controlled_retrieval | 2816 | 0.984 | 0.107 | 0.000159 | 0.34 | 0.119 |
| ouro-2.6b | hotpotqa_distractor | 3545 | 0.826 | 0.00407 | 0.000307 | 0.644 | 0.00384 |

## Query-conditioned probe magnitudes vs the numerical floor

Sources binned by clean question-anchor attention (within prompt deciles). The lowest decile serves as the empirical 'near-zero perturbation' floor: in BF16 any mask changes rounding, so hidden-state E has a floor even when the removed attention mass is ~0; the question is whether higher-attention sources rise above it.

| model | dataset | attention decile | n | median qa all-loop KL | median E_R | median ‖Δh_R‖ | median top-1 flip |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0 | 288 | 7.26e-06 | 0.014 | 1.07 | 0 |
| huginn-0125 | controlled_retrieval | 1 | 280 | 1.45e-05 | 0.0141 | 1.08 | 0 |
| huginn-0125 | controlled_retrieval | 2 | 280 | 1.19e-05 | 0.0141 | 1.07 | 0 |
| huginn-0125 | controlled_retrieval | 3 | 280 | 1.52e-05 | 0.0141 | 1.08 | 0 |
| huginn-0125 | controlled_retrieval | 4 | 280 | 1.45e-05 | 0.0141 | 1.07 | 0 |
| huginn-0125 | controlled_retrieval | 5 | 280 | 6.72e-06 | 0.014 | 1.07 | 0 |
| huginn-0125 | controlled_retrieval | 6 | 280 | 1.53e-05 | 0.0141 | 1.08 | 0 |
| huginn-0125 | controlled_retrieval | 7 | 280 | 6.46e-06 | 0.0142 | 1.08 | 0 |
| huginn-0125 | controlled_retrieval | 8 | 280 | 6.38e-06 | 0.0142 | 1.08 | 0 |
| huginn-0125 | controlled_retrieval | 9 | 288 | 7.41e-06 | 0.0145 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 0 | 372 | 1.48e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 1 | 367 | 1.78e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 2 | 365 | 1.41e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 3 | 367 | 1.51e-05 | 0.0142 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 4 | 371 | 1.64e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 5 | 365 | 1.45e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 6 | 367 | 1.7e-05 | 0.0144 | 1.1 | 0 |
| huginn-0125 | hotpotqa_distractor | 7 | 365 | 1.5e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | hotpotqa_distractor | 8 | 367 | 1.84e-05 | 0.0144 | 1.1 | 0 |
| huginn-0125 | hotpotqa_distractor | 9 | 371 | 2.19e-05 | 0.0151 | 1.15 | 0 |
| ouro-2.6b | controlled_retrieval | 0 | 288 | 3.12e-05 | 0.0129 | 0.621 | 0 |
| ouro-2.6b | controlled_retrieval | 1 | 280 | 3.17e-05 | 0.0129 | 0.625 | 0 |
| ouro-2.6b | controlled_retrieval | 2 | 280 | 3.12e-05 | 0.0129 | 0.624 | 0 |
| ouro-2.6b | controlled_retrieval | 3 | 280 | 3.32e-05 | 0.0127 | 0.615 | 0 |
| ouro-2.6b | controlled_retrieval | 4 | 280 | 3.32e-05 | 0.0129 | 0.623 | 0 |
| ouro-2.6b | controlled_retrieval | 5 | 280 | 3.32e-05 | 0.0129 | 0.622 | 0 |
| ouro-2.6b | controlled_retrieval | 6 | 280 | 3.15e-05 | 0.0129 | 0.624 | 0 |
| ouro-2.6b | controlled_retrieval | 7 | 280 | 2.89e-05 | 0.0129 | 0.621 | 0 |
| ouro-2.6b | controlled_retrieval | 8 | 280 | 3.12e-05 | 0.013 | 0.63 | 0 |
| ouro-2.6b | controlled_retrieval | 9 | 288 | 3.74e-05 | 0.0135 | 0.651 | 0 |
| ouro-2.6b | hotpotqa_distractor | 0 | 358 | 9.12e-05 | 0.014 | 0.652 | 0 |
| ouro-2.6b | hotpotqa_distractor | 1 | 356 | 9.06e-05 | 0.014 | 0.644 | 0 |
| ouro-2.6b | hotpotqa_distractor | 2 | 352 | 9.08e-05 | 0.0138 | 0.643 | 0 |
| ouro-2.6b | hotpotqa_distractor | 3 | 356 | 9.8e-05 | 0.0139 | 0.643 | 0 |
| ouro-2.6b | hotpotqa_distractor | 4 | 353 | 8.78e-05 | 0.014 | 0.648 | 0 |
| ouro-2.6b | hotpotqa_distractor | 5 | 353 | 8.68e-05 | 0.014 | 0.651 | 0 |
| ouro-2.6b | hotpotqa_distractor | 6 | 353 | 9.63e-05 | 0.014 | 0.644 | 0 |
| ouro-2.6b | hotpotqa_distractor | 7 | 355 | 9.55e-05 | 0.0141 | 0.653 | 0 |
| ouro-2.6b | hotpotqa_distractor | 8 | 353 | 9.92e-05 | 0.0144 | 0.668 | 0 |
| ouro-2.6b | hotpotqa_distractor | 9 | 356 | 0.000105 | 0.0146 | 0.672 | 0 |

## One-loop query masks: propagation gain E_R / E_s by masked loop s (median over sources; IQR)

| model | dataset | anchor | s | n | median E_s | median E_R | median gain | IQR gain | median KL | p90 KL |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | question anchor | 1 | 2816 | 0.00694 | 0.014 | 2.03 | [1.92,2.13] | 7.73e-06 | 0.000375 |
| huginn-0125 | controlled_retrieval | question anchor | 2 | 2816 | 0.00685 | 0.014 | 2.04 | [1.93,2.15] | 7.75e-06 | 0.000374 |
| huginn-0125 | controlled_retrieval | question anchor | 4 | 2816 | 0.00671 | 0.0141 | 2.1 | [1.97,2.19] | 1.33e-05 | 0.000375 |
| huginn-0125 | controlled_retrieval | question anchor | 8 | 2816 | 0.00652 | 0.014 | 2.15 | [2.04,2.26] | 1.44e-05 | 0.00038 |
| huginn-0125 | controlled_retrieval | question anchor | 16 | 2816 | 0.00639 | 0.014 | 2.2 | [2.1,2.31] | 1.25e-05 | 0.000378 |
| huginn-0125 | controlled_retrieval | question anchor | 24 | 2816 | 0.00644 | 0.0139 | 2.16 | [2.04,2.26] | 1.21e-05 | 0.000381 |
| huginn-0125 | controlled_retrieval | question anchor | 32 | 2816 | 0.00624 | 0.00624 | 1 | [1,1] | 1.61e-06 | 2e-05 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 1 | 2816 | 0.0074 | 0.0141 | 1.9 | [1.7,2.07] | 2.09e-06 | 0.000222 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 2 | 2816 | 0.00729 | 0.0141 | 1.92 | [1.63,2.1] | 2.02e-06 | 0.000226 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 4 | 2816 | 0.00769 | 0.0141 | 1.82 | [1.47,2.05] | 1.97e-06 | 0.000213 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 8 | 2816 | 0.00704 | 0.0141 | 1.98 | [1.67,2.2] | 2.03e-06 | 0.000235 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 16 | 2816 | 0.00715 | 0.0141 | 1.96 | [1.63,2.18] | 1.86e-06 | 0.000209 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 24 | 2816 | 0.00714 | 0.0141 | 1.95 | [1.65,2.15] | 2.18e-06 | 0.000207 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 32 | 2816 | 0.0071 | 0.0071 | 1 | [1,1] | 1.33e-06 | 0.000149 |
| huginn-0125 | hotpotqa_distractor | question anchor | 1 | 3677 | 0.00656 | 0.0142 | 2.16 | [2.02,2.29] | 1.61e-05 | 0.000514 |
| huginn-0125 | hotpotqa_distractor | question anchor | 2 | 3677 | 0.00691 | 0.0142 | 2.05 | [1.94,2.17] | 1.5e-05 | 0.000513 |
| huginn-0125 | hotpotqa_distractor | question anchor | 4 | 3677 | 0.00671 | 0.0143 | 2.13 | [2,2.24] | 1.47e-05 | 0.000513 |
| huginn-0125 | hotpotqa_distractor | question anchor | 8 | 3677 | 0.00652 | 0.0142 | 2.17 | [2.05,2.31] | 1.59e-05 | 0.000516 |
| huginn-0125 | hotpotqa_distractor | question anchor | 16 | 3677 | 0.0064 | 0.0142 | 2.22 | [2.1,2.33] | 1.54e-05 | 0.000516 |
| huginn-0125 | hotpotqa_distractor | question anchor | 24 | 3677 | 0.00632 | 0.014 | 2.22 | [2.09,2.33] | 1.66e-05 | 0.000528 |
| huginn-0125 | hotpotqa_distractor | question anchor | 32 | 3677 | 0.00636 | 0.00636 | 1 | [1,1] | 4.19e-06 | 0.000422 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 1 | 3677 | 0.00681 | 0.0129 | 1.91 | [1.74,2.08] | 3.7e-05 | 0.00101 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 2 | 3677 | 0.00673 | 0.0129 | 1.92 | [1.77,2.06] | 3.86e-05 | 0.00118 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 4 | 3677 | 0.00715 | 0.0129 | 1.82 | [1.59,1.99] | 3.88e-05 | 0.00116 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 8 | 3677 | 0.0068 | 0.013 | 1.91 | [1.72,2.07] | 4.03e-05 | 0.00121 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 16 | 3677 | 0.00671 | 0.0129 | 1.92 | [1.73,2.08] | 3.86e-05 | 0.00119 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 24 | 3677 | 0.00673 | 0.0129 | 1.9 | [1.71,2.06] | 3.99e-05 | 0.00128 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 32 | 3677 | 0.00674 | 0.00674 | 1 | [1,1] | 2.82e-05 | 0.000838 |
| ouro-2.6b | controlled_retrieval | question anchor | 1 | 2816 | 0.0111 | 0.0129 | 1.16 | [1.07,1.23] | 3.19e-05 | 0.000292 |
| ouro-2.6b | controlled_retrieval | question anchor | 2 | 2816 | 0.0116 | 0.0128 | 1.1 | [1.04,1.17] | 3.25e-05 | 0.000324 |
| ouro-2.6b | controlled_retrieval | question anchor | 3 | 2816 | 0.0119 | 0.0124 | 1.04 | [0.984,1.1] | 3.2e-05 | 0.000284 |
| ouro-2.6b | controlled_retrieval | question anchor | 4 | 2816 | 0.0114 | 0.0114 | 1 | [1,1] | 3.41e-05 | 0.000332 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 1 | 2816 | 0.0105 | 0.0141 | 1.31 | [1.04,2] | 5.68e-06 | 9.37e-05 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 2 | 2816 | 0.0122 | 0.0143 | 1.17 | [0.985,1.48] | 5.42e-06 | 8.96e-05 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 3 | 2816 | 0.013 | 0.0137 | 1.07 | [0.958,1.2] | 4.94e-06 | 8.29e-05 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 4 | 2816 | 0.0127 | 0.0127 | 1 | [1,1] | 3.99e-06 | 9.2e-05 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 1 | 3545 | 0.0117 | 0.0138 | 1.16 | [0.981,1.3] | 8.69e-05 | 0.000428 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 2 | 3545 | 0.0123 | 0.0138 | 1.11 | [1.01,1.2] | 0.000103 | 0.000577 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 3 | 3545 | 0.0126 | 0.0134 | 1.06 | [0.999,1.12] | 7.42e-05 | 0.000426 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 4 | 3545 | 0.0125 | 0.0125 | 1 | [1,1] | 5.82e-05 | 0.000561 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 1 | 3545 | 0.0145 | 0.0141 | 0.978 | [0.873,1.09] | 5.84e-05 | 0.000498 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 2 | 3545 | 0.0132 | 0.0139 | 1.05 | [0.931,1.16] | 5.97e-05 | 0.000508 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 3 | 3545 | 0.0126 | 0.0135 | 1.07 | [0.976,1.15] | 4.74e-05 | 0.000491 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 4 | 3545 | 0.0125 | 0.0125 | 1 | [1,1] | 4.46e-05 | 0.000483 |

## Within-prompt Spearman of single features with future damage (mean over prompts [95% CI])

| model | dataset | feature | ρ(mean KL) | ρ(mean NLL shift) |
|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | qa_attn_mean | 0.118 [0.078,0.153] | -0.0349 [-0.080,0.004] |
| huginn-0125 | controlled_retrieval | qa_attn_last | 0.119 [0.076,0.154] | -0.0313 [-0.074,0.007] |
| huginn-0125 | controlled_retrieval | qa_attn_maxhead | 0.0873 [0.062,0.112] | -0.0329 [-0.081,0.006] |
| huginn-0125 | controlled_retrieval | recv_mean | 0.0928 [0.058,0.123] | -0.0269 [-0.062,0.002] |
| huginn-0125 | controlled_retrieval | lastq_mean | 0.175 [0.131,0.215] | -0.026 [-0.064,0.003] |
| huginn-0125 | controlled_retrieval | qb_attn_mean | 0.175 [0.131,0.215] | -0.026 [-0.064,0.003] |
| huginn-0125 | controlled_retrieval | qa_delta_mean | 0.119 [0.077,0.155] | -0.035 [-0.080,0.003] |
| huginn-0125 | controlled_retrieval | qa_delta_last | 0.119 [0.076,0.154] | -0.0328 [-0.076,0.005] |
| huginn-0125 | controlled_retrieval | qa_delta_loopw | 0.121 [0.080,0.156] | -0.0349 [-0.080,0.003] |
| huginn-0125 | controlled_retrieval | qa_vnorm_mean | -0.0191 [-0.072,0.027] | 0.0122 [-0.016,0.045] |
| huginn-0125 | controlled_retrieval | src_U_mean | -0.0504 [-0.077,-0.017] | 0.0111 [-0.041,0.058] |
| huginn-0125 | controlled_retrieval | src_settle | 0.00794 [-0.066,0.076] | -0.0318 [-0.070,0.008] |
| huginn-0125 | controlled_retrieval | fut_attn_mean | 0.192 [0.151,0.228] | 0.0311 [0.002,0.059] |
| huginn-0125 | controlled_retrieval | fut_attn_max_t | 0.202 [0.166,0.234] | 0.0415 [0.011,0.072] |
| huginn-0125 | controlled_retrieval | fut_delta_mean | 0.195 [0.154,0.230] | 0.0274 [-0.002,0.055] |
| huginn-0125 | controlled_retrieval | fut_delta_max_t | 0.21 [0.174,0.239] | 0.0388 [0.010,0.068] |
| huginn-0125 | controlled_retrieval | fut_attn_at_t | 0.202 [0.166,0.234] | 0.0415 [0.011,0.072] |
| huginn-0125 | controlled_retrieval | fut_delta_at_t | 0.21 [0.175,0.239] | 0.0387 [0.010,0.068] |
| huginn-0125 | controlled_retrieval | qa_all_kl | -0.0161 [-0.051,0.016] | 0.00684 [-0.030,0.047] |
| huginn-0125 | controlled_retrieval | qa_all_E_R | 0.0381 [0.017,0.065] | 0.00111 [-0.034,0.031] |
| huginn-0125 | controlled_retrieval | qa_all_dn_R | 0.0381 [0.017,0.065] | 0.00111 [-0.034,0.031] |
| huginn-0125 | controlled_retrieval | qb_all_kl | 0.0697 [0.034,0.102] | -0.0261 [-0.054,0.010] |
| huginn-0125 | controlled_retrieval | qb_all_E_R | 0.11 [0.080,0.141] | -0.0249 [-0.078,0.028] |
| huginn-0125 | controlled_retrieval | qf_all_kl | 0.169 [0.130,0.212] | 0.0313 [0.004,0.061] |
| huginn-0125 | controlled_retrieval | qf_all_E_R | 0.225 [0.173,0.266] | 0.0455 [0.019,0.071] |
| huginn-0125 | controlled_retrieval | qf_all_dn_R | 0.225 [0.173,0.266] | 0.0456 [0.019,0.071] |
| huginn-0125 | controlled_retrieval | qf_loop1_gain | -0.0379 [-0.089,0.005] | -0.0336 [-0.079,0.011] |
| huginn-0125 | controlled_retrieval | qa_loop1_gain | -0.00949 [-0.054,0.038] | 0.0245 [-0.010,0.060] |
| huginn-0125 | controlled_retrieval | clean_kl_mean | 0.156 [0.121,0.191] | 0.0545 [0.022,0.086] |
| huginn-0125 | controlled_retrieval | la1_attn | 0.15 [0.103,0.189] | -0.0207 [-0.059,0.009] |
| huginn-0125 | controlled_retrieval | la2_attn | 0.188 [0.133,0.237] | 0.000676 [-0.030,0.028] |
| huginn-0125 | controlled_retrieval | la12_attn_max | 0.196 [0.140,0.243] | 0.0034 [-0.026,0.031] |
| huginn-0125 | controlled_retrieval | la12_delta_max | 0.198 [0.142,0.244] | 0.00294 [-0.027,0.031] |
| huginn-0125 | controlled_retrieval | fut_lastloop_frac | 0.114 [0.071,0.154] | 0.0356 [0.008,0.063] |
| huginn-0125 | controlled_retrieval | fut_loop_entropy | 0.0292 [-0.009,0.065] | 0.0343 [0.001,0.066] |
| huginn-0125 | controlled_retrieval | qa_lastloop_frac | 0.0479 [0.001,0.092] | -0.0114 [-0.041,0.017] |
| huginn-0125 | hotpotqa_distractor | qa_attn_mean | 0.12 [0.062,0.174] | 0.0421 [0.015,0.064] |
| huginn-0125 | hotpotqa_distractor | qa_attn_last | 0.121 [0.060,0.178] | 0.0366 [0.013,0.055] |
| huginn-0125 | hotpotqa_distractor | qa_attn_maxhead | 0.0777 [0.031,0.121] | 0.0179 [-0.005,0.039] |
| huginn-0125 | hotpotqa_distractor | recv_mean | 0.0623 [0.008,0.109] | 0.0128 [-0.013,0.044] |
| huginn-0125 | hotpotqa_distractor | lastq_mean | 0.197 [0.116,0.268] | 0.055 [0.025,0.085] |
| huginn-0125 | hotpotqa_distractor | qb_attn_mean | 0.197 [0.116,0.268] | 0.055 [0.025,0.085] |
| huginn-0125 | hotpotqa_distractor | qa_delta_mean | 0.126 [0.064,0.181] | 0.0418 [0.015,0.063] |
| huginn-0125 | hotpotqa_distractor | qa_delta_last | 0.13 [0.068,0.186] | 0.0366 [0.012,0.056] |
| huginn-0125 | hotpotqa_distractor | qa_delta_loopw | 0.125 [0.064,0.178] | 0.043 [0.015,0.065] |
| huginn-0125 | hotpotqa_distractor | qa_vnorm_mean | -0.0459 [-0.087,-0.010] | -0.00568 [-0.039,0.026] |
| huginn-0125 | hotpotqa_distractor | src_U_mean | -0.0495 [-0.094,-0.007] | 0.00445 [-0.039,0.046] |
| huginn-0125 | hotpotqa_distractor | src_settle | 0.026 [-0.042,0.094] | 0.00772 [-0.026,0.037] |
| huginn-0125 | hotpotqa_distractor | fut_attn_mean | 0.231 [0.151,0.298] | 0.0432 [0.022,0.065] |
| huginn-0125 | hotpotqa_distractor | fut_attn_max_t | 0.228 [0.152,0.292] | 0.0399 [0.019,0.061] |
| huginn-0125 | hotpotqa_distractor | fut_delta_mean | 0.233 [0.154,0.301] | 0.0435 [0.023,0.065] |
| huginn-0125 | hotpotqa_distractor | fut_delta_max_t | 0.233 [0.156,0.298] | 0.0402 [0.020,0.061] |
| huginn-0125 | hotpotqa_distractor | fut_attn_at_t | 0.228 [0.152,0.292] | 0.0399 [0.019,0.061] |
| huginn-0125 | hotpotqa_distractor | fut_delta_at_t | 0.232 [0.156,0.298] | 0.04 [0.019,0.061] |
| huginn-0125 | hotpotqa_distractor | qa_all_kl | 0.00426 [-0.028,0.035] | -0.0177 [-0.040,0.004] |
| huginn-0125 | hotpotqa_distractor | qa_all_E_R | 0.0552 [0.028,0.082] | 0.047 [0.000,0.093] |
| huginn-0125 | hotpotqa_distractor | qa_all_dn_R | 0.0552 [0.028,0.082] | 0.047 [0.000,0.093] |
| huginn-0125 | hotpotqa_distractor | qb_all_kl | 0.0925 [0.033,0.151] | 0.0297 [0.008,0.053] |
| huginn-0125 | hotpotqa_distractor | qb_all_E_R | 0.118 [0.054,0.175] | 0.0175 [-0.028,0.061] |
| huginn-0125 | hotpotqa_distractor | qf_all_kl | 0.175 [0.119,0.225] | 0.0457 [0.021,0.072] |
| huginn-0125 | hotpotqa_distractor | qf_all_E_R | 0.24 [0.168,0.298] | 0.0346 [0.015,0.056] |
| huginn-0125 | hotpotqa_distractor | qf_all_dn_R | 0.24 [0.168,0.298] | 0.0346 [0.015,0.056] |
| huginn-0125 | hotpotqa_distractor | qf_loop1_gain | -0.0167 [-0.035,0.001] | -0.0167 [-0.052,0.015] |
| huginn-0125 | hotpotqa_distractor | qa_loop1_gain | 0.0252 [-0.004,0.052] | -0.0072 [-0.032,0.016] |
| huginn-0125 | hotpotqa_distractor | clean_kl_mean | 0.202 [0.140,0.259] | 0.0172 [-0.020,0.053] |
| huginn-0125 | hotpotqa_distractor | la1_attn | 0.108 [0.056,0.158] | 0.0449 [0.009,0.078] |
| huginn-0125 | hotpotqa_distractor | la2_attn | 0.234 [0.149,0.305] | 0.0567 [0.034,0.080] |
| huginn-0125 | hotpotqa_distractor | la12_attn_max | 0.235 [0.150,0.305] | 0.0579 [0.035,0.082] |
| huginn-0125 | hotpotqa_distractor | la12_delta_max | 0.238 [0.152,0.308] | 0.058 [0.036,0.081] |
| huginn-0125 | hotpotqa_distractor | fut_lastloop_frac | 0.123 [0.087,0.167] | 0.00735 [-0.016,0.032] |
| huginn-0125 | hotpotqa_distractor | fut_loop_entropy | -0.0886 [-0.132,-0.050] | -0.0125 [-0.055,0.026] |
| huginn-0125 | hotpotqa_distractor | qa_lastloop_frac | 0.0668 [0.003,0.124] | 0.00244 [-0.015,0.021] |
| ouro-2.6b | controlled_retrieval | qa_attn_mean | 0.0949 [0.078,0.119] | -0.0295 [-0.061,-0.005] |
| ouro-2.6b | controlled_retrieval | qa_attn_last | 0.0846 [0.067,0.103] | -0.0236 [-0.057,0.004] |
| ouro-2.6b | controlled_retrieval | qa_attn_maxhead | 0.0672 [0.047,0.090] | -0.0315 [-0.057,-0.011] |
| ouro-2.6b | controlled_retrieval | recv_mean | 0.0993 [0.075,0.125] | -0.0281 [-0.048,-0.002] |
| ouro-2.6b | controlled_retrieval | lastq_mean | 0.105 [0.086,0.127] | -0.0296 [-0.061,-0.000] |
| ouro-2.6b | controlled_retrieval | qb_attn_mean | 0.105 [0.086,0.127] | -0.0296 [-0.061,-0.000] |
| ouro-2.6b | controlled_retrieval | qa_delta_mean | 0.101 [0.086,0.123] | -0.0293 [-0.059,-0.005] |
| ouro-2.6b | controlled_retrieval | qa_delta_last | 0.09 [0.073,0.106] | -0.0231 [-0.055,0.004] |
| ouro-2.6b | controlled_retrieval | qa_delta_loopw | 0.101 [0.083,0.127] | -0.0324 [-0.061,-0.009] |
| ouro-2.6b | controlled_retrieval | qa_vnorm_mean | 0.07 [0.041,0.093] | -0.0311 [-0.053,-0.013] |
| ouro-2.6b | controlled_retrieval | src_U_mean | -0.0324 [-0.054,-0.012] | -0.0373 [-0.070,-0.010] |
| ouro-2.6b | controlled_retrieval | src_settle | -0.0158 [-0.052,0.016] | 0.000667 [-0.034,0.033] |
| ouro-2.6b | controlled_retrieval | fut_attn_mean | 0.155 [0.134,0.178] | -0.00885 [-0.052,0.021] |
| ouro-2.6b | controlled_retrieval | fut_attn_max_t | 0.147 [0.129,0.168] | -0.00218 [-0.045,0.032] |
| ouro-2.6b | controlled_retrieval | fut_delta_mean | 0.161 [0.136,0.188] | -0.00699 [-0.050,0.023] |
| ouro-2.6b | controlled_retrieval | fut_delta_max_t | 0.156 [0.133,0.181] | -9.9e-05 [-0.044,0.033] |
| ouro-2.6b | controlled_retrieval | fut_attn_at_t | 0.147 [0.129,0.168] | -0.00218 [-0.045,0.032] |
| ouro-2.6b | controlled_retrieval | fut_delta_at_t | 0.157 [0.133,0.182] | 0.000971 [-0.043,0.034] |
| ouro-2.6b | controlled_retrieval | qa_all_kl | 0.00611 [-0.027,0.037] | -0.00462 [-0.026,0.020] |
| ouro-2.6b | controlled_retrieval | qa_all_E_R | 0.0478 [0.012,0.082] | -0.00273 [-0.029,0.029] |
| ouro-2.6b | controlled_retrieval | qa_all_dn_R | 0.0478 [0.012,0.082] | -0.00273 [-0.029,0.029] |
| ouro-2.6b | controlled_retrieval | qb_all_kl | 0.0127 [-0.024,0.050] | 0.0251 [-0.001,0.050] |
| ouro-2.6b | controlled_retrieval | qb_all_E_R | 0.0808 [0.026,0.140] | -0.00184 [-0.030,0.029] |
| ouro-2.6b | controlled_retrieval | qf_all_kl | 0.107 [0.070,0.139] | 0.0285 [-0.009,0.064] |
| ouro-2.6b | controlled_retrieval | qf_all_E_R | 0.153 [0.104,0.200] | 0.0481 [0.015,0.079] |
| ouro-2.6b | controlled_retrieval | qf_all_dn_R | 0.164 [0.118,0.206] | 0.0542 [0.024,0.080] |
| ouro-2.6b | controlled_retrieval | qf_loop1_gain | -0.0954 [-0.141,-0.051] | 0.00278 [-0.046,0.051] |
| ouro-2.6b | controlled_retrieval | qa_loop1_gain | -0.0512 [-0.075,-0.025] | 0.0132 [-0.029,0.055] |
| ouro-2.6b | controlled_retrieval | clean_kl_mean | 0.147 [0.121,0.171] | 0.0397 [0.021,0.057] |
| ouro-2.6b | controlled_retrieval | la1_attn | 0.114 [0.097,0.134] | -0.0259 [-0.057,-0.002] |
| ouro-2.6b | controlled_retrieval | la2_attn | 0.146 [0.136,0.157] | -0.000688 [-0.036,0.026] |
| ouro-2.6b | controlled_retrieval | la12_attn_max | 0.148 [0.137,0.159] | -0.000723 [-0.036,0.026] |
| ouro-2.6b | controlled_retrieval | la12_delta_max | 0.154 [0.142,0.169] | 0.00108 [-0.034,0.027] |
| ouro-2.6b | controlled_retrieval | fut_lastloop_frac | 0.0222 [0.002,0.040] | 0.0134 [-0.015,0.043] |
| ouro-2.6b | controlled_retrieval | fut_loop_entropy | -0.0267 [-0.054,0.004] | -0.0163 [-0.056,0.019] |
| ouro-2.6b | controlled_retrieval | qa_lastloop_frac | 0.00205 [-0.033,0.039] | 0.00741 [-0.025,0.038] |
| ouro-2.6b | hotpotqa_distractor | qa_attn_mean | 0.112 [0.084,0.147] | 0.019 [0.005,0.035] |
| ouro-2.6b | hotpotqa_distractor | qa_attn_last | 0.0896 [0.042,0.139] | 0.0069 [-0.011,0.023] |
| ouro-2.6b | hotpotqa_distractor | qa_attn_maxhead | 0.103 [0.075,0.130] | 0.0193 [-0.005,0.046] |
| ouro-2.6b | hotpotqa_distractor | recv_mean | 0.0469 [0.010,0.089] | 0.00173 [-0.014,0.017] |
| ouro-2.6b | hotpotqa_distractor | lastq_mean | 0.0877 [0.053,0.128] | 0.0181 [0.000,0.036] |
| ouro-2.6b | hotpotqa_distractor | qb_attn_mean | 0.0877 [0.053,0.128] | 0.0181 [0.000,0.036] |
| ouro-2.6b | hotpotqa_distractor | qa_delta_mean | 0.12 [0.089,0.157] | 0.0221 [0.007,0.039] |
| ouro-2.6b | hotpotqa_distractor | qa_delta_last | 0.0958 [0.050,0.145] | 0.00861 [-0.009,0.025] |
| ouro-2.6b | hotpotqa_distractor | qa_delta_loopw | 0.122 [0.095,0.154] | 0.0235 [0.008,0.041] |
| ouro-2.6b | hotpotqa_distractor | qa_vnorm_mean | -0.0206 [-0.052,0.013] | -0.00597 [-0.052,0.034] |
| ouro-2.6b | hotpotqa_distractor | src_U_mean | -0.0219 [-0.051,0.008] | 0.000533 [-0.037,0.045] |
| ouro-2.6b | hotpotqa_distractor | src_settle | -0.0128 [-0.056,0.032] | 0.0032 [-0.023,0.024] |
| ouro-2.6b | hotpotqa_distractor | fut_attn_mean | 0.173 [0.146,0.199] | 0.0246 [0.007,0.044] |
| ouro-2.6b | hotpotqa_distractor | fut_attn_max_t | 0.175 [0.144,0.203] | 0.0285 [0.010,0.049] |
| ouro-2.6b | hotpotqa_distractor | fut_delta_mean | 0.175 [0.144,0.203] | 0.0246 [0.003,0.047] |
| ouro-2.6b | hotpotqa_distractor | fut_delta_max_t | 0.175 [0.142,0.205] | 0.0278 [0.004,0.052] |
| ouro-2.6b | hotpotqa_distractor | fut_attn_at_t | 0.175 [0.144,0.203] | 0.0285 [0.010,0.049] |
| ouro-2.6b | hotpotqa_distractor | fut_delta_at_t | 0.175 [0.141,0.205] | 0.0275 [0.004,0.052] |
| ouro-2.6b | hotpotqa_distractor | qa_all_kl | 0.0126 [-0.022,0.044] | 0.043 [0.017,0.068] |
| ouro-2.6b | hotpotqa_distractor | qa_all_E_R | 0.0479 [-0.005,0.097] | -0.00105 [-0.037,0.041] |
| ouro-2.6b | hotpotqa_distractor | qa_all_dn_R | 0.0479 [-0.005,0.097] | -0.00105 [-0.037,0.041] |
| ouro-2.6b | hotpotqa_distractor | qb_all_kl | 0.015 [-0.019,0.050] | 0.00113 [-0.033,0.037] |
| ouro-2.6b | hotpotqa_distractor | qb_all_E_R | 0.061 [0.029,0.097] | -0.000377 [-0.019,0.018] |
| ouro-2.6b | hotpotqa_distractor | qf_all_kl | 0.071 [0.044,0.101] | 0.0047 [-0.019,0.023] |
| ouro-2.6b | hotpotqa_distractor | qf_all_E_R | 0.147 [0.118,0.173] | 0.0338 [0.004,0.068] |
| ouro-2.6b | hotpotqa_distractor | qf_all_dn_R | 0.163 [0.133,0.191] | 0.0342 [0.009,0.063] |
| ouro-2.6b | hotpotqa_distractor | qf_loop1_gain | -0.0861 [-0.132,-0.042] | -0.00225 [-0.029,0.025] |
| ouro-2.6b | hotpotqa_distractor | qa_loop1_gain | -0.0585 [-0.121,-0.012] | 0.0183 [0.000,0.036] |
| ouro-2.6b | hotpotqa_distractor | clean_kl_mean | 0.149 [0.127,0.169] | -9.45e-05 [-0.029,0.031] |
| ouro-2.6b | hotpotqa_distractor | la1_attn | 0.0898 [0.062,0.120] | 0.012 [-0.011,0.034] |
| ouro-2.6b | hotpotqa_distractor | la2_attn | 0.167 [0.137,0.199] | 0.0189 [0.000,0.037] |
| ouro-2.6b | hotpotqa_distractor | la12_attn_max | 0.167 [0.136,0.200] | 0.0192 [0.001,0.037] |
| ouro-2.6b | hotpotqa_distractor | la12_delta_max | 0.17 [0.137,0.204] | 0.021 [0.001,0.040] |
| ouro-2.6b | hotpotqa_distractor | fut_lastloop_frac | 0.072 [0.027,0.121] | -0.00768 [-0.040,0.018] |
| ouro-2.6b | hotpotqa_distractor | fut_loop_entropy | -0.00249 [-0.033,0.027] | 0.0165 [-0.017,0.050] |
| ouro-2.6b | hotpotqa_distractor | qa_lastloop_frac | -0.00896 [-0.056,0.035] | -0.011 [-0.043,0.013] |

## H1/H2 — attention-matched and local-perturbation-matched source pairs

Pairs are same-role, |Δpos|≤16, within prompt. 'attention' = clean question-anchor attention (deployable); 'future' = the consuming generated query's clean attention (oracle). δ = closed-form local attention-output perturbation at the same query. If Y dispersion stays large under future-attn+δ matching, the damage is not a local/attention quantity at the consuming query either (propagation / recurrence or later history). Pairs where both members have mean KL < 1e-3 (BF16 floor) are excluded as uninformative.

| model | dataset | pairing | n pairs | median |log10 Y ratio| | frac >10× | probe-KL (anchor of the pairing) sign agreement with Y [CI] | future-attn sign agreement [CI] |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | attention-matched | 45 | 2.14 | 0.8 | 0.489 [0.356,0.644] | 0.867 [0.756,0.956] |
| huginn-0125 | controlled_retrieval | attention+δ-matched | 44 | 2.08 | 0.795 | 0.477 [0.341,0.614] | 0.864 [0.750,0.955] |
| huginn-0125 | controlled_retrieval | future-attn-matched | 34 | 0.655 | 0.412 | 0.912 [0.824,1.000] | 0.588 [0.412,0.735] |
| huginn-0125 | controlled_retrieval | future-attn+δ-matched | 33 | 0.614 | 0.364 | 0.909 [0.788,1.000] | 0.606 [0.424,0.758] |
| huginn-0125 | controlled_retrieval | random same-role | 59 | 3.47 | 0.949 | 0.559 [0.441,0.678] | 0.949 [0.881,1.000] |
| huginn-0125 | hotpotqa_distractor | attention-matched | 321 | 0.39 | 0.162 | 0.517 [0.464,0.570] | 0.632 [0.583,0.682] |
| huginn-0125 | hotpotqa_distractor | attention+δ-matched | 316 | 0.386 | 0.158 | 0.522 [0.465,0.576] | 0.633 [0.579,0.684] |
| huginn-0125 | hotpotqa_distractor | future-attn-matched | 291 | 0.333 | 0.0687 | 0.598 [0.543,0.649] | 0.485 [0.426,0.540] |
| huginn-0125 | hotpotqa_distractor | future-attn+δ-matched | 284 | 0.328 | 0.0669 | 0.588 [0.532,0.641] | 0.479 [0.419,0.535] |
| huginn-0125 | hotpotqa_distractor | random same-role | 358 | 0.376 | 0.162 | 0.475 [0.422,0.525] | 0.561 [0.508,0.612] |
| ouro-2.6b | controlled_retrieval | attention-matched | 36 | 3.28 | 0.806 | 0.389 [0.222,0.556] | 0.833 [0.694,0.944] |
| ouro-2.6b | controlled_retrieval | attention+δ-matched | 36 | 3.28 | 0.806 | 0.389 [0.222,0.556] | 0.833 [0.722,0.944] |
| ouro-2.6b | controlled_retrieval | future-attn-matched | 29 | 0.808 | 0.379 | 0.517 [0.345,0.690] | 0.552 [0.379,0.724] |
| ouro-2.6b | controlled_retrieval | future-attn+δ-matched | 27 | 0.808 | 0.333 | 0.481 [0.296,0.667] | 0.556 [0.370,0.741] |
| ouro-2.6b | controlled_retrieval | random same-role | 46 | 3.44 | 0.87 | 0.587 [0.435,0.717] | 0.913 [0.826,0.978] |
| ouro-2.6b | hotpotqa_distractor | attention-matched | 59 | 1.18 | 0.576 | 0.542 [0.424,0.661] | 0.831 [0.729,0.915] |
| ouro-2.6b | hotpotqa_distractor | attention+δ-matched | 54 | 1.17 | 0.556 | 0.574 [0.444,0.704] | 0.815 [0.704,0.907] |
| ouro-2.6b | hotpotqa_distractor | future-attn-matched | 40 | 0.651 | 0.4 | 0.725 [0.600,0.850] | 0.5 [0.350,0.650] |
| ouro-2.6b | hotpotqa_distractor | future-attn+δ-matched | 36 | 0.603 | 0.361 | 0.722 [0.556,0.861] | 0.5 [0.333,0.667] |
| ouro-2.6b | hotpotqa_distractor | random same-role | 63 | 1.06 | 0.54 | 0.508 [0.381,0.635] | 0.873 [0.778,0.952] |

## Loop profile of clean attention to the source (normalised over loops): top-5% harm vs rest

| model | dataset | group | n | query | first loop | last loop | peak loop | entropy |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | rest | 2672 | consuming query t* | 0.0566 | 0.0281 | 1 | 3.44 |
| huginn-0125 | controlled_retrieval | top5 | 144 | consuming query t* | 0.0127 | 0.0336 | 23 | 3.46 |
| huginn-0125 | controlled_retrieval | rest | 2672 | question anchor | 0.056 | 0.0257 | 2 | 3.4 |
| huginn-0125 | controlled_retrieval | top5 | 144 | question anchor | 0.0357 | 0.0284 | 2 | 3.44 |
| huginn-0125 | hotpotqa_distractor | rest | 3489 | consuming query t* | 0.0238 | 0.0311 | 4 | 3.46 |
| huginn-0125 | hotpotqa_distractor | top5 | 188 | consuming query t* | 0.00887 | 0.036 | 25 | 3.44 |
| huginn-0125 | hotpotqa_distractor | rest | 3489 | question anchor | 0.0439 | 0.0285 | 2 | 3.45 |
| huginn-0125 | hotpotqa_distractor | top5 | 188 | question anchor | 0.0323 | 0.0298 | 3 | 3.45 |
| ouro-2.6b | controlled_retrieval | rest | 2672 | consuming query t* | 0.22 | 0.298 | 4 | 1.38 |
| ouro-2.6b | controlled_retrieval | top5 | 144 | consuming query t* | 0.235 | 0.299 | 4 | 1.38 |
| ouro-2.6b | controlled_retrieval | rest | 2672 | question anchor | 0.355 | 0.221 | 1 | 1.36 |
| ouro-2.6b | controlled_retrieval | top5 | 144 | question anchor | 0.338 | 0.25 | 1 | 1.36 |
| ouro-2.6b | hotpotqa_distractor | rest | 3363 | consuming query t* | 0.302 | 0.228 | 1 | 1.38 |
| ouro-2.6b | hotpotqa_distractor | top5 | 182 | consuming query t* | 0.216 | 0.284 | 4 | 1.38 |
| ouro-2.6b | hotpotqa_distractor | rest | 3363 | question anchor | 0.385 | 0.181 | 1 | 1.34 |
| ouro-2.6b | hotpotqa_distractor | top5 | 182 | question anchor | 0.367 | 0.164 | 1 | 1.34 |

## Loop-resolved attention importance: ρ(attention at loop r, Y) within prompt (mean over prompts)

| model | dataset | query | r=1 | r=2 | r=3 | r=4 | r=8 | r=16 | r=24 | r=32 | mean over r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | consuming query t* (oracle) | 0.0749 | 0.154 | 0.179 | 0.175 | 0.183 | 0.194 | 0.204 | 0.203 | 0.202 |
| huginn-0125 | controlled_retrieval | 2-token lookahead | 0.0389 | 0.0863 | 0.133 | 0.161 | 0.185 | 0.189 | 0.188 | 0.19 | 0.196 |
| huginn-0125 | controlled_retrieval | question anchor | 0.0409 | 0.111 | 0.136 | 0.123 | 0.139 | 0.113 | 0.119 | 0.119 | 0.118 |
| huginn-0125 | hotpotqa_distractor | consuming query t* (oracle) | 0.0744 | 0.126 | 0.118 | 0.117 | 0.195 | 0.229 | 0.234 | 0.235 | 0.228 |
| huginn-0125 | hotpotqa_distractor | 2-token lookahead | 0.0352 | 0.105 | 0.0942 | 0.1 | 0.19 | 0.231 | 0.242 | 0.241 | 0.235 |
| huginn-0125 | hotpotqa_distractor | question anchor | 0.0375 | 0.119 | 0.111 | 0.11 | 0.0992 | 0.114 | 0.119 | 0.121 | 0.12 |
| ouro-2.6b | controlled_retrieval | consuming query t* (oracle) | 0.0888 | 0.0986 | 0.134 | 0.15 | — | — | — | — | 0.147 |
| ouro-2.6b | controlled_retrieval | 2-token lookahead | 0.0733 | 0.106 | 0.137 | 0.157 | — | — | — | — | 0.148 |
| ouro-2.6b | controlled_retrieval | question anchor | 0.0564 | 0.086 | 0.0894 | 0.0846 | — | — | — | — | 0.0949 |
| ouro-2.6b | hotpotqa_distractor | consuming query t* (oracle) | 0.122 | 0.163 | 0.177 | 0.174 | — | — | — | — | 0.175 |
| ouro-2.6b | hotpotqa_distractor | 2-token lookahead | 0.0871 | 0.107 | 0.168 | 0.169 | — | — | — | — | 0.167 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 0.0979 | 0.111 | 0.106 | 0.0896 | — | — | — | — | 0.112 |

## H3 — leave-one-question-UID-out prediction of future eviction damage (log10 mean KL over the gold continuation)

Ridge on log-magnitude features; Spearman within each held-out prompt, mean [95% prompt bootstrap]. D and F are offline oracles (query recomputation / future-query attention); S is the cheap one-pass surrogate.

| model | dataset | features | mean ρ [CI] | AUROC top-10% harm |
|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | A attention/position/lexical | 0.145 [0.098,0.193] | 0.7 [0.628,0.770] |
| huginn-0125 | controlled_retrieval | B = A + source trajectory | 0.136 [0.095,0.176] | 0.692 [0.627,0.755] |
| huginn-0125 | controlled_retrieval | C = A + value/local-perturbation | 0.149 [0.105,0.194] | 0.703 [0.643,0.764] |
| huginn-0125 | controlled_retrieval | S = C + cheap surrogates | 0.152 [0.115,0.190] | 0.709 [0.648,0.769] |
| huginn-0125 | controlled_retrieval | L = C + 2-token format lookahead (deployable, +2 steps) | 0.157 [0.111,0.206] | 0.724 [0.662,0.783] |
| huginn-0125 | controlled_retrieval | D = C + prompt-anchor query probes (oracle) | 0.139 [0.097,0.185] | 0.69 [0.627,0.751] |
| huginn-0125 | controlled_retrieval | F = C + future-query census (oracle) | 0.187 [0.145,0.232] | 0.737 [0.684,0.794] |
| huginn-0125 | controlled_retrieval | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.215 [0.182,0.245] | 0.759 [0.711,0.812] |
| huginn-0125 | controlled_retrieval | G1g = F + propagation gains only at t* (oracle) | 0.18 [0.138,0.227] | 0.74 [0.689,0.794] |
| huginn-0125 | controlled_retrieval | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.221 [0.165,0.262] | 0.751 [0.701,0.805] |
| huginn-0125 | controlled_retrieval | G = F + all future-query probes (oracle) | 0.209 [0.158,0.252] | 0.75 [0.697,0.804] |
| huginn-0125 | controlled_retrieval | D+F | 0.183 [0.137,0.231] | 0.738 [0.685,0.794] |
| huginn-0125 | controlled_retrieval | LIFT B − A | -0.00963 [-0.022,-0.000] | — |
| huginn-0125 | controlled_retrieval | LIFT C − A | 0.00354 [-0.007,0.013] | — |
| huginn-0125 | controlled_retrieval | LIFT S − C | 0.00324 [-0.008,0.014] | — |
| huginn-0125 | controlled_retrieval | LIFT L − C | 0.00838 [-0.009,0.025] | — |
| huginn-0125 | controlled_retrieval | LIFT F − L | 0.03 [-0.005,0.067] | — |
| huginn-0125 | controlled_retrieval | LIFT D − C | -0.0097 [-0.025,0.004] | — |
| huginn-0125 | controlled_retrieval | LIFT F − C | 0.0383 [-0.005,0.084] | — |
| huginn-0125 | controlled_retrieval | LIFT G1 − F | 0.0281 [0.001,0.056] | — |
| huginn-0125 | controlled_retrieval | LIFT G1g − F | -0.00758 [-0.019,0.003] | — |
| huginn-0125 | controlled_retrieval | LIFT G2 − F | 0.0337 [-0.007,0.073] | — |
| huginn-0125 | controlled_retrieval | LIFT D+F − F | -0.00414 [-0.014,0.003] | — |
| huginn-0125 | hotpotqa_distractor | A attention/position/lexical | 0.161 [0.101,0.220] | 0.712 [0.636,0.783] |
| huginn-0125 | hotpotqa_distractor | B = A + source trajectory | 0.158 [0.093,0.222] | 0.707 [0.627,0.782] |
| huginn-0125 | hotpotqa_distractor | C = A + value/local-perturbation | 0.156 [0.099,0.211] | 0.708 [0.631,0.779] |
| huginn-0125 | hotpotqa_distractor | S = C + cheap surrogates | 0.156 [0.098,0.210] | 0.708 [0.631,0.778] |
| huginn-0125 | hotpotqa_distractor | L = C + 2-token format lookahead (deployable, +2 steps) | 0.172 [0.110,0.225] | 0.736 [0.664,0.802] |
| huginn-0125 | hotpotqa_distractor | D = C + prompt-anchor query probes (oracle) | 0.147 [0.093,0.199] | 0.693 [0.617,0.763] |
| huginn-0125 | hotpotqa_distractor | F = C + future-query census (oracle) | 0.223 [0.160,0.271] | 0.794 [0.720,0.858] |
| huginn-0125 | hotpotqa_distractor | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.209 [0.147,0.260] | 0.771 [0.702,0.833] |
| huginn-0125 | hotpotqa_distractor | G1g = F + propagation gains only at t* (oracle) | 0.227 [0.166,0.275] | 0.793 [0.721,0.851] |
| huginn-0125 | hotpotqa_distractor | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.226 [0.164,0.275] | 0.792 [0.717,0.856] |
| huginn-0125 | hotpotqa_distractor | G = F + all future-query probes (oracle) | 0.22 [0.156,0.277] | 0.775 [0.697,0.842] |
| huginn-0125 | hotpotqa_distractor | D+F | 0.22 [0.159,0.269] | 0.788 [0.717,0.850] |
| huginn-0125 | hotpotqa_distractor | LIFT B − A | -0.00303 [-0.011,0.005] | — |
| huginn-0125 | hotpotqa_distractor | LIFT C − A | -0.00539 [-0.011,0.001] | — |
| huginn-0125 | hotpotqa_distractor | LIFT S − C | -0.000284 [-0.002,0.002] | — |
| huginn-0125 | hotpotqa_distractor | LIFT L − C | 0.0163 [0.006,0.026] | — |
| huginn-0125 | hotpotqa_distractor | LIFT F − L | 0.0504 [0.006,0.102] | — |
| huginn-0125 | hotpotqa_distractor | LIFT D − C | -0.00888 [-0.014,-0.003] | — |
| huginn-0125 | hotpotqa_distractor | LIFT F − C | 0.0667 [0.029,0.112] | — |
| huginn-0125 | hotpotqa_distractor | LIFT G1 − F | -0.0137 [-0.043,0.012] | — |
| huginn-0125 | hotpotqa_distractor | LIFT G1g − F | 0.00445 [-0.001,0.010] | — |
| huginn-0125 | hotpotqa_distractor | LIFT G2 − F | 0.00364 [-0.006,0.012] | — |
| huginn-0125 | hotpotqa_distractor | LIFT D+F − F | -0.00297 [-0.012,0.004] | — |
| ouro-2.6b | controlled_retrieval | A attention/position/lexical | 0.0642 [0.032,0.094] | 0.613 [0.607,0.619] |
| ouro-2.6b | controlled_retrieval | B = A + source trajectory | 0.0939 [0.063,0.126] | 0.644 [0.621,0.671] |
| ouro-2.6b | controlled_retrieval | C = A + value/local-perturbation | 0.089 [0.057,0.118] | 0.628 [0.613,0.645] |
| ouro-2.6b | controlled_retrieval | S = C + cheap surrogates | 0.0938 [0.061,0.124] | 0.63 [0.617,0.643] |
| ouro-2.6b | controlled_retrieval | L = C + 2-token format lookahead (deployable, +2 steps) | 0.112 [0.077,0.142] | 0.657 [0.624,0.685] |
| ouro-2.6b | controlled_retrieval | D = C + prompt-anchor query probes (oracle) | 0.0793 [0.041,0.119] | 0.614 [0.596,0.635] |
| ouro-2.6b | controlled_retrieval | F = C + future-query census (oracle) | 0.113 [0.083,0.147] | 0.668 [0.653,0.682] |
| ouro-2.6b | controlled_retrieval | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.123 [0.096,0.150] | 0.679 [0.661,0.696] |
| ouro-2.6b | controlled_retrieval | G1g = F + propagation gains only at t* (oracle) | 0.113 [0.085,0.146] | 0.668 [0.653,0.681] |
| ouro-2.6b | controlled_retrieval | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.123 [0.087,0.160] | 0.672 [0.649,0.695] |
| ouro-2.6b | controlled_retrieval | G = F + all future-query probes (oracle) | 0.135 [0.116,0.159] | 0.684 [0.671,0.698] |
| ouro-2.6b | controlled_retrieval | D+F | 0.117 [0.089,0.147] | 0.671 [0.657,0.684] |
| ouro-2.6b | controlled_retrieval | LIFT B − A | 0.0298 [0.012,0.048] | — |
| ouro-2.6b | controlled_retrieval | LIFT C − A | 0.0248 [0.013,0.036] | — |
| ouro-2.6b | controlled_retrieval | LIFT S − C | 0.00481 [-0.004,0.013] | — |
| ouro-2.6b | controlled_retrieval | LIFT L − C | 0.0234 [0.002,0.047] | — |
| ouro-2.6b | controlled_retrieval | LIFT F − L | 0.000321 [-0.030,0.037] | — |
| ouro-2.6b | controlled_retrieval | LIFT D − C | -0.00968 [-0.022,0.006] | — |
| ouro-2.6b | controlled_retrieval | LIFT F − C | 0.0237 [-0.004,0.052] | — |
| ouro-2.6b | controlled_retrieval | LIFT G1 − F | 0.00981 [-0.009,0.032] | — |
| ouro-2.6b | controlled_retrieval | LIFT G1g − F | 0.000151 [-0.005,0.005] | — |
| ouro-2.6b | controlled_retrieval | LIFT G2 − F | 0.0107 [-0.008,0.027] | — |
| ouro-2.6b | controlled_retrieval | LIFT D+F − F | 0.00439 [-0.004,0.013] | — |
| ouro-2.6b | hotpotqa_distractor | A attention/position/lexical | 0.0802 [0.053,0.110] | 0.609 [0.570,0.651] |
| ouro-2.6b | hotpotqa_distractor | B = A + source trajectory | 0.0696 [0.030,0.111] | 0.587 [0.544,0.635] |
| ouro-2.6b | hotpotqa_distractor | C = A + value/local-perturbation | 0.0987 [0.072,0.128] | 0.627 [0.577,0.674] |
| ouro-2.6b | hotpotqa_distractor | S = C + cheap surrogates | 0.107 [0.071,0.146] | 0.628 [0.577,0.683] |
| ouro-2.6b | hotpotqa_distractor | L = C + 2-token format lookahead (deployable, +2 steps) | 0.133 [0.092,0.175] | 0.699 [0.663,0.741] |
| ouro-2.6b | hotpotqa_distractor | D = C + prompt-anchor query probes (oracle) | 0.0811 [0.053,0.109] | 0.59 [0.558,0.622] |
| ouro-2.6b | hotpotqa_distractor | F = C + future-query census (oracle) | 0.159 [0.128,0.191] | 0.713 [0.670,0.754] |
| ouro-2.6b | hotpotqa_distractor | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.144 [0.114,0.175] | 0.72 [0.679,0.760] |
| ouro-2.6b | hotpotqa_distractor | G1g = F + propagation gains only at t* (oracle) | 0.153 [0.123,0.187] | 0.716 [0.680,0.751] |
| ouro-2.6b | hotpotqa_distractor | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.148 [0.116,0.181] | 0.715 [0.671,0.757] |
| ouro-2.6b | hotpotqa_distractor | G = F + all future-query probes (oracle) | 0.141 [0.106,0.179] | 0.713 [0.675,0.749] |
| ouro-2.6b | hotpotqa_distractor | D+F | 0.167 [0.132,0.199] | 0.712 [0.671,0.749] |
| ouro-2.6b | hotpotqa_distractor | LIFT B − A | -0.0106 [-0.027,0.005] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT C − A | 0.0185 [0.004,0.037] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT S − C | 0.00802 [-0.005,0.021] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT L − C | 0.0345 [-0.019,0.084] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT F − L | 0.0254 [-0.017,0.072] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT D − C | -0.0176 [-0.037,0.003] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT F − C | 0.0599 [0.039,0.079] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT G1 − F | -0.0143 [-0.032,0.002] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT G1g − F | -0.00574 [-0.021,0.007] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT G2 − F | -0.0104 [-0.019,-0.003] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT D+F − F | 0.00801 [-0.000,0.016] | — |

## Deferred-eviction lookahead curve: LOPO ranking with the first k generated queries' clean attention

| model | dataset | k | mean ρ [CI] | AUROC top-10% | LIFT over k=0 [CI] | univariate ρ(max attn t≤k) |
|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0 (C) | 0.149 [0.105,0.194] | — | — | — |
| huginn-0125 | controlled_retrieval | 1 | 0.149 [0.110,0.191] | 0.707 | 0.000448 [-0.005,0.005] | 0.15 |
| huginn-0125 | controlled_retrieval | 2 | 0.164 [0.121,0.211] | 0.73 | 0.0146 [-0.007,0.034] | 0.196 |
| huginn-0125 | controlled_retrieval | 3 | 0.153 [0.119,0.187] | 0.719 | 0.00388 [-0.018,0.024] | 0.191 |
| huginn-0125 | controlled_retrieval | 4 | 0.157 [0.124,0.192] | 0.724 | 0.00791 [-0.011,0.028] | 0.188 |
| huginn-0125 | controlled_retrieval | 6 | 0.186 [0.145,0.233] | 0.75 | 0.0369 [0.004,0.067] | 0.193 |
| huginn-0125 | controlled_retrieval | 8 | 0.198 [0.148,0.250] | 0.751 | 0.0496 [0.017,0.082] | 0.184 |
| huginn-0125 | hotpotqa_distractor | 0 (C) | 0.156 [0.099,0.211] | — | — | — |
| huginn-0125 | hotpotqa_distractor | 1 | 0.154 [0.096,0.209] | 0.707 | -0.00194 [-0.003,-0.001] | 0.108 |
| huginn-0125 | hotpotqa_distractor | 2 | 0.176 [0.118,0.228] | 0.738 | 0.0206 [0.011,0.029] | 0.235 |
| huginn-0125 | hotpotqa_distractor | 3 | 0.204 [0.148,0.253] | 0.771 | 0.0486 [0.022,0.087] | 0.231 |
| huginn-0125 | hotpotqa_distractor | 4 | 0.211 [0.153,0.260] | 0.779 | 0.0551 [0.027,0.096] | 0.23 |
| huginn-0125 | hotpotqa_distractor | 6 | 0.219 [0.160,0.268] | 0.792 | 0.0628 [0.033,0.102] | 0.23 |
| huginn-0125 | hotpotqa_distractor | 8 | 0.219 [0.160,0.269] | 0.792 | 0.063 [0.033,0.104] | 0.227 |
| ouro-2.6b | controlled_retrieval | 0 (C) | 0.089 [0.057,0.118] | — | — | — |
| ouro-2.6b | controlled_retrieval | 1 | 0.0971 [0.069,0.126] | 0.643 | 0.00811 [-0.003,0.023] | 0.114 |
| ouro-2.6b | controlled_retrieval | 2 | 0.145 [0.111,0.180] | 0.671 | 0.0557 [0.020,0.085] | 0.148 |
| ouro-2.6b | controlled_retrieval | 3 | 0.131 [0.104,0.162] | 0.678 | 0.0423 [0.018,0.065] | 0.141 |
| ouro-2.6b | controlled_retrieval | 4 | 0.119 [0.097,0.143] | 0.669 | 0.0302 [-0.003,0.063] | 0.125 |
| ouro-2.6b | controlled_retrieval | 6 | 0.124 [0.095,0.148] | 0.67 | 0.0347 [0.014,0.056] | 0.125 |
| ouro-2.6b | controlled_retrieval | 8 | 0.122 [0.098,0.148] | 0.661 | 0.0327 [0.014,0.053] | 0.137 |
| ouro-2.6b | hotpotqa_distractor | 0 (C) | 0.0987 [0.072,0.128] | — | — | — |
| ouro-2.6b | hotpotqa_distractor | 1 | 0.11 [0.081,0.139] | 0.636 | 0.0109 [-0.003,0.025] | 0.0898 |
| ouro-2.6b | hotpotqa_distractor | 2 | 0.141 [0.105,0.178] | 0.707 | 0.0423 [-0.007,0.085] | 0.167 |
| ouro-2.6b | hotpotqa_distractor | 3 | 0.143 [0.103,0.182] | 0.713 | 0.0446 [0.009,0.077] | 0.175 |
| ouro-2.6b | hotpotqa_distractor | 4 | 0.147 [0.112,0.182] | 0.715 | 0.0486 [0.018,0.077] | 0.177 |
| ouro-2.6b | hotpotqa_distractor | 6 | 0.152 [0.115,0.188] | 0.713 | 0.0538 [0.024,0.082] | 0.174 |
| ouro-2.6b | hotpotqa_distractor | 8 | 0.155 [0.115,0.195] | 0.713 | 0.0561 [0.025,0.087] | 0.174 |

# Generalisation arm: v6 W1/W2 prompts (128-token greedy continuation outcome from the frozen v6 census)

## Query-conditioned probe magnitudes vs the numerical floor

Sources binned by clean question-anchor attention (within prompt deciles). The lowest decile serves as the empirical 'near-zero perturbation' floor: in BF16 any mask changes rounding, so hidden-state E has a floor even when the removed attention mass is ~0; the question is whether higher-attention sources rise above it.

| model | dataset | attention decile | n | median qa all-loop KL | median E_R | median ‖Δh_R‖ | median top-1 flip |
|---|---|---|---|---|---|---|---|
| huginn-0125 | v6_W1 | 0 | 58 | 4.21e-05 | 0.0147 | 1.13 | 0 |
| huginn-0125 | v6_W1 | 1 | 54 | 0.000125 | 0.0149 | 1.14 | 0 |
| huginn-0125 | v6_W1 | 2 | 56 | 5.4e-05 | 0.0148 | 1.13 | 0 |
| huginn-0125 | v6_W1 | 3 | 54 | 4.74e-05 | 0.0149 | 1.14 | 0 |
| huginn-0125 | v6_W1 | 4 | 57 | 0.000157 | 0.0149 | 1.14 | 0 |
| huginn-0125 | v6_W1 | 5 | 54 | 0.000136 | 0.015 | 1.14 | 0 |
| huginn-0125 | v6_W1 | 6 | 54 | 0.000136 | 0.015 | 1.14 | 0 |
| huginn-0125 | v6_W1 | 7 | 56 | 0.00015 | 0.0151 | 1.15 | 0 |
| huginn-0125 | v6_W1 | 8 | 54 | 0.000136 | 0.0155 | 1.18 | 0 |
| huginn-0125 | v6_W1 | 9 | 57 | 0.00119 | 0.0424 | 3.24 | 0 |
| huginn-0125 | v6_W2 | 0 | 52 | 3.67e-05 | 0.0142 | 1.09 | 0 |
| huginn-0125 | v6_W2 | 1 | 49 | 1.99e-05 | 0.0143 | 1.09 | 0 |
| huginn-0125 | v6_W2 | 2 | 50 | 3.65e-05 | 0.0144 | 1.1 | 0 |
| huginn-0125 | v6_W2 | 3 | 49 | 3.98e-05 | 0.0144 | 1.1 | 0 |
| huginn-0125 | v6_W2 | 4 | 51 | 3.05e-05 | 0.0145 | 1.11 | 0 |
| huginn-0125 | v6_W2 | 5 | 49 | 3.77e-05 | 0.0145 | 1.11 | 0 |
| huginn-0125 | v6_W2 | 6 | 49 | 3e-05 | 0.0145 | 1.1 | 0 |
| huginn-0125 | v6_W2 | 7 | 50 | 3.15e-05 | 0.0147 | 1.12 | 0 |
| huginn-0125 | v6_W2 | 8 | 49 | 4.22e-05 | 0.0155 | 1.18 | 0 |
| huginn-0125 | v6_W2 | 9 | 52 | 0.000493 | 0.0707 | 5.4 | 0 |
| ouro-2.6b | v6_W1 | 0 | 56 | 0.000308 | 0.0135 | 0.699 | 0 |
| ouro-2.6b | v6_W1 | 1 | 54 | 0.000277 | 0.0139 | 0.72 | 0 |
| ouro-2.6b | v6_W1 | 2 | 54 | 0.000275 | 0.0137 | 0.711 | 0 |
| ouro-2.6b | v6_W1 | 3 | 54 | 0.000293 | 0.0142 | 0.747 | 0 |
| ouro-2.6b | v6_W1 | 4 | 54 | 0.000295 | 0.0138 | 0.723 | 0 |
| ouro-2.6b | v6_W1 | 5 | 54 | 0.000282 | 0.0136 | 0.721 | 0 |
| ouro-2.6b | v6_W1 | 6 | 54 | 0.000312 | 0.0143 | 0.751 | 0 |
| ouro-2.6b | v6_W1 | 7 | 54 | 0.000353 | 0.0146 | 0.764 | 0 |
| ouro-2.6b | v6_W1 | 8 | 54 | 0.000314 | 0.0164 | 0.824 | 0 |
| ouro-2.6b | v6_W1 | 9 | 56 | 0.0022 | 0.0497 | 2.47 | 0 |
| ouro-2.6b | v6_W2 | 0 | 52 | 0.000263 | 0.0121 | 0.603 | 0 |
| ouro-2.6b | v6_W2 | 1 | 48 | 0.00023 | 0.0122 | 0.611 | 0 |
| ouro-2.6b | v6_W2 | 2 | 48 | 0.000226 | 0.012 | 0.601 | 0 |
| ouro-2.6b | v6_W2 | 3 | 49 | 0.000251 | 0.0123 | 0.62 | 0 |
| ouro-2.6b | v6_W2 | 4 | 49 | 0.00023 | 0.0123 | 0.61 | 0 |
| ouro-2.6b | v6_W2 | 5 | 48 | 0.000277 | 0.0124 | 0.617 | 0 |
| ouro-2.6b | v6_W2 | 6 | 49 | 0.000281 | 0.0124 | 0.62 | 0 |
| ouro-2.6b | v6_W2 | 7 | 48 | 0.000292 | 0.0131 | 0.651 | 0 |
| ouro-2.6b | v6_W2 | 8 | 48 | 0.000293 | 0.0148 | 0.729 | 0 |
| ouro-2.6b | v6_W2 | 9 | 52 | 0.00192 | 0.0498 | 2.45 | 0 |

## One-loop query masks: propagation gain E_R / E_s by masked loop s (median over sources; IQR)

| model | dataset | anchor | s | n | median E_s | median E_R | median gain | IQR gain | median KL | p90 KL |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | v6_W1 | question anchor | 1 | 554 | 0.0073 | 0.0147 | 2.01 | [1.46,2.24] | 0.000119 | 0.000262 |
| huginn-0125 | v6_W1 | question anchor | 2 | 554 | 0.00763 | 0.0147 | 1.92 | [1.67,2.06] | 5.47e-05 | 0.000261 |
| huginn-0125 | v6_W1 | question anchor | 4 | 554 | 0.00715 | 0.0147 | 2.04 | [1.94,2.15] | 5.94e-05 | 0.00026 |
| huginn-0125 | v6_W1 | question anchor | 8 | 554 | 0.007 | 0.0147 | 2.09 | [1.96,2.19] | 6.98e-05 | 0.000264 |
| huginn-0125 | v6_W1 | question anchor | 16 | 554 | 0.00692 | 0.0147 | 2.12 | [2.01,2.21] | 5.3e-05 | 0.000261 |
| huginn-0125 | v6_W1 | question anchor | 24 | 554 | 0.00681 | 0.0145 | 2.12 | [2.01,2.22] | 0.000121 | 0.000262 |
| huginn-0125 | v6_W1 | question anchor | 32 | 554 | 0.00681 | 0.00681 | 1 | [1,1] | 2e-05 | 0.000207 |
| huginn-0125 | v6_W1 | future consuming query t* | 1 | 554 | 0.00912 | 0.0121 | 1.3 | [0.655,1.64] | 0.000148 | 0.000731 |
| huginn-0125 | v6_W1 | future consuming query t* | 2 | 554 | 0.0116 | 0.012 | 1.04 | [0.407,1.48] | 0.000152 | 0.0011 |
| huginn-0125 | v6_W1 | future consuming query t* | 4 | 554 | 0.0158 | 0.0121 | 0.789 | [0.289,1.37] | 0.000152 | 0.000859 |
| huginn-0125 | v6_W1 | future consuming query t* | 8 | 554 | 0.0158 | 0.0121 | 0.746 | [0.286,1.3] | 0.000149 | 0.00087 |
| huginn-0125 | v6_W1 | future consuming query t* | 16 | 554 | 0.0136 | 0.0121 | 0.886 | [0.372,1.43] | 0.000157 | 0.0011 |
| huginn-0125 | v6_W1 | future consuming query t* | 24 | 554 | 0.014 | 0.0122 | 0.862 | [0.387,1.41] | 0.000161 | 0.000874 |
| huginn-0125 | v6_W1 | future consuming query t* | 32 | 554 | 0.0138 | 0.0138 | 1 | [1,1] | 0.000292 | 0.00776 |
| huginn-0125 | v6_W2 | question anchor | 1 | 500 | 0.00769 | 0.0142 | 1.85 | [1.25,2.08] | 3.03e-05 | 0.000113 |
| huginn-0125 | v6_W2 | question anchor | 2 | 500 | 0.00745 | 0.0142 | 1.91 | [1.65,2.03] | 3.06e-05 | 0.000143 |
| huginn-0125 | v6_W2 | question anchor | 4 | 500 | 0.00709 | 0.0142 | 2 | [1.83,2.11] | 2.84e-05 | 0.000109 |
| huginn-0125 | v6_W2 | question anchor | 8 | 500 | 0.00685 | 0.0142 | 2.07 | [1.94,2.18] | 2.77e-05 | 0.000117 |
| huginn-0125 | v6_W2 | question anchor | 16 | 500 | 0.00677 | 0.0142 | 2.1 | [1.96,2.18] | 3.27e-05 | 0.000131 |
| huginn-0125 | v6_W2 | question anchor | 24 | 500 | 0.00682 | 0.0141 | 2.06 | [1.92,2.15] | 2.96e-05 | 0.000108 |
| huginn-0125 | v6_W2 | question anchor | 32 | 500 | 0.00684 | 0.00684 | 1 | [1,1] | 3.61e-05 | 0.000104 |
| huginn-0125 | v6_W2 | future consuming query t* | 1 | 500 | 0.0149 | 0.0128 | 0.888 | [0.354,1.62] | 8.69e-06 | 0.000556 |
| huginn-0125 | v6_W2 | future consuming query t* | 2 | 500 | 0.0578 | 0.0128 | 0.222 | [0.0859,1.05] | 9.67e-06 | 0.000432 |
| huginn-0125 | v6_W2 | future consuming query t* | 4 | 500 | 0.109 | 0.0129 | 0.118 | [0.0655,0.632] | 1.12e-05 | 0.000542 |
| huginn-0125 | v6_W2 | future consuming query t* | 8 | 500 | 0.11 | 0.0128 | 0.119 | [0.063,0.57] | 9.57e-06 | 0.000536 |
| huginn-0125 | v6_W2 | future consuming query t* | 16 | 500 | 0.0975 | 0.013 | 0.139 | [0.0745,0.519] | 8.77e-06 | 0.000521 |
| huginn-0125 | v6_W2 | future consuming query t* | 24 | 500 | 0.0995 | 0.0171 | 0.211 | [0.151,0.547] | 2.07e-05 | 0.000881 |
| huginn-0125 | v6_W2 | future consuming query t* | 32 | 500 | 0.0952 | 0.0952 | 1 | [1,1] | 0.00146 | 0.14 |
| ouro-2.6b | v6_W1 | question anchor | 1 | 544 | 0.017 | 0.0136 | 0.804 | [0.709,0.977] | 0.000298 | 0.00154 |
| ouro-2.6b | v6_W1 | question anchor | 2 | 544 | 0.013 | 0.0136 | 1.08 | [0.989,1.21] | 0.000302 | 0.00171 |
| ouro-2.6b | v6_W1 | question anchor | 3 | 544 | 0.0132 | 0.0133 | 1.03 | [0.923,1.12] | 0.000294 | 0.00185 |
| ouro-2.6b | v6_W1 | question anchor | 4 | 544 | 0.0128 | 0.0128 | 1 | [1,1] | 0.000296 | 0.002 |
| ouro-2.6b | v6_W1 | future consuming query t* | 1 | 544 | 0.0886 | 0.0186 | 0.272 | [0.154,0.565] | 0.000172 | 0.00199 |
| ouro-2.6b | v6_W1 | future consuming query t* | 2 | 544 | 0.0628 | 0.0179 | 0.341 | [0.23,0.632] | 0.000179 | 0.002 |
| ouro-2.6b | v6_W1 | future consuming query t* | 3 | 544 | 0.0513 | 0.0212 | 0.505 | [0.371,0.737] | 0.000194 | 0.00197 |
| ouro-2.6b | v6_W1 | future consuming query t* | 4 | 544 | 0.0474 | 0.0474 | 1 | [1,1] | 0.000668 | 0.321 |
| ouro-2.6b | v6_W2 | question anchor | 1 | 491 | 0.0145 | 0.0122 | 0.808 | [0.731,0.92] | 0.00026 | 0.00122 |
| ouro-2.6b | v6_W2 | question anchor | 2 | 491 | 0.0125 | 0.0121 | 0.901 | [0.693,1.04] | 0.000252 | 0.00121 |
| ouro-2.6b | v6_W2 | question anchor | 3 | 491 | 0.012 | 0.0119 | 0.967 | [0.885,1.03] | 0.000243 | 0.00116 |
| ouro-2.6b | v6_W2 | question anchor | 4 | 491 | 0.0115 | 0.0115 | 1 | [1,1] | 0.000248 | 0.00122 |
| ouro-2.6b | v6_W2 | future consuming query t* | 1 | 491 | 0.22 | 0.0368 | 0.219 | [0.141,0.437] | 0.000211 | 0.00278 |
| ouro-2.6b | v6_W2 | future consuming query t* | 2 | 491 | 0.15 | 0.037 | 0.289 | [0.2,0.446] | 0.000137 | 0.00411 |
| ouro-2.6b | v6_W2 | future consuming query t* | 3 | 491 | 0.134 | 0.0534 | 0.441 | [0.343,0.571] | 0.000152 | 0.00369 |
| ouro-2.6b | v6_W2 | future consuming query t* | 4 | 491 | 0.133 | 0.133 | 1 | [1,1] | 0.00105 | 2.17 |

## Within-prompt Spearman of single features with future damage (mean over prompts [95% CI])

| model | dataset | feature | ρ(mean KL) | ρ(mean NLL shift) |
|---|---|---|---|---|
| huginn-0125 | v6_W1 | qa_attn_mean | 0.258 [0.131,0.356] | 0.0038 [-0.075,0.103] |
| huginn-0125 | v6_W1 | qa_attn_last | 0.275 [0.129,0.378] | 0.0378 [-0.037,0.133] |
| huginn-0125 | v6_W1 | qa_attn_maxhead | 0.102 [0.023,0.169] | -0.0548 [-0.118,0.015] |
| huginn-0125 | v6_W1 | recv_mean | 0.231 [0.175,0.329] | -0.0335 [-0.124,0.019] |
| huginn-0125 | v6_W1 | lastq_mean | 0.357 [0.255,0.457] | 0.0132 [-0.068,0.094] |
| huginn-0125 | v6_W1 | qb_attn_mean | 0.357 [0.255,0.457] | 0.0132 [-0.068,0.094] |
| huginn-0125 | v6_W1 | qa_delta_mean | 0.251 [0.132,0.343] | 0.00026 [-0.075,0.089] |
| huginn-0125 | v6_W1 | qa_delta_last | 0.264 [0.119,0.363] | 0.0311 [-0.043,0.121] |
| huginn-0125 | v6_W1 | qa_delta_loopw | 0.241 [0.131,0.329] | -0.0135 [-0.085,0.070] |
| huginn-0125 | v6_W1 | qa_vnorm_mean | 0.0612 [-0.074,0.197] | -0.0816 [-0.164,0.012] |
| huginn-0125 | v6_W1 | src_U_mean | -0.108 [-0.183,-0.011] | -0.1 [-0.150,-0.043] |
| huginn-0125 | v6_W1 | src_settle | -0.00824 [-0.097,0.080] | 0.0256 [-0.018,0.055] |
| huginn-0125 | v6_W1 | fut_attn_mean | 0.749 [0.695,0.808] | 0.19 [0.047,0.333] |
| huginn-0125 | v6_W1 | fut_attn_max_t | 0.766 [0.684,0.819] | 0.245 [0.072,0.418] |
| huginn-0125 | v6_W1 | fut_delta_mean | 0.754 [0.699,0.815] | 0.185 [0.045,0.325] |
| huginn-0125 | v6_W1 | fut_delta_max_t | 0.768 [0.686,0.823] | 0.243 [0.072,0.413] |
| huginn-0125 | v6_W1 | fut_attn_at_t | 0.766 [0.684,0.819] | 0.245 [0.072,0.418] |
| huginn-0125 | v6_W1 | fut_delta_at_t | 0.768 [0.688,0.823] | 0.242 [0.069,0.414] |
| huginn-0125 | v6_W1 | qa_all_kl | -0.0151 [-0.052,0.024] | -0.0291 [-0.096,0.019] |
| huginn-0125 | v6_W1 | qa_all_E_R | 0.0467 [-0.069,0.121] | -0.0522 [-0.076,-0.028] |
| huginn-0125 | v6_W1 | qa_all_dn_R | 0.0467 [-0.069,0.121] | -0.0522 [-0.076,-0.028] |
| huginn-0125 | v6_W1 | qb_all_kl | 0.246 [0.150,0.305] | 0.0894 [0.034,0.163] |
| huginn-0125 | v6_W1 | qb_all_E_R | 0.235 [0.197,0.273] | -0.0147 [-0.060,0.031] |
| huginn-0125 | v6_W1 | qf_all_kl | 0.676 [0.575,0.791] | 0.218 [0.060,0.386] |
| huginn-0125 | v6_W1 | qf_all_E_R | 0.762 [0.670,0.831] | 0.209 [0.022,0.396] |
| huginn-0125 | v6_W1 | qf_all_dn_R | 0.762 [0.670,0.831] | 0.209 [0.022,0.396] |
| huginn-0125 | v6_W1 | qf_loop1_gain | -0.354 [-0.421,-0.291] | -0.188 [-0.330,-0.063] |
| huginn-0125 | v6_W1 | qa_loop1_gain | -0.0539 [-0.094,-0.013] | 0.108 [0.066,0.145] |
| huginn-0125 | v6_W1 | clean_kl_mean | — | — |
| huginn-0125 | v6_W1 | la1_attn | 0.496 [0.299,0.619] | 0.014 [-0.130,0.153] |
| huginn-0125 | v6_W1 | la2_attn | 0.503 [0.290,0.621] | 0.0421 [-0.095,0.184] |
| huginn-0125 | v6_W1 | la12_attn_max | 0.51 [0.285,0.631] | 0.0155 [-0.125,0.153] |
| huginn-0125 | v6_W1 | la12_delta_max | 0.529 [0.301,0.659] | 0.0165 [-0.131,0.162] |
| huginn-0125 | v6_W1 | fut_lastloop_frac | 0.192 [0.101,0.282] | 0.0448 [-0.022,0.089] |
| huginn-0125 | v6_W1 | fut_loop_entropy | 0.0862 [0.021,0.151] | 0.113 [-0.007,0.274] |
| huginn-0125 | v6_W1 | qa_lastloop_frac | 0.0167 [-0.044,0.055] | 0.122 [0.077,0.167] |
| huginn-0125 | v6_W2 | qa_attn_mean | 0.0259 [-0.107,0.159] | -0.107 [-0.203,0.021] |
| huginn-0125 | v6_W2 | qa_attn_last | 0.0612 [-0.060,0.182] | -0.0636 [-0.168,0.055] |
| huginn-0125 | v6_W2 | qa_attn_maxhead | -0.0987 [-0.237,0.040] | -0.166 [-0.299,-0.034] |
| huginn-0125 | v6_W2 | recv_mean | 0.105 [0.037,0.154] | -0.0114 [-0.150,0.127] |
| huginn-0125 | v6_W2 | lastq_mean | 0.0452 [-0.095,0.242] | -0.121 [-0.214,-0.034] |
| huginn-0125 | v6_W2 | qb_attn_mean | 0.0452 [-0.095,0.242] | -0.121 [-0.214,-0.034] |
| huginn-0125 | v6_W2 | qa_delta_mean | 0.0149 [-0.121,0.151] | -0.118 [-0.212,0.006] |
| huginn-0125 | v6_W2 | qa_delta_last | 0.0357 [-0.083,0.155] | -0.0869 [-0.179,0.024] |
| huginn-0125 | v6_W2 | qa_delta_loopw | 0.00376 [-0.124,0.132] | -0.133 [-0.224,-0.016] |
| huginn-0125 | v6_W2 | qa_vnorm_mean | 0.00293 [-0.109,0.171] | -0.0806 [-0.229,0.106] |
| huginn-0125 | v6_W2 | src_U_mean | -0.172 [-0.267,-0.102] | -0.163 [-0.206,-0.106] |
| huginn-0125 | v6_W2 | src_settle | -0.205 [-0.299,-0.111] | -0.132 [-0.242,-0.045] |
| huginn-0125 | v6_W2 | fut_attn_mean | 0.704 [0.599,0.810] | 0.401 [0.257,0.538] |
| huginn-0125 | v6_W2 | fut_attn_max_t | 0.789 [0.729,0.848] | 0.535 [0.413,0.622] |
| huginn-0125 | v6_W2 | fut_delta_mean | 0.713 [0.613,0.813] | 0.402 [0.273,0.523] |
| huginn-0125 | v6_W2 | fut_delta_max_t | 0.824 [0.775,0.873] | 0.572 [0.443,0.681] |
| huginn-0125 | v6_W2 | fut_attn_at_t | 0.789 [0.729,0.848] | 0.535 [0.413,0.622] |
| huginn-0125 | v6_W2 | fut_delta_at_t | 0.823 [0.775,0.872] | 0.573 [0.445,0.681] |
| huginn-0125 | v6_W2 | qa_all_kl | -0.156 [-0.289,-0.028] | -0.225 [-0.252,-0.202] |
| huginn-0125 | v6_W2 | qa_all_E_R | -0.0907 [-0.155,-0.002] | -0.123 [-0.206,-0.010] |
| huginn-0125 | v6_W2 | qa_all_dn_R | -0.0907 [-0.155,-0.002] | -0.123 [-0.206,-0.010] |
| huginn-0125 | v6_W2 | qb_all_kl | 0.0186 [-0.072,0.123] | -0.0127 [-0.071,0.050] |
| huginn-0125 | v6_W2 | qb_all_E_R | -0.0592 [-0.173,0.097] | -0.139 [-0.243,-0.034] |
| huginn-0125 | v6_W2 | qf_all_kl | 0.871 [0.849,0.906] | 0.603 [0.464,0.743] |
| huginn-0125 | v6_W2 | qf_all_E_R | 0.813 [0.768,0.858] | 0.576 [0.463,0.667] |
| huginn-0125 | v6_W2 | qf_all_dn_R | 0.813 [0.768,0.858] | 0.576 [0.463,0.667] |
| huginn-0125 | v6_W2 | qf_loop1_gain | -0.424 [-0.489,-0.341] | -0.304 [-0.411,-0.243] |
| huginn-0125 | v6_W2 | qa_loop1_gain | 0.109 [0.010,0.212] | 0.139 [0.060,0.223] |
| huginn-0125 | v6_W2 | clean_kl_mean | — | — |
| huginn-0125 | v6_W2 | la1_attn | 0.17 [0.040,0.305] | -0.00897 [-0.132,0.115] |
| huginn-0125 | v6_W2 | la2_attn | 0.243 [0.172,0.345] | 0.0359 [-0.081,0.153] |
| huginn-0125 | v6_W2 | la12_attn_max | 0.2 [0.081,0.337] | 0.0116 [-0.104,0.127] |
| huginn-0125 | v6_W2 | la12_delta_max | 0.202 [0.078,0.340] | 0.0152 [-0.096,0.128] |
| huginn-0125 | v6_W2 | fut_lastloop_frac | -0.0167 [-0.061,0.035] | -0.0402 [-0.141,0.061] |
| huginn-0125 | v6_W2 | fut_loop_entropy | 0.328 [0.299,0.362] | 0.269 [0.189,0.366] |
| huginn-0125 | v6_W2 | qa_lastloop_frac | 0.108 [-0.029,0.230] | 0.151 [0.060,0.233] |
| ouro-2.6b | v6_W1 | qa_attn_mean | 0.14 [-0.231,0.510] | -0.00757 [-0.230,0.215] |
| ouro-2.6b | v6_W1 | qa_attn_last | 0.174 [-0.158,0.507] | 0.016 [-0.195,0.227] |
| ouro-2.6b | v6_W1 | qa_attn_maxhead | -0.0781 [-0.420,0.263] | -0.0971 [-0.335,0.141] |
| ouro-2.6b | v6_W1 | recv_mean | 0.137 [-0.146,0.420] | 0.0283 [-0.114,0.167] |
| ouro-2.6b | v6_W1 | lastq_mean | 0.17 [-0.216,0.556] | -0.0628 [-0.260,0.135] |
| ouro-2.6b | v6_W1 | qb_attn_mean | 0.17 [-0.216,0.556] | -0.0628 [-0.260,0.135] |
| ouro-2.6b | v6_W1 | qa_delta_mean | 0.185 [-0.159,0.530] | 0.0209 [-0.172,0.214] |
| ouro-2.6b | v6_W1 | qa_delta_last | 0.187 [-0.136,0.509] | 0.0244 [-0.171,0.220] |
| ouro-2.6b | v6_W1 | qa_delta_loopw | 0.18 [-0.173,0.534] | 0.0234 [-0.176,0.223] |
| ouro-2.6b | v6_W1 | qa_vnorm_mean | 0.203 [-0.037,0.410] | 0.00857 [-0.128,0.095] |
| ouro-2.6b | v6_W1 | src_U_mean | -0.187 [-0.200,-0.171] | -0.218 [-0.297,-0.138] |
| ouro-2.6b | v6_W1 | src_settle | 0.0636 [-0.005,0.119] | 0.0513 [-0.002,0.125] |
| ouro-2.6b | v6_W1 | fut_attn_mean | 0.576 [0.285,0.786] | 0.26 [0.118,0.354] |
| ouro-2.6b | v6_W1 | fut_attn_max_t | 0.778 [0.644,0.867] | 0.485 [0.432,0.532] |
| ouro-2.6b | v6_W1 | fut_delta_mean | 0.598 [0.312,0.798] | 0.277 [0.143,0.382] |
| ouro-2.6b | v6_W1 | fut_delta_max_t | 0.795 [0.673,0.872] | 0.495 [0.408,0.549] |
| ouro-2.6b | v6_W1 | fut_attn_at_t | 0.778 [0.644,0.867] | 0.485 [0.432,0.532] |
| ouro-2.6b | v6_W1 | fut_delta_at_t | 0.789 [0.661,0.870] | 0.487 [0.408,0.538] |
| ouro-2.6b | v6_W1 | qa_all_kl | -0.00266 [-0.165,0.167] | 0.002 [-0.151,0.155] |
| ouro-2.6b | v6_W1 | qa_all_E_R | 0.0885 [-0.167,0.344] | -0.0191 [-0.169,0.131] |
| ouro-2.6b | v6_W1 | qa_all_dn_R | 0.0885 [-0.167,0.344] | -0.0191 [-0.169,0.131] |
| ouro-2.6b | v6_W1 | qb_all_kl | 0.176 [-0.031,0.441] | 0.00971 [-0.043,0.095] |
| ouro-2.6b | v6_W1 | qb_all_E_R | 0.276 [0.045,0.559] | 0.0795 [-0.009,0.147] |
| ouro-2.6b | v6_W1 | qf_all_kl | 0.662 [0.608,0.716] | 0.409 [0.223,0.590] |
| ouro-2.6b | v6_W1 | qf_all_E_R | 0.751 [0.674,0.822] | 0.497 [0.412,0.552] |
| ouro-2.6b | v6_W1 | qf_all_dn_R | 0.75 [0.663,0.815] | 0.491 [0.403,0.543] |
| ouro-2.6b | v6_W1 | qf_loop1_gain | -0.521 [-0.629,-0.378] | -0.296 [-0.405,-0.188] |
| ouro-2.6b | v6_W1 | qa_loop1_gain | -0.0444 [-0.264,0.175] | -0.0221 [-0.150,0.109] |
| ouro-2.6b | v6_W1 | clean_kl_mean | — | — |
| ouro-2.6b | v6_W1 | la1_attn | 0.232 [-0.120,0.585] | -0.0264 [-0.219,0.123] |
| ouro-2.6b | v6_W1 | la2_attn | 0.222 [-0.068,0.509] | 0.0201 [-0.153,0.177] |
| ouro-2.6b | v6_W1 | la12_attn_max | 0.238 [-0.093,0.569] | 0.00338 [-0.198,0.188] |
| ouro-2.6b | v6_W1 | la12_delta_max | 0.266 [-0.051,0.575] | 0.0296 [-0.165,0.202] |
| ouro-2.6b | v6_W1 | fut_lastloop_frac | 0.0954 [-0.006,0.197] | 0.0271 [-0.022,0.061] |
| ouro-2.6b | v6_W1 | fut_loop_entropy | 0.378 [0.249,0.461] | 0.292 [0.207,0.377] |
| ouro-2.6b | v6_W1 | qa_lastloop_frac | 0.17 [0.049,0.259] | 0.106 [0.044,0.172] |
| ouro-2.6b | v6_W2 | qa_attn_mean | 0.0995 [-0.038,0.196] | 0.0536 [-0.066,0.173] |
| ouro-2.6b | v6_W2 | qa_attn_last | 0.133 [-0.015,0.246] | 0.0626 [-0.065,0.202] |
| ouro-2.6b | v6_W2 | qa_attn_maxhead | -0.104 [-0.313,0.090] | -0.0658 [-0.274,0.115] |
| ouro-2.6b | v6_W2 | recv_mean | 0.126 [-0.053,0.297] | 0.0556 [-0.102,0.211] |
| ouro-2.6b | v6_W2 | lastq_mean | 0.0561 [-0.127,0.162] | 0.0191 [-0.129,0.164] |
| ouro-2.6b | v6_W2 | qb_attn_mean | 0.0561 [-0.127,0.162] | 0.0191 [-0.129,0.164] |
| ouro-2.6b | v6_W2 | qa_delta_mean | 0.143 [-0.001,0.260] | 0.0779 [-0.044,0.222] |
| ouro-2.6b | v6_W2 | qa_delta_last | 0.159 [0.005,0.270] | 0.0781 [-0.050,0.222] |
| ouro-2.6b | v6_W2 | qa_delta_loopw | 0.144 [0.007,0.254] | 0.0814 [-0.037,0.219] |
| ouro-2.6b | v6_W2 | qa_vnorm_mean | 0.31 [0.116,0.493] | 0.169 [0.046,0.256] |
| ouro-2.6b | v6_W2 | src_U_mean | -0.361 [-0.409,-0.312] | -0.267 [-0.365,-0.170] |
| ouro-2.6b | v6_W2 | src_settle | 0.0723 [0.001,0.183] | 0.0531 [0.010,0.097] |
| ouro-2.6b | v6_W2 | fut_attn_mean | 0.43 [0.215,0.646] | 0.261 [0.171,0.352] |
| ouro-2.6b | v6_W2 | fut_attn_max_t | 0.74 [0.650,0.829] | 0.584 [0.559,0.601] |
| ouro-2.6b | v6_W2 | fut_delta_mean | 0.47 [0.263,0.676] | 0.292 [0.200,0.362] |
| ouro-2.6b | v6_W2 | fut_delta_max_t | 0.76 [0.668,0.851] | 0.591 [0.555,0.630] |
| ouro-2.6b | v6_W2 | fut_attn_at_t | 0.74 [0.650,0.829] | 0.584 [0.559,0.601] |
| ouro-2.6b | v6_W2 | fut_delta_at_t | 0.75 [0.654,0.846] | 0.583 [0.553,0.616] |
| ouro-2.6b | v6_W2 | qa_all_kl | -0.011 [-0.121,0.081] | -0.0446 [-0.149,0.048] |
| ouro-2.6b | v6_W2 | qa_all_E_R | 0.0737 [-0.075,0.246] | -0.00523 [-0.150,0.139] |
| ouro-2.6b | v6_W2 | qa_all_dn_R | 0.0737 [-0.075,0.246] | -0.00523 [-0.150,0.139] |
| ouro-2.6b | v6_W2 | qb_all_kl | 0.0437 [-0.073,0.160] | 0.0105 [-0.076,0.085] |
| ouro-2.6b | v6_W2 | qb_all_E_R | 0.164 [0.032,0.282] | 0.111 [0.020,0.185] |
| ouro-2.6b | v6_W2 | qf_all_kl | 0.717 [0.680,0.748] | 0.575 [0.459,0.692] |
| ouro-2.6b | v6_W2 | qf_all_E_R | 0.688 [0.568,0.807] | 0.53 [0.514,0.548] |
| ouro-2.6b | v6_W2 | qf_all_dn_R | 0.68 [0.557,0.802] | 0.522 [0.500,0.540] |
| ouro-2.6b | v6_W2 | qf_loop1_gain | -0.484 [-0.622,-0.310] | -0.375 [-0.473,-0.283] |
| ouro-2.6b | v6_W2 | qa_loop1_gain | -0.0825 [-0.208,0.035] | -0.0273 [-0.170,0.116] |
| ouro-2.6b | v6_W2 | clean_kl_mean | — | — |
| ouro-2.6b | v6_W2 | la1_attn | 0.085 [-0.064,0.194] | 0.00482 [-0.095,0.119] |
| ouro-2.6b | v6_W2 | la2_attn | 0.124 [0.064,0.184] | 0.0237 [-0.079,0.141] |
| ouro-2.6b | v6_W2 | la12_attn_max | 0.103 [-0.031,0.212] | 0.0117 [-0.076,0.127] |
| ouro-2.6b | v6_W2 | la12_delta_max | 0.137 [-0.001,0.242] | 0.0344 [-0.057,0.147] |
| ouro-2.6b | v6_W2 | fut_lastloop_frac | 0.113 [-0.031,0.258] | 0.116 [-0.065,0.297] |
| ouro-2.6b | v6_W2 | fut_loop_entropy | 0.345 [0.292,0.420] | 0.288 [0.241,0.339] |
| ouro-2.6b | v6_W2 | qa_lastloop_frac | 0.133 [0.107,0.156] | 0.0754 [0.026,0.128] |

## H3 — leave-one-question-UID-out prediction of future eviction damage (log10 mean KL over the gold continuation)

Ridge on log-magnitude features; Spearman within each held-out prompt, mean [95% prompt bootstrap]. D and F are offline oracles (query recomputation / future-query attention); S is the cheap one-pass surrogate.

| model | dataset | features | mean ρ [CI] | AUROC top-10% harm |
|---|---|---|---|---|
| huginn-0125 | v6_W1 | A attention/position/lexical | 0.42 [0.252,0.569] | 0.711 [0.613,0.784] |
| huginn-0125 | v6_W1 | B = A + source trajectory | 0.42 [0.261,0.556] | 0.718 [0.617,0.798] |
| huginn-0125 | v6_W1 | C = A + value/local-perturbation | 0.427 [0.264,0.571] | 0.709 [0.595,0.785] |
| huginn-0125 | v6_W1 | S = C + cheap surrogates | 0.425 [0.254,0.577] | 0.711 [0.597,0.785] |
| huginn-0125 | v6_W1 | L = C + 2-token format lookahead (deployable, +2 steps) | 0.466 [0.316,0.605] | 0.731 [0.585,0.826] |
| huginn-0125 | v6_W1 | D = C + prompt-anchor query probes (oracle) | 0.377 [0.264,0.453] | 0.704 [0.617,0.774] |
| huginn-0125 | v6_W1 | F = C + future-query census (oracle) | 0.75 [0.685,0.789] | 0.951 [0.902,0.985] |
| huginn-0125 | v6_W1 | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.735 [0.666,0.796] | 0.959 [0.936,0.983] |
| huginn-0125 | v6_W1 | G1g = F + propagation gains only at t* (oracle) | 0.712 [0.641,0.777] | 0.948 [0.905,0.982] |
| huginn-0125 | v6_W1 | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.767 [0.702,0.828] | 0.967 [0.950,0.987] |
| huginn-0125 | v6_W1 | G = F + all future-query probes (oracle) | 0.737 [0.689,0.780] | 0.963 [0.941,0.984] |
| huginn-0125 | v6_W1 | D+F | 0.729 [0.667,0.773] | 0.943 [0.898,0.976] |
| huginn-0125 | v6_W1 | LIFT B − A | 0.000574 [-0.014,0.018] | — |
| huginn-0125 | v6_W1 | LIFT C − A | 0.00761 [-0.004,0.020] | — |
| huginn-0125 | v6_W1 | LIFT S − C | -0.00232 [-0.010,0.005] | — |
| huginn-0125 | v6_W1 | LIFT L − C | 0.039 [0.016,0.062] | — |
| huginn-0125 | v6_W1 | LIFT F − L | 0.284 [0.159,0.447] | — |
| huginn-0125 | v6_W1 | LIFT D − C | -0.05 [-0.126,-0.000] | — |
| huginn-0125 | v6_W1 | LIFT F − C | 0.323 [0.192,0.512] | — |
| huginn-0125 | v6_W1 | LIFT G1 − F | -0.0156 [-0.040,0.013] | — |
| huginn-0125 | v6_W1 | LIFT G1g − F | -0.0384 [-0.073,-0.004] | — |
| huginn-0125 | v6_W1 | LIFT G2 − F | 0.0174 [-0.011,0.043] | — |
| huginn-0125 | v6_W1 | LIFT D+F − F | -0.0213 [-0.040,-0.009] | — |
| huginn-0125 | v6_W2 | A attention/position/lexical | 0.0414 [-0.012,0.091] | 0.472 [0.425,0.519] |
| huginn-0125 | v6_W2 | B = A + source trajectory | 0.106 [0.040,0.171] | 0.518 [0.437,0.598] |
| huginn-0125 | v6_W2 | C = A + value/local-perturbation | 0.0541 [-0.011,0.106] | 0.477 [0.421,0.533] |
| huginn-0125 | v6_W2 | S = C + cheap surrogates | 0.0475 [-0.011,0.106] | 0.487 [0.437,0.549] |
| huginn-0125 | v6_W2 | L = C + 2-token format lookahead (deployable, +2 steps) | 0.104 [0.035,0.144] | 0.516 [0.449,0.584] |
| huginn-0125 | v6_W2 | D = C + prompt-anchor query probes (oracle) | 0.0319 [-0.076,0.139] | 0.464 [0.432,0.514] |
| huginn-0125 | v6_W2 | F = C + future-query census (oracle) | 0.836 [0.807,0.871] | 0.874 [0.829,0.918] |
| huginn-0125 | v6_W2 | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.859 [0.827,0.891] | 0.913 [0.892,0.929] |
| huginn-0125 | v6_W2 | G1g = F + propagation gains only at t* (oracle) | 0.832 [0.795,0.877] | 0.883 [0.841,0.923] |
| huginn-0125 | v6_W2 | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.912 [0.872,0.937] | 0.981 [0.973,0.991] |
| huginn-0125 | v6_W2 | G = F + all future-query probes (oracle) | 0.903 [0.868,0.925] | 0.973 [0.961,0.986] |
| huginn-0125 | v6_W2 | D+F | 0.806 [0.753,0.857] | 0.859 [0.805,0.895] |
| huginn-0125 | v6_W2 | LIFT B − A | 0.0643 [0.003,0.137] | — |
| huginn-0125 | v6_W2 | LIFT C − A | 0.0127 [-0.009,0.034] | — |
| huginn-0125 | v6_W2 | LIFT S − C | -0.00661 [-0.034,0.020] | — |
| huginn-0125 | v6_W2 | LIFT L − C | 0.0494 [0.026,0.080] | — |
| huginn-0125 | v6_W2 | LIFT F − L | 0.732 [0.683,0.781] | — |
| huginn-0125 | v6_W2 | LIFT D − C | -0.0222 [-0.083,0.034] | — |
| huginn-0125 | v6_W2 | LIFT F − C | 0.782 [0.720,0.841] | — |
| huginn-0125 | v6_W2 | LIFT G1 − F | 0.0228 [0.009,0.039] | — |
| huginn-0125 | v6_W2 | LIFT G1g − F | -0.00411 [-0.017,0.009] | — |
| huginn-0125 | v6_W2 | LIFT G2 − F | 0.0758 [0.040,0.112] | — |
| huginn-0125 | v6_W2 | LIFT D+F − F | -0.0299 [-0.081,0.002] | — |
| ouro-2.6b | v6_W1 | A attention/position/lexical | -0.0174 [-0.141,0.154] | 0.481 [0.362,0.662] |
| ouro-2.6b | v6_W1 | B = A + source trajectory | -0.000747 [-0.190,0.216] | 0.517 [0.355,0.701] |
| ouro-2.6b | v6_W1 | C = A + value/local-perturbation | -0.00919 [-0.145,0.160] | 0.478 [0.355,0.672] |
| ouro-2.6b | v6_W1 | S = C + cheap surrogates | -0.00971 [-0.145,0.159] | 0.477 [0.354,0.672] |
| ouro-2.6b | v6_W1 | L = C + 2-token format lookahead (deployable, +2 steps) | 0.00445 [-0.124,0.164] | 0.458 [0.321,0.660] |
| ouro-2.6b | v6_W1 | D = C + prompt-anchor query probes (oracle) | 0.0148 [-0.093,0.165] | 0.47 [0.326,0.630] |
| ouro-2.6b | v6_W1 | F = C + future-query census (oracle) | 0.701 [0.614,0.789] | 0.854 [0.771,0.910] |
| ouro-2.6b | v6_W1 | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.718 [0.649,0.807] | 0.899 [0.838,0.948] |
| ouro-2.6b | v6_W1 | G1g = F + propagation gains only at t* (oracle) | 0.67 [0.551,0.789] | 0.859 [0.759,0.927] |
| ouro-2.6b | v6_W1 | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.748 [0.682,0.822] | 0.92 [0.883,0.946] |
| ouro-2.6b | v6_W1 | G = F + all future-query probes (oracle) | 0.762 [0.705,0.822] | 0.93 [0.884,0.956] |
| ouro-2.6b | v6_W1 | D+F | 0.707 [0.623,0.794] | 0.869 [0.800,0.917] |
| ouro-2.6b | v6_W1 | LIFT B − A | 0.0166 [-0.056,0.089] | — |
| ouro-2.6b | v6_W1 | LIFT C − A | 0.00817 [-0.023,0.042] | — |
| ouro-2.6b | v6_W1 | LIFT S − C | -0.000521 [-0.004,0.003] | — |
| ouro-2.6b | v6_W1 | LIFT L − C | 0.0136 [0.002,0.023] | — |
| ouro-2.6b | v6_W1 | LIFT F − L | 0.697 [0.619,0.747] | — |
| ouro-2.6b | v6_W1 | LIFT D − C | 0.024 [-0.035,0.083] | — |
| ouro-2.6b | v6_W1 | LIFT F − C | 0.71 [0.621,0.767] | — |
| ouro-2.6b | v6_W1 | LIFT G1 − F | 0.0169 [-0.021,0.044] | — |
| ouro-2.6b | v6_W1 | LIFT G1g − F | -0.0314 [-0.107,0.017] | — |
| ouro-2.6b | v6_W1 | LIFT G2 − F | 0.0465 [0.001,0.091] | — |
| ouro-2.6b | v6_W1 | LIFT D+F − F | 0.00617 [-0.004,0.013] | — |
| ouro-2.6b | v6_W2 | A attention/position/lexical | 0.231 [0.100,0.430] | 0.565 [0.512,0.617] |
| ouro-2.6b | v6_W2 | B = A + source trajectory | 0.359 [0.223,0.551] | 0.679 [0.620,0.739] |
| ouro-2.6b | v6_W2 | C = A + value/local-perturbation | 0.23 [0.071,0.433] | 0.548 [0.488,0.608] |
| ouro-2.6b | v6_W2 | S = C + cheap surrogates | 0.219 [0.040,0.434] | 0.544 [0.481,0.608] |
| ouro-2.6b | v6_W2 | L = C + 2-token format lookahead (deployable, +2 steps) | 0.209 [0.025,0.403] | 0.532 [0.465,0.588] |
| ouro-2.6b | v6_W2 | D = C + prompt-anchor query probes (oracle) | 0.268 [0.142,0.447] | 0.543 [0.455,0.632] |
| ouro-2.6b | v6_W2 | F = C + future-query census (oracle) | 0.727 [0.665,0.790] | 0.867 [0.830,0.915] |
| ouro-2.6b | v6_W2 | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.733 [0.695,0.771] | 0.885 [0.836,0.924] |
| ouro-2.6b | v6_W2 | G1g = F + propagation gains only at t* (oracle) | 0.709 [0.666,0.753] | 0.859 [0.827,0.903] |
| ouro-2.6b | v6_W2 | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.803 [0.769,0.837] | 0.923 [0.890,0.955] |
| ouro-2.6b | v6_W2 | G = F + all future-query probes (oracle) | 0.782 [0.756,0.808] | 0.922 [0.894,0.951] |
| ouro-2.6b | v6_W2 | D+F | 0.718 [0.660,0.776] | 0.858 [0.829,0.906] |
| ouro-2.6b | v6_W2 | LIFT B − A | 0.128 [0.104,0.157] | — |
| ouro-2.6b | v6_W2 | LIFT C − A | -0.000997 [-0.041,0.035] | — |
| ouro-2.6b | v6_W2 | LIFT S − C | -0.0111 [-0.028,0.001] | — |
| ouro-2.6b | v6_W2 | LIFT L − C | -0.021 [-0.055,0.013] | — |
| ouro-2.6b | v6_W2 | LIFT F − L | 0.518 [0.384,0.705] | — |
| ouro-2.6b | v6_W2 | LIFT D − C | 0.0381 [0.014,0.071] | — |
| ouro-2.6b | v6_W2 | LIFT F − C | 0.497 [0.354,0.659] | — |
| ouro-2.6b | v6_W2 | LIFT G1 − F | 0.00548 [-0.034,0.030] | — |
| ouro-2.6b | v6_W2 | LIFT G1g − F | -0.0179 [-0.042,0.001] | — |
| ouro-2.6b | v6_W2 | LIFT G2 − F | 0.076 [0.042,0.105] | — |
| ouro-2.6b | v6_W2 | LIFT D+F − F | -0.00915 [-0.015,-0.005] | — |

## Gate D — native dynamic exit under post-prefill single-token eviction (label-blind subset)

| model | dataset | n (prompt×source) | clean mean depth | frac queries depth-changed | mean Σdepth Δ | mean fixed ΔNLL | mean replay ΔNLL | mean live ΔNLL | live−replay [CI] | EM full→evict |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 192 | 25.8 | 0.291 | 0.026 | 0.00792 | 0.0105 | 0.00971 | -0.00083 [-0.002,0.001] | 1→0.99 |
| huginn-0125 | hotpotqa_distractor | 192 | 22.8 | 0.211 | 0.104 | -0.00361 | 0.00197 | 0.00375 | 0.00179 [0.000,0.004] | 0.25→0.25 |
| ouro-2.6b | controlled_retrieval | 192 | 3.45 | 0.00833 | 0.0833 | 0.012 | 0.0104 | 0.0112 | 0.000891 [-0.000,0.003] | 1→0.984 |
| ouro-2.6b | hotpotqa_distractor | 192 | 3.74 | 0.00964 | 0.0208 | 0.000159 | 3.1e-05 | 8.49e-05 | 5.39e-05 [-0.000,0.000] | 0.5→0.505 |

| model | dataset | ρ(frac depth-changed, fixed ΔNLL) | ρ(Σdepth Δ, fixed ΔNLL) | frac Σdepth Δ>0 | frac <0 |
|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0.161 | 0.0687 | 0.401 | 0.422 |
| huginn-0125 | hotpotqa_distractor | 0.147 | 0.225 | 0.401 | 0.333 |
| ouro-2.6b | controlled_retrieval | 0.0259 | 0.0259 | 0.0417 | 0 |
| ouro-2.6b | hotpotqa_distractor | 0.108 | 0.122 | 0.0312 | 0.0104 |

## Clean-only exit calibration (for freezing live thresholds; no eviction data used)

| model / phase / threshold | n tokens | mean depth | frac at R | histogram |
|---|---|---|---|---|
| ouro-2.6b/prompt_exit/q=0.5 | 7414 | 3.25 | 0.298 | {'2': 353, '3': 4852, '4': 2209} |
| ouro-2.6b/prompt_exit/q=0.6 | 7414 | 3.44 | 0.44 | {'2': 16, '3': 4136, '4': 3262} |
| ouro-2.6b/prompt_exit/q=0.7 | 7414 | 3.65 | 0.65 | {'2': 16, '3': 2582, '4': 4816} |
| ouro-2.6b/prompt_exit/q=0.8 | 7414 | 4 | 0.997 | {'2': 16, '3': 4, '4': 7394} |
| ouro-2.6b/prompt_exit/q=0.9 | 7414 | 4 | 0.997 | {'2': 16, '3': 3, '4': 7395} |
| ouro-2.6b/prompt_exit/q=0.95 | 7414 | 4 | 0.998 | {'2': 16, '3': 2, '4': 7396} |
| ouro-2.6b/gen_exit/q=0.5 | 141 | 3.38 | 0.383 | {'3': 87, '4': 54} |
| ouro-2.6b/gen_exit/q=0.6 | 141 | 3.43 | 0.426 | {'3': 81, '4': 60} |
| ouro-2.6b/gen_exit/q=0.7 | 141 | 3.6 | 0.596 | {'3': 57, '4': 84} |
| ouro-2.6b/gen_exit/q=0.8 | 141 | 4 | 1 | {'4': 141} |
| ouro-2.6b/gen_exit/q=0.9 | 141 | 4 | 1 | {'4': 141} |
| ouro-2.6b/gen_exit/q=0.95 | 141 | 4 | 1 | {'4': 141} |
| huginn-0125/prompt_exit/cos=0.01 | 7922 | 12.4 | 0.000126 | {'2': 16, '4': 16, '5': 32, '6': 16, '7': 57, '8': 39, '9': 137, '10': 630, '11': 1207, '12': 1904, '13': 1929, '14': 1160, '15': 521, '16': 185, '17': 46, '18': 15, '19': 7, '20': 1, '22': 1, '23': 2, '25': 1} |
| huginn-0125/prompt_exit/cos=0.003 | 7922 | 16 | 0.000884 | {'2': 16, '4': 16, '5': 4, '6': 28, '9': 29, '10': 32, '11': 70, '12': 227, '13': 537, '14': 912, '15': 1230, '16': 1508, '17': 1460, '18': 916, '19': 519, '20': 260, '21': 74, '22': 36, '23': 19, '24': 8, '25': 5, '26': 1, '27': 2, '28': 2, '30': 4, '32': 7} |
| huginn-0125/prompt_exit/cos=0.001 | 7922 | 19.6 | 0.00442 | {'3': 16, '5': 16, '6': 13, '7': 19, '11': 24, '12': 19, '13': 45, '14': 88, '15': 260, '16': 480, '17': 714, '18': 869, '19': 1063, '20': 1289, '21': 1133, '22': 847, '23': 478, '24': 262, '25': 122, '26': 64, '27': 33, '28': 11, '29': 16, '30': 3, '31': 3, '32': 35} |
| huginn-0125/prompt_exit/cos=0.0003 | 7922 | 24.6 | 0.0377 | {'3': 16, '5': 10, '6': 6, '7': 9, '8': 21, '9': 2, '13': 11, '14': 18, '15': 15, '16': 50, '17': 41, '18': 125, '19': 277, '20': 397, '21': 521, '22': 618, '23': 617, '24': 837, '25': 1007, '26': 980, '27': 751, '28': 505, '29': 393, '30': 257, '31': 139, '32': 299} |
| huginn-0125/gen_exit/cos=0.01 | 141 | 15.3 | 0.00709 | {'11': 1, '12': 13, '13': 32, '14': 16, '15': 21, '16': 19, '17': 16, '18': 7, '19': 2, '20': 6, '21': 4, '22': 1, '23': 2, '25': 1} |
| huginn-0125/gen_exit/cos=0.003 | 141 | 19.8 | 0.00709 | {'14': 1, '16': 13, '17': 25, '18': 16, '19': 24, '20': 11, '21': 16, '22': 11, '23': 5, '24': 7, '25': 3, '26': 4, '27': 2, '28': 1, '29': 1, '32': 1} |
| huginn-0125/gen_exit/cos=0.001 | 141 | 24.6 | 0.0496 | {'19': 2, '20': 8, '21': 8, '22': 20, '23': 21, '24': 27, '25': 8, '26': 11, '27': 10, '28': 7, '29': 7, '30': 4, '31': 1, '32': 7} |
| huginn-0125/gen_exit/cos=0.0003 | 141 | 30.1 | 0.418 | {'23': 1, '24': 2, '25': 1, '26': 3, '27': 8, '28': 20, '29': 13, '30': 20, '31': 14, '32': 59} |
