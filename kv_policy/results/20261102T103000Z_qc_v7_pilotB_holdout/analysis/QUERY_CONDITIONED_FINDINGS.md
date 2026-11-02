# v7 query-conditioned sensitivity — findings (20261102T103000Z_qc_v7_pilotB_holdout)

Completeness: 610/610 jobs, errors 0, problems 0, complete=True

Sources in table: 18951 over 48 units; per model/dataset: huginn-0125/controlled_retrieval: 4224 sources / 12 prompts, huginn-0125/hotpotqa_distractor: 5270 sources / 12 prompts, ouro-2.6b/controlled_retrieval: 4224 sources / 12 prompts, ouro-2.6b/hotpotqa_distractor: 5233 sources / 12 prompts

## Outcome structure: heavy tail and evidence labels (post-hoc stratification only)

| model | dataset | n sources | share of total KL in top 5% | mean KL evidence | mean KL non-evidence | evidence share among top-5% | mean ΔNLL_future evidence |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 4224 | 0.98 | 0.106 | 0.000206 | 0.412 | 0.107 |
| huginn-0125 | hotpotqa_distractor | 5270 | 0.892 | 0.00852 | 0.000365 | 0.561 | 0.0112 |
| ouro-2.6b | controlled_retrieval | 4224 | 0.986 | 0.0983 | 0.000101 | 0.341 | 0.109 |
| ouro-2.6b | hotpotqa_distractor | 5233 | 0.901 | 0.00852 | 0.000257 | 0.553 | 0.0133 |

## Query-conditioned probe magnitudes vs the numerical floor

Sources binned by clean question-anchor attention (within prompt deciles). The lowest decile serves as the empirical 'near-zero perturbation' floor: in BF16 any mask changes rounding, so hidden-state E has a floor even when the removed attention mass is ~0; the question is whether higher-attention sources rise above it.

| model | dataset | attention decile | n | median qa all-loop KL | median E_R | median ‖Δh_R‖ | median top-1 flip |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0 | 432 | 5.89e-06 | 0.0144 | 1.1 | 0 |
| huginn-0125 | controlled_retrieval | 1 | 420 | 6.82e-06 | 0.0145 | 1.11 | 0 |
| huginn-0125 | controlled_retrieval | 2 | 420 | 8.17e-06 | 0.0145 | 1.1 | 0 |
| huginn-0125 | controlled_retrieval | 3 | 420 | 6.23e-06 | 0.0145 | 1.1 | 0 |
| huginn-0125 | controlled_retrieval | 4 | 420 | 6.87e-06 | 0.0145 | 1.11 | 0 |
| huginn-0125 | controlled_retrieval | 5 | 420 | 6.04e-06 | 0.0146 | 1.11 | 0 |
| huginn-0125 | controlled_retrieval | 6 | 420 | 7.09e-06 | 0.0146 | 1.11 | 0 |
| huginn-0125 | controlled_retrieval | 7 | 420 | 7.97e-06 | 0.0145 | 1.11 | 0 |
| huginn-0125 | controlled_retrieval | 8 | 420 | 7.36e-06 | 0.0146 | 1.11 | 0 |
| huginn-0125 | controlled_retrieval | 9 | 432 | 8.28e-06 | 0.0151 | 1.15 | 0 |
| huginn-0125 | hotpotqa_distractor | 0 | 534 | 2.64e-05 | 0.0145 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 1 | 524 | 2.75e-05 | 0.0145 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 2 | 525 | 3.03e-05 | 0.0145 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 3 | 527 | 2.81e-05 | 0.0146 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 4 | 528 | 2.65e-05 | 0.0146 | 1.12 | 0 |
| huginn-0125 | hotpotqa_distractor | 5 | 522 | 2.8e-05 | 0.0146 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 6 | 527 | 2.79e-05 | 0.0145 | 1.11 | 0 |
| huginn-0125 | hotpotqa_distractor | 7 | 525 | 2.7e-05 | 0.0147 | 1.12 | 0 |
| huginn-0125 | hotpotqa_distractor | 8 | 524 | 2.73e-05 | 0.0148 | 1.13 | 0 |
| huginn-0125 | hotpotqa_distractor | 9 | 534 | 2.84e-05 | 0.0154 | 1.18 | 0 |
| ouro-2.6b | controlled_retrieval | 0 | 432 | 3.77e-05 | 0.0129 | 0.623 | 0 |
| ouro-2.6b | controlled_retrieval | 1 | 420 | 3.32e-05 | 0.013 | 0.626 | 0 |
| ouro-2.6b | controlled_retrieval | 2 | 420 | 3.09e-05 | 0.013 | 0.625 | 0 |
| ouro-2.6b | controlled_retrieval | 3 | 420 | 3.8e-05 | 0.0131 | 0.63 | 0 |
| ouro-2.6b | controlled_retrieval | 4 | 420 | 3.07e-05 | 0.0129 | 0.622 | 0 |
| ouro-2.6b | controlled_retrieval | 5 | 420 | 3.51e-05 | 0.013 | 0.625 | 0 |
| ouro-2.6b | controlled_retrieval | 6 | 420 | 3.85e-05 | 0.013 | 0.625 | 0 |
| ouro-2.6b | controlled_retrieval | 7 | 420 | 3.5e-05 | 0.0131 | 0.63 | 0 |
| ouro-2.6b | controlled_retrieval | 8 | 420 | 3.45e-05 | 0.0132 | 0.634 | 0 |
| ouro-2.6b | controlled_retrieval | 9 | 432 | 3.92e-05 | 0.0138 | 0.663 | 0 |
| ouro-2.6b | hotpotqa_distractor | 0 | 529 | 0.000113 | 0.0138 | 0.642 | 0 |
| ouro-2.6b | hotpotqa_distractor | 1 | 522 | 0.000115 | 0.0139 | 0.644 | 0 |
| ouro-2.6b | hotpotqa_distractor | 2 | 522 | 0.000113 | 0.0139 | 0.646 | 0 |
| ouro-2.6b | hotpotqa_distractor | 3 | 521 | 0.000114 | 0.014 | 0.649 | 0 |
| ouro-2.6b | hotpotqa_distractor | 4 | 525 | 0.000107 | 0.014 | 0.652 | 0 |
| ouro-2.6b | hotpotqa_distractor | 5 | 521 | 0.00011 | 0.0141 | 0.656 | 0 |
| ouro-2.6b | hotpotqa_distractor | 6 | 520 | 0.000112 | 0.0141 | 0.66 | 0 |
| ouro-2.6b | hotpotqa_distractor | 7 | 523 | 0.000108 | 0.0142 | 0.659 | 0 |
| ouro-2.6b | hotpotqa_distractor | 8 | 521 | 0.000114 | 0.0144 | 0.667 | 0 |
| ouro-2.6b | hotpotqa_distractor | 9 | 529 | 0.000109 | 0.0148 | 0.685 | 0 |

## One-loop query masks: propagation gain E_R / E_s by masked loop s (median over sources; IQR)

| model | dataset | anchor | s | n | median E_s | median E_R | median gain | IQR gain | median KL | p90 KL |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | question anchor | 1 | 4224 | 0.00695 | 0.0145 | 2.09 | [1.98,2.19] | 7.05e-06 | 0.00038 |
| huginn-0125 | controlled_retrieval | question anchor | 2 | 4224 | 0.00675 | 0.0145 | 2.14 | [2.02,2.25] | 6.81e-06 | 0.00038 |
| huginn-0125 | controlled_retrieval | question anchor | 4 | 4224 | 0.00674 | 0.0145 | 2.14 | [2.01,2.26] | 7.3e-06 | 0.00038 |
| huginn-0125 | controlled_retrieval | question anchor | 8 | 4224 | 0.00644 | 0.0144 | 2.24 | [2.14,2.35] | 6.95e-06 | 0.000381 |
| huginn-0125 | controlled_retrieval | question anchor | 16 | 4224 | 0.00632 | 0.0145 | 2.28 | [2.18,2.4] | 7.37e-06 | 0.000379 |
| huginn-0125 | controlled_retrieval | question anchor | 24 | 4224 | 0.00639 | 0.0143 | 2.24 | [2.14,2.34] | 6.85e-06 | 0.00038 |
| huginn-0125 | controlled_retrieval | question anchor | 32 | 4224 | 0.00634 | 0.00634 | 1 | [1,1] | 3.05e-06 | 4.32e-05 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 1 | 4224 | 0.00742 | 0.0141 | 1.89 | [1.7,2.05] | 2.03e-06 | 0.000407 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 2 | 4224 | 0.00722 | 0.0141 | 1.94 | [1.66,2.12] | 1.92e-06 | 0.000404 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 4 | 4224 | 0.00766 | 0.0141 | 1.83 | [1.48,2.06] | 2.16e-06 | 0.000398 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 8 | 4224 | 0.0071 | 0.0141 | 1.97 | [1.65,2.16] | 2.04e-06 | 0.000404 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 16 | 4224 | 0.00715 | 0.0141 | 1.97 | [1.63,2.16] | 2.12e-06 | 0.000399 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 24 | 4224 | 0.00715 | 0.0141 | 1.95 | [1.62,2.15] | 2.38e-06 | 0.000415 |
| huginn-0125 | controlled_retrieval | future consuming query t* | 32 | 4224 | 0.00713 | 0.00713 | 1 | [1,1] | 1.15e-06 | 0.000394 |
| huginn-0125 | hotpotqa_distractor | question anchor | 1 | 5270 | 0.00675 | 0.0145 | 2.14 | [2,2.28] | 2.69e-05 | 0.000419 |
| huginn-0125 | hotpotqa_distractor | question anchor | 2 | 5270 | 0.00686 | 0.0145 | 2.11 | [2.01,2.22] | 2.67e-05 | 0.000416 |
| huginn-0125 | hotpotqa_distractor | question anchor | 4 | 5270 | 0.00677 | 0.0145 | 2.14 | [2.02,2.26] | 2.6e-05 | 0.000413 |
| huginn-0125 | hotpotqa_distractor | question anchor | 8 | 5270 | 0.00647 | 0.0145 | 2.23 | [2.12,2.34] | 2.65e-05 | 0.000392 |
| huginn-0125 | hotpotqa_distractor | question anchor | 16 | 5270 | 0.00635 | 0.0145 | 2.28 | [2.16,2.41] | 2.61e-05 | 0.000394 |
| huginn-0125 | hotpotqa_distractor | question anchor | 24 | 5270 | 0.00634 | 0.0143 | 2.25 | [2.14,2.35] | 2.62e-05 | 0.000395 |
| huginn-0125 | hotpotqa_distractor | question anchor | 32 | 5270 | 0.00639 | 0.00639 | 1 | [1,1] | 5.56e-06 | 5.34e-05 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 1 | 5270 | 0.0068 | 0.0128 | 1.9 | [1.73,2.07] | 5.94e-05 | 0.000696 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 2 | 5270 | 0.00674 | 0.0128 | 1.92 | [1.75,2.07] | 5.94e-05 | 0.0007 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 4 | 5270 | 0.0072 | 0.0128 | 1.77 | [1.55,1.96] | 5.98e-05 | 0.0007 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 8 | 5270 | 0.00679 | 0.0128 | 1.88 | [1.68,2.05] | 5.88e-05 | 0.000696 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 16 | 5270 | 0.00676 | 0.0128 | 1.88 | [1.71,2.04] | 5.89e-05 | 0.000718 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 24 | 5270 | 0.00675 | 0.0128 | 1.88 | [1.71,2.03] | 6.02e-05 | 0.000748 |
| huginn-0125 | hotpotqa_distractor | future consuming query t* | 32 | 5270 | 0.00673 | 0.00673 | 1 | [1,1] | 3.88e-05 | 0.0005 |
| ouro-2.6b | controlled_retrieval | question anchor | 1 | 4224 | 0.0111 | 0.013 | 1.16 | [1.09,1.24] | 3.47e-05 | 0.000194 |
| ouro-2.6b | controlled_retrieval | question anchor | 2 | 4224 | 0.0119 | 0.0128 | 1.08 | [1.02,1.14] | 3.36e-05 | 0.00025 |
| ouro-2.6b | controlled_retrieval | question anchor | 3 | 4224 | 0.0116 | 0.0125 | 1.08 | [1.03,1.14] | 3.14e-05 | 0.000238 |
| ouro-2.6b | controlled_retrieval | question anchor | 4 | 4224 | 0.0115 | 0.0115 | 1 | [1,1] | 2.86e-05 | 0.000232 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 1 | 4224 | 0.0107 | 0.0146 | 1.3 | [1.08,1.85] | 5e-06 | 6.92e-05 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 2 | 4224 | 0.0122 | 0.0144 | 1.19 | [1.02,1.41] | 4.63e-06 | 5.91e-05 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 3 | 4224 | 0.013 | 0.0137 | 1.06 | [0.95,1.16] | 3.97e-06 | 5.66e-05 |
| ouro-2.6b | controlled_retrieval | future consuming query t* | 4 | 4224 | 0.0127 | 0.0127 | 1 | [1,1] | 3.48e-06 | 7.56e-05 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 1 | 5233 | 0.0122 | 0.0138 | 1.14 | [0.966,1.25] | 0.000111 | 0.000392 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 2 | 5233 | 0.0124 | 0.0137 | 1.1 | [1.03,1.18] | 0.000109 | 0.000379 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 3 | 5233 | 0.0127 | 0.0133 | 1.04 | [0.979,1.11] | 9.27e-05 | 0.000414 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 4 | 5233 | 0.0124 | 0.0124 | 1 | [1,1] | 8.22e-05 | 0.000481 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 1 | 5233 | 0.0134 | 0.014 | 1.07 | [0.924,1.21] | 1.97e-05 | 0.000562 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 2 | 5233 | 0.0132 | 0.0139 | 1.06 | [0.959,1.17] | 2e-05 | 0.000547 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 3 | 5233 | 0.0128 | 0.0133 | 1.05 | [0.95,1.12] | 1.92e-05 | 0.000532 |
| ouro-2.6b | hotpotqa_distractor | future consuming query t* | 4 | 5233 | 0.0123 | 0.0123 | 1 | [1,1] | 1.85e-05 | 0.000507 |

## Within-prompt Spearman of single features with future damage (mean over prompts [95% CI])

| model | dataset | feature | ρ(mean KL) | ρ(mean NLL shift) |
|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | qa_attn_mean | 0.0775 [0.049,0.108] | -0.045 [-0.075,-0.013] |
| huginn-0125 | controlled_retrieval | qa_attn_last | 0.0844 [0.054,0.115] | -0.0434 [-0.071,-0.014] |
| huginn-0125 | controlled_retrieval | qa_attn_maxhead | 0.0646 [0.040,0.090] | -0.0338 [-0.072,0.004] |
| huginn-0125 | controlled_retrieval | recv_mean | 0.0698 [0.039,0.100] | -0.041 [-0.075,-0.007] |
| huginn-0125 | controlled_retrieval | lastq_mean | 0.133 [0.098,0.167] | -0.0326 [-0.064,0.000] |
| huginn-0125 | controlled_retrieval | qb_attn_mean | 0.133 [0.098,0.167] | -0.0326 [-0.064,0.000] |
| huginn-0125 | controlled_retrieval | qa_delta_mean | 0.0771 [0.050,0.107] | -0.0437 [-0.074,-0.012] |
| huginn-0125 | controlled_retrieval | qa_delta_last | 0.0829 [0.054,0.113] | -0.0423 [-0.070,-0.013] |
| huginn-0125 | controlled_retrieval | qa_delta_loopw | 0.0745 [0.048,0.105] | -0.0424 [-0.075,-0.009] |
| huginn-0125 | controlled_retrieval | qa_vnorm_mean | -0.0384 [-0.070,-0.012] | 0.0152 [-0.023,0.058] |
| huginn-0125 | controlled_retrieval | src_U_mean | 0.0246 [0.005,0.049] | -0.0018 [-0.018,0.015] |
| huginn-0125 | controlled_retrieval | src_settle | 0.0587 [0.029,0.090] | -0.0243 [-0.046,-0.002] |
| huginn-0125 | controlled_retrieval | fut_attn_mean | 0.203 [0.160,0.245] | 0.0139 [-0.023,0.050] |
| huginn-0125 | controlled_retrieval | fut_attn_max_t | 0.208 [0.162,0.253] | 0.0256 [-0.012,0.063] |
| huginn-0125 | controlled_retrieval | fut_delta_mean | 0.205 [0.163,0.246] | 0.0115 [-0.024,0.046] |
| huginn-0125 | controlled_retrieval | fut_delta_max_t | 0.211 [0.166,0.256] | 0.0239 [-0.013,0.061] |
| huginn-0125 | controlled_retrieval | fut_attn_at_t | 0.208 [0.162,0.253] | 0.0256 [-0.012,0.063] |
| huginn-0125 | controlled_retrieval | fut_delta_at_t | 0.211 [0.166,0.255] | 0.0239 [-0.013,0.061] |
| huginn-0125 | controlled_retrieval | qa_all_kl | -0.000526 [-0.026,0.023] | -0.0361 [-0.064,-0.011] |
| huginn-0125 | controlled_retrieval | qa_all_E_R | 0.0163 [-0.016,0.043] | 0.00143 [-0.020,0.022] |
| huginn-0125 | controlled_retrieval | qa_all_dn_R | 0.0163 [-0.016,0.043] | 0.00143 [-0.020,0.022] |
| huginn-0125 | controlled_retrieval | qb_all_kl | 0.0284 [0.009,0.049] | -0.0203 [-0.052,0.013] |
| huginn-0125 | controlled_retrieval | qb_all_E_R | 0.0344 [0.005,0.065] | -0.0135 [-0.035,0.010] |
| huginn-0125 | controlled_retrieval | qf_all_kl | 0.149 [0.106,0.190] | -0.00458 [-0.031,0.024] |
| huginn-0125 | controlled_retrieval | qf_all_E_R | 0.198 [0.153,0.244] | 0.0458 [0.009,0.084] |
| huginn-0125 | controlled_retrieval | qf_all_dn_R | 0.198 [0.153,0.244] | 0.0457 [0.010,0.084] |
| huginn-0125 | controlled_retrieval | qf_loop1_gain | -0.0737 [-0.107,-0.043] | 0.0153 [-0.005,0.034] |
| huginn-0125 | controlled_retrieval | qa_loop1_gain | 0.000693 [-0.026,0.031] | 0.0469 [0.022,0.068] |
| huginn-0125 | controlled_retrieval | clean_kl_mean | 0.177 [0.151,0.202] | 0.03 [-0.006,0.072] |
| huginn-0125 | controlled_retrieval | la1_attn | 0.114 [0.079,0.148] | -0.0293 [-0.058,0.002] |
| huginn-0125 | controlled_retrieval | la2_attn | 0.181 [0.141,0.221] | 0.000541 [-0.032,0.033] |
| huginn-0125 | controlled_retrieval | la12_attn_max | 0.187 [0.145,0.227] | 0.00295 [-0.029,0.034] |
| huginn-0125 | controlled_retrieval | la12_delta_max | 0.191 [0.149,0.230] | 0.00337 [-0.028,0.034] |
| huginn-0125 | controlled_retrieval | fut_lastloop_frac | 0.146 [0.125,0.166] | 0.0228 [-0.009,0.053] |
| huginn-0125 | controlled_retrieval | fut_loop_entropy | 0.0523 [0.023,0.085] | 0.0116 [-0.014,0.038] |
| huginn-0125 | controlled_retrieval | qa_lastloop_frac | 0.0688 [0.030,0.103] | -0.017 [-0.037,0.003] |
| huginn-0125 | hotpotqa_distractor | qa_attn_mean | 0.116 [0.070,0.164] | -0.00552 [-0.041,0.030] |
| huginn-0125 | hotpotqa_distractor | qa_attn_last | 0.124 [0.079,0.171] | -0.00497 [-0.040,0.032] |
| huginn-0125 | hotpotqa_distractor | qa_attn_maxhead | 0.0875 [0.044,0.130] | -0.00776 [-0.036,0.023] |
| huginn-0125 | hotpotqa_distractor | recv_mean | 0.0341 [-0.008,0.077] | 0.000288 [-0.025,0.026] |
| huginn-0125 | hotpotqa_distractor | lastq_mean | 0.168 [0.110,0.231] | 0.0148 [-0.024,0.056] |
| huginn-0125 | hotpotqa_distractor | qb_attn_mean | 0.168 [0.110,0.231] | 0.0148 [-0.024,0.056] |
| huginn-0125 | hotpotqa_distractor | qa_delta_mean | 0.123 [0.078,0.170] | -0.00432 [-0.041,0.032] |
| huginn-0125 | hotpotqa_distractor | qa_delta_last | 0.13 [0.084,0.179] | -0.00464 [-0.041,0.033] |
| huginn-0125 | hotpotqa_distractor | qa_delta_loopw | 0.119 [0.074,0.166] | -0.00361 [-0.040,0.033] |
| huginn-0125 | hotpotqa_distractor | qa_vnorm_mean | -0.0409 [-0.063,-0.021] | -0.00534 [-0.027,0.014] |
| huginn-0125 | hotpotqa_distractor | src_U_mean | -0.022 [-0.041,-0.004] | 0.00489 [-0.016,0.029] |
| huginn-0125 | hotpotqa_distractor | src_settle | 0.0668 [0.025,0.112] | 0.0267 [0.007,0.046] |
| huginn-0125 | hotpotqa_distractor | fut_attn_mean | 0.193 [0.135,0.255] | 0.0148 [-0.025,0.062] |
| huginn-0125 | hotpotqa_distractor | fut_attn_max_t | 0.198 [0.139,0.259] | 0.0129 [-0.026,0.058] |
| huginn-0125 | hotpotqa_distractor | fut_delta_mean | 0.196 [0.137,0.257] | 0.0173 [-0.023,0.064] |
| huginn-0125 | hotpotqa_distractor | fut_delta_max_t | 0.2 [0.142,0.261] | 0.0155 [-0.023,0.061] |
| huginn-0125 | hotpotqa_distractor | fut_attn_at_t | 0.198 [0.139,0.259] | 0.0129 [-0.026,0.058] |
| huginn-0125 | hotpotqa_distractor | fut_delta_at_t | 0.201 [0.142,0.261] | 0.0154 [-0.023,0.060] |
| huginn-0125 | hotpotqa_distractor | qa_all_kl | 0.0284 [0.009,0.049] | 0.00318 [-0.013,0.020] |
| huginn-0125 | hotpotqa_distractor | qa_all_E_R | 0.0384 [0.006,0.070] | 0.0151 [-0.011,0.042] |
| huginn-0125 | hotpotqa_distractor | qa_all_dn_R | 0.0384 [0.006,0.070] | 0.0151 [-0.011,0.042] |
| huginn-0125 | hotpotqa_distractor | qb_all_kl | 0.0848 [0.064,0.107] | 2.11e-05 [-0.036,0.042] |
| huginn-0125 | hotpotqa_distractor | qb_all_E_R | 0.105 [0.062,0.149] | -0.02 [-0.052,0.022] |
| huginn-0125 | hotpotqa_distractor | qf_all_kl | 0.193 [0.150,0.239] | -0.00706 [-0.036,0.026] |
| huginn-0125 | hotpotqa_distractor | qf_all_E_R | 0.193 [0.139,0.252] | 0.0116 [-0.025,0.052] |
| huginn-0125 | hotpotqa_distractor | qf_all_dn_R | 0.193 [0.139,0.252] | 0.0115 [-0.025,0.052] |
| huginn-0125 | hotpotqa_distractor | qf_loop1_gain | -0.0164 [-0.057,0.022] | -0.0127 [-0.036,0.010] |
| huginn-0125 | hotpotqa_distractor | qa_loop1_gain | 0.0232 [-0.006,0.053] | 0.0115 [-0.012,0.032] |
| huginn-0125 | hotpotqa_distractor | clean_kl_mean | 0.202 [0.149,0.262] | -0.00825 [-0.039,0.027] |
| huginn-0125 | hotpotqa_distractor | la1_attn | 0.093 [0.050,0.137] | 0.0145 [-0.014,0.041] |
| huginn-0125 | hotpotqa_distractor | la2_attn | 0.201 [0.142,0.262] | 0.0103 [-0.030,0.058] |
| huginn-0125 | hotpotqa_distractor | la12_attn_max | 0.201 [0.142,0.262] | 0.0097 [-0.030,0.057] |
| huginn-0125 | hotpotqa_distractor | la12_delta_max | 0.202 [0.143,0.263] | 0.00983 [-0.030,0.057] |
| huginn-0125 | hotpotqa_distractor | fut_lastloop_frac | 0.124 [0.083,0.168] | -0.0309 [-0.061,0.005] |
| huginn-0125 | hotpotqa_distractor | fut_loop_entropy | -0.0855 [-0.122,-0.041] | -0.00609 [-0.027,0.012] |
| huginn-0125 | hotpotqa_distractor | qa_lastloop_frac | 0.0647 [0.028,0.106] | -0.0122 [-0.036,0.009] |
| ouro-2.6b | controlled_retrieval | qa_attn_mean | 0.0949 [0.067,0.122] | -0.0504 [-0.070,-0.030] |
| ouro-2.6b | controlled_retrieval | qa_attn_last | 0.0975 [0.072,0.124] | -0.0454 [-0.060,-0.030] |
| ouro-2.6b | controlled_retrieval | qa_attn_maxhead | 0.0728 [0.051,0.096] | -0.0384 [-0.057,-0.018] |
| ouro-2.6b | controlled_retrieval | recv_mean | 0.0944 [0.064,0.123] | -0.0364 [-0.062,-0.012] |
| ouro-2.6b | controlled_retrieval | lastq_mean | 0.0852 [0.057,0.113] | -0.049 [-0.070,-0.029] |
| ouro-2.6b | controlled_retrieval | qb_attn_mean | 0.0852 [0.057,0.113] | -0.049 [-0.070,-0.029] |
| ouro-2.6b | controlled_retrieval | qa_delta_mean | 0.1 [0.074,0.127] | -0.0494 [-0.068,-0.030] |
| ouro-2.6b | controlled_retrieval | qa_delta_last | 0.102 [0.077,0.128] | -0.0463 [-0.061,-0.031] |
| ouro-2.6b | controlled_retrieval | qa_delta_loopw | 0.0983 [0.070,0.126] | -0.0503 [-0.072,-0.029] |
| ouro-2.6b | controlled_retrieval | qa_vnorm_mean | 0.0782 [0.053,0.106] | -0.0477 [-0.074,-0.019] |
| ouro-2.6b | controlled_retrieval | src_U_mean | 0.000294 [-0.031,0.035] | -0.0502 [-0.080,-0.023] |
| ouro-2.6b | controlled_retrieval | src_settle | -0.00809 [-0.037,0.019] | -0.0158 [-0.041,0.009] |
| ouro-2.6b | controlled_retrieval | fut_attn_mean | 0.161 [0.140,0.186] | -0.0217 [-0.052,0.011] |
| ouro-2.6b | controlled_retrieval | fut_attn_max_t | 0.153 [0.133,0.177] | -0.011 [-0.044,0.024] |
| ouro-2.6b | controlled_retrieval | fut_delta_mean | 0.167 [0.146,0.192] | -0.0179 [-0.047,0.011] |
| ouro-2.6b | controlled_retrieval | fut_delta_max_t | 0.161 [0.142,0.185] | -0.00934 [-0.041,0.023] |
| ouro-2.6b | controlled_retrieval | fut_attn_at_t | 0.153 [0.133,0.177] | -0.011 [-0.044,0.024] |
| ouro-2.6b | controlled_retrieval | fut_delta_at_t | 0.161 [0.142,0.184] | -0.00951 [-0.041,0.023] |
| ouro-2.6b | controlled_retrieval | qa_all_kl | 0.0354 [0.008,0.065] | -0.0107 [-0.041,0.023] |
| ouro-2.6b | controlled_retrieval | qa_all_E_R | 0.0932 [0.055,0.135] | -0.00493 [-0.040,0.032] |
| ouro-2.6b | controlled_retrieval | qa_all_dn_R | 0.0932 [0.055,0.135] | -0.00493 [-0.040,0.032] |
| ouro-2.6b | controlled_retrieval | qb_all_kl | 0.0284 [-0.002,0.064] | 0.00232 [-0.017,0.022] |
| ouro-2.6b | controlled_retrieval | qb_all_E_R | 0.0317 [0.007,0.059] | 0.0027 [-0.021,0.028] |
| ouro-2.6b | controlled_retrieval | qf_all_kl | 0.097 [0.057,0.135] | 0.0277 [-0.000,0.058] |
| ouro-2.6b | controlled_retrieval | qf_all_E_R | 0.128 [0.095,0.158] | 0.0146 [-0.003,0.032] |
| ouro-2.6b | controlled_retrieval | qf_all_dn_R | 0.135 [0.102,0.164] | 0.0145 [-0.007,0.034] |
| ouro-2.6b | controlled_retrieval | qf_loop1_gain | -0.0985 [-0.140,-0.063] | -0.0284 [-0.046,-0.010] |
| ouro-2.6b | controlled_retrieval | qa_loop1_gain | -0.0535 [-0.092,-0.016] | 0.0274 [-0.018,0.075] |
| ouro-2.6b | controlled_retrieval | clean_kl_mean | 0.126 [0.094,0.162] | -0.0169 [-0.052,0.016] |
| ouro-2.6b | controlled_retrieval | la1_attn | 0.101 [0.073,0.129] | -0.0448 [-0.066,-0.023] |
| ouro-2.6b | controlled_retrieval | la2_attn | 0.15 [0.125,0.173] | -0.0263 [-0.054,0.003] |
| ouro-2.6b | controlled_retrieval | la12_attn_max | 0.152 [0.126,0.176] | -0.0252 [-0.053,0.004] |
| ouro-2.6b | controlled_retrieval | la12_delta_max | 0.156 [0.131,0.180] | -0.0226 [-0.049,0.005] |
| ouro-2.6b | controlled_retrieval | fut_lastloop_frac | 0.0238 [-0.001,0.053] | -0.00786 [-0.038,0.027] |
| ouro-2.6b | controlled_retrieval | fut_loop_entropy | -0.0167 [-0.048,0.017] | 0.0245 [-0.002,0.050] |
| ouro-2.6b | controlled_retrieval | qa_lastloop_frac | 0.0296 [0.001,0.058] | -0.00217 [-0.021,0.017] |
| ouro-2.6b | hotpotqa_distractor | qa_attn_mean | 0.12 [0.099,0.141] | -0.00538 [-0.035,0.019] |
| ouro-2.6b | hotpotqa_distractor | qa_attn_last | 0.115 [0.090,0.143] | -0.0089 [-0.037,0.015] |
| ouro-2.6b | hotpotqa_distractor | qa_attn_maxhead | 0.0957 [0.067,0.124] | 0.00105 [-0.034,0.032] |
| ouro-2.6b | hotpotqa_distractor | recv_mean | 0.0328 [0.005,0.059] | -0.00478 [-0.029,0.021] |
| ouro-2.6b | hotpotqa_distractor | lastq_mean | 0.107 [0.079,0.135] | -0.0138 [-0.031,0.002] |
| ouro-2.6b | hotpotqa_distractor | qb_attn_mean | 0.107 [0.079,0.135] | -0.0138 [-0.031,0.002] |
| ouro-2.6b | hotpotqa_distractor | qa_delta_mean | 0.134 [0.110,0.157] | -0.00334 [-0.036,0.023] |
| ouro-2.6b | hotpotqa_distractor | qa_delta_last | 0.123 [0.097,0.152] | -0.00707 [-0.036,0.016] |
| ouro-2.6b | hotpotqa_distractor | qa_delta_loopw | 0.131 [0.107,0.154] | -0.00419 [-0.038,0.022] |
| ouro-2.6b | hotpotqa_distractor | qa_vnorm_mean | -0.0102 [-0.046,0.029] | 0.0026 [-0.024,0.029] |
| ouro-2.6b | hotpotqa_distractor | src_U_mean | -0.0292 [-0.061,-0.000] | -0.00786 [-0.040,0.022] |
| ouro-2.6b | hotpotqa_distractor | src_settle | 0.0132 [-0.013,0.046] | -0.0155 [-0.042,0.012] |
| ouro-2.6b | hotpotqa_distractor | fut_attn_mean | 0.17 [0.136,0.211] | 0.0225 [-0.005,0.050] |
| ouro-2.6b | hotpotqa_distractor | fut_attn_max_t | 0.167 [0.131,0.209] | 0.0212 [-0.009,0.050] |
| ouro-2.6b | hotpotqa_distractor | fut_delta_mean | 0.176 [0.141,0.217] | 0.0257 [-0.002,0.052] |
| ouro-2.6b | hotpotqa_distractor | fut_delta_max_t | 0.172 [0.137,0.214] | 0.0226 [-0.007,0.050] |
| ouro-2.6b | hotpotqa_distractor | fut_attn_at_t | 0.167 [0.131,0.209] | 0.0212 [-0.009,0.050] |
| ouro-2.6b | hotpotqa_distractor | fut_delta_at_t | 0.17 [0.136,0.211] | 0.0233 [-0.006,0.051] |
| ouro-2.6b | hotpotqa_distractor | qa_all_kl | 0.0323 [0.001,0.062] | 0.00254 [-0.026,0.031] |
| ouro-2.6b | hotpotqa_distractor | qa_all_E_R | 0.0669 [0.035,0.103] | 0.00891 [-0.020,0.035] |
| ouro-2.6b | hotpotqa_distractor | qa_all_dn_R | 0.0669 [0.035,0.103] | 0.00891 [-0.020,0.035] |
| ouro-2.6b | hotpotqa_distractor | qb_all_kl | 0.00896 [-0.011,0.029] | 0.000553 [-0.026,0.026] |
| ouro-2.6b | hotpotqa_distractor | qb_all_E_R | 0.052 [0.023,0.077] | 0.0201 [0.002,0.038] |
| ouro-2.6b | hotpotqa_distractor | qf_all_kl | 0.0856 [0.049,0.120] | 0.0133 [-0.010,0.034] |
| ouro-2.6b | hotpotqa_distractor | qf_all_E_R | 0.147 [0.110,0.184] | 0.0101 [-0.017,0.032] |
| ouro-2.6b | hotpotqa_distractor | qf_all_dn_R | 0.144 [0.107,0.185] | 0.00792 [-0.016,0.030] |
| ouro-2.6b | hotpotqa_distractor | qf_loop1_gain | -0.0871 [-0.136,-0.038] | -0.0336 [-0.055,-0.011] |
| ouro-2.6b | hotpotqa_distractor | qa_loop1_gain | -0.0606 [-0.087,-0.031] | -0.0118 [-0.038,0.018] |
| ouro-2.6b | hotpotqa_distractor | clean_kl_mean | 0.142 [0.098,0.193] | 0.0204 [-0.012,0.048] |
| ouro-2.6b | hotpotqa_distractor | la1_attn | 0.112 [0.081,0.141] | -0.00597 [-0.025,0.012] |
| ouro-2.6b | hotpotqa_distractor | la2_attn | 0.182 [0.139,0.230] | 0.012 [-0.020,0.040] |
| ouro-2.6b | hotpotqa_distractor | la12_attn_max | 0.183 [0.141,0.231] | 0.0129 [-0.019,0.040] |
| ouro-2.6b | hotpotqa_distractor | la12_delta_max | 0.191 [0.148,0.240] | 0.0145 [-0.017,0.041] |
| ouro-2.6b | hotpotqa_distractor | fut_lastloop_frac | 0.0697 [0.036,0.103] | 0.00939 [-0.013,0.032] |
| ouro-2.6b | hotpotqa_distractor | fut_loop_entropy | 0.00882 [-0.024,0.041] | 0.0288 [0.001,0.062] |
| ouro-2.6b | hotpotqa_distractor | qa_lastloop_frac | 0.0209 [-0.007,0.051] | -0.00687 [-0.038,0.024] |

## H1/H2 — attention-matched and local-perturbation-matched source pairs

Pairs are same-role, |Δpos|≤16, within prompt. 'attention' = clean question-anchor attention (deployable); 'future' = the consuming generated query's clean attention (oracle). δ = closed-form local attention-output perturbation at the same query. If Y dispersion stays large under future-attn+δ matching, the damage is not a local/attention quantity at the consuming query either (propagation / recurrence or later history). Pairs where both members have mean KL < 1e-3 (BF16 floor) are excluded as uninformative.

| model | dataset | pairing | n pairs | median |log10 Y ratio| | frac >10× | probe-KL (anchor of the pairing) sign agreement with Y [CI] | future-attn sign agreement [CI] |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | attention-matched | 75 | 1.92 | 0.787 | 0.467 [0.360,0.587] | 0.933 [0.867,0.987] |
| huginn-0125 | controlled_retrieval | attention+δ-matched | 73 | 1.86 | 0.781 | 0.479 [0.370,0.603] | 0.932 [0.863,0.986] |
| huginn-0125 | controlled_retrieval | future-attn-matched | 52 | 0.774 | 0.365 | 0.923 [0.846,0.981] | 0.558 [0.423,0.692] |
| huginn-0125 | controlled_retrieval | future-attn+δ-matched | 43 | 0.716 | 0.302 | 0.93 [0.837,1.000] | 0.465 [0.326,0.605] |
| huginn-0125 | controlled_retrieval | random same-role | 91 | 2.81 | 0.791 | 0.462 [0.352,0.571] | 0.901 [0.835,0.956] |
| huginn-0125 | hotpotqa_distractor | attention-matched | 136 | 1.09 | 0.544 | 0.493 [0.404,0.574] | 0.824 [0.757,0.882] |
| huginn-0125 | hotpotqa_distractor | attention+δ-matched | 134 | 1.1 | 0.545 | 0.485 [0.403,0.567] | 0.821 [0.754,0.888] |
| huginn-0125 | hotpotqa_distractor | future-attn-matched | 105 | 0.791 | 0.305 | 0.829 [0.752,0.895] | 0.657 [0.562,0.743] |
| huginn-0125 | hotpotqa_distractor | future-attn+δ-matched | 96 | 0.758 | 0.281 | 0.812 [0.729,0.896] | 0.625 [0.531,0.719] |
| huginn-0125 | hotpotqa_distractor | random same-role | 163 | 1.1 | 0.564 | 0.564 [0.485,0.638] | 0.853 [0.791,0.902] |
| ouro-2.6b | controlled_retrieval | attention-matched | 43 | 3.56 | 0.767 | 0.651 [0.512,0.791] | 0.93 [0.837,1.000] |
| ouro-2.6b | controlled_retrieval | attention+δ-matched | 41 | 3.56 | 0.78 | 0.659 [0.512,0.805] | 0.951 [0.878,1.000] |
| ouro-2.6b | controlled_retrieval | future-attn-matched | 39 | 1.18 | 0.564 | 0.667 [0.513,0.821] | 0.667 [0.513,0.795] |
| ouro-2.6b | controlled_retrieval | future-attn+δ-matched | 38 | 1.18 | 0.553 | 0.632 [0.474,0.789] | 0.684 [0.526,0.816] |
| ouro-2.6b | controlled_retrieval | random same-role | 63 | 3.33 | 0.921 | 0.524 [0.397,0.651] | 0.905 [0.825,0.968] |
| ouro-2.6b | hotpotqa_distractor | attention-matched | 72 | 1.22 | 0.597 | 0.514 [0.403,0.625] | 0.792 [0.694,0.875] |
| ouro-2.6b | hotpotqa_distractor | attention+δ-matched | 66 | 1.11 | 0.576 | 0.5 [0.379,0.621] | 0.788 [0.682,0.879] |
| ouro-2.6b | hotpotqa_distractor | future-attn-matched | 55 | 0.885 | 0.418 | 0.745 [0.618,0.855] | 0.4 [0.273,0.527] |
| ouro-2.6b | hotpotqa_distractor | future-attn+δ-matched | 52 | 0.879 | 0.423 | 0.731 [0.596,0.846] | 0.442 [0.308,0.577] |
| ouro-2.6b | hotpotqa_distractor | random same-role | 78 | 1.32 | 0.615 | 0.513 [0.410,0.628] | 0.872 [0.795,0.936] |

## Loop profile of clean attention to the source (normalised over loops): top-5% harm vs rest

| model | dataset | group | n | query | first loop | last loop | peak loop | entropy |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | rest | 4008 | consuming query t* | 0.0541 | 0.0285 | 1 | 3.45 |
| huginn-0125 | controlled_retrieval | top5 | 216 | consuming query t* | 0.0107 | 0.0339 | 21 | 3.45 |
| huginn-0125 | controlled_retrieval | rest | 4008 | question anchor | 0.0518 | 0.0264 | 2 | 3.41 |
| huginn-0125 | controlled_retrieval | top5 | 216 | question anchor | 0.0376 | 0.0289 | 2 | 3.44 |
| huginn-0125 | hotpotqa_distractor | rest | 4998 | consuming query t* | 0.0267 | 0.031 | 4 | 3.46 |
| huginn-0125 | hotpotqa_distractor | top5 | 272 | consuming query t* | 0.0111 | 0.0358 | 25 | 3.44 |
| huginn-0125 | hotpotqa_distractor | rest | 4998 | question anchor | 0.0483 | 0.0292 | 1 | 3.45 |
| huginn-0125 | hotpotqa_distractor | top5 | 272 | question anchor | 0.0354 | 0.0308 | 4 | 3.45 |
| ouro-2.6b | controlled_retrieval | rest | 4008 | consuming query t* | 0.219 | 0.295 | 4 | 1.38 |
| ouro-2.6b | controlled_retrieval | top5 | 216 | consuming query t* | 0.226 | 0.304 | 4 | 1.38 |
| ouro-2.6b | controlled_retrieval | rest | 4008 | question anchor | 0.349 | 0.216 | 1 | 1.36 |
| ouro-2.6b | controlled_retrieval | top5 | 216 | question anchor | 0.35 | 0.232 | 1 | 1.36 |
| ouro-2.6b | hotpotqa_distractor | rest | 4965 | consuming query t* | 0.306 | 0.232 | 1 | 1.38 |
| ouro-2.6b | hotpotqa_distractor | top5 | 268 | consuming query t* | 0.242 | 0.277 | 4 | 1.38 |
| ouro-2.6b | hotpotqa_distractor | rest | 4965 | question anchor | 0.378 | 0.187 | 1 | 1.34 |
| ouro-2.6b | hotpotqa_distractor | top5 | 268 | question anchor | 0.352 | 0.187 | 1 | 1.36 |

## Loop-resolved attention importance: ρ(attention at loop r, Y) within prompt (mean over prompts)

| model | dataset | query | r=1 | r=2 | r=3 | r=4 | r=8 | r=16 | r=24 | r=32 | mean over r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | consuming query t* (oracle) | 0.079 | 0.145 | 0.16 | 0.169 | 0.189 | 0.209 | 0.213 | 0.214 | 0.208 |
| huginn-0125 | controlled_retrieval | 2-token lookahead | -0.00456 | 0.0265 | 0.0824 | 0.112 | 0.168 | 0.189 | 0.188 | 0.188 | 0.187 |
| huginn-0125 | controlled_retrieval | question anchor | 0.0152 | 0.062 | 0.0847 | 0.0626 | 0.0914 | 0.0806 | 0.084 | 0.0844 | 0.0775 |
| huginn-0125 | hotpotqa_distractor | consuming query t* (oracle) | 0.0522 | 0.0718 | 0.0956 | 0.101 | 0.156 | 0.2 | 0.203 | 0.202 | 0.198 |
| huginn-0125 | hotpotqa_distractor | 2-token lookahead | 0.0334 | 0.0704 | 0.0883 | 0.0886 | 0.146 | 0.202 | 0.206 | 0.206 | 0.201 |
| huginn-0125 | hotpotqa_distractor | question anchor | 0.0337 | 0.0775 | 0.101 | 0.107 | 0.107 | 0.115 | 0.122 | 0.124 | 0.116 |
| ouro-2.6b | controlled_retrieval | consuming query t* (oracle) | 0.0978 | 0.112 | 0.147 | 0.156 | — | — | — | — | 0.153 |
| ouro-2.6b | controlled_retrieval | 2-token lookahead | 0.0704 | 0.102 | 0.142 | 0.163 | — | — | — | — | 0.152 |
| ouro-2.6b | controlled_retrieval | question anchor | 0.0667 | 0.0899 | 0.0922 | 0.0975 | — | — | — | — | 0.0949 |
| ouro-2.6b | hotpotqa_distractor | consuming query t* (oracle) | 0.112 | 0.147 | 0.174 | 0.173 | — | — | — | — | 0.167 |
| ouro-2.6b | hotpotqa_distractor | 2-token lookahead | 0.0921 | 0.125 | 0.178 | 0.184 | — | — | — | — | 0.183 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 0.0907 | 0.122 | 0.123 | 0.115 | — | — | — | — | 0.12 |

## H3 — leave-one-question-UID-out prediction of future eviction damage (log10 mean KL over the gold continuation)

Ridge on log-magnitude features; Spearman within each held-out prompt, mean [95% prompt bootstrap]. D and F are offline oracles (query recomputation / future-query attention); S is the cheap one-pass surrogate.

| model | dataset | features | mean ρ [CI] | AUROC top-10% harm |
|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | A attention/position/lexical | 0.138 [0.090,0.189] | 0.683 [0.651,0.714] |
| huginn-0125 | controlled_retrieval | B = A + source trajectory | 0.136 [0.091,0.183] | 0.681 [0.656,0.707] |
| huginn-0125 | controlled_retrieval | C = A + value/local-perturbation | 0.135 [0.085,0.190] | 0.684 [0.653,0.715] |
| huginn-0125 | controlled_retrieval | S = C + cheap surrogates | 0.133 [0.084,0.187] | 0.683 [0.652,0.715] |
| huginn-0125 | controlled_retrieval | L = C + 2-token format lookahead (deployable, +2 steps) | 0.151 [0.106,0.202] | 0.699 [0.669,0.727] |
| huginn-0125 | controlled_retrieval | D = C + prompt-anchor query probes (oracle) | 0.14 [0.099,0.186] | 0.682 [0.647,0.716] |
| huginn-0125 | controlled_retrieval | F = C + future-query census (oracle) | 0.193 [0.155,0.234] | 0.73 [0.687,0.768] |
| huginn-0125 | controlled_retrieval | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.2 [0.159,0.241] | 0.748 [0.708,0.785] |
| huginn-0125 | controlled_retrieval | G1g = F + propagation gains only at t* (oracle) | 0.197 [0.156,0.238] | 0.73 [0.680,0.774] |
| huginn-0125 | controlled_retrieval | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.209 [0.169,0.251] | 0.751 [0.719,0.783] |
| huginn-0125 | controlled_retrieval | G = F + all future-query probes (oracle) | 0.21 [0.169,0.254] | 0.748 [0.708,0.786] |
| huginn-0125 | controlled_retrieval | D+F | 0.192 [0.155,0.232] | 0.73 [0.687,0.770] |
| huginn-0125 | controlled_retrieval | LIFT B − A | -0.00183 [-0.008,0.005] | — |
| huginn-0125 | controlled_retrieval | LIFT C − A | -0.0026 [-0.010,0.005] | — |
| huginn-0125 | controlled_retrieval | LIFT S − C | -0.00231 [-0.007,0.002] | — |
| huginn-0125 | controlled_retrieval | LIFT L − C | 0.0158 [0.005,0.027] | — |
| huginn-0125 | controlled_retrieval | LIFT F − L | 0.0422 [0.009,0.076] | — |
| huginn-0125 | controlled_retrieval | LIFT D − C | 0.00488 [-0.009,0.019] | — |
| huginn-0125 | controlled_retrieval | LIFT F − C | 0.058 [0.019,0.098] | — |
| huginn-0125 | controlled_retrieval | LIFT G1 − F | 0.00697 [-0.015,0.029] | — |
| huginn-0125 | controlled_retrieval | LIFT G1g − F | 0.00313 [-0.007,0.011] | — |
| huginn-0125 | controlled_retrieval | LIFT G2 − F | 0.0153 [-0.008,0.036] | — |
| huginn-0125 | controlled_retrieval | LIFT D+F − F | -0.00147 [-0.007,0.004] | — |
| huginn-0125 | hotpotqa_distractor | A attention/position/lexical | 0.17 [0.123,0.223] | 0.719 [0.658,0.784] |
| huginn-0125 | hotpotqa_distractor | B = A + source trajectory | 0.168 [0.119,0.224] | 0.72 [0.658,0.784] |
| huginn-0125 | hotpotqa_distractor | C = A + value/local-perturbation | 0.167 [0.121,0.219] | 0.718 [0.658,0.780] |
| huginn-0125 | hotpotqa_distractor | S = C + cheap surrogates | 0.164 [0.118,0.218] | 0.716 [0.654,0.780] |
| huginn-0125 | hotpotqa_distractor | L = C + 2-token format lookahead (deployable, +2 steps) | 0.179 [0.129,0.229] | 0.76 [0.690,0.828] |
| huginn-0125 | hotpotqa_distractor | D = C + prompt-anchor query probes (oracle) | 0.166 [0.119,0.218] | 0.715 [0.656,0.776] |
| huginn-0125 | hotpotqa_distractor | F = C + future-query census (oracle) | 0.196 [0.144,0.251] | 0.775 [0.707,0.841] |
| huginn-0125 | hotpotqa_distractor | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.199 [0.142,0.259] | 0.777 [0.710,0.840] |
| huginn-0125 | hotpotqa_distractor | G1g = F + propagation gains only at t* (oracle) | 0.191 [0.131,0.252] | 0.779 [0.710,0.843] |
| huginn-0125 | hotpotqa_distractor | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.204 [0.153,0.257] | 0.788 [0.719,0.850] |
| huginn-0125 | hotpotqa_distractor | G = F + all future-query probes (oracle) | 0.201 [0.147,0.260] | 0.786 [0.716,0.848] |
| huginn-0125 | hotpotqa_distractor | D+F | 0.2 [0.148,0.253] | 0.78 [0.713,0.845] |
| huginn-0125 | hotpotqa_distractor | LIFT B − A | -0.00229 [-0.011,0.005] | — |
| huginn-0125 | hotpotqa_distractor | LIFT C − A | -0.00314 [-0.012,0.006] | — |
| huginn-0125 | hotpotqa_distractor | LIFT S − C | -0.00261 [-0.008,0.001] | — |
| huginn-0125 | hotpotqa_distractor | LIFT L − C | 0.0115 [-0.016,0.042] | — |
| huginn-0125 | hotpotqa_distractor | LIFT F − L | 0.0178 [-0.010,0.048] | — |
| huginn-0125 | hotpotqa_distractor | LIFT D − C | -0.00156 [-0.008,0.005] | — |
| huginn-0125 | hotpotqa_distractor | LIFT F − C | 0.0293 [0.001,0.060] | — |
| huginn-0125 | hotpotqa_distractor | LIFT G1 − F | 0.00273 [-0.018,0.024] | — |
| huginn-0125 | hotpotqa_distractor | LIFT G1g − F | -0.00529 [-0.019,0.008] | — |
| huginn-0125 | hotpotqa_distractor | LIFT G2 − F | 0.00713 [-0.001,0.016] | — |
| huginn-0125 | hotpotqa_distractor | LIFT D+F − F | 0.00312 [-0.003,0.009] | — |
| ouro-2.6b | controlled_retrieval | A attention/position/lexical | 0.0948 [0.068,0.123] | 0.623 [0.592,0.654] |
| ouro-2.6b | controlled_retrieval | B = A + source trajectory | 0.0944 [0.067,0.124] | 0.62 [0.590,0.654] |
| ouro-2.6b | controlled_retrieval | C = A + value/local-perturbation | 0.111 [0.085,0.137] | 0.638 [0.608,0.669] |
| ouro-2.6b | controlled_retrieval | S = C + cheap surrogates | 0.113 [0.087,0.140] | 0.64 [0.610,0.672] |
| ouro-2.6b | controlled_retrieval | L = C + 2-token format lookahead (deployable, +2 steps) | 0.132 [0.107,0.158] | 0.665 [0.627,0.700] |
| ouro-2.6b | controlled_retrieval | D = C + prompt-anchor query probes (oracle) | 0.119 [0.089,0.148] | 0.635 [0.601,0.673] |
| ouro-2.6b | controlled_retrieval | F = C + future-query census (oracle) | 0.128 [0.098,0.159] | 0.661 [0.629,0.697] |
| ouro-2.6b | controlled_retrieval | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.155 [0.119,0.186] | 0.696 [0.660,0.733] |
| ouro-2.6b | controlled_retrieval | G1g = F + propagation gains only at t* (oracle) | 0.13 [0.100,0.160] | 0.664 [0.631,0.700] |
| ouro-2.6b | controlled_retrieval | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.157 [0.125,0.184] | 0.705 [0.675,0.737] |
| ouro-2.6b | controlled_retrieval | G = F + all future-query probes (oracle) | 0.161 [0.125,0.192] | 0.701 [0.667,0.735] |
| ouro-2.6b | controlled_retrieval | D+F | 0.137 [0.107,0.167] | 0.669 [0.639,0.702] |
| ouro-2.6b | controlled_retrieval | LIFT B − A | -0.000394 [-0.017,0.016] | — |
| ouro-2.6b | controlled_retrieval | LIFT C − A | 0.0164 [0.006,0.027] | — |
| ouro-2.6b | controlled_retrieval | LIFT S − C | 0.00192 [-0.000,0.004] | — |
| ouro-2.6b | controlled_retrieval | LIFT L − C | 0.0211 [-0.001,0.043] | — |
| ouro-2.6b | controlled_retrieval | LIFT F − L | -0.00408 [-0.037,0.030] | — |
| ouro-2.6b | controlled_retrieval | LIFT D − C | 0.0073 [-0.013,0.030] | — |
| ouro-2.6b | controlled_retrieval | LIFT F − C | 0.017 [-0.011,0.043] | — |
| ouro-2.6b | controlled_retrieval | LIFT G1 − F | 0.0272 [0.009,0.045] | — |
| ouro-2.6b | controlled_retrieval | LIFT G1g − F | 0.00156 [-0.003,0.007] | — |
| ouro-2.6b | controlled_retrieval | LIFT G2 − F | 0.0288 [0.001,0.054] | — |
| ouro-2.6b | controlled_retrieval | LIFT D+F − F | 0.00875 [-0.000,0.017] | — |
| ouro-2.6b | hotpotqa_distractor | A attention/position/lexical | 0.125 [0.096,0.157] | 0.64 [0.607,0.672] |
| ouro-2.6b | hotpotqa_distractor | B = A + source trajectory | 0.124 [0.102,0.153] | 0.638 [0.604,0.671] |
| ouro-2.6b | hotpotqa_distractor | C = A + value/local-perturbation | 0.123 [0.090,0.156] | 0.637 [0.598,0.670] |
| ouro-2.6b | hotpotqa_distractor | S = C + cheap surrogates | 0.121 [0.084,0.155] | 0.634 [0.595,0.667] |
| ouro-2.6b | hotpotqa_distractor | L = C + 2-token format lookahead (deployable, +2 steps) | 0.151 [0.109,0.203] | 0.693 [0.645,0.743] |
| ouro-2.6b | hotpotqa_distractor | D = C + prompt-anchor query probes (oracle) | 0.105 [0.075,0.136] | 0.627 [0.599,0.655] |
| ouro-2.6b | hotpotqa_distractor | F = C + future-query census (oracle) | 0.15 [0.116,0.188] | 0.684 [0.649,0.718] |
| ouro-2.6b | hotpotqa_distractor | G1 = F + one-loop probes at t* (E_s,E_R,gain; oracle) | 0.147 [0.102,0.196] | 0.679 [0.630,0.725] |
| ouro-2.6b | hotpotqa_distractor | G1g = F + propagation gains only at t* (oracle) | 0.147 [0.107,0.193] | 0.689 [0.651,0.726] |
| ouro-2.6b | hotpotqa_distractor | G2 = F + all-loop probe at t* (≈ within-query damage component) | 0.141 [0.095,0.184] | 0.67 [0.628,0.710] |
| ouro-2.6b | hotpotqa_distractor | G = F + all future-query probes (oracle) | 0.125 [0.080,0.169] | 0.658 [0.615,0.697] |
| ouro-2.6b | hotpotqa_distractor | D+F | 0.147 [0.111,0.190] | 0.691 [0.656,0.726] |
| ouro-2.6b | hotpotqa_distractor | LIFT B − A | -0.000569 [-0.011,0.009] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT C − A | -0.00216 [-0.014,0.010] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT S − C | -0.00209 [-0.007,0.002] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT L − C | 0.028 [-0.009,0.066] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT F − L | -0.000822 [-0.030,0.022] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT D − C | -0.0175 [-0.041,0.010] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT F − C | 0.0271 [-0.005,0.067] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT G1 − F | -0.00248 [-0.019,0.015] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT G1g − F | -0.00324 [-0.014,0.010] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT G2 − F | -0.00922 [-0.027,0.009] | — |
| ouro-2.6b | hotpotqa_distractor | LIFT D+F − F | -0.00292 [-0.025,0.020] | — |

## Deferred-eviction lookahead curve: LOPO ranking with the first k generated queries' clean attention

| model | dataset | k | mean ρ [CI] | AUROC top-10% | LIFT over k=0 [CI] | univariate ρ(max attn t≤k) |
|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0 (C) | 0.135 [0.085,0.190] | — | — | — |
| huginn-0125 | controlled_retrieval | 1 | 0.137 [0.087,0.192] | 0.692 | 0.00193 [-0.001,0.005] | 0.114 |
| huginn-0125 | controlled_retrieval | 2 | 0.152 [0.106,0.201] | 0.706 | 0.0168 [0.006,0.029] | 0.187 |
| huginn-0125 | controlled_retrieval | 3 | 0.143 [0.096,0.194] | 0.698 | 0.00787 [0.000,0.017] | 0.179 |
| huginn-0125 | controlled_retrieval | 4 | 0.146 [0.096,0.199] | 0.702 | 0.0106 [0.003,0.018] | 0.186 |
| huginn-0125 | controlled_retrieval | 6 | 0.18 [0.131,0.233] | 0.729 | 0.0448 [0.024,0.066] | 0.205 |
| huginn-0125 | controlled_retrieval | 8 | 0.191 [0.141,0.244] | 0.744 | 0.0558 [0.031,0.080] | 0.204 |
| huginn-0125 | hotpotqa_distractor | 0 (C) | 0.167 [0.121,0.219] | — | — | — |
| huginn-0125 | hotpotqa_distractor | 1 | 0.168 [0.122,0.219] | 0.719 | 0.00118 [-0.001,0.004] | 0.093 |
| huginn-0125 | hotpotqa_distractor | 2 | 0.186 [0.135,0.236] | 0.761 | 0.0185 [-0.005,0.045] | 0.201 |
| huginn-0125 | hotpotqa_distractor | 3 | 0.2 [0.148,0.251] | 0.777 | 0.0325 [0.009,0.058] | 0.204 |
| huginn-0125 | hotpotqa_distractor | 4 | 0.206 [0.156,0.256] | 0.78 | 0.0385 [0.017,0.063] | 0.2 |
| huginn-0125 | hotpotqa_distractor | 6 | 0.207 [0.158,0.257] | 0.783 | 0.0398 [0.019,0.066] | 0.199 |
| huginn-0125 | hotpotqa_distractor | 8 | 0.203 [0.155,0.251] | 0.781 | 0.0356 [0.016,0.060] | 0.198 |
| ouro-2.6b | controlled_retrieval | 0 (C) | 0.111 [0.085,0.137] | — | — | — |
| ouro-2.6b | controlled_retrieval | 1 | 0.118 [0.090,0.150] | 0.652 | 0.00723 [-0.005,0.020] | 0.101 |
| ouro-2.6b | controlled_retrieval | 2 | 0.119 [0.094,0.151] | 0.656 | 0.00799 [-0.016,0.029] | 0.152 |
| ouro-2.6b | controlled_retrieval | 3 | 0.129 [0.104,0.158] | 0.654 | 0.0182 [-0.001,0.037] | 0.146 |
| ouro-2.6b | controlled_retrieval | 4 | 0.14 [0.116,0.164] | 0.659 | 0.029 [0.013,0.044] | 0.137 |
| ouro-2.6b | controlled_retrieval | 6 | 0.138 [0.115,0.161] | 0.656 | 0.0267 [0.006,0.046] | 0.136 |
| ouro-2.6b | controlled_retrieval | 8 | 0.143 [0.113,0.174] | 0.664 | 0.0321 [0.007,0.059] | 0.143 |
| ouro-2.6b | hotpotqa_distractor | 0 (C) | 0.123 [0.090,0.156] | — | — | — |
| ouro-2.6b | hotpotqa_distractor | 1 | 0.133 [0.104,0.163] | 0.642 | 0.00995 [-0.001,0.021] | 0.112 |
| ouro-2.6b | hotpotqa_distractor | 2 | 0.161 [0.124,0.207] | 0.696 | 0.0386 [0.008,0.072] | 0.183 |
| ouro-2.6b | hotpotqa_distractor | 3 | 0.158 [0.122,0.201] | 0.693 | 0.035 [0.007,0.070] | 0.173 |
| ouro-2.6b | hotpotqa_distractor | 4 | 0.157 [0.122,0.199] | 0.692 | 0.034 [0.003,0.069] | 0.168 |
| ouro-2.6b | hotpotqa_distractor | 6 | 0.156 [0.120,0.199] | 0.689 | 0.0329 [0.000,0.070] | 0.167 |
| ouro-2.6b | hotpotqa_distractor | 8 | 0.157 [0.121,0.199] | 0.69 | 0.0346 [0.002,0.070] | 0.166 |

## Gate D — native dynamic exit under post-prefill single-token eviction (label-blind subset)

| model | dataset | n (prompt×source) | clean mean depth | frac queries depth-changed | mean Σdepth Δ | mean fixed ΔNLL | mean replay ΔNLL | mean live ΔNLL | live−replay [CI] | EM full→evict |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 288 | 25.7 | 0.333 | -0.222 | 0.00316 | 0.00327 | 0.0038 | 0.000534 [-0.001,0.002] | 0.917→0.917 |
| huginn-0125 | hotpotqa_distractor | 288 | 22.3 | 0.211 | 0.292 | 0.00182 | 0.00446 | -0.00185 | -0.00631 [-0.008,-0.005] | 0.583→0.583 |
| ouro-2.6b | controlled_retrieval | 288 | 3.46 | 0.00486 | 0 | 0.00494 | 0.00483 | 0.00487 | 3.73e-05 [-0.000,0.000] | 1→0.993 |
| ouro-2.6b | hotpotqa_distractor | 288 | 3.72 | 0.0113 | -0.0243 | 0.00281 | 0.00277 | 0.00277 | -1.58e-06 [-0.000,0.000] | 0.583→0.587 |

| model | dataset | ρ(frac depth-changed, fixed ΔNLL) | ρ(Σdepth Δ, fixed ΔNLL) | frac Σdepth Δ>0 | frac <0 |
|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | -0.252 | 0.0584 | 0.33 | 0.444 |
| huginn-0125 | hotpotqa_distractor | -0.0514 | 0.0843 | 0.392 | 0.198 |
| ouro-2.6b | controlled_retrieval | -0.0793 | 0.269 | 0.0104 | 0.0243 |
| ouro-2.6b | hotpotqa_distractor | 0.0757 | 0.044 | 0.0174 | 0.0417 |

## Clean-only exit calibration (for freezing live thresholds; no eviction data used)

| model / phase / threshold | n tokens | mean depth | frac at R | histogram |
|---|---|---|---|---|
| ouro-2.6b/prompt_exit/q=0.5 | 11062 | 3.22 | 0.268 | {'2': 488, '3': 7604, '4': 2970} |
| ouro-2.6b/prompt_exit/q=0.6 | 11062 | 3.42 | 0.42 | {'2': 24, '3': 6388, '4': 4650} |
| ouro-2.6b/prompt_exit/q=0.7 | 11062 | 3.64 | 0.638 | {'2': 24, '3': 3983, '4': 7055} |
| ouro-2.6b/prompt_exit/q=0.8 | 11062 | 4 | 0.998 | {'2': 24, '3': 3, '4': 11035} |
| ouro-2.6b/prompt_exit/q=0.9 | 11062 | 4 | 0.998 | {'2': 24, '4': 11038} |
| ouro-2.6b/prompt_exit/q=0.95 | 11062 | 4 | 0.998 | {'2': 24, '4': 11038} |
| ouro-2.6b/gen_exit/q=0.5 | 217 | 3.35 | 0.355 | {'2': 1, '3': 139, '4': 77} |
| ouro-2.6b/gen_exit/q=0.6 | 217 | 3.4 | 0.396 | {'3': 131, '4': 86} |
| ouro-2.6b/gen_exit/q=0.7 | 217 | 3.59 | 0.594 | {'3': 88, '4': 129} |
| ouro-2.6b/gen_exit/q=0.8 | 217 | 4 | 1 | {'4': 217} |
| ouro-2.6b/gen_exit/q=0.9 | 217 | 4 | 1 | {'4': 217} |
| ouro-2.6b/gen_exit/q=0.95 | 217 | 4 | 1 | {'4': 217} |
| huginn-0125/prompt_exit/cos=0.01 | 11651 | 12.5 | 8.58e-05 | {'2': 24, '4': 24, '5': 46, '6': 26, '7': 82, '8': 55, '9': 200, '10': 819, '11': 1676, '12': 2890, '13': 2813, '14': 1643, '15': 828, '16': 354, '17': 113, '18': 32, '19': 15, '20': 2, '21': 2, '22': 4, '23': 2, '25': 1} |
| huginn-0125/prompt_exit/cos=0.003 | 11651 | 16 | 0.00112 | {'2': 24, '4': 23, '5': 5, '6': 43, '7': 1, '9': 43, '10': 46, '11': 77, '12': 281, '13': 793, '14': 1292, '15': 1775, '16': 2405, '17': 2050, '18': 1356, '19': 774, '20': 406, '21': 128, '22': 52, '23': 26, '24': 10, '25': 11, '26': 7, '28': 4, '29': 3, '30': 2, '31': 1, '32': 13} |
| huginn-0125/prompt_exit/cos=0.001 | 11651 | 19.7 | 0.00532 | {'3': 24, '5': 24, '6': 16, '7': 31, '8': 1, '11': 35, '12': 24, '13': 68, '14': 121, '15': 330, '16': 740, '17': 968, '18': 1299, '19': 1698, '20': 1809, '21': 1663, '22': 1208, '23': 722, '24': 407, '25': 194, '26': 103, '27': 48, '28': 19, '29': 16, '30': 9, '31': 12, '32': 62} |
| huginn-0125/prompt_exit/cos=0.0003 | 11651 | 24.7 | 0.0409 | {'3': 24, '5': 7, '6': 17, '7': 6, '8': 37, '9': 5, '13': 26, '14': 17, '15': 22, '16': 71, '17': 55, '18': 170, '19': 377, '20': 614, '21': 815, '22': 830, '23': 846, '24': 1130, '25': 1405, '26': 1488, '27': 1181, '28': 824, '29': 565, '30': 394, '31': 248, '32': 477} |
| huginn-0125/gen_exit/cos=0.01 | 218 | 15.2 | 0.00459 | {'10': 1, '11': 3, '12': 12, '13': 56, '14': 21, '15': 24, '16': 39, '17': 26, '18': 16, '19': 12, '20': 4, '21': 3, '22': 1} |
| huginn-0125/gen_exit/cos=0.003 | 218 | 19.7 | 0.00917 | {'14': 3, '15': 6, '16': 11, '17': 37, '18': 25, '19': 23, '20': 29, '21': 26, '22': 30, '23': 12, '24': 9, '25': 4, '26': 1, '32': 2} |
| huginn-0125/gen_exit/cos=0.001 | 218 | 24.3 | 0.0138 | {'18': 4, '19': 5, '20': 6, '21': 21, '22': 25, '23': 25, '24': 25, '25': 33, '26': 28, '27': 20, '28': 13, '29': 4, '30': 4, '31': 2, '32': 3} |
| huginn-0125/gen_exit/cos=0.0003 | 218 | 30 | 0.463 | {'21': 1, '23': 4, '24': 2, '25': 6, '26': 9, '27': 17, '28': 20, '29': 11, '30': 26, '31': 21, '32': 101} |
