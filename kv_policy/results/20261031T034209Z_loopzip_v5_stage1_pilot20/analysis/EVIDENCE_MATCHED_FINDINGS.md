# LoopZip v5 Stage 1 — evidence-locked policies (OFFLINE ORACLE CONTROL)

complete=True; jobs 80/80; feasible request-budget cells 190/240; parity failures 0.

Every policy keeps the same evidence positions E and exactly K Context tokens.

| model | dataset | b | policy | n | ΔNLL [95% CI] | KL(Full‖policy) | acc (full-correct) | ΔNLL − mean_attn [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0.1 | mean_attn | 20 | 0.071 [-0.023,0.174] | 0.1919 | 0.79 (n=19) | 0.000 [0.000,0.000] |
| huginn-0125 | controlled_retrieval | 0.1 | last_attn | 20 | 0.049 [-0.028,0.136] | 0.1721 | 0.79 (n=19) | -0.023 [-0.049,-0.004] |
| huginn-0125 | controlled_retrieval | 0.1 | max_attn | 20 | 0.086 [-0.014,0.197] | 0.2042 | 0.68 (n=19) | 0.014 [-0.006,0.037] |
| huginn-0125 | controlled_retrieval | 0.1 | recent_attn | 20 | 0.045 [-0.031,0.131] | 0.1697 | 0.84 (n=19) | -0.026 [-0.057,-0.005] |
| huginn-0125 | controlled_retrieval | 0.1 | random | 20 | -0.049 [-0.080,-0.015] | 0.0900 | 0.98 (n=19) | -0.120 [-0.206,-0.049] |
| huginn-0125 | controlled_retrieval | 0.3 | mean_attn | 20 | -0.025 [-0.059,0.002] | 0.0608 | 1.00 (n=19) | 0.000 [0.000,0.000] |
| huginn-0125 | controlled_retrieval | 0.3 | last_attn | 20 | -0.027 [-0.061,0.000] | 0.0586 | 1.00 (n=19) | -0.002 [-0.006,0.002] |
| huginn-0125 | controlled_retrieval | 0.3 | max_attn | 20 | -0.010 [-0.049,0.024] | 0.0759 | 0.95 (n=19) | 0.015 [-0.001,0.037] |
| huginn-0125 | controlled_retrieval | 0.3 | recent_attn | 20 | -0.020 [-0.060,0.020] | 0.0664 | 0.89 (n=19) | 0.005 [-0.017,0.032] |
| huginn-0125 | controlled_retrieval | 0.3 | random | 20 | -0.046 [-0.056,-0.037] | 0.0351 | 1.00 (n=19) | -0.021 [-0.044,0.011] |
| huginn-0125 | controlled_retrieval | 0.5 | mean_attn | 20 | -0.012 [-0.026,-0.001] | 0.0360 | 1.00 (n=19) | 0.000 [0.000,0.000] |
| huginn-0125 | controlled_retrieval | 0.5 | last_attn | 20 | -0.013 [-0.027,-0.002] | 0.0310 | 1.00 (n=19) | -0.001 [-0.005,0.003] |
| huginn-0125 | controlled_retrieval | 0.5 | max_attn | 20 | -0.025 [-0.052,-0.006] | 0.0398 | 1.00 (n=19) | -0.013 [-0.028,-0.003] |
| huginn-0125 | controlled_retrieval | 0.5 | recent_attn | 20 | -0.013 [-0.024,-0.003] | 0.0251 | 1.00 (n=19) | -0.001 [-0.007,0.007] |
| huginn-0125 | controlled_retrieval | 0.5 | random | 20 | -0.028 [-0.035,-0.022] | 0.0261 | 1.00 (n=19) | -0.016 [-0.025,-0.005] |
| huginn-0125 | hotpotqa_distractor | 0.3 | mean_attn | 15 | -0.052 [-0.158,0.021] | 0.1052 | 1.00 (n=8) | 0.000 [0.000,0.000] |
| huginn-0125 | hotpotqa_distractor | 0.3 | last_attn | 15 | -0.054 [-0.161,0.013] | 0.0984 | 1.00 (n=8) | -0.002 [-0.017,0.018] |
| huginn-0125 | hotpotqa_distractor | 0.3 | max_attn | 15 | -0.056 [-0.154,0.009] | 0.1012 | 1.00 (n=8) | -0.004 [-0.024,0.016] |
| huginn-0125 | hotpotqa_distractor | 0.3 | recent_attn | 15 | -0.045 [-0.144,0.020] | 0.1042 | 1.00 (n=8) | 0.007 [-0.007,0.024] |
| huginn-0125 | hotpotqa_distractor | 0.3 | random | 15 | -0.052 [-0.147,0.010] | 0.0959 | 1.00 (n=8) | 0.000 [-0.024,0.032] |
| huginn-0125 | hotpotqa_distractor | 0.5 | mean_attn | 20 | -0.027 [-0.082,0.014] | 0.0282 | 1.00 (n=9) | 0.000 [0.000,0.000] |
| huginn-0125 | hotpotqa_distractor | 0.5 | last_attn | 20 | -0.034 [-0.088,0.004] | 0.0183 | 1.00 (n=9) | -0.008 [-0.018,0.001] |
| huginn-0125 | hotpotqa_distractor | 0.5 | max_attn | 20 | -0.022 [-0.090,0.032] | 0.0574 | 1.00 (n=9) | 0.005 [-0.024,0.038] |
| huginn-0125 | hotpotqa_distractor | 0.5 | recent_attn | 20 | -0.035 [-0.088,0.008] | 0.0462 | 1.00 (n=9) | -0.008 [-0.021,0.004] |
| huginn-0125 | hotpotqa_distractor | 0.5 | random | 20 | -0.033 [-0.077,-0.001] | 0.0488 | 1.00 (n=9) | -0.006 [-0.032,0.019] |
| ouro-2.6b | controlled_retrieval | 0.1 | mean_attn | 20 | 0.293 [0.102,0.485] | 0.0966 | 0.70 (n=20) | 0.000 [0.000,0.000] |
| ouro-2.6b | controlled_retrieval | 0.1 | last_attn | 20 | 0.327 [0.119,0.541] | 0.1110 | 0.70 (n=20) | 0.034 [0.007,0.065] |
| ouro-2.6b | controlled_retrieval | 0.1 | max_attn | 20 | 0.287 [0.098,0.475] | 0.0949 | 0.70 (n=20) | -0.006 [-0.017,0.004] |
| ouro-2.6b | controlled_retrieval | 0.1 | recent_attn | 20 | 0.308 [0.107,0.503] | 0.1018 | 0.70 (n=20) | 0.015 [-0.001,0.035] |
| ouro-2.6b | controlled_retrieval | 0.1 | random | 20 | 0.130 [0.036,0.238] | 0.0552 | 0.87 (n=20) | -0.163 [-0.293,-0.057] |
| ouro-2.6b | controlled_retrieval | 0.3 | mean_attn | 20 | 0.013 [0.000,0.029] | 0.0065 | 0.95 (n=20) | 0.000 [0.000,0.000] |
| ouro-2.6b | controlled_retrieval | 0.3 | last_attn | 20 | 0.078 [0.010,0.163] | 0.0276 | 0.80 (n=20) | 0.065 [-0.003,0.146] |
| ouro-2.6b | controlled_retrieval | 0.3 | max_attn | 20 | 0.056 [0.006,0.126] | 0.0209 | 0.85 (n=20) | 0.043 [-0.001,0.110] |
| ouro-2.6b | controlled_retrieval | 0.3 | recent_attn | 20 | 0.012 [0.001,0.026] | 0.0064 | 0.95 (n=20) | -0.001 [-0.011,0.007] |
| ouro-2.6b | controlled_retrieval | 0.3 | random | 20 | -0.008 [-0.013,-0.002] | 0.0082 | 0.98 (n=20) | -0.021 [-0.034,-0.010] |
| ouro-2.6b | controlled_retrieval | 0.5 | mean_attn | 20 | 0.004 [-0.000,0.010] | 0.0019 | 1.00 (n=20) | 0.000 [0.000,0.000] |
| ouro-2.6b | controlled_retrieval | 0.5 | last_attn | 20 | 0.005 [0.000,0.009] | 0.0034 | 1.00 (n=20) | 0.001 [-0.002,0.003] |
| ouro-2.6b | controlled_retrieval | 0.5 | max_attn | 20 | 0.003 [-0.001,0.006] | 0.0035 | 1.00 (n=20) | -0.001 [-0.005,0.003] |
| ouro-2.6b | controlled_retrieval | 0.5 | recent_attn | 20 | 0.003 [-0.001,0.006] | 0.0019 | 1.00 (n=20) | -0.001 [-0.005,0.002] |
| ouro-2.6b | controlled_retrieval | 0.5 | random | 20 | -0.007 [-0.009,-0.005] | 0.0046 | 1.00 (n=20) | -0.011 [-0.017,-0.007] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | mean_attn | 15 | -0.009 [-0.033,0.009] | 0.0024 | 1.00 (n=9) | 0.000 [0.000,0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | last_attn | 15 | -0.010 [-0.036,0.009] | 0.0024 | 1.00 (n=9) | -0.001 [-0.005,0.003] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | max_attn | 15 | -0.015 [-0.040,0.005] | 0.0026 | 1.00 (n=9) | -0.005 [-0.010,-0.001] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | recent_attn | 15 | -0.011 [-0.034,0.008] | 0.0027 | 1.00 (n=9) | -0.002 [-0.007,0.002] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | random | 15 | -0.005 [-0.040,0.038] | 0.0045 | 1.00 (n=9) | 0.004 [-0.015,0.036] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | mean_attn | 20 | -0.005 [-0.021,0.008] | 0.0010 | 1.00 (n=11) | 0.000 [0.000,0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | last_attn | 20 | -0.009 [-0.027,0.004] | 0.0009 | 1.00 (n=11) | -0.004 [-0.007,-0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | max_attn | 20 | -0.007 [-0.026,0.007] | 0.0010 | 1.00 (n=11) | -0.002 [-0.006,0.002] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | recent_attn | 20 | -0.008 [-0.024,0.005] | 0.0010 | 1.00 (n=11) | -0.003 [-0.005,-0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | random | 20 | -0.004 [-0.025,0.021] | 0.0032 | 1.00 (n=11) | 0.001 [-0.020,0.026] |

Arm 1A (Phase-1 attention policies, identical retained evidence set): 157 pairs; greedy correctness differs in 0; pairs with full evidence kept: 4.

Annotation-informed oracle control; residual differences are not by themselves recurrence-specific.
