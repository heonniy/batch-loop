# LoopZip v5 Stage 1 — evidence-locked policies (OFFLINE ORACLE CONTROL)

complete=True; jobs 20/20; feasible request-budget cells 50/60; parity failures 0.

Every policy keeps the same evidence positions E and exactly K Context tokens.

| model | dataset | b | policy | n | ΔNLL [95% CI] | KL(Full‖policy) | acc (full-correct) | ΔNLL − mean_attn [95% CI] |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0.1 | mean_attn | 5 | -0.002 [-0.051,0.085] | 0.0684 | 1.00 (n=5) | 0.000 [0.000,0.000] |
| huginn-0125 | controlled_retrieval | 0.1 | last_attn | 5 | -0.001 [-0.049,0.085] | 0.0668 | 1.00 (n=5) | 0.001 [-0.002,0.005] |
| huginn-0125 | controlled_retrieval | 0.1 | max_attn | 5 | 0.006 [-0.046,0.103] | 0.0712 | 0.80 (n=5) | 0.009 [-0.001,0.021] |
| huginn-0125 | controlled_retrieval | 0.1 | recent_attn | 5 | -0.004 [-0.053,0.085] | 0.0673 | 1.00 (n=5) | -0.002 [-0.005,0.000] |
| huginn-0125 | controlled_retrieval | 0.1 | random | 5 | -0.065 [-0.077,-0.053] | 0.0233 | 1.00 (n=5) | -0.062 [-0.141,-0.020] |
| huginn-0125 | controlled_retrieval | 0.3 | mean_attn | 5 | -0.014 [-0.031,0.004] | 0.0160 | 1.00 (n=5) | 0.000 [0.000,0.000] |
| huginn-0125 | controlled_retrieval | 0.3 | last_attn | 5 | -0.015 [-0.038,0.010] | 0.0163 | 1.00 (n=5) | -0.001 [-0.014,0.010] |
| huginn-0125 | controlled_retrieval | 0.3 | max_attn | 5 | -0.021 [-0.040,-0.001] | 0.0181 | 1.00 (n=5) | -0.007 [-0.018,0.001] |
| huginn-0125 | controlled_retrieval | 0.3 | recent_attn | 5 | -0.026 [-0.046,-0.010] | 0.0109 | 1.00 (n=5) | -0.012 [-0.020,0.000] |
| huginn-0125 | controlled_retrieval | 0.3 | random | 5 | -0.041 [-0.056,-0.025] | 0.0115 | 1.00 (n=5) | -0.027 [-0.036,-0.020] |
| huginn-0125 | controlled_retrieval | 0.5 | mean_attn | 5 | -0.003 [-0.009,0.006] | 0.0082 | 1.00 (n=5) | 0.000 [0.000,0.000] |
| huginn-0125 | controlled_retrieval | 0.5 | last_attn | 5 | -0.004 [-0.008,-0.001] | 0.0072 | 1.00 (n=5) | -0.002 [-0.008,0.006] |
| huginn-0125 | controlled_retrieval | 0.5 | max_attn | 5 | -0.009 [-0.014,-0.004] | 0.0113 | 1.00 (n=5) | -0.006 [-0.013,0.002] |
| huginn-0125 | controlled_retrieval | 0.5 | recent_attn | 5 | -0.009 [-0.017,0.000] | 0.0036 | 1.00 (n=5) | -0.006 [-0.017,0.005] |
| huginn-0125 | controlled_retrieval | 0.5 | random | 5 | -0.027 [-0.034,-0.016] | 0.0070 | 1.00 (n=5) | -0.024 [-0.030,-0.018] |
| huginn-0125 | hotpotqa_distractor | 0.3 | mean_attn | 5 | -0.134 [-0.383,0.066] | 0.1715 | 1.00 (n=1) | 0.000 [0.000,0.000] |
| huginn-0125 | hotpotqa_distractor | 0.3 | last_attn | 5 | -0.114 [-0.367,0.062] | 0.1548 | 1.00 (n=1) | 0.020 [-0.007,0.068] |
| huginn-0125 | hotpotqa_distractor | 0.3 | max_attn | 5 | -0.113 [-0.346,0.062] | 0.1790 | 1.00 (n=1) | 0.021 [-0.011,0.056] |
| huginn-0125 | hotpotqa_distractor | 0.3 | recent_attn | 5 | -0.111 [-0.339,0.057] | 0.1673 | 1.00 (n=1) | 0.023 [-0.009,0.062] |
| huginn-0125 | hotpotqa_distractor | 0.3 | random | 5 | -0.093 [-0.331,0.054] | 0.1616 | 1.00 (n=1) | 0.041 [-0.020,0.112] |
| huginn-0125 | hotpotqa_distractor | 0.5 | mean_attn | 5 | -0.122 [-0.291,-0.008] | 0.0221 | 1.00 (n=1) | 0.000 [0.000,0.000] |
| huginn-0125 | hotpotqa_distractor | 0.5 | last_attn | 5 | -0.128 [-0.294,-0.012] | 0.0201 | 1.00 (n=1) | -0.006 [-0.018,0.004] |
| huginn-0125 | hotpotqa_distractor | 0.5 | max_attn | 5 | -0.087 [-0.323,0.099] | 0.1154 | 1.00 (n=1) | 0.036 [-0.074,0.140] |
| huginn-0125 | hotpotqa_distractor | 0.5 | recent_attn | 5 | -0.138 [-0.293,-0.028] | 0.0832 | 1.00 (n=1) | -0.016 [-0.035,0.003] |
| huginn-0125 | hotpotqa_distractor | 0.5 | random | 5 | -0.077 [-0.218,0.021] | 0.0874 | 1.00 (n=1) | 0.045 [0.013,0.078] |
| ouro-2.6b | controlled_retrieval | 0.1 | mean_attn | 5 | 0.341 [-0.007,0.795] | 0.1147 | 0.60 (n=5) | 0.000 [0.000,0.000] |
| ouro-2.6b | controlled_retrieval | 0.1 | last_attn | 5 | 0.387 [-0.008,0.843] | 0.1320 | 0.60 (n=5) | 0.046 [-0.002,0.139] |
| ouro-2.6b | controlled_retrieval | 0.1 | max_attn | 5 | 0.312 [-0.007,0.737] | 0.1060 | 0.60 (n=5) | -0.029 [-0.058,0.000] |
| ouro-2.6b | controlled_retrieval | 0.1 | recent_attn | 5 | 0.332 [-0.006,0.783] | 0.1125 | 0.60 (n=5) | -0.009 [-0.029,0.002] |
| ouro-2.6b | controlled_retrieval | 0.1 | random | 5 | 0.125 [-0.016,0.302] | 0.0555 | 0.87 (n=5) | -0.216 [-0.569,-0.006] |
| ouro-2.6b | controlled_retrieval | 0.3 | mean_attn | 5 | 0.024 [-0.009,0.074] | 0.0143 | 0.80 (n=5) | 0.000 [0.000,0.000] |
| ouro-2.6b | controlled_retrieval | 0.3 | last_attn | 5 | 0.101 [-0.006,0.274] | 0.0398 | 0.60 (n=5) | 0.077 [-0.016,0.248] |
| ouro-2.6b | controlled_retrieval | 0.3 | max_attn | 5 | 0.023 [-0.010,0.072] | 0.0142 | 0.80 (n=5) | -0.000 [-0.002,0.001] |
| ouro-2.6b | controlled_retrieval | 0.3 | recent_attn | 5 | 0.018 [-0.014,0.067] | 0.0130 | 0.80 (n=5) | -0.006 [-0.012,-0.000] |
| ouro-2.6b | controlled_retrieval | 0.3 | random | 5 | -0.004 [-0.019,0.018] | 0.0123 | 0.93 (n=5) | -0.028 [-0.056,-0.007] |
| ouro-2.6b | controlled_retrieval | 0.5 | mean_attn | 5 | 0.008 [-0.006,0.029] | 0.0031 | 1.00 (n=5) | 0.000 [0.000,0.000] |
| ouro-2.6b | controlled_retrieval | 0.5 | last_attn | 5 | 0.007 [-0.008,0.022] | 0.0086 | 1.00 (n=5) | -0.002 [-0.008,0.003] |
| ouro-2.6b | controlled_retrieval | 0.5 | max_attn | 5 | -0.000 [-0.013,0.010] | 0.0080 | 1.00 (n=5) | -0.009 [-0.020,-0.001] |
| ouro-2.6b | controlled_retrieval | 0.5 | recent_attn | 5 | 0.002 [-0.008,0.015] | 0.0024 | 1.00 (n=5) | -0.007 [-0.013,-0.002] |
| ouro-2.6b | controlled_retrieval | 0.5 | random | 5 | -0.007 [-0.011,-0.002] | 0.0064 | 1.00 (n=5) | -0.015 [-0.034,-0.004] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | mean_attn | 5 | -0.009 [-0.077,0.029] | 0.0035 | 1.00 (n=2) | 0.000 [0.000,0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | last_attn | 5 | -0.014 [-0.090,0.031] | 0.0034 | 1.00 (n=2) | -0.005 [-0.013,0.003] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | max_attn | 5 | -0.018 [-0.095,0.027] | 0.0036 | 1.00 (n=2) | -0.009 [-0.017,-0.002] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | recent_attn | 5 | -0.011 [-0.079,0.027] | 0.0040 | 1.00 (n=2) | -0.002 [-0.005,0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | random | 5 | 0.024 [-0.085,0.142] | 0.0075 | 1.00 (n=2) | 0.033 [-0.019,0.122] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | mean_attn | 5 | -0.016 [-0.078,0.016] | 0.0016 | 1.00 (n=2) | 0.000 [0.000,0.000] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | last_attn | 5 | -0.027 [-0.090,0.006] | 0.0012 | 1.00 (n=2) | -0.011 [-0.014,-0.006] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | max_attn | 5 | -0.025 [-0.093,0.013] | 0.0015 | 1.00 (n=2) | -0.008 [-0.017,0.001] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | recent_attn | 5 | -0.020 [-0.083,0.014] | 0.0014 | 1.00 (n=2) | -0.003 [-0.008,0.001] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | random | 5 | 0.020 [-0.054,0.106] | 0.0053 | 1.00 (n=2) | 0.036 [-0.009,0.105] |

Arm 1A (Phase-1 attention policies, identical retained evidence set): 41 pairs; greedy correctness differs in 0; pairs with full evidence kept: 1.

Annotation-informed oracle control; residual differences are not by themselves recurrence-specific.
