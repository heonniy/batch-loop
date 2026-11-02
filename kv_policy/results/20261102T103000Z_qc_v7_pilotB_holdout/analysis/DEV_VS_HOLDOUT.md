# Dev (pilot A) vs holdout (pilot B): LOPO lifts and loop-resolved attention importance

| model | dataset | lift | dev (8 UIDs) | holdout (12 UIDs) |
|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | LIFT C − A | +0.004 [-0.007,+0.013] | -0.003 [-0.010,+0.005] |
| huginn-0125 | hotpotqa_distractor | LIFT C − A | -0.005 [-0.011,+0.001] | -0.003 [-0.012,+0.006] |
| ouro-2.6b | controlled_retrieval | LIFT C − A | +0.025 [+0.013,+0.036] | +0.016 [+0.006,+0.027] |
| ouro-2.6b | hotpotqa_distractor | LIFT C − A | +0.019 [+0.004,+0.037] | -0.002 [-0.014,+0.010] |
| huginn-0125 | controlled_retrieval | LIFT L − C | +0.008 [-0.009,+0.025] | +0.016 [+0.005,+0.027] |
| huginn-0125 | hotpotqa_distractor | LIFT L − C | +0.016 [+0.006,+0.026] | +0.012 [-0.016,+0.042] |
| ouro-2.6b | controlled_retrieval | LIFT L − C | +0.023 [+0.002,+0.047] | +0.021 [-0.001,+0.043] |
| ouro-2.6b | hotpotqa_distractor | LIFT L − C | +0.034 [-0.019,+0.084] | +0.028 [-0.009,+0.066] |
| huginn-0125 | controlled_retrieval | LIFT F − C | +0.038 [-0.005,+0.084] | +0.058 [+0.019,+0.098] |
| huginn-0125 | hotpotqa_distractor | LIFT F − C | +0.067 [+0.029,+0.112] | +0.029 [+0.001,+0.060] |
| ouro-2.6b | controlled_retrieval | LIFT F − C | +0.024 [-0.004,+0.052] | +0.017 [-0.011,+0.043] |
| ouro-2.6b | hotpotqa_distractor | LIFT F − C | +0.060 [+0.039,+0.079] | +0.027 [-0.005,+0.067] |
| huginn-0125 | controlled_retrieval | LIFT D − C | -0.010 [-0.025,+0.004] | +0.005 [-0.009,+0.019] |
| huginn-0125 | hotpotqa_distractor | LIFT D − C | -0.009 [-0.014,-0.003] | -0.002 [-0.008,+0.005] |
| ouro-2.6b | controlled_retrieval | LIFT D − C | -0.010 [-0.022,+0.006] | +0.007 [-0.013,+0.030] |
| ouro-2.6b | hotpotqa_distractor | LIFT D − C | -0.018 [-0.037,+0.003] | -0.018 [-0.041,+0.010] |
| huginn-0125 | controlled_retrieval | LIFT G1 − F | +0.028 [+0.001,+0.056] | +0.007 [-0.015,+0.029] |
| huginn-0125 | hotpotqa_distractor | LIFT G1 − F | -0.014 [-0.043,+0.012] | +0.003 [-0.018,+0.024] |
| ouro-2.6b | controlled_retrieval | LIFT G1 − F | +0.010 [-0.009,+0.032] | +0.027 [+0.009,+0.045] |
| ouro-2.6b | hotpotqa_distractor | LIFT G1 − F | -0.014 [-0.032,+0.002] | -0.002 [-0.019,+0.015] |
| huginn-0125 | controlled_retrieval | LIFT G1g − F | -0.008 [-0.019,+0.003] | +0.003 [-0.007,+0.011] |
| huginn-0125 | hotpotqa_distractor | LIFT G1g − F | +0.004 [-0.001,+0.010] | -0.005 [-0.019,+0.008] |
| ouro-2.6b | controlled_retrieval | LIFT G1g − F | +0.000 [-0.005,+0.005] | +0.002 [-0.003,+0.007] |
| ouro-2.6b | hotpotqa_distractor | LIFT G1g − F | -0.006 [-0.021,+0.007] | -0.003 [-0.014,+0.010] |
| huginn-0125 | controlled_retrieval | LIFT G2 − F | +0.034 [-0.007,+0.073] | +0.015 [-0.008,+0.036] |
| huginn-0125 | hotpotqa_distractor | LIFT G2 − F | +0.004 [-0.006,+0.012] | +0.007 [-0.001,+0.016] |
| ouro-2.6b | controlled_retrieval | LIFT G2 − F | +0.011 [-0.008,+0.027] | +0.029 [+0.001,+0.054] |
| ouro-2.6b | hotpotqa_distractor | LIFT G2 − F | -0.010 [-0.019,-0.003] | -0.009 [-0.027,+0.009] |

| model | dataset | query | ρ@loop1 dev / holdout | ρ@late dev / holdout |
|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | consuming query t* (oracle) | 0.075 / 0.079 | 0.203 / 0.214 |
| huginn-0125 | controlled_retrieval | 2-token lookahead | 0.039 / -0.005 | 0.190 / 0.188 |
| huginn-0125 | controlled_retrieval | question anchor | 0.041 / 0.015 | 0.119 / 0.084 |
| huginn-0125 | hotpotqa_distractor | consuming query t* (oracle) | 0.074 / 0.052 | 0.235 / 0.202 |
| huginn-0125 | hotpotqa_distractor | 2-token lookahead | 0.035 / 0.033 | 0.241 / 0.206 |
| huginn-0125 | hotpotqa_distractor | question anchor | 0.038 / 0.034 | 0.121 / 0.124 |
| ouro-2.6b | controlled_retrieval | consuming query t* (oracle) | 0.089 / 0.098 | 0.150 / 0.156 |
| ouro-2.6b | controlled_retrieval | 2-token lookahead | 0.073 / 0.070 | 0.157 / 0.163 |
| ouro-2.6b | controlled_retrieval | question anchor | 0.056 / 0.067 | 0.085 / 0.097 |
| ouro-2.6b | hotpotqa_distractor | consuming query t* (oracle) | 0.122 / 0.112 | 0.174 / 0.173 |
| ouro-2.6b | hotpotqa_distractor | 2-token lookahead | 0.087 / 0.092 | 0.169 / 0.184 |
| ouro-2.6b | hotpotqa_distractor | question anchor | 0.098 / 0.091 | 0.090 / 0.115 |
