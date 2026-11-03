# V10 Full vs last recurrent KV: Context eviction × trajectories × exit

Model huginn-0125; controlled_retrieval; 1/1 prompts, 6 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 4 | -0.008105 | 5.115e-05 | 0.008156 | 0.02021 | 0.02014 |
| oracle_distractor | 1 | -0.007085 | -0.0003206 | 0.006764 | 0.01755 | 0.01741 |
| oracle_evidence | 1 | 0.004561 | 0.005673 | 0.001113 | 0.02024 | 0.02154 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 8 | 0.00928 | -0.00689 | -0.01617 |
| 16 | -0.002992 | -0.004054 | -0.001063 |
| 24 | -0.002679 | 4.699e-05 | 0.002726 |
| 32 | -0.005824 | 0.0009262 | 0.00675 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 5.867362022399902e-05; CLEAN pooled early fraction 0.205; saturated=False
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | -0.005824 | 0 | 0 | 0 |
| last | 0.4242 | 0.4697 | 0.002184 | -0.002616 | 0 | 0 |

Threshold 0.00010689496994018557; CLEAN pooled early fraction 0.500; saturated=False
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | -0.005824 | 0 | 0 | 0 |
| last | 0.7424 | 0.7121 | 0.006878 | 0.003111 | 0 | 0 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
