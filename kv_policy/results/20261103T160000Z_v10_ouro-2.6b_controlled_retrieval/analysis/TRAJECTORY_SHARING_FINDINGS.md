# V10 Full vs first_last recurrent KV: Context eviction × trajectories × exit

Model ouro-2.6b; controlled_retrieval; 5/5 prompts, 30 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.001515 | -0.000246 | -0.001761 | 0.02139 | 0.01806 |
| oracle_distractor | 5 | 0.001853 | -0.0005393 | -0.002392 | 0.01962 | 0.01553 |
| oracle_evidence | 5 | 0.2342 | 0.2281 | -0.006059 | 0.1114 | 0.1019 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 2 | 0.02587 | 0.03068 | 0.004817 |
| 3 | 0.04027 | 0.03918 | -0.001089 |
| 4 | 0.04035 | 0.03776 | -0.002582 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 0.8 (anchor=pooled); CLEAN early fraction pooled 0.212, by policy {'full': 0.0, 'first_last': 0.424}; saturated policies=['full']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.04035 | 0 | 0.06667 | 0 |
| first_last | 0.02424 | 0.01818 | 0.03901 | -0.0005391 | 0.03333 | 0 |

Threshold 0.7 (anchor=pooled); CLEAN early fraction pooled 0.758, by policy {'full': 0.545, 'first_last': 0.97}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.01515 | 0.01515 | 0.04103 | 0.003217 | 0.06667 | 0 |
| first_last | 0.006061 | 0.006061 | 0.03985 | 0.0006749 | 0.03333 | 0 |

Threshold 0.6 (anchor=full_only); CLEAN early fraction pooled 0.636, by policy {'full': 0.636, 'first_last': 0.97}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.01818 | 0.01818 | 0.03984 | 0.004886 | 0.06667 | 0 |
| first_last | 0.02121 | 0.0303 | 0.03951 | 0.001306 | 0.03333 | 0 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
