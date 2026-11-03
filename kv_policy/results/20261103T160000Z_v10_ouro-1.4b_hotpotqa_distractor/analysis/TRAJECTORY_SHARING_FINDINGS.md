# V10 Full vs first_last recurrent KV: Context eviction × trajectories × exit

Model ouro-1.4b; hotpotqa_distractor; 5/5 prompts, 30 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.004182 | 0.0004622 | -0.003719 | 0.02445 | 0.02411 |
| oracle_distractor | 5 | -0.003572 | -0.002054 | 0.001518 | 0.01771 | 0.01662 |
| oracle_evidence | 5 | -0.02364 | -0.009051 | 0.01458 | 0.02553 | 0.02533 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 2 | 0.01699 | -0.01917 | -0.03616 |
| 3 | 0.01308 | 0.0003843 | -0.0127 |
| 4 | -0.001747 | -0.001543 | 0.0002041 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 0.7 (anchor=pooled); CLEAN early fraction pooled 0.342, by policy {'full': 0.105, 'first_last': 0.579}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.01 | 0.01 | -0.001642 | -3.526e-05 | 0 | 0 |
| first_last | 0.008889 | 0.01 | 0.005783 | -0.0005235 | 0 | -0.03333 |

Threshold 0.5 (anchor=pooled); CLEAN early fraction pooled 0.658, by policy {'full': 0.368, 'first_last': 0.947}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0125 | 0.0125 | -0.001315 | -0.0003178 | 0 | 0 |
| first_last | 0.08083 | 0.05222 | -0.004192 | 0.004062 | 0 | -0.03333 |

Threshold 0.6 (anchor=full_only); CLEAN early fraction pooled 0.211, by policy {'full': 0.211, 'first_last': 0.842}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | -0.002528 | 0 | 0 | 0 |
| first_last | 0.01667 | 0.04333 | 0.004487 | 0.002051 | 0 | -0.03333 |

Threshold 0.3 (anchor=full_only); CLEAN early fraction pooled 0.632, by policy {'full': 0.632, 'first_last': 1.0}; saturated policies=['first_last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.006667 | 0.006667 | 0.001888 | -0.001124 | 0 | 0 |
| first_last | 0.03333 | 0.01 | -0.003079 | 0 | 0 | -0.03333 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
