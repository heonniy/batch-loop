# V10 Full vs last recurrent KV: Context eviction × trajectories × exit

Model huginn-0125; hotpotqa_distractor; 5/5 prompts, 30 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | -0.006665 | -0.008555 | -0.00189 | 0.02409 | 0.02333 |
| oracle_distractor | 5 | -0.004341 | -0.0123 | -0.007955 | 0.01637 | 0.01608 |
| oracle_evidence | 5 | -0.02121 | -0.02137 | -0.0001584 | 0.0329 | 0.0479 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 8 | 0.003046 | 0.006583 | 0.003537 |
| 16 | -0.001445 | -0.007343 | -0.005899 |
| 24 | 0.004226 | -0.002893 | -0.00712 |
| 32 | -0.008702 | -0.01131 | -0.002613 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 5.829334259033203e-05 (anchor=pooled); CLEAN early fraction pooled 0.289, by policy {'full': 0.0, 'last': 0.579}; saturated policies=['full']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | -0.008702 | 0 | 0 | 0 |
| last | 0.6769 | 0.6331 | -0.001613 | 0.001606 | 0 | 0 |

Threshold 0.0001 (anchor=pooled); CLEAN early fraction pooled 0.447, by policy {'full': 0.0, 'last': 0.895}; saturated policies=['full']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | -0.008702 | 0 | 0 | 0 |
| last | 0.5517 | 0.5733 | -0.0041 | -0.001721 | 0 | 0 |

Threshold 0.00018330931663513184 (anchor=full_only); CLEAN early fraction pooled 0.421, by policy {'full': 0.421, 'last': 0.947}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2372 | 0.2306 | -0.006914 | -0.001757 | 0 | 0 |
| last | 0.2867 | 0.2925 | -0.003607 | 0.004908 | 0 | 0 |

Threshold 0.0002111107110977173 (anchor=full_only); CLEAN early fraction pooled 0.632, by policy {'full': 0.632, 'last': 0.947}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.3186 | 0.3019 | 0.001407 | -0.009197 | 0 | 0 |
| last | 0.3217 | 0.3303 | -0.008207 | 0.0008604 | 0 | 0 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
