# V10 Full vs first_last recurrent KV: Context eviction × trajectories × exit

Model ouro-1.4b; controlled_retrieval; 5/5 prompts, 30 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | -0.0007047 | 0.0003646 | 0.001069 | 0.02209 | 0.01964 |
| oracle_distractor | 5 | -0.001027 | -0.001758 | -0.000731 | 0.01887 | 0.01439 |
| oracle_evidence | 5 | 0.1816 | 0.1399 | -0.04167 | 0.1028 | 0.09237 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 2 | 0.02571 | 0.03353 | 0.007819 |
| 3 | 0.03147 | 0.02789 | -0.003588 |
| 4 | 0.02962 | 0.02327 | -0.006354 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 0.8 (anchor=pooled); CLEAN early fraction pooled 0.045, by policy {'full': 0.0, 'first_last': 0.091}; saturated policies=['full']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.02962 | 0 | 0.03333 | 0 |
| first_last | 0.0303 | 0.02121 | 0.024 | -0.0001728 | 0.03333 | 0 |

Threshold 0.7 (anchor=pooled); CLEAN early fraction pooled 0.758, by policy {'full': 0.515, 'first_last': 1.0}; saturated policies=['first_last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.03636 | 0.04545 | 0.03027 | -0.001355 | 0.03333 | 0 |
| first_last | 0.01212 | 0.01212 | 0.02562 | -0.002271 | 0.03333 | 0 |

Threshold 0.6 (anchor=full_only); CLEAN early fraction pooled 0.667, by policy {'full': 0.667, 'first_last': 1.0}; saturated policies=['first_last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.02727 | 0.02727 | 0.03234 | -0.0003404 | 0.03333 | 0 |
| first_last | 0.02727 | 0.04242 | 0.02562 | -0.004306 | 0.03333 | 0 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
