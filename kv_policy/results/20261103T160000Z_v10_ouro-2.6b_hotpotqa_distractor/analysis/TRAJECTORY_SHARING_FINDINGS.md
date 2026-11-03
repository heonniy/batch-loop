# V10 Full vs first_last recurrent KV: Context eviction × trajectories × exit

Model ouro-2.6b; hotpotqa_distractor; 5/5 prompts, 30 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.004915 | -0.0007737 | -0.005688 | 0.02367 | 0.02343 |
| oracle_distractor | 5 | 0.0005196 | -0.005858 | -0.006378 | 0.01629 | 0.01638 |
| oracle_evidence | 5 | -0.004725 | -0.003618 | 0.001106 | 0.02115 | 0.02054 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 2 | 0.002112 | 0.002345 | 0.0002331 |
| 3 | 0.002293 | 0.0006794 | -0.001613 |
| 4 | 0.002576 | -0.002095 | -0.004671 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 0.8 (anchor=pooled); CLEAN early fraction pooled 0.026, by policy {'full': 0.0, 'first_last': 0.053}; saturated policies=['full']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.002576 | 0 | 0.03333 | 0 |
| first_last | 0 | 0 | -0.002003 | 0 | 0.06667 | 0 |

Threshold 0.6 (anchor=pooled); CLEAN early fraction pooled 0.632, by policy {'full': 0.368, 'first_last': 0.895}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.006667 | 0.006667 | 0.00364 | 0.0006657 | 0 | -0.03333 |
| first_last | 0 | 0 | 0.005147 | 0 | 0 | -0.06667 |

Threshold 0.7 (anchor=full_only); CLEAN early fraction pooled 0.158, by policy {'full': 0.158, 'first_last': 0.842}; saturated policies=[]
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.01333 | 0.006667 | 0.0007825 | 0.0001352 | 0.03333 | 0 |
| first_last | 0 | 0.006667 | 0.001064 | 0 | 0 | -0.06667 |

Threshold 0.4 (anchor=full_only); CLEAN early fraction pooled 0.737, by policy {'full': 0.737, 'first_last': 1.0}; saturated policies=['first_last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.02 | 0.02 | 0.002189 | -0.002064 | 0 | -0.03333 |
| first_last | 0 | 0 | 0.004448 | 0 | 0 | -0.06667 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
