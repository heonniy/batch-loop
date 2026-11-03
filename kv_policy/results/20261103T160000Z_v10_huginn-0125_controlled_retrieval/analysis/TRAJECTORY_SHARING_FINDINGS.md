# V10 Full vs last recurrent KV: Context eviction × trajectories × exit

Model huginn-0125; controlled_retrieval; 5/5 prompts, 30 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | -0.0009968 | 0.002691 | 0.003688 | 0.01624 | 0.01605 |
| oracle_distractor | 5 | -0.004151 | 0.003067 | 0.007219 | 0.01538 | 0.01538 |
| oracle_evidence | 5 | 0.338 | 0.3366 | -0.001424 | 0.1896 | 0.2152 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 8 | 0.05531 | 0.05472 | -0.0005887 |
| 16 | 0.0589 | 0.05289 | -0.006013 |
| 24 | 0.05275 | 0.05949 | 0.006745 |
| 32 | 0.05498 | 0.05841 | 0.003424 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 5.901813507080078e-05 (anchor=pooled); CLEAN early fraction pooled 0.227, by policy {'full': 0.0, 'last': 0.455}; saturated policies=['full']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.05498 | 0 | 0.06667 | 0 |
| last | 0.4636 | 0.4394 | 0.05687 | 0.001102 | 0.06667 | 0 |

Threshold 0.00010836124420166016 (anchor=pooled); CLEAN early fraction pooled 0.500, by policy {'full': 0.0, 'last': 1.0}; saturated policies=['full', 'last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.05498 | 0 | 0.06667 | 0 |
| last | 0.6152 | 0.6424 | 0.05815 | -0.00257 | 0.06667 | 0 |

Threshold 0.00026405453681945803 (anchor=full_only); CLEAN early fraction pooled 0.303, by policy {'full': 0.303, 'last': 1.0}; saturated policies=['last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2394 | 0.2394 | 0.05444 | -0.0007973 | 0.06667 | 0 |
| last | 0.4121 | 0.3939 | 0.05256 | -0.0002774 | 0.06667 | 0 |

Threshold 0.0003294616937637329 (anchor=full_only); CLEAN early fraction pooled 0.636, by policy {'full': 0.636, 'last': 1.0}; saturated policies=['last']
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.3576 | 0.3424 | 0.05916 | 0.003779 | 0.06667 | 0 |
| last | 0.4 | 0.3182 | 0.05842 | -0.004558 | 0.06667 | 0 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
