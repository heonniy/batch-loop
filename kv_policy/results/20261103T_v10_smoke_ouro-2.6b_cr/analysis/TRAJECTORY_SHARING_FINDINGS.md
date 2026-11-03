# V10 Full vs first_last recurrent KV: Context eviction × trajectories × exit

Model ouro-2.6b; controlled_retrieval; 1/1 prompts, 6 source windows.
All arms: normal FULL-depth prefill, identical source window and frozen gold continuation. Full and shared policies are compared WITHIN model.
The first generated token is predicted before post-prefill eviction and excluded from damage.

| stratum | n | Full ΔNLL | Shared ΔNLL | (Shared−Full) eviction interaction | Full trajectory E | Shared trajectory E |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 4 | 0.001129 | 5.362e-06 | -0.001123 | 0.01709 | 0.01673 |
| oracle_distractor | 1 | 0.003912 | 0.0003301 | -0.003582 | 0.01683 | 0.01691 |
| oracle_evidence | 1 | 0.002391 | 0.002027 | -0.000363 | 0.01501 | 0.01558 |

## Fixed-depth recurrent damage curve (equal depth within each policy)
| forced depth | Full eviction ΔNLL | Shared eviction ΔNLL | Interaction |
|---|---:|---:|---:|
| 2 | 0.001786 | 0.00345 | 0.001664 |
| 3 | 0.003719 | 0.0001023 | -0.003617 |
| 4 | 0.001803 | 0.0003965 | -0.001406 |

## Native exit conditions (clean-calibrated, same threshold for both policies)

Threshold 0.8; CLEAN pooled early fraction 0.250; saturated=False
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.001803 | 0 | 0 | 0 |
| first_last | 0 | 0.0303 | 0.001438 | 0 | 0 | 0 |

Threshold 0.7; CLEAN pooled early fraction 0.750; saturated=False
| Policy | Mean fraction query exits changed by eviction | Offline would-exit fraction | Mean live ΔNLL | Mean live−replay gap | Correctness flips vs clean live | Exit×Eviction correctness interaction |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0.0003909 | 0 | 0 | 0 |
| first_last | 0.01515 | 0 | 6.291e-05 | -0.0001366 | 0 | 0 |

Analysis is descriptive; use independent prompt bootstrap and nontrivial damage floor before paper claims.
Dynamic exit alone does not prove quality repair; live vs replay includes altered generated KV history.
Huginn's FP32 cosine is a diagnostic on BF16 states, not official BF16 arithmetic.
This reference uses full-size KV + read mapping (no actual HBM/TPOT savings).
