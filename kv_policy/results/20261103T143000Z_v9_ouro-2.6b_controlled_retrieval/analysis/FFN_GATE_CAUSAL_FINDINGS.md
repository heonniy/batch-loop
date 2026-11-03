# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model ouro-2.6b, dataset controlled_retrieval, pilot UIDs 5, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | -0.01718 | -0.01914 [-0.057414430930512066, 0.0009074273487203772] | -22.02 | 0.03661 | -15.5 |
| oracle_distractor | 5 | -0.01034 | -0.0119 [-0.0319983242941089, 0.00011019934550858534] | -21.63 | -0.02944 | -15.37 |
| oracle_evidence | 5 | 0.08107 | -0.3279 [-0.8072616590000689, 0.0024910996691323824] | -21.63 | -0.2015 | -15.31 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 1_2 | -0.02213 | [-0.06606803590664645, 0.0006535511225229148] |
| main_position_hash | 3_4 | -0.0274 | [-0.08174662742530926, 0.0009165941976243674] |
| oracle_distractor | 1_2 | -0.01679 | [-0.051338113397359796, 0.0021388078946620232] |
| oracle_distractor | 3_4 | -0.007762 | [-0.021294480867218118, 0.0014497858984395817] |
| oracle_evidence | 1_2 | -0.1451 | [-0.7187093955697491, 0.28497908806893973] |
| oracle_evidence | 3_4 | -0.2675 | [-0.7190505488379858, 0.009704951879102744] |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.95 | 0 | 0.05 | 0.3881 | 0.3881 | 0.4074 |
| main_position_hash | 0.85 | 0 | 0.35 | 0.9563 | 0.9563 | 1.031 |
| oracle_distractor | 0.95 | 0 | 0 | 0.3958 | 0.3958 | 0.4068 |
| oracle_distractor | 0.85 | 0 | 0.4 | 0.6475 | 0.6475 | 1.039 |
| oracle_evidence | 0.95 | 0.2 | 0.2 | 0.4847 | 0.366 | 0.7346 |
| oracle_evidence | 0.85 | 0.4 | 1 | 1.587 | 1.268 | 1.153 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
