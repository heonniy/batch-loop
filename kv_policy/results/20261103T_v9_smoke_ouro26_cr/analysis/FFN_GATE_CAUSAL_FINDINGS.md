# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model ouro-2.6b, dataset controlled_retrieval, pilot UIDs 1, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 4 | -5.028e-06 | 0.0006755 None | -22.22 | 0.0003846 | -16.27 |
| oracle_distractor | 1 | 0.001575 | 0.001035 None | -23.11 | -0.0009777 | -15.99 |
| oracle_evidence | 1 | 1.146 | -0.2377 None | -21.53 | -0.8195 | -15.03 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 1_2 | -0.001797 | None |
| main_position_hash | 3_4 | 0.000614 | None |
| oracle_distractor | 1_2 | 0.00148 | None |
| oracle_distractor | 3_4 | 0.005328 | None |
| oracle_evidence | 1_2 | 0.02682 | None |
| oracle_evidence | 3_4 | -0.359 | None |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.95 | 0.25 | 0 | 0.2723 | 0.319 | 0.3303 |
| main_position_hash | 0.7 | 0 | 0 | 0.2838 | 0.2838 | 0.2815 |
| oracle_distractor | 0.95 | 0 | 0 | 0.3169 | 0.3169 | 0.3253 |
| oracle_distractor | 0.7 | 1 | 0 | 0.2859 | 0.2862 | 0.2812 |
| oracle_evidence | 0.95 | 0 | 1 | 1.166 | 1.166 | 2.01 |
| oracle_evidence | 0.7 | 1 | 1 | 1.222 | 1.236 | 2.449 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
