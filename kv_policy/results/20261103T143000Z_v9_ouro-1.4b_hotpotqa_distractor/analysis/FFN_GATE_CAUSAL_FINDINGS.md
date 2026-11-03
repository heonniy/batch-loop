# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model ouro-1.4b, dataset hotpotqa_distractor, pilot UIDs 5, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.02565 | 0.03125 [-0.04582352864879393, 0.13120602985844013] | -11.39 | -2.969 | -8.846 |
| oracle_distractor | 5 | -0.008351 | 0.008538 [-0.02178505187746076, 0.04183150875071693] | -11.22 | -1.911 | -8.707 |
| oracle_evidence | 5 | 0.04859 | 0.04282 [-0.05490361099979548, 0.14048724303997692] | -11.6 | -3.868 | -8.75 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 1_2 | 0.02713 | [-0.07028709237264205, 0.15707109193184543] |
| main_position_hash | 3_4 | -0.02171 | [-0.07652157040179841, 0.012335595085896767] |
| oracle_distractor | 1_2 | 0.01636 | [-0.001150328264754652, 0.03203065637085167] |
| oracle_distractor | 3_4 | 0.01777 | [-0.01980833630701202, 0.05351993252833651] |
| oracle_evidence | 1_2 | 0.08819 | [-0.015345239660236843, 0.21012547842110507] |
| oracle_evidence | 3_4 | 0.06459 | [0.022219995361354085, 0.12134527117013923] |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.85 | 0.05 | 0.05 | 3.283 | 3.283 | 3.254 |
| main_position_hash | 0.7 | 0.4 | 0.2 | 2.81 | 2.789 | 2.777 |
| oracle_distractor | 0.85 | 0.2 | 0 | 3.263 | 3.263 | 3.243 |
| oracle_distractor | 0.7 | 0.2 | 0.2 | 2.722 | 2.722 | 2.76 |
| oracle_evidence | 0.85 | 0 | 0.2 | 3.327 | 3.327 | 3.278 |
| oracle_evidence | 0.7 | 0.4 | 0 | 2.804 | 2.808 | 2.764 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
