# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model ouro-1.4b, dataset controlled_retrieval, pilot UIDs 5, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.1231 | 0.09282 [-0.07731143390759823, 0.322422589091584] | -8.282 | -1.538 | -5.425 |
| oracle_distractor | 5 | 0.1262 | 0.1335 [0.0010917013138532106, 0.28758174989372454] | -8.212 | -1.797 | -5.371 |
| oracle_evidence | 5 | 0.1253 | 0.03155 [-0.04674825243651872, 0.09532624036073711] | -8.16 | -0.9718 | -5.617 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 1_2 | 0.08762 | [-0.09096081353724017, 0.32432193612679827] |
| main_position_hash | 3_4 | 0.1477 | [0.0196954356692731, 0.35049207673873745] |
| oracle_distractor | 1_2 | 0.1162 | [-0.010532440766692108, 0.2685032529756427] |
| oracle_distractor | 3_4 | 0.1748 | [0.02078178860247144, 0.35797966942191106] |
| oracle_evidence | 1_2 | 0.007423 | [-0.06891741186380367, 0.058666911777109074] |
| oracle_evidence | 3_4 | 0.03519 | [-0.03293452817946685, 0.08937858758494292] |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.85 | 0.2 | 0.7 | 3.878 | 3.9 | 3.804 |
| main_position_hash | 0.7 | 0.25 | 0.3 | 3.592 | 3.578 | 3.476 |
| oracle_distractor | 0.85 | 0.4 | 0.4 | 3.96 | 3.891 | 3.766 |
| oracle_distractor | 0.7 | 0.4 | 0.4 | 3.595 | 3.581 | 3.426 |
| oracle_evidence | 0.85 | 0.4 | 0.8 | 3.844 | 3.89 | 3.876 |
| oracle_evidence | 0.7 | 0.2 | 0.4 | 3.737 | 3.738 | 3.939 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
