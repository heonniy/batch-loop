# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model huginn-0125, dataset hotpotqa_distractor, pilot UIDs 5, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | -0.008555 | -0.002595 [-0.009003858235317042, 0.0030718929923023277] | -0.01063 | 0.002286 | -0.007311 |
| oracle_distractor | 5 | -0.0123 | 0.003256 [-0.0011363925780945472, 0.007731738424322376] | -0.005259 | -0.0064 | 0.001798 |
| oracle_evidence | 5 | -0.02137 | -0.0127 [-0.03975965753530284, 0.00935245824221056] | -0.2994 | 0.01574 | -0.01143 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 17_24 | 0.002179 | [-0.005895129917355735, 0.010221218110798092] |
| main_position_hash | 1_8 | 0.001864 | [-0.001277557121731132, 0.005384228342477893] |
| main_position_hash | 25_32 | -0.003522 | [-0.012234471788845735, 0.003715312786308844] |
| main_position_hash | 9_16 | 0.00402 | [-0.002763958301499425, 0.013907317280453368] |
| oracle_distractor | 17_24 | -0.007372 | [-0.03253907601562487, 0.009033739997345736] |
| oracle_distractor | 1_8 | -0.001758 | [-0.01190799437618506, 0.008391034822090627] |
| oracle_distractor | 25_32 | -0.002239 | [-0.009802983868236526, 0.0053772696257789136] |
| oracle_distractor | 9_16 | 0.00267 | [-0.00672798033742259, 0.013313694293643152] |
| oracle_evidence | 17_24 | 1.664e-05 | [-0.01646770753515221, 0.011677143733848228] |
| oracle_evidence | 1_8 | 0.001147 | [-0.006440805886107004, 0.007455891922221625] |
| oracle_evidence | 25_32 | -0.01838 | [-0.046873807243953, 0.00531935816271778] |
| oracle_evidence | 9_16 | 0.0002019 | [-0.008957825792021978, 0.009361632645203155] |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.0001 | 1 | 0 | 1.664 | 1.666 | 1.66 |
| main_position_hash | 0.00025 | 0.8 | 0 | 1.661 | 1.668 | 1.672 |
| oracle_distractor | 0.0001 | 1 | 0 | 1.661 | 1.658 | 1.659 |
| oracle_distractor | 0.00025 | 0.8 | 0 | 1.666 | 1.665 | 1.659 |
| oracle_evidence | 0.0001 | 1 | 0 | 1.643 | 1.648 | 1.661 |
| oracle_evidence | 0.00025 | 0.8 | 0 | 1.65 | 1.646 | 1.676 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
