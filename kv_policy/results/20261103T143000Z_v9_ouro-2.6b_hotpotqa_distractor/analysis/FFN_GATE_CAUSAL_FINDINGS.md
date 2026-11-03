# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model ouro-2.6b, dataset hotpotqa_distractor, pilot UIDs 5, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.03358 | 0.02718 [-0.010648023340544316, 0.09721570942188713] | -18.13 | 0.01862 | -18.55 |
| oracle_distractor | 5 | 0.02761 | 0.03035 [-0.00893820660477056, 0.10029907078404598] | -18.36 | 0.02148 | -18.44 |
| oracle_evidence | 5 | 0.01924 | 0.02726 [-0.01548462880642285, 0.09092933901921957] | -18.31 | 0.03681 | -18.52 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 1_2 | 0.0369 | [0.004657907118200383, 0.09940328111406413] |
| main_position_hash | 3_4 | 0.006787 | [-0.010008410095664045, 0.03743866961643407] |
| oracle_distractor | 1_2 | 0.02636 | [-0.008197831450813698, 0.0895067466630616] |
| oracle_distractor | 3_4 | 0.01351 | [-0.004592105948055769, 0.04123010387411341] |
| oracle_evidence | 1_2 | 0.02321 | [-0.004670445228527687, 0.07116522681381966] |
| oracle_evidence | 3_4 | 0.01205 | [-0.0077376306969820455, 0.03865623454615067] |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.85 | 0.05 | 0.2 | 0.7743 | 0.7736 | 0.766 |
| main_position_hash | 0.7 | 0.05 | 0 | 0.7209 | 0.7215 | 0.7144 |
| oracle_distractor | 0.85 | 0 | 0.2 | 0.7646 | 0.7646 | 0.7628 |
| oracle_distractor | 0.7 | 0 | 0 | 0.7217 | 0.7217 | 0.7136 |
| oracle_evidence | 0.85 | 0 | 0.2 | 0.7752 | 0.7752 | 0.7543 |
| oracle_evidence | 0.7 | 0 | 0 | 0.7177 | 0.7177 | 0.7037 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
