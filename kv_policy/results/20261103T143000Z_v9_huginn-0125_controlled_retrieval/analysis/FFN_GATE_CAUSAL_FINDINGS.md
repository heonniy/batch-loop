# v9 causal gate and dynamic exit: results readout

Status: all requested pilot units complete
Model huginn-0125, dataset controlled_retrieval, pilot UIDs 5, source window width 4

Important: FFN gate restoration is a privileged oracle, not a deployed policy.
A positive gate repair is NOT, by itself, proof of loop-specific causality.
Dynamic live experiments have actual exit decisions; replay arms force clean depths.
The Huginn FP32 cosine exit is diagnostic on BF16 states.

## Matched FFN restoration (gold future NLL; mean over source windows, [prompt bootstrap 95% CI])

| stratum | n windows | mean eviction penalty | gate restored (positive = helps) | up restored | shifted gate control | full FFN restored |
|---|---:|---:|---:|---:|---:|---:|
| main_position_hash | 20 | 0.002691 | 0.002221 [-0.00011552437332284504, 0.004629711243323997] | -0.06882 | 0.001097 | 0.001948 |
| oracle_distractor | 5 | 0.003067 | 0.003192 [-0.00028305232990535336, 0.006250913584372042] | 0.004783 | -0.0007254 | 0.0007293 |
| oracle_evidence | 5 | 0.3366 | -0.388 [-0.8898727887080167, 0.011092869362910318] | -0.3725 | -0.04778 | -0.05673 |

## Loop-local gate restoration (positive recovery = correct gate patch helps)

| stratum | loops | mean NLL recovered | bootstrap by prompt |
|---|---|---:|---|
| main_position_hash | 17_24 | 0.0007703 | [-0.00182299593972855, 0.003213535366885495] |
| main_position_hash | 1_8 | 0.0006591 | [-0.0012204073109751245, 0.002138188343305958] |
| main_position_hash | 25_32 | 0.002603 | [0.001262641169305434, 0.0040361949709040446] |
| main_position_hash | 9_16 | 0.001517 | [-0.00047119584705798136, 0.003548679274026645] |
| oracle_distractor | 17_24 | 0.003192 | [-0.0029957325709983663, 0.00847173298243431] |
| oracle_distractor | 1_8 | 0.004342 | [-0.0030118028121069564, 0.011630170173011711] |
| oracle_distractor | 25_32 | 0.005416 | [0.0019943187580793165, 0.009215964526229036] |
| oracle_distractor | 9_16 | 0.004211 | [-0.007600021745311114, 0.015217708048876389] |
| oracle_evidence | 17_24 | -0.01286 | [-0.02929661292786485, 5.2092983969531656e-05] |
| oracle_evidence | 1_8 | -0.002932 | [-0.009175112154043763, 0.002874598858470567] |
| oracle_evidence | 25_32 | -0.3812 | [-0.8690073458926055, 0.010046056948485815] |
| oracle_evidence | 9_16 | -0.01581 | [-0.046197103006998044, 0.0038971097726607784] |

## Exit and answer interaction (same clean-only frozen threshold)

| stratum | exit setting | fraction of source windows with changed query depth | fraction with changed greedy answer | mean NLL live | mean NLL replay | mean NLL gate-replay |
|---|---|---:|---:|---:|---:|---:|
| main_position_hash | 0.0001 | 1 | 0 | 0.8993 | 0.9 | 0.8985 |
| main_position_hash | 0.00025 | 1 | 0 | 0.9014 | 0.8988 | 0.8997 |
| oracle_distractor | 0.0001 | 1 | 0 | 0.9034 | 0.8997 | 0.8961 |
| oracle_distractor | 0.00025 | 1 | 0 | 0.9004 | 0.8997 | 0.9031 |
| oracle_evidence | 0.0001 | 1 | 0.4 | 1.227 | 1.235 | 1.616 |
| oracle_evidence | 0.00025 | 1 | 0.4 | 1.232 | 1.231 | 1.614 |

## GO / NO-GO requirements

One must observe a non-noise eviction penalty and a reproducible gate-only
repair that beats up/shift controls, then verify loop-local effect beyond
ordinary layer-wise FFN patching before claiming a loop-specific mechanism.
If live exit changes depth but answers are stable, do not claim quality impact.
If only ORACLE gate patches help, the method remains an offline diagnosis.
This table is exploratory; n=5 prompts per cell is NOT an adequately powered study.
