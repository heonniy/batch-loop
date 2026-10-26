# E2 — failure localisation (current-token interventions, frozen FULL history)

Run `results/20261026T080106Z_huginn_ouro_full_last_convert`: 96 prompt records (ouro-1.4b: 32, ouro-2.6b: 32, huginn-0125: 32); 3 decode sites per prompt on each model's own FULL greedy prefix; BF16. Huginn r4/r8/r16 = separately labelled non-native depth sweep. E = relative state error vs the unperturbed FULL trajectory; KL = KL(FULL ‖ intervened) of the final logits. Values: mean over prompts [95% bootstrap CI].

Parity (functional single-token map vs engine, BF16): max |Δlogit| 0.578 over 288 sites, top-1 identical 288/288.

## a. Fixed-query cross-depth attention (layer mean): relative attention-output change when step r reads step-s history

- **ouro-1.4b** (R=4): reading terminal history (s=R): r=1: 0.455, r=2: 0.275, r=3: 0.147; reading step-1 history from the terminal step (r=R, s=1): 2.016
- **ouro-2.6b** (R=4): reading terminal history (s=R): r=1: 0.416, r=2: 0.232, r=3: 0.121; reading step-1 history from the terminal step (r=R, s=1): 1.540
- **huginn-0125** (R=32): reading terminal history (s=R): r=1: 0.822, r=2: 0.801, r=4: 0.556, r=8: 0.274, r=16: 0.077, r=24: 0.025, r=31: 0.013; reading step-1 history from the terminal step (r=R, s=1): 1.011

## b. Single-call terminal read (one layer, one step) vs matched-norm random shift, propagated to the output

| model | r | E_inj (kv) | E_final kv | E_final random | per-step decay kv | final KL kv | final KL random | top-1 kept kv |
|---|---:|---|---|---|---|---|---|---:|
| ouro-1.4b | 1 | 0.88 | 0.0609 [0.0524, 0.0702] | 0.0897 | 0.503 | 0.0543 [0.0376, 0.071] | 0.198 | 0.988 |
| ouro-1.4b | 2 | 0.097 | 0.029 [0.0222, 0.0371] | 0.0266 | 0.522 | 0.000205 [0.000137, 0.00029] | 0.000215 | 0.998 |
| ouro-1.4b | 3 | 0.0482 | 0.0216 [0.0193, 0.0242] | 0.0212 | 0.498 | 0.000196 [0.000134, 0.000264] | 0.000198 | 0.998 |
| ouro-2.6b | 1 | 0.604 | 0.0373 [0.03, 0.0455] | 0.0631 | 0.541 | 0.0118 [0.00429, 0.0212] | 0.131 | 0.997 |
| ouro-2.6b | 2 | 0.0771 | 0.0277 [0.0202, 0.0368] | 0.0269 | 0.585 | 0.000268 [0.000179, 0.000369] | 0.000289 | 1.000 |
| ouro-2.6b | 3 | 0.0395 | 0.0218 [0.0184, 0.0264] | 0.0213 | 0.64 | 0.000261 [0.000166, 0.000365] | 0.000265 | 1.000 |
| huginn-0125 | 1 | 0.338 | 0.0136 [0.0134, 0.0138] | 0.0136 | 0.905 | 0.000197 [0.000126, 0.000279] | 0.000207 | 1.000 |
| huginn-0125 | 2 | 0.28 | 0.0136 [0.0134, 0.0138] | 0.0136 | 0.906 | 0.000217 [0.000143, 0.000302] | 0.000183 | 0.997 |
| huginn-0125 | 4 | 0.173 | 0.0136 [0.0134, 0.0139] | 0.0136 | 0.915 | 0.000215 [0.000142, 0.000299] | 0.000204 | 1.000 |
| huginn-0125 | 8 | 0.0765 | 0.0137 [0.0135, 0.0139] | 0.0137 | 0.933 | 0.000225 [0.000134, 0.000335] | 0.000213 | 1.000 |
| huginn-0125 | 16 | 0.0215 | 0.0137 [0.0135, 0.0139] | 0.0137 | 0.976 | 0.000209 [0.000132, 0.000295] | 0.000223 | 0.997 |
| huginn-0125 | 24 | 0.00816 | 0.0137 [0.0135, 0.0139] | 0.0137 | 1.07 | 0.000202 [0.000125, 0.000287] | 0.000216 | 1.000 |
| huginn-0125 | 31 | 0.00521 | 0.0147 [0.0143, 0.0153] | 0.0143 | 3.02 | 0.000229 [0.000148, 0.000326] | 0.00021 | 1.000 |

Layer x step sensitivity (final KL of the single-call terminal read, mean over prompts; top-5 cells per model):

- ouro-1.4b: layer 0 step 1: KL 1.29; layer 1 step 1: KL 0.0111; layer 2 step 1: KL 0.00102; layer 3 step 1: KL 0.000582; layer 9 step 1: KL 0.000322
- ouro-2.6b: layer 0 step 1: KL 0.506; layer 1 step 1: KL 0.0303; layer 5 step 1: KL 0.00976; layer 2 step 1: KL 0.0057; layer 38 step 3: KL 0.000419
- huginn-0125: layer 3 step 31: KL 0.000315; layer 3 step 4: KL 0.000286; layer 3 step 8: KL 0.000274; layer 3 step 2: KL 0.000269; layer 1 step 8: KL 0.000249

## c. Persistent terminal reads for the current token (history unchanged)

| model | reads replaced | E_final | final KL | top-1 kept |
|---|---|---|---|---:|
| ouro-1.4b | step1_all_layers | 0.624 [0.545, 0.708] | 0.914 [0.55, 1.29] | 0.854 |
| ouro-1.4b | step2_all_layers | 0.119 [0.0841, 0.162] | 0.00103 [0.000645, 0.0015] | 1.000 |
| ouro-1.4b | step3_all_layers | 0.0821 [0.0698, 0.0964] | 0.0013 [0.000476, 0.00259] | 1.000 |
| ouro-1.4b | all_steps_all_layers | 0.691 [0.622, 0.765] | 0.895 [0.581, 1.23] | 0.833 |
| ouro-2.6b | step1_all_layers | 0.378 [0.295, 0.458] | 0.611 [0.216, 1.15] | 0.865 |
| ouro-2.6b | step2_all_layers | 0.0973 [0.0732, 0.127] | 0.00117 [0.000732, 0.0018] | 1.000 |
| ouro-2.6b | step3_all_layers | 0.0875 [0.0674, 0.113] | 0.00102 [0.000606, 0.00147] | 1.000 |
| ouro-2.6b | all_steps_all_layers | 0.401 [0.323, 0.478] | 0.449 [0.0852, 0.991] | 0.917 |
| huginn-0125 | step1_all_layers | 0.0137 [0.0135, 0.0139] | 0.00024 [0.000162, 0.00033] | 1.000 |
| huginn-0125 | step2_all_layers | 0.0137 [0.0135, 0.0139] | 0.000217 [0.000111, 0.000351] | 1.000 |
| huginn-0125 | step4_all_layers | 0.0136 [0.0134, 0.0139] | 0.000191 [0.000109, 0.000284] | 0.990 |
| huginn-0125 | step8_all_layers | 0.0136 [0.0134, 0.0138] | 0.000233 [0.000144, 0.000356] | 1.000 |
| huginn-0125 | step16_all_layers | 0.0137 [0.0135, 0.014] | 0.000219 [0.000139, 0.000303] | 1.000 |
| huginn-0125 | step24_all_layers | 0.0138 [0.0136, 0.0141] | 0.000182 [9.79e-05, 0.000309] | 1.000 |
| huginn-0125 | step31_all_layers | 0.0132 [0.0123, 0.0145] | 0.000273 [0.000142, 0.000446] | 1.000 |
| huginn-0125 | all_steps_all_layers | 0.0225 [0.0203, 0.0256] | 0.000451 [0.000249, 0.000696] | 1.000 |

Single-call shifts are recovered under an unchanged history if E_final ≪ E_inj; persistent rows show whether replacing every read of a step moves the token to a different output (changed fixed point) rather than a recoverable perturbation.

