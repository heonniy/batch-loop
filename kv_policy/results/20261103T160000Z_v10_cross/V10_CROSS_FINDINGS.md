# V10 cross-cell findings: FULL vs architecture-specific shared recurrent KV × Context eviction × trajectories × exit

Shared policy: Huginn = LAST plane for every query loop; Ouro = FIRST plane for query loop 1, LAST plane for loops ≥2. Same prompts, same frozen gold continuation, same 4-token windows (4 label-blind 'main' + 1 evidence + 1 distractor oracle per prompt). Interaction = (shared eviction ΔNLL) − (full eviction ΔNLL); positive = sharing makes the model MORE sensitive to eviction. CIs: prompt bootstrap (5 prompts per cell: wide).

- huginn-0125/controlled_retrieval: 5/5 prompts; thresholds 5.9e-05 (pooled, clean early full/shared 0.00/0.45); 0.000108 (pooled, clean early full/shared 0.00/1.00); 0.000264 (full_only, clean early full/shared 0.30/1.00); 0.000329 (full_only, clean early full/shared 0.64/1.00)
- huginn-0125/hotpotqa_distractor: 5/5 prompts; thresholds 5.83e-05 (pooled, clean early full/shared 0.00/0.58); 0.0001 (pooled, clean early full/shared 0.00/0.89); 0.000183 (full_only, clean early full/shared 0.42/0.95); 0.000211 (full_only, clean early full/shared 0.63/0.95)
- ouro-1.4b/controlled_retrieval: 5/5 prompts; thresholds 0.8 (pooled, clean early full/shared 0.00/0.09); 0.7 (pooled, clean early full/shared 0.52/1.00); 0.6 (full_only, clean early full/shared 0.67/1.00)
- ouro-1.4b/hotpotqa_distractor: 5/5 prompts; thresholds 0.7 (pooled, clean early full/shared 0.11/0.58); 0.5 (pooled, clean early full/shared 0.37/0.95); 0.6 (full_only, clean early full/shared 0.21/0.84); 0.3 (full_only, clean early full/shared 0.63/1.00)
- ouro-2.6b/controlled_retrieval: 5/5 prompts; thresholds 0.8 (pooled, clean early full/shared 0.00/0.42); 0.7 (pooled, clean early full/shared 0.55/0.97); 0.6 (full_only, clean early full/shared 0.64/0.97)
- ouro-2.6b/hotpotqa_distractor: 5/5 prompts; thresholds 0.8 (pooled, clean early full/shared 0.00/0.05); 0.6 (pooled, clean early full/shared 0.37/0.89); 0.7 (full_only, clean early full/shared 0.16/0.84); 0.4 (full_only, clean early full/shared 0.74/1.00)

## 0. Clean (no eviction) cost of the shared policy and numerical floors

| model | dataset | shared − full NLL [CI] | KL(full‖shared) | final-loop state rel. diff full vs shared | EM fixed full → shared | repeat floor (logprob) |
|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | -0.0113 [-0.0187, -0.0052] | 0.000103 | 0.0276 | 1 → 1 | 0 |
| huginn-0125 | hotpotqa_distractor | 0.00675 [-0.0147, 0.0279] | 0.000794 | 0.0277 | 0.2 → 0.2 | 0 |
| ouro-1.4b | controlled_retrieval | 0.00641 [0.00197, 0.0109] | 0.0133 | 0.208 | 1 → 1 | 0 |
| ouro-1.4b | hotpotqa_distractor | -0.162 [-0.385, -0.00745] | 0.0984 | 0.194 | 0.2 → 0.2 | 0 |
| ouro-2.6b | controlled_retrieval | 0.0252 [0.0221, 0.0275] | 0.0069 | 0.2 | 1 → 1 | 0 |
| ouro-2.6b | hotpotqa_distractor | 0.0923 [-0.0158, 0.188] | 0.0295 | 0.149 | 0.4 → 0.2 | 0 |

## 1. H2 — eviction penalty under FULL vs SHARED at native depth (fixed R), and the interaction

| model | dataset | stratum | n | full ΔNLL [CI] | shared ΔNLL [CI] | interaction [CI] | KL full / shared | EM flips full / shared |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 20 | -0.000997 [-0.00289, 0.000547] | 0.00269 [-0.00204, 0.00742] | 0.00369 [-0.00224, 0.00961] | 6.44e-05 / 6.86e-05 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 5 | -0.00415 [-0.00718, -0.000678] | 0.00307 [0.000277, 0.00586] | 0.00722 [0.00335, 0.0117] | 7.9e-05 / 7.19e-05 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 5 | 0.338 [0.000347, 0.699] | 0.337 [0.00671, 0.689] | -0.00142 [-0.0122, 0.0108] | 0.41 / 0.403 | 0.4 / 0.4 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 20 | -0.00666 [-0.0321, 0.02] | -0.00856 [-0.0259, 0.0118] | -0.00189 [-0.0131, 0.0153] | 0.00255 / 0.00238 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 5 | -0.00434 [-0.0197, 0.0147] | -0.0123 [-0.0284, 0.000286] | -0.00796 [-0.0297, 0.0138] | 0.000377 / 0.000336 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 5 | -0.0212 [-0.043, 0.00462] | -0.0214 [-0.0473, 0.00214] | -0.000158 [-0.0158, 0.0204] | 0.00221 / 0.00261 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 20 | -0.000705 [-0.00254, 0.000699] | 0.000365 [-0.00183, 0.00279] | 0.00107 [-0.00202, 0.00379] | 0.000123 / 0.000165 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 5 | -0.00103 [-0.0037, 0.000798] | -0.00176 [-0.00384, 0.000837] | -0.000731 [-0.00366, 0.00256] | 0.000121 / 0.000137 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 5 | 0.182 [-0.00293, 0.539] | 0.14 [-0.00401, 0.417] | -0.0417 [-0.124, 0.002] | 0.18 / 0.138 | 0.2 / 0.2 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 20 | 0.00418 [-0.00803, 0.0225] | 0.000462 [-0.0144, 0.0226] | -0.00372 [-0.0101, 0.0017] | 0.00297 / 0.00307 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 5 | -0.00357 [-0.0108, 0.00547] | -0.00205 [-0.00705, 0.00215] | 0.00152 [-0.00493, 0.00762] | 0.000245 / 0.000308 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 5 | -0.0236 [-0.0589, -0.00204] | -0.00905 [-0.0289, 0.00829] | 0.0146 [-0.00543, 0.0359] | 0.00139 / 0.00429 | 0 / 0.2 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 20 | 0.00151 [-0.000101, 0.00282] | -0.000246 [-0.0025, 0.00186] | -0.00176 [-0.00373, 0.000213] | 0.000125 / 0.00011 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 5 | 0.00185 [0.000919, 0.00299] | -0.000539 [-0.0034, 0.00289] | -0.00239 [-0.00521, 0.00149] | 6.97e-05 / 0.000104 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 5 | 0.234 [0.00118, 0.606] | 0.228 [-0.00324, 0.665] | -0.00606 [-0.105, 0.0913] | 0.229 / 0.227 | 0.4 / 0.2 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 20 | 0.00491 [-0.00625, 0.0194] | -0.000774 [-0.00529, 0.00356] | -0.00569 [-0.0165, 0.00163] | 0.00332 / 0.00114 | 0 / 0.05 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 5 | 0.00052 [-0.00211, 0.0034] | -0.00586 [-0.00868, -0.00313] | -0.00638 [-0.01, -0.00271] | 0.000153 / 0.00025 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 5 | -0.00472 [-0.0215, 0.0123] | -0.00362 [-0.0171, 0.00725] | 0.00111 [-0.0228, 0.022] | 0.00254 / 0.000764 | 0.2 / 0.2 |

## 2. H2 at matched forced depth (eviction ΔNLL full / shared, interaction), main + oracle strata pooled per label

| model | dataset | stratum | depth → full / shared / interaction |
|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | d=8: -0.000739 / 0.000575 / 0.00131; d=16: 0.00243 / -0.00199 / -0.00442; d=24: -0.00349 / 0.00224 / 0.00573; d=32: -0.000997 / 0.00269 / 0.00369 |
| huginn-0125 | controlled_retrieval | oracle_distractor | d=8: 0.00101 / -0.00129 / -0.0023; d=16: 0.000607 / 0.0016 / 0.000995; d=24: -0.00156 / 0.000742 / 0.0023; d=32: -0.00415 / 0.00307 / 0.00722 |
| huginn-0125 | controlled_retrieval | oracle_evidence | d=8: 0.334 / 0.327 / -0.00649; d=16: 0.343 / 0.324 / -0.0194; d=24: 0.332 / 0.347 / 0.0152; d=32: 0.338 / 0.337 / -0.00142 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | d=8: 0.00593 / 0.00653 / 0.000605; d=16: 0.000766 / -0.00714 / -0.00791; d=24: 0.00374 / 0.000133 / -0.0036; d=32: -0.00666 / -0.00856 / -0.00189 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | d=8: -0.0107 / -0.00542 / 0.00525; d=16: -0.00312 / -0.00637 / -0.00324; d=24: 0.00905 / -0.00288 / -0.0119; d=32: -0.00434 / -0.0123 / -0.00796 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | d=8: 0.00524 / 0.0188 / 0.0136; d=16: -0.00861 / -0.00913 / -0.000526; d=24: 0.00136 / -0.015 / -0.0164; d=32: -0.0212 / -0.0214 / -0.000158 |
| ouro-1.4b | controlled_retrieval | main_position_hash | d=2: -0.000941 / -0.000327 / 0.000614; d=3: -0.000238 / -0.000621 / -0.000383; d=4: -0.000705 / 0.000365 / 0.00107 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | d=2: -0.00299 / 0.000451 / 0.00344; d=3: 0.0012 / -0.000144 / -0.00134; d=4: -0.00103 / -0.00176 / -0.000731 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | d=2: 0.161 / 0.202 / 0.041; d=3: 0.189 / 0.17 / -0.0187; d=4: 0.182 / 0.14 / -0.0417 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | d=2: 0.0268 / -0.0212 / -0.048; d=3: 0.0123 / 0.00362 / -0.00873; d=4: 0.00418 / 0.000462 / -0.00372 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | d=2: -0.00149 / -0.0099 / -0.00841; d=3: 0.00852 / -0.00273 / -0.0113; d=4: -0.00357 / -0.00205 / 0.00152 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | d=2: -0.00385 / -0.0205 / -0.0166; d=3: 0.0206 / -0.00943 / -0.03; d=4: -0.0236 / -0.00905 / 0.0146 |
| ouro-2.6b | controlled_retrieval | main_position_hash | d=2: -0.00224 / 0.000412 / 0.00265; d=3: 6.11e-05 / -2.74e-05 / -8.85e-05; d=4: 0.00151 / -0.000246 / -0.00176 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | d=2: -0.00766 / -0.000592 / 0.00706; d=3: -0.000194 / 5.13e-05 / 0.000246; d=4: 0.00185 / -0.000539 / -0.00239 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | d=2: 0.172 / 0.183 / 0.0112; d=3: 0.242 / 0.235 / -0.00643; d=4: 0.234 / 0.228 / -0.00606 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | d=2: 0.00224 / 0.00177 / -0.00047; d=3: 0.00341 / 0.0018 / -0.00161; d=4: 0.00491 / -0.000774 / -0.00569 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | d=2: 0.00367 / 0.00682 / 0.00315; d=3: 0.00289 / 0.000325 / -0.00257; d=4: 0.00052 / -0.00586 / -0.00638 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | d=2: 4.26e-05 / 0.000169 / 0.000126; d=3: -0.00276 / -0.00345 / -0.000692; d=4: -0.00472 / -0.00362 / 0.00111 |

## 3. H1 — eviction-induced change of the generated query's recurrent state (relative to the same policy's clean run)

| model | dataset | stratum | E full: loop 1 → final | E shared: loop 1 → final | |Δh| final full / shared | per-loop curve (shared − full) at r=1, R/4, R/2, R |
|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 0.00944 → 0.0167 | 0.0105 → 0.0163 | 1.27 / 1.24 | r1: 0.00108, r8: 0.000534, r16: -0.00082, r32: -0.000418 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.00911 → 0.016 | 0.00995 → 0.0156 | 1.22 / 1.19 | r1: 0.00084, r8: 0.000682, r16: -0.000529, r32: -0.000418 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.0118 → 0.225 | 0.095 → 0.224 | 17.1 / 17.1 | r1: 0.0832, r8: 0.0338, r16: 0.000896, r32: -0.000298 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.0112 → 0.0244 | 0.0122 → 0.0241 | 1.86 / 1.84 | r1: 0.000968, r8: 0.000702, r16: -0.00279, r32: -0.000301 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 0.0102 → 0.0161 | 0.0112 → 0.0163 | 1.23 / 1.24 | r1: 0.000989, r8: -0.00152, r16: 0.00042, r32: 0.000144 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.00967 → 0.0482 | 0.0172 → 0.0508 | 3.68 / 3.88 | r1: 0.00753, r8: 0.0153, r16: 0.0241, r32: 0.00264 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.0188 → 0.0267 | 0.0188 → 0.0209 | 1.01 / 0.793 | r1: 0, r2: -0.00165, r4: -0.00581 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.0127 → 0.0247 | 0.0127 → 0.0159 | 0.933 / 0.601 | r1: 0, r2: -0.00294, r4: -0.00881 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.0817 → 0.114 | 0.0817 → 0.096 | 4.22 / 3.51 | r1: 0, r2: -0.0113, r4: -0.0176 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.026 → 0.0238 | 0.026 → 0.0234 | 0.996 / 0.982 | r1: 0, r2: -0.00167, r4: -0.000373 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.0177 → 0.0178 | 0.0177 → 0.0163 | 0.757 / 0.697 | r1: 0, r2: -0.000402, r4: -0.00152 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.0293 → 0.0243 | 0.0293 → 0.0249 | 1.04 / 1.06 | r1: 0, r2: -0.00103, r4: 0.000524 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 0.0165 → 0.027 | 0.0165 → 0.0195 | 1.13 / 0.827 | r1: 0, r2: -0.00111, r4: -0.00746 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 0.0125 → 0.0278 | 0.0125 → 0.0175 | 1.15 / 0.746 | r1: 0, r2: 0.00043, r4: -0.0103 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.0981 → 0.125 | 0.0981 → 0.106 | 5.09 / 4.37 | r1: 0, r2: -0.00638, r4: -0.019 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.0214 → 0.0263 | 0.0214 → 0.0244 | 1.2 / 1.12 | r1: 0, r2: 0.0018, r4: -0.00187 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.0161 → 0.0172 | 0.0161 → 0.0169 | 0.803 / 0.787 | r1: 0, r2: 0.000132, r4: -0.000341 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.0207 → 0.0223 | 0.0207 → 0.0211 | 1.03 / 0.982 | r1: 0, r2: -0.0008, r4: -0.00116 |

## 4. Clean native exit under each policy (same threshold for both) — realized depths and their cost

| model | dataset | threshold (anchor) | clean mean depth full / shared | clean early fraction full / shared | live − fixed ΔNLL full / shared | greedy EM live full / shared |
|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 5.9e-05 (pooled) | 32 / 28.3 | 0 / 0.42 | 0 / 0.000388 | 1 / 1 |
| huginn-0125 | controlled_retrieval | 0.000108 (pooled) | 32 / 20.2 | 0 / 0.93 | 0 / -0.0014 | 1 / 1 |
| huginn-0125 | controlled_retrieval | 0.000264 (full_only) | 31.3 / 16.8 | 0.29 / 0.93 | -0.00183 / 0.00658 | 1 / 1 |
| huginn-0125 | controlled_retrieval | 0.000329 (full_only) | 30.4 / 16 | 0.53 / 0.98 | -0.00393 / 0.00106 | 1 / 1 |
| huginn-0125 | hotpotqa_distractor | 5.83e-05 (pooled) | 32 / 26.2 | 0 / 0.67 | 0 / -0.00965 | 0.2 / 0.2 |
| huginn-0125 | hotpotqa_distractor | 0.0001 (pooled) | 32 / 18.5 | 0 / 0.96 | 0 / -0.00648 | 0.2 / 0.2 |
| huginn-0125 | hotpotqa_distractor | 0.000183 (full_only) | 31.4 / 16.3 | 0.31 / 0.96 | 0.00162 / -0.00432 | 0.2 / 0.2 |
| huginn-0125 | hotpotqa_distractor | 0.000211 (full_only) | 30.9 / 16 | 0.52 / 0.96 | -0.0043 / -0.00168 | 0.2 / 0.2 |
| ouro-1.4b | controlled_retrieval | 0.8 (pooled) | 4 / 3.96 | 0 / 0.036 | 0 / -0.00068 | 1 / 1 |
| ouro-1.4b | controlled_retrieval | 0.7 (pooled) | 3.51 / 3 | 0.49 / 1 | 0.000946 / -0.00264 | 1 / 1 |
| ouro-1.4b | controlled_retrieval | 0.6 (full_only) | 3.33 / 2.71 | 0.67 / 1 | 0.0102 / -0.00333 | 1 / 1 |
| ouro-1.4b | hotpotqa_distractor | 0.7 (pooled) | 3.85 / 3.48 | 0.15 / 0.52 | 0.00208 / 0.0159 | 0.2 / 0.2 |
| ouro-1.4b | hotpotqa_distractor | 0.5 (pooled) | 3.61 / 2.79 | 0.39 / 0.88 | 0.0984 / 0.0359 | 0.2 / 0.2 |
| ouro-1.4b | hotpotqa_distractor | 0.6 (full_only) | 3.71 / 3.09 | 0.29 / 0.82 | 0.0471 / 0.0119 | 0.2 / 0.2 |
| ouro-1.4b | hotpotqa_distractor | 0.3 (full_only) | 3.21 / 2.38 | 0.61 / 0.94 | 0.0833 / 0.0347 | 0.2 / 0.2 |
| ouro-2.6b | controlled_retrieval | 0.8 (pooled) | 4 / 3.64 | 0 / 0.36 | 0 / -0.000786 | 1 / 1 |
| ouro-2.6b | controlled_retrieval | 0.7 (pooled) | 3.53 / 3 | 0.47 / 1 | 3.45e-05 / -0.0143 | 1 / 1 |
| ouro-2.6b | controlled_retrieval | 0.6 (full_only) | 3.31 / 2.64 | 0.69 / 1 | 0.000779 / -0.0132 | 1 / 1 |
| ouro-2.6b | hotpotqa_distractor | 0.8 (pooled) | 4 / 3.85 | 0 / 0.15 | 0 / 0.000938 | 0.4 / 0.2 |
| ouro-2.6b | hotpotqa_distractor | 0.6 (pooled) | 3.6 / 3 | 0.4 / 0.88 | 0.019 / -0.0708 | 0.4 / 0.4 |
| ouro-2.6b | hotpotqa_distractor | 0.7 (full_only) | 3.75 / 3.2 | 0.26 / 0.8 | -0.00141 / -0.0611 | 0.4 / 0.4 |
| ouro-2.6b | hotpotqa_distractor | 0.4 (full_only) | 3.19 / 2.24 | 0.63 / 1 | 0.00538 / 0.0793 | 0.4 / 0.4 |

## 5. H3/H4 — eviction × exit: would-exit (from fixed-depth traces), live depth change, live penalty, answers

| model | dataset | stratum | threshold (anchor) | would-exit changed full / shared | live depth changed full / shared | Σdepth Δ full / shared | live ΔNLL full / shared | live interaction [CI] | live−replay full / shared | answer changed full / shared | correctness flip full / shared |
|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | main_position_hash | 5.9e-05 (pooled) | 0 / 0.44 | 0 / 0.47 | 0 / 12.1 | -0.000997 / 0.00154 | 0.00254 [0.00064, 0.00511] | 0 / 0.00111 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | main_position_hash | 0.000108 (pooled) | 0 / 0.63 | 0 / 0.6 | 0 / 5.35 | -0.000997 / 0.00233 | 0.00332 [-0.00249, 0.0103] | 0 / -0.000913 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | main_position_hash | 0.000264 (full_only) | 0.25 / 0.34 | 0.25 / 0.39 | -1.45 / 0.1 | -0.00173 / -0.00382 | -0.00209 [-0.0103, 0.00585] | -0.00169 / -0.00149 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | main_position_hash | 0.000329 (full_only) | 0.34 / 0.27 | 0.36 / 0.4 | -0.2 / 3.2 | 0.00359 / 0.0031 | -0.000485 [-0.00578, 0.00481] | 0.00498 / 0.00027 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 5.9e-05 (pooled) | 0 / 0.44 | 0 / 0.44 | 0 / 10.8 | -0.00415 / 0.00116 | 0.00531 [-0.00172, 0.0108] | 0 / 0.00205 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.000108 (pooled) | 0 / 0.62 | 0 / 0.64 | 0 / -1.4 | -0.00415 / 0.00511 | 0.00927 [0.00281, 0.0157] | 0 / 0.00122 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.000264 (full_only) | 0.18 / 0.47 | 0.18 / 0.35 | 0.4 / 1.2 | -0.00427 / 0.000185 | 0.00446 [0.000906, 0.00807] | -0.00279 / 0.00082 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_distractor | 0.000329 (full_only) | 0.36 / 0.35 | 0.36 / 0.35 | 0.6 / 1.4 | 0.000334 / -0.000231 | -0.000565 [-0.01, 0.0101] | 0.00123 / -0.00285 | 0 / 0 | 0 / 0 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 5.9e-05 (pooled) | 0 / 0.45 | 0 / 0.45 | 0 / 16 | 0.338 / 0.334 | -0.00415 [-0.0164, 0.00396] | 0 / 0.000106 | 0.4 / 0.4 | 0.4 / 0.4 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.000108 (pooled) | 0 / 0.73 | 0 / 0.64 | 0 / 17.4 | 0.338 / 0.334 | -0.00356 [-0.0129, 0.00795] | 0 / -0.013 | 0.4 / 0.4 | 0.4 / 0.4 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.000264 (full_only) | 0.24 / 0.55 | 0.24 / 0.58 | 0.4 / 12.8 | 0.338 / 0.33 | -0.00739 [-0.0321, 0.0105] | 0.00477 / 0.00348 | 0.4 / 0.4 | 0.4 / 0.4 |
| huginn-0125 | controlled_retrieval | oracle_evidence | 0.000329 (full_only) | 0.35 / 0.49 | 0.35 / 0.47 | 3.4 / 13.4 | 0.34 / 0.338 | -0.00194 [-0.0124, 0.00567] | 0.00155 / -0.0256 | 0.4 / 0.4 | 0.4 / 0.4 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 5.83e-05 (pooled) | 0 / 0.6 | 0 / 0.67 | 0 / 1.75 | -0.00666 / 0.00123 | 0.0079 [-0.00941, 0.0248] | 0 / 0.000232 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.0001 (pooled) | 0 / 0.56 | 0 / 0.53 | 0 / 2.2 | -0.00666 / -0.00022 | 0.00644 [-0.00749, 0.0206] | 0 / -0.00205 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.000183 (full_only) | 0.23 / 0.3 | 0.24 / 0.29 | 0.25 / 0.55 | -0.00407 / 0.00217 | 0.00624 [-0.00452, 0.0208] | -0.00211 / 0.00809 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | main_position_hash | 0.000211 (full_only) | 0.28 / 0.35 | 0.3 / 0.35 | 0.2 / -0.15 | 0.000627 / -0.00533 | -0.00596 [-0.0108, -0.00102] | -0.00899 / 0.000857 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 5.83e-05 (pooled) | 0 / 0.65 | 0 / 0.69 | 0 / -0.2 | -0.00434 / -0.00225 | 0.00209 [-0.0186, 0.0158] | 0 / 0.0014 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 0.0001 (pooled) | 0 / 0.63 | 0 / 0.7 | 0 / 3.2 | -0.00434 / -0.00263 | 0.00171 [-0.0101, 0.0113] | 0 / 0.00274 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 0.000183 (full_only) | 0.22 / 0.28 | 0.22 / 0.21 | 0.8 / -0.8 | -0.0106 / -0.0094 | 0.00125 [-0.00359, 0.00885] | -0.00409 / 9.87e-05 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_distractor | 0.000211 (full_only) | 0.31 / 0.21 | 0.38 / 0.19 | 0 / -0.2 | 0.00418 / -0.00879 | -0.013 [-0.0232, -0.00419] | -0.0149 / -0.00114 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 5.83e-05 (pooled) | 0 / 0.74 | 0 / 0.7 | 0 / -2.8 | -0.0212 / -0.0123 | 0.00887 [-0.0198, 0.0375] | 0 / 0.00731 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.0001 (pooled) | 0 / 0.58 | 0 / 0.48 | 0 / 1 | -0.0212 / -0.0211 | 0.000123 [-0.0304, 0.0219] | 0 / -0.00487 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.000183 (full_only) | 0.24 / 0.28 | 0.24 / 0.36 | 0.2 / -0.6 | -0.0145 / -0.0209 | -0.00638 [-0.0271, 0.0144] | 0.00199 / -0.00301 | 0 / 0 | 0 / 0 |
| huginn-0125 | hotpotqa_distractor | oracle_evidence | 0.000211 (full_only) | 0.37 / 0.38 | 0.34 / 0.34 | -0.2 / -0.6 | 0.00175 / -0.0191 | -0.0209 [-0.0356, -0.00595] | -0.0043 / 0.00287 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.8 (pooled) | 0 / 0.023 | 0 / 0.032 | 0 / -0.15 | -0.000705 / 0.000592 | 0.0013 [-0.000508, 0.0031] | 0 / -0.000752 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.7 (pooled) | 0.036 / 0 | 0.027 / 0 | 0.3 / 0 | -0.000529 / -0.000621 | -9.26e-05 [-0.00128, 0.00106] | -0.000408 / 0 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | main_position_hash | 0.6 (full_only) | 0.0091 / 0.027 | 0.0091 / 0.014 | 0.1 / 0.05 | 3.99e-05 / -0.000746 | -0.000786 [-0.0024, 0.00065] | -5.64e-05 / -7.5e-06 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.8 (pooled) | 0 / 0.018 | 0 / 0.018 | 0 / -0.2 | -0.00103 / -0.000321 | 0.000705 [-0.00266, 0.00432] | 0 / 0.000684 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.7 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | -0.00048 / -0.000144 | 0.000336 [-0.00127, 0.00217] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | oracle_distractor | 0.6 (full_only) | 0.036 / 0.036 | 0.036 / 0 | 0.4 / 0 | -0.000322 / -0.000706 | -0.000384 [-0.00272, 0.00138] | -0.000257 / 0 | 0 / 0 | 0 / 0 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.8 (pooled) | 0 / 0.018 | 0 / 0.036 | 0 / -0.4 | 0.182 / 0.142 | -0.0396 [-0.122, 0.00395] | 0 / 0.00129 | 0.2 / 0.2 | 0.2 / 0.2 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.7 (pooled) | 0.13 / 0.073 | 0.11 / 0.073 | 1.2 / 0.8 | 0.184 / 0.156 | -0.0279 [-0.0828, 0.0011] | -0.0065 / -0.0136 | 0.2 / 0.2 | 0.2 / 0.2 |
| ouro-1.4b | controlled_retrieval | oracle_evidence | 0.6 (full_only) | 0.091 / 0.11 | 0.091 / 0.11 | 1 / 1.4 | 0.194 / 0.157 | -0.0368 [-0.112, 0.00237] | -0.00156 / -0.0258 | 0.2 / 0.2 | 0.2 / 0.2 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.7 (pooled) | 0.005 / 0.015 | 0.005 / 0.013 | -0.05 / 0 | 0.00444 / 0.0104 | 0.00595 [-0.00241, 0.0175] | -1.96e-05 / -0.000785 | 0 / 0.05 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.5 (pooled) | 0.0063 / 0.058 | 0.0063 / 0.11 | -0.05 / 0.05 | 0.00268 / 0.00161 | -0.00107 [-0.0266, 0.0214] | -0.000188 / 0.00701 | 0.05 / 0.2 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.6 (full_only) | 0 / 0.044 | 0 / 0.02 | 0 / -0.05 | 0.00168 / 0.00955 | 0.00787 [-0.0129, 0.0313] | 0 / 0.0029 | 0 / 0.05 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | main_position_hash | 0.3 (full_only) | 0.01 / 0.01 | 0.01 / 0.02 | 0.05 / 0.1 | 0.00574 / 0.000992 | -0.00475 [-0.014, 0.00367] | -0.00169 / 0 | 0.05 / 0.05 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.7 (pooled) | 0.02 / 0 | 0.02 / 0 | 0.2 / 0 | -0.00398 / 0.000814 | 0.00479 [-0.00836, 0.0178] | -0.000808 / 0 | 0 / 0 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.5 (pooled) | 0.025 / 0 | 0.025 / 0.025 | -0.2 / 0.2 | 0.00197 / -0.0125 | -0.0145 [-0.0519, 0.013] | -0.000478 / -0.00267 | 0 / 0.2 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.6 (full_only) | 0 / 0.04 | 0 / 0 | 0 / 0 | 0.00169 / -0.00341 | -0.0051 [-0.0249, 0.00994] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_distractor | 0.3 (full_only) | 0 / 0 | 0 / 0.04 | 0 / 0.2 | 0.00782 / -0.00358 | -0.0114 [-0.0321, 0.00927] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.7 (pooled) | 0.02 / 0 | 0.02 / 0 | -0.2 / 0 | -0.0237 / -0.00769 | 0.016 [-0.00541, 0.0373] | 0.000675 / 0 | 0 / 0.4 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.5 (pooled) | 0.025 / 0.08 | 0.025 / 0.025 | -0.2 / 0.2 | -0.0206 / -0.0191 | 0.00151 [-0.0212, 0.0242] | -0.000675 / -0.000988 | 0.2 / 0.2 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.6 (full_only) | 0 / 0.045 | 0 / 0.02 | 0 / -0.2 | -0.0236 / -0.00786 | 0.0157 [-0.0137, 0.0451] | 0 / 0.000716 | 0 / 0.4 | 0 / 0 |
| ouro-1.4b | hotpotqa_distractor | oracle_evidence | 0.3 (full_only) | 0 / 0.02 | 0 / 0.08 | 0 / 0 | -0.0195 / -0.0189 | 0.000613 [-0.0136, 0.0172] | 0 / 0 | 0.2 / 0.2 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 0.8 (pooled) | 0 / 0.0091 | 0 / 0.018 | 0 / 0 | 0.00151 / 0.000237 | -0.00128 [-0.00367, 0.00104] | 0 / -3.53e-05 | 0 / 0 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 0.7 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | 0.00178 / -2.74e-05 | -0.00181 [-0.0023, -0.00129] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | main_position_hash | 0.6 (full_only) | 0 / 0.0045 | 0 / 0.0045 | 0 / -0.05 | 0.00032 / -0.000631 | -0.000951 [-0.00254, 0.00162] | 0 / -0.000134 | 0 / 0 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 0.8 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | 0.00185 / -0.000879 | -0.00273 [-0.00606, 0.00127] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 0.7 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | 0.000479 / 5.13e-05 | -0.000427 [-0.00127, 0.000649] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | oracle_distractor | 0.6 (full_only) | 0 / 0.018 | 0 / 0 | 0 / 0 | 0.000286 / -0.000285 | -0.000572 [-0.00296, 0.00181] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.8 (pooled) | 0 / 0.073 | 0 / 0.073 | 0 / 0.8 | 0.234 / 0.234 | -0.000162 [-0.0993, 0.102] | 0 / -0.00309 | 0.4 / 0.2 | 0.4 / 0.2 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.7 (pooled) | 0.091 / 0.036 | 0.091 / 0.036 | 1 / 0.4 | 0.239 / 0.239 | 0.000604 [-0.121, 0.124] | 0.0193 / 0.00405 | 0.4 / 0.2 | 0.4 / 0.2 |
| ouro-2.6b | controlled_retrieval | oracle_evidence | 0.6 (full_only) | 0.11 / 0.15 | 0.11 / 0.11 | 1.2 / 1.2 | 0.237 / 0.24 | 0.00239 [-0.127, 0.134] | 0.0293 / 0.00837 | 0.4 / 0.2 | 0.4 / 0.2 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.8 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | 0.00491 / -0.000311 | -0.00523 [-0.0161, 0.00171] | 0 / 0 | 0.05 / 0.05 | 0 / 0.05 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.6 (pooled) | 0.01 / 0 | 0.01 / 0 | 0.05 / 0 | 0.00546 / 0.00693 | 0.00147 [-0.0114, 0.015] | 0.000998 / 0 | 0 / 0.3 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.7 (full_only) | 0.01 / 0.005 | 0.02 / 0 | 0.1 / 0 | 0.00405 / 0.0034 | -0.000647 [-0.0115, 0.00965] | 0.000203 / 0 | 0.05 / 0 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | main_position_hash | 0.4 (full_only) | 0.01 / 0 | 0.01 / 0 | -0.05 / 0 | 0.00422 / 0.00467 | 0.000456 [-0.00727, 0.00988] | -0.00103 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.8 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | 0.00052 / -0.00594 | -0.00646 [-0.01, -0.00289] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.6 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | -0.000785 / 0.0045 | 0.00529 [-0.004, 0.0161] | 0 / 0 | 0 / 0.4 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.7 (full_only) | 0 / 0 | 0 / 0 | 0 / 0 | -0.00298 / -0.00057 | 0.00241 [-0.00831, 0.0122] | 0 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_distractor | 0.4 (full_only) | 0.04 / 0 | 0.04 / 0 | -0.2 / 0 | -0.00148 / -0.00029 | 0.00119 [-0.00315, 0.00552] | -0.00445 / 0 | 0 / 0 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.8 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | -0.00472 / -0.00483 | -0.000109 [-0.0236, 0.0212] | 0 / 0 | 0.2 / 0.2 | 0.2 / 0.2 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.6 (pooled) | 0 / 0 | 0 / 0 | 0 / 0 | 0.000773 / -0.00134 | -0.00211 [-0.0171, 0.00886] | 0 / 0 | 0.2 / 0.4 | 0 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.7 (full_only) | 0 / 0.02 | 0 / 0 | 0 / 0 | -0.00853 / -0.00666 | 0.00187 [-0.0146, 0.0183] | 0 / 0 | 0.2 / 0.2 | 0.2 / 0 |
| ouro-2.6b | hotpotqa_distractor | oracle_evidence | 0.4 (full_only) | 0.04 / 0 | 0.04 / 0 | -0.2 / 0 | -0.00226 / 0.00828 | 0.0105 [-0.0102, 0.0348] | -0.00379 / 0 | 0.2 / 0 | 0 / 0 |
