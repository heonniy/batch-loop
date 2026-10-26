# Huginn R32 — mechanism of FULL-vs-terminal KV robustness (pilot)

Run `results/20261023T125444Z_huginn_r32_attn_mechanism`: 64 prompts (32 GSM8K + 32 MATH-500, Stage-1 stratified selection), frozen greedy FULL prefix, BF16 model, FP32 instrumentation. Values: mean over prompts of per-prompt token means [95% bootstrap CI over prompts]. Loop 32 = terminal.

Controls: hook check max |Δlogit| traced vs untraced = 0 on 52 prompts; NaN counts B/C = 0/0; prelude FULL-vs-LAST error 5.4e-03 (must be 0).

Control deviation: 1 prompt(s) show a nonzero prelude FULL-vs-LAST difference although both rows have identical inputs: `test/precalculus/1199.json` (P=1024). The two rows of the same B=2 BF16 prefill differ numerically at this shape (batch-row kernel noise, not a policy effect), so this prompt's C/boundary numbers include that noise. Sensitivity: loop-32 hidden error / KL with all prompts 0.0275 / 4.962e-04, excluding flagged 0.0273 / 5.037e-04. A/B/D use only the FULL row and are unaffected.

## A. KV geometry on the FULL trajectory (continuation columns; median over columns×heads)

| layer | metric | r=1 | r=2 | r=4 | r=8 | r=16 | r=24 | r=28 | r=31 | r=32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | dK | 0.654 | 0.577 | 0.382 | 0.189 | 0.052 | 0.018 | 0.012 | 0.009 | 0.000 |
| 0 | dV | 1.172 | 1.018 | 0.795 | 0.424 | 0.117 | 0.039 | 0.026 | 0.018 | 0.000 |
| 0 | cosK | 0.796 | 0.832 | 0.929 | 0.983 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0 | cosV | 0.306 | 0.433 | 0.674 | 0.910 | 0.993 | 0.999 | 1.000 | 1.000 | 1.000 |
| 0 | cK | 0.700 | 0.444 | 0.284 | 0.130 | 0.034 | 0.013 | 0.010 | 0.009 | — |
| 0 | cV | 1.146 | 0.835 | 0.593 | 0.281 | 0.073 | 0.026 | 0.020 | 0.018 | — |
| 1 | dK | 0.552 | 0.598 | 0.364 | 0.182 | 0.050 | 0.017 | 0.012 | 0.008 | 0.000 |
| 1 | dV | 1.015 | 0.955 | 0.706 | 0.367 | 0.102 | 0.034 | 0.023 | 0.015 | 0.000 |
| 1 | cosK | 0.842 | 0.831 | 0.935 | 0.984 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | cosV | 0.460 | 0.531 | 0.746 | 0.933 | 0.995 | 0.999 | 1.000 | 1.000 | 1.000 |
| 1 | cK | 0.489 | 0.418 | 0.262 | 0.124 | 0.032 | 0.012 | 0.009 | 0.008 | — |
| 1 | cV | 0.827 | 0.699 | 0.513 | 0.246 | 0.064 | 0.023 | 0.018 | 0.015 | — |
| 2 | dK | 0.578 | 0.626 | 0.374 | 0.184 | 0.052 | 0.017 | 0.012 | 0.008 | 0.000 |
| 2 | dV | 0.911 | 0.895 | 0.628 | 0.324 | 0.090 | 0.030 | 0.020 | 0.013 | 0.000 |
| 2 | cosK | 0.827 | 0.819 | 0.931 | 0.984 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | cosV | 0.552 | 0.602 | 0.799 | 0.948 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | cK | 0.455 | 0.411 | 0.259 | 0.121 | 0.032 | 0.012 | 0.009 | 0.008 | — |
| 2 | cV | 0.689 | 0.617 | 0.433 | 0.207 | 0.055 | 0.020 | 0.015 | 0.013 | — |
| 3 | dK | 0.615 | 0.610 | 0.381 | 0.188 | 0.053 | 0.018 | 0.012 | 0.008 | 0.000 |
| 3 | dV | 0.986 | 0.933 | 0.671 | 0.349 | 0.097 | 0.032 | 0.022 | 0.014 | 0.000 |
| 3 | cosK | 0.807 | 0.826 | 0.929 | 0.983 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | cosV | 0.478 | 0.570 | 0.773 | 0.940 | 0.995 | 0.999 | 1.000 | 1.000 | 1.000 |
| 3 | cK | 0.448 | 0.392 | 0.252 | 0.119 | 0.031 | 0.012 | 0.009 | 0.008 | — |
| 3 | cV | 0.726 | 0.620 | 0.442 | 0.212 | 0.057 | 0.021 | 0.016 | 0.014 | — |

Prompt columns (median dK / dV, layer-mean): r=1: 0.571/0.984, r=8: 0.179/0.340, r=16: 0.055/0.104, r=31: 0.008/0.013. Per-head dispersion: `kv_geometry_per_head.csv`; by position: `kv_geometry_by_position.csv`.

## B. Fixed-query attention counterfactual (FULL query; only past recurrent K,V swapped to loop 32)

| layer | metric | r=1 | r=2 | r=4 | r=8 | r=16 | r=24 | r=28 | r=31 | r=32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | D_swap | 0.934 | 0.875 | 0.617 | 0.292 | 0.0811 | 0.0268 | 0.0177 | 0.0132 | 0 |
| 0 | D_konly | 0.665 | 0.447 | 0.299 | 0.136 | 0.0377 | 0.0131 | 0.00891 | 0.00741 | 0 |
| 0 | D_vonly | 0.845 | 0.77 | 0.564 | 0.264 | 0.0718 | 0.0232 | 0.0151 | 0.0108 | 0 |
| 0 | tv_mean | 0.55 | 0.428 | 0.267 | 0.118 | 0.033 | 0.0114 | 0.00788 | 0.00592 | 0 |
| 0 | cos_swap | 0.479 | 0.552 | 0.801 | 0.956 | 0.996 | 1 | 1 | 1 | 1 |
| 0 | attn_over_resid | 0.871 | 0.55 | 0.463 | 0.415 | 0.399 | 0.398 | 0.398 | 0.398 | 0.398 |
| 0 | Dswap_over_resid | 0.814 | 0.484 | 0.286 | 0.122 | 0.0323 | 0.0106 | 0.007 | 0.00524 | 0 |
| 1 | D_swap | 0.792 | 0.822 | 0.547 | 0.274 | 0.0789 | 0.0261 | 0.0169 | 0.0125 | 0 |
| 1 | D_konly | 0.49 | 0.388 | 0.271 | 0.125 | 0.0352 | 0.0119 | 0.00813 | 0.00655 | 0 |
| 1 | D_vonly | 0.758 | 0.786 | 0.518 | 0.249 | 0.0711 | 0.0232 | 0.0148 | 0.0106 | 0 |
| 1 | tv_mean | 0.479 | 0.425 | 0.272 | 0.128 | 0.037 | 0.0126 | 0.00865 | 0.00654 | 0 |
| 1 | cos_swap | 0.624 | 0.627 | 0.84 | 0.962 | 0.997 | 1 | 1 | 1 | 1 |
| 1 | attn_over_resid | 0.409 | 0.358 | 0.315 | 0.29 | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 |
| 1 | Dswap_over_resid | 0.323 | 0.294 | 0.173 | 0.0798 | 0.0228 | 0.00754 | 0.00487 | 0.00361 | 0 |
| 2 | D_swap | 0.788 | 0.772 | 0.56 | 0.271 | 0.0802 | 0.0267 | 0.0171 | 0.0119 | 0 |
| 2 | D_konly | 0.595 | 0.518 | 0.383 | 0.163 | 0.0471 | 0.016 | 0.0106 | 0.00801 | 0 |
| 2 | D_vonly | 0.679 | 0.697 | 0.458 | 0.223 | 0.0655 | 0.0213 | 0.0134 | 0.00869 | 0 |
| 2 | tv_mean | 0.318 | 0.454 | 0.26 | 0.122 | 0.0365 | 0.0126 | 0.00845 | 0.00608 | 0 |
| 2 | cos_swap | 0.617 | 0.653 | 0.821 | 0.963 | 0.996 | 1 | 1 | 1 | 1 |
| 2 | attn_over_resid | 0.224 | 0.287 | 0.248 | 0.229 | 0.225 | 0.225 | 0.225 | 0.225 | 0.225 |
| 2 | Dswap_over_resid | 0.178 | 0.221 | 0.142 | 0.0625 | 0.0181 | 0.00599 | 0.00381 | 0.00263 | 0 |
| 3 | D_swap | 0.779 | 0.736 | 0.505 | 0.252 | 0.0723 | 0.0237 | 0.0152 | 0.0104 | 0 |
| 3 | D_konly | 0.505 | 0.437 | 0.303 | 0.141 | 0.0403 | 0.0138 | 0.00925 | 0.00708 | 0 |
| 3 | D_vonly | 0.692 | 0.744 | 0.465 | 0.215 | 0.0608 | 0.0195 | 0.0122 | 0.00761 | 0 |
| 3 | tv_mean | 0.459 | 0.465 | 0.289 | 0.139 | 0.0418 | 0.0145 | 0.0099 | 0.00709 | 0 |
| 3 | cos_swap | 0.623 | 0.695 | 0.864 | 0.968 | 0.997 | 1 | 1 | 1 | 1 |
| 3 | attn_over_resid | 0.274 | 0.253 | 0.249 | 0.231 | 0.228 | 0.228 | 0.228 | 0.228 | 0.228 |
| 3 | Dswap_over_resid | 0.214 | 0.186 | 0.127 | 0.0584 | 0.0165 | 0.0054 | 0.00345 | 0.00235 | 0 |

Controls (layer mean, r=16): BF16 engine vs FP32 direct attention D_precision 0.00121 [0.0012, 0.00122]; reversed-layout D_layout 2.35e-07 [2.34e-07, 2.37e-07]; r=32 swap is an exact no-op. D_* = ||A_x − A_ref|| / ||A_ref|| after the output projection; Dswap_over_resid = ||A_swap − A_ref|| / ||block input||.

## C. Actual FULL vs LAST sequential execution (same prefix and latent): observed differences

| loop r | err_in | err_out (layer 3) | rel ΔA | cos(e_in, ΔA) | rel ΔM | cos(e_mid, ΔM) | attn/resid | mlp/resid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.71e-05 | 0.787 | 0.908 | +0.028 | 0.731 | -0.069 | 0.445 | 0.507 |
| 2 | 0.772 | 0.936 | 0.908 | +0.016 | 0.926 | -0.103 | 0.362 | 0.428 |
| 4 | 0.814 | 0.805 | 0.671 | +0.015 | 0.771 | -0.058 | 0.319 | 0.394 |
| 8 | 0.482 | 0.456 | 0.35 | +0.023 | 0.428 | -0.036 | 0.291 | 0.386 |
| 16 | 0.143 | 0.135 | 0.104 | +0.028 | 0.124 | -0.029 | 0.285 | 0.384 |
| 24 | 0.0506 | 0.0494 | 0.0381 | +0.026 | 0.0453 | -0.033 | 0.285 | 0.384 |
| 28 | 0.0347 | 0.035 | 0.0272 | +0.018 | 0.0322 | -0.041 | 0.285 | 0.384 |
| 31 | 0.0282 | 0.029 | 0.0223 | +0.012 | 0.0268 | -0.049 | 0.285 | 0.384 |
| 32 | 0.0267 | 0.0275 | 0.0207 | +0.007 | 0.0255 | -0.052 | 0.285 | 0.384 |

(err_in at layer 0 of loop r = error entering loop r; ΔA, ΔM, cosines and ratios are layer means; see CSV per layer.)

Boundaries (decode tokens): coda0 0.0276 [0.0251, 0.0311]; coda1 0.0279 [0.0254, 0.0316]; final_lnf 0.028 [0.0255, 0.0313]; post_core_lnf 0.0277 [0.0243, 0.0324]; prelude0 0 [0, 0]; prelude1 5.49e-06 [0, 1.65e-05]; kl 0.000496 [0.000349, 0.00071]; argmax_same 0.995 [0.993, 0.997].

- gsm8k: hidden error at loop end r=1 0.796, r=16 0.149, r=32 0.0264; final logits KL 7.06e-04.
- math500: hidden error at loop end r=1 0.778, r=16 0.121, r=32 0.0286; final logits KL 2.86e-04.

## D. Single-call intervention (one attention call reads terminal history; rest FULL)

| intervened loop | imm. attn Δ | post-layer Δh | post-R32 Δh | final ln_f Δ | KL(ref‖int) | argmax same | prompts |
|---:|---|---|---|---|---|---:|---:|
| 1 | 0.819 [0.813, 0.826] | 0.335 [0.331, 0.34] | 0.0139 [0.0137, 0.0141] | 0.0168 [0.0166, 0.0171] | 0.000173 [0.000139, 0.000208] | 0.9961 | 64 |
| 2 | 0.798 [0.791, 0.805] | 0.274 [0.269, 0.279] | 0.0139 [0.0137, 0.0141] | 0.0168 [0.0166, 0.0171] | 0.000182 [0.000143, 0.000224] | 0.9966 | 64 |
| 4 | 0.556 [0.546, 0.567] | 0.172 [0.165, 0.179] | 0.0139 [0.0137, 0.0141] | 0.0168 [0.0166, 0.0171] | 0.000166 [0.000134, 0.000202] | 0.9971 | 64 |
| 8 | 0.271 [0.263, 0.28] | 0.0758 [0.072, 0.0794] | 0.0139 [0.0137, 0.0143] | 0.0169 [0.0166, 0.0173] | 0.000179 [0.000141, 0.000221] | 0.9976 | 64 |
| 16 | 0.0778 [0.073, 0.0829] | 0.0216 [0.0201, 0.023] | 0.0138 [0.0137, 0.014] | 0.0168 [0.0166, 0.017] | 0.000171 [0.000141, 0.000207] | 0.9966 | 64 |
| 24 | 0.0261 [0.0246, 0.0276] | 0.00829 [0.00789, 0.00866] | 0.0138 [0.0136, 0.0139] | 0.0167 [0.0165, 0.0169] | 0.000173 [0.000138, 0.000212] | 0.9971 | 64 |
| 31 | 0.0119 [0.0112, 0.0128] | 0.00498 [0.00479, 0.00519] | 0.0102 [0.01, 0.0103] | 0.0137 [0.0135, 0.0139] | 0.000125 [9.79e-05, 0.000155] | 0.9985 | 64 |
| 32 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 1.0000 | 64 |

Per layer: `intervention_summary.csv`; per token: `intervention_effects.csv`.

## Precision control: same C and D in FP32 (short prompts: `gsm8k-test-0722`, `gsm8k-test-0752`, `gsm8k-test-0985`, `gsm8k-test-1119`, `test/algebra/109.json`, `test/geometry/477.json`, `test/intermediate_algebra/860.json`, `test/number_theory/255.json`)

| quantity | FP32 | BF16 (same prompts) |
|---|---:|---:|
| D, intervene loop 1: post_layer | 0.328 | 0.328 |
| D, intervene loop 1: post_R | 0.000134 | 0.0141 |
| D, intervene loop 1: kl_ref_int | 2.46e-08 | 0.000272 |
| D, intervene loop 2: post_layer | 0.266 | 0.266 |
| D, intervene loop 2: post_R | 0.000167 | 0.014 |
| D, intervene loop 2: kl_ref_int | 2.42e-08 | 0.000321 |
| D, intervene loop 4: post_layer | 0.156 | 0.156 |
| D, intervene loop 4: post_R | 0.000203 | 0.014 |
| D, intervene loop 4: kl_ref_int | 9.86e-08 | 0.000299 |
| D, intervene loop 8: post_layer | 0.0709 | 0.071 |
| D, intervene loop 8: post_R | 0.000244 | 0.014 |
| D, intervene loop 8: kl_ref_int | 4.51e-08 | 0.000285 |
| D, intervene loop 16: post_layer | 0.0227 | 0.0234 |
| D, intervene loop 16: post_R | 0.000497 | 0.0141 |
| D, intervene loop 16: kl_ref_int | 3.06e-07 | 0.000242 |
| D, intervene loop 24: post_layer | 0.00731 | 0.0091 |
| D, intervene loop 24: post_R | 0.000988 | 0.0139 |
| D, intervene loop 24: kl_ref_int | 6.32e-07 | 0.000275 |
| D, intervene loop 31: post_layer | 0.0016 | 0.00492 |
| D, intervene loop 31: post_R | 0.00148 | 0.0101 |
| D, intervene loop 31: kl_ref_int | 1.07e-06 | 0.000157 |
| C, FULL-vs-LAST hidden error at end of loop 1 | 0.78 | 0.779 |
| C, FULL-vs-LAST hidden error at end of loop 4 | 0.793 | 0.792 |
| C, FULL-vs-LAST hidden error at end of loop 8 | 0.448 | 0.447 |
| C, FULL-vs-LAST hidden error at end of loop 16 | 0.143 | 0.144 |
| C, FULL-vs-LAST hidden error at end of loop 24 | 0.0464 | 0.0509 |
| C, FULL-vs-LAST hidden error at end of loop 32 | 0.0144 | 0.024 |
| C, FULL-vs-LAST logits KL (decode mean) | 0.000139 | 0.000405 |

## Negative / outlier cases

Largest FULL-vs-LAST mean KL (C): gsm8k-test-0383 KL 5.50e-03, argmax same 0.9935, n=153, err_out r32 0.0473; gsm8k-test-0584 KL 2.12e-03, argmax same 0.9762, n=126, err_out r32 0.0431; gsm8k-test-0924 KL 2.08e-03, argmax same 0.9873, n=79, err_out r32 0.0317; gsm8k-test-0790 KL 1.39e-03, argmax same 0.9839, n=62, err_out r32 0.0278; test/algebra/849.json KL 1.17e-03, argmax same 1.0000, n=69, err_out r32 0.0214

Prompts whose loop-end hidden error grows from r=16 to r=32: 0/64

Figures: `fig_A_B_by_loop.png`, `fig_C_D_error_propagation.png`


## Interpretation against the plan's rubric (written after inspecting the results above)

1. **Raw KV redundancy — only for late loops.** Against the terminal loop, early-loop K/V are far apart
   (continuation median dK ≈ 0.55–0.65, dV ≈ 0.91–1.17, cosV ≈ 0.31–0.55 at r = 1). They become near-identical only
   from r ≈ 16 (dK ≈ 0.05, dV ≈ 0.1, cos ≥ 0.99) and r ≥ 24 (dK, dV ≤ 0.04). All four layers and the prompt
   columns behave the same way.
2. **Attention readout insensitivity — not supported.** At a fixed FULL query, swapping past K/V to loop 32
   changes the post-projection attention output about as much as V itself differs (D_swap ≈ 0.78–0.93 at
   r = 1, ≈ 0.07–0.08 at r = 16, ≈ 0.010–0.013 at r = 31). The value swap dominates (D_vonly > D_konly in every layer at
   r ≥ 4), and the attention weights move with it (TV 0.32–0.55 at r = 1). The readout passes the KV difference
   through rather than damping it.
3. **Recurrent recovery — strongly supported; the main mechanism for early loops.** In the actual LAST run the
   loop-end hidden error peaks near 0.8–0.9 at r = 2–4. It then shrinks monotonically to ≈ 0.027 (BF16) at
   r = 32, with no prompt growing between r = 16 and r = 32 (0/64). A single early perturbation
   (post-layer Δh ≈ 0.33) is contracted by the remaining loops to 1.3e-4 in FP32, about 2500×. The observed
   update/error alignments are weak: cos(e, ΔA) ≈ +0.01–0.03 and cos(e, ΔM) ≈ −0.03 to −0.10. So the contraction
   is not a residual update that cancels the error directly. It is a property of the whole recurrence
   (re-injected embedding plus normalised sandwich blocks). This is observational and has not been isolated
   further.
4. **Coda / LM-head insensitivity — not a separate effect.** The relative hidden error passes the coda
   essentially unchanged (≈ 0.028 → 0.028). The small output KL (≈ 5e-4 nats; argmax agreement 99.5%) comes
   from that small residual error, not from coda damping.
5. **BF16 floor.** In BF16, any single perturbation, large or tiny, ends at ≈ 1.4% hidden error at r = 32.
   Late interventions grow up to this level and early ones shrink down to it. In FP32 the same interventions
   end two orders of magnitude lower. LAST's persistent effect in FP32 (≈ 1.4% at r = 32, KL 1.4e-4) is the
   same size as this BF16 rounding floor. This matches Stage 2, where LAST's top-1 change rate was close to the
   latent-noise floor.

**Mixed mechanism.** The answer has two parts:
- Early-loop terminal substitution is a large perturbation, both in KV space and in the fixed-query attention
  output. The remaining recurrence erases it.
- Late-loop substitution is small because the per-token K/V have already converged.

The residual effect at the output is about the BF16 numerical floor of the 32-loop recurrence. These are
diagnostics on a frozen FULL prefix. They make no performance or optimisation claim.

Scope notes:
- The pilot is 32 GSM8K + 32 MATH-500 prompts (Stage-1 selection). D uses 8 tokens per prompt × 8 loops ×
  4 layers.
- The FP32 control uses 8 short prompts.
- One prompt (P = 1024) has batch-row prefill noise; see the control-deviation note at the top.
- The repository root `AGENTS.md` blocks GPU use for the Phase-04R rank-stability task. This study is a
  separate kv_policy task that the user explicitly requested, and its plan asks for the H200 ×8 pilot.

## Provenance

- Jobs ran from uncommitted working-tree code on top of `3498415`, which is the `source_sha` in the raw
  records and in `mechanism/manifest.json` (that file also lists the dirty files). That code is committed
  unchanged as `0d0bf7c`. Test log: `mechanism/logs/pytest_cpu.log` (129 passed, 1 skipped).
- Per-step arrays `mechanism/raw/*.npz` (64 files, 468 MB) are NOT committed. Their sha256 are in
  `mechanism/raw_npz_sha256.txt`, and the files stay on the H200 host under this run directory. Every CSV and
  table here is derived from them plus the committed `raw/*.json`.
- FP32 control: `mechanism/float32_control/*.json`. Analysis env: kvp_tf453 clone with matplotlib (figures only).
