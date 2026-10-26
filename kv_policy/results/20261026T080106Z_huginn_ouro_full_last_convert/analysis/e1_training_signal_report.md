# E1 — training-depth signal characterisation under FULL

Run `results/20261026T080106Z_huginn_ouro_full_last_convert`; 1200 item records. Gold teacher forcing on the frozen GSM8K-200 / MATH500-200 targets, B=1, BF16. Ouro per-step predictions are its native exit heads (lm_head(norm h_r)); Huginn depth-r predictions use the coda readout of the loop-r latent (its own adaptive-compute path). Exit probabilities are model outputs, not evidence of training exposure.

## A. Gold next-token loss by recurrent depth (token-weighted; normalised improvement = (NLL_1 − NLL_r)/(NLL_1 − NLL_R))

- **huginn-0125 gsm8k** (200 problems, 17106 tokens): r=1: NLL 3.019, top-1 0.356, norm.impr 0.00; r=2: NLL 2.042, top-1 0.503, norm.impr 0.46; r=4: NLL 1.359, top-1 0.646, norm.impr 0.78; r=8: NLL 0.940, top-1 0.760, norm.impr 0.98; r=16: NLL 0.884, top-1 0.777, norm.impr 1.01; r=24: NLL 0.898, top-1 0.774, norm.impr 1.00; r=28: NLL 0.899, top-1 0.773, norm.impr 1.00; r=31: NLL 0.900, top-1 0.772, norm.impr 1.00; r=32: NLL 0.899, top-1 0.772, norm.impr 1.00
  - lowest NLL at r=12 (0.878); final-step NLL 0.899 — final step is NOT the best (non-monotone depth curve)
- **huginn-0125 math500** (200 problems, 47776 tokens): r=1: NLL 2.145, top-1 0.481, norm.impr 0.00; r=2: NLL 1.476, top-1 0.621, norm.impr 0.44; r=4: NLL 0.998, top-1 0.724, norm.impr 0.76; r=8: NLL 0.719, top-1 0.798, norm.impr 0.94; r=16: NLL 0.637, top-1 0.821, norm.impr 1.00; r=24: NLL 0.632, top-1 0.822, norm.impr 1.00; r=28: NLL 0.632, top-1 0.822, norm.impr 1.00; r=31: NLL 0.632, top-1 0.822, norm.impr 1.00; r=32: NLL 0.632, top-1 0.822, norm.impr 1.00
  - lowest NLL at r=28 (0.632); final-step NLL 0.632 — final step is NOT the best (non-monotone depth curve)
- **ouro-1.4b gsm8k** (200 problems, 19000 tokens): r=1: NLL 1.217, top-1 0.760, norm.impr 0.00; r=2: NLL 0.788, top-1 0.801, norm.impr 0.88; r=3: NLL 0.735, top-1 0.807, norm.impr 0.99; r=4: NLL 0.732, top-1 0.809, norm.impr 1.00
  - lowest NLL at r=4 (0.732); final-step NLL 0.732
- **ouro-1.4b math500** (200 problems, 50383 tokens): r=1: NLL 0.980, top-1 0.779, norm.impr 0.00; r=2: NLL 0.627, top-1 0.832, norm.impr 0.81; r=3: NLL 0.561, top-1 0.844, norm.impr 0.96; r=4: NLL 0.544, top-1 0.848, norm.impr 1.00
  - lowest NLL at r=4 (0.544); final-step NLL 0.544
- **ouro-2.6b gsm8k** (200 problems, 19000 tokens): r=1: NLL 0.935, top-1 0.786, norm.impr 0.00; r=2: NLL 0.726, top-1 0.815, norm.impr 0.87; r=3: NLL 0.697, top-1 0.818, norm.impr 1.00; r=4: NLL 0.696, top-1 0.816, norm.impr 1.00
  - lowest NLL at r=4 (0.696); final-step NLL 0.696
- **ouro-2.6b math500** (200 problems, 50383 tokens): r=1: NLL 0.706, top-1 0.817, norm.impr 0.00; r=2: NLL 0.506, top-1 0.858, norm.impr 0.83; r=3: NLL 0.475, top-1 0.863, norm.impr 0.96; r=4: NLL 0.465, top-1 0.865, norm.impr 1.00
  - lowest NLL at r=4 (0.465); final-step NLL 0.465

## B. Ouro ACT exit distribution (official algorithm: sigmoid gates, remaining mass on the last step)

- ouro-1.4b gsm8k prompt: mean pdf p1=0.067, p2=0.145, p3=0.301, p4=0.486; E[exit] 3.21 ± 0.47; pdf entropy 0.97 nats; mode step 1: 0.01, 2: 0.01, 3: 0.12, 4: 0.86
- ouro-1.4b gsm8k target: mean pdf p1=0.098, p2=0.201, p3=0.323, p4=0.379; E[exit] 2.98 ± 0.38; pdf entropy 1.16 nats; mode step 1: 0.01, 2: 0.03, 3: 0.17, 4: 0.80; within-problem Spearman(E[exit], final NLL) +0.817 [+0.808, +0.826]
- ouro-1.4b math500 prompt: mean pdf p1=0.083, p2=0.167, p3=0.292, p4=0.457; E[exit] 3.12 ± 0.47; pdf entropy 1.04 nats; mode step 1: 0.02, 2: 0.02, 3: 0.11, 4: 0.84
- ouro-1.4b math500 target: mean pdf p1=0.099, p2=0.190, p3=0.304, p4=0.407; E[exit] 3.02 ± 0.43; pdf entropy 1.13 nats; mode step 1: 0.03, 2: 0.03, 3: 0.09, 4: 0.85; within-problem Spearman(E[exit], final NLL) +0.787 [+0.780, +0.793]
- ouro-2.6b gsm8k prompt: mean pdf p1=0.075, p2=0.160, p3=0.294, p4=0.471; E[exit] 3.16 ± 0.48; pdf entropy 1.01 nats; mode step 1: 0.01, 2: 0.00, 3: 0.14, 4: 0.84
- ouro-2.6b gsm8k target: mean pdf p1=0.114, p2=0.224, p3=0.316, p4=0.347; E[exit] 2.89 ± 0.33; pdf entropy 1.22 nats; mode step 1: 0.01, 2: 0.01, 3: 0.20, 4: 0.78; within-problem Spearman(E[exit], final NLL) +0.812 [+0.803, +0.821]
- ouro-2.6b math500 prompt: mean pdf p1=0.093, p2=0.200, p3=0.316, p4=0.392; E[exit] 3.01 ± 0.41; pdf entropy 1.13 nats; mode step 1: 0.01, 2: 0.01, 3: 0.26, 4: 0.72
- ouro-2.6b math500 target: mean pdf p1=0.116, p2=0.218, p3=0.310, p4=0.356; E[exit] 2.91 ± 0.37; pdf entropy 1.21 nats; mode step 1: 0.00, 2: 0.02, 3: 0.26, 4: 0.71; within-problem Spearman(E[exit], final NLL) +0.789 [+0.782, +0.796]

Inference with the checkpoint default `early_exit_threshold=1.0` exits at the last step (cumulative mass reaches 1 only there); the fixed-R4 evaluation therefore matches default inference.

## C. Cross-depth K/V geometry under FULL (layer-mean of ||X^(r) − X^(R)|| / ||X^(R)|| and cos; within-model only)

- huginn-0125 prompt K_dist vs terminal (layer mean): r=1: 0.656, r=2: 0.592, r=4: 0.364, r=8: 0.183, r=16: 0.059, r=24: 0.020, r=31: 0.009, r=32: 0.000
- huginn-0125 prompt V_dist vs terminal (layer mean): r=1: 1.049, r=2: 0.936, r=4: 0.667, r=8: 0.346, r=16: 0.111, r=24: 0.038, r=31: 0.015, r=32: 0.000
- huginn-0125 prompt K_cos vs terminal (layer mean): r=1: 0.822, r=2: 0.826, r=4: 0.933, r=8: 0.982, r=16: 0.998, r=24: 1.000, r=31: 1.000, r=32: 1.000
- huginn-0125 prompt V_cos vs terminal (layer mean): r=1: 0.484, r=2: 0.572, r=4: 0.773, r=8: 0.935, r=16: 0.993, r=24: 0.999, r=31: 1.000, r=32: 1.000
- huginn-0125 cont K_dist vs terminal (layer mean): r=1: 0.666, r=2: 0.619, r=4: 0.401, r=8: 0.204, r=16: 0.060, r=24: 0.021, r=31: 0.009, r=32: 0.000
- huginn-0125 cont V_dist vs terminal (layer mean): r=1: 1.077, r=2: 0.984, r=4: 0.735, r=8: 0.391, r=16: 0.117, r=24: 0.040, r=31: 0.017, r=32: 0.000
- huginn-0125 cont K_cos vs terminal (layer mean): r=1: 0.812, r=2: 0.808, r=4: 0.921, r=8: 0.978, r=16: 0.998, r=24: 1.000, r=31: 1.000, r=32: 1.000
- huginn-0125 cont V_cos vs terminal (layer mean): r=1: 0.456, r=2: 0.529, r=4: 0.731, r=8: 0.918, r=16: 0.992, r=24: 0.999, r=31: 1.000, r=32: 1.000
- ouro-1.4b prompt K_dist vs terminal (layer mean): r=1: 0.493, r=2: 0.259, r=3: 0.146, r=4: 0.000
- ouro-1.4b prompt V_dist vs terminal (layer mean): r=1: 0.889, r=2: 0.535, r=3: 0.313, r=4: 0.000
- ouro-1.4b prompt K_cos vs terminal (layer mean): r=1: 0.840, r=2: 0.960, r=3: 0.986, r=4: 1.000
- ouro-1.4b prompt V_cos vs terminal (layer mean): r=1: 0.625, r=2: 0.843, r=3: 0.940, r=4: 1.000
- ouro-1.4b cont K_dist vs terminal (layer mean): r=1: 0.489, r=2: 0.266, r=3: 0.147, r=4: 0.000
- ouro-1.4b cont V_dist vs terminal (layer mean): r=1: 0.914, r=2: 0.562, r=3: 0.324, r=4: 0.000
- ouro-1.4b cont K_cos vs terminal (layer mean): r=1: 0.845, r=2: 0.958, r=3: 0.986, r=4: 1.000
- ouro-1.4b cont V_cos vs terminal (layer mean): r=1: 0.605, r=2: 0.826, r=3: 0.935, r=4: 1.000
- ouro-2.6b prompt K_dist vs terminal (layer mean): r=1: 0.443, r=2: 0.232, r=3: 0.125, r=4: 0.000
- ouro-2.6b prompt V_dist vs terminal (layer mean): r=1: 0.851, r=2: 0.478, r=3: 0.269, r=4: 0.000
- ouro-2.6b prompt K_cos vs terminal (layer mean): r=1: 0.871, r=2: 0.965, r=3: 0.989, r=4: 1.000
- ouro-2.6b prompt V_cos vs terminal (layer mean): r=1: 0.655, r=2: 0.865, r=3: 0.953, r=4: 1.000
- ouro-2.6b cont K_dist vs terminal (layer mean): r=1: 0.450, r=2: 0.250, r=3: 0.134, r=4: 0.000
- ouro-2.6b cont V_dist vs terminal (layer mean): r=1: 0.901, r=2: 0.530, r=3: 0.297, r=4: 0.000
- ouro-2.6b cont K_cos vs terminal (layer mean): r=1: 0.869, r=2: 0.959, r=3: 0.987, r=4: 1.000
- ouro-2.6b cont V_cos vs terminal (layer mean): r=1: 0.613, r=2: 0.834, r=3: 0.940, r=4: 1.000

## D. Ouro: effect of terminal-substituting ONE historical token's K/V vs that token's features

Effect = mean over the next 32 gold predictions after the cut of KL(p_FULL ‖ p_FULL−i), inside an otherwise FULL cache. Source i = cache column of token x_i (prompt or continuation, always before the window). Spearman correlations are computed within each prompt and averaged; [95% bootstrap CI over prompts].

- **ouro-1.4b** (62 prompts; negative control duplicate-FULL row effect max 0.0e+00; positive reference all-terminal effect median 2.61; single-token effect median 0.000564, max 0.179)
  - Spearman(effect, p_step1) = +0.141 [+0.086, +0.195] (n=62)
  - Spearman(effect, p_last) = -0.116 [-0.170, -0.059] (n=62)
  - Spearman(effect, expected_exit) = -0.125 [-0.179, -0.072] (n=62)
  - Spearman(effect, pdf_entropy) = +0.122 [+0.068, +0.176] (n=62)
  - Spearman(effect, nll_final) = -0.089 [-0.145, -0.038] (n=62)
  - Spearman(effect, nll_improve_1_to_R) = -0.054 [-0.107, -0.002] (n=62)
  - Spearman(effect, kv_dist_K) = +0.162 [+0.121, +0.202] (n=62)
  - Spearman(effect, kv_dist_V) = +0.166 [+0.125, +0.207] (n=62)
  - Spearman(effect, recency) = +0.293 [+0.238, +0.354] (n=62)
  - Spearman(effect, in_prompt) = -0.259 [-0.316, -0.192] (n=62)
  - negative control: Spearman(effect shuffled within prompt, expected_exit) = -0.013 [-0.060, +0.034]
- **ouro-2.6b** (62 prompts; negative control duplicate-FULL row effect max 0.0e+00; positive reference all-terminal effect median 0.464; single-token effect median 0.000386, max 0.0603)
  - Spearman(effect, p_step1) = +0.070 [+0.020, +0.113] (n=62)
  - Spearman(effect, p_last) = -0.088 [-0.132, -0.035] (n=62)
  - Spearman(effect, expected_exit) = -0.085 [-0.132, -0.037] (n=62)
  - Spearman(effect, pdf_entropy) = +0.066 [+0.020, +0.108] (n=62)
  - Spearman(effect, nll_final) = -0.003 [-0.056, +0.051] (n=62)
  - Spearman(effect, nll_improve_1_to_R) = -0.039 [-0.085, +0.010] (n=62)
  - Spearman(effect, kv_dist_K) = +0.114 [+0.062, +0.164] (n=62)
  - Spearman(effect, kv_dist_V) = +0.153 [+0.104, +0.204] (n=62)
  - Spearman(effect, recency) = +0.516 [+0.461, +0.564] (n=62)
  - Spearman(effect, in_prompt) = -0.446 [-0.492, -0.396] (n=62)
  - negative control: Spearman(effect shuffled within prompt, expected_exit) = +0.006 [-0.035, +0.049]

Correlations are associations within these checkpoints; they do not show that ACT training causes the sensitivity.

