# LoopZip v6 — Stage A patterns and Stage C analyses

Descriptive pilot (4 prompts per model × workload in Stage B/C). Bootstrap over prompts. Full-sized reference KV + read masks: no HBM/latency claims.

## Stage A — descriptive trajectory patterns (dev-frozen thresholds, holdout check)

Settle depth = first loop after which F_t(r) < 0.1 through R (a-priori); a token with ≥2 update rises (U(r+1) > 1.1·U(r)) is labeled nonmonotonic. Early/mid/late cut points are the DEV tertiles of relative settle depth per model; shares are then compared on HOLDOUT prompts. If the dev tertiles coincide (Ouro, R=4), tokens are labeled by their exact settle loop instead.

| model | label | dev share | holdout share |
|---|---|---|---|
| huginn-0125 | early_settling | 0.393 | 0.333 |
| huginn-0125 | late_settling | 0.238 | 0.294 |
| huginn-0125 | mid_settling | 0.345 | 0.353 |
| huginn-0125 | nonmonotonic | 0.0235 | 0.0198 |
| ouro-2.6b | nonmonotonic | 0.000924 | 0.000903 |
| ouro-2.6b | settles_at_r/R=0.50 | 0.00406 | 0.00361 |
| ouro-2.6b | settles_at_r/R=0.75 | 0.225 | 0.226 |
| ouro-2.6b | settles_at_r/R=1.00 | 0.77 | 0.77 |

| model | dev cut points (settle r/R) | Spearman token-ID settle depth dev↔holdout (n IDs) | var share: prompt | role | token ID (in-sample) |
|---|---|---|---|---|---|
| huginn-0125 | [0.438, 0.531] | 0.587 (261) | 0.101 | 0.0798 | 0.453 |
| ouro-2.6b | [1.0, 1.0] | 0.478 (258) | 0.134 | 0.0284 | 0.361 |

Generated vs prompt and role-wise label shares are in stagea_label_share_by_role.csv.
Ouro per-loop readout is reported in the controls section below (Huginn: not model-native).

## Stage B audit

16 Stage-B prompts; census complete for all: True; eligible sources 2105, excluded 'other' tokens 377. Worst G0 parity value over prompts: 0.

## Stage C — matched-source comparisons (same prompt, same role, |Δposition| ≤ 4, attention-received percentile within 0.10)

|log10 KL ratio| for matched pairs vs random same-role pairs from the same prompt; per-prompt medians, then mean [95% CI] over prompts.

| model | attention stratum | matched pairs | prompts | matched median | random same-role median | share of matched pairs >10× apart |
|---|---|---|---|---|---|---|
| huginn-0125 | high | 277 | 8 | 0.325 [0.227,0.425] | 0.359 | 0.152 |
| huginn-0125 | low | 261 | 8 | 0.304 [0.163,0.516] | 0.359 | 0.157 |
| huginn-0125 | mid | 271 | 8 | 0.326 [0.207,0.472] | 0.359 | 0.159 |
| ouro-2.6b | high | 299 | 8 | 0.503 [0.347,0.669] | 0.500 | 0.241 |
| ouro-2.6b | low | 306 | 8 | 0.441 [0.273,0.579] | 0.500 | 0.232 |
| ouro-2.6b | mid | 277 | 8 | 0.540 [0.317,0.751] | 0.500 | 0.249 |

Interpretation guard: matched sources are adjacent tokens with similar received attention; residual dispersion is NOT by itself recurrence-specific (value content, token identity, and local syntax are uncontrolled).

## Stage C — role comparison support

Position overlap between Content and Instruction/Question sources: 0 tokens in every prompt (Instruction/Question always follow Context), so there is NO common position/age support and no role-causality claim is made. Within the common range of received attention only (descriptive):

| model | prompts | content n | instr/question n | mean KL content | mean KL instr/question |
|---|---|---|---|---|---|
| huginn-0125 | 8 | 335 | 50 | 0.0109 | 0.00242 |
| ouro-2.6b | 8 | 27 | 19 | 0.00609 | 0.0141 |

## Stage C — descriptive leave-one-prompt-out ranking of source harm (NOT Stage D)

Ridge on log10 mean KL with clean-prefill features; Spearman within the held-out prompt. The 16 census prompts are exhausted for case/threshold choices, so this is not an out-of-sample claim.

| model | features | mean within-prompt Spearman [95% CI over prompts] |
|---|---|---|
| huginn-0125 | attention/position/role | 0.263 [0.158,0.374] |
| huginn-0125 | lexical | 0.174 [0.103,0.242] |
| huginn-0125 | control (attn+lexical) | 0.247 [0.158,0.353] |
| huginn-0125 | recurrence | 0.211 [0.139,0.282] |
| huginn-0125 | control+recurrence | 0.258 [0.179,0.351] |
| huginn-0125 | LIFT: control+recurrence − control | 0.011 [-0.039,0.060] |
| ouro-2.6b | attention/position/role | 0.069 [0.003,0.139] |
| ouro-2.6b | lexical | 0.264 [0.171,0.344] |
| ouro-2.6b | control (attn+lexical) | 0.208 [0.120,0.297] |
| ouro-2.6b | recurrence | 0.299 [0.201,0.389] |
| ouro-2.6b | control+recurrence | 0.279 [0.147,0.390] |
| ouro-2.6b | LIFT: control+recurrence − control | 0.071 [0.016,0.117] |

## Stage C — example selection protocol (dev 2 prompts / holdout 2 prompts per model×workload)

Largest/median/smallest sources are picked on DEV only (stagec_dev_examples.csv). Holdout confirms whether the heavy-tail and discordant-pair statistics carry over:

| model | split | max/median KL | top-5% KL share | attention pct of largest source | matched pairs >10× |
|---|---|---|---|---|---|
| huginn-0125 | dev | 248.3 | 0.534 | 0.41 | 0.168 |
| huginn-0125 | holdout | 65.6 | 0.406 | 0.41 | 0.143 |
| ouro-2.6b | dev | 111.9 | 0.435 | 0.33 | 0.21 |
| ouro-2.6b | holdout | 53.7 | 0.341 | 0.40 | 0.268 |

## Stage C — mechanism controls (clean history, small labeled diagnostics)

controls run complete=True (164/164 jobs).

Per case (source × t), all at query token t with pristine FULL history. One-loop mask = recurrent layers of loop r only; 'own delta' injects the one-loop mask's h_r change (should reproduce it up to BF16 rounding); 'random' injects a random direction with the same norm.

KL is heavy-tailed across cases, so medians are shown next to means.

| model | arm | n | mean KL | median KL | median E(t,R) |
|---|---|---|---|---|---|
| huginn-0125 | all_loops_plain | 48 | 0.001132 | 3.994e-05 | 0.01341 |
| huginn-0125 | inject_own_delta | 336 | 0.0002421 | 2.105e-05 | 0.01225 |
| huginn-0125 | inject_random_matched | 672 | 0.0002332 | 2.151e-05 | 0.01218 |
| huginn-0125 | one_loop_mask | 336 | 0.0002296 | 2.687e-05 | 0.01218 |
| huginn-0125 | rec_only_all_loops | 48 | 0.0007724 | 3.869e-05 | 0.01343 |
| ouro-2.6b | all_loops_plain | 48 | 0.2783 | 9.009e-05 | 0.01502 |
| ouro-2.6b | inject_own_delta | 192 | 0.03123 | 3.894e-05 | 0.01353 |
| ouro-2.6b | inject_random_matched | 384 | 0.06575 | 6.193e-05 | 0.01858 |
| ouro-2.6b | one_loop_mask | 192 | 0.03123 | 2.928e-05 | 0.01356 |
| ouro-2.6b | rec_only_all_loops | 48 | 0.2783 | 9.009e-05 | 0.01502 |

| model | loop r | one-loop mask KL mean / median | random-matched KL median | median random/mask KL ratio | median E(R)/E(r) after one-loop mask | own-delta≈mask: median rel. gap, Spearman |
|---|---|---|---|---|---|---|
| huginn-0125 | 1 | 0.000258 / 4.1e-05 | 1.97e-05 | 0.869 | 1.75 | 0.201, 0.786 |
| huginn-0125 | 2 | 0.000235 / 2.38e-05 | 3.72e-05 | 1.03 | 1.64 | 0.415, 0.891 |
| huginn-0125 | 4 | 0.000203 / 2.05e-05 | 2.42e-05 | 1.37 | 1.61 | 0.767, 0.805 |
| huginn-0125 | 8 | 0.000233 / 1.99e-05 | 2.08e-05 | 1.16 | 1.76 | 0.294, 0.846 |
| huginn-0125 | 16 | 0.000208 / 1.53e-05 | 2.92e-05 | 1.1 | 1.74 | 0.494, 0.813 |
| huginn-0125 | 24 | 0.00022 / 2.91e-05 | 3.9e-05 | 1.11 | 1.73 | 0.554, 0.809 |
| huginn-0125 | 32 | 0.00025 / 3.36e-05 | 2.06e-05 | 0.926 | 1 | 0.00333, 0.967 |
| ouro-2.6b | 1 | 0.000264 / 4.36e-05 | 0.000192 | 1.87 | 1.03 | 0.532, 0.808 |
| ouro-2.6b | 2 | 0.000291 / 4.84e-05 | 0.000302 | 3.24 | 1.03 | 0.317, 0.877 |
| ouro-2.6b | 3 | 0.000258 / 2.69e-05 | 0.000148 | 2.83 | 1.04 | 0.438, 0.615 |
| ouro-2.6b | 4 | 0.124 / 2.21e-05 | 2.12e-05 | 1 | 1 | 1.98e-05, 1.000 |

| model | Σ one-loop KL (probed loops) | recurrent-only all-loop KL | all-loop + plain KL |
|---|---|---|---|
| huginn-0125 | 0.001607 | 0.0007724 | 0.001132 |
| ouro-2.6b | 0.1249 | 0.2783 | 0.2783 |

Huginn probes 7 of 32 loops, so its Σ is not a full additive decomposition.

## Stage A supplement — Ouro-native per-loop readout (generated tokens, lm_head(h_r))

| workload | units | KL(p_r‖p_R) r=1 | r=2 | r=3 | top-1 = final at r=1 | r=2 | r=3 |
|---|---|---|---|---|---|---|---|
| W1 | 40 | 0.712 | 0.0673 | 0.00922 | 0.843 | 0.934 | 0.973 |
| W2 | 40 | 0.391 | 0.0467 | 0.00635 | 0.919 | 0.963 | 0.985 |

## Auxiliary — v5 Controlled Retrieval persistent census (evidence labels post hoc only)

| model | prompts | sources | evidence | mean KL evidence | mean KL non-evidence | evidence share of top-5% | median evidence KL percentile |
|---|---|---|---|---|---|---|---|
| huginn-0125 | 2 | 774 | 44 | 0.1121 | 0.0009998 | 0.289 | 0.773 |
| ouro-2.6b | 2 | 776 | 44 | 0.01169 | 0.002491 | 0.237 | 0.655 |

