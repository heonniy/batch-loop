# LoopZip v6 pilot — assessment (written after reviewing STAGEA_STAGEC_FINDINGS.md and TRAJECTORY_OBSERVATIONS.md)

## Go/no-go reading against plan §6 (pilot; not a Stage D result)

1. **RQ1 / normal trajectories — structured and repeatable, not homogeneous.** Huginn settle depth splits into
   dev-frozen early/mid/late shares that hold on holdout prompts (0.39/0.35/0.24 dev vs 0.33/0.35/0.29 holdout);
   Ouro (R=4) settles at r=3 for ~23% and only at R for ~77% of tokens in both splits. Per-token-ID settle depth
   repeats across disjoint prompts (Spearman 0.59 Huginn, 0.48 Ouro), and token identity carries more in-sample
   variance than prompt or role. Not inconclusive, but much of the structure is lexical.
2. **Source harm is heavy-tailed and only weakly explained by attention/position/role.** Matched adjacent same-role,
   similar-attention pairs are as dispersed as random same-role pairs (|log10 KL ratio| ≈0.3–0.5, 15–25% >10×
   apart), and this survives the dev→holdout split. Recurrent source features add **no** leave-one-prompt-out lift
   for Huginn (+0.011 [−0.039, 0.060]) and a small lift for Ouro (+0.071 [0.016, 0.117]) once lexical controls are
   included; lexical features alone already rank Ouro harm at 0.26. Gate 2 therefore does **not** support a
   recurrence-specific importance claim for Huginn; Ouro is at most a weak, unreplicated hint.
3. **Mechanism controls.** Single-loop, single-query masks are tiny (median KL ~1e-5, near the BF16 floor; own-delta
   reinjection reproduces them only to a ~20–80% median relative gap except the last loop). Within that resolution:
   Ouro's last-loop read dominates (mean KL 0.12 vs ≤3e-4 for loops 1–3), an error injected at loop r is carried to
   R without amplification (E(R)/E(r) ≈ 1.03), and random-direction perturbations of the same norm are ~2–3× more
   harmful than the actual mask direction at middle loops. Huginn grows an injected error ~1.6–1.75× by R and shows
   no direction specificity (random/mask ratio 0.9–1.4). Plain prelude/coda reads add ~30% of Huginn's all-loop KL.
4. **Evidence control (v5 CR).** Evidence tokens are far more harmful when evicted (mean KL 0.112 vs 0.001 Huginn,
   0.012 vs 0.0025 Ouro), yet only 24–29% of the top-5% most harmful sources are evidence.

Stage D (heldout predictability on independently sourced prompts) was **not run**: the reproducible part of the
effect is heavy-tailed source harm that is largely lexical/attention-related, and the recurrence-feature lift is null
for Huginn and small for Ouro. A Stage D run would be justified only as a focused Ouro test with lexical controls on
the frozen holdout pool (40 prompts/workload in manifest.json).
