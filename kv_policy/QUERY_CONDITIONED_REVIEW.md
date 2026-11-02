# v7 review — query-conditioned Context-KV sensitivity × dynamic exit (Huginn-0125 R=32, Ouro-2.6B R=4)

Date 2026-11-02. Branch research/query-conditioned-kv-recurrence-exit-20261102. Plan: QUERY_CONDITIONED_RECURRENT_KV_PLAN.md.
Journal: QUERY_CONDITIONED_JOURNAL.md (E0–E7). All GPU runs on the H200×8 host, B=1, BF16 forward, full-sized reference KV +
read masks (no HBM/TPOT claim). Code: kvp/query_conditioned_probe.py, scripts/query_conditioned_sweep.py,
scripts/query_conditioned_analysis.py; CPU tests tests/test_query_conditioned_probe.py (8) + tests/test_query_conditioned_analysis.py (9).

## Runs (all complete, 0 errors, every unit passed the v7 validation gate)

| run | purpose | units | jobs | code |
|---|---|---|---|---|
| results/20261102T_qc_v7_timing_smoke | parity/timing receipts + clean-only exit calibration (stopped after 1 CR unit/model) | 2 | partial | 47138db5 |
| results/20261102T093000Z_qc_v7_pilotA | DEV: 8 CR + 8 HotpotQA UIDs per model, exhaustive Context sources, LIVE arms, v6 W1/W2 arm | 48 | 466 | 2d5b6136 |
| results/20261102T103000Z_qc_v7_pilotB_holdout | HOLDOUT: remaining 12 UIDs per dataset per model, same frozen protocol | 48 | 610 | 2d5b6136 |
| results/20261102T112500Z_qc_v7_pilotA_policy | equal-budget retention policies (dev) | 32 | 32 | 013cbbe5 |
| results/20261102T132500Z_qc_v7_pilotA_policy_v2 | loop-resolved lookahead/deferred variants (dev) | 32 | 32 | 5bce5d4d |
| results/20261102T140000Z_qc_v7_pilotB_policy | equal-budget policies incl. loop-resolved variants (holdout) | 48 | 48 | 7333c3dd |

Prompts/targets: the frozen v5 pilot20 Controlled-Retrieval (16 facts) and HotpotQA-Distractor evidence_excerpt prompts
(results/20261030T164908Z_loopzip_failure_pilot20/jobs.json, seed 20261031), gold continuation "Answer: <x>\n" teacher-forced
(first target excluded from utility). Sources: every aligned Context token (352–589 per prompt; 33,894 source outcomes in
dev+holdout). Live-exit thresholds frozen from clean-only smoke receipts: Ouro q=0.7, Huginn fp32 cosine 1e-3, min depth 2.

## Validation / parity (validate_v7, every unit)

No-mask query probe E=0 and KL=0; probe replay exact (0); probe rollback leaves the prompt KV digest unchanged (0); one-loop mask
at s leaves loops < s exactly unchanged; fixed decode == reference (0); forced-R replay == fixed (0); accumulated arm == fixed
evicted decode (0); first target invariant (0); FULL-LIVE repeat identical; FULL-LIVE == recorded replay (0). BF16 floor: any mask
changes the hidden state by E_R ≈ 0.013–0.016 (both models, independent of the removed attention mass) and the output by
KL ~1e-5; hidden-state error is therefore NOT a usable sensitivity metric; output KL / top-1 are used.

## Go/no-go against plan §5

1. **Measurable, repeatable query-conditioned effects at the prompt anchors?** NO. At the last Question token and at the last
   prompt token the all-loop / one-loop mask responses sit at the floor for every attention decile (dev: median KL 7e-6–1e-4,
   lowest = highest decile). Prompt-anchor probe features add nothing to the ranking of future damage (D − C: −0.018…+0.007,
   every CI crosses 0, dev and holdout, both models, both datasets).
2. **Different responses at matched immediate perturbation (H2)?** Not as a per-source property. At the query that actually
   consumes the KV (t*), the propagation gains E_R/E_s are architecture constants (Huginn ≈ 2.0, Ouro ≈ 1.0–1.2 at every loop)
   and add zero lift (G1g − F ≈ 0 in all 8 cells). Matching sources on the consuming query's attention AND closed-form local
   perturbation shrinks the within-pair damage dispersion from 1–3.6 decades (prompt-attention matching) to 0.6–0.8 decades but
   does not remove it; the residual is explained by the within-query all-loop damage at t* (probe KL orders 81–93% of Huginn
   pairs), i.e. by the consuming query's reading of the KV, not by a source-specific recurrent amplification.
3. **Does query information predict future eviction damage beyond A/B/C on held-out UIDs (H3)?** YES for the FUTURE consuming
   query, NO for the prompt query. F − C (future-query attention/δ, oracle): holdout +0.058 [0.019,0.098] Huginn-CR, +0.029
   [0.001,0.060] Huginn-Hotpot, +0.017 / +0.027 Ouro (CIs cross 0); dev +0.024…+0.067. Source trajectory (B − A) and value/local
   perturbation (C − A) add ≤0.025. The cheap surrogates S add nothing. The deployable 2-token format lookahead (L − C) adds
   +0.008…+0.034 (CI>0 in 2/8 cells); deferring the decision to after k generated tokens recovers most of the oracle by k=3–8.
4. **Native dynamic exit (H4).** Huginn re-evaluated exits change depth for 21–33% of generated queries after a single-token
   eviction (both directions; Σdepth Δ>0 in 33–40%, <0 in 20–44% of subset sources); Ouro changes 0.5–2% (almost always deeper).
   Live − recorded-replay future NLL is ≈0 in 7/8 cells (Huginn-Hotpot holdout −0.006 [−0.008,−0.005]); EM unchanged. The exit
   response exists but is not a quality mechanism.
5. **Recurrence-specific signal that does replicate: WHICH LOOP's attention.** Within-prompt ρ(attention at loop r, damage):
   Huginn loop 1 ≈ 0.00–0.08 vs 0.19–0.24 at loops ≥16, for the consuming query, the 2-token lookahead and the question anchor,
   dev and holdout; Ouro loop 1 0.07–0.12 vs loop 4 0.15–0.18. Top-5% harm sources are read by Huginn at loop ≈23–25, the rest
   peak at loops 1–4. In the equal-budget policy check the loop-1 variants of the lookahead/deferred scores are far worse than
   the loop-R/late variants (dev CR 10%: Huginn la2 ΔNLL 2.41 vs 1.05, def4 1.40 vs 0.66; Ouro la2 1.79 vs 1.08, def4 0.91 vs 0.32).

## Equal-budget policy check (keep 10/30/50% of Context; label-free scores; greedy EM + gold ΔNLL)

Dev (results/20261102T112500Z_qc_v7_pilotA_policy/analysis): every prompt-time attention policy (question-anchor loop-1/loop-R/
mean/late, all-later-prompt-queries, last prompt token) is WORSE than random at 10% retention in all four cells (ΔNLL 2.4–2.9 vs
random 1.1–1.8; CR EM 0) and ≈ random at 30–50%. The 2-token "Answer:" lookahead and the deferred-4 decision keep CR EM 0.38/0.62
(Huginn) and 0.75/0.75 (Ouro) at 30% vs 0 for every prompt-time policy and 1.0 for the future-query oracle; HotpotQA ΔNLL ≈ 0 vs
0.3–1.7. Huginn's oracle built from loop-1 attention reaches CR EM 0.25/0.5/0.62 (10/30/50%) vs 0.75/1.0/1.0 with loop-R.
Holdout (results/20261102T140000Z_qc_v7_pilotB_policy/analysis, 12 UIDs/dataset/model, all variants): replicated.
At 10% retention the prompt-time attention policies are worse than random in both CR cells (ΔNLL 2.5–2.8 vs 1.46 Huginn;
2.6–2.75 vs 1.70 Ouro) and ≈ random on HotpotQA (1.0–1.3 vs 1.2 / 0.96); CR EM 0 everywhere for them. Loop-1 vs loop-R
variants of the deployable scores (10% / 30%): Huginn-CR la2 2.26 vs 1.14 / 0.98 vs 0.63, def4 1.37 vs 0.89 / 1.13 vs 0.41;
Ouro-CR la2 1.94 vs 0.78 / 1.30 vs 0.06 (EM 0 vs 0.92), def4 0.82 vs 0.54 / 0.34 vs −0.007 (EM 0.67 vs 1.0); Huginn-Hotpot
la2 1.16 vs 0.37, def4 0.47 vs −0.12; Ouro-Hotpot la2 1.11 vs 0.26. The future-query oracle with loop-1 attention is also
worse than with loop-R on Huginn (CR 10%: ΔNLL 0.74 / EM 0.17 vs −0.05 / EM 0.83); on Ouro both oracles are at EM 1.0.

## Interpretation (what the user's hypothesis looks like after the data)

"Attention-importance selection is insufficient" — confirmed, and quantified: on these retrieval workloads prompt-time attention is
not merely weak but below random at tight budgets. The reasons, in order of size:
(a) QUERY MISMATCH: the KV that matters is read by a later generated query whose attention pattern differs from every prompt
    query's; its attention (oracle) explains the bulk of the attention-matched damage dispersion. This is "consider the later
    query", but it is an attention statement about a different query, not a recurrence statement.
(b) LOOP CHOICE (recurrence-specific): in recurrent-depth models the first recurrence's attention is a poor importance signal;
    the late recurrences' attention carries it. One-pass-style (first-loop) scores are the wrong statistic.
(c) NOT supported: a per-source "recurrent sensitivity" of the later query's computation (one-loop propagation gains, prompt-anchor
    recomputation, source trajectory) beyond the attention the consuming query pays at its late loops.
Practical consequences (not a serving claim): defer the eviction decision to after the first few generated tokens, or use the
known format prefix as a lookahead query, and score with late-loop attention.

## Not done / limitations

- Only single-token evictions for the census; the policy check masks sets greedily by score (non-additive effects not modelled).
- The gold continuation is short (4–11 tokens); the v6 128-token arm (dev only, 4+4 prompts/model) shows the same query-mismatch
  pattern (ρ 0.77–0.79 for consuming-query attention vs 0.03–0.26 for the question anchor) but was not run on holdout prompts.
- HotpotQA FULL EM is low (0.25–0.58), so EM differences there are noisy; ΔNLL/KL are the primary outcomes.
- Huginn cosine exit uses fp32 arithmetic (model-dtype cosine is degenerate); recorded as a labelled diagnostic, not official BF16.
- No HBM/TPOT measurement; full-sized reference KV with read masks throughout.
- Pilot A was used to choose analyses (future anchor, lookahead curve, loop-resolved tables); pilot B is the confirmation set.
