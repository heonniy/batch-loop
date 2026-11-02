# v7 lab journal — query-conditioned Context-KV sensitivity (Huginn-0125 R=32, Ouro-2.6B R=4)

Branch research/query-conditioned-kv-recurrence-exit-20261102. Plan: QUERY_CONDITIONED_RECURRENT_KV_PLAN.md.
Entries are chronological; each records (hypothesis, clue, decision, configuration, what stays comparable).

## 2026-11-02 — E0: what v6 already answered (read before designing)

- v6 (results/20261031T070815Z_loopzip_v6_trajectory_pilot, STAGEA_STAGEC_FINDINGS.md): single-loop, single-query masks
  at generated queries are near the BF16 floor (median KL ~1e-5); Ouro's last loop dominates; Huginn grows an
  injected hidden error ~1.7x by R with no direction specificity; source-only trajectory features add no LOPO lift
  for Huginn (+0.011, CI crosses 0) and a small one for Ouro (+0.071) over attention+lexical controls.
- v3 smoke (results/20261030T153302Z_v3_main_smoke5): masking a Context token while processing the Instruction anchor
  gives Huginn final-state errors of 0.016–0.024 for *every* source, including sources with 1e-4 attention. That is a
  BF16 recomputation floor of the hidden state under any perturbation, not a source effect; hidden-state E alone is
  therefore not a usable sensitivity metric for Huginn — need matched no-effect controls and output-level KL.
- Depth study (results/20261030T061344Z_depth_main100_fp32cos): Ouro-2.6B q=0.7 is degenerate (all tokens at depth 4)
  on GSM8K/MATH; Huginn BF16 cosine 1e-3 is degenerate (97% at R); FP32 cosine gives variance. Live thresholds must be
  calibrated on clean traces of THESE retrieval prompts before any eviction outcome is read.

## 2026-11-02 — E1: design decisions (preregistered before GPU)

Hypothesis under test: the future post-prefill eviction damage Y_i of Context token i is explained beyond clean
prompt-query attention / position / lexical / value controls by how the *later query's recurrent computation*
responds to losing i. Three distinct explanations for "attention-matched sources with different Y" are kept apart:
  (a) query mismatch — the relevant attention is the FUTURE generated query's, not the prompt query's (oracle);
  (b) value/local-output — the immediate attention-output change `w_i/(1-w_i)·||o-v_i||` differs (one-pass, cheap);
  (c) recurrent propagation — equal local perturbation, different propagated h_q(R)/output change (the claim).

Workloads: the frozen v5 pilot20 prompts (results/20261030T164908Z_loopzip_failure_pilot20/jobs.json; Controlled
Retrieval 16 facts, HotpotQA-Distractor evidence_excerpt; 20 UIDs each, both models, same text) with the gold
"Answer: <x>\n" continuation teacher-forced as the frozen future (first target excluded from utility). Sources:
EXHAUSTIVE over aligned Context tokens (no sampling, no labels). Secondary generalisation arm: the v6 W1/W2 Bref
prompts (4 per workload per model) whose 128-step persistent-eviction census already exists — only the new
query-conditioned features are computed there.

Measurements per (unit, source i):
  Y (outcome): accumulated persistent eviction over all generated queries — per-t KL(p_full||p_ev), NLL shift,
     top-1 change, E_t(r); clean-history probe at every t (within-query vs history accumulation).
  Anchor probes (offline oracle): q = last Question token (primary) and last prompt token (separate ablation);
     masks: all loops (+plain), rec-only / plain-only (Huginn), one-loop s∈{1,2,3,4} (Ouro) / {1,2,4,8,16,24,32}
     (Huginn). Recorded: E(r), update-direction cosine, ||Δh_r||, next-token KL/top-1, exit statistics of the
     perturbed trajectory (Ouro gate λ_r, Huginn fp32 + model-dtype cosine distance).
  Clean census (one pass): per loop r: head-mean attention, max-head attention, closed-form local perturbation
     δ(i,r), value norm — at both anchors (deployable) and at every generated query t (future ORACLE).
  Source trajectory baseline: U/F/C of token i's own prefill trajectory (v6 metrics).
  Live exit (Gate D, only with frozen thresholds): FULL-LIVE x2 / FULL-REPLAY / FULL-FIXED; per subset source
     EVICT-LIVE / EVICT-REPLAY / EVICT-FIXED + fixed-depth greedy EM. Subset = 16 positional-hash + 8 top
     question-anchor attention (label-blind).

Validation gate per unit (validate_v7): no-mask probe E=0/KL=0, probe replay exact, probe rollback leaves prompt KV
digest unchanged, one-loop mask at s leaves loops < s exactly unchanged, fixed decode == reference, forced-R replay ==
fixed, accumulated arm == fixed evicted decode, first target invariant, live exit repeatable, live == recorded replay.

Threshold calibration: from clean REF receipts only (prompt + generated tokens): Ouro q chosen so that a non-trivial
fraction of clean generated tokens exits before 4; Huginn fp32 cosine 1e-3 (depth study) unless degenerate here.
Frozen in the pilot config before any eviction outcome is inspected.

Decision rules (plan §5): (1) are query-conditioned effects above the floor and repeatable; (2) do they differ at
matched attention AND matched local perturbation; (3) do they predict Y on held-out question UIDs beyond A/B/C
features (LOPO, prompt bootstrap); (4) does eviction change live depth and does that relate to quality.

## 2026-11-02 — E2: timing/validation smoke (results/20261102T_qc_v7_timing_smoke, code 47138db5)

- validate_v7 passes on both real checkpoints (all parity values exactly 0; masked max E 0.013–0.016; one-loop mask
  leaves earlier loops exactly unchanged). Prefill 49–54 s per 420–440-token prompt; CEN 3.2 s/source (Ouro),
  3.9 s/source (Huginn) incl. accumulated + clean probes at all t + 5 (Ouro) / 10 (Huginn) anchor probes x 2 anchors.
- First CR unit (84 sources seen): the answer-code digit tokens carry almost all future damage (mean KL 0.3–1.4)
  but have question-anchor attention ~1e-4 and anchor-probe KL ~1e-5 (floor), while the FUTURE generated query's
  attention to them is 10–50x larger. Clue: a large part of "attention is insufficient" is QUERY MISMATCH (the
  consuming query is a later generated token), not the prompt query's recurrence.
- Hidden-state E_R under any mask is ~0.013–0.015 for every source in both models = BF16 recomputation floor;
  Huginn one-loop gain ~2.0 and Ouro ~1.0–1.3 at that floor. E alone cannot rank sources; use output KL / top-1
  and compare against the lowest-attention decile as the empirical floor.
- DECISION (before the pilot): add a third, per-source ORACLE anchor "future_maxattn" = the generated query t* with
  the largest clean attention to the source; run the same all-loop / one-loop probes there. This measures recurrent
  propagation at the query that actually consumes the KV. Prompt anchors stay as preregistered.
- Live thresholds frozen from CLEAN prompt+generated exit statistics of the smoke REF receipts only: Ouro q=0.7
  (prompt tokens: 61% exit at 3, 39% at 4; generated 6/11 at 3), Huginn fp32 cosine 1e-3 (generated mean 24.8,
  9% at R; BF16 model-dtype cosine is recorded only as a diagnostic). min_depth 2.
- The smoke was stopped after 1 CR unit per model (partial CEN kept as receipts); it is NOT part of the pilot data.

## 2026-11-02 — E3: pilot A (dev), results/20261102T093000Z_qc_v7_pilotA, code 2d5b6136 (analysis up to d93eba0f)

466/466 jobs, 0 errors, 48 units (8 CR + 8 HotpotQA per model, exhaustive 352–589 Context sources each; 4 v6 W1 +
4 v6 W2 Bref prompts per model), all validation gates pass. Findings file: analysis/QUERY_CONDITIONED_FINDINGS.md.

1. Prompt-anchor query-conditioned probes are a NULL. At the last Question token (and the last prompt token) the
   all-loop / one-loop mask responses sit at the BF16 floor for every attention decile (KL ~1e-5, E_R ≈ 0.014 in
   both models, identical in the lowest and highest attention decile) and add negative LOPO lift (D − C: −0.01 to
   −0.02 in all four model×dataset cells). The later query's recurrence AT PROMPT TIME does not carry information.
2. The dominant reason attention fails is QUERY MISMATCH, not recurrence. The damage is consumed by a later
   generated query t*; its clean attention (oracle F) adds +0.04–0.07 LOPO lift (CI>0 in 3/4 cells) and, on the
   v6 128-token continuations, within-prompt ρ 0.77–0.79 vs 0.03–0.26 for question-anchor attention. Among
   prompt-attention-matched informative pairs (Y dispersion 2–3 decades, 80% >10x apart) the future query's attention
   orders 83–87%; matching on future attention + local δ shrinks the dispersion to 0.6–0.8 decades.
3. Recurrent propagation at the consuming query is NOT a per-source property that predicts damage: propagation
   gains E_R/E_s (G1g) give zero lift in all cells; one-loop E_s/E_R (G1) give +0.028 [0.001,0.056] only for
   Huginn-CR and ≤0 elsewhere. Huginn's gain ≈ 2 and Ouro's ≈ 1.0–1.2 at every loop are floor-amplification constants.
4. The recurrence-specific signal that DOES exist is WHICH LOOP'S attention one reads: for Huginn, attention at
   loops ≥16 is a 2–3x stronger within-prompt predictor than first-loop attention (ρ 0.07 → 0.20–0.24; same for the
   consuming query, the 2-token lookahead and the question anchor); the top-5% harm sources are read at loop ~23–25
   whereas the rest peak at loops 1–4. Ouro: loop 4 (0.15) > loop 1 (0.09). A one-pass (first-loop) attention
   score is the weakest choice in a recurrent model.
5. Deployable partial fixes: the 2-token format lookahead ("Answer:") gives +0.01–0.035 lift; deferring the eviction
   decision to after k generated tokens recovers most of the oracle (k=3–8: +0.03–0.06, CI>0 for Huginn-Hotpot and
   both Ouro cells).
6. Gate D: Huginn changes exit depth for 21–29% of generated queries (both directions; ρ(Σdepth Δ, damage)=0.07–0.23);
   Ouro changes depth for 1–2% (always deeper). Live − recorded-replay future NLL ≈ 0 (CI includes 0 in 3/4 cells):
   the native exit response neither rescues nor worsens quality; it is an exit-system observation only.

DECISION: no protocol change. Pilot B = the remaining 12 UIDs per dataset per model (held-out confirmation, same
code/thresholds; results/20261102T103000Z_qc_v7_pilotB_holdout). In parallel, add an equal-budget POLICY check of
the cheap alternatives suggested by 4–5 (loop-resolved anchor attention, lookahead, deferred-k) vs the oracle and
random, on the frozen dev prompts first (kind POL). The expensive query-recomputation oracle is NOT carried forward.
