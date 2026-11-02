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
