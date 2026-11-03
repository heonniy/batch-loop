# V10 results — FULL vs architecture-specific shared recurrent KV × Context eviction × query trajectory / exit (2026-11-03)

Branch research/full-vs-architecture-sharing-trajectory-exit-20261103 (parent v9). Plan: TRAJECTORY_SHARING_EXIT_PLAN.md.
Shared policies: Huginn-0125 (R=32) = LAST source plane for every query loop; Ouro-2.6B / Ouro-1.4B (R=4) = FIRST plane for query
loop 1, LAST for loops ≥2. Full-sized reference KV + read maps; no HBM/TPOT claim. Journal: QUERY_CONDITIONED_JOURNAL.md E13–E15.

Runs (all complete, 0 errors):
- Smoke: results/20261103T_v10_smoke_{huginn-0125,ouro-2.6b}_cr (parity receipts; revealed threshold saturation, E13).
- Main: results/20261103T160000Z_v10_{huginn-0125,ouro-2.6b,ouro-1.4b}_{controlled_retrieval,hotpotqa_distractor}
  (3 clean calibration + 5 evaluation UIDs; 6 width-4 windows per prompt = 4 label-blind + evidence + distractor oracle),
  pooled: results/20261103T160000Z_v10_cross/V10_CROSS_FINDINGS.md (prompt bootstrap).
- Follow-up (E14, clean only): results/20261103T170000Z_v10pareto_* — common threshold grid for both policies, 8 UIDs per cell,
  quality at matched realized depth (analysis/PARETO.md per cell).

## Validation
FULL read map reproduces the independent reference exactly (max |Δlogp| 0); clean repeat floor 0; every live run equals its
recorded replay (0); the first generated token is invariant under every post-prefill arm (asserted).

## Protocol change (E13, before any eviction outcome)
Pooled Full+Shared clean calibration chose thresholds under which FULL never exits (Huginn: FULL 0% early vs LAST 45–100%; Ouro
q=0.8: FULL 0% vs FIRST+LAST 5–42%). A second, Full-anchored clean-only pair was added and applied to both policies; per-policy
clean early fractions are recorded in each manifest.

## Findings
1. **H2 — sharing × eviction interaction on future NLL: null.** Interaction (shared penalty − full penalty) is within ±0.01 nats
   in 16/18 stratum cells and never has a CI excluding 0 beyond floor-level windows (largest: Ouro-1.4B CR evidence −0.04
   [−0.12, 0.002]). Evidence windows cost 0.14–0.34 nats and flip 20–40% of answers under BOTH policies alike. Same at matched forced
   depths (Huginn 8/16/24/32, Ouro 2/3/4).
2. **H1 — query trajectory.** Huginn LAST shows a much larger eviction-induced deviation in the FIRST loops for evidence windows
   (r1: 0.095 vs 0.012) that converges to the FULL value by R (0.224 vs 0.225); Ouro FIRST+LAST reduces the final-loop deviation
   by 20–40% (main 0.027 → 0.020; evidence 0.125 → 0.106) — neither changes NLL.
3. **H3/H4 — exit.** At the same threshold, shared-KV queries exit far earlier (Huginn LAST mean depth 16–20 vs FULL 30–32; Ouro ≈3.0
   vs 3.5) with no NLL cost, and eviction changes 35–70% of Huginn-LAST exit decisions (FULL: 0–36%) — again without a quality
   effect (live−replay ≤0.026, live interaction ≤0.02, answer flips identical across policies).
4. **Follow-up — quality at matched compute (clean).** Huginn: at the same realized mean depth, LAST+exit is closer to FULL-fixed than
   FULL+exit in 96–97% of matched points, median KL ratio 25–34× (e.g. CR depth ≈13: LAST KL 1.5e-4 vs FULL 8e-3; Hotpot depth ≈13:
   1e-3 vs 5e-2); LAST+exit stays at its own fixed-depth KL (~1e-4) down to ~10–13 loops, i.e. under LAST sharing the query has
   converged after ~1/3 of the recurrence. Ouro (both sizes): the opposite — FIRST+LAST carries a constant cost (KL 0.01–0.13 vs
   FULL even at depth 4) and FULL+exit is closer to FULL-fixed at matched depth (shared better in only 0–25% of points, median
   ratio 0.2–0.65). (Gold-NLL differences in this sweep are noisy and occasionally favour the shared policy; KL to FULL-fixed is
   the fidelity metric.)

## Verdict
- No evidence that the shared cache makes the models more (or less) sensitive to Context eviction (H2 null in all 3 models) — the
  plan's main interaction is a negative result.
- The sharing × recurrence interaction that IS real concerns the query's own dynamics: Huginn's LAST-shared history makes generated
  queries converge in ~1/3 of the loops, so LAST sharing + native exit dominates FULL + exit on the fidelity-vs-compute frontier
  (25–34× lower KL at equal depth). Ouro shows no such benefit. Architecture-specific, consistent with v8 (Huginn planes converge,
  Ouro's do not).
- Limitations: 5 evaluation (8 Pareto) prompts per cell, short gold continuations, fp32 cosine on BF16 states for Huginn exit,
  full-sized reference (no real memory/latency), Ouro-1.4B Hotpot shows large shared-policy prediction shifts (KL 0.13) on n=8.
