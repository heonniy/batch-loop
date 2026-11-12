# Implementation handoff — Stage A first, no GPU jobs

Status: PLAN. This file authorizes code changes and analysis planning in a *new additive directory* only, not automatic execution on GPUs or modification of previous receipts.

## Priority 1 — CPU retrospective (one short PR/commit)
1. Read `dynamic_sd/decision_characterization_20261113/EXPERIMENT_PLAN.md`, prior `dynamic_sd/followup_20261112/{RESULTS.md,fu_analyze.py,oracle_frontier.py,counterfactual_actions.py}`, previous wiki F-079–F-085, and existing artifact manifest.
2. Implement `decision_decompose.py` CLI that reads `.jsonl.gz` and metadata without unpacking or changing committed receipts, parses all repeat/action rows and verifies identical prefix+target contract, completeness, strict accuracy, and repeated timing geometry. Unit-test missing grid, nonexact action, changed prefix, zero wall, and prompt leakage.
3. Produce `decomposition.json` and `decomposition.md` on per-exit cells: globally calibrated best static; adaptive AR-only, K-only (fixed d), d-only (fixed K), Joint. Use prompt-cluster CI, 3-repeat cross-fitting only for **same-prefix descriptive oracle**; do not label it test generalization. Print selected d0,K0 and action counts, and no-gain results.
4. Implement `conditioned_crossover.py` that stratifies paired actions by **reference D** at the checkpoint and fixed K, with per-bin n and prompt CI; flag sparse bins. Clearly mark post-hoc observations. If current rows lack per-action acceptance/work breakdown, report exactly which fields are missing and stop rather than reconstruct them.
5. Commit code/tests before newly derived results; create a fresh results folder and manifest. Do not modify F-079–F-085 or existing experiment receipts.

## Priority 2 — Conditional GPU follow-up (requires explicit authorized GPU execution)
- If fixed-versus-live decomposition reveals an actual d-specific incremental gain, add per-action recurrence/acceptance and stage-wall instrumentation, run 32-prompt smoke/100-prompt pilot on both architectures; require FP32 FULL native parity. Do not run any GPU/model process merely because this document exists.
- Only after A/B evidence, implement historical acceptance-cost baseline, previous depth, and pre-core recurrent features on prompt-disjoint train/validation/test; measure net wall after controller cost. No RL until cheap baselines fail with unexplained headroom.
- Optimized B>1 SD is deferred until offline positive signal and separate native-exit/graph parity; do not invent a CUDA-Graph-compatible dynamic SD engine.

## Reproducibility / negative results

Log raw receipt path/hash, exact source SHAs, row counts, prompt split, model/exit regime, action grid and validation failures. The most useful result may be **NO-GO for adaptive draft d**, even if K-only oracle remains positive. Report that plainly. For any related already-tested setting cite F-079/F-080/F-081/F-082/F-083/F-085 rather than rebranding it as new.
