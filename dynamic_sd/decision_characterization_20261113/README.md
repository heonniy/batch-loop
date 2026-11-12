# Dynamic SD decision characterization (2026-11-13)

This directory contains an **experiment plan and agent handoff only**; no new derived results or GPU jobs were run for this commit.

Sequence: **A** (CPU-only decompose the already physically measured same-prefix one-round oracle: AR fallback vs adaptive K vs adaptive draft recurrence d) → **B** (post-hoc verifier-D conditional crossover, with causal/matching limits) → **C** (prompt-held-out incremental information value of recurrent features over acceptance+cost) → **D** (conditionally optimized CUDA-Graph/B>1 physical validation).

Read [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) before [AGENT_TASK.md](AGENT_TASK.md). Keep historical `dynamic_sd/followup_20261112/results_20261112` immutable. Linked source is [RESULTS.md](../followup_20261112/RESULTS.md), experiment target `heonniy/batch-loop@5c6bbea3`.

## Executed 2026-11-13 — Stage A/B (CPU only, no GPU)
- Code: `cf_receipts.py`, `decision_decompose.py`, `conditioned_crossover.py`, `tests/test_cpu.py` (CI: `.github/workflows/dsd_decision_cpu.yml`, re-derives and diffs the outputs).
- Results: [RESULTS.md](RESULTS.md), `results_20261113/` (decomposition / crossover md+json, MANIFEST.json with input/code/output sha256).
- Stage C/D not started: need new GPU traces and explicit authorization.
