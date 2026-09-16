# AGENTS.md

## Current priority — practical recurrent-depth predictor tournament (2026-09-16)

Read in order:
1. `phase03/PREDICTOR_TOURNAMENT_PLAN.md` — canonical Phase-3 predictor/search contract.
2. `phase02/oracle_upper_bound_results_20260916/RESULT_REVIEW.md` — physical FIFO/Oracle ceilings that the predictor must recover.
3. `phase02/ORACLE_UPPER_BOUND_PLAN.md` — RIC definition and Oracle methodology.
4. `phase02/PHYSICAL_EARLY_EXIT_IMPLEMENTATION.md` — survivor runtime/KV contract.
5. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — provenance/result-review rules.

### Research target

The physical Oracle opportunity is now established. Move to a bounded practical-predictor study.

The task is **not** primarily to predict the exact integer recurrence depth. The task is to produce a causal pre-recurrence difficulty score that makes the recurrent microbatch ordering approximate the perfect-current-depth Oracle and minimizes the resulting microbatch maximum depths / RIC.

Run the full lightweight candidate tournament in `phase03/PREDICTOR_TOURNAMENT_PLAN.md`; do not stop after the first positive method. Identify the best measured predictor for each model family and then confirm the top methods in the physical survivor runtime.

### GPU ownership

Use exactly these physical GPUs for Phase 3 model execution:

- **GPU 7 — Huginn-3.5B only**
- **GPU 8 — Ouro-1.4B only**

Do not use GPU 0–6 for Phase 3. Every GPU runner must fail closed unless exactly the assigned physical GPU is visible and must log physical index, UUID, CUDA_VISIBLE_DEVICES mapping, source SHA, model revision, and run arguments.

If GPU 8 is unavailable or not the expected accelerator, stop that lane and record the failure instead of silently moving to another GPU.

### Strong reference regimes

Start from the validated large-pool Oracle regimes, not the old N32 smoke.

Huginn primary:

```text
threshold = 0.12
N = 256
B = 32
shared physical-layer KV
```

Physical Phase-2 reference: FIFO→Oracle recurrent calls 2995→2377 and prefill-excluded decode wall 24.30→20.37 s (16.2% reduction) for the selected 32-token run.

Ouro primary:

```text
q = 0.2 and/or 0.4
N = 256
B = 32
last-exited depth-indexed KV
```

For Phase 3, collect self-consistent predictor features/labels under the same prompt/runtime contract. Do not train on mismatched uncapped depth labels and capped prompt features without explicitly labeling that as an exploratory ablation.

### Ten-candidate tournament

The canonical candidate definitions are in `phase03/PREDICTOR_TOURNAMENT_PLAN.md`:

- C0 previous depth;
- C1 recent depth-history statistics;
- C2 previous gate/convergence trajectory;
- C3 current token uncertainty;
- C4 current token embedding probe;
- C5 representation probe (Huginn Prelude hidden; Ouro previous exit hidden + current embedding);
- C6 fused-feature depth regression;
- C7 ordinal depth distribution;
- C8 weighted pairwise ranking;
- C9 RIC-aware listwise ranking.

Base LMs remain frozen.

### Causality / no leakage

At token `t`, a practical predictor may use only data available before current recurrence begins. The true current depth and current recurrent/gate state are labels only and must never enter runtime features.

Split datasets by **request**, not by token. No request may occur in both train and held-out test.

### Primary evaluation

The trace-side primary metric is Oracle RIC recovery:

```text
RIC_recovery = (FIFO_RIC - PRED_RIC) / (FIFO_RIC - ORACLE_RIC)
```

The final physical metric is Oracle wall-time recovery including predictor/sort/staging overhead:

```text
wall_recovery = (T_FIFO - T_PRED) / (T_FIFO - T_ORACLE)
```

Depth MAE, exact accuracy, Spearman, and Kendall tau are diagnostics only. A predictor with worse MAE may still be better if it creates lower-RIC microbatches.

### Huginn Prelude staging fairness

Prelude-hidden candidates require Prelude to run before recurrent grouping. Therefore physical Huginn comparisons must include a **STAGED_FIFO** control using the exact same Prelude staging/cache path but no predictor reordering. Compare Predictor and Oracle against that control; do not attribute a Prelude-runtime restructuring gain to the predictor.

### Search and selection protocol

1. Collect/freeze causal features and true-depth labels once on GPU7/GPU8.
2. Train/evaluate all ten candidates on the same request split.
3. Rank by held-out test RIC recovery, preserving all negative methods.
4. Take the top three distinct candidates into physical timing.
5. Use >=2 warmups and >=5 measured repeats per final strategy where feasible.
6. Include predictor inference/sort overhead in wall time.
7. Declare the final winner by held-out **physical wall-time recovery**, with trace RIC recovery as explanatory support.

Always identify the best measured candidate even if all candidates are weak, but never manufacture a positive result by changing splits/thresholds after looking at test outcomes.

### Runtime/KV contracts remain unchanged

- Pure no-refill.
- Ouro: physical survivor compaction + last-exited depth-indexed KV.
- Huginn: physical survivor compaction + shared physical-layer KV.
- Stable logical request↔KV-row identities while active tensors shrink.
- Huginn head-dim-96/padding systems checks must remain valid.

### Scope boundary

Continue to exclude native/full-depth quality comparison, GSM8K/perplexity, refill, and base-LM fine-tuning. This phase studies scheduling prediction and systems recovery of the measured Oracle bound.

### Required result packet

Publish under a new Phase-3 result root:

- frozen feature/split manifest;
- all-ten-method `predictor_tournament.csv`;
- validation/test metrics;
- top-candidate configs/checkpoints;
- physical FIFO/Predictor/Oracle matrix;
- raw manifests/repeat summaries;
- `RESULT_REVIEW.md`;
- `review_manifest.json`.

The review must answer: **Which predictor wins on Huginn? Which wins on Ouro? How much Oracle RIC and physical wall-time opportunity does each recover, after overhead?**

### Evidence discipline

Do not call trace-only RIC recovery a speedup. Do not call training/validation performance a held-out result. Do not omit negative candidates or failed runs. Do not invent GPU runs, predictor gains, or unavailable measurements.

Preferred subjects: `phase03: collect ...`, `phase03: implement predictors ...`, `phase03: run tournament ...`, `phase03: publish predictor ...`, `phase03: review predictor ...`.
