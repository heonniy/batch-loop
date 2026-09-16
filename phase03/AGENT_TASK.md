# Phase 03 — Coding/Experiment Agent Task

Implement and execute the recurrent-depth predictor tournament from `PREDICTOR_TOURNAMENT_PLAN.md`.

## Hard constraints

- GPU7 = Huginn only.
- GPU8 = Ouro only.
- Do not use GPUs 0–6.
- Verify physical GPU index + UUID before each lane starts; fail closed on mismatch.
- Base model weights stay frozen.
- True current depth is label/evaluation only; never feed it into runtime features.
- Split by request, not token.
- Preserve all Phase-1/2 code and results; use a fresh Phase-3 result root.
- Predictor overhead must be included in final physical wall time.
- Run the full lightweight candidate tournament before declaring a winner.

## 1. Implement a predictor package

Suggested additive layout:

```text
phase03_predictor/
  __init__.py
  features.py
  datasets.py
  models.py
  losses.py
  train.py
  evaluate_ric.py
  physical_predictor_runtime.py
  summarize.py
  tests/
```

Keep common predictor interfaces model-agnostic where possible:

```python
score = predictor(features)  # [num_ready]
order = stable_argsort(score)
```

Then use the existing Phase-2 physical survivor runtime after grouping.

## 2. Feature collection

### GPU7 Huginn

Primary regime: threshold 0.12. Also retain 0.26 if collection cost permits.

For every request/token record causal features before recurrence:

- request/sample key (metadata only, never a model feature);
- token index;
- current token id;
- current token embedding or a frozen compact projection;
- previous true depth (causal history feature);
- recent depth history summaries;
- previous convergence trajectory and summaries;
- previous token output entropy/top1 prob/top1-top2 margin/sampled-token surprisal;
- token position/context length;
- current Prelude hidden representation or a frozen projection thereof;
- current true depth label stored separately.

If storing full 5280-d Prelude hidden is too large, save a deterministic frozen projection (e.g. seeded Gaussian/orthogonal projection to 256/512 dims) and document/hash the projection. For the representation probe, optionally retain full vectors for a bounded train subset if memory permits.

The Prelude hidden must be produced before current recurrence and is therefore causal.

### GPU8 Ouro

Primary q=0.2 and q=0.4. Collect self-consistent labels/features using the exact same prompt cap/runtime contract.

Record:

- current token id/embedding;
- previous exit hidden representation or deterministic projection;
- previous learned-gate trajectory/probabilities/margins;
- previous depth and recent-depth summaries;
- previous-token uncertainty scalars;
- token position/context length;
- true current depth label separately.

Token 0 uses an explicit fallback/missing mask for previous-token features.

## 3. Request split

Preferred final split per model:

```text
train >= 256 requests
validation >= 128 requests
test >= 256 requests
```

If collection must be staged, first screen on a smaller request-split dataset, then collect enough fresh held-out requests so the final top-3 test has N256 available.

Save request IDs and hashes in `feature_manifest.json`.

## 4. Implement all ten candidates

Implement C0–C9 exactly as defined in `PREDICTOR_TOURNAMENT_PLAN.md`.

For trainable candidates use a small hyperparameter grid selected on validation only. Keep search bounded, e.g.:

```text
hidden width: 64 / 256
learning rate: 1e-3 / 3e-4
weight decay: 0 / 1e-4
```

Do not use the test set to choose epochs/hyperparameters.

Use early stopping on validation RIC recovery or validation ranking loss, and record the selection rule.

## 5. Trace-side tournament

For every candidate on held-out test, use its predicted score at every token boundary to stable-sort requests, split into fixed B microbatches, then compute RIC using the true test labels.

Primary cells:

```text
Huginn threshold 0.12: N256/B32
Huginn threshold 0.26: N256/B32 (secondary)
Ouro q0.2/q0.4: N256/B32
```

Also include `N=B=32` control where meaningful.

Output at least:

```text
candidate
model/regime
FIFO RIC
Pred RIC
Oracle RIC
Pred RIC reduction vs FIFO
Oracle RIC recovery
Spearman
Kendall tau
MAE
exact-depth accuracy
predictor params
prediction time
sort time
```

Create `predictor_tournament.csv` with all methods, including negative outcomes.

## 6. Select top three

Per model, select top three **distinct** predictor recipes by held-out test RIC recovery, with overhead as a tiebreaker.

Do not choose based on physical results yet.

At minimum, if they are competitive, ensure one representation-based and one ranking/RIC-aware method reaches physical testing so the tournament tests the core hypotheses.

## 7. Physical Huginn comparison on GPU7

Prelude hidden requires stage decoupling. Implement one common physical path:

```text
ready pool
 -> common Prelude stage
 -> STAGED_FIFO / Predictor / Oracle ordering
 -> B32 recurrent survivor batches
 -> common Coda
```

All strategies must use the same prelude/cache staging. Predictor score/sort overhead is inside decode wall timing.

Final matrix:

```text
STAGED_FIFO
TOP1
TOP2
TOP3
ORACLE
N=B=32 control
```

Primary threshold 0.12, N256/B32, 32 fixed decode tokens first. If stable and affordable, repeat winner at 64 tokens and/or threshold 0.26.

Use >=2 warmups + >=5 measured repeats for the final winner/FIFO/Oracle cell.

## 8. Physical Ouro comparison on GPU8

Use the same prompt/label contract as feature collection. Before each token recurrence:

```text
prior-token features + current token
 -> predictor score
 -> fixed-B grouping
 -> existing Ouro survivor runtime + last-exited KV
```

Compare:

```text
FIFO
TOP1
TOP2
TOP3
ORACLE
N=B control
```

Use N256/B32, 32 fixed decode tokens, >=2 warmups + >=5 measured repeats for final cells when feasible.

## 9. Winner definition

The final winner is determined by held-out **physical decode wall-time Oracle recovery after all predictor/staging overhead**:

```text
wall_recovery = (T_FIFO - T_PRED) / (T_FIFO - T_ORACLE)
```

For Huginn, `T_FIFO` means STAGED_FIFO from the identical Prelude staging path.

Also report:

```text
RIC_recovery = (FIFO_RIC - PRED_RIC) / (FIFO_RIC - ORACLE_RIC)
```

If the trace-side winner is not the physical winner because of overhead, call that out explicitly and select the physical winner.

## 10. Required review questions

`RESULT_REVIEW.md` must answer:

1. Which of C0–C9 has the best test RIC recovery on Huginn?
2. Which has the best test RIC recovery on Ouro?
3. Which method wins physical wall time after overhead on each model?
4. How much of Oracle RIC opportunity is recovered?
5. How much of Oracle physical wall-time opportunity is recovered?
6. Is Prelude hidden actually predictive enough to justify Huginn stage decoupling?
7. Does pairwise/listwise RIC-aware training beat ordinary depth regression?
8. Which feature family contributes most in ablation?
9. Does the winner generalize to the secondary threshold/q cell?
10. What is the next finite action: refine predictor, redesign objective, or move to final system evaluation?

## 11. Do not stop early

A positive C5/C6 result is not sufficient. Complete the ten-method trace tournament, run the top three physically, and publish the best measured method with its negative/positive competitors.

If all methods recover little Oracle room, that is the result; do not leak Oracle labels into runtime or change the held-out split to force a winner.
