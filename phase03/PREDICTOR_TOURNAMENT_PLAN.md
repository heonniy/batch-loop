# Phase 03 — Recurrent-Depth Predictor Tournament

## Goal

Find the **best practical pre-recurrence ordering signal** for depth-compatible no-refill batching, then measure how much of the physical Oracle opportunity it actually recovers.

This phase is not an exact-depth-prediction contest. The scheduler only needs a scalar score that orders requests similarly to the perfect-depth Oracle before the recurrent interval starts.

Primary research question:

> Given only information causally available before the current recurrent interval, how much of the FIFO→Oracle RIC and wall-time gap can a practical predictor recover?

Use the Phase-2 Oracle ceilings as references, especially:

- Huginn threshold 0.12, N256/B32: physical FIFO→Oracle calls 2995→2377 and decode wall 24.30→20.37 s (16.2% wall reduction).
- Huginn threshold 0.26 and 0.50 as secondary regimes.
- Ouro q0.2/q0.4 as the second model family, but collect predictor labels/features under a self-consistent prompt/runtime contract rather than silently mixing the old uncapped Phase-1 labels with capped physical prompts.

## GPU assignment

Use exactly two physical GPUs for this phase:

- **GPU 7: Huginn-3.5B predictor lane**
- **GPU 8: Ouro-1.4B predictor lane**

Do not use GPU 0–6 for Phase 3. Each model-forward/feature-collection/physical runner must fail closed unless exactly its assigned physical GPU is visible. Record host physical index, UUID, model revision, source SHA, and CUDA_VISIBLE_DEVICES mapping in every manifest.

CPU-only analysis/training of tiny heads is allowed when it does not invoke the model, but all model feature extraction and physical timing remains on the assigned GPU.

## What counts as a causal predictor

At token boundary `t`, the predictor may use only data already available **before current-token recurrence begins**.

Allowed examples:

- previous token exit depth/history;
- previous token gate or convergence trajectory;
- previous token final/exit hidden representation;
- current sampled token ID/embedding;
- logits from the previous token that produced the current token: entropy, top-1 probability, top1-top2 margin, sampled-token surprisal;
- token position/context length;
- **Huginn only:** current token Prelude output, because Prelude executes before the recurrent core.

Forbidden leakage:

- current true exit depth;
- current recurrent states after the first recurrent step;
- current gate/convergence values produced by recurrence;
- any future token information;
- request IDs or split membership used as predictive features.

The true current depth is a training/evaluation label only.

## Data split and leakage control

Split by **request**, never by token, so tokens from the same request cannot appear in both train and test.

Preferred final dataset contract:

```text
train requests: >=256
validation requests: >=128
held-out test requests: >=256
fixed decode horizon: 64 tokens where feasible
```

If the initial feature-collection cost is too high, run a smaller screening split first, but the final winner must be re-evaluated on a fresh held-out N256 test pool before a physical claim.

Freeze and hash:

- prompt IDs/content;
- token trajectories used for labels/features;
- true depth labels;
- request split membership;
- feature schema/version.

For Huginn, use shared-KV B=1/live adaptive semantics to collect labels, matching Phase 2. For Ouro, collect a self-consistent label/feature cohort with the same prompt truncation and model/runtime policy used by the predictor experiment.

## Ten predictor candidates

Run all ten recipes under the same request split. A candidate can have small internal hyperparameter variants selected on validation only; report the final frozen variant on test.

### C0 — Previous depth baseline

```text
score_t = depth_{t-1}
```

Token 0 uses a constant/fallback score. No training. This is the minimum causal baseline.

### C1 — Depth-history statistics

Features:

- last 1/2/4/8 depths;
- rolling mean/std/min/max;
- trend and token position.

Fit a tiny linear/MLP regressor to scalar depth/difficulty. This tests temporal locality beyond the old `prev/recent4` heuristic.

### C2 — Gate / convergence trajectory

Model-specific previous-token trajectory:

- Ouro: previous gate logits/probabilities/margins over recurrence;
- Huginn: previous latent-difference/convergence values over recurrence.

Include previous exit depth and trajectory summary. Tiny MLP → scalar score.

### C3 — Token uncertainty

Features already available from logits that generated the current token:

- sampled-token log probability / surprisal;
- entropy;
- top-1 probability;
- top1-top2 margin;
- token position / context length.

Tiny MLP → scalar score. No extra Transformer compute.

### C4 — Current token embedding probe

Use the current sampled-token embedding plus lightweight positional/context scalar features.

Compare a linear probe and one tiny MLP; choose by validation RIC recovery.

### C5 — Representation probe

This is the strongest representation-only candidate.

- **Huginn:** current-token Prelude hidden state `h_pre(t)` → linear/MLP scalar score.
- **Ouro:** previous exit hidden state + current token embedding → linear/MLP scalar score.

The base LM stays frozen.

### C6 — Cheap feature fusion regression

Fuse the best causal features from C1–C5 into a small MLP and regress depth with Huber/L1-style loss.

The output is still used only as an ordering score; exact-depth accuracy is secondary.

### C7 — Ordinal depth distribution

Predict `P(D=r | features)` with an ordinal/cumulative formulation rather than flat multiclass classification.

Test ordering by:

- expected depth `E[D]`;
- validation-selected conservative quantile if useful.

Report uncertainty calibration as a secondary diagnostic.

### C8 — Weighted pairwise ranking

Use the same best fused feature set but train directly for ordering.

For true depths `D_i < D_j`, enforce `s_i < s_j` with pairwise logistic/ranking loss. Weight pairs by a clipped function of depth gap, e.g. `|D_i-D_j|`, so confusing depth 3 with 15 matters much more than confusing 7 with 8.

This is a leading method candidate because the Oracle itself is an ordering policy.

### C9 — RIC-aware listwise ranking

Train a listwise scorer on groups sampled from a token boundary to approximate the actual scheduler objective:

```text
minimize sum_m max(true_depths in microbatch_m)
```

Because hard sort/max is nondifferentiable, use a documented surrogate such as a soft-sort/listwise ranking objective or a max-sensitive pair/list loss. Keep the implementation small and reproducible; do not build a giant policy model.

This candidate should explicitly penalize placing a deep outlier into an otherwise shallow predicted microbatch.

## Feature standardization and head size

Keep predictor overhead tiny relative to the base model.

Recommended default heads:

```text
Linear probe
or
input -> 256 -> 64 -> scalar / ordinal outputs
```

Base models are frozen. Record parameter count, feature extraction cost, prediction GPU/CPU time, sort time, and any extra memory.

Avoid large sequence models in the first tournament. A GRU/Transformer predictor may be added only if all ten lightweight candidates fail to recover meaningful Oracle RIC and the evidence suggests temporal modeling is the missing signal.

## Huginn staging requirement

Prelude-hidden prediction changes where grouping can happen. Do not confound predictor gain with a different Prelude batching implementation.

For physical Huginn comparison, use the **same staged runtime** for all strategies:

```text
ready pool
  -> Prelude execution / cached h_pre
  -> score or no-score
  -> FIFO / Predictor / Oracle grouping
  -> recurrent survivor runtime
  -> Coda
```

Include a `STAGED_FIFO` control with identical Prelude staging but stable FIFO recurrent grouping. Compare Predictor and Oracle against this control. Predictor overhead must be included in decode wall time.

## Ouro staging requirement

Ouro has no Prelude. Before grouping at token `t`, derive features from prior-token state plus the already sampled current token. Use the exact same recurrent runtime/KV backend for FIFO, Predictor, and Oracle.

For token 0, define and log a deterministic fallback feature path rather than leaking current recurrence.

## Tournament metrics

### Primary trace-side metric: Oracle RIC recovery

For a held-out test cohort:

```text
RIC_recovery = (FIFO_RIC - PRED_RIC) / (FIFO_RIC - ORACLE_RIC)
```

Interpretation:

- 0% = no better than FIFO;
- 100% = same RIC as perfect-depth Oracle;
- negative = worse than FIFO.

Use the exact same requests, token boundaries, B, and true work for FIFO/Predictor/Oracle.

### Primary physical metric: Oracle wall-time recovery

```text
wall_recovery = (T_FIFO - T_PRED) / (T_FIFO - T_ORACLE)
```

The final tournament winner is the candidate with the best **held-out physical wall-time recovery after predictor overhead**, not necessarily the best depth MAE.

### Secondary diagnostics

Report:

- predictor RIC and RIC reduction;
- Spearman rank correlation;
- Kendall tau;
- depth MAE / exact depth only as diagnostics;
- microbatch max-depth distribution;
- fraction of shallow microbatches terminating before Rmax;
- predictor inference + sorting overhead;
- recurrent GPU time and decode wall time;
- throughput for the fixed decode contract;
- active-width curves;
- peak HBM.

## Screening and selection

### Stage 1 — feature collection

On GPU7/GPU8, collect causal features and true depth labels with frozen request splits. Store compact feature datasets so repeated head training does not rerun the LLM.

### Stage 2 — all-10 trace tournament

Train/evaluate C0–C9. For each model/regime produce:

```text
candidate
feature set
loss
predictor params
validation RIC recovery
test RIC recovery
Spearman
Kendall
depth MAE
inference overhead estimate
```

Rank primarily by held-out test RIC recovery; use validation only for hyperparameter/model selection.

### Stage 3 — top-3 physical tournament

Take the top three distinct methods per model (or shared top methods if the same recipes win) into the real survivor runtime.

Physical comparison order for each selected regime:

```text
STAGED_FIFO / FIFO
PREDICTOR_A
PREDICTOR_B
PREDICTOR_C
ORACLE
N=B zero-headroom control where applicable
```

Use >=2 warmups and >=5 measured repeats per final strategy when feasible. Preserve individual repeat distributions.

### Stage 4 — declare the winner honestly

Always report a ranked winner, but distinguish:

- **best measured candidate**;
- whether it provides a meaningful positive gain;
- how much Oracle RIC/wall opportunity it recovers.

Do not cherry-pick a train split, threshold, request subset, or token horizon merely to make a method win. Negative candidates remain in the result table.

## Primary physical regimes

Start with the strongest validated Oracle settings:

### Huginn on GPU7

```text
threshold = 0.12
N = 256
B = 32
32-token physical confirmation first
64-token trace/test characterization where available
shared physical-layer KV
```

Then confirm the winner at threshold 0.26 as a second regime if cost permits.

### Ouro on GPU8

Start with q=0.2 and/or q=0.4 using a self-consistent feature/label/runtime prompt contract. Prefer N256/B32 because Phase 2 showed strong Oracle hardware room, but do not reuse mismatched uncapped labels as if they were live labels for capped predictor features.

## Success criteria

Do not require an arbitrary number in order to publish the result, but use these interpretation bands:

```text
RIC/wall Oracle recovery < 25%   -> weak predictor
25–50%                           -> partial signal
50–70%                           -> useful
>70%                             -> strong practical approximation of Oracle
```

The physical wall result is authoritative. A candidate with high RIC recovery but excessive predictor/staging overhead can lose to a simpler candidate.

## Required artifacts

Create a new result root, e.g.:

```text
phase03/predictor_tournament_results_YYYYMMDD/
```

Publish:

- `feature_manifest.json` with request split and causal feature schema;
- compact frozen feature/label files or hashes/receipts if too large;
- `predictor_tournament.csv` with all ten methods;
- model-specific validation/test ranking tables;
- training configs and checkpoints for top candidates;
- top-3 physical `physical_matrix.csv`;
- raw run manifests and per-repeat summaries;
- `RESULT_REVIEW.md`;
- `review_manifest.json`.

The review must identify the final best method for Huginn and Ouro, its Oracle RIC recovery, its physical wall-time recovery, and predictor overhead.

## Scope exclusions

Still exclude:

- native/full-depth quality comparison;
- GSM8K/perplexity;
- refill;
- changing the base LM weights;
- using true current depth at runtime;
- reporting trace-only recovery as physical speedup.

## Decision at the end

At completion choose one:

1. one shared predictor design works well for both model families;
2. Huginn and Ouro need model-specific feature heads but share the same RIC-aware scheduling objective;
3. lightweight predictors recover too little Oracle room, motivating a new predictor architecture/problem framing.

Do not stop after the first positive candidate. Run the full lightweight tournament and identify the best measured method.