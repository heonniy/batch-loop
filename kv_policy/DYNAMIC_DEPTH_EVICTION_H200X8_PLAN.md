# Dynamic Exit Depth as a Signal for KV Eviction — Huginn and Ouro (2026-10-30)

**Repository**: `heonniy/batch-loop`
**Branch**: `research/dynamic-depth-kv-eviction-h200x8-20261030`
**Status**: reference engine + experiment runner + tests committed.
**IMPORTANT**: This branch does NOT contain H200 result measurements. GPU work can only
be run from the actual H200 host with model/checkpoint permissions.

## Hypothesis and falsification

**User hypothesis**: a token that exits after more recurrent iterations carries
more valuable history, so ignoring its exit depth during KV eviction may hurt quality.

**Necessary objection**: token (i)'s exit depth (d_i) measures compute spent
to predict the **next** token from prefix (x_{<=i}), not how useful (x_i)'s
KV will be to **future** query tokens. E.g., a difficult prediction may be
followed by an easy but crucial fact token. Deep tokens need NOT be valuable
sources; deep tokens can even be costlier to preserve if storing all loops.

**Testable revised hypothesis**:

> Holding attention, age and prompt/answer region constant, does the
> native exit depth of historical token `i` add held-out predictive
> information about the future NLL damage caused by evicting token `i`?

Let `T` be a fixed teacher-forced source prefix, `Y` its gold suffix.
For historical source `i`, define its single-source eviction oracle:

```
U_i = mean_NLL(Y | evict ALL KV of source i, recorded exit trace)
      - mean_NLL(Y | retain ALL KV, SAME recorded exit trace)
```

Positive U means evicting the source hurts gold next-token prediction;
negative U means an eviction can (locally) help. **Do not clip negatives.**

The first gate is `Corr(depth_i, U_i)`; the stronger gate is whether
`depth_i` adds information after conditioning on:
- Attention received by token i from EARLIER, already-seen queries
  during prompt processing (not oracle future Attention);
- age/position at the time eviction would take place;
- prompt region (initial study restricts all candidate sources to prompt);
- model checkpoint, dataset, exit criterion, source/target lengths.

A high source exit depth is a possible signal, NOT an established
importance label. Raw R=32 Huginn depths and R=4 Ouro depths are not
commensurate; compare within each model/dataset.

## Model-specific live exit rules

| Model | Pinned revision | Rule (B=1) | Initial experiment |
|---|---|---|---|
| Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | learned early_exit_gate sigmoid / cumulative Q-exit | q=0.7, min 2, max R4 |
| Ouro-2.6B | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | same model-native gate | q=0.7, min 2, max R4 |
| Huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | official cosine convergence `1-cos(h_r,h_(r-1)) < 1e-3` | min 2, max R32 |

**Do not equate learned Ouro ponder probabilities with Huginn's
unsupervised convergence evaluator.** The chosen min-depth 2 is
a controlled experiment setting (not claimed to be every official default).

Official source references:
- Ouro released model forward: `https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/modeling_ouro.py`
  (`model.early_exit_gate`, `lambda=sigmoid`, residual final-loop probability,
  and `exit_threshold` cumulative selection).
- Huginn `https://huggingface.co/tomg-group-umd/huginn-0125/blob/main/raven_modeling_minimal.py`
  (`CosineExitEvaluator`, auto threshold 1e-3).
- CDB `https://arxiv.org/abs/2608.09444` for ragged-depth/cache semantics.

The new experimental reference physically executes only
`1..d_i` recurrent steps for each **token**, then fills the **missing
historical** cache planes `r>d_i` with its last actually computed
`KV_i^(d_i)`. Queries at depth r read:
`KV_i^(min(r,d_i))`. No source token is recomputed (no-refill).
Current token's own KV is always computed at current r.

**Exact eviction timing:** After the LAST prompt token has completely
executed and its next-token logits have been produced, delete the chosen
historical token's KV. Prompt computation and prompt token exit depths are
therefore IDENTICAL to the FULL dynamic run. The first generated token
prediction is already determined; future gold targets after it can be
affected. Attention used as an online predictor is collected only from
prompt queries (excluding self), so no future gold query enters selection.

For Huginn, prelude and coda are executed once per token. Eviction masks
the source token in **all** physical layers, including prelude and coda.
For Ouro, all physical recurrent attention layers are masked.

**Critical technical gate**: tiny-model dynamic q=1/no-exit must
match existing full/strict output logits and NLL. CPU test model parity
must pass BEFORE using pretrained H200 outputs. Any mismatch blocks
the study; do not report synthetic proxy depths as native exits.

## Experimental workflow

### E0 — CPU and model parity (mandatory)

```bash
git fetch origin
git checkout research/dynamic-depth-kv-eviction-h200x8-20261030
cd kv_policy
python -m pytest -q tests/test_depth_eviction_probe.py
python -m pytest -q tests/test_depth_eviction_model_parity.py
```

If the Huginn evaluator differs from the official BF16 output on a
small B1 input, diagnose dtype/normalization, checkpoint revision
and caching order before any full run. Ouro gate should be
cross-checked against official HF forward cumulative probability.

### E1 — 3 checkpoints × 2 datasets × B=1 teacher-forced pilot

Frozen GSM8K and MATH500 prompt/gold tokenization from `kvp.jobs`,
with the original model-specific prompt templates and revisions.

- First: 4 requests per model per dataset (24 total requests);
- For each: last 96 prompt tokens (**explicit left truncation**) plus
  first 16 gold target tokens; random 4 **past prompt** source tokens
  selected from positions >=4 (exclude attention sinks);
- Record FULL-dynamic exit trace on the resulting EXACT input;
- Re-run unchanged trace on same B1 input; require max target-token
  NLL difference <=1e-4; if not, FAIL;
- For each selected source i, mask its KV only AFTER complete prompt prefill,
  at all layers while forcing the same exit depths. Compute U_i;
- Record both already-available past attention and (separately
  labeled) **future/oracle** attention for diagnosis;
- Run independent single-device workers across 8 authorized H200 GPUs.

Pilot should produce up to 24 requests × 4 sources = 96 oracle rows.
This is NOT GSM8K/MATH500 exact-match Accuracy: by design prompts are
truncated, and future targets are teacher forced. Do not interpret the
short-window results as full benchmark quality.

**Cost controls**: first execute on 1 GPU with
`--n-examples 1 --n-candidates 2` (new run directory). Check R,
that exit-depth values vary, and the B1 replay gate. Only then use
all 8 H200s. Prevent retries of failed locked jobs without manual audit.

### E2 — Statistical characterization (no new models)

Autogenerated artifacts:
- `raw/<stable_job_key>.json`: exact token hashes, model revision,
  per-source oracle U, frozen trace, pass/fail replay.
- `analysis/token_oracle.csv`: depth, positive/negative U, age,
  past attention and **oracle-only future attention**.
- `analysis/depth_utility_summary.json`: depth vs U Spearman,
  past attention vs U Spearman, **descriptive in-sample**
  incremental R² conditioned on age/attention,
  exit-depth histogram summary, missing/failing jobs.

**Promote only if** there are enough unique exit depths and signs of
nontrivial positive association (not merely a few outlier examples).
If a model exits at nearly the same depth for all prompt tokens, change
q (Ouro 0.5,0.9) or threshold (Huginn 5e-4,5e-3) in **new**
pre-registered run directories. Do not tune on the held-out test.

**No automatic causal or efficacy claim from correlation**.
Follow-up should group-bootstrap requests (NOT individual correlated
tokens), compare held-out depth added to age/attention predictors, and
include depth-shuffle controls within request and prompt strata.

### E3 — Conditional multi-token policy comparison (NOT implemented in this patch)

If E1/E2 supports added information, run **actual jointly evicted
multi-source sets** under a fixed cache budget (not sum of individual
oracle U_i; interactions are nonadditive). Compare:
1. Prompt-protected Random and age/recency;
2. Online past-attention-only;
3. Source-depth-only (prefer deep and reverse-depth as negative control);
4. Online attention + source depth;
5. Shuffled-depth (same histogram, broken token association);
6. Oracle single-source utility as a diagnostic *upper-bound clue*.

All baselines use SAME number of cached source tokens for shared
one-plane caches; for Full ragged-depth caching, match exact bytes,
since an early-exit source has fewer physical KV planes.
Repeat with LAST/strict and dynamic FULL separately: eviction vs
cross-loop KV substitution are distinct interventions.

Score free greedy generation + Accuracy, teacher-forced NLL and actual
compact-implementation HBM/TTFT/TPOT only AFTER the quality gate.
Replay and live exit are two different conditions, tested separately.
No trace-dependent future token reads allowed in an online selector.

## Execution commands (H200 host only)

```bash
cd kv_policy

# 1. PREPARE only — no CUDA/models are loaded, tokenizer/dataset access needed
python scripts/depth_eviction_sweep.py --prepare \
  --run-dir results/dynamic_depth_eviction_pilot_20261030 \
  --models ouro-1.4b,ouro-2.6b,huginn-0125 \
  --datasets gsm8k,math500 --n-examples 4 \
  --max-prompt-tokens 96 --max-target-tokens 16 \
  --n-candidates 4 --ouro-q 0.7 --huginn-cosine 0.001

# 2. RUN — only when H200 0..7 are confirmed idle/authorized.
python scripts/depth_eviction_sweep.py --execute \
  --run-dir results/dynamic_depth_eviction_pilot_20261030 \
  --gpu-ids 0,1,2,3,4,5,6,7

# 3. ANALYZE — offline, CPU-only, possible on a different machine
python scripts/depth_eviction_sweep.py --analyze \
  --run-dir results/dynamic_depth_eviction_pilot_20261030
```

Same code SHA and exact frozen manifest must be used for the entire
run. The launcher validates source SHA and checkpoint revisions
before starting workers. Partial results can be analyzed explicitly;
missing jobs are never silently treated as successful. Workers keep
exclusive lock claims until manual intervention after failure.

## Decision rules

- **Data-quality gate**: no errors, all required model/dataset cells,
  q/threshold not degenerate, exact independent Full-no-exit parity,
  frozen trace replay NLL difference <=1e-4.
- **A / Positive evidence**: within at least one model, depth has a
  reproducible positive association with future KV eviction utility,
  adds held-out predictive value beyond past attention+recency, and
  improves actual joint eviction at matched bytes.
- **B / Negative evidence**: no added association, high-depth tokens
  are no more important than lower-depth controls, or a strong simple
  attention/prompt-random baseline matches depth-based methods.
- **C / Model difference**: Ouro positive/Huginn negative (or vice
  versa) is a legitimate architecture-specific result; NEVER
  force common q/threshold or compare raw exit depths across models.
- **D / Inconclusive**: insufficient depth variance or nonnative
  context truncation dominates — report as inconclusive and expand
  controlled datasets before framing a result.

## Important scope

This patch contains the E1 oracle and E2 descriptive analysis; it does
not implement E3 online multi-eviction serving. No GPU experiment has
been executed by this ChatGPT session: GitHub connector grants
repository editing, not SSH/H200 CUDA process access. Passing the
CPU-only tests confirms a fraction of E0, not production HF parity.
Do not disturb root Phase04R predictor experiments, any other user's
GPUs, or the FERRY repo.
