# More Knobs, More Memory Room — H200×8 Loop-KV Research Execution Contract
Date: 2026-10-29. **Status: code and plans committed; no new H200 benchmark results claimed.**


## 2026-10-29 revision — professor feedback: prioritize Ouro sub-50% protection

**Revised priority B is OURO, not Huginn token-window eviction.**
The actual optimization question is:

> Can we preserve approximately the accuracy of LAST + all first-recurrence
> KV banks (50% of Full recurrent KV) at significantly smaller memory,
> ideally 30–40%, without retraining?

This is *not* an attempt to preserve "50% accuracy". 50% refers to the
**KV memory footprint of Loop1-all relative to Full**, not its task score.

### Frozen, same-semantics R4 Strict baseline (200 GSM8K / MATH500)

| Model / dataset | FULL 100% | LAST 25% | fixed calibrated 25.5/27.1% | all R1 banks 50% |
|---|---:|---:|---:|---:|
| Ouro-1.4B GSM8K | .810 | .000 | .660 (2 banks) | .720 |
| Ouro-1.4B MATH500 | .740 | .000 | .485 (2 banks) | .610 |
| Ouro-2.6B GSM8K | .880 | .600 | .770 (1 bank) | .845 |
| Ouro-2.6B MATH500 | .565 | .330 | .410 (1 bank) | .500 |

Especially on Ouro-1.4B MATH500, the region BETWEEN 27.1% and 50%
contains a 12.5 percentage-point accuracy gap. This is more promising
for selective bank assignment than Huginn R32 where LAST already
matches Full. Avoid treating additional 50%-vs-100% loss as fixed by
our method: we target **matching Loop1-all at less memory first**.

### B1 — Runnable Layer × Loop1 sub-50% Pareto screening

Policy `bankTopB{30,35,40,45}`: take the top-k layers by the original
disjoint 32-prompt single-bank NLL ranking; preserve their first-loop
KV plus the terminal KV for **every** token. This ranking was obtained
under LAST-CONVERT (not Strict Shared), hence is a *transfer hypothesis*.
Compare at exactly the same count to:
- `bankPrefixB*`: first k physical layers (simple fixed heuristic);
- `bankRandB*`: a frozen seeded random set (1 pilot seed; >=3 for paper);
- `last` (25%), `bankP0`, `bankcal`, `bankR1all` (50%), FULL (100%).

For Ouro R4, L physical attention layers, k extra first-recurrence
layer banks retain `(L+k)/(4L)` of Full recurrent KV.

| Budget ceiling | Ouro-1.4B k / actual | Ouro-2.6B k / actual |
|---|---|---|
| 30% | 4 / 29.17% | 9 / 29.69% |
| 35% | 9 / 34.38% | 19 / 34.90% |
| 40% | 14 / 39.58% | 28 / 39.58% |
| 45% | 19 / 44.79% | 38 / 44.79% |
| 50% | 24 / 50% (all Layer R1) | 48 / 50% (all Layer R1) |

The rank source and source hashes are stored in `sub50_provenance.json`;
there is an **enforced calibration-UID vs evaluation-UID disjointness
check**. Do not optimize a layer mask using the same held-out examples
later used for evaluation. If the old LAST-CONVERT rank fails on Strict,
run **new Strict-mode calibration using a separately frozen disjoint
cohort**, not posthoc quality selection on the 200 test items.

Only `Layer × Loop` static selection is implemented in B1. No
token-specific bank storage or physical HBM compaction is claimed.

### Commands (H200 x8, opt-in GPU; from kv_policy/)

```bash
# CPU tests first
python -m pytest -q tests/test_sub50_budget.py tests/test_knob_budget.py tests/test_knob_launcher.py
python -m pytest -q tests/test_policy_semantics.py tests/test_banks.py

# Preview/pin matrix and nominal budgets; still no GPU launch
python scripts/kv_knob_sweep.py --phase sub50 --task nll \
  --sample pilot --run-dir results/ouro_sub50_nll_pilot

# On user-authorized, idle H200 devices only: 8 independent workers
python scripts/kv_knob_sweep.py --phase sub50 --task nll \
  --sample pilot --run-dir results/ouro_sub50_nll_pilot \
  --execute --gpu-ids 0,1,2,3,4,5,6,7

# Select 2–3 promising budgets on SEPARATE generation-pilot run,
# retaining mandatory controls; --policies does not mutate an old run.
python scripts/kv_knob_sweep.py --phase sub50 --task gen \
  --sample pilot --run-dir results/ouro_sub50_gen_pilot \
  --policies full-strict,last-strict,bankP0-strict,bankR1all-strict,bankTopB35-strict,bankTopB40-strict,bankPrefixB40-strict

# Expand selected policies to main=200 per dataset after pilot gate.
# Use a NEW run directory, --sample main, --execute on permitted devices.
```

Artifacts: frozen `banks.json`, `sub50_provenance.json` (with source
file SHA), `nominal_logical_kv_budget.csv`, paired vs Full and paired
vs **Loop1-all** (`analysis/sub50_paired_vs_loop1all.csv`).

**Research GO:** Under identical strict semantics, achieve:
- strong: <=35% Full equivalent with >=Loop1-all Acc minus 3pp in
  both GSM8K and MATH500, over held-out 200 items;
- moderate: <=40% with that gap, or <=45% if accompanied by a
  clear, generalizable bank selection rule;
- NO-GO for proposed method if only 45–50% reaches baseline quality
  without beating prefix/random and transfer controls.
The 3pp window is a descriptive screening target, not proof of
statistical non-inferiority: publish paired confidence intervals and
larger samples if an equivalence claim is desired.

**Next steps conditional on B1:**
- **B1 successful, Top > Prefix/Random:** B2 conditional greedy
  marginal restoration after L0R1; test layers and held-out datasets.
- **Top ~= Prefix/Random but a small k works:** use a SIMPLE fixed
  first-loop protection recipe, avoid claiming calibration complexity.
- **Only 45–50% works:** B3 token-selective first-loop storage
  (protect only sensitive historical tokens/spans) and/or compact
  low-bit residual K/V; evaluate accuracy/metadata/overhead at same bytes.
- **NLL good, generation bad:** B4 on-policy/same-prefix KL and
  generation-aware protected-bank selection.
- **All sub50 masks fail:** try first/last one-plane interpolation,
  asymmetric K/V, low-bit correction before broad token eviction.

### Important comparison discipline

The old `postlast` track (Huginn LAST + token window masking) remains
available as **secondary exploratory track M**, not the primary B.
A token window removes **entire historical token K/V**, which changes
the reasoning problem and cannot be used to claim a selective first
recurrence bank's efficiency. If token eviction is later combined with
B1, compare separable vs joint policies at truly identical TOTAL bytes.

Current PolicyKV/BankKV allocates full broadcast-sized cache; the
quoted percentages are **logical retained KV**. After quality GO,
implement a compact packed store, validate reference parity, and report
actual H200 HBM peak, metadata bytes, TTFT/TPOT/TPS and feasible batch
concurrency. No measured KV reduction is claimed yet.


## Why this experiment exists

The 10/29 lab-feedback concern is not simply "try more policies." It is
"find a setting in which the proposed method can produce a *meaningful*
quality-at-memory advantage **over an already-strong LAST baseline**."

Original presentation (user-provided `backup.pdf`, slides 10–13):
token-adaptive recurrent-KV retention, inter-token eviction, and the
question of whether inter- and intra-token decisions should be JOINT.
The earlier Strict Shared receipts show:
- Huginn R32: FULL and LAST ~equal in GSM8K/MATH500.
- Ouro-1.4B R4: LAST collapse, partially recovered by L0R1 and L1R1.
- Ouro-2.6B R4: partial LAST robustness and residual bank-protection gap.
These observations are **historical baselines**; never relabel as new results.

### Research decision

**Track B (find a hard-but-viable regime) and Track M (get below LAST's
memory floor) are both first-class.** Don't spend months trying to protect
one tiny recurrent bank in R32 Huginn if LAST already meets quality.

Primary optimization target, only after quality screening:

```
min actual compact KV bytes / maximize feasible batch concurrency,
subject to a pre-registered acceptable accuracy (or NLL) regression
relative to matched FULL/strict on held-out prompts.
```

Report complete Accuracy–NLL–memory–latency Pareto fronts. Do not
conflate relative savings from FULL with **incremental** savings over LAST.

## Why the Huginn room is different

Huginn-0125 has 2 prelude attention layers + 4 recurrent physical
attention layers + 2 coda attention layers. With R loops, uniform
per-layer KV shape/dtype/context (as in the reference implementation):

```
FULL-equivalent planes = 4 + 4 R
LAST-equivalent planes = 4 + 4 = 8
```

| R | Full planes | Last planes | LAST/FULL total-KV fraction |
|---|---:|---:|---:|
| 4 | 20 | 8 | 40.0% |
| 8 | 36 | 8 | 22.2% |
| 16 | 68 | 8 | 11.8% |
| 24 | 100 | 8 | 8.0% |
| 32 | 132 | 8 | 6.1% |

`1/R` applies to *recurrent-only* KV, not prelude/coda. R32 + one
extra bank increases the total-FULL fraction by only 1/132 ≈ 0.76pp.
So there is little rationale for a paper that only deletes that extra
bank. There IS potentially large incremental room if *the remaining
one-plane history* can be token-evicted/quantized and actual compact
storage realizes the saving.

All such fractions are **logical per-token accounting**: K/V tensors
are currently still FULL-sized in the broadcast reference engine.
`kvp/knob_budget.py` and `analysis/nominal_logical_kv_budget.csv`
are illustrative models, not measured GPU HBM reductions.

## Four runnable QUALITY screening tracks

All cells use the fixed model revisions in `kvp/specs.py`, original
frozen GSM8K/MATH500, BF16, greedy, correct per-model templates and
identical STRICT token-major prefill across policies. Data and scoring
share the existing `make_quality_jobs.py`/`quality_worker.py` pipeline.
Pilot = 32 fixed items per dataset; main = 200 items per dataset.
New analyses include paired accuracy and mean-per-problem NLL bootstrap
intervals against FULL/strict on matching UIDs.

### B0 — depth screen: Huginn R4, R8, R16, R24, R32

Models: `huginn-0125-r4`, `-r8`, `-r16`, `-r24`, native `huginn-0125` R32.
Policies `full-strict`, `last-strict`, `average-strict`.
First task: teacher-forced NLL; promote candidates to free-generation
Accuracy and recovery-vs-depth tracing.

The *same-R* FULL is always the baseline. A setting is not a promising
compression target if its own FULL model is unusably inaccurate.
Do not label non-native R4/8/16/24 checkpoints "the model's intended
depth." Compare error **within** R then quality **across** R.
For a good regime, study R-conditioned bank protection or mixed policy.
If LAST ~= FULL for every viable R, STOP depth-bank selection on Huginn.

### M0 — POST-LAST token-memory screen (primary memory-room knob)

Models: Huginn R32 and R16 (custom subset allowed). Policies:
`full`, `last`, `lastwin64/128/256`,
`lastpromptwin64/128/256` (all strict).

The latter masks all but the recent W token positions plus the first
four **real** sink tokens, and optionally pins the entire prompt.
Causal mask, actual left padding and the current token are preserved.
A pruned historical token cannot reappear later.
The same historical token mask is applied at every attention layer
including Huginn prelude/core/coda; this is an *inter-token* eviction
experiment applied **after** recurrence sharing.

This is a **logical masking QUALITY prototype** using `WindowKV`
that still holds FULL-size broadcast buffers. Its measured GPU peak
memory is NOT expected to fall. Only after quality GO should the code
agent implement and validate actual compact storage, and then report
memory peak, KV bytes, TTFT/TPOT/TPS with matched workload.

Compare prompt-pinned vs fully prunable prompt. Strong controls:
FULL, LAST, random token survival, H2O/SnapKV-like token scoring
(if a faithful available implementation is integrated later).
Do not claim the current window-only implementation is H2O/SnapKV.
Include long-context retrieval and coding datasets only AFTER their
prompts/scorers have independent parity audits (not currently wired).

### P0 — one-plane representation (accuracy gain without a second bank)

Models: Ouro-1.4B/2.6B. At **one recurrent KV plane** compare
`last`, `first`, `average`, `blend25`, `blend50`, `blend75`
against FULL and fixed `bankP0`, calibration `bankcal`,
`bankR1all`. First/last blends independently mix Key and Value with
the same fixed alpha in {0.25,0.5,0.75}:

`KV_blend = alpha * KV_first + (1-alpha) * KV_terminal`

The blend retains ONE KV version; transient computation/scratch is
not counted as compressed persistent memory. Mean and blends are
screening baselines, NOT claimed as novel methods.
Potential follow-up: per-layer alpha, asymmetric alpha_K/alpha_V,
one-key/value exact protected bank, bitwidth/delta coding.
Compare against ResidualQuant, LLA, and BeNice-style protection before
claiming a method. If no one-plane representation improves accuracy,
STOP hand-tuned interpolation; move to targeted residual or learned
calibration with honest method-overhead accounting.

### J0 — independent inter×intra composition (joint-necessity gate)

Models: Ouro-1.4B/2.6B. Compare `lastwin`, `bankP0`,
`bankP0win`, `bankcalwin`, `lastpromptwin`, and corresponding
FULL/strict controls at a common window W. This tests the INTERACTION
and **independent composition** of:
1. Which historical tokens survive (causal window/sinks).
2. Which recurrent bank(s) each surviving token can read.

Here `bankP0winW` protects **the same bank for every surviving token**.
It is NOT a token-adaptive joint selection algorithm.
Compute effect interaction on matching prefixes:
`I = L(E+C) - L(E) - L(C) + L(FULL)`; nonzero suggests
nonadditive effects, but neither proves a joint method is required nor
predicts its benefits. A **real** joint-necessity claim requires an
upper-bound oracle that selects (token,layer,recurrence) cells under the
*same total bytes* and beats the best independent composition on
held-out prompts. Implement that ONLY if J0 reveals meaningful
unexplained loss and A0 shows stable token heterogeneity.

## Exploratory knobs NOT YET runnable in this patch

- **A0** conditional per-token L0R1/L1R1 importance, span/categorical
  importance and held-out predictability; prior Huginn local restoration
  was unstable; accept a negative result.
- **Q0** INT8/INT4 one-plane quantization; 2-bit/4-bit residuals with
  K/V-asymmetric precision, per-token vs global bit allocation;
  benchmark against ResidualQuant at identical metadata+bytes.
- **D0** Dory1.0-4B-BF16 R{1,2,4} native per_loop/last_loop and
  SWA vs global-attention layer breakdown. Do NOT load Dory through
  current Ouro/Huginn-only reference engine. Its native vLLM implementation
  is a separate codebase and needs an attention/memory parity audit.
  Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
- **T0** Ouro Thinking as a distinct trained checkpoint, not a runtime
  toggle; LongBench/RULER, math/coding, prompt vs generated reasoning.
  Standardize Full-output length buckets and query templates before
  interpreting "task sensitivity."
- **X0** dynamic-depth recorded replay with official native exits,
  referenced from `NATIVE_EXIT_REPLAY_PLAN.md`. Variable-depth runtime
  has NOT yet been measured in this patch; maintain fixed-R reference
  as independent quality ground truth.

## H200 × 8 execution instructions

**IMPORTANT:** no H200 job was launched in this GitHub commit session.
The user has eight H200s available; verify idle GPU IDs and local
authorization *before* adding `--execute`. A worker is isolated to
one CUDA device; no TP, NCCL or background processes are invoked.

From a checkout of this branch, `cd kv_policy`:

```bash
# CPU/code audit. Model-parity tests can download pinned checkpoint configs.
python -m pytest -q tests/test_knob_budget.py
python -m pytest -q tests/test_policy_semantics.py

# Prepare jobs, prompts and nominal logical memory sheets; NO GPU execution.
python scripts/kv_knob_sweep.py --phase depth --task nll --sample pilot \
  --run-dir results/knob_depth_nll_pilot
python scripts/kv_knob_sweep.py --phase postlast --task nll --sample pilot \
  --run-dir results/knob_postlast_nll_pilot
python scripts/kv_knob_sweep.py --phase oneplane --task nll --sample pilot \
  --run-dir results/knob_oneplane_nll_pilot
python scripts/kv_knob_sweep.py --phase factored --task nll --sample pilot \
  --run-dir results/knob_factored_nll_pilot

# Explicitly authorize actual H200 work for a prepared run.
python scripts/kv_knob_sweep.py --phase depth --task nll --sample pilot \
  --run-dir results/knob_depth_nll_pilot \
  --execute --gpu-ids 0,1,2,3,4,5,6,7

# Then evaluate generation separately, with a new directory and unchanged
# sample/model/policy identities. `--models` can restrict to GO conditions:
python scripts/kv_knob_sweep.py --phase depth --task gen --sample pilot \
  --run-dir results/knob_depth_gen_pilot \
  --models huginn-0125-r8,huginn-0125-r16,huginn-0125

# Repeat selected cells at main=200; use unique directories by phase,
# task, B and source revision.
# Safe B sensitivity check (do not launch larger B without HBM preflight):
python scripts/kv_knob_sweep.py --phase postlast --task nll --sample pilot \
  --batch-size-gsm8k 1 --batch-size-math500 1 \
  --run-dir results/knob_postlast_b1_nll
```

Use default local batch GSM8K B=8, MATH-500 B=4 for continuity;
B1 and B4 controls are important for numerical/noise comparisons.
B16 is a conditional stress knob only after peak-memory safety review.
Each run binds one model to one H200 per worker; the eight devices
process independent jobs, not a single tensor-parallel model.
Do not kill or modify unrelated processes.

The launcher refuses run-directory reuse when the matrix, batch sizes
or source git SHA changes. Workers use atomic lock files; a failed
worker's claim must be audited before an explicit retry. Raw and full
baseline results are never silently overwritten.

Artifacts:
`jobs.json`, `prompts/*.jsonl`, `run_manifest.json`,
`knob_manifest.json`, optional `banks.json`,
`analysis/nominal_logical_kv_budget.csv`,
`raw/*.jsonl`, `completeness.json`,
`analysis/screen_summary.csv`, `analysis/paired_vs_full_strict.csv`,
`logs/*` and `errors/*.json`.

## Decision tree for next agent iteration (MUST NOT silently launch)

1. **B0 positive:** a viable lower-R Huginn model has a reproducible
   quality gap FULL > LAST. Do within-R bank protection/codec and depth-
   conditioned error propagation before implementing an optimizer.
   If lower-R FULL collapses or LAST~=FULL, mark B0 NO-GO.
2. **B0 negative, M0 positive:** LAST+token masking saves >30% of LAST's
   *logical active histories* at credible quality. Promote 2–3 strongest
   cells to a compact implementation and run actual measured memory/BS/TPOT.
   If masked quality collapses, measure prompt-only/decoding-only
   eviction and R16 before abandoning token eviction.
3. **P0 positive:** a one-plane blend/average gives reproducibly
   better accuracy than LAST at identical one-plane storage. A′: tune
   alpha_K/alpha_V separately on disjoint calibration, test held-out.
   A″: test low-bit residual versus ResidualQuant. Report the cost of
   calculating the intermediate blend.
4. **J0 positive:** factored pruning+bank protection has nonadditive loss.
   Conditional A0: probe per-token protected-bank oracle on fixed FULL
   prefixes. If token heterogeneity is stable, implement a true joint
   upper bound and practical selector. If joint upper bound adds no
   held-out quality at matched bytes, NO-GO JOINT, even if the
   interaction score is nonzero.
5. **All fail:** run Dory native and verified Thinking/long-context
   checkpoints, plus low-precision LAST. **Do not manufacture a
   winning baseline by selecting only favorable tasks.**

### Go/No-Go measurement rubric

- Technical GO: independent policy reference / B1 parity, no future
  KV reads, Full/strict numerical control, no undisclosed parsing/length
  cap failures, all jobs complete. Record BF16 batch-shape instability.
- Research GO: repeated held-out paired gain at a controlled total KV
  byte budget, not only a larger percentage of FULL savings.
  Compare NLL AND free-generation Accuracy (teacher forcing may badly
  overstate autoregressive recovery). Include at least 3 random seeds
  for random policies and report negative cells.
- Systems GO: physically packed or evicted cache, measured saved
  HBM, and throughput/batch capacity improvements beyond LAST.
  Current reference WindowKV uses a full-sized broadcast store and
  MUST NOT be used as systems evidence.

## Explicit scope/safety

The root AGENTS.md focuses on an unrelated Phase-04R CPU-only
scheduling audit and forbids its GPU runs. Do not modify Phase-04R
source/traces or resurrect blocked predictor workflows. This is a new
user-requested `kv_policy` experiment track with separately authorized
H200 resources; running the new track still requires the host operator
to opt in with `--execute`. Never modify the FERRY/mgo repository.

Prior results or a successful `git push` alone do not constitute a
completed H200 run. After completion, push analyzed receipts and a
`DECISION.md` with explicit B0/M0/P0/J0 gates.
