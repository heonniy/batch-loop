# Focused Two-Track Research Plan — 2026-10-29
## Only (A) Ouro token-specific Loop1 retention and (B) Huginn hard-but-usable regimes

**Status**: reproducible code and gates; no H200 output/quality claim yet.
**Research hardware available**: user-authorized H200 x8; workers can run
independently on CUDA ordinals 0..7 **only after idle/device checks**.
**This plan SUPERSEDES the priority of** `KNOB_EXPANSION_H200X8_PLAN.md`
for the new branch. Post-Last token eviction, one-plane mixtures, Dory,
standalone dynamic-depth replay and global inter×intra joint policies are
**ON HOLD**. Their code/history remain unchanged.

## What the 10/29 professor feedback means

Do not assume every token needs a first-loop cache (Ouro), or that
Huginn R32 offers useful Full/Last quality headroom. The two linked RQs are:

- **A / Ouro**: When Last+ALL first-recurrence banks stores *50% of FULL
  recurrent KV* and recovers substantial accuracy, can we choose *which
  historical tokens* additionally keep their first-recurrence KV to match
  the **50%-KV baseline accuracy** with **less than 50%-KV memory**?
  The 50% refers to MEMORY, not 50% task accuracy.
- **B / Huginn**: Is there a realistic, sufficiently accurate checkpoint
  and reasoning/context/depth regime where Last-sharing loses significant
  quality against same-R FULL, rather than the already near-lossless R32
  GSM8K/MATH500 setting?

### Frozen strict reference results (historical; not rerun in this commit)

GSM8K-200, MATH500-200, BF16/greedy, all token-major STRICT:
| Model/task | FULL 100% | LAST 25% | L0R1 bank 25–26% | Calibrated 25–27% | all first-recurrence banks 50% |
|---|---:|---:|---:|---:|---:|
| Ouro-1.4B GSM8K | .810 | .000 | .405 | .660 | **.720** |
| Ouro-1.4B MATH500 | .740 | .000 | .170 | .485 | **.610** |
| Ouro-2.6B GSM8K | .880 | .600 | .770 | .770 | **.845** |
| Ouro-2.6B MATH500 | .565 | .330 | .410 | .410 | **.500** |

Huginn-0125 R32 FULL .420/.145 vs LAST .405/.160 (GSM8K/MATH);
differences not significant. These results are a reason to seek a
different Huginn setting rather than overclaiming a new optimizer.

## A. O U R O — Token-selective first-recurrence KV

### Precisely what is protected

For a historical token i, physical attention layer l, current query loop r,
R=4, keep KV_i^(4) unconditionally; additionally keep KV_i^(1) if the
causally selected token i is protected. Query r=1 reads KV_i^(1) if
selected, else KV_i^(4). Query r=2,3,4 reads KV_i^(4) always.
Current token self-KV is still computed at its CURRENT recurrence;
the token makes its protection decision only after completing R loops.
Both prompt and decode tokens follow the exact same STRICT semantics.

**One decision per token across all physical layers in A0.**
This isolates the professor's question from layer-selection confounds.
Static layer-masks are controls, not a claimed token-adaptive method.

Let f be the fraction of stored token positions whose first-recurrence
banks are protected across ALL Ouro physical layers:
- LAST: Full-KV fraction = 0.25
- ALL-R1: Full-KV fraction = 0.50
- Token-selected: Full-KV fraction = 0.25 + 0.25 f
- **Incremental reduction relative to ALL-R1** = (0.50-(.25+.25f))/.50
  = (1-f)/2.

| Expected f | Full-KV fraction | Memory saved **vs ALL-R1** |
|---|---|---|
| 20% | 30% | 40% |
| 40% | 35% | 30% |
| 60% | 40% | 20% |
| 80% | 45% | 10% |
Hash selection rates above are *expected*, not exact quota. Use the
logged actual chosen token count, filtered for ended batch rows.
The physical reference allocation is full-sized and not an HBM saving.

### A0 controls and candidate token selectors

Required policies:
- `full/strict`, `last/strict`, `bankR1all/strict` (the **50% reference**);
- `bankcal/strict` and Layer-only top/prefix masks (`bankTopB35/B40`,
  `bankPrefixB35/B40`) from disjoint historical calibration;
- `tokL1hash20/40/60/80`: deterministic causal token-level Bernoulli
  selection; `tokL1hash0` = LAST exact negative control,
  `tokL1hash100` = all-R1 exact positive control;
- `tokL1prompt`, `tokL1decode`: preserve first-loop history only for
  prompt or generated tokens, as a low-cost semantic-region ablation;
- `tokL1delta50/100`: **heuristic only**. After the token completes R,
  compute relative first-vs-terminal K/V difference across physical
  layers, select if above a fixed threshold. It uses ONLY already
  computed K/V of the current token; no future attention, gold labels,
  or Full-model rollout. Threshold calibration/selection is a later
  task; record the selected fractions rather than claiming a fixed budget.

Run the same frozen 32+32 pilot then (if meaningful) 200+200.
Report generation Accuracy, teacher-forced token-weighted NLL,
post-EOS / repetition failures, prompt/generation protected rates,
quality vs ALL-R1 **paired by UID**, quality vs FULL/strict,
and actual LOGICAL fraction / saving vs ALL-R1.

### A0 → A' / A'' decision logic

- **A0-GO**: for *both* GSM8K and MATH500, a token policy achieves
  within 3 percentage points of ALL-R1 at <=35% FULL KV (>=30%
  incremental saving vs ALL-R1). A <=40% footprint with same quality
  is a secondary, weaker GO. Accuracy noninferiority is not proven
  by a point estimate: report paired 95% intervals/large-sample check.
- **A'-Token-selective necessary?** At matched bytes compare the best
  token policy against layer-only `bankTopB35/B40`, first-L layers,
  random bank and uniform-hash tokens. Only claim token-level selection
  as a method if it reproducibly improves Pareto over layer-only.
  One random seed is a pilot, not evidence of random-baseline superiority.
- **A''-If token hash/prompt/delta shows heterogeneity:** run disjoint
  train/calibration/holdout cohorts, compute optional costly
  per-token restore oracle on 16–32 items, test correlation with
  current-token KV delta/position/prompt-vs-decode and, if stable,
  implement a causal online selector with budget enforcement.
- **A''-If exact same two banks beat all token variants:** keep the
  simple layer-only recipe; mark token-adaptive hypothesis NO-GO.
- **A''-If only 45–50% achieves baseline quality:** consider low-bit
  recurrence-1 residuals or per-token asymmetric K/V **as a later new
  study**, not bundled into this experiment.
- Use both models. The prior `bankcal` was optimized under
  LAST-CONVERT, so it is a transfer baseline in STRICT, not an oracle.
- **Critical resource warning**: current token-loop KV uses full-sized
  broadcast buffers. It is valid for *quality and theoretical logical
  memory* only. Implement a compact packed KV on conditional success
  before saying "we saved GPU memory".

## B. H U G I N N — Find meaningful Full-vs-Last degradation

### B0: depth and actual-quality gate

Same frozen pretrained Huginn-0125 checkpoint, varying *test-time*
recurrence R = 8, 12, 16, 20, 24, 32. R=40/48/64 are
**conditional extrapolation controls**, not "native" depths.
Execute FULL/strict and LAST/strict at the same R, on
GSM8K and MATH500. For R32 reuse historical accuracy as a sanity
reference, but regenerate matched paired baselines when comparing
new stress prompts or batch shapes.

Measure: FULL accuracy, LAST accuracy, paired Δacc CI,
token-weighted ΔNLL, first differing generation token, cap hits,
full accuracy normalized to native R32. R8 might show large
sharing loss but also poor FULL quality; that is **not useful GO**.

Screening GO: at least one condition with
`A_full(R,task) >= 0.8 * A_full(R32,task)`, and
`A_full(R,task)-A_last(R,task) >= 0.05` accuracy (>=5pp)
with a paired CI that supports a real degradation.
The constants are screening thresholds, not intrinsic significance
tests. Consider model's native R32 quality ceiling before investing.

### B1: reasoning/context stress at selected depths

Same frozen GSM8K and MATH500 question IDs. New *prompt-style*
parameter, encoded into prompt files, job names, manifests and raw rows:
- `default`: original lm-eval prompt; **control**.
- `reason`: appends a step-by-step instruction to Huginn's system
  message; not a trained think-mode switch.
- `verify`: requests explicit intermediate checking.
- `distract512`: prepends >=512-token irrelevant passage to the
  last question. It is not a standard benchmark, but isolates
  extra context load without introducing new answer facts.

At R16/R24/R32 (one can restrict models with --models), compare
FULL vs LAST *within the same style* on both datasets. For longer
generation inspect output length/cap strata and Full prefix length.
Do NOT attribute a quality loss to "thinking" unless the model
actually has a separate trained thinking mode. Huginn's official
latent recurrence/adaptive compute does not by itself confer such
a switch (https://github.com/seal-rg/recurrent-pretraining).

### B1 → B'/B'' decision logic

- **B'-Positive depth/stress regime**: reproduce with 200 held-out
  questions, B1 and B8 shape controls, and repeated FP32 perturbation
  traces. Compare whether recurrent recovery saturates too early.
- **B''-Only R4/R8 weak and FULL also unusable**: do not sell it as
  memory-compression headroom. Try R16/24 stress first.
- **B''-No gap on existing GSM8K/MATH500:** a distinct, audited
  hard-dataset adapter for AIME-like math, multi-hop/RULER retrieval,
  or code reasoning is the next conditional task. No such adapter is
  implemented in this patch. Freeze task prompts and scorer first.
- **B''-Long-context distractor alone kills FULL**: label general
  context limitation, not sharing sensitivity.
- **B''-No positive usable Huginn regime** after candidate testing:
  mark Huginn architecture as negative generalization baseline,
  concentrate method effort on Ouro; do not invent a "hard" result.

## Run it on the user's H200 x8 host

This patch **does not start GPUs or run remote commands**.
Single-process workers claim independent jobs across 8 H200s;
there is no tensor parallel or inference CUDA availability here.
Inspect GPU ownership/idle state; do not kill unrelated jobs.

```bash
git fetch origin
git checkout research/ouro-tokenloop-huginn-hard-20261029
cd kv_policy

# CPU/no-checkpoint sanity
python -m pytest -q tests/test_token_loop.py tests/test_knob_budget.py \
  tests/test_knob_launcher.py tests/test_huginn_stress.py

# Native checkpoint tiny-oracle parity (requires transformers + HF access)
python -m pytest -q tests/test_policy_semantics.py -k tokloop

# Ouro: first run a narrow quality pilot, 32+32, NLL + generation
python scripts/kv_knob_sweep.py --phase ouro_token --task both --sample pilot \
  --run-dir results/focus_ouro_token_pilot \
  --policies full-strict,last-strict,bankR1all-strict,bankcal-strict,bankTopB35-strict,bankTopB40-strict,tokL1hash0-strict,tokL1hash40-strict,tokL1hash60-strict,tokL1hash100-strict,tokL1prompt-strict,tokL1decode-strict,tokL1delta50-strict

# Huginn: first depth pilot under original prompt, 32+32
python scripts/kv_knob_sweep.py --phase huginn_hard --task both --sample pilot \
  --models huginn-0125-r8,huginn-0125-r16,huginn-0125-r24,huginn-0125 \
  --run-dir results/focus_huginn_depth_pilot

# Then explicit --execute --gpu-ids 0,1,2,3,4,5,6,7 for the desired
# already-prepared run and EXACT same CLI args, with no commit changes.
# Do NOT automatically launch all remaining matrices.
```

For B1 (after deciding on viable R):
```bash
python scripts/kv_knob_sweep.py --phase huginn_hard --task gen \
  --sample pilot --models huginn-0125-r16,huginn-0125 \
  --prompt-style reason --run-dir results/focus_huginn_reason_pilot
python scripts/kv_knob_sweep.py --phase huginn_hard --task gen \
  --sample pilot --models huginn-0125-r16,huginn-0125 \
  --prompt-style distract512 --run-dir results/focus_huginn_distractor_pilot
```

After the pilot, freeze the **best few hypotheses** (not just the
winner without controls) and run 200 GSM8K + 200 MATH500 per model
with `--sample main` in new directories and declared code SHA.

### Required outputs

Automatic:
- `jobs.json`, `prompts/*.jsonl`, `run_manifest.json`,
  `knob_manifest.json`, `banks.json` and calibration provenance
  (Ouro controls), `raw/*.jsonl`, completion/errors/worker logs.
- `analysis/screen_summary.csv`,
  `analysis/paired_vs_full_strict.csv`,
  `analysis/sub50_paired_vs_loop1all.csv` for Ouro
  and `analysis/token_loop_logical_memory.csv`.
- `analysis/nominal_logical_kv_budget.csv` marks expected rates
  separately from logged realized fractions.

After GPU run, agent writes **two separate** outcome memos:
1. `analysis/OURO_TOKEN_DECISION.md`: how much additional memory
   versus 50%-KV Loop1-all, whether quality is noninferior in
   practical terms, token vs layer controls, runtime logical accounting.
2. `analysis/HUGINN_HARD_DECISION.md`: FULL viability, under what
   fixed R/prompt/length LAST drops, or explicitly NO usable gap.

Do not report token level as theoretically better than layer masks
until held-out gen results corroborate matched-byte superiority.
Do not report "measured HBM reduction" from this broadcast engine.
DO NOT modify the unrelated Phase04R predictor code or the FERRY repo.
