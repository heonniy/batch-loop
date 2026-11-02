# Agent handoff — investigate Query-conditioned KV sensitivity; do not force a positive result

This handoff was requested on 2026-11-02. Read
QUERY_CONDITIONED_RECURRENT_KV_PLAN.md **first** and examine the actual
v6 results and repo APIs before editing. The user wants the **coding agent
to take research initiative** after controlled tests, not to blindly
implement one untested scoring formula.

## Research objective

Find out whether **an earlier Context KV's effect on the later
Instruction/Question query's repeated computation** (not the source
token's own normal trajectory) explains future decode-time eviction
damage beyond strong conventional features. Extend to **whether the
identity of the evicted Context KV changes actual dynamic exit depth**
for Huginn and Ouro.

Do NOT assume that such a phenomenon exists. A clear, adequately
controlled negative result is acceptable.

## Scope

- Models: pinned Huginn-0125 (full depth 32), Ouro-2.6B (full depth 4).
  Do not introduce Ouro-1.4B until justified by a diagnostic.
- Clean FULL-depth prefill for all prompts; source eviction from
  generated-token reads only; frozen same-prefix fixed decode first.
- Datasets: Controlled Retrieval and HotpotQA-Distractor custom
  excerpt, with exact Context -> Instruction/Question role offsets.
  Reuse frozen dataset/revision/UID infrastructure when available.
- Two distinct comparisons: (a) prompt Query-conditioned
  counterfactual sensitivity (offline oracle) versus Source self
  trajectory; (b) genuinely post-prefill future eviction utility.
- Dynamic exits: paired FULL-LIVE, EVICT-LIVE, FULL-REPLAY,
  EVICT-REPLAY, FULL-FIXED, EVICT-FIXED on the identical token sequence.
  Ouro Q-exit and Huginn cosine are different criteria. Record native
  dtype and diagnostic FP32 cosine separately when relevant.

## Engineering discretion (explicitly encouraged)

You own the implementation choice. First audit which reusable portions
of kvp/role_sensitivity.py, kvp/depth_eviction_probe.py,
kvp/loopzip_persistent_eviction.py, and kvp/loopzip_cached_eval.py
actually implement the required semantics. You MAY:

- add a clean-prefix query replay hook rather than altering old APIs;
- add better local Value/Attention Output controls if discovered;
- choose a cheaper exact-parity scheme before scaling up;
- run a separate, fixed 4- or 8-Context-token-window experiment
  if one-token BF16 effects are too small;
- look for a stronger alternative query signal (e.g. attention-output
  correction direction, loop-specific query sensitivity or exit-gate
  margin), provided it is scientifically distinct and measured;
- expand a small mechanism pilot based on **written** observations;
- stop the mechanism claim if only lexical/value/evidence effects
  explain the data, or if official exits rarely trigger.

**Do not select source pairs based on their future damage to produce
a compelling plot.** In MAIN, candidates/pairs are frozen using ONLY
clean prefill Attention, source position and query role. Keep
evidence-labeled oracle controls and positive-example plots separately.

## Start here: what to implement and validate first

1. Review v6 V6_PILOT_ASSESSMENT and Stage C controls; document what is
   already answered (single-loop KL near BF16 floor, Ouro late-loop
   dominant, source-trajectory feature weak for Huginn).
2. Write a small causal diagram/dataflow: source i -> later clean q,
   query-local mask at loop s -> q trajectory (DIAGNOSTIC), and separately
   untouched prompt -> post-prefill source eviction -> generated
   query trajectory/quality/depth (OUTCOME).
3. Build or reuse a deterministic candidate sampler and query-anchor
   selector. Test that swapping gold answer/evidence annotations does
   NOT change MAIN selection. Ensure no future decode features appear
   in rankings.
4. Implement CPU semantics tests and 1-checkpoint smoke parity for both
   Huginn/Ouro. No-mask and FULL replay should match; check BF16 noise;
   only selected source, selected query and selected loop may be masked
   for the one-loop probe. Confirm repeated probes do not mutate prefix
   history or affect first post-prefill generated-token prediction.
5. Make ONE readable per-prompt report: matching pairs, attention and
   local output perturbation, h_q(r) errors and Coda-output changes,
   future fixed NLL/KL, and depth shifts in the dynamic arm. Print
   denominator, missingness, and tiny-effect flags.
6. On a 5-question/model smoke decide whether signals are resolvable;
   on 20/model/dataset pilot compare Source-only, Attention/Value/local
   output and Query-conditioned features **before** proposing policy.
   A held-out 100/model/dataset expansion is optional, based on gate
   results and compute profile.
7. Test native dynamic depth with clean-only selected thresholds and
   forced FULL trace replay. Distinguish exit events from actual
   output quality; report depth histogram/exit saturation and
   percent of instances with depth change. Include persistent one-KV
   eviction and optional matched-width windows, with separate labels.

## High-priority science controls

- Source-side trajectory: a baseline, never the flagship feature.
- Clean Query changes alone: a baseline, not causal per-source effect.
- Query-conditioned masked recomputation: **expensive oracle**; show
  a realistic one-pass surrogate separately if possible.
- Single-loop versus all-loop query mask: distinguish downstream
  propagation from repeated error injection.
- Local Attention Output perturbation and Value: stronger controls than
  mean attention. Value content alone may explain damage.
- Clean-history versus accumulated generated KV: distinguish within
  current query from past generated-token-history accumulation.
- Attention/position/content-role matched examples, plus evidence
  annotation-only ablations; prompt-level CIs and early/late decode.
- Huginn prelude/coda versus recurrent-only mask; don't ascribe all
  all-layer harm to recurrence.
- Ouro's R4 last-loop dominance and Huginn's R32 loops: a simple
  first/last-loop control is essential if any loop-specific claim emerges.
- Actual fixed and live exits; do not treat recorded replay as live
  computation or use full-final states to choose an online exit.
- If Huginn output changes are below BF16 re-run floor, do not
  report apparent amplification from unstable ratios.

## Scientific decision branches (the agent should use judgment)

IF source pairs have different Y but q-conditioned metrics add nothing
after local output/lexical controls -> generic Context importance.
IF query-local masks have measurable, source-specific propagated error
above numerical floor -> investigate mechanism with q-specific anchors,
loop s and matched local perturbation, but keep online utility separate.
IF query diagnostic predicts Y on a dev cohort but no inexpensive
clean-prefill surrogate does -> describe an expensive diagnostic/oracle,
NOT a deployed selector.
IF EVICT-LIVE changes depth but EVICT-REPLAY quality is similar -> depth
response is interesting; test exit margin, gate stability and whether
depth adaptation helps **without** making online-oracle assumptions.
IF one model has no signal, report asymmetry; do not silently remove it.
IF one-token evictions are too weak, use a NEW frozen equal-width window
arm rather than changing the existing experiment's budget.

## Output and provenance expected from the coding agent

- A new source branch (already created), with tested code and
  **separate commits for implementation and produced GPU evidence**.
- A design journal under a new v7 path explaining: hypothesis,
  protocol/UIDs, changes made after observing which pilot finding,
  why the follow-up is informative, and what alternative explanation
  it rules out. Distinguish exploratory vs preregistered results.
- A compact per-question/source/anchor/loop table, matched pairs,
  clean/evicted future-NLL/stepwise KL, depth histograms,
  live/replay/fixed comparison and bootstrap uncertainty.
- A review file that flags failed or unrun jobs and explicit
  go/no-go status for recurrence-specific feature usefulness.
- Exact source SHA, checkpoints, datasets, commands, workload counts,
  peak memory, validation outcomes and *no unsupported HBM claim*.

No GitHub authoring session can itself validate H200 checkpoints.
Do not claim test or performance success until the execution agent
actually runs them and publishes receipts.

## Suggested work cadence (not a rigid implementation recipe)

- Initial commit: adapter/probe + CPU tests + a bounded dry-run parser.
- First GPU receipt: both-model full/replay/no-mask parity + 5-example
  causal sanity on clean Query.
- Second GPU receipt: fixed-depth query-feature versus future damage.
- Third GPU receipt: paired actual dynamic depth, only if exit checks
  are informative.
- Optional follow-up: independently held-out validation and equal-budget
  candidate policy.

**The agent is free to change ordering for a concrete technical reason,
but must document why and retain valid comparable baselines.**
