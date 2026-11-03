# V10 — Architecture-specific sharing × Context KV eviction × Query recurrent trajectory / exit

Date: 2026-11-03
Branch: research/full-vs-architecture-sharing-trajectory-exit-20261103
Parent: research/kv-ffn-gate-depth-causal-20261103 (v9).
**Status: CODE/PLAN COMMITTED; NO NEW H200 EXECUTION OR RESULTS CLAIMED.**

## What was requested

Do not continue FFN gate restoration (v9 NO-GO). Instead, compare **the SAME Context
KV eviction in a full-loop cache versus a model-appropriate shared cache**, and
trace the *later generated Query's own recurrent hidden states, native early-exit
statistics, depth decisions and actual answer quality*.

Primary models:
- Huginn-0125: R=32, 4 physical recurrent core layers. Shared condition = **LAST**
  historical recurrent KV bank for every generated-query loop.
- Ouro-2.6B: R=4, recurrent transformer. Shared condition = **FIRST+LAST**.
  At query loop 1 read completed source's loop-1 KV, at query loops 2–4
  read completed source's terminal KV. Two KV planes are conceptually kept.
- Ouro-1.4B: **optional follow-up, not included in the primary two-model claim**.
  Important if Ouro2.6B is robust but 1.4B repeats its v8b generated-KV failure.

Each model also has the true **FULL** history control, in which query loop r
reads a completed source's corresponding loop-r KV. This is a causal comparison
of *sharing policies*, not a same-byte-budget performance comparison.

An ordinary prelude/coda cache (Huginn) is neither compressed nor
replaced. Both sharing policies redirect historical recurrent KV **reads
only** after fully completing a normal full-depth token-major prefill.
Source KV remains in the full-sized reference bank. We do not claim physical
HBM savings, TPOT improvements, or a deployable dynamic schedule.

### Exact first/last semantics (potential source of ambiguity)

V8 had a generic nearest-plane first+last mapping for physical plane-budget
sweeps. The V10 PRIMARY Ouro condition instead protects the FIRST source
plane specifically for the consuming query's first recurrent step:
  source r=0 bank for query r=0; source r=R-1 bank for query r=1..R-1.
This resembles a "first-loop protection + last-loop sharing" policy.
It is not "nearest of the two" for r=1 (that would still read the first bank).
A *nearest* alternative may be run as a separately labeled ablation; do not
mix its result into the primary first-for-r1 cohort. If the model exits
at depth d<R, the unused terminal bank is filled with its last computed
KV by the existing DepthOracleKV.finish(); therefore dynamic source
depth and query depth are well-defined and no-refill is respected.

## Main questions and hypotheses (all directions are open)

H1: Does the SAME Context KV eviction change the generated Query's loop-wise
trajectory differently under Full vs the architecture-specific shared KV?
  E^(policy)_(q,i)(r) =
     ||h^(policy,evict_i)_(q,r) - h^(policy,keep)_(q,r)||_2 /
     (||h^(policy,keep)_(q,r)||_2 + eps).
The observer checkpoint is exactly the v6 model-native checkpoint (Huginn
before Coda; Ouro after its recurrent model.norm). Large differences can be
numeric; compare to no-mask repeat floor.

H2: Does the sharing policy interact with eviction damage at matched depth?
  interaction_i(d) =
    [NLL_(shared,evict_i,d) - NLL_(shared,keep,d)] -
    [NLL_(full,evict_i,d) - NLL_(full,keep,d)].
Positive => sharing is more sensitive to eviction; negative => less sensitive.
Control each policy's baseline quality independently, otherwise sharing
damage can be mistaken for evaporation damage. Along with final native R,
test Huginn d=8/16/24/32 and Ouro d=2/3/4. These are **forced fixed-depth**
evaluations, not claims of genuine early exit; Huginn includes its Coda
at the evaluated depth. Use the *same frozen continuation* in all arms.

H3: Does source eviction change the native would-exit decision even at a
FULL fixed-depth recurrence, where no early termination has occurred yet?
- Huginn: d_r = 1 - cos(h_r,h_(r-1)); early exit at first d_r<threshold
  after min-depth 2. Record FP32 and model-dtype diagnostics separately;
  the FP32 arithmetic is a research diagnostic, not official BF16 serving.
- Ouro: learned early-exit gate -> cumulative Q-exit probability; stop at
  first cumulative>=threshold after min-depth 2.
Full and Evicted fixed-depth trajectories provide the *statistic* and
hypothetical would-exit crossing with common histories. This is distinct
from the actual Live exit executed below. An exit flip alone is not output harm.

H4: With *actual* native Live Exit enabled, does the sharing × eviction
interaction change sum of executed loops, future NLL and free-generation
correctness? Separate the threshold's effect from eviction:
  Full-fixed, Full-live, Full-replay (exact FULL-LIVE trace),
  Evict-fixed, Evict-live, Evict-replay (FULL-LIVE trace).
The LIVE run MUST really re-evaluate exit on every generated query.
Live-vs-replay changes both generated-KV history and depth and must not
be labeled a pure mediation effect. Save exact per-generated-query depths
and per-query output distribution divergence where available.

## Workloads and pairing protocol

- Pinned Controlled Retrieval and HotpotQA-Distractor evidence excerpts:
  same v5 frozen prompt UIDs and target IDs across arms within a model.
- Main: four equally sized contiguous 4-token Context windows, frozen
  by position/hashing before looking at eviction outcomes. Additional
  relevant-evidence and irrelevant-distractor windows are annotated
  *oracle diagnostic cohorts*, not privileged features at prediction time.
  Main and diagnostic cohorts MUST NOT be pooled without labels.
- All prompts: FULL-depth normal prefill, mask only after it completes;
  all generated tokens replay the identical frozen gold continuation for
  NLL/KL and trajectory analysis. First generated-token prediction is
  invariant under post-prefill masking and excluded from NLL effect.
- Actual correctness: native real free greedy generation and answer
  extractor at fixed R and at both LIVE exit thresholds, for both KV
  policies and Keep/Evict. Report token sequence differences separately
  from exact-match correctness; track EOS/truncation and wrong->right.
  HotpotQA's low baseline EM means NLL/KL are primary there.
- Initial CPU tests and one prompt/model smoke, then 3 CLEAN calibration
  prompts + 5 different test prompts/model-dataset. 20+ prompt expansion
  only after nontrivial, repeatable effects and compute profiling.
  The existing v7/v9 n=5 pilots are not independent holdout evidence
  for a newly discovered V10 hypothesis: use new UIDs if scaling.

## Exit thresholds — fix v9 saturation without outcome leakage

Estimate statistics on both **CLEAN FULL and CLEAN SHARED** fixed-depth
decodes on calibration UIDs. Choose one pair of thresholds per model from
their combined gate-statistic distributions, targeting ~25% and ~60%
early exits; **use exactly the same two thresholds in Full and Shared**.
Q-exit uses a clean grid 0.2–0.995. Huginn includes much smaller FP32
distance thresholds and clean-data gate quantiles below the v9 minimum
1e-4, which prematurely saturated in v9. If still saturated, flag it
and do not tune based on outcome or failure cases.
Record calibration histograms and realized FULL/SHARED live histograms;
thresholds are not directly comparable across models.

## Concrete observables to inspect

1. Plot E(q,r) for each sharing-policy/eviction pair on the same Query,
   and a second curve of within-run exit statistic along r.
   Report both absolute state distance and normalized E; values near
   BF16 floor are negative evidence for a sensitivity claim.
2. Plot \Delta NLL_eviction(d) at matched fixed d for Full and Shared;
   display interaction difference rather than comparing raw NLL
   between architectures/checkpoints.
3. Plot difference between fixed-depth would-exit and actual live
   per-query exit, and all associated true answer-change cases.
4. Use per-prompt grouping/bootstrap (different source windows on one
   prompt are correlated), and separate oracle evidence vs MAIN.
5. If there is a reproducible sharing × evaporation × recurrent-depth
   interaction, *then* test one-loop vs persistent mask and standard
   transformer layer-wise controls for loop-specific interpretation.
   A generic deep-network effect should not be called loop-specific.

## Execution starting point (requires allocated H200; no command has been run here)

  cd kv_policy
  python -m pytest -q tests/test_trajectory_sharing_policy.py

  python scripts/trajectory_sharing_sweep.py --prepare --model huginn-0125 \
    --dataset controlled_retrieval --run-dir results/v10_huginn_cr
  CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_sharing_sweep.py --calibrate \
    --model huginn-0125 --dataset controlled_retrieval \
    --run-dir results/v10_huginn_cr
  CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_sharing_sweep.py --execute \
    --model huginn-0125 --dataset controlled_retrieval \
    --run-dir results/v10_huginn_cr
  python scripts/trajectory_sharing_sweep.py --analyze \
    --model huginn-0125 --dataset controlled_retrieval \
    --run-dir results/v10_huginn_cr

Repeat on Ouro-2.6B in a distinct run dir and GPU, then HotpotQA.
Optional Ouro-1.4B in a third series. Do not interfere with existing GPU
processes or original v6/v7/v8/v9/FERRY outputs.

## Agent autonomy / scientific integrity

The execution agent is encouraged to audit the mapping, perform parity,
run a 1-prompt smoke, inspect informative errors, and adapt scheduling
or add **separately frozen** negative controls or larger windows.
Before each changed experiment, record: (clue, alternative explanation,
changed condition and source UIDs, what stays comparable, go/no-go).
Do not claim memory speedups from reference map reads. A null interaction
or no relation to quality is a valid result.

Existing material:
- kvp/query_conditioned_probe.py: PlaneMapProbeKV, FULL-prefill reference,
  causal post-prefill source mask, live/replay/fixed decode.
- kvp/loopzip_trajectory_recorder.py: native checkpoints and paired states.
- kvp/depth_eviction_probe.py: native Ouro Q-exit / Huginn cosine logic.
- scripts/ffn_gate_causal_sweep.py: v5 frozen prompt loading, source_windows,
  free answer scoring (NO v9 gate-restoration arms are run in V10).
- New: kvp/trajectory_sharing_policy.py,
  kvp/trajectory_sharing_probe.py,
  scripts/trajectory_sharing_sweep.py,
  tests/test_trajectory_sharing_policy.py.
