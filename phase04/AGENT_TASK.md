# Phase 04 — Executor task: validate, explain depth, then predict

This is a task specification. None of the GPU tests or predictor results in this document is claimed to have been executed by the planning commit.

Read:
1. `phase04/IMPLEMENTATION_AUDIT.md`
2. `phase04/DEPTH_PATTERN_PREDICTOR_PLAN.md`
3. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md`

## First action

Inspect the worktree and running jobs without interrupting them. Preserve every previous result root. Record the starting commit and imported model-source hashes. Resolve GPU authorization before model execution: the old plan specifies GPU7/Huginn and GPU8/Ouro, whereas Phase-3 code hard-codes GPU7/Huginn and GPU6/Ouro. Do not infer permission from that contradictory code or invent a user correction. Require an explicit authorized physical-index/UUID mapping. Source review, coding, and CPU tests can proceed while GPU authorization is unresolved.

## Work package 1 — Repair the measurement foundation

Implement a bounded validation harness and fixes for audit findings F1-F6 before expensive new collection. Keep changes reviewable and publish a source commit before executing it.

Required deliverables:
- an explicit, shared exit-policy contract for each model;
- per-request/token stochastic-state identity, including Huginn prefill and configured loop noise;
- an independent same-policy per-request reference and compact live-exit comparison;
- debug/profile and low-instrumentation benchmark modes with bounded event/receipt storage;
- effective-tokenized-input split and consumed-artifact hash checks;
- validation-only candidate selection and new sealed confirmation cohorts;
- fail-closed GPU mapping and unsupported-config checks.

Run existing CPU regression tests and add tests for the gaps. Existing command (not a claim of execution):

```bash
python -m unittest \
  phase02_runtime.test_runtime \
  phase02_runtime.test_oracle_ric \
  phase02_runtime.test_summarize_oracle_physical \
  phase03_predictor.test_tournament \
  phase03_predictor.test_full_tournament
```

The exact new module/CLI names are implementation choices; document runnable commands after creating them. Do not publish a planned command as an already existing executable.

With authorized GPUs, run Gate A's small real-model state/KV/exit tests first, then remeasure clean FIFO/Oracle anchors. Report PASS/FAIL/BLOCKED independently for each model and each check. Same-policy logit/hidden/gate parity is correctness validation, not a native-quality comparison. Preserve near-threshold numerical differences and first-divergence evidence; neither force them away nor excuse unexplained errors as BF16 noise.

## Work package 2 — Collect data that can explain predictability

Only after that model lane passes Gate A:
- begin with the train256/validation128 short-prompt pilot;
- collect raw previous exit hidden, current known token embedding, history/trajectory, Huginn Prelude and actual h0, and separate post-execution labels;
- store realized stochastic-state checksums and tokenized prompt/split/policy hashes;
- analyze token priors, within-token contextual variation, within-request temporal variation, within-step ranking, initialization sensitivity and projection loss;
- distinguish threshold crossings from Rmax caps.

Use the plan's bounded expansion and sealed test sizes. Do not re-use old test outcomes to select new winners. Never use a future token/depth/current-recurrence representation as a pre-loop predictor input. Offline calibration data must be disjoint from confirmation requests.

## Work package 3 — Prior + hidden predictor and physical confirmation

Compare P0-P7 from the plan on the same validated data. First keep fixed-B score sorting common to all predictors. In particular compare:
- offline smoothed token/policy prior;
- prior plus causal history;
- full previous-hidden/current-embedding contextual probes;
- token-prior plus context residual distribution head.

Then compare mean-score ordering and tail-risk/group-max construction on identical predicted distributions. Use validation to choose two finalists and freeze their checkpoints/configurations. Open fresh confirmation data only after freezing. Include negative candidates and three-seed validation for the top two families.

Physical comparisons must charge feature maintenance, predictor inference, ordering and data movement. Use the same staged FIFO and corrected Oracle. Keep replayed-depth/input systems timings separate from live-exit verification and optional live generation. Report prefill, decode-only wall time and genuine full-workload BCT separately. Do not label a frozen-depth replay as native/live generation.

## Scope locks

- No refill, no survivor merging across started microbatches, no after-r1/r2 regrouping.
- Predictor controls grouping only; actual exit rule and depth budget stay unchanged.
- Ouro last-exited KV; Huginn shared physical-layer KV.
- Previous hidden is a predictor feature, not a model warm-start change.
- Do not implement independent boundary batch sizing or mixed prefill/decode as a way to mask weak predictor results; that is a later research decision.
- No quality/perplexity/GSM8K benchmark or base-LM fine-tuning.
- No changes to old results, no GPU reassignment without authorization, no indefinite background soak.

## Publication and stopping

Use new roots under `phase04/results/<timestamp>_<source_sha>/`. Source/plan, implementation, measured results and review are separate commits. Follow the existing five-axis review protocol and retain failed/blocked runs.

If Gate A fails, publish the diagnosis and stop that model's expensive collection. If the bounded predictor search yields weak recovery, publish the best measured candidate as weak; do not alter test sets, thresholds or the no-refill problem to produce a positive number.

The final handoff must state:
1. What was actually wrong versus merely unverified, and which fixes were GPU-tested.
2. Whether per-request exits/KV/state remain consistent across B and ordering.
3. Which pre-loop information explains useful within-ready-pool depth differences.
4. Which predictor wins on untouched confirmation requests and its actual overhead.
5. FIFO/Predictor/Oracle RIC, decode wall, throughput and full-workload BCT with exact protocol labels.
