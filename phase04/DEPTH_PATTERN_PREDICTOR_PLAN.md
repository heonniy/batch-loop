# Phase 04 — Validated exit traces, depth patterns, and prior + hidden predictors

Date: 2026-09-17. Baseline reviewed: `a4a089811f53d4acf300074cfb56d27f5c4f1f8b`.
Status: **implementation/research plan, not completed experiments**.
Read `IMPLEMENTATION_AUDIT.md` and the existing experiment-review protocol first.

## Research contract

Claim to test: **In offline depth-adaptive Looped-LM inference, pre-recurrence information can identify tokens with compatible exit-depth behavior, improving no-refill microbatch composition and reducing recurrent tail/valley cost.**

The predictor never decides actual exits. Ouro keeps last-exited depth-indexed KV; Huginn keeps shared physical-layer KV. Freeze checkpoint, KV policy, initialization distribution, exit rule, threshold and maximum depth within each comparison. At the start of a token interval, form a batch; thereafter only remove exited rows. No refill, no cross-microbatch survivor repacking, no after-r1/r2 regrouping. Previous hidden is a predictor input, NOT a warm-start modification to model execution. Base LMs stay frozen.

Defer independent Prelude/Core/Coda batch-size optimization, mixed prefill/decode scheduling and dynamic initial batch sizing until predictor confirmation. No task-accuracy, GSM8K, perplexity, or native-versus-approximate quality benchmark. State/KV/gate/logit comparisons below are implementation correctness tests under the SAME selected semantics, not a quality study.

Main study axes are not interchangeable:
1. offline profiling/calibration learns reusable priors from TRAIN data;
2. contextual hidden features supply information available before the current recurrence;
3. a small predictor combines those signals for grouping.

## Notation and target definition

At decode interval t, x[i,t] is already sampled/known; processing it predicts x[i,t+1]. D[i,t] is the first exit chosen by the frozen execution policy, capped at Rmax. It is NOT a label for the minimum compute needed for a correct answer.

Record `exit_reason = threshold | rmax_cap`, the exit margin, and whether threshold also happened to pass at the cap. The generation index, absolute token position and physical padded column are different fields.

For fixed full microbatches of size B and fixed work W:
- RIC = sum over tokens and microbatches of maximum D in each microbatch.
- logical idle slots = B * RIC - W; slot utilization = W / (B * RIC).
- same-B depth-sorted Oracle minimizes this grouping RIC when N is divisible by B. It is not an absolute wall-time optimum over arbitrary runtimes.
- N=B is a no-composition-freedom control. B=1 is another control.
- for N not divisible by B, put the smallest partial group first in ascending sorted order (or use descending B-sized groups with a final partial group) for the max-sum Oracle; do not blindly reuse the full-group formula.

Primary metrics: RIC reduction versus FIFO, Oracle RIC recovery, decode wall reduction, tokens/s, and full-workload BCT where actually timed. Recovery is undefined when FIFO=Oracle; report NA, not 0 or infinity. Include absolute differences. Predicted RIC above FIFO gives negative recovery and must be retained.

# Block A — Validate and repair before collecting new targets

This block is a prerequisite, not an optional appendix. Parallelize independent CPU/source tasks, not unvalidated model runs.

## A1. Freeze exact execution semantics

Inspect and hash the ACTUALLY imported cached model source, not just the checkpoint directory name. Record model/tokenizer revision, config, PyTorch/Transformers/CUDA, attention backend, layer counts, head counts, rope settings, inference dtype, test-time noise config and the exit function. Check the audit's named source functions against the pinned cached implementation.

Ouro: normalized recurrent state; raw gate and sigmoid; per-row cumulative remaining probability; >= q comparator; current effective Rmin=1 and Rmax=4; final-cap behavior. Do not change to Rmin=2 simply to match another paper. Huginn: raw versus normalized latent conventions; FP32 relative L2 with epsilon and strict < threshold; current Rmax=16; prelude/core/coda indices; random initialization and any loop noise. Normalization and precision differences must be named, not hidden by a loose tolerance.

Implement a shared exit-policy component used by collector, compact runtime and independent reference. Distinguish duplicated gate computation for diagnostics from actual stateful gate updates: the probability accumulator must update exactly once per row/recurrence.

## A2. Make stochastic state independent of batching

Use canonical request identities and token positions to bind the sampled h0. Isolate model initialization, projection construction, predictor training, token sampling and optional recurrent noise RNG streams. Keep the original stochastic distribution. Record an h0 checksum per request/token in debug receipts. Reference/compact runs must receive the same realized states, regardless of B, row permutation, number of prefilling peers or preceding RNG calls.

For frozen-replay experiments, explicit stored states are acceptable when labeled as replay; do not use this to hide live-state generation cost. The deployable path must generate the same identity-keyed state and include the cost. For initial correctness tests, explicit shared tensors are simpler than assuming equal seeds suffice. Check prefill as well as decode.

## A3. Independent-reference validation matrix

Use an uncomplicated per-request reference under the selected KV/exit policy and compare against physical compaction. An independent masked full-B reference may additionally freeze already-exited states and emulate the chosen KV fills. Do not compare early-exit copied KVs to genuinely computed full-depth KVs and call that a compaction error.

| Test | Required cases | Evidence |
| --- | --- | --- |
| Exit arithmetic | Ouro known gate vectors, state reset, reversed row IDs, all/none exit, threshold ties, cap; Huginn independent per-row changes, near-zero norm, ties, nonfinite inputs | per-row exit masks/depths, exact toy invariants |
| Physical survivor removal | forced [1,2,4,2] -> widths [4,3,1,1], then independently generated live masks | no exited row reappears; actual tensor width receipts |
| KV writes and reads | Ouro every (row,token,layer,depth) fill; Huginn overwritten current position with immutable prior history | sentinel tensors AND actual attention-visible K/V checks |
| Index/padding invariance | B=1,4,8,16; spot-check 32,64; nonidentity rows; heterogeneous short prompts; N=B+3 final partial batch | same canonical request results after unpermuting |
| Stateful multi-token path | 1,8,32 decode tokens; one full-depth and one shallow setting per model | first divergence, logits/hidden errors, gate margin, KV checksums |
| Boundary correctness | prompt-only last logits, first generated input x_t, subsequent KV position, history length increases once | no duplicate/skipped token or off-by-one RoPE position |

Exact equality is required for integer IDs, explicit copied tensors, KV slot mapping, toy gates and depth results under replayed decisions. Real floating-point parity requires tolerances chosen BEFORE tests and logging both absolute/relative errors and near-threshold cases. Do not simply require bitwise BF16 outputs at all batch shapes, and do not dismiss unexplained systematic drift as numerical noise. Near-threshold disagreements get isolated high-precision probes and a documented policy; no test-driven threshold adjustment.

GPU anchor: 32 distinct prompts x 8 tokens, mixed lengths, then 64 prompts x 32 tokens after the small test passes. B64 capacity failures remain failures, not silently reduced B. Check correctness before scaling prompt length.

## A4. Remove diagnostic distortion from scored timing

Split `debug/profile` from `benchmark` operation. Turn off per-KV CUDA-event creation and unbounded row-copy receipts by default; bound/drain them in profiling. Clear phase accounting after prefill/staging. Remove per-operation synchronization from scored runs except required dependency management; retain synchronization at timing endpoints. Preserve useful debug assertions in separate runs. Predictor-required feature/trajectory gathering is real cost and must remain timed.

Separate core elapsed time, gate, KV fill/gather, compaction, feature extraction/maintenance, predictor, ordering and boundary costs. Avoid summing nested/overlapping GPU times as independent costs. Sorting inside the Oracle wall timer must not misleadingly appear as a zero-cost decomposition field. Preallocate where validated; preserve the existing KV semantics. Request-batch chunking of prefill can reduce peak HBM; token-axis chunking under shared KV is a semantic change until tested and is not authorized as an automatic fallback.

Re-establish profile-free FIFO/Oracle at primary N256/B32, 32-token anchors before applying predictors. Old prototype speedups are not guaranteed to survive instrumentation changes.

## A5. Data, selection, and ownership checks

Create new canonical IDs and split by request/conversation/document family. Hash actual tokenized, templated, capped prompt inputs; reject cross-split duplicates even when raw strings differ. Check all consumed feature, label, projection and predictor checkpoint artifacts. Save source SHA before execution; dirty source requires explicit hashes and cannot be silently labeled as a committed implementation.

Old primary test sets have been used to choose candidates. Treat them as development evidence. Use validation to select finalists and freeze all choices before a fresh confirmation test. GPU map conflicts in old code must not be silently resolved: require an explicit authorized model -> physical-index -> UUID record and fail closed if unavailable. Use no other GPU, no background soak, and do not interrupt existing work.

### Gate A decision

Publish `runtime_validation.json`, row-level mismatch receipts, source/config hashes, debug-vs-clean timings and `IMPLEMENTATION_REVIEW.md`. Status is PASS, FAIL or BLOCKED per test and model. Continue a model lane to Block B only after its high-priority semantic/reproducibility checks pass. Do not label unexecuted tests PASS. A failure in one model does not authorize switching models, devices or policies without a record.

# Block B — Analyze where pre-loop predictability comes from

## B1. Bounded workload plan

Primary regimes: Huginn threshold 0.12/shared/Rmax16; Ouro q0.2/last-exited/Rmax4. Secondary: Huginn 0.26, Ouro q0.4. Hold each policy fixed. Other historical thresholds may be analyzed from compatible saved data, but do not select a new favorable threshold using confirmation results.

Anchor collection: Alpaca, reproducibility prompt cap64, decode32. Clearly label this short-prompt anchor, not unrestricted workload evidence. Target 1024 TRAIN + 512 VALIDATION + 512 sealed TEST requests per model; partition validation into 256 model/selection and 256 probability-calibration/physical-selection requests, with family grouping respected. Test consists of two disjoint N256 pools. First run a train256/validation128 pilot without opening TEST; extend training through 128/256/512/1024 learning-curve prefixes. No unbounded training/collection loop.

Generalization: ShareGPT with fresh conversation-disjoint prompts, cap512 and decode64; begin with memory validation. Freeze the winning primary predictor before the transfer test. Refitting on a separate ShareGPT training split is a separate result, not frozen transfer. Set the exact permitted pool sizes/caps in a manifest before scoring. If N256 cannot fit, report the regime blocked and benchmark an explicitly different smaller-N regime; do not call it the same Oracle opportunity. Request-level prefill chunking must use the validated stochastic identity and KV semantics.

If budget requires a smaller first packet, publish that finite subset and its missing comparisons. Never manufacture diversity, omit difficult requests, or substitute repeated prompts for unique sample counts.

## B2. Canonical feature and label schema

Save before current recurrence:
- canonical request ID for bookkeeping ONLY (never a predictive feature), dataset/family IDs for grouping/splits;
- input token x[i,t], tokenizer revision, generation index, absolute position, valid context length, prompt length;
- previous observed exit depth and previous gate/convergence trajectory with explicit masks;
- previous exit hidden in raw FP16/BF16, current token embedding, and Huginn current Prelude hidden and initial h0;
- previous logits-derived surprisal/entropy/margin only for the already-known current input token;
- a documented token-0 fallback; do not invent a previous adaptive depth for prefill. Prefill last hidden may be available as a separate feature with its different provenance.

Save separate labels AFTER execution: actual D[i,t], exit reason, exit margin, actual step count, optional full current trajectory for diagnostics, h0 checksum, next-token ID, KV/update receipts when profiling. The next output token, current-loop hidden/gates, future depth statistics and reference continuation are not practical predictor inputs.

Keep uncompressed hidden features in sharded host storage. Apply projection/normalization fitted only on TRAIN. No whole-history KV dump is required. Teacher forcing is allowed only as an explicitly named workload protocol, not as hidden use of future outputs.

## B3. Required insight analyses

| Question | Analysis and control | Output |
| --- | --- | --- |
| Are depth labels stable? | identical token/KV/h0 repeat; separately change seed while holding everything else fixed | per-token depth variance, gate margin and repeat agreement |
| Is token identity predictive? | TRAIN token-conditional depth histogram versus held-out same-token occurrences; stratify counts and unknown/rare tokens | token-prior recovery, histogram, support coverage |
| Does context matter for the same token? | compare same-token occurrences across held-out requests; add prior hidden/Prelude features | incremental within-token explanatory power and RIC recovery |
| Is a sequence persistently deep? | between-request vs within-request variation; previous-depth autocorrelation without future sequence means | sequence-prior versus token-context comparison |
| Does generation position explain useful ordering? | global mean-by-step AND rank/error within each simultaneous ready pool | distinguish common time trend from cross-request signal |
| Was projection lossy? | same head/loss on old projection, larger projection, raw hidden and trainable bottleneck | learning curves and held-out recovery |
| What errors create valleys? | false-shallow long-tail insertions, batch-max regret, number of deep contaminants per batch | fraction of RIC regret from each error type |

A generation-step term common to all requests cannot change their ordering; monotone scalar calibration cannot change ordering either. Do not sell either as an independent scheduling gain. Global token frequency/position correlations are not causal findings. Do not infer semantic reasoning difficulty from exit depth alone.

Counterfactual threshold sweeps on a frozen trajectory describe that SAME starting state only. They do not replace self-consistent multi-token recollection at a new threshold because prior exits affect future state/KV. Cap-limited tokens must be distinguished from true threshold crossings.

# Block C — Predictor comparisons, then fixed-B scheduling confirmation

## C1. Candidate ladder (same split and policy)

| ID | Candidate | Purpose |
| --- | --- | --- |
| P0 | Reproduced C8 on validated data; FIFO/random/previous-depth controls | Comparable old-method floor, not an assumed winner |
| P1 | Smoothed token/policy depth distribution with rare-token backoff | Pure offline-pattern method; no neural model required |
| P2 | Token prior + causal depth/trajectory/history features | How far low-cost pattern calibration gets |
| P3 | Raw previous exit hidden + current embedding, learned bottleneck | Test missing contextual information, particularly Ouro |
| P4 | Context fusion: P3 plus Huginn Prelude/h0 or Ouro gate history | Full allowed pre-loop representation probe |
| P5 | Token-prior logits + contextual residual head | Primary proposed method: reusable prior corrected by context |
| P6 | P5 depth distribution with calibrated tail-aware grouping | Test whether risk-aware use of probabilities adds value |
| P7 | Stronger group-aware training on P4/P5 features | Test multi-contaminant regret rather than only single-swap signal |

For P1, smooth TRAIN counts into a global/policy prior and back off on sparse tokens; choose smoothing on validation. Do not use dataset-specific lookup values fitted on TEST. Any causal per-request adaptation uses observed PAST exits and has a matched baseline.

For neural candidates, first check that a sufficiently expressive head can fit a tiny TRAIN subset (sanity, not performance). Compare frozen Gaussian projection and trainable projection explicitly. Use bounded width choices 64/256 and up to 50 epochs with validation early stopping; record seed, parameter count and optimization budget. Repeat the two best validation families with three seeds. A larger diagnostic teacher is optional and explicitly budgeted; distill only after it demonstrates additional held-out signal.

Use expected-depth/ranking losses and cumulative survival distribution objectives as matched comparisons. C7 already exists: merely renaming it survival prediction is not a new contribution. Enforce or verify monotone cumulative probabilities; calibrate on the designated calibration split. Treat the independence assumption in expected group-max calculations as an approximation to validate, not a theorem about model outputs.

Primary selection: validation Oracle RIC recovery plus measured cost. Report mean/median and bootstrap intervals over independent requests/cohorts, not treating repeated tokens as independent requests. Choose two finalists before TEST is opened. Keep all negative candidates and do not declare that a best method must be good.

## C2. Microbatch composition contract and EasyBalance inspiration

Borrow ready/deferred/RunSet bookkeeping from EasyBalance Algorithm 1/section 4.2, not its cross-layer execution or device-load objective. The decision happens BEFORE a recurrent interval. Ready tokens/chunks not selected stay available for later batches; selected membership is locked until exit. All N items are eventually processed once at that boundary. No waiting for hypothetical future information while the GPU has eligible work.

First compare predictors using identical scalar sorting and fixed B. Then, using the SAME distribution output, compare expected-depth sort, upper-tail score and calibrated expected-group-max greedy composition. This separates feature prediction from scheduling gains. Deferred items cannot be dropped or excluded from the timer. Reordering already fixed microbatches alone does not reduce their sum of maximum depths on this serial fixed-work runtime.

N=256 and B=32 imply eight microbatches; their count is not an independent control. Sweep fixed B=4,8,16,32,64 across separate runs. Optional token/chunk selection granularity of 1/4/8 is a later validation ablation using homogeneous predicted chunks, not an arbitrary large mixed chunk fixed in advance. Keep Rmin/Rmax and chosen exit policy unchanged. B1 and N=B controls must show no composition RIC gain.

## C3. Physical evaluation protocols

1. **Frozen-work replay:** identical input token IDs, prescribed exit depths, h0/noise, KV policy and cohort across FIFO/Predictor/Oracle. Compute predictor features causally from this execution. Report fixed-work systems throughput only; record deviations of recomputed live gate as diagnostics, never secretly force them into a live claim.
2. **Live-exit check:** real exit policy, same input-token protocol and stochastic identity. If batch arithmetic changes depths, report the differences and actual work; do not assert same-work checksums should match by fiat.
3. **Optional fixed-policy free-generation timing after correctness:** same prompts and generation cap/EOS protocol; report actual generated tokens and BCT. No task-quality evaluation. Different output lengths/work prevent attributing the entire timing difference purely to packing.

Start physical finalists at N256/B32, 32 tokens under anchor policies. Include N=B control; then fixed-B sweep and long-prompt transfer subject to validated HBM. At least 2 warmups and 5 measured repetitions per strategy; alternate/randomize strategy order with a recorded seed. Exclude model load/compilation/warmup from scored runtime, but include predictor feature maintenance, copies, CPU/GPU synchronization actually required, inference, sort, KV gathering/fills, compaction and current boundary path. Keep the same staged FIFO control for Prelude-based predictors.

Measure separately: prefill wall, decode wall, all-request BCT=prefill+decode+scheduling under the stated protocol, core GPU time, predictor/sort/feature overhead, KV footprint, active-width histogram, RIC and total executed token-loop work. Do not label decode-only time full E2E. Validate standalone clean timing against profiling; tiny changes within dispersion are not gains. Compare predictors to the Oracle measured on the SAME corrected runtime and cohort, not a historical denominator.

## Interpretation and stop rules

A weak token prior is a result, not a reason to alter the test set. Strong TRAIN but weak validation suggests generalization problems. Strong saved-feature recovery but weak live recovery requires feature/state alignment investigation. Strong RIC but weak wall recovery requires overhead analysis. All candidates weak does not prove pre-loop prediction impossible.

Report recovery bands (<25%, 25-50%, 50-75%, >75%) as descriptive, not guaranteed targets. A useful primary milestone is >50% validation RIC recovery followed by positive net physical time; no success is assumed. Do not start boundary-batching design B to conceal a failed predictor comparison. Publish the bounded result and ask for a research decision if the maximum planned search is exhausted.

## Required artifacts and commit sequence

New result roots only: `phase04/results/<timestamp>_<source_sha>/`.

Commit source/plan first, execute from the recorded source, then publish result and review commits separately. Additive code can live under `phase04_analysis/` or clearly versioned runtime modules; do not silently alter existing result definitions.

Artifacts:
- `policy_manifest.json`, `runtime_validation.json`, `rng_identity_report.json`;
- frozen dataset/tokenized-input/split/feature manifests and hashes;
- `depth_pattern_analysis.csv`, within-step/within-token analyses, train learning curves;
- all-candidate validation tables; sealed-test receipt and final frozen configuration;
- `predictor_comparison.csv`, `physical_matrix.csv`, raw repeat summaries;
- real-model active-row/KV debug receipts outside scored timings;
- `RESULT_REVIEW.md` and `review_manifest.json` following the existing five-axis protocol.

The final review must answer: Which implementation issues were reproduced and fixed? Which remain unverified? What information explains useful within-pool depth differences? Does offline prior help? Does previous hidden add held-out value? Which predictor wins on fresh confirmation data? How much net Oracle opportunity is recovered? What is the actual full workload time, with which token/exit protocol?
