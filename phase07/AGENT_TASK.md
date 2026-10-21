# Phase 07 — Depth Predictability Characterization (4 × H100)

**STATUS: experiment handoff committed; code implementation, GPU runs and results NOT yet completed.** Execute this plan on `research/phase07-depth-predictability-4gpu-20261021`. Preserve phase01–06 evidence and all prior negative predictor results.

## Objective and research questions

We need to decide whether a **low-cost predictor available before a no-refill recurrent wave** can accurately rank/decompose future depths enough to improve real batch formation. Do not optimize depth MAE alone.

- **RQ1 Token/position/history:** Are depth and shallow/deep propensity associated with token identity/type, prompt category, context length, decode position, and previous depths? Quantify between-request and within-request variation; avoid semantic-difficulty causal claims.
- **RQ2 Pre-Core state:** How much predictable depth information is present *before* recurrent work, from Ouro embedding/initial state and Huginn Prelude output? Frozen model; linear probe and tiny MLP; include a matched token/history baseline.
- **RQ3 Early recurrent trajectory:** How much additional predictive signal is supplied by loop-1/loop-2 hidden states, Ouro hazard/CDF, Huginn latent-difference trajectory? Handle tokens that have already exited. This is a **diagnostic upper bound** and NOT directly usable for decode-boundary no-refill grouping without an architecture change.
- **RQ4 Scheduling value:** Do predictor outputs improve **same-work** RIC, survivor active-width shape, expected microbatch maximum, and eventually optimized CDB physical decode wall versus FIFO and same-cohort oracle?

## Baseline evidence and pitfalls

Read root `AGENTS.md`, phase02–05 RESULTS/REVIEW and phase06/AGENT_TASK. Phase03 C0–C9 already tested history, representation, regression, ordinal, ranking and listwise predictors; its measured Oracle wall recovery was 11.4% Ouro / -4.6% Huginn. This study MUST compare with/reproduce those baselines on matched cohorts, **not rebrand a linear probe as novel**. Phase04 BF16 batch-dependent live depths failed gate A; frozen B1 labels cannot be claimed batch-invariant live truth. Ouro original HF gate is post hoc full-depth in Phase01. All frozen-depth GPU grouping is `PHYSICAL_FORCED_DEPTH`, not live accuracy.

## Pin implementations and reproducibility

Upstream optimized CDB: `kschwethelm/continuous-depth-batching` commit `abdfae0adbdd12be8c6123ccb34eee98a38afb39`, or record an explicitly reviewed replacement SHA if unavailable. Clone in ignored location (or fork/pin patch), do not silently vendor. Ouro-1.4B revision `37bc7e8ff85a90d813dc096383405f521b6a67b0`; Ouro-2.6B `f8e38372d72a8aa7f4ea9f14b98739a311589088`; Huginn-3.5B `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. Fix model-specific Rmax (Ouro 4; Huginn 16), exit semantics and minimum exit depth; record chosen thresholds before data inspection (Ouro q=0.2/0.4, Huginn tau=0.12/0.26 as existing anchors). Pin BF16, FlashAttention backend, shared/depth-indexed KV semantics, dataset revision/hash, prompt tokenization, teacher-forced token sequence, seed, split IDs, source code SHA.

**4 GPU lanes:** use four *currently authorized and idle* H100s, with `CUDA_VISIBLE_DEVICES` per process, explicit physical GPU IDs/UUIDs, and no assumptions that historical GPU 0–3 or 4–7 are free. Suggested independent workloads: Lane A Ouro-1.4B; Lane B Ouro-2.6B; Lane C Huginn-3.5B; Lane D independently validate parity / alternate threshold or run frozen predictor scoring. Do not launch while ownership uncertain; keep lanes isolated. CPU training may follow GPU feature collection. Avoid multi-GPU model parallelism for this characterization.

## E0 — Gate and instrumentation (mandatory)

1. Reproduce optimized CDB no-refill smoke and existing model-specific CUDA/paged-KV tests. Freeze exit semantics. Check B1 and B>1 live gate/next-token parity under real batching; if discrepancies persist, preserve as negative result. Use B1 canonical features/labels for *prediction science*, clearly distinguish from executed B>1 labels; no live-E2E quality/speed claims on failed parity.
2. Add low-impact hooks to extract *per request-token* token ID, decode position, context length, request/prompt hashes, past depths, pre-Core hidden state (Ouro embedding/initial state; Huginn final Prelude hidden), optional early-loop states, exit hazard/logit/CDF or latent-difference, and final observed depth. Align current token vs next-token prediction correctly; prevent lookahead label leakage.
3. Teacher-forced reference trajectory to keep target token IDs stable; frozen labels with gate+KV metadata. Use feature capture ONLY for profiling/training; time/disable feature extraction separately in actual inference. No all-layer/full-hidden-history dump by default; bounded snapshots, optional PCA/summary statistics, disk/HBM safety.
4. Predeclare train/validation/test **request-disjoint** splits, with disjoint prompt hashes and fixed seed. Never split tokens from same request across train and test. Evaluate Alpaca and ShareGPT (length buckets) separately and as held-out workload transfer where feasible. Record OOMs and missingness.

## E1 — Depth attribution, autocorrelation and baselines

- Histograms/CDFs for depth by model/threshold/position bucket/context bucket/token class (punctuation, digit, whitespace, code, other); estimate conditional entropy, within-request lag-k autocorrelation and depth-transition matrix (report class imbalance and CI). Compare token type vs context-controlled correlations; **never claim semantic causality** from correlation.
- Baselines: always-majority/marginal PDF, last depth, moving mean, seq mean, token-ID lookup fit only on train, token class + position + context, inexpensive gradient-boosted or multinomial regression as sensible. Evaluate held-out cross-prompt and cross-workload and per-threshold transfer.
- Compare Ouro hazard accumulation against Huginn relative-L2 convergence; preserve Ouro cumulative `exit_cdf` threshold and Huginn `latent_diff=||h_r-h_{r-1}||/max(||h_r||,1e-9)`, not an invented softmax/confidence gate.

## E2 — Pre-Core representation probes

- Freeze all pretrained model weights and exit policy. Capture features **after** Huginn Prelude but **before** any recurrent Core; Ouro has no Transformer Prelude, so characterize token embedding, initial recurrent state and cheap context/history separately. Never claim Ouro embedding is equivalent to Huginn contextual Prelude.
- Compare standardized linear softmax classifier, tiny 2-layer MLP (e.g. hidden widths 64/128), optionally small low-rank input projection; tune only on validation. Train 3 seeds; specify optimizer, early-stop rule, probability calibration (temperature scaling on validation only), test reliability/expected calibration error.
- Predict (P(D=r\mid x)) across allowed discrete depths (including max-depth mass) and (P(D\ge r\mid x)) survival values; model-specific exit thresholds. Per-model and joint representations as diagnostic, with class-weight ablation if imbalance.
- Feature cost: extra GPU read/copy, CPU inference if any, GPU MLP launch, sorting and graph staging. Count parameter/MACs, memory bytes and synchronized wall overhead. Pre-core availability is mandatory for deployable policy.

## E3 — Early-loop predictive information (diagnostic)

- Probe at Core loop 1 and 2, using the same features plus hidden-state delta, Ouro per-loop hazard/CDF/survival, Huginn latent diff. Report conditional prediction for **survivors** and avoid treating already-exited tokens as future participants.
- Compare exact-depth classification, ordinal survival, pairwise shallow/deep ranking and expected microbatch-maximum surrogate. Quantify gain over pre-core at **equal cohort and survivor-conditioned target**.
- Distinguish source step and decision horizon: upstream delayed/lookahead consumption can shift when an exit signal becomes actionable. Do not use future loop signals or final depth as features.
- This is not a drop-in no-refill decoder regrouping scheme. If early-loop prediction succeeds, report opportunity for a *future* intra-Core regrouping policy without implementing it silently.

## E4 — Offline grouping utility / selection of predictor

Frozen fair test cohorts: N={64,128,256} requests where available, microbatch B={4,8,16,32,64} only when B<=N, same token boundaries and KV feasibility. FIFO, static grouping based on first-token known depth (oracle-but-static diagnostic), previous-depth policy, calibrated PDF/survival grouping, predicted point-depth, pairwise ranking, dynamic true-depth oracle. No resident KV swapping in this phase; select resident-only matched cells, otherwise mark as `TRACE_ONLY` rather than claiming deployability.

Metrics: exact depth accuracy, MAE, rank tau with tie handling, NLL/Brier/ECE, `RIC=sum_{m} max_{i in m}d_i`, active-width trace, logical slot utilization, predicted-vs-true expected-max regret, `RIC recovery=(RIC_FIFO-RIC_PRED)/(RIC_FIFO-RIC_ORACLE)` (undefined if denominator=0), predictor+regroup overhead, and percent **wall-time reduction** `(T_FIFO-T_PRED)/T_FIFO`. Never confuse that with speedup `T_FIFO/T_PRED`.

Use `E2` optimized CDB runtime from Phase06 when available. A small predeclared 4-GPU independent lane comparison should run FIFO, actual pre-core predictor grouping, depth oracle, and CDB no-refill at the *same resident cap, microbatch width B, model, cache, workload and fixed token/depth trace*, with randomized policy order, >=2 warmups and >=5 timed reps. Run live-exit evaluation only after parity/quality gates pass. If Phase06 optimized runtime remains unimplemented or E0 fails, mark physical cells BLOCKED and still publish CPU/trace characterization.

## Required artifacts and completion

Create `phase07/README.md` with exact install/commands and upstream pinned checkout; `phase07/scripts/`, `phase07/tests/`, `phase07/results/<run_id>/` containing `RUN_MANIFEST.json`, `RESULTS.md`, `REVIEW.md`, raw data and matching plot sources (depth CDF/lag autocorrelation, confusion/calibration, predictor overhead vs RIC recovery, oracle gap by B/model). Commit scripts/tests and authentic receipts; retain negative/OOM/failed parity cells. Never fabricate metrics. Review five axes: measurement, correctness, simulator-vs-physical, fair comparison, hypothesis.

**Decision:** choose among PRE-CORE GO (predictor beats simple previous-depth and yields positive net E2E gain on both model families), EARLY-LOOP ONLY (early signals strong but pre-core weak; new problem formulation needed), or NO-GO (predictor cannot recover enough gain after overhead). Report limitations: true-depth oracle not deployable, KV residency and N/C/B split, prompt/generalization, model-specific exit definitions. Add a concise agent summary and push commit SHAs to this branch.
