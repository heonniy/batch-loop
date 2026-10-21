# Phase 07 — Exit-depth predictability and grouping utility: results (run 20261021_4a3c11e)

**Decision: PRE-CORE GO, qualified. It holds only under `PHYSICAL_FORCED_DEPTH`. Live end-to-end is BLOCKED by E0.**

* A frozen pre-core MLP-64 beats previous-depth grouping on every lane (TRACE_ONLY).
* On the optimized CDB no-refill runtime, it reduces measured decode wall time against FIFO on both model families. This holds after adding the separately measured predictor cost. All timed repetitions are disjoint from FIFO.
* Network-level numbers (forced depth, N=256, T=32):

  | Model | B | Wall-time reduction | Net Oracle-wall recovery |
  |---|---:|---:|---:|
  | Ouro-1.4B | 8 | −9.5% | 31% |
  | Ouro-1.4B | 32 | −6.5% | 22% |
  | Huginn-3.5B | 8 | −3.4% | 24% |
  | Huginn-3.5B | 32 | −4.2% | 25% |
* Qualifications (see REVIEW.md):
  1. B1 and B>1 live-exit parity FAILS on both the eager and the optimized runtime. All depths are frozen B=1 profiling labels, so this is not live speed or quality evidence.
  2. Phase 03 defined a 25% useful band. The primary predictor clears it only at Ouro B8 and Huginn B32. Ouro B32 (21.6%) and Huginn B8 (24.0%) fall short.
  3. The no-refill B<N regime is 4.8× (Ouro) and 3.3× (Huginn) slower than the native single-cohort run at B=N=256. Grouping only pays where B<N is forced, for example by KV/memory limits.
  4. Most of the predictive signal comes from the previous token's final recurrent state (`prevh`). The current-token pre-core context contributes little: the Huginn Prelude output alone recovers about 6%.

Evidence classes:
* TRACE_ONLY: E1, E2, E3 and E4-utility.
* PHYSICAL_FORCED_DEPTH: E4-physical.
* PHYSICAL_LIVE_EXIT correctness: E0.

All numbers are on request-disjoint test splits (256 requests × T=64 decode tokens unless stated). Probe metrics are means over 3 seeds. Raw tables are in `analysis/summary/`, `analysis/e1/`, `analysis/e4/`, `probes/` and `e4_physical/`.

## Runs

| Lane | GPU (physical) | Model | Exit rule | Dataset | Status |
|---|---|---|---|---|---|
| A | 0 | Ouro-1.4B | q=0.2 | Alpaca, ShareGPT | complete |
| B | 1 | Ouro-2.6B | q=0.2 | Alpaca, ShareGPT | complete |
| C | 2 | Huginn-3.5B | τ=0.12 | Alpaca, ShareGPT | complete |
| D | 3 | Ouro-1.4B q=0.4; Huginn-3.5B τ=0.26 | — | Alpaca | complete |

* Splits per lane:
  * Train: 512 requests.
  * Validation: 128 requests.
  * Test: 256 requests.
  * Request-disjoint over the deduplicated prompt pool, split seed 20261021.
* Prompts are capped at 64 tokens; T = 64 B=1 teacher-forced decode tokens.
* The E4 physical runs and the optimized-CDB parity cover Ouro-1.4B and Huginn-3.5B only. Ouro-2.6B has no CDB-side model entry in `cdb_ext`, so it was not run there.
* Failures are kept:
  * `e4_physical_failed_attempt_1/`: crash in the overhead microbenchmark; the rows were lost and the run was rerun after a fix.
  * `e0/failed_attempts/`: upstream width-1/2 prefill-graph warmup errors.

## E0 — Parity gate (FAIL, Phase 04 reproduced)

**Eager Phase-02 runtime.** First 64 test requests, teacher-forced with B=1 tokens. Live exits at B=8/32 compared with B=1 labels (`e0/e0_parity_eager_*.json`).

| Lane | B | Depth flips | Next-token flips | Δdepth histogram |
|---|---:|---:|---:|---|
| ouro14 q0.2 | 8 / 32 | 1.32% / 1.20% | 1.20% / 1.37% | ±1 only |
| ouro14 q0.4 | 8 / 32 | 0.44% / 0.59% | 0.95% / 1.22% | ±1 only |
| ouro26 q0.2 | 8 / 32 | 1.05% / 0.68% | 1.15% / 1.17% | ±1 only |
| huginn τ0.12 | 8 / 32 | 5.22% / 5.40% | 0.29% / 0.51% | mostly ±1; tails −5…+3 |
| huginn τ0.26 | 8 / 32 | 3.20% / 2.54% | 0.66% / 0.73% | mostly ±1; tails −5…+4 |

**Optimized CDB.** Upstream abdfae0, free-running live exits, 32 requests × 16 tokens (`e0/e0_parity_cdb_*.json`).

| Model | B=8: identical-token requests / exit flips | B=32: identical-token requests / exit flips |
|---|---|---|
| Ouro-1.4B q0.2 | 15/32 / 71 of 512 | 22/32 / 45 of 512 |
| Huginn-3.5B τ0.12 | 28/32 / 35 of 512 | 26/32 / 46 of 512 |

Free-running decode lets one flip cascade into token divergence, so these counts are upper-bound-like. Both runtimes FAIL. Every later result uses B=1 canonical labels and makes no live-exit claim.

## E1 — Attribution, autocorrelation and simple baselines (TRACE_ONLY)

**Information gain** in bits on test (conditional cross-entropy reduction vs the marginal; tables fit on train):

| Lane | H(D) | token class | position | context | token ID | prev depth |
|---|---:|---:|---:|---:|---:|---:|
| ouro14 q0.2 alpaca | 1.97 | 0.017 | 0.025 | 0.012 | 0.138 | 0.144 |
| ouro14 q0.2 sharegpt | 1.95 | 0.012 | 0.033 | 0.018 | 0.074 | 0.157 |
| ouro14 q0.4 alpaca | 1.40 | 0.031 | 0.019 | 0.013 | 0.146 | 0.124 |
| ouro26 q0.2 alpaca | 1.85 | 0.024 | 0.025 | 0.015 | 0.129 | 0.125 |
| ouro26 q0.2 sharegpt | 1.84 | 0.016 | 0.022 | 0.010 | 0.094 | 0.125 |
| huginn τ0.12 alpaca | 2.29 | 0.067 | 0.011 | 0.013 | 0.095 | 0.124 |
| huginn τ0.12 sharegpt | 2.25 | 0.052 | 0.022 | 0.026 | 0.111 | 0.121 |
| huginn τ0.26 alpaca | 1.65 | 0.049 | 0.011 | 0.015 | 0.086 | 0.088 |

* Token identity and previous depth each carry about 0.1–0.15 bits, which is 5–9% of H(D). Position and context length carry about 0.01–0.03 bits.
* Between-request variance share of depth:

  | Family | Share | Mean depth | CV |
  |---|---:|---:|---:|
  | Ouro | 0.12–0.13 | 2.2–2.8 | 0.23–0.43 |
  | Huginn | 0.18–0.21 | 6.3–8.8 | 0.13–0.15 |

  Most variation is therefore within-request and token-to-token.
* Lag-1 autocorrelation (`analysis/e1/e1_autocorrelation.csv`, bootstrap 95% CI):
  * Raw: 0.31–0.39.
  * Within-request centered: 0.19–0.26.
  * Centered lag-8: ≤ 0.07.
  * Depth memory is short.
* All of these are associations, not semantic-difficulty causes.

**Simple baselines, RIC recovery at B=8/16/32** (N=256, lockstep, `analysis/e1/e1_baselines.csv`):

| Lane | last depth | moving mean 4 | token-ID lookup | class+pos+ctx+history logistic |
|---|---|---|---|---|
| ouro14 q0.2 alpaca | .064/.041/.023 | .077/.054/.020 | .095/.068/.040 | .120/.074/.043 |
| huginn τ0.12 alpaca | .123/.129/.125 | .130/.147/.161 | .045/.049/.050 | .176/.196/.198 |

* The other lanes are in the CSV.
* The Huginn τ0.26 logistic baseline is degenerate (τ_b −0.09, recovery ≈ 0). The cause is a fitting failure, not a measured property; see REVIEW.
* Cross-workload transfer (Alpaca↔ShareGPT) of the logistic baseline stays within ±0.03 recovery of in-domain fits.

## E2 — Pre-core probes (TRACE_ONLY)

Seed means. Recovery is the RIC recovery at B=8/16/32. `all_precore` combines cheap + token-lookup + embedding + previous final state, plus the Prelude output for Huginn.

| Lane | cheap MLP64 | cheap+tok MLP64 | emb MLP64 | prevh MLP64 | Prelude MLP64 | all linear | **all MLP64** | all MLP128 |
|---|---|---|---|---|---|---|---|---|
| ouro14 q0.2 alpaca | .13/.09/.05 | .19/.14/.07 | .10/.07/.05 | .29/.23/.15 | n/a | .29/.22/.14 | **.32/.26/.18** | .33/.26/.19 |
| ouro14 q0.2 sharegpt | .13/.10/.05 | .17/.12/.07 | .05/.04/.02 | .25/.20/.14 | n/a | .25/.19/.12 | **.28/.22/.15** | .29/.23/.16 |
| ouro14 q0.4 alpaca | .20/.20/.12 | .25/.23/.15 | .12/.11/.07 | .33/.30/.22 | n/a | .31/.30/.24 | **.36/.34/.25** | .37/.35/.26 |
| ouro26 q0.2 alpaca | .15/.13/.09 | .22/.18/.14 | .09/.05/.03 | .32/.28/.24 | n/a | .32/.29/.24 | **.35/.31/.26** | .36/.31/.27 |
| ouro26 q0.2 sharegpt | .12/.12/.09 | .16/.14/.12 | .07/.07/.06 | .28/.27/.24 | n/a | .28/.26/.24 | **.30/.28/.25** | .30/.28/.25 |
| huginn τ0.12 alpaca | .20/.22/.23 | .22/.24/.24 | .05/.05/.04 | .30/.31/.32 | .08/.08/.06 | .25/.27/.27 | **.32/.33/.33** | .31/.33/.33 |
| huginn τ0.12 sharegpt | .16/.15/.14 | .19/.19/.19 | .06/.04/.03 | .25/.26/.26 | .08/.07/.07 | .22/.22/.23 | **.28/.28/.28** | .28/.28/.28 |
| huginn τ0.26 alpaca | .22/.23/.26 | .24/.25/.28 | .06/.05/.04 | .30/.33/.36 | .09/.08/.07 | .12/.13/.15 | **.32/.35/.36** | .32/.34/.37 |

**Probe quality for all_precore MLP64:**

| Family | Exact-depth accuracy | NLL | Within-step Kendall τ_b |
|---|---:|---:|---:|
| Ouro | 0.57–0.72 | 0.63–0.98 | 0.53–0.56 |
| Huginn | 0.45–0.61 | 0.94–1.34 | 0.48–0.52 |

* ECE after temperature scaling is 0.010–0.020 for Ouro and 0.033–0.051 for Huginn. Huginn is over-confident in the top confidence bins (`analysis/summary/e2_calibration_confusion.png`).
* **Linear vs MLP:** the MLP adds 0.02–0.06 recovery over linear on the full set. Huginn τ0.26 linear is unstable (NLL 2.1). MLP128 ≈ MLP64.
* Class-weighted loss ≈ unweighted on ranking metrics, with worse NLL.
* **Where the signal is:**
  * `prevh` alone reaches 80–100% of the full set's recovery.
  * The current-token embedding is weak (≤ 0.10).
  * The Huginn final Prelude output alone recovers only 0.06–0.09.
* Pre-core depth information is mostly carried-over recurrent state from the previous token, not the current token's contextual pre-core representation. The Ouro embedding and the Huginn Prelude are never treated as equivalent.

**Cost.**
* The all_precore MLP64 has 2.6×10⁵ (Ouro, dim 4111) to 1.0×10⁶ (Huginn, dim 15867) parameters, so MACs/token ≈ params (`e2_e3_probe_summary.csv`).
* Measured on H100 at N=256 (synthetic weights, `e4_physical/*/RUN_MANIFEST.json`): feature copy + MLP + E[D] + argsort takes 0.14 ms GPU and 0.155 ms synchronized wall per decode step. The previous-depth argsort alone takes 0.08 ms.

## E3 — Early-loop information (diagnostic, survivor-conditioned, TRACE_ONLY)

Target: P(D = s+c | D ≥ s+1), on the same survivor subset, comparing the same pre-core features against pre-core + early-loop features. Values are accuracy for the MLP64 (precore → early).

| Lane | after loop 1 (D≥2) | after loop 2 (D≥3) |
|---|---|---|
| ouro14 q0.2 alpaca | .60 → .72 | .73 → .81 |
| ouro14 q0.2 sharegpt | .60 → .71 | .74 → .82 |
| ouro14 q0.4 alpaca | .73 → .87 | .83 → .91 |
| ouro26 q0.2 alpaca | .64 → .78 | .81 → .87 |
| ouro26 q0.2 sharegpt | .63 → .78 | .82 → .87 |
| huginn τ0.12 alpaca | .45 → .45 | .44 → .47 |
| huginn τ0.12 sharegpt | .45 → .46 | .45 → .46 |
| huginn τ0.26 alpaca | .62 → .62 | .61 → .63 |

* **Ouro:** loop-1/2 hidden state, hazard and CDF add +0.06 to +0.14 accuracy over pre-core.
* **Huginn:** latent-diff and loop states add ≈ 0 before loop 3, because Huginn exits at depth 6–9.
* This would only enable a future intra-Core regrouping policy, which was not implemented and is out of no-refill scope.

## E4 — Grouping utility

**TRACE_ONLY,** N=256 lockstep, test, T=64 (`analysis/e4/e4_grouping_utility.csv`). Values are RIC recovery at B=8 / B=32.

| Lane | Oracle RIC reduction | static first-token (diag.) | prev depth | **all_precore MLP64 E[D]** | P(D≥r*) MLP64 |
|---|---|---|---|---|---|
| ouro14 q0.2 alpaca | 34% / 35% | .01 / .01 | .06 / .02 | **.32 / .18** | .29 / .19 |
| ouro14 q0.2 sharegpt | 33% / 34% | .01 / .02 | .07 / .02 | **.27 / .14** | .27 / .18 |
| ouro14 q0.4 alpaca | 22% / 26% | .03 / .01 | .11 / .05 | **.36 / .24** | .27 / .20 |
| ouro26 q0.2 alpaca | 36% / 40% | .01 / −.02 | .06 / .01 | **.35 / .24** | .27 / .18 |
| ouro26 q0.2 sharegpt | 34% / 39% | .02 / .03 | .08 / .05 | **.30 / .24** | .22 / .18 |
| huginn τ0.12 alpaca | 18% / 23% | .03 / .01 | .12 / .13 | **.31 / .32** | .30 / .31 |
| huginn τ0.12 sharegpt | 17% / 22% | .02 / .02 | .10 / .08 | **.28 / .28** | .25 / .25 |
| huginn τ0.26 alpaca | 15% / 20% | .01 / .01 | .10 / .15 | **.32 / .35** | .23 / .28 |

* Logical slot utilization (FIFO → MLP64 → Oracle), Ouro q0.2 B=8: 0.65 → 0.73 → 0.98.
* Expected-cohort-max error of the PDF is +0.03…0.14 for Ouro and +0.35…1.1 for Huginn, an over-estimate under the independence assumption.
* Oracle headroom is largest around B=16–32 (`e4_oracle_gap_by_B.png`). Ouro predictor recovery falls as B grows toward N/4.

**PHYSICAL_FORCED_DEPTH,** pinned optimized CDB (CUDA graphs, paged FA3, async, no-refill), N=256, T=32 (`analysis/summary/e4_physical_summary.csv`).
* Protocol: frozen B=1 test depths, 2 warmups + 5 timed repetitions per policy, randomized policy order.
* Every repetition reported 0 errors and every policy had one work hash.
* The median decode wall is reported as min…max.

| Model | B | FIFO (s) | Predicted (s) | Oracle (s) | RIC rec. | Wall reduction | Wall rec. | Net wall rec. (+0.155 ms/step) |
|---|---:|---|---|---|---:|---:|---:|---:|
| Ouro-1.4B q0.2 | 8 | 18.975 (18.96…19.02) | 17.172 (17.15…17.38) | 13.210 | 30.9% | 9.5% | 31.3% | 31.2% |
| Ouro-1.4B q0.2 | 32 | 5.855 (5.79…5.97) | 5.467 (5.45…5.64) | 4.077 | 20.2% | 6.6% | 21.8% | 21.6% |
| Huginn-3.5B τ0.12 | 8 | 24.450 (24.45…24.62) | 23.617 (23.60…23.76) | 20.985 | 24.7% | 3.4% | 24.0% | 23.9% |
| Huginn-3.5B τ0.12 | 32 | 8.043 (8.03…8.20) | 7.698 (7.68…7.82) | 6.682 | 26.1% | 4.3% | 25.3% | 25.0% |

* Native single cohort (B=N=256, same trace): Ouro 1.225 s, Huginn 2.447 s.
* The predicted policy replays offline B=1-feature scores. The predictor cost is added from the microbenchmark rather than timed inside the engine.
* Wall reduction = (T_FIFO − T_pred)/T_FIFO. This is not a speedup.

## Comparison with Phase 03

Matched cohort: held-out N=256, B=32, 32-token test cohorts, same models and thresholds. Phase 07 trace values use T=64. Data: `analysis/summary/phase03_comparison.csv`.

| | Ouro-1.4B q0.2 | Huginn-3.5B τ0.12 |
|---|---:|---:|
| P03 C0 previous depth (trace) | 3.4% | 6.9% |
| P07 previous depth (trace) | 2.3% | 12.5% |
| P03 C1 history (trace) | 6.3% | 9.2% |
| P07 cheap history MLP (trace) | 4.7% | 22.8% |
| P03 C4 embedding / C5 representation (trace) | 8.5% / 10.8% | 4.9% / 5.2% |
| P07 emb MLP / Huginn Prelude MLP (trace) | 4.7% / — | 3.9% / 6.1% |
| P03 C6 fused regression (trace) | 13.4% | 15.3% |
| P03 best, C8 pairwise (trace) | 17.9% | 20.9% |
| **P07 all_precore MLP64 (trace)** | **18.2%** | **31.8%** |
| P03 C8 physical RIC / wall recovery | 18.5% / 11.4% | 21.4% / −4.6% |
| **P07 physical RIC / net wall recovery** | **20.2% / 21.6%** | **26.1% / 25.0%** |

* **Ouro:** the Phase 07 probe matches Phase 03's best pairwise ranker C8 on trace RIC recovery, with no gain. The physical improvement over Phase 03 comes from two sources rather than from better prediction:
  * Predictor cost: Phase 03's CPU feature extraction + head + sort took 0.49 s per 32-token cohort run. The GPU-resident path here is measured at about 0.005 s.
  * A runtime whose wall time tracks RIC more closely: the optimized CDB.
* **Huginn:** recovery rises from 20.9% to 31.8% (trace). The gain comes from the previous token's final recurrent state, which Phase 03's C0–C9 did not use as a representation, not from the Prelude. The Prelude-only probe (6.1%) reproduces Phase 03's C5 (5.2%).
* The physical runtimes differ (Phase 03 eager vs Phase 07 optimized CDB), and Phase 03 timed its predictor inline. The physical rows are therefore not a same-runtime A/B of the predictors.

## Decision against the predeclared rule

| Rule | Status |
|---|---|
| Predictor beats simple previous-depth | **Met** in every lane, trace and B. |
| Positive net E2E gain on both families | **Met under PHYSICAL_FORCED_DEPTH** (Ouro −6.5…−9.5%, Huginn −3.4…−4.2% wall). **Not established live:** E0 parity fails, so B>1 live labels differ from those used here. |

* Early-loop information helps only Ouro, so EARLY-LOOP ONLY does not describe the outcome.
* NO-GO is not supported by the forced-depth measurements.
* Recorded decision: **PRE-CORE GO (qualified: forced-depth, B<N no-refill regime, live-exit BLOCKED).**
* The Phase 03 25% band is cleared only by Ouro B8 and Huginn B32. The Ouro B32 net wall recovery is 21.6%.
* The next finite step is to integrate the GPU-resident feature gather + MLP into the engine (timed inline), and to resolve or bound live-exit batch dependence (rank-stability contract, Phase 04R).

## Plots

* `analysis/e1/e1_depth_cdf_by_token_class.png`
* `analysis/e1/e1_autocorrelation.png`
* `analysis/summary/e2_calibration_confusion.png`
* `analysis/summary/e4_overhead_vs_recovery.png`
* `analysis/summary/e4_oracle_gap_by_B.png`
* `analysis/e4/e4_ric_reduction_vs_B.png`
