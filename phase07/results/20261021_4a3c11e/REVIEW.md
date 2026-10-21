# Phase 07 — Review (run 20261021_4a3c11e)

This review covers five axes: measurement, correctness, simulator vs physical, fair comparison and hypothesis. Each item states a problem or limit and its effect on the claims in RESULTS.md.

## 1. Measurement

* **E4 physical timing.**
  * Protocol: 2 warmups and 5 timed repetitions per policy, randomized order, the same engine per cell.
  * Every timed predicted repetition is faster than every FIFO repetition: the min/max ranges do not overlap in any of the 4 cells.
  * RIC and the work hash are identical across repetitions, with 0 engine errors.
  * Spread is small: FIFO max−min is ≤ 0.18 s, while the predicted−FIFO gap is ≥ 0.34 s.
  * CPU affinity was pinned with `taskset` (GPU0 → cores 0–7, GPU2 → 16–23).
* **Predictor cost is not inside the engine timer.**
  * The `predicted` policy replays scores that were computed offline from B=1 features.
  * The online cost comes from a separate H100 microbenchmark: 0.155 ms synchronized per step at N=256. It uses synthetic weights and a bf16→fp32 copy as a stand-in for the feature gather.
  * The real cost of gathering `prevh` and the embedding from engine buffers, and of maintaining the history scalars, was **not** measured inside the CDB runtime.
  * The "net" columns add 32 × 0.155 ms = 5 ms. That is about 0.1% of decode wall time, so even a 10× underestimate would not flip the sign. It is still an estimate, not an integrated measurement.
* **Attempt 1 of E4 physical crashed** in the prev-depth overhead microbenchmark (a zero-width matrix) after all runs had completed. Rows were written only at the end, so they were lost.
  * The script was fixed in f36941c (rows written first) and the run was repeated in full.
  * The attempt-1 logs are kept in `e4_physical_failed_attempt_1/`. No attempt-1 number is used.
* **Co-residency on GPU3 during the CDB parity runs.** The manifest records another process at start (PID 1056365, almost certainly lane D's collector from this study; not re-verified). Parity is a correctness measurement, so co-residency does not affect it, and no timing was taken from it.

## 2. Correctness

* **Parity gate.** E0 fails on both runtimes, reproducing Phase 04 Gate A.
  * Ouro flips are ±1 and occur at about 0.4–1.3% of positions.
  * Huginn flips are about 2.5–5.4%, with tails up to ±5.
  * Consequences:
    * All E1–E4 labels are canonical B=1 profiling depths.
    * No live-exit speed or quality claim is made.
    * Forced-depth wall times assume B=1 depths, which a live B>1 run would not reproduce exactly.
* **AGENTS.md is stale relative to this phase.** It still carries the Phase 04R "no GPU / no new predictor" block. This phase ran under the explicit Phase 07 task (`AGENT_TASK.md`, commit d7b6783) and the user's GPU 0–3 confirmation. The Phase 04R rank-stability contract is still undecided, and nothing here activates `PROPOSED_B2_CONTRACT.md`.
* **Leakage controls.**
  * Splits are request-disjoint over deduplicated prompt texts.
  * The token-ID lookup is fit out-of-fold on train.
  * Standardization statistics and temperature come from train and validation only.
  * `prevh` is the previous token's final state, which is available before the current token's Core.
  * E3 uses only loop-≤s signals for survivor targets D ≥ s+1.
* **Degenerate baselines.**
  * Huginn τ0.26: the `class+pos+ctx+history_logistic` baseline (τ_b −0.09, recovery ≈ 0) and the linear probes (NLL 2.1–7.4) are fitting failures, not evidence of absent signal. The MLPs on the same features are stable.
  * Cross-threshold transfer from τ0.26 to τ0.12 gives exact accuracy ≈ 0, because the label supports barely overlap (depths 5–10 vs 7–14).
  * Neither result is used in any conclusion.
* **Huginn hidden states** are stored as float32 because fp16 overflowed. Ouro hidden states are stored as fp16.
* **Optimized-CDB B=1 reference.** Upstream width-1 and width-2 engines fail prefill-graph warmup. The workaround is one request per call on a width-2 engine, with prefill graphs disabled at all widths (`e0/failed_attempts/README.txt`).

## 3. Simulator vs physical

* E4 utility is TRACE_ONLY and lockstep. It assumes every request is resident and cohorts are re-formed at every token. The physical CDB path forms cohorts per no-refill wave via `cdb_ext`.
* Physical RIC recovery tracks trace recovery closely:

  | Model | B | Physical RIC recovery | Trace recovery (T=64) |
  |---|---:|---:|---:|
  | Ouro | 8 | 30.9% | 32.1% |
  | Ouro | 32 | 20.2% | 18.2% |
  | Huginn | 8 | 24.7% | 30.8% |
  | Huginn | 32 | 26.1% | 31.8% |

  Wall recovery tracks RIC recovery to within about 1.5 points. This is unlike Phase 03, where wall recovery was far below RIC recovery.
* **Regime caveat.** At the same resident set, the native single cohort (B=N=256) takes 1.2 s for Ouro and 2.4 s for Huginn, against 5.9–19 s and 8–24 s for B=32/8 FIFO.
  * Partitioning N into B<N cohorts is itself a large slowdown on this runtime.
  * The grouping gain is relevant only where B<N is imposed, for example when the KV budget or HBM limits the concurrent batch. That regime was **not** induced or measured here.
* Ouro-2.6B and the ShareGPT, q0.4 and τ0.26 lanes have trace results only.

## 4. Fair comparison

* All policies within a cell share the same model, resident set, B, T, prompts, frozen depths, engine instance, cache and work hash.
* The depth oracle is not deployable. It is the denominator, not a competitor.
* **The primary predictor was fixed by rule, not by test ranking:** `all_precore|h64|w0` (the full pre-core set at the smallest MLP width), hard-coded in `run_post.sh` (commit 37e69fe).
  * Some Ouro-alpaca probe outputs had been seen before that commit. The rule did not change as a result, and on test it is not the best variant for most lanes.
  * The "best" E4 policies in `e4_grouping_utility.csv` are selected on test and are optimistic. They are not used in the decision.
* **Phase 03 comparison.** Cohort size, B, T=32 and the thresholds match, but the prompt sample, split seed and physical runtime differ. Phase 03 timed its CPU predictor inline (0.49 s per run); Phase 07 adds a GPU microbenchmark estimate.
  * Treat the trace rows as comparable predictor quality.
  * Treat the physical rows as different implementations, not a predictor A/B.
* On Ouro the Phase 07 probe (18.2%) equals Phase 03 C8 (17.9%). It is not presented as a novel predictor.

## 5. Hypothesis

* **Pre-core information is real but mostly "history in representation form".** The previous token's final recurrent state carries 80–100% of the attainable pre-core signal.
  * The current-token embedding and the Huginn Prelude output each give ≤ 0.10 recovery.
  * Depth memory is short (centered lag-8 autocorrelation ≤ 0.07) and mostly within-request (between-request variance share 0.12–0.21).
* **Early-loop information** (E3) is strong for Ouro (+6–14 accuracy points at loop 1/2) and absent for Huginn before loop 3. An intra-Core regrouping policy could exploit it for Ouro, but that falls outside strict no-refill and was not implemented.
* **The utility ceiling is bounded.**
  * The primary predictor recovers 14–36% of trace Oracle headroom.
  * Ouro recovery drops at larger B (B32 0.14–0.24 vs B8 0.27–0.36).
  * Oracle headroom itself is 15–23% (Huginn) and 22–40% (Ouro) of FIFO RIC.
  * At these levels, wall-time reductions are at most about 10% in the B<N regime.

## Limits that bound the decision

1. Live exit is batch-dependent (E0 FAIL). Forced-depth gains are an upper-bound-style estimate for live execution.
2. The B<N regime itself is not justified by the measurements here (point 3).
3. Predictor feature extraction is not integrated in the engine. Its cost is a microbenchmark estimate.
4. Two datasets only (Alpaca, ShareGPT), with 64-token prompt caps and 64 decode tokens. Dataset revisions are local caches; prompt hashes are in the manifests.
5. Exit definitions are model-specific (Ouro cumulative hazard CDF ≥ q; Huginn relative latent diff < τ). Thresholds were fixed before inspecting the data.
6. KV residency: resident-only cells, no swapping. The N/C/B split was not explored beyond N=256 and B ∈ {8, 32} physically.
