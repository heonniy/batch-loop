# E1 — Retrospective no-refill regime analysis (CPU only)

Generated 2026-10-19 by `phase05/scripts/e1_regime_reanalysis.py` (no model load, no CUDA,
`CUDA_VISIBLE_DEVICES=""`). Input paths, SHA-256, row counts and schemas: `e1_input_manifest.json`.
No new GPU measurement was made; every timing number below is re-derived from committed phase02/03
per-repetition receipts.

## What each evidence class means here

| Evidence class | Rows | What it is | What it is NOT |
| --- | ---: | --- | --- |
| `TRACE_ONLY` | 790 | FIFO vs depth-sorted Oracle recurrent-invocation counts (RIC) on frozen B1 depth traces: phase02 32/64-token sweeps (384), phase01 fair headroom, Oracle plus practical/random strategies (400), phase04 N32/T8 canonical-B1 cohort (6) | any latency or speed |
| `LATENCY_MODEL` | 128 | phase01 hardware-calibrated replay (`HARDWARE_CALIBRATED_TRACE_REPLAY_NOT_INTEGRATED_RUNTIME`), Ouro B4/B8 only; plus `e1_calibration_surface.csv` fits | GPU-measured policy latency; it is built on native-reference HF fixed-depth calibration, not the survivor runtime |
| `PHYSICAL_FORCED_DEPTH` | 23 | survivor-compacted no-refill runtime with trace-forced frozen B1 depths, 2 warmups + 5 (one cell: 3) measured repeats; CUDA-event Core time and synchronized decode wall (prefill excluded) | live early exit, quality equivalence, native dataset throughput |
| `PHYSICAL_LIVE_EXIT` | 18 | phase04 fresh BF16 live batched exits, N32 × 8 positions, B2/4/8 × 3 orders: measured live RIC/work counts | timing (debug timing was not scored), any Oracle (no live Oracle grouping was executed) |

Native Ouro HF E2E (`ouro_native_e2e.csv`, 80 rows) applies the exit gate post hoc after full recurrence;
it is never used as an early-exit baseline. All 80 rows are listed in `e1_excluded_rows.csv`, the 20
`OOM_PARTIAL` cohorts with an explicit throughput-exclusion reason. Also excluded (listed, not dropped):
12 calibration OOM cells, 3 exploratory 3-repeat phase02 Ouro runs superseded by 5-repeat cells, the
interrupted Huginn memory-growth collector (failure receipt), and two N4/2-token validation runs.

## Reconciliation

All 959 output rows reconcile; 0 mismatches.
- phase02 `trace_sim_32tok` and `trace_sim_64tok` (192 rows each) were recomputed from the raw
  `trace_inputs/*.jsonl.gz`: trace hash, FIFO RIC, Oracle RIC, work and same-work checksum all match.
- **The task text said `trace_sim_64tok/oracle_ric_sweep.csv` is absent; it exists and is tracked in this
  branch** (192 rows). The read-only legacy sweeps under `/home/hwlee/batch-loop/results/oracle_ric_*`
  (ouro 128 rows, Huginn q0.26/0.35/0.5 16 rows each, all 64-token) are row-identical subsets of it;
  Huginn threshold 0.12 exists only in the repo copy. Nothing was copied.
- phase01 headroom: RIC = `total_capacity_slots / B` (all microbatches full); for Ouro-1.4B it was
  recomputed from raw traces (64 Alpaca / 128 ShareGPT tokens) and matches. **Ouro-2.6B raw traces are
  not committed**, so its rows are `NOT_RECOMPUTABLE` (identity check only); depth stats come from the
  phase01 trace-audit histograms.
- phase02 physical: per-repetition `summary.csv` medians equal `physical_matrix.csv`; recurrence-event
  counts equal FIFO/Oracle RIC × repeats; RIC equals the 32-token trace recompute.
- phase03 physical: FIFO/Oracle RIC recomputed from `depth_labels.npy` (held-out ids 384–639; controls
  384–415); published medians and the per-repetition-summed max-depth distribution match.
- phase04: live RIC and work recomputed from `per_token_validation.csv` using the published
  `random.Random(seed).shuffle` orders, match `batch_shape_validation.csv`.

## Physical evidence (all `PHYSICAL_FORCED_DEPTH`, prompt cap 64, 32 decode tokens)

`reduction = (T_fifo − T_policy)/T_fifo`; `×` = speedup `T_fifo/T_policy`. Brackets give the policy's
min–max over measured repeats. "ms/call" = median Core time / RIC; "width" = mean active rows per Core
call (FIFO/policy). Tracking = measured reduction / RIC reduction (Core / wall).

| Cell | q | N/B | depth mean / CV | FIFO→policy RIC | Core (CUDA events) | Decode wall | ms/call | width | tracking | reps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p2 huginn_alpaca_0.12 | 0.12 | 256/32 | 8.62 / 0.134 | 2995→2377 (20.6%) | 20.22→16.66 s (17.6%) [16.63–17.14] | 24.30→20.37 s (16.2%; ×1.193) [20.34–20.95] | 6.75/7.01 | 23.6/29.7 | 0.85 / 0.78 | 5 |
| p2 huginn_alpaca_0.26 | 0.26 | 256/32 | 6.20 / 0.119 | 2054→1707 (16.9%) | 14.43→12.36 s (14.3%) [12.30–12.54] | 18.11→15.83 s (12.6%; ×1.144) [15.75–16.02] | 7.03/7.24 | 24.7/29.7 | 0.85 / 0.75 | 5 |
| p2 huginn_alpaca_0.5 | 0.5 | 256/32 | 4.21 / 0.103 | 1317→1116 (15.3%) | 13.06→11.30 s (13.5%) [11.20–11.41] | 17.54→15.62 s (11.0%; ×1.123) [15.49–15.72] | 9.92/10.13 | 26.2/30.9 | 0.88 / 0.72 | 5 |
| p2 huginn N=B control | 0.12 | 32/32 | 8.54 / 0.125 | 361→361 (0.0%) | 2.44→2.42 s (0.6%) | 2.95→2.94 s (0.5%) | 6.75/6.71 | 24.2/24.2 | – | 5 |
| p2 ouro_alpaca_0.2 | 0.2 | 256/32 | 2.34 / 0.357 | 935→655 (29.9%) | 19.29→13.53 s (29.9%) [13.37–13.68] | 23.17→16.00 s (30.9%; ×1.448) [15.83–16.15] | 20.64/20.66 | 20.5/29.3 | 1.00 / 1.03 | 5 |
| p2 ouro_alpaca_0.4 | 0.4 | 256/32 | 2.94 / 0.219 | 1018→780 (23.4%) | 22.14→16.94 s (23.5%) [16.87–17.00] | 25.19→19.30 s (23.4%; ×1.305) [19.18–19.40] | 21.74/21.72 | 23.6/30.9 | 1.00 / 1.00 | 5 |
| p2 ouro_sharegpt_0.2 | 0.2 | 256/32 | 2.39 / 0.348 | 954→667 (30.1%) | 20.53→14.37 s (30.0%) [14.25–14.41] | 24.55→16.91 s (31.1%; ×1.452) [16.78–16.95] | 21.52/21.54 | 20.5/29.4 | 1.00 / 1.03 | 5 |
| p2 ouro_alpaca_0.2 | 0.2 | 128/16 | 2.32 / 0.361 | 877→644 (26.6%) | 18.58→13.87 s (25.4%) [13.57–13.92] | 21.99→15.95 s (27.5%; ×1.379) [15.65–16.03] | 21.19/21.54 | 10.9/14.8 | 0.95 / 1.03 | **3** |
| p2 ouro N=B control | 0.2 | 32/32 | 2.32 / 0.366 | 112→112 (0.0%) | 2.39→2.39 s (−0.3%) | 3.02→3.02 s (0.1%) | 21.30/21.36 | 21.2/21.2 | – | 5 |
| p3 ouro q0.2 ORACLE | 0.2 | 256/32 | 2.44 / 0.397 | 1021→670 (34.4%) | 22.10→14.54 s (34.2%) [14.42–14.59] | 26.20→17.05 s (34.9%; ×1.536) [16.93–17.10] | 21.64/21.70 | 19.5/29.8 | 1.00 / 1.02 | 5 |
| p3 ouro q0.2 C8 predictor | 0.2 | 256/32 | 2.44 / 0.397 | 1021→956 (6.4%) | 22.10→20.66 s (6.5%) | 26.20→25.15 s (4.0%; ×1.041) | 21.64/21.61 | 19.5/20.9 | 1.02 / 0.62 | 5 |
| p3 ouro q0.4 ORACLE | 0.4 | 256/32 | 2.87 / 0.226 | 1022→763 (25.3%) | 22.32→16.73 s (25.1%) [16.67–19.17] | 25.40→19.06 s (25.0%; ×1.333) [18.99–21.64] | 21.84/21.92 | 23.0/30.8 | 0.99 / 0.98 | 5 |
| p3 huginn 0.12 ORACLE | 0.12 | 256/32 | 8.64 / 0.137 | 3031→2376 (21.6%) | 19.82→16.21 s (18.2%) [16.21–16.24] | 23.60→19.61 s (16.9%; ×1.203) [19.60–19.64] | 6.54/6.82 | 23.4/29.8 | 0.84 / 0.78 | 5 |
| p3 huginn 0.12 C8 predictor | 0.12 | 256/32 | 8.64 / 0.137 | 3031→2891 (4.6%) | 19.82→19.12 s (3.5%) | 23.60→23.78 s (−0.8%; ×0.992) | 6.54/6.61 | 23.4/24.5 | 0.76 / −0.17 | 5 |
| p3 huginn 0.26 ORACLE | 0.26 | 256/32 | 6.22 / 0.123 | 2075→1696 (18.3%) | 14.30→12.07 s (15.6%) [12.02–18.65] | 17.70→15.24 s (13.9%; ×1.162) [15.19–23.54] | 6.89/7.11 | 24.5/30.0 | 0.86 / 0.76 | 5 |
| p3 ouro / huginn N=B controls | – | 32/32 | – | 126→126 / 365→365 | 0.5% / 0.4% | 0.4% / 0.3% | – | – | – | 5 |

(Full rows, incl. C7/C9 and q0.4/0.26 C8, are in `e1_regime_cells.csv`. Phase03 Huginn baseline is
`STAGED_FIFO`; predictor walls include ~2% predictor+sort overhead, `selection_overhead_ratio`.) The
q0.4/0.26 phase03 cells contain one slow outlier repeat per strategy (max column); medians are used and
outliers are retained, not dropped.

### RQ1 (E1 part): is physical gain explained by removed whole Core invocations?

- **Ouro-1.4B: yes, within the measured cells.** Core-time reduction equals RIC reduction to within
  0.01–0.05 (tracking 0.95–1.00) and per-call Core time is unchanged by Oracle grouping
  (20.64 vs 20.66 ms/call) even though Oracle raises mean active width from ~20 to ~29 rows. In the
  per-width receipts (`e1_core_call_cost_by_width.csv`) an Ouro Core call costs ~19.5 ms at width 1 and
  ~20.6 ms at width 9–32: time is set by the number of calls, not by filled slots. Wall reduction is
  equal or slightly larger (tracking 1.00–1.03).
- **Huginn-3.5B: mostly, but attenuated.** Core reduction is 0.84–0.88 of RIC reduction and wall
  reduction 0.72–0.78. The measured per-call Core cost rises with active width (~5.8 ms at width 1 to
  ~7.1 ms at width 32 for q0.12), and Oracle calls are wider (29.7 vs 23.6 rows), so each surviving call
  costs ~3–4% more. Non-Core decode time (Prelude/Coda, KV, staging) is 17–26% of FIFO wall and is not
  reduced by regrouping, which further dilutes wall gain.
- N=B controls (grouping cannot change RIC) stay within ±0.6% Core / ±0.5% wall for both models.
- Practical predictors in phase03 remove few calls (3–6% RIC); Ouro Core still tracks RIC (~1.0) but the
  ~2% selection overhead halves the wall gain; Huginn wall is negative.

The slot-utilization gain alone does not predict speed: what matters in these receipts is the count of
whole Core invocations, weighted (on Huginn only) by a modest width-dependent call cost. This is a
statement about these B=32 (and one B=16), short-context, 32-token cells only.

## Trace-only regime structure (not speed)

At N256/B32/64 tokens, FIFO→Oracle RIC reduction: Ouro q0.2 32.2% (Alpaca) / 32.9% (ShareGPT), q0.4
25.6/25.8%, q0.5 17.5/17.2%, q0.7 5.8/6.0%; Huginn 0.12/0.26/0.35/0.50: 22.0/19.6/18.0/15.0%.
Across the 64-token sweep (N>B), reduction correlates with depth CV (Pearson r 0.83 Ouro, 0.37 Huginn)
much more than with log2(N/B) at fixed model (0.18, 0.03) because B and N/B are confounded in the
N∈{32..256}×B∈{4..32} grid: at fixed B, reduction rises with N (e.g. Huginn 0.12 B32: 0→11.8→18.7→22.0%
for N/B 1→8), while at fixed N/B larger B usually gains more. In phase01 (N fixed at 256) the Ouro
reduction peaks at B16–32 and falls at B64 (fewer microbatches per boundary, N/B=4). Plots:
`ric_reduction_vs_N_over_B.png`, `ric_reduction_vs_B.png`, `ric_reduction_vs_depth_cv.png`.
Model and KV policy are confounded (each model has exactly one KV policy), so "stratified by KV" is the
same split as "stratified by model". Ouro trace-only rows use native-reference post-hoc traces on
uncapped prompts; the phase02 physical Ouro runs cap prompts at 64 tokens (4/256 Alpaca and 80/256
ShareGPT prompts truncated) — the FIFO/Oracle pair inside each physical cell is matched.

## Latency model (phase01 replay) vs trace

For Ouro-1.4B the replayed decode reduction at B4/B8 is close to the RIC reduction (e.g. Alpaca q0.2 B8:
RIC 30.4%, model 29.0%). For Ouro-2.6B it is erratic (B4 model ≈ half of RIC, B8 model exceeds RIC: e.g.
Alpaca q0.2 B8 RIC 30.7% vs model 36.4%) because its calibration fits are non-monotone in B (e.g.
context128 step ms 26.5/32.1/46.1/27.3/67.2 for B1/2/4/8/16; several r² 0.75–0.85). These are model
outputs from native-reference HF fixed-depth calibration; there is **no** physical survivor-runtime cell
at B4/B8 or for Ouro-2.6B to validate them. The calibration surface (`e1_calibration_surface.csv`) is
noisy and is **not** used to infer a compute-, weight- or KV-bound regime; a flat or non-monotone
time-versus-B curve is not evidence of a hardware regime.

## Live exit (phase04, N32 × 8 positions)

Live BF16 batched RIC differs from the frozen-B1 grouping RIC of the same order by −2…+3 calls
(≤1.7%; work ±4 of 636/2118). TRACE_ONLY Oracle headroom on the same canonical depths is small at this
pool size (Ouro 5.9/10.6/8.8%, Huginn 2.2/4.1/6.2% for B2/4/8). No live Oracle was executed and no
timing was scored, so no live speed statement is possible.

## Cells lacking comparable evidence

- **Ouro-2.6B**: trace-only and latency-model only; no physical cell, raw traces not committed.
- **Physical B other than 32**: only Ouro N128/B16 (3 repeats, below the ≥5 rule). No physical B4/8/64.
- **Context/horizon**: all physical cells are prompt cap 64 + 32 decode tokens; no 64-token physical
  horizon, no long-context (512–2048) physical cell, no mixed-length context model.
- **Huginn latency model**: none (phase01 Huginn batched adaptive path failed; no Huginn replay).
- **Live exit**: RIC counts only, N32, no timing, no Oracle; phase04 Gate A remains FAIL.
- **Stage breakdown**: Prelude/Coda/KV timers overlap and are not an additive decomposition; E2 is
  needed for per-stage B/L curves and any bottleneck attribution (RQ2).
- Dataset revisions are `UNKNOWN` in all source manifests; prompt/trace SHA-256 are bound instead.
