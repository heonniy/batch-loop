# E1 — Retrospective regime re-analysis (method and reproduction)

CPU-only re-analysis of committed phase01–04 receipts. No model is loaded and no CUDA device is used.
Results: `phase05/results/20261019_e1/RESULTS.md`.

## Reproduce

```bash
cd <repo root>
export CUDA_VISIBLE_DEVICES=""
PY=/home/hwlee/batch-loop/.venv/bin/python        # numpy, pandas, matplotlib
$PY phase05/scripts/e1_regime_reanalysis.py --smoke        # tiny synthetic input, writes nothing
$PY phase05/scripts/e1_regime_reanalysis.py                # -> phase05/results/20261019_e1/
$PY phase05/tests/test_e1.py -v                            # arithmetic + output-contract tests
```

Runtime ≈ 10 s. Output is deterministic (fixed orders, no sampling). `e1_input_manifest.json` records the
command, git HEAD, script SHA-256, package versions, and for every input file its path, SHA-256 (raw bytes;
plus decompressed SHA-256 for `.jsonl.gz`), row count and column schema.

## Inputs

| Phase | Files | Use |
| --- | --- | --- |
| phase01 `results/20260916_four_gpu_validation/` | `ouro_headroom_fair.csv`, `ouro_trace_audit.csv`, `ouro_fairness_validation.csv` | TRACE_ONLY FIFO vs Oracle/practical RIC (RIC = capacity slots / B), depth histograms |
| | `ouro_hardware_replay_b4_b8.csv` | LATENCY_MODEL replay (decode_replay_ms; BCT incl. prefill noted) |
| | `ouro_calibration.csv`, `ouro_calibration_fit.csv` | calibration surface (LATENCY_MODEL); OOM cells listed as excluded |
| | `ouro_native_e2e.csv` | excluded entirely (native HF post-hoc exit; 20 OOM-partial cohorts) |
| phase02 `oracle_upper_bound_results_20260916/` | `trace_inputs/*.jsonl.gz` | raw B1 depths; every trace-only RIC is recomputed from these |
| | `trace_sim_{32,64}tok/oracle_ric_sweep.csv` | TRACE_ONLY rows, reconciled against the recompute |
| | `physical_matrix.csv`, `physical/*/*/{summary.csv,recurrence_events.jsonl.gz,run_manifest.json}` | PHYSICAL_FORCED_DEPTH medians/spread, per-call width and cost |
| phase03 `predictor_tournament_results_20260916/` | `*_physical/`, `*_n_eq_b_control/` summaries + events, `*/depth_labels.npy`, matrices | PHYSICAL_FORCED_DEPTH FIFO/Oracle/predictor rows, held-out ids 384–639 |
| phase04 `results/…_bf16_resolution/`, `…_rank_stability_cpu/` | `per_token_validation.csv`, `batch_shape_validation.csv`, rank-stability CSVs | TRACE_ONLY (canonical B1) and PHYSICAL_LIVE_EXIT (live RIC counts, no timing) |
| external, read-only | `/home/hwlee/batch-loop/results/oracle_ric_*/oracle_ric_sweep.csv` | hashed and compared to the repo 64-token sweep; not copied |

## Definitions

For a decode boundary t, microbatches m (requests chosen before the recurrent interval; no refill) and
per-token depths d_i:

- `RIC = Σ_t Σ_m max_{i∈m} d_i` (whole recurrent Core invocations under survivor compaction)
- `work = Σ_t Σ_i d_i` (token-loop work; identical across policies by construction)
- `slot utilization = work / Σ_t Σ_m |m|·max_{i∈m} d_i`
- FIFO: request order chunked by B. Oracle: per-boundary stable sort by true depth, then request id.
  Sorting is optimal for Σ max over equal-size groups (brute-force checked in the tests).
- `reduction = (T_fifo − T_policy)/T_fifo`; `speedup = T_fifo/T_policy = 1/(1−reduction)`. Both are
  reported separately for RIC, Core, wall and model time.
- `tracking ratio = measured (Core|wall) reduction / RIC reduction` (undefined for N=B controls).
- `ms/call = median Core ms / RIC`; `mean active width` from recurrence-event receipts.
- `noncore_wall_fraction = 1 − Core/wall`; `selection_overhead_ratio = (predictor+sort wall)/policy wall`.
- Medians over measured repetitions only (warmups excluded); min/max kept as spread; outliers retained.
- Depth stats (mean, population std, CV, p10/p50/p90) are computed on the exact request×token cohort of
  the row (first N requests × first T tokens, or held-out ids for phase03).

## Evidence classes and guards

Each output row has exactly one of `TRACE_ONLY`, `LATENCY_MODEL`, `PHYSICAL_FORCED_DEPTH`,
`PHYSICAL_LIVE_EXIT`, plus `source_files`, `source_rows` (file:1-based data row) and a `reconciliation`
status. `validate_cells` refuses any non-physical row carrying measured Core/wall time and any non-model
row carrying model time. Missing values are written as blank (NaN), never zero. Live-exit rows leave
`ric_reduction` blank because no live Oracle was executed.

## Outputs (`phase05/results/20261019_e1/`)

`e1_regime_cells.csv` (959 rows), `e1_excluded_rows.csv` (98), `e1_input_manifest.json`,
`e1_calibration_surface.csv`, `e1_core_call_cost_by_width.csv`, `RESULTS.md`, and plots
`ric_reduction_vs_N_over_B.png`, `ric_reduction_vs_B.png`, `ric_reduction_vs_depth_cv.png`,
`physical_reduction_vs_ric_reduction.png`.

## Deviations / limits

- The task referenced `trace_sim_64tok/oracle_ric_sweep.csv` as missing; it is present and used.
- Ouro-2.6B rows cannot be recomputed (raw traces not committed); they rely on phase01 tables.
- Model and KV policy are confounded (one KV policy per model); stratification by KV equals by model.
- No phase05/README.md or REVIEW.md was written by this E1 step (shared with E2/E3 work).
