# Ouro has substantial oracle scheduling headroom; Huginn does not

## Executive summary

- The GPU 1 PoC completed successfully on 2026-09-15 from batch-loop commit
  [`830eb5a`](https://github.com/heonniy/batch-loop/commit/830eb5a0259cff9305d55a0cddd23bebb0825faa).
- Ouro-1.4B shows substantial oracle headroom at practical simulated batch sizes:
  22.13 percentage points at B=4, peaking at 27.51 points at B=16. The causal
  predictors recover only about 1.6-2.4 points over FIFO at B=4-64.
- Huginn starts near 94-95% FIFO slot utilization. Oracle grouping adds at most
  4.20 points at B=4 and 1.75 points at B=16; causal grouping is negligible or
  negative at most tested sizes.
- This is a trace-based slot-utilization result, not an end-to-end latency result.
  The evidence supports continuing an Ouro-focused scheduler/runtime experiment,
  while the measured Huginn configuration has little scheduling headroom.

## Oracle headroom

`oracle_headroom_pp` is oracle slot utilization minus FIFO slot utilization,
reported in percentage points.

| Micro-batch | Ouro FIFO | Ouro oracle | Ouro headroom | Huginn FIFO | Huginn oracle | Huginn headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 77.02% | 99.15% | 22.13 pp | 94.99% | 99.19% | 4.20 pp |
| 8 | 72.12% | 98.10% | 25.98 pp | 94.50% | 97.89% | 3.40 pp |
| 16 | 68.23% | 95.74% | 27.51 pp | 94.07% | 95.82% | 1.75 pp |
| 32 | 64.48% | 90.98% | 26.50 pp | 93.59% | 93.78% | 0.19 pp |
| 64 | 61.68% | 79.96% | 18.28 pp | 92.89% | 93.34% | 0.44 pp |
| 128 | 61.84% | 68.22% | 6.37 pp | N/A | N/A | N/A |
| 256 | 76.32% | 76.32% | 0.00 pp | N/A | N/A | N/A |

Huginn B=128 and B=256 are unavailable because no token position contained a
complete micro-batch at those sizes. These missing measurements are not zeros.
At Ouro B=256 there is only one full-batch ordering, so all strategies coincide.

## Causal depth predictability

| Model | Predictor | MAE (steps) | Exact accuracy | Pearson correlation |
| --- | --- | ---: | ---: | ---: |
| Ouro-1.4B | previous token | 0.598 | 51.1% | 0.400 |
| Ouro-1.4B | recent-4 mean | 0.579 | 50.3% | 0.465 |
| Ouro-1.4B | sequence mean | 0.604 | 50.6% | 0.441 |
| Huginn | previous token | 1.662 | 71.9% | 0.073 |
| Huginn | recent-4 mean | 2.247 | 31.1% | 0.066 |
| Huginn | sequence mean | 2.566 | 1.4% | 0.050 |

The high Huginn previous-token exact accuracy coexists with near-zero correlation
because most observed depths are concentrated near the maximum. It does not
translate into meaningful grouping gains.

## Experiment scope

| Setting | Ouro-1.4B | Huginn |
| --- | ---: | ---: |
| Dataset | Alpaca | Alpaca |
| Requests | 256 | 128 |
| Maximum generated tokens | 24 | 24 |
| Maximum prompt tokens | 512 | 512 |
| Exit threshold | learned gate, 0.2 | latent difference, 0.12 |
| Maximum recurrent depth | 4 | 16 |
| Trace rows | 4,817 | 724 |

- Hardware: NVIDIA H100 80GB HBM3, physical GPU 1; driver 570.211.01.
- Runtime: Python 3.11.15, PyTorch 2.6.0+cu124, Transformers 4.54.1,
  Accelerate 1.15.0, Datasets 5.0.1, NumPy 2.4.6, Pandas 3.0.5, and
  Matplotlib 3.11.2.
- Dataset revision: `tatsu-lab/alpaca@dce01c9b08f87459cf36a430d809084718273017`.
- Huginn revision: `tomg-group-umd/huginn-0125@bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
- Ouro weights/code source: `ByteDance/Ouro-1.4B`. The run pinned the last
  working official pre-cache-update code revision
  `37bc7e8ff85a90d813dc096383405f521b6a67b0`; current revision
  `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` failed against the repository's
  supported Transformers range because its custom cache is incompatible with the
  base `Cache` API.
- Huginn needed the process-local cache-property compatibility shim in
  [`repro/sitecustomize.py`](repro/sitecustomize.py). No tracked source code was
  changed for the experiment.

## Artifacts

- Combined metrics: [`analysis/poc/summary.csv`](analysis/poc/summary.csv),
  [`analysis/poc/predictability.csv`](analysis/poc/predictability.csv), and
  [`analysis/poc/valley_curves.csv`](analysis/poc/valley_curves.csv).
- Ouro plots and per-run evidence:
  [`analysis/poc/ouro14_alpaca_q02__.cache_ouro-1.4b-rev37/`](analysis/poc/ouro14_alpaca_q02__.cache_ouro-1.4b-rev37/).
- Huginn plots and per-run evidence:
  [`analysis/poc/huginn_alpaca_d012__tomg-group-umd_huginn-0125/`](analysis/poc/huginn_alpaca_d012__tomg-group-umd_huginn-0125/).
- Raw traces: [`traces/ouro14_alpaca_q02.jsonl`](traces/ouro14_alpaca_q02.jsonl)
  and [`traces/huginn_alpaca_d012.jsonl`](traces/huginn_alpaca_d012.jsonl).
- Synthetic sanity check: [`analysis/synthetic/`](analysis/synthetic/) and
  [`traces/synthetic.jsonl`](traces/synthetic.jsonl).
- Full execution log: [`run_gpu1_20260915.log`](run_gpu1_20260915.log).
- Reproduction launcher: [`repro/run_gpu1_poc_compat.sh`](repro/run_gpu1_poc_compat.sh).

## Limitations

- Slot utilization counts recurrent token-loop work; it does not measure H100
  wall-clock latency or scheduler overhead.
- Ouro's public implementation computes all recurrent steps and exposes its gate
  as a post-hoc readout, so this run estimates scheduling opportunity rather than
  real early-exit speedup.
- The Ouro compatibility pin predates the upstream cache update. Results should be
  rerun after the upstream model provides a working cache implementation for the
  supported Transformers release.
- Request termination reduces the number of live requests at later token indices.
  The simulator intentionally excludes incomplete micro-batches.
