# Phase 02 — physical survivor-compaction smoke review (2026-09-16)

## First question and answer

With **the same forced exit-depth work**, does actual no-refill survivor compaction plus Oracle micro-batch formation reduce H100 time versus FIFO? **Sometimes, not universally.** The recurrent core really received smaller tensors after exits on both models. Oracle helped when sorting caused a micro-batch to drain early enough to remove a recurrent-core invocation; when the call count stayed the same, reduced active width alone often produced negligible latency change. These are physical model executions, not trace-latency replay. The trace controls only *when* a request exits.

The selected 32-token, B16/N32 examples show the practical size of this initial systems signal (medians of two warmed runs, prefill excluded):

| Model / forced-depth source | Same recurrence work | Recurrent GPU time, FIFO → Oracle | Instrumented decode wall, FIFO → Oracle |
| --- | ---: | ---: | ---: |
| Ouro-1.4B, q0.4 Phase-1 trace | 2,951 | 4,982 → 4,562 ms (−8.4%) | 5,657 → 5,151 ms (−9.0%) |
| Huginn-3.5B, threshold 0.5 B1 shared-KV trace | 4,106 | 1,797 → 1,694 ms (−5.7%) | 2,512 → 2,398 ms (−4.5%) |

These timings are **instrumented smoke results**, not a production throughput or quality claim. Oracle has perfect depth knowledge, and no practical predictor was timed. See the exact per-run CSVs and manifests in [the M2 matrix](m2_matrix.csv) and the `gpu6_ouro/` and `gpu7_huginn/` directories.

## What ran

- Pulled main at `94b0245`, read `AGENTS.md` and the Phase-2 contract, then committed source before GPU runs. GPU6 ran only Ouro-1.4B (revision `37bc7e8`); GPU7 ran only Huginn-3.5B (revision `bb6621b`). The physical GPU UUIDs, full source SHAs, model revisions, trace hashes, and invocation arguments are in each `run_manifest.json` and [review_manifest.json](review_manifest.json).
- The common runtime scatters exited final states, gathers only survivors, and preserves stable logical request/KV-row IDs. Ouro uses depth-indexed last-exited KV copies; Huginn overwrites one KV slot per physical recurrent layer/request. Huginn's head-dim-96 attention uses an explicit causal/padding mask with SDPA math, without editing the cached model source.
- M0 forced-depth smoke passed on both GPUs: Ouro `[1,2,4,2]` yielded active widths `4,3,1,1`; Huginn `[1,2,16,2]` yielded `4,3,1` through recurrence 16. Ouro last-exited fill receipts and both active-ID event logs are preserved. Four CPU invariant tests passed. Both models produced finite smoke logits.
- The Huginn heterogeneous-length padding check compared B4 against each B1 prompt (6/14/20/26 tokens): all four top-1 tokens matched; normalized logit RMS differences were 0.0055–0.0067, below the explicit 0.01 check ([receipt](huginn_padding_check.json)). This is a systems masking check, not a model-quality evaluation.

## One-token M1: where Oracle did and did not help

All selected cells used N32, one generated token, two warmups per strategy, then 3–5 repeats. The table reports median **recurrent CUDA-event time** reduction, `100 × (FIFO − Oracle) / FIFO`; full milliseconds, wall time, source IDs, and equal-work counts are in [m1_matrix.csv](m1_matrix.csv).

| Model / threshold | B8 | B16 | B32 |
| --- | ---: | ---: | ---: |
| Ouro q0.2 | 7.3% | 0.1% | −0.2% |
| Ouro q0.4 | 12.5% | 12.6% | −0.3% |
| Huginn 0.35 | 5.2% | 0.4% | 0.8% |
| Huginn 0.5 | 5.9% | 0.1% | −0.2% |

For example, Ouro q0.4/B16 had identical total recurrent work (106 request-steps). FIFO's two micro-batches both ran four recurrences, with active widths `16,16,16,5` each. Oracle ran `16,16,16` and `16,16,16,10`: one fewer full model recurrence invocation. Huginn 0.5/B8 similarly went from 17 FIFO recurrence invocations to 16 Oracle invocations. By contrast, B32 is a single micro-batch, so sorting cannot remove a separate batch's recurrence invocation; the near-zero results are expected. The q0.2/B16 stable repeat also showed almost no recurrent-time benefit despite changed membership. This is why logical headroom alone cannot be reported as physical speedup.

The first Ouro q0.2/B16 two-repeat run and the Huginn synthetic-depth exploratory M1 are retained in the raw bundle but **excluded** from the matrix: the former had insufficient warmup and unstable first-repeat timing, and the latter preceded the model-specific B1 exit trace. Both are described in [review_manifest.json](review_manifest.json).

## Multi-token M2 and boundaries

After M1, fixed 16- and 32-token B16/N32 runs completed for Ouro q0.4 and Huginn threshold 0.5. FIFO and Oracle executed exactly the same total request-recurrence work within each cell (see [m2_matrix.csv](m2_matrix.csv)); the cache length advanced at token boundaries, not during survivor compaction. Ouro's 16-token core/wall reductions were 11.4%/11.6%, and Huginn's were 6.1%/4.8%. The 32-token values are in the opening table. These M2 cells have only two warmed repeats per strategy and should be treated as an implementation gate, not a broad performance characterization.

The prompt content is deterministic synthetic text with heterogeneous lengths. Ouro's published Phase-1 Alpaca traces and Huginn's locally collected shared-KV B1 traces provide **depths only**; no Alpaca/ShareGPT workload-throughput claim is made. B1 live-exit timing is kept separate from trace-forced FIFO/Oracle timing. No GSM8K, perplexity, native full-depth quality, or practical-predictor run was performed. No GPU 0–5 was used.

## Evidence and reproduction

Every raw run directory under `gpu6_ouro/` or `gpu7_huginn/` contains `run_manifest.json`, `summary.csv`, and `recurrence_events.jsonl`; the B1 live runs also include `live_exit_depth_trace.jsonl`. The published [trace_inputs](trace_inputs/) preserve the full Ouro q0.2/q0.4 trace files with hashes in the review manifest. `recurrence_events.jsonl` gives active B, logical IDs on debug runs, per-recurrence GPU time, compaction, KV work, prelude, and coda. The manifests distinguish prefill-excluded decode wall time from peak HBM, which includes prefill.

Re-run the CPU checks with `python -m unittest phase02_runtime.test_runtime -v`. On a host with the pinned local model snapshots, use `./run_phase02_gpu6_ouro.sh` or `./run_phase02_gpu7_huginn.sh` with `--stage m1`/`m2`, `--trace-file`, B/N/token arguments, and a fresh result root; each runner rejects an incorrect visible physical GPU. For the Huginn padding check, set `CUDA_VISIBLE_DEVICES=7` and run `python -m phase02_runtime.check_huginn_padding`.
