# Phase 02 — Oracle recurrent-invocation headroom review (2026-09-16)

## Answer to the immediate question

The large offline ready pool does create material Oracle headroom on Huginn, but less than the strongest Ouro trace cells. With 256 Huginn requests, B32, 64 fixed decode tokens, and shared-KV B=1 live exit depths, the maximum observed **trace-only** recurrent-invocation count (RIC) reduction is **22.0%** at threshold 0.12. The other Huginn thresholds yield 19.6% (0.26), 18.0% (0.35), and 15.0% (0.50). This is a count of model-core calls, **not** a measured speedup.

The corresponding selected threshold-0.12, 32-token **physical survivor** run on H100 GPU7 executed 2,995 FIFO versus 2,377 Oracle recurrent calls (−20.6%) on identical forced-depth request/token work. Across two warmups and five measured repetitions per strategy, median recurrent CUDA-event time was 20.22→16.66 s (−17.6%), and median prefill-excluded decode wall time was 24.30→20.37 s (−16.2%). This is an instrumented, perfect-depth Oracle result; no practical predictor was trained or timed.

The smaller physical Huginn opportunities followed the trace ranking: threshold 0.26 yielded −16.9% calls and −12.6% wall time; threshold 0.50 yielded −15.3% calls and −11.0% wall time. The N=B=32 control had exactly the same 361 calls and only a noisy 0.5% wall difference. Therefore the observed mechanism is removable whole recurrent invocations at larger N/B, not a generic benefit from reordering one microbatch.

## Trace-first characterization

Huginn-3.5B (revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`) was run with BF16, shared physical-layer KV, its latent-difference exit rule, Rmax16, native-seeded state, Alpaca-sourced prompts (seed 1; 64-token prompt cap), N256, and 64 fixed decode tokens. Each of four thresholds has a complete 16,384-row B=1 live trace. The collector uses the same adapter and token-boundary cache semantics as the physical runtime; a separate N4×2-token comparison found identical eight exit depths and prompt hashes against the original live benchmark path. The initial large collector was stopped at 74,219 MiB after repeated full-history KV concatenation caused memory growth; [the failure receipt](physical/gpu7_huginn/phase02_physical_gpu7_huginn_20260916T130250Z_4129364/failure_receipt.json) is retained. Commit `48ed8d9` changed shared-KV history to geometric capacity plus in-place token-boundary updates. The complete collector subsequently held peak reserved HBM at 15.74 GiB for each threshold.

For Ouro-1.4B (revision `37bc7e8ff85a90d813dc096383405f521b6a67b0`), the sweep reuses eight existing Phase-1 **native-reference, post-hoc learned-gate** Alpaca/ShareGPT traces at q=0.2/0.4/0.5/0.7. They are exit-depth inputs, not evidence that the Phase-1 runtime physically stopped at the exit. New model executions in this block used only GPU6 (Ouro) and GPU7 (Huginn).

All trace-only cells use the same frozen first-N request IDs and 64 token boundaries for FIFO and Oracle. Oracle sorts by that token's true depth, then stable request ID; no refill occurs, and every N is divisible by B. The [ranked 64-token sweep](trace_sim_64tok/oracle_ric_sweep.csv) covers 12 model/source/threshold inputs × N32/64/128/256 × B4/8/16/32 = 192 cells. [Depth distributions](trace_sim_64tok/depth_distributions.csv), [per-token RIC rows](trace_sim_64tok/token_boundary_ric.csv), and a [32-token horizon sweep](trace_sim_32tok/oracle_ric_sweep.csv) are also published. The CSVs contain exact trace hashes, same-work checksums, slot utilization, microbatch-max distributions, and per-token RIC dispersion.

| Trace-only input, N256/B32, 64 tokens | Depth p10/p50/p90 | FIFO→Oracle RIC | RIC reduction |
| --- | ---: | ---: | ---: |
| Huginn/Alpaca threshold 0.50 | 4/4/5 | 2,640→2,243 | 15.0% |
| Huginn/Alpaca threshold 0.35 | 5/5/6 | 3,503→2,873 | 18.0% |
| Huginn/Alpaca threshold 0.26 | 5/6/7 | 4,290→3,449 | 19.6% |
| Huginn/Alpaca threshold 0.12 | 7/9/10 | 6,151→4,797 | 22.0% |
| Ouro/Alpaca q0.2 | see depth CSV | 1,809→1,227 | 32.2% |
| Ouro/ShareGPT q0.2 | see depth CSV | 1,887→1,266 | 32.9% |

For Huginn threshold 0.12 at B32, increasing the ready pool from N=B=32 to N64, N128, and N256 changes RIC reduction from 0.0% to 11.8%, 18.7%, and 22.0%. Thus a small N32 smoke does not measure the offline Oracle ceiling. Across the 64 token boundaries of the N256/B32 threshold-0.12 cell, the per-boundary reduction p10/p50/p90 is 18.2%/22.1%/25.5%. The zero N=B controls are explicitly retained in the full sweep.

## Physical FIFO/Oracle confirmation

All selected cells use the real no-refill survivor runtime. The trace controls the exit depth; the model computes recurrent layers on the surviving tensor rows and updates its model-specific KV backend. Each cell has the same request IDs, same generated-token count, and same per-request depth work checksum. An independent audit matched **all nine physical cells** to the 32-token trace sweep on both FIFO/Oracle RIC and the work checksum. Timings below are medians of the measured repetitions after warmups; prefill and model load are excluded from decode wall time. [The full physical matrix](physical_matrix.csv) gives each time, overhead component, run directory, prompt hash, trace hash, source SHA, GPU UUID, and peak HBM. Compressed per-call events and unmodified manifests are under `physical/`.

| Model / depth trace / physical prompt source | N/B, tokens; warmup+repeats | FIFO→Oracle calls | Recurrent GPU time reduction | Decode wall reduction |
| --- | ---: | ---: | ---: | ---: |
| Huginn 0.12 / Alpaca | 256/32, 32; 2+5 | 2,995→2,377 (−20.6%) | 17.6% | 16.2% |
| Huginn 0.26 / Alpaca | 256/32, 32; 2+5 | 2,054→1,707 (−16.9%) | 14.3% | 12.6% |
| Huginn 0.50 / Alpaca | 256/32, 32; 2+5 | 1,317→1,116 (−15.3%) | 13.5% | 11.0% |
| Huginn 0.12 N=B control / Alpaca | 32/32, 32; 2+5 | 361→361 (0.0%) | 0.6% | 0.5% |
| Ouro q0.2 / Alpaca | 256/32, 32; 2+5 | 935→655 (−29.9%) | 29.9% | 30.9% |
| Ouro q0.4 / Alpaca | 256/32, 32; 2+5 | 1,018→780 (−23.4%) | 23.5% | 23.4% |
| Ouro q0.2 / ShareGPT | 256/32, 32; 2+5 | 954→667 (−30.1%) | 30.0% | 31.1% |
| Ouro q0.2 / Alpaca, smaller pool | 128/16, 32; 2+3 | 877→644 (−26.6%) | 25.4% | 27.5% |
| Ouro N=B control / Alpaca | 32/32, 32; 2+5 | 112→112 (0.0%) | −0.3% | 0.1% |

The Huginn threshold-0.12 32-token result is narrower than its 64-token trace result: −20.6% versus −22.0% RIC. Its 16.2% wall reduction is also smaller than its 20.6% call reduction. The measured prelude/coda times stay around 1.1/1.06 s per strategy, and KV-update time changes less than core time; fixed and cache costs plausibly attenuate the wall gain. These timing categories can overlap and are not summed as a decomposition. The physical FIFO/Oracle median comparison, not trace-only slot utilization, is the speed evidence.

## Comparison limits and decision

Huginn's physical Alpaca prompt hash exactly matches its B=1 trace collector (256 requests, seed 1, cap 64). The Ouro Phase-1 depth traces were collected on uncapped native prompts; the new physical benchmark capped all prompts at 64 tokens. Four of 256 Alpaca prompts and 80 of 256 ShareGPT prompts were therefore truncated in the Ouro physical runs. **Within each physical FIFO/Oracle pair the prompt and forced depth work are identical**, so the hardware comparison remains valid; the Ouro results must not be interpreted as a native live-exit or original-context dataset throughput result. The physical model outputs may diverge between strategies and from the B=1 source trace; no quality equivalence is asserted. Dataset revisions were not pinned in the run manifests and remain `UNKNOWN`; prompt SHA-256, seed, and exact trace hashes are bound instead.

This is an offline, pure no-refill Oracle bound for a fixed 32/64-token horizon, not production serving, refill, native/full-depth quality, GSM8K, perplexity, or a practical predictor result. The Huginn source uses BF16 and head-dim-96 SDPA math; Ouro uses BF16 and its last-exited depth-indexed KV policy. Peak-HBM values in run summaries include prefill even though reported decode wall excludes it.

**Decision: proceed to a bounded practical-predictor study targeting expected microbatch maximum depth/RIC**, rather than depth MAE alone. The decision is a systems-research GO: large ready pools produced 22.0% Huginn trace RIC headroom and 16.2% measured Oracle wall improvement, while zero-headroom controls stayed near zero. It is not a quality, predictor-recovery, or deployment GO. Because Huginn wall gains lagged RIC gains, measure prelude/coda, KV, and attention/kernel costs alongside any later predictor result; do not assume 22% can be recovered as throughput.

## Reproduction and audit

The published `.jsonl.gz` files under [trace_inputs](trace_inputs/) are lossless copies; `phase02_runtime.oracle_ric` reads them directly and hashes the decompressed bytes, matching the original run manifests. Re-run CPU checks with `python -m unittest phase02_runtime.test_oracle_ric phase02_runtime.test_summarize_oracle_physical phase02_runtime.test_runtime -v`. The [trace-simulation manifests](trace_sim_64tok/run_manifest.json) record all 12 inputs and the exact analysis source SHA. Physical run manifests under `physical/gpu6_ouro/` and `physical/gpu7_huginn/` record the exact runner argv, source SHA, model revision, prompt/trace hashes, dataset, N/B, threshold, seed, GPU identity, and KV backend. `recurrence_events.jsonl.gz` contains active-width and per-call timing receipts. Exploratory three-repeat Ouro runs and the interrupted Huginn memory-growth run are preserved but not substituted for the final five-repeat cells.
