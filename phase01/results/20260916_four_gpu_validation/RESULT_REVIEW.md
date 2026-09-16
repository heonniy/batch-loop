# Four-GPU model validation: Ouro trace headroom survives; Huginn main lane stops at the KV/batch gate

## Executive summary

- Ouro-1.4B and Ouro-2.6B each completed all 8 Alpaca/ShareGPT × threshold exit traces: **16 traces, 256 requests each, 393,216 recorded token positions**. All 80 fair-subset checks passed. At B=64 the oracle-minus-FIFO *logical slot-utilization* gain spans **4.3–28.0 percentage points**; the best tested history-only grouping gains at most **3.2 pp**. This is scheduling headroom, not GPU speedup.
- Native whole-model E2E has 80 model × workload × threshold × B rows: **60 complete 256-request cells, 20 ShareGPT OOM-partial cells**. Maximum fully completed B is 64 for Alpaca, 16 for Ouro-1.4B ShareGPT, and 8 for Ouro-2.6B ShareGPT. Retain the partial rows as capacity evidence; do not compare their throughput with complete cells.
- Huginn's corrected Rmax=16 KV gate has **23 OOMs in 168 fixed-depth surface cells**, plus both explicit B64/context2048/depth16 target probes OOM. Both tested adaptive B4 policies fail with an `E=96` batched-attention error; B1 output agreement is 0/4 against the policy reference in each gate comparison. The isolated GPU0 diagnosis confirms that compressed KV reduces one B8/context512 peak from about 22.0 to 9.95 GiB, but B64/context1536–2048 still OOMs even for one generated token. No Huginn canonical scheduling matrix or quality result is claimed.

## Evidence and interpretation

The table gives oracle-minus-FIFO slot-utilization gain in percentage points at B=64, on the same selected token IDs and loop work for every strategy. Higher threshold 0.7 is a useful negative/control region: FIFO is already efficient, leaving only 4.3–7.5 pp oracle headroom.

| Model | Workload | q=0.2 | q=0.4 | q=0.5 | q=0.7 |
| --- | --- | ---: | ---: | ---: | ---: |
| Ouro-1.4B | Alpaca | 23.4 | 21.0 | 16.0 | 4.3 |
| Ouro-1.4B | ShareGPT | 28.0 | 24.0 | 16.4 | 7.5 |
| Ouro-2.6B | Alpaca | 23.1 | 20.3 | 16.7 | 5.6 |
| Ouro-2.6B | ShareGPT | 26.1 | 22.5 | 17.0 | 7.4 |

These are `trace-sim` values from [ouro_headroom_fair.csv](ouro_headroom_fair.csv), not physical runtime measurements. The practical `prev`/`recent`/`seqmean` strategies recover little of the oracle bound (at B=64, best gain across the 16 cells is 3.21 pp). Raw trace completeness and depth distributions are audited in [ouro_trace_audit.csv](ouro_trace_audit.csv); [ouro_fairness_validation.csv](ouro_fairness_validation.csv) retains the per-B equal-work checks.

[ouro_native_e2e.csv](ouro_native_e2e.csv) preserves all complete and OOM-partial native rows, allocated/reserved HBM peaks, batch latency, original failure counts and messages. This is the public Ouro HF path: the gate is interpreted **post hoc after all recurrent steps are computed**. It is a whole-model baseline, not physical early exit or a no-refill engine. There was a B1 warmup per threshold, but no repeat campaign, so the timings are descriptive single-pass measurements only.

CUDA error strings may call the visible device “GPU 0” inside an isolated process; `gpu_physical` and [run receipts](run_receipts/) identify the host's physical GPU. That text does not mean these Ouro cells ran on physical GPU0.

There is also a denominator defect in the source E2E CSV: `_generated_lengths` stops at the first EOS token even when `min_new_tokens=max_new_tokens` forced fixed-length generation. The raw `generated_tokens` and raw generated-tokens/s columns therefore undercount work (the complete-cell raw counts are 86.0–98.5% of the fixed contract). The publication file retains them for audit, and adds `contract_generated_tokens_if_complete = 256 × {64,128}` and `contract_tokens_per_wall_s_if_complete` only where all 256 requests completed. No corrected token throughput is supplied for OOM-partial rows. `peak_gpu_reserved_gib` can include allocator reservations carried from preceding sweep cells; `peak_gpu_allocated_gib` is the per-cell reset peak.

The [Ouro calibration surface](ouro_calibration.csv) has 80/84 OK cells for 1.4B and 76/84 for 2.6B; the 12 OOM cells are preserved. [Fits](ouro_calibration_fit.csv) are measured latency-surface approximations, not an integrated scheduler. Replay is restricted to B=4,8 because both models have full-depth calibration at those B values across all measured contexts; higher-B/long-context fits are incomplete or absent. The [hardware-calibrated trace replay](ouro_hardware_replay_b4_b8.csv) has 160 fair rows. At B=8/q=0.2 it models oracle/FIFO BCT ratios of 1.40 (1.4B Alpaca), 1.47 (1.4B ShareGPT), 1.55 (2.6B Alpaca), and 1.61 (2.6B ShareGPT). These are **model outputs**, not observed no-refill throughput/BCT or native E2E speedups.

The corrected Huginn gate uses `latest-m4` (68 OK / 16 OOM surface cells) and `latest-m4-compress-s4` (77 OK / 7 OOM), with both B64/context2048/depth16 target probes OOM in [huginn_gate_capacity.csv](huginn_gate_capacity.csv). [B4 parity probes](huginn_gate_batch_parity.csv) complete 0/4 requests under both policies because the current batched adaptive path raises `ValueError: NYI: Currently non power of 2 embedding dimension are not supported. Got E=96 and Ev=96.` B1 [full vs latest-m4](lane_c2_huginn_latest_gate_b1_full_vs_latest_m4.json) and [latest-m4 vs compressed](lane_d2_huginn_compress_gate_b1_latest_m4_vs_compress_s4.json) each have 0/4 exact output agreement in the hashed-output receipts. This is a *gate failure*, not proof that either KV policy has poor GSM8K quality; the planned N=128 quality sanity was not run.

The isolated [GPU0 memory diagnosis](huginn_gpu0_memory.csv) runs at source `af049175aa05034779796d6e2fea0ee26e52685c`: 11 OK / 4 OOM among 15 probes. The B8/context512/depth16 one-token allocated peaks are 22.03 GiB (`full`), 22.03 GiB (`latest-m4`), and 9.95 GiB (compressed). Compressed B64/context1024 passes at 59.03 GiB allocated peak; B64/context1536 and 2048 OOM. B16/context2048 and B32/context1024 one-vs-eight-token probes have essentially unchanged peak allocated memory. This suggests the failing capacity boundary is already reached in prefill; it does **not** establish that decode or a future integrated engine is safe. The [GPU0 adaptive B1 comparison](huginn_gpu0_adaptive_b1_policy_compare.json) remains 0/4 exact agreement.

## Scope and provenance

| Evidence | Source commit | H100 binding | Checkpoint / runtime contract |
| --- | --- | --- | --- |
| Ouro calibration | `b15a3765c4517cf0d8ab1a9f33cfe083c098f41a` (inferred; see below) | physical 4 (1.4B), 5 (2.6B) | pinned revisions below; native-reference KV, Rmax=4 |
| Ouro trace/fair/native shards | `dbce2e8d1171faf3607f4e8043511c1d1e44ec46` | trace shards physical 4/6 (1.4B), 5/7 (2.6B); native ShareGPT 2.6B follow-ups physical 4/6 | learned gate q={0.2,0.4,0.5,0.7}, seed 1 |
| Huginn corrected KV gate | `d10c6b7` | physical 6 (`latest-m4`), 7 (compressed) | Rmax=16, latent-diff q=0.26 for parity; fixed decode adapter |
| Huginn isolated diagnosis | `af049175aa05034779796d6e2fea0ee26e52685c` | physical 0 only | fixed-depth memory and adaptive B1 policy probes |

Checkpoint revisions: Ouro-1.4B `37bc7e8ff85a90d813dc096383405f521b6a67b0`; Ouro-2.6B `f8e38372d72a8aa7f4ea9f14b98739a311589088`; Huginn-3.5B `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. H100 80GB HBM3; BF16 model loading. Package versions in the manifest were observed at publication, **not bound at each run**. The calibration files predate the `dbce2e8` shard restart (11:47–11:48 KST, before the 11:49 `d10c6b7` commit); their `b15a376` source assignment follows this chronology because the restart overwrote the original lane manifest. Original 2.6B ShareGPT E2E follow-ups also lack a self-contained run manifest: their GPU/source binding comes from launch context and the pinned worktree, rather than the CSV alone. Treat both as provenance limitations.

Alpaca uses `tatsu-lab/alpaca` train, cap 512 prompt tokens and fixed 64 decode tokens; ShareGPT uses `ewof/sharegpt-instruct-unfiltered-deduped` train, cap 2048 and fixed 128; 256 requests per workload with seed 1. Dataset `load_dataset` calls did **not pin a revision**. Current local cache has Alpaca snapshot `dce01c9b08f87459cf36a430d809084718273017` and ShareGPT snapshot `d4b3c8d85e6b9e0dd4e3ba85f046a9517a847d0c`, but those are post-hoc observations, not run-bound revision receipts. This limits exact future regeneration. Raw per-token traces are too large for this compact commit; they remain at the relative paths and SHA-256 values in the inventory. The committed tables, audit script, source commits and run commands support review without conflating raw/modelled/runtime evidence.

## Review status and next action

| Axis | Status | Basis |
| --- | --- | --- |
| Measurement | `VALID_IN_SCOPE` for traces; `DATA_DEFECT_DISCLOSED` for native raw token denominator | 16 complete trace cohorts; 20 E2E OOM-partial cells excluded from whole-cohort throughput |
| Implementation | `CHECKED_IN_SCOPE` for trace/native code; `INTEGRATED_NOT_IMPLEMENTED` | Ouro HF gate post hoc; Huginn batched adaptive path errors |
| Simulator | `CHECKED_IN_SCOPE_TRACE_SIM_ONLY` | 80/80 fair equal-work checks; no physical scheduler timing |
| Comparison | `COMPARABLE` only for within-trace grouping; `INCOMPLETE` for physical speed | same selected items/work per strategy; no integrated no-refill runtime |
| Hypothesis | `EXPLORATORY_SIGNAL` at q=0.2–0.5, `NEGATIVE_IN_SCOPE` at q=0.7/practical predictors | oracle headroom varies materially, history-only gains small |

The next research block should first pin dataset revisions and repair the Huginn B>1 adaptive kernel/parity path, then decide whether the compressed-KV quality gate is acceptable. A physical Ouro speed claim requires a survivor-skipping recurrent adapter and repeated synchronized measurements. Until then, the present result supports a logical batching opportunity and native capacity boundaries only.
