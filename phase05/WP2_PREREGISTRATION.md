# Phase 05 WP2 — physical no-refill regrouping headroom (preregistration)

Committed **before** any scored WP2 run. Authorization: user message 2026-10-17 ("WP2 계획을 사용해서, 진행해줘") after the WP1 review (`phase05/results/20261017T052501Z_b82d7ed_wp1/RESULT_REVIEW.md`, B2 semantic Gate A PASS 4/4). WP3 (predictor) stays blocked until a decision on this review.

## Question

Under strict no-refill on the real survivor-compaction runtime, how much decode wall time does regrouping at the decode boundary save versus FIFO, when each grouping executes its **own** live BF16 exits? Is the clairvoyant headroom large enough to justify a predictor, and how much do simple causal history policies already recover?

## Fixed contract (identical across policies)

- Models/thresholds/Rmax/KV/h0 exactly as confirmed in WP1: Ouro-1.4B q=0.2, Rmax 4, last-exited depth-indexed KV; Huginn-3.5B threshold 0.12, Rmax 16, shared physical-layer KV, request-keyed h0 seed 1. BF16, eager phase02 runtime, survivor-only recurrent execution.
- Data: Alpaca (`tatsu-lab/alpaca`) and ShareGPT (`ewof/sharegpt-instruct-unfiltered-deduped`), cached revisions. Request split (`phase05/wp2_headroom.py:split_prompts`): drop the first 4,096 prompts of the seed-1 shuffle that earlier phases drew from, deduplicate, shuffle with seed 20261017; `[0,256)` = WP2 headroom (used now); `[256,1280)` WP3 train, `[1280,1792)` WP3 validation, `[1792,2304)` WP3 test — **untouched** by WP2.
- N=256, prompt cap 64 tokens (truncation), T=64 decode tokens, no EOS.
- A per-request B=1 live greedy run defines the canonical B1 depths and token IDs; every batched run teacher-forces those token IDs. Next-token and depth differences are measured outcomes (B2).
- Lockstep decode boundary: at each token step all N requests are ready (runtime KV commits per step); the policy partitions them into N/B cohorts; each cohort runs Prelude → shrink-only core → Coda. No refill, no inter-cohort merge, no intra-interval regrouping.

## Policies (ascending sort, ties by request id)

| Policy | Key at step t | Information |
| --- | --- | --- |
| `fifo` | request id | none (CDB no-refill FIFO baseline) |
| `random` | seeded shuffle per (rep, t) | none |
| `static_first` | this run's own live depth at t=0 | causal |
| `history_last` | own live depth at t−1 | causal |
| `history_mean` | mean own live depth over 0..t−1 | causal |
| `oracle_b1` | canonical B1 depth at t | **clairvoyant**, but executed live with its own exits |

History policies use FIFO at t=0. `oracle_b1` is a frozen-label-informed grouping executed physically — not a guaranteed optimum and not an equal-work comparison.

## Grid and repeats

B ∈ {4, 8, 16, 32, 64} × 6 policies × 3 repeats, plus B=N=256 no-composition control (3 repeats) and the B=1 per-request reference (1 run, timing labeled "B=1 control"). One discarded warmup (B=64 fifo). Order: per repeat, B order shuffled (seed `wp2-plan:<rep>`), policy order rotated by repeat. All runs, OOMs and errors retained.

Lanes (independent processes, one GPU each, CPU pinned to 8 distinct cores): A Ouro×Alpaca GPU0 (cores 0–7), B Huginn×Alpaca GPU1 (8–15), C Ouro×ShareGPT GPU2 (16–23), D Huginn×ShareGPT GPU3 (24–31).

## Timing

`decode_wall_s`: `perf_counter` between `cuda.synchronize()` after prefill and after the last token, including policy/sort CPU time (also reported separately as `scheduler_cpu_s`), Prelude, core, gate, compaction, KV and Coda. One host sync per decode step for every policy (needed by history policies). Prefill excluded. `benchmark` instrumentation (no per-stage syncs).

## Hard checks per run (lane stops on failure)

Each (t, request) executed exactly once; per cohort width[0]=B, shrink-only widths, `len(widths)=max depth`, width at r = #members with depth ≥ r. Provenance must match the WP1 contract and committed source.

## Primary metrics and decision rules (fixed now)

Per lane and B: median over 3 repeats of decode wall; **headroom(policy) = 1 − median(policy)/median(fifo)**. Also live RIC, live work, depth/next-token flips vs B1, core row slots, scheduler CPU time, peak memory.

- **Oracle headroom present** for a lane at B if headroom(`oracle_b1`) ≥ 5% **and** all 3 oracle repeats are faster than all 3 FIFO repeats.
- **WP3 GO (per model)** if oracle headroom is present at ≥ 2 of B ∈ {8, 16, 32, 64} in **both** datasets. Otherwise **NO-GO** for that model.
- **History recovery** = headroom(best causal policy chosen by mean over B on that lane) / headroom(`oracle_b1`), reported per cell; if ≥ 50% in most GO cells, the review notes that a learned predictor's marginal value is bounded by the remaining gap.
- Live-work change of each policy vs FIFO is reported next to every speedup; a speedup accompanied by lower live work is flagged as **not equal-work**.

## Engineering smoke (declared)

An unscored N=32/T=4 smoke (GPU0 Ouro×ShareGPT, GPU1 Huginn×Alpaca, `--smoke`) was run before this commit only to verify the runner; its output stays in the session scratchpad and is not evidence.

## Outputs

`phase05/results/<UTC>_<source_sha>_wp2/lane_<X>_<model>_<dataset>_gpu<i>/`: `run_manifest.json`, `status.jsonl`, `b1_reference.csv.gz`, `runs.csv`, `live_tokens.csv.gz`; then `analysis/` from `phase05/wp2_analyze.py`, `RESULT_REVIEW.md`, `review_manifest.json`.
