# Phase-4 Gate A implementation review — 2026-09-17

## Decision

**FAIL for both primary BF16 model lanes.** The independent per-request B=1 reference and physically compacted live-exit run disagree on the prospectively specified 32-request × 8-token, B=8 anchor. Per `phase04/DEPTH_PATTERN_PREDICTOR_PLAN.md` and `phase04/GATE_A_PARITY_RULE.md`, do not collect Phase-4 predictor data or claim a new FIFO/Oracle speed result. The separate FP32 control passes; it does not retroactively pass the BF16 experiment or establish the exact source of its drift.

All runs use the pinned checkpoints, the same selected model-specific KV/exit policy, 32 unique effective tokenized synthetic prompts (max 64 tokens), cohort-width reference padding, fixed 8-token horizon without EOS, and reference-generated input token IDs supplied to the compact run. The compact run still computes **its own live gate and next-token logits**; no reference depth is forced. This tests same-input semantics, not free-running text quality. Detailed commands, GPU UUIDs, source/config/tokenizer hashes, and versions are in each `policy_manifest.json`.

| Physical GPU | Role and largest test | Evidence | Outcome |
| --- | --- | --- | --- |
| 2 | Ouro-1.4B primary, N32/B8/T8 BF16 and FP32 | `../20260917T052824Z_2ff6ab1/gate_a_ouro_gpu2_n32_b8_t8_bf16/`; `../20260917T052700Z_2ff6ab1/gate_a_ouro_gpu2_n32_b8_t8_fp32/` | BF16 FAIL; FP32 PASS |
| 3 | Ouro replication, N4/B4/T2 BF16 | `../20260917T051209Z_98891b2/gate_a_ouro_gpu3_replication/` | strict numerical parity FAIL; no N32 claim |
| 5 | Huginn replication, N4/B4/T2 BF16 | `../20260917T051209Z_98891b2/gate_a_huginn_gpu5_replication/` | strict numerical parity FAIL; no N32 claim |
| 7 | Huginn-3.5B primary, N32/B8/T8 BF16 and FP32 | `../20260917T052824Z_2ff6ab1/gate_a_huginn_gpu7_n32_b8_t8_bf16/`; `../20260917T052700Z_2ff6ab1/gate_a_huginn_gpu7_n32_b8_t8_fp32/` | BF16 FAIL; FP32 PASS |

The primary BF16 anchor initially ran at source `c591671c497f47b23de52d005197685b5efcb6b8`; it was repeated unchanged at `2ff6ab18cc289f5161ed9af4aaf03d8059c6ed4b` after a profile-event synchronization repair and produced identical decision/work discrepancies. Source `2ff6ab1` also produced the FP32 control. No GPU outside 2, 3, 5, 7 was used for Phase 4; other jobs were left alone.

## Live-anchor evidence

| BF16 N32/B8/T8 | Ouro GPU2 (q=0.2, Rmax4) | Huginn GPU7 (threshold 0.12, Rmax16) |
| --- | ---: | ---: |
| Depth disagreements / 256 request-token positions | 2 | 7 |
| Next-token ID disagreements | 0 | 3 |
| B1 / compact recurrent calls | 617 / 615 | 2108 / 2105 |
| Max hidden relative L2 | 10.82% | 5.52% |
| Max logit relative L2 | 22.16% | 9.24% |
| Max attention-visible KV relative L2 | 31.51% | 2.78% |
| Max gate-score absolute difference | 0.01289 | 0.00718 |
| Structural KV mismatches | 0 | 0 |
| Request-keyed h0 hash mismatches | not applicable | 0 / 288 prefill-and-decode states |

The prospective release limit was exact depths/next-token IDs and ≤3% relative L2 for hidden/logits/prefill/KV, with gate-score absolute difference ≤0.02. Both BF16 lanes fail the exact-decision and numerical conditions, despite passing the observed gate-absolute and KV-structure conditions. These relative maxima include later states after trajectories diverged and should not be interpreted as the initial rounding error.

Ouro's first decision disagreement is request 22, token 6: the reference first-step cumulative gate is 0.199307978, compact is 0.200557709, straddling q=0.2, so depths are 2 versus 1. A second request also crosses the threshold. Huginn's earliest decision disagreement is a next-token ID at request 7, token 0 (32 versus 18370) although both exits occur at depth 10; subsequent near-threshold exit differences appear, including request 25/token 1 (depth 10 versus 9 at scores 0.120288983 versus 0.119749524). Full per-row score/margin, first-divergence and near-threshold receipts are in the raw JSON, not overwritten by a forced-depth replay.

At FP32, the same N32/B8/T8 diagnostic passes all recorded elementwise checks for both models: 0 depth and next-token disagreements, 616/616 Ouro and 2097/2097 Huginn recurrent calls, no structural KV or h0 mismatch. Those work totals need not equal BF16 totals because precision changes the live decisions. This is strong evidence that BF16 batch-shape numerical effects contribute to the failure, **not** proof that every layer-level cause has been isolated or that FP32 is an approved replacement primary runtime. Do not relax thresholds or substitute FP32 results for the BF16 release gate without an explicit protocol change.

## Audit finding disposition

| Finding | Tested disposition | Remaining limit |
| --- | --- | --- |
| F1 stochastic identity | Huginn request/position-keyed h0 implemented; all 288 anchor state hashes match B1/compact; pinned config has test-time noise 0. | Other checkpoints/noise settings untested. |
| F2 forced replay versus live | Independent row reference and actual compact live gate implemented and run; **live parity FAIL** on both BF16 anchors. | No schedule-invariant live result; prior forced-replay timing cannot establish it. |
| F3 timing instrumentation | Benchmark/profile paths split, KV events/receipts bounded or disabled as appropriate; newly exposed asynchronous profile-event read fixed in `2ff6ab1` and CPU-tested. | Clean FIFO/Oracle timing **BLOCKED** until semantics pass; `elapsed_wall_s` here is a diagnostic wall clock, not scored runtime. |
| F4 test-based selection | Selection moved to validation; CPU regression passed. | New sealed confirmation and predictor selection **BLOCKED**. Old Phase-3 test-informed selection remains exploratory. |
| F5 effective input/artifact audit | Effective tokenized prompt hashing and cross-split guard added and CPU-tested; 32/32 unique effective inputs in anchor. | No actual Phase-4 train/validation corpus or consumed predictor artifacts yet. |
| F6 config/shape/boundary | Fail-closed config/padding assertions, row/KV tests, N4 padding/equal-length/FP32 controls and N32 mixed-length run completed. | Full B=1/4/8/16/32/64, N64×T32, permutations and EOS matrix not completed after release failure. |
| F7 predictor limits | Not addressed experimentally. | Block B/C **BLOCKED** by Gate A. |
| F8 GPU ownership | User-authorized mapping recorded and enforced by exact physical index, UUID, H100 type, single visible device, and model role. All four assigned GPUs have receipts. | No reassignment implied by old Phase-3 GPU6/8 code. |

The profile-control's first FP32 Ouro attempt at source `382ed15` ended before comparison with `CUDA error: device not ready`; its `failure_receipt.json` is preserved under `../20260917T052532Z_382ed15/`. The event-drain repair is profile-only; the BF16 decision mismatch persisted in the repeat. CPU regression after the repair: 19 tests passed with `python -m unittest phase04_runtime.test_contracts phase02_runtime.test_runtime phase02_runtime.test_oracle_ric phase02_runtime.test_summarize_oracle_physical phase03_predictor.test_tournament phase03_predictor.test_full_tournament -q` in `.venv`.

## Consequence and bounded next check

Prior Phase-2 Oracle RIC and Phase-3 predictor traces remain evidence under their original frozen/replayed protocols; neither is invalidated as a historical measurement. Their physical prototype wall times cannot be promoted to verified live-adaptive Phase-4 speedups. No Phase-4 predictor winner, RIC recovery, FIFO/Oracle rebenchmark, throughput, decode wall or full-workload BCT was measured.

To resume this model work, first localize the *earliest* BF16 per-layer/attention/KV deviation on the saved near-threshold requests, compare controlled batch shapes and precision, and propose a protocol-consistent numerical/semantic remedy. Re-run the prospectively defined live Gate A before any expensive data collection. This review does not authorize changing the threshold, checkpoint, precision or workload solely to obtain a pass.
