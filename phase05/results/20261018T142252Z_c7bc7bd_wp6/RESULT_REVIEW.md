# Phase 05 WP6 review — extensions of the WP5 ranker

**Decisions (preregistered per-ranker rule: ≥ 5% wall saving vs FIFO with all repeats faster, in ≥ 2 of 4 B per block, all 4 blocks):**

| Study | Ranker | Ouro-1.4B | Huginn-3.5B |
| --- | --- | --- | --- |
| 6a cross-dataset transfer | `ranker_cross` (frozen WP5 R from the other dataset) | **GO** (retention 0.90) | NO-GO (retention 0.88 of a sub-5% gain) |
| 6a replication | `ranker_same` (frozen WP5 R) | NO-GO by rule: 3/4 blocks pass, one block hit by a GPU1 timing disturbance (see below) | NO-GO |
| 6b no trajectory | `ranker_notraj` | **GO** | NO-GO (medians mostly 4–5.5%, 1/4 blocks pass), but **trajectory-cost hypothesis supported** (faster than `ranker_same` in 16/16 cells) |
| 6c q=0.4 | `ranker_q04` (new) / `ranker_q02` (WP5 R transferred) | **GO / GO** | not run |
| 6d EOS variable length | `ranker_same` | **GO** | not run |

The Ouro result survives a fresh test split, transfer across datasets, removal of trajectory features, a different exit threshold and EOS-terminated variable-length decoding. Huginn stays just under the 5% bar. Dropping the trajectory features costs no ranking quality and moves Huginn's best cells to 5.2–7.2%.

## Provenance

- Preregistration and code: `313291f`; drivers: `c7bc7bd`, `1f33f82`, `c8fd0e0`. All committed before collection and training.
- **Deviation from WP5 practice:** the new rankers (6b ×4, 6c ×2) were trained and selected on validation only, inside the per-GPU job chains. Their checkpoints were therefore committed together with the results (`808c3d8`), not in a separate commit before the physical runs. Each run manifest records the checkpoint SHA-256, and training never read test arrays.
- Fresh splits:
  - `wp6_test` [3328,3840): 6a/6b, both models;
  - `wp6_q_test` [3840,4352): 6c, Ouro, collected at q=0.4 together with q=0.4 train/validation labels;
  - `wp6_eos_test` [4352,4864): 6d, Ouro, EOS stop. 186/512 (Alpaca) and 166/512 (ShareGPT) requests ended by EOS; mean length 45.2 and 50.3 of 64.
- **Gate 0: PASS 8/8.** No-trajectory and q=0.4 rankers, plus the variable-length path with random imposed lengths; live-vs-offline score Spearman 0.9985–0.9996.
- Physical: **1,016/1,016 runs OK**, 0 OOM, 0 hard-check failures.
  - 6a/6b: 4 lanes × 145 runs.
  - 6c: 2 × 121.
  - 6d: 2 × 97.
- Upstream optimized engine: scoped out before the runs (single cohort per wave; different checkpoint; torch 2.11/FA3). See the preregistration.

## 6a/6b — fresh `wp6_test`: decode-wall saving vs FIFO (median of 3; * = wins the cell)

| Lane / cohort | B | history_last | ranker_same | ranker_cross | ranker_notraj | oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ouro×Alpaca c0 (GPU1, disturbed) | 8 / 16 / 32 | 3.2 / 1.1 / 0.7 | 13.0 / 10.3 / 8.5 | 12.0 / 9.0\* / 6.1\* | 13.8\* / 10.9\* / 7.8 | 34.2 / 34.1 / 31.5 |
| Ouro×Alpaca c1 | 8 / 16 / 32 | 2.8 / 1.4 / 1.4 | 13.5\* / 11.3\* / 8.6\* | 12.1\* / 10.6\* / 8.2\* | 14.1\* / 12.8\* / 8.9\* | 34.8 / 34.9 / 31.3 |
| Ouro×ShareGPT c0 | 8 / 16 / 32 | 2.3 / 1.8 / 0.7 | 12.1\* / 11.3\* / 8.8\* | 11.1\* / 9.1\* / 6.7\* | 12.6\* / 11.3\* / 8.3\* | 32.2 / 33.1 / 29.8 |
| Ouro×ShareGPT c1 | 8 / 16 / 32 | 2.6 / 0.9 / 0.1 | 10.6\* / 8.7\* / 5.8\* | 9.8\* / 7.6\* / 5.4\* | 10.8\* / 9.0\* / 5.9\* | 31.8 / 32.4 / 29.2 |
| Huginn×Alpaca c0 (GPU1, disturbed) | 8 / 16 / 32 / 64 | 5.3 / 2.5 / 2.1 / 2.4 | 6.6\* / 4.3 / 4.7 / 4.4 | 6.1 / 3.4 / 3.6 / 3.9 | 7.2\* / 5.4 / 5.4 / 5.2\* | 16.8 / 14.9 / 13.9 / 12.1 |
| Huginn×Alpaca c1 | 8 / 16 / 32 / 64 | 1.5 / 1.3 / 1.6 / 1.5 | 3.0 / 4.6 / 4.2 / −2.7 | 2.2 / 3.4 / 3.6 / 2.7 | 4.4 / 5.4 / 5.5 / 4.2 | 13.8 / 15.3 / 14.7 / 11.5 |
| Huginn×ShareGPT c0 | 8 / 16 / 32 / 64 | 1.6 / 1.8 / 2.3 / 1.9 | 1.9 / 3.5 / 3.9 / 2.7 | 0.8 / 3.5 / 1.7 / 1.8 | 3.4 / 4.3 / 5.2\* / 3.1 | 13.3 / 14.1 / 14.2 / 10.5 |
| Huginn×ShareGPT c1 | 8 / 16 / 32 / 64 | 0.9 / 2.0 / 1.3 / 1.2 | 1.7 / 2.5 / 2.7 / 2.2 | 2.9 / 2.5 / 3.4 / 2.3 | 4.6 / 4.0 / 4.9 / 3.1 | 13.6 / 15.2 / 14.1 / 11.0 |

Ouro B=64 is 2–4% for every ranker and is not shown. Huginn×Alpaca c1 B=64 `ranker_same` (−2.7%) is a median over one disturbed repeat.

- **Transfer:** a ranker trained on the other dataset keeps ~90% of the same-dataset gain (Ouro 0.90, Huginn 0.88).
- **No trajectory:** validation recovery is essentially unchanged (Ouro 0.301/0.269 vs 0.301/0.273 with trajectory; Huginn 0.362/0.325 vs 0.363/0.327). The previous-token convergence/gate trajectory adds no ranking signal beyond the full-dimensional states, only serving cost. Removing it is faster than `ranker_same` in 16/16 Huginn cells and 12/16 Ouro cells. Huginn rises by ≈ +1–2 pp to 4–5.5%, still below the rule in most cells.

### Measurement disturbance on GPU1

Both lanes that ran on GPU1 have one slow repeat in some cells, slowing *every* policy by ~10–15%:
- Ouro×Alpaca c0 FIFO B8: 164.8 / 166.5 / **192.6 s**.
- Huginn×Alpaca c0 FIFO B8: 146.2 / 150.5 / **163.6 s**.

This breaks the "all repeats faster than all FIFO repeats" condition. Ouro×Alpaca c0 `ranker_same` fails every cell despite medians of 8.5–13.0%, and this single block decides the 6a replication NO-GO. It also inflates Huginn×Alpaca c0 medians (history_last 5.3% at B8). The same lane-wide pattern appeared on GPU0 in WP4. The cause is unknown; other users' jobs on GPUs 4–7 are a candidate. All runs are retained and the rule is applied as written. Clean lanes (GPU0, GPU2, GPU3) show sub-1% spreads.

## 6c — Ouro q=0.4, fresh `wp6_q_test`

| Lane / cohort | B=8 | B=16 | B=32 | B=64 | oracle B8–32 |
| --- | --- | --- | --- | --- | --- |
| Alpaca c0 q02 / **q04** | 7.6\* / **8.4\*** | 8.1\* / **8.2\*** | 7.2\* / **7.3\*** | 4.6 / **5.5\*** | 21.6–24.0 |
| Alpaca c1 | 8.1\* / **9.5\*** | 8.4\* / **10.1\*** | 8.2\* / **8.4\*** | 3.9 / 5.0 | 21.5–24.9 |
| ShareGPT c0 | 7.2\* / **8.3\*** | 7.4\* / **9.6\*** | 6.0\* / **7.2\*** | 2.9 / 4.2 | 20.9–23.5 |
| ShareGPT c1 | 6.6\* / **7.4\*** | 7.7\* / **8.4\*** | 5.5\* / **6.1\*** | 3.3 / 2.6 | 21.0–23.0 |

At q=0.4 depths are deeper and less spread, so the oracle headroom drops from 32–35% to 21–25%. A ranker trained at q=0.4 (validation recovery 0.39 / 0.33) captures 7–10%. The q=0.2 ranker transferred without refit still captures 6–8% and also passes. The ranking signal generalizes across exit thresholds.

## 6d — Ouro EOS-terminated variable length, fresh `wp6_eos_test`

| Lane / cohort | B=8 | B=16 | B=32 | B=64 | oracle B8 |
| --- | --- | --- | --- | --- | --- |
| Alpaca c0 / c1 | 10.7\* / 11.4\* | 9.9\* / 10.2\* | 5.4\* / 5.6\* | 0.9 / 3.3 | 32.1 / 32.0 |
| ShareGPT c0 / c1 | 8.2\* / 8.7\* | 8.9\* / 7.8\* | 4.6 / 5.2\* | 4.4 / 1.1 | 28.8 / 28.9 |

With about a third of requests leaving early and the last cohort per step partial, the frozen WP5 ranker still saves 8–11% at B=8/16. FIFO wall time drops by about a quarter because fewer tokens are decoded. Active-token work differs from FIFO by < 0.1%.

## Five-axis review

| Axis | Status |
| --- | --- |
| Measurement | VALID on GPU0/2/3; **GPU1 lanes disturbed** (one slow repeat per cell group), retained and flagged |
| Implementation | Gate-0 8/8; hard checks 1,016/1,016; EOS path uses masked placeholder KV positions (`kv.py:retire_rows`) |
| Simulator | None; all decisions physical |
| Comparison | Same inputs, B, serving path; equal live work (< 0.1%) |
| Hypothesis | Ouro gain **robust** (transfer, no-traj, q=0.4, EOS all GO); Huginn remains below the bar; trajectory features are pure cost |

## Limits

- Eager runtime, lockstep scheduling, 64-token prompt cap, T ≤ 64.
- One additional threshold (Ouro only); EOS study on Ouro only.
- The upstream optimized engine was not tested. Its single-cohort no-refill scheduler would first need a multi-cohort redesign.
- B-constrained regime only.
