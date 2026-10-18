# Phase 05 WP2 review — physical no-refill regrouping headroom

**Decision (preregistered rule): WP3 = GO for both Ouro-1.4B and Huginn-3.5B.** The clairvoyant `oracle_b1` grouping, physically executed with its own live BF16 exits, cuts decode wall time against CDB-style no-refill FIFO by **20–35% (Ouro)** and **10–15% (Huginn)** at every B ∈ {4…64}, on both Alpaca and ShareGPT, and all 3 oracle repeats beat all 3 FIFO repeats in every one of the 20 cells. **The simple causal policies recover almost none of it** (best +2.6%, ≤17% of the oracle gap). So the headroom is real, but a WP3 predictor has to be much better than "previous depth" to capture it. WP3 has not been started.

## Scope and provenance

- Preregistration [`WP2_PREREGISTRATION.md`](../../WP2_PREREGISTRATION.md) and runner committed at `44abcb5` before any scored run. The analysis script `wp2_analyze.py` was committed at `7e8ecf4` *after* launch but before any result was read. Its rules are a verbatim implementation of the preregistered rules.
- Lanes: A Ouro×Alpaca GPU0, B Huginn×Alpaca GPU1, C Ouro×ShareGPT GPU2, D Huginn×ShareGPT GPU3. One process per H100, CPU pinned to 8 cores. 94 runs per lane: 1 warmup, 90 scored, 3 B=256. **376/376 OK, 0 OOM, 0 hard-check failures**.
- Contract identical to WP1: BF16, Ouro q=0.2/Rmax 4, Huginn 0.12/Rmax 16, eager phase02 survivor runtime, strict no-refill, lockstep decode boundary. N=256 fresh requests per dataset (excluding prompts earlier phases could have used), 64-token prompt cap, T=64, B1 tokens teacher-forced.
- Raw data per lane: `runs.csv`, `b1_reference.csv.gz`, `live_tokens.csv.gz` (per-run per-token live depth and next token), `status.jsonl`, `run_manifest.json`. Derived: `analysis/cell_summary.csv`, `analysis/decision.json` (with input SHA-256s), `analysis/headroom_vs_B.png`.

## Result: decode-wall saving vs FIFO (median of 3; * = all oracle repeats faster than all FIFO repeats)

| Lane | B | FIFO wall | random | static_first | history_last | history_mean | **oracle_b1** | oracle / FIFO live RIC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ouro×Alpaca | 4 | 294.4 s | −0.4% | −3.6% | −1.0% | +1.0% | **29.6%\*** | 9,975 / 14,004 |
| | 8 | 163.3 s | −0.6% | −0.1% | +0.2% | +1.6% | **34.0%\*** | 5,090 / 7,625 |
| | 16 | 87.8 s | 0.0% | −0.1% | +1.6% | +1.3% | **34.8%\*** | 2,633 / 4,001 |
| | 32 | 46.5 s | −0.7% | −1.3% | 0.0% | −0.1% | **31.2%\*** | 1,388 / 2,040 |
| | 64 | 25.6 s | +0.2% | −1.8% | +0.5% | +0.3% | **24.8%\*** | 745 / 1,023 |
| Ouro×ShareGPT | 4 | 294.3 s | 0.0% | +0.5% | +2.5% | +1.6% | **27.0%\*** | 10,411 / 14,093 |
| | 8 | 162.9 s | 0.0% | +0.9% | +2.6% | +1.1% | **31.3%\*** | 5,325 / 7,620 |
| | 16 | 87.6 s | −0.3% | +0.9% | +2.3% | +1.6% | **32.0%\*** | 2,755 / 4,006 |
| | 32 | 46.9 s | +0.7% | +1.7% | +1.5% | +1.4% | **29.2%\*** | 1,453 / 2,042 |
| | 64 | 25.4 s | −0.4% | −0.3% | −0.3% | −1.2% | **19.7%\*** | 795 / 1,024 |
| Huginn×Alpaca | 4 | 269.1 s | −0.1% | 0.0% | +0.8% | +0.9% | **10.2%\*** | 36,483 / 41,117 |
| | 8 | 144.4 s | +0.3% | −0.5% | +0.8% | +1.3% | **13.5%\*** | 18,534 / 21,839 |
| | 16 | 81.6 s | +0.4% | +0.5% | +2.3% | +1.8% | **15.0%\*** | 9,507 / 11,599 |
| | 32 | 48.1 s | 0.0% | +0.4% | +1.8% | +1.7% | **13.9%\*** | 4,964 / 6,137 |
| | 64 | 30.6 s | −0.2% | −0.2% | +1.6% | +1.2% | **10.6%\*** | 2,659 / 3,228 |
| Huginn×ShareGPT | 4 | 265.1 s | +0.4% | +0.3% | +1.4% | +1.3% | **10.1%\*** | 35,952 / 40,478 |
| | 8 | 141.8 s | 0.0% | −0.5% | +1.2% | +0.4% | **13.6%\*** | 18,246 / 21,536 |
| | 16 | 80.2 s | −0.8% | +0.1% | +2.4% | +1.7% | **15.0%\*** | 9,398 / 11,471 |
| | 32 | 47.4 s | −0.2% | −0.2% | +1.8% | +0.9% | **13.8%\*** | 4,942 / 6,113 |
| | 64 | 30.6 s | +0.5% | +0.8% | +2.0% | +1.2% | **11.7%\*** | 2,645 / 3,266 |

Repeat spread is small: for example, Ouro×Alpaca B8 FIFO ran 162.6–165.6 s and oracle 107.1–112.7 s. Huginn×ShareGPT B8 FIFO ran 141.6–141.9 s and oracle 122.2–123.2 s. Policy sort CPU time is ≤ 8 ms per run.

## Equal-work check

Wall savings come from **RIC (recurrent invocations)**, not from doing less work. The oracle's median live work differs from FIFO's by −36…+43 token-loop steps on totals of ~39–41k (Ouro) and ~141–144k (Huginn), i.e. |Δ| ≤ 0.1%. No speedup is flagged as unequal work. The frozen-B1 lockstep RIC (`cell_summary.csv:frozen_b1_lockstep_RIC`) is within ~1–4% of the live oracle RIC, so live BF16 depth perturbation erodes little of the frozen packing gain.

## Measured BF16 outcomes (B2: retained, not corrected)

Live vs B1 depth flips per run: Ouro ≈ 230–250 / 16,384 (≈1.5%); Huginn ≈ 710–790 / 16,384 (≈4.5%). Next-token flips: Ouro ≈ 205–230 (≈1.3%); Huginn ≈ 104–119 (≈0.7%). Flip counts are similar across FIFO and the oracle, and inputs are teacher-forced, so these flips do not compound along the trajectory.

## Why the causal policies fail (B1 traces, descriptive)

| Lane | Within-step Spearman ρ(depth t−1, depth t), median | ρ(history mean, depth t), t ≥ 8 | Between-request share of depth variance |
| --- | ---: | ---: | ---: |
| Ouro×Alpaca | 0.32 | 0.30 | 0.13 |
| Ouro×ShareGPT | 0.37 | 0.28 | 0.13 |
| Huginn×Alpaca | 0.35 | 0.44 | 0.21 |
| Huginn×ShareGPT | 0.36 | 0.40 | 0.17 |

80–87% of depth variance is **within a request, token by token**. A request's history therefore predicts the current token's depth only weakly, and sorting on a ρ≈0.3–0.4 signal moves wall time by ≈1–2%. A useful WP3 predictor must use **current-token, pre-recurrence information**, such as the Prelude or embedding of the token being processed, or the previous step's output distribution. History alone is not enough. (Huginn's depth distribution is concentrated at 7–10 of 16, which also limits its oracle gain compared with Ouro's spread over 1–4.)

## Important context: batch size dominates

The no-composition control B=N=256 (one cohort, no grouping choice) takes **10.2–10.6 s for Ouro and 19.0–19.1 s for Huginn**. That is faster than the best B=64 oracle (19.2–20.4 s and 26.9–27.4 s). The B=1 sequential control takes 873–946 s. This eager runtime is launch-bound: the regrouping headroom applies **only when B is constrained**, for example by memory, latency SLOs or online arrival. It is not a reason to choose a small B. The B constraint for the target deployment must be stated before WP3 claims anything end-to-end.

## Five-axis review

| Axis | Status |
| --- | --- |
| Measurement | VALID: 3 repeats per cell, interleaved order, CPU pinning, all runs retained, tight spread |
| Implementation | Per-run row/width hard checks 376/376 PASS; runtime = WP1-gated phase02 path |
| Simulator | Frozen RIC reported as structural only; every headline number is physically executed |
| Comparison | Same inputs, threshold, KV, precision, B; equal live work within 0.1% |
| Hypothesis | Oracle headroom large and robust → **WP3 GO (both models)**; history policies insufficient |

## Limitations

Oracle labels come from a B1 run on the same cohort, so they are clairvoyant and not achievable as-is. Lockstep scheduling only (no asynchronous ready pool). Eager runtime without CUDA graphs: absolute wall times and the B tradeoff may differ on an optimized engine such as upstream CDB. One threshold per model. Prompts truncated to 64 tokens. No EOS (fixed 64 tokens).

## Next decision required (not taken here)

Approve WP3 scope: a causal **current-token** depth-propensity predictor (features available before the recurrent core), selection on `wp3_validation`, a frozen model evaluated physically on the sealed `wp3_test`, and feature/predictor/sort overhead inside the timer. FIFO, `history_last` and `oracle_b1` serve as reference points.
