# Phase 05 WP4 review — host-sync-free GPU serving of the frozen WP3 predictors

**Decision (preregistered):**
- **Serving-cost hypothesis: supported.** GPU serving beats the WP3 host serving in 27/32 cells (84%). The mean RIC-to-wall translation rises from −0.11 (host) to +0.17 (GPU); excluding the noisy lane it rises from 0.09 to 0.48.
- **Practical predictor: still NO-GO for both models.** Huginn wins no cell (best +3.4%). Ouro×ShareGPT meets the rule in both cohorts (B=8 and B=16, 5.3–6.0%). Ouro×Alpaca fails it on a lane whose timing was visibly disturbed (see Measurement).
- **Bottom line:** removing host syncs fixes most of the serving loss, but ranking quality is now the binding limit.

## Provenance

- Preregistration, GPU serving code and analysis `e18e02a`; lane driver `d4714b5`. All committed before any scored run.
- Predictors: the WP3-frozen checkpoints (C7 / C8 / C9 / C8), SHA-256 verified at start. No retraining.
- Fresh split `wp4_confirm` (pool indices 2304–2815), never used before: 2 cohorts × 256 per lane. Large feature arrays stay local; labels, tokens and manifests are committed.
- **Gate 0 equivalence audit: PASS 4/4.** Within-step score Spearman ≥ 0.9999993, identical cohorts 100%, max |feature diff| ≤ 9.5e-7, max |score diff| ≤ 1.9e-6, at B=8 over 64 steps. The GPU path makes exactly the same grouping decisions as the WP3 host path.
- Physical: 4 lanes × 121 runs = **484/484 OK**, 0 OOM, 0 hard-check failures. Policies: fifo, history_last, predictor_gpu, predictor_host (WP3 implementation), oracle_b1. B ∈ {8,16,32,64}, 3 repeats. The per-step host sync was removed from every policy in this runner.

## Results: decode-wall saving vs FIFO (median of 3; * = predictor_gpu wins the cell)

| Lane / cohort | B=8 | B=16 | B=32 | B=64 |
| --- | --- | --- | --- | --- |
| **Huginn×Alpaca c0** host / **GPU** / oracle | 0.5 / **2.1** / 13.9% | 0.6 / **2.1** / 15.6% | −0.2 / **2.8** / 14.2% | −1.4 / **2.7** / 10.9% |
| Huginn×Alpaca c1 | 0.2 / **1.4** / 13.9 | 0.3 / **2.1** / 15.5 | 0.5 / **3.4** / 14.8 | −1.9 / **2.5** / 10.8 |
| Huginn×ShareGPT c0 | 0.0 / **1.6** / 13.6 | −0.3 / **1.8** / 14.9 | −0.4 / **2.3** / 14.6 | −2.7 / **1.3** / 11.2 |
| Huginn×ShareGPT c1 | −0.4 / **1.2** / 13.3 | −0.2 / **1.6** / 14.7 | −1.4 / **1.3** / 13.5 | −2.2 / **2.0** / 10.4 |
| **Ouro×ShareGPT c0** | 6.1 / **6.0\*** / 31.8 | 5.1 / **5.3\*** / 33.1 | 1.9 / **2.7** / 29.7 | −2.4 / **0.5** / 21.6 |
| Ouro×ShareGPT c1 | 5.8 / **5.8\*** / 31.4 | 4.3 / **5.3\*** / 32.3 | 0.7 / **2.5** / 27.5 | −3.0 / **0.9** / 21.5 |
| Ouro×Alpaca c0 (disturbed) | 18.9 / **18.1\*** / 33.9 | 6.2 / **7.2** / 34.6 | 2.2 / **−3.1** / 30.4 | −27.7 / **−21.9** / 6.6 |
| Ouro×Alpaca c1 (disturbed) | 9.3 / **−11.6** / 34.6 | 5.2 / **6.8\*** / 34.7 | 2.4 / **4.0** / 19.0 | 0.7 / **3.3** / 23.8 |

`history_last` saves 0.9–2.9% on clean lanes. Live work differs from FIFO by < 0.1% in every cell.

- **Huginn:** removing host syncs turns the WP3 serving loss into a consistent gain, **+2.6 pp vs predictor_host on average**. The predictor now edges out `history_last` (≈1.6–3.4% vs ≈0.9–2.3%). However, its RIC saving is only 4–7% (oracle: 15–20%), so even near-perfect translation cannot reach 5%.
- **Ouro×ShareGPT:** GPU serving is about equal to host at B=8–16 and better at B=32–64 (+0.8–3.9 pp). The B=8/16 wins replicate WP3's sealed-test result on a fresh split.

## Measurement: Ouro×Alpaca lane (GPU0) disturbed

Throughout repeats 0–1 of this lane, runs of *every* policy ran 1.05–1.34× slower than the clean Ouro×ShareGPT lane's median for the same cell. Repeat 2 was clean. Examples: FIFO B8 167–218 s; oracle B64 only 6.6% versus 21–24% elsewhere. The other three lanes, with the same code and model, show 0.1–1.7% spread. The source is unknown; no foreign process was on GPU0 when checked afterwards. The other users' jobs on GPUs 4–7 restarted (new PIDs) during the run, which is a plausible host-side cause but unproven. **All runs are retained and the preregistered rule was applied as written.** Ouro×Alpaca's medians in this WP are therefore unreliable in both directions (+18.1% and −11.6% at B=8), and Ouro's NO-GO here rests on that lane. A clean rerun of this one lane, labeled as such, would be a new decision. It cannot be applied retroactively to change this verdict.

## RIC→wall translation (wall saving ÷ RIC saving vs FIFO, cells with RIC saving ≥ 2%)

| Scope | predictor_host | predictor_gpu | oracle_b1 |
| --- | ---: | ---: | ---: |
| All lanes | −0.11 | 0.17 | 0.84 |
| Excluding Ouro×Alpaca | 0.09 | **0.48** | 0.84 |

GPU serving recovers most of the predictor-specific loss. It still converts about half of its RIC saving, against the oracle's 0.84. The residual is plausibly the per-recurrence trajectory hook kernels, the per-cohort uncertainty reductions and the head launch. These costs are fixed per step, so their share is largest when the RIC saving is small (4–7%). This explanation was not separately timed.

## Five-axis review

| Axis | Status |
| --- | --- |
| Measurement | VALID for 3 lanes; **Ouro×Alpaca disturbed** (unexplained slowdowns in repeats 0–1), retained and flagged |
| Implementation | Gate 0 equivalence PASS 4/4; row/width hard checks 484/484 |
| Simulator | None; all numbers physical |
| Comparison | Same frozen predictor and cohort; host vs GPU serving paired in session |
| Hypothesis | Serving cost explained most of WP3's loss (**supported**); practical predictor still **NO-GO** |

## What this means for Phase 05

1. With serving overhead mostly removed, the remaining gap to the oracle comes from ranking quality: the predictors recover only 17–30% of the oracle's RIC saving. For Huginn this caps the gain at ≈ 2–3% wall time on this runtime.
2. Ouro×ShareGPT at B ∈ {8,16} is a replicated, rule-passing ≈ 5–6% gain. Ouro is the only model where a better ranker plus GPU serving looks promising.
3. Next steps that would still be valid need new explicit decisions and fresh splits: a stronger ranker (better current-token features), cheaper trajectory features, or a clean rerun of the disturbed lane as a separately labeled replication.
