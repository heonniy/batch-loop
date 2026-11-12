# Decision characterization — Stage A (decomposition) and Stage B (D-conditioned crossover)

**Status:** CPU-only re-analysis, run 2026-11-13. No GPU or model process was run.
**Input:** the committed receipts `dynamic_sd/followup_20261112/results_20261112/cf/*.jsonl.gz` at `5c6bbea3`, read directly from `.gz` and unchanged. Their sha256 values are checked against `followup_20261112/results_20261112/MANIFEST.sha256` and recorded in `results_20261113/MANIFEST.json`.
**Outputs:** `results_20261113/{decomposition,crossover}.{md,json}`.
**Code:** `cf_receipts.py`, `decision_decompose.py`, `conditioned_crossover.py`, `tests/test_cpu.py` (13 tests).

Labels: *SOURCE MEASURED* means a value physically measured at 5c6bbea3 (B=1, eager, one round, FULL KV). *DERIVED FROM RECEIPTS* means computed here from those receipts. *HYPOTHESIS* marks interpretation. Every number below is DERIVED FROM RECEIPTS unless labelled otherwise.

## Audit

| cell | prefixes kept / total | prompts | excluded (any inexact trial) |
|---|---:|---:|---|
| Ouro fixed / τ0.4 / τ0.1 (FP32) | 512 / 512 each | 32 | 0 |
| Ouro τ0.4 BF16 | 491 / 512 | 32 | 21 prefixes (183 inexact trials); excluded prefixes have higher mean reference D (2.81 vs 2.57) and lower mean k (109 vs 125), so the BF16 cell is biased toward easier and later prefixes |
| Huginn fixed / ε0.1 (FP32) | 384 / 384 each | 24 | 0 |

The validator checks every prefix for:
- identical prefix, target and runtime contract;
- the full action grid with 3 repeats;
- `measured=true`, `wall_ms>0` and `committed_tokens>=1`;
- identical committed tokens and recorded geometry across repeats.

There were no validation failures.

## Method notes (deviations, stated up front)

1. **Selection rule.** The plan requires matched rates as Σtokens/Σms. With that objective, choosing each prefix's action by its own tokens/ms ratio (the rule `oracle_frontier.py` used at 5c6bbea3) is *not* optimal, and a superset class can then score **below** its subset. For example, Ouro τ0.4 gives Joint/K = 0.976 under that rule.
   - The primary results therefore choose, within each class, the per-prefix actions that maximise the class's aggregate rate. This uses Dinkelbach iteration on the two training repeats, then scores on the held-out repeat.
   - The old rule is kept as a sensitivity table in `decomposition.md`.
   - Consequence: the 5c6bbea3 "oracle / best static" numbers (1.18–1.26) slightly understate the aggregate-optimal one-round frontier (1.21–1.34 below).
2. **Calibration.** S, d0 and K0 come from the other prompt fold (2-fold, prompt-disjoint, so each prompt is evaluated once).
   - The per-prefix choice inside a class is a **same-prefix descriptive oracle**: it is picked on 2 timing repeats and scored on the third. It is not a deployable policy and not a measure of generalization to new prompts.
   - The oracle knows the realised greedy acceptance of each action at that prefix.
3. **Units.** All ratios are one-round aggregate rates. Actions with different K advance different numbers of tokens, so none of these ratios is a multi-round speedup. Controller cost is excluded. Timing is eager B=1, and FP32 runs on the SDPA fallback attention kernel.

## Stage A — nested decomposition

Classes: S = calibrated best fixed action; A = {AR, best fixed SD}; K = {AR, DtV(d0, any K)}; d = {AR, DtV(any d, K0)}; Joint = everything. The nesting is S ⊂ A ⊂ K ⊂ Joint and A ⊂ d ⊂ Joint. Values are aggregate-optimal, with 95% prompt-cluster CIs.

| cell | S vs AR | A/S (AR fallback) | K/A (adaptive K) | d/A (adaptive d) | Joint/K (d beyond K) | Joint/d (K beyond d) | Joint/S |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ouro fixed | 1.850 | 1.194 [1.162, 1.224] | 1.017 [1.012, 1.022] | 1.000 [1.000, 1.000] | 0.999 [0.999, 1.000] | 1.016 [1.011, 1.022] | 1.213 [1.178, 1.247] |
| Ouro τ0.4 | 1.340 | 1.209 [1.167, 1.249] | 1.056 [1.037, 1.078] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.056 [1.037, 1.078] | 1.277 [1.237, 1.315] |
| Ouro τ0.1 | 1.038 | 1.179 [1.151, 1.207] | 1.134 [1.114, 1.154] | 1.003 [1.000, 1.007] | 1.000 [0.998, 1.003] | 1.130 [1.111, 1.148] | 1.336 [1.294, 1.376] |
| Ouro τ0.4 BF16 | 1.443 | 1.191 [1.150, 1.237] | 1.065 [1.047, 1.087] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.065 [1.047, 1.087] | 1.268 [1.225, 1.321] |
| Huginn fixed | 1.860 | 1.097 [1.073, 1.130] | 1.005 [1.002, 1.008] | 1.138 [1.128, 1.150] | 1.149 [1.138, 1.160] | 1.015 [1.011, 1.018] | 1.267 [1.237, 1.304] |
| Huginn ε0.1 | 1.470 | 1.260 [1.227, 1.294] | 1.008 [1.005, 1.012] | 1.015 [1.010, 1.020] | 1.019 [1.015, 1.023] | 1.012 [1.008, 1.017] | 1.295 [1.259, 1.333] |

Calibrated (d0, K0): Ouro fixed (1,8); Ouro τ0.4 (1,4) or (1,8) depending on the fold; Ouro τ0.1 (1,2); Huginn fixed (8,8); Huginn ε0.1 (4,8).

**Cross-regime, live − fixed.** Paired over the same prompt ids. Live and fixed have different native targets, so this comparison is descriptive, not causal.

| pair | Joint/S | K/A | d/A | Joint/K |
|---|---:|---:|---:|---:|
| Ouro τ0.4 − fixed | +0.064 [+0.025, +0.106] | +0.039 [+0.020, +0.062] | +0.000 [0, 0] | +0.001 [+0.000, +0.001] |
| Ouro τ0.1 − fixed | +0.123 [+0.071, +0.172] | +0.117 [+0.097, +0.138] | +0.003 [−0.000, +0.007] | +0.000 [−0.002, +0.003] |
| Huginn ε0.1 − fixed | +0.028 [−0.024, +0.074] | +0.003 [−0.001, +0.008] | **−0.123** [−0.134, −0.112] | **−0.130** [−0.141, −0.119] |

**Stage A reading:**
- **AR fallback is the largest component in every cell.** A/S is 1.10–1.26. The oracle picks AR in 25–49% of rounds; HYPOTHESIS: these are rounds where it knows the draft would be rejected. This is acceptance foresight, and it exists at fixed depth too.
- **Adaptive K matters for Ouro, and more under exits.** K/A is 1.017 at fixed depth, 1.056 at τ0.4 and 1.134 at τ0.1. The increase over fixed depth is significant. For Huginn adaptive K is small (≤1%).
- **Adaptive d, Ouro: NO-GO.** Joint/K = 1.000 and d/A = 1.000 in every regime and precision. Given AR fallback and adaptive K, d=2 is never needed.
- **Adaptive d, Huginn: not live-specific.**
  - At fixed depth, adaptive d is the dominant component (Joint/K = 1.149). This is a per-prefix d4 vs d8 choice driven by d4's realised acceptance (Stage B).
  - Under ε0.1 exits it shrinks to 1.019 (CI > 1, so still above the plan's 1% screen) because calibrated d0 moves to 4.
  - Exits *reduce* the value of adaptive d by 0.12–0.13.
- **Screen (plan rule):**
  - Ouro: "adaptive d low priority" in all 4 cells; K/A is material.
  - Huginn: Joint/K is above 1% in both cells, so d is not screened out. But the d-specific gain is a fixed-depth phenomenon, not a dynamic-depth one.
- **Never a depth-specific claim:** Joint/S > 1 everywhere, including at fixed depth.

## Stage B — reference-D-conditioned crossover

POST-HOC. D is the native AR exit depth of the token processed at the checkpoint, which is a post-verification quantity. Ratios are aggregate U(hi)/U(lo), paired on identical KV snapshots, with prompt-cluster bootstrap CIs. Full tables, including per-bin n, latency differences, acceptance and derived core rows, are in `results_20261113/crossover.md`.

**Draft depth d (fixed K):**

| cell | comparison | per-D-bin U(d_hi)/U(d_lo) | label |
|---|---|---|---|
| Ouro τ0.4 | d1→d2, K4 | D2 0.726 [0.711, 0.744] · D3 0.792 [0.758, 0.827] · D4 0.873 [0.756, 1.026] | no crossover; d1 better or tied in every bin |
| Ouro τ0.1 | d1→d2, K4 | D1 0.979 · D2 0.986 · D3 0.922 · D4 1.005 [0.847, 1.197] | noisy (point estimates cross 1, CIs do not separate) |
| Huginn ε0.1 | d4→d8, K8 | ≤14 0.873 · 15 0.888 · 16 0.903 · 17 0.858 · 18–20 0.940 [0.851, 1.058] · 21–31 1.009 (n=9, sparse) | no crossover; d4 better in all adequately sized bins |
| Huginn fixed | d4→d8, K8 | D=32 only: 1.028 [0.994, 1.065] | single bin |

- Larger d does **not** reduce core rows (rows lo→hi are equal, e.g. 146→146 for Huginn ε0.1 D16). Verification always runs every source to its own exit depth D. Larger d only moves rows from the batched verify into sequential single-row draft calls. Its cost is latency (+34 ms per round for Ouro d1→d2 at K4; +68 ms for Huginn d4→d8 at K8). Its benefit is +0.1–1.4 accepted drafts.

**Confounding control: the d choice follows acceptance, not D.** These are U(d_hi)/U(d_lo) values stratified by the accepted length of the shallower draft (an outcome, so post-hoc):

| cell | accepted(lo)=0 | 1 | 2 | 3 | … | all accepted |
|---|---:|---:|---:|---:|---|---:|
| Ouro τ0.4, d1→d2 K4 | 2.07 | 1.28 | 0.97 | 0.85 | | 0.70 (n=380 of 512) |
| Huginn ε0.1, d4→d8 K8 | 4.65 | 2.19 | 1.77 | 1.17 | 0.85–1.05 | 0.74 (n=231 of 384) |
| Huginn fixed, d4→d8 K8 | 4.64 | 2.84 | 2.08 | 1.28 | 0.89–1.17 | 0.79 (n=187 of 384) |

Across D bins the d ratio moves within 0.73–0.94. Across acceptance strata it moves within 0.70–4.65. **The plan's falsifier holds:** the d optimum depends almost entirely on the shallower draft's acceptance and not on D, and deep-D bins do not favour a larger d once cost is included. This is a valid negative result for "deep target exit ⇒ deeper draft". For Huginn ε0.1 the d ratio also drifts with generation position k (1.13 for k<84 → 0.79 for k≥164), which is another non-D confounder.

**Speculate or not (AR vs SD), by D: reproducible crossover in Ouro τ0.1.** U(DtV d1 K2)/U(AR) is 0.80 [0.77, 0.83] at D=1, 1.09 at D=2, 1.28 at D=3 and 1.16 at D=4. At τ0.4 SD wins in every bin, with its peak at D=3. For Huginn ε0.1, SD/AR (d4 K8) rises with D: 1.30 (≤14) → 1.49 (16) → 1.62 (17).

**Length K, by D:**
- Ouro τ0.4, K4→K8: 1.022 [1.004, 1.040] at D=2 and 0.881 [0.796, 0.961] at D=4. This is a reproducible crossover.
- Ouro τ0.1, K2→K4: 0.946 at D=1 and 1.045 at D=3.
- HYPOTHESIS: these follow the depth-conditional acceptance already reported in F-080 (deep D → lower acceptance), so they do not by themselves show that D carries information beyond acceptance.

**Missing fields (reported rather than reconstructed):** per-stage Core/Prelude/Coda wall, CUDA-event stage timings, CPU stall and host sync, graph bucket. Core rows, core calls and rejected rows are DERIVED exactly from the recorded source depths. The forced-verification-budget / matched-replay intervention arm needs the GPU and was not run.

## Conclusions (research judgment, not a Decision)

1. **Adaptive draft recurrence d is NO-GO for Ouro.** Its incremental value over AR fallback plus adaptive K is zero in all regimes.
2. **For Huginn, adaptive d is not dynamic-depth-specific.** Its oracle value is large at fixed depth (driven by acceptance) and shrinks by about 13 points under exits.
3. **The two "native exit changes the optimal action" effects are AR fallback and adaptive K, and both look explainable by acceptance.**
   - Ouro: K/A grows from 1.7% to 13.4% under exits, and D is associated with the AR-vs-SD and K crossovers.
   - Both effects are already present to a large degree at fixed depth, and both follow acceptance, which F-080 links to D.
   - Whether D or recurrent features add *deployable* information beyond acceptance history and cost is Stage C's question, and it is not answered here.
4. Every value above is a same-prefix one-round oracle with perfect acceptance foresight, measured in eager B=1. A strong oracle paired with a weak controller would not count as a positive result.

## Next (requires explicit GPU authorization; not started)

Stage C needs new traces the current receipts lack:
- causal per-token history (past acceptance, rejected suffix lengths, previous token's D);
- pre-core hidden or gate statistics with their measurement cost;
- stage-wall instrumentation;
- prompt-disjoint train, validation and test splits, with ≥100 test prompts.

Following AGENT_TASK, the d-specific GPU follow-up is *not* indicated for Ouro. For Huginn it is indicated only as a fixed-depth acceptance question. The K and AR-fallback predictors (baseline B1 history + cost vs B2/B3 recurrent features) are the remaining candidates for Stage C.
