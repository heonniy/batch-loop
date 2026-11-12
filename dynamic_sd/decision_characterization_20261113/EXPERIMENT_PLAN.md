# Dynamic-Depth SD Decision Characterization — observation before controller

**Date:** 2026-11-13 · **Status:** PROPOSED / PLAN ONLY. No new CPU or GPU experiment has been executed by this commit.  
**Starting point:** `heonniy/batch-loop@5c6bbea3`, [follow-up RESULTS](../followup_20261112/RESULTS.md), original [dynamic SD review](../RESULT_REVIEW.md). Pin WaveFront `4345209` until a separately gated optimized-engine stage.

## Main research question

Does native token-specific recurrent exit make **which speculative action is optimal** depend on recurrent computation depth, *beyond* ordinary variation in draft acceptance, draft length, token difficulty and hardware cost?

Do **not** presuppose that a dynamic controller is useful or that deep target exit should imply deeper draft. Separate three decisions:
- D1: **AR vs SD** (whether to speculate);
- D2: draft recurrent depth `d` (number of core iterations used per draft token);
- D3: speculative length `K` (number of proposed future tokens).

`D` means **post-verification native target exit depth**. It is not a causal pre-verification feature. `R` is the maximum target recurrence. Neither `K` nor draft-tree depth in standard SD papers is recurrent `d`.

## What exists already and what differs

| Record | Already measured | Required *new* delta |
| --- | --- | --- |
| Research wiki `F-079` | Fixed/live full-run eager WFD/DtV vs matched AR on Ouro/Huginn; prefill-inclusive BF16 GSM8K n200 | Same-prefix decision attribution with FP32 parity and *separate* d/K/AR contribution |
| `F-080` | Post-hoc P(accept\|D) for Ouro/Huginn | Match acceptance / prefix position / remaining round budget and compare physical **cost**, not only acceptance |
| `F-081` | WFD d-sweep: Huginn best d4 already fixed; Ouro best d1 | Native DtV and, separately, WFD action-conditioned d crossover; no conflation of execution method |
| `F-082`, `H-SPEC-002` | Ouro lockstep B<=64, AR and DtV only | Later optimized B>1 concurrent branch + depth physical comparator |
| `F-083` & new `5c6bbea3` | FP32 FULL native live parity on n16–32; BF16 failures persist | All new action choices must preserve matched native AR target; report BF16 separately |
| `F-085` & `5c6bbea3` | Eager gate shadow + 1-round oracle; gate not graph-profiled | No production speed claim until Stage 0b |
| `F-032/F-033` | Pre-core depth prediction for **batch grouping**, not SD | Test incremental *speculation-action* predictive value on unseen prompts |
| `IDEA-002/IDEA-003` | Serving/runtime characterization already in backlog | Speculation-specific branch×depth decision interactions only; no duplicate generic CDB study |

Current `5c6bbea3` results: one-round cross-fitted oracle / best static on GSM8K FP32 B1 is Ouro fixed 1.176, tau0.4 1.220, tau0.1 1.257; Huginn fixed 1.259 and eps0.1 1.251. **This headroom exists with fixed depth** and uses oracle knowledge from action trials of the same prefix, not an online predictor.

## Stage A — CPU-only decomposition on existing measured counterfactuals (FIRST)

**No GPU needed; reuse actual committed** `dynamic_sd/followup_20261112/results_20261112/cf/*.jsonl.gz` only.

1. Audit SHA manifest and metadata; load direct compressed JSONL. Verify each `prefix_id` has identical `prefix_hash`, `target_hash`, `exit_policy`, FULL KV, precision, backend, action grid, 3 repeats, measured=true, `wall_ms>0`, `committed_tokens>=1`. Reject inexact actions *as a complete-prefix grid*; report number and selection bias.
2. For each exit/model cell choose the best single fixed action on a prompt-disjoint calibration split, NOT the evaluation prompts.
3. Partition one-round actions into **nested** policy classes:
   - `S`: best fixed action (AR or d/K), constant.
   - `A`: AR vs the best fixed SD action (speculate or not).
   - `K`: AR vs DtV(d0,K) where d0 is the calibrated fixed d; K varies.
   - `d`: AR vs DtV(d,K0) where K0 is the calibrated fixed K; d varies.
   - `Joint`: AR vs all measured DtV(d,K).
   Ensure `S⊂A⊂K/Joint` and `S⊂A⊂d/Joint`; K and d are not mutually nested. Report fixed d0,K0, all candidate actions and fallback count.
4. For descriptive *oracle* comparison, select the action for each identical prefix on two timing repeats, score on third, with prompt-cluster bootstrap CI; do NOT label this causal or held-out-prompt generalization. For a generalization screen, learn or select only **deployable historical features** on train prompts and evaluate completely unseen prompts; do not look up prompt/prefix IDs.
5. Compute incremental gains: `A/S`, `K/A`, `d/A`, `Joint/K`, `Joint/d`, and cross-regime comparison `(Joint/S)_live -(Joint/S)_fixed` (differences are descriptive across distinct native target trajectories, not causal effect of only D). Compute matched static/one-round rate via aggregate committed tokens / total measured time, not mean of per-round ratios. Also plot fraction choosing AR, d, K, round acceptance, rejected work and wall.
6. **No conclusions from aggregate-only d/K values**: stratify by model, exit regime, prompt and verification route; when K varies, tokens advanced differ, so one-round reward comparison cannot be turned into a multi-round speedup.

**Decision screen:** If `Joint/K` is within timing uncertainty or <=1% in every FP32 live and fixed cell, token-adaptive d is a low-priority direction. If `K/A` is negligible, dynamic length adaptation alone is also weak. Never call the mere existence of `Joint/S>1` a depth-specific finding.

## Stage B — Depth-conditioned cost and crossover characterization

Start with existing FP32 FULL same-prefix DtV round rows. Capture every action's `reference_next_D`, draft acceptance prefix length, `committed_tokens`, verifier Core rows/calls, stage Core/Prelude/Coda wall and temporal work; **verify whether existing receipts already contain them before collecting new traces**.

- Compare `U(d,K|D)=committed_tokens/wall_ms` with **fixed K** when testing d (Huginn d2/4/8, Ouro d1/2) and **fixed d** when testing K (K1/2/4/8 for Ouro; K2/4/8 Huginn). Reference D is a post-hoc stratum; label it explicitly.
- Report `ΔU(d2,d1;D,acceptance,K)` with paired prefix trials (identical committed-prefix cache) and prompt-bootstrap CI, plus a simple absolute *latency* difference. Distinguish no-crossover, noisy crossover, and reproducible crossover; require minimum n per D bin and show all n. Huginn rare D>20 goes to tail bin with uncertainty, not exaggerated.
- **Confounding controls:** match/stratify number accepted or acceptance-prefix length, prefix context length, prompt ID and token position; two actions can have different accepted lengths so common-support matching must be reported; conditioning on outcomes can induce selection bias, never call stratification causal.
- Separate (i) natural live D and (ii) an explicit intervention to hold committed prefix/target path fixed while forcing *counterfactual verification budget* or matched replay depth on equivalent action. Any forced-depth arm has a **different target** unless matched semantics are proved; classify as diagnostic only, never mix quality-matched speed.
- **Falsifier:** deep-D cases do not favor larger d after actual cost is included, or d/K optimum depends almost entirely on acceptance length regardless of D. This is a *valid negative result*.

## Stage C — Does recurrence add causal, deployable information?

Only if A/B reveal enough incremental headroom:
- Predictor target: `best action` or measured `expected committed tokens / wall_ms`; explicitly define pre-decision time. Unseen **prompt-disjoint** train/validation/test; entire prompts assigned before feature extraction; all 3 repeat trials/prefix stay together.
- Baseline B0: globally calibrated fixed d/K and AR. B1: causal past-acceptance EWMA, past rejected suffix lengths, previous-token prior, context length and profiled cost; B2: B1 + prior emitted token's final exit D and/or depth histogram; B3: B2 + current pre-core hidden and optional earliest-loop gate statistics **only if their measurement cost is accounted for**.
- Explicit `D_future` oracle only as an inaccessible diagnostic, never a predictor feature. No post-verification acceptance of the *current* candidates; no same-prefix action-trial performance in train features. Compare B2/B3 **incremental net utility over B1** for live and fixed target regimes; freeze feature extraction and thresholds on validation.
- Measure calibration of action decisions, top-1 d and K, minority AR fallbacks, per-prompt net throughput, predictor wall overhead (including GPU->CPU sync). Include smaller sample (smoke 32 prompts/model/regime) only as validation of instrumentation, then >=100 held-out test prompts per condition if early signals justify it.
- **Falsifier:** causal recurrent features do not lift true held-out / end-to-end utility over history+cost, or benefit vanishes after runtime overhead. A strong oracle plus a weak controller is **not** a viable positive result.

## Stage D — Physical optimized serving gate (conditional)

Implement only after A–C indicate live-specific action gains:
- Optimized backend shared by AR/SD, proper CUDA Graph support, device-native exits and CUDA Events/NVTX for per-loop gate, host sync and branch/verification. Compare B1/8/32 and optionally 64, FULL KV, no-refill cohorts with explicit strict-vs-relaxed BF16 parity, same prompt/length/admission and checkpoints. No assumed `vllm-rlt` native dynamic SD path without a separate port/parity gate.
- Baselines: strongest static action, acceptance-only dynamic K, previous-depth-only, recurrent-aware dynamic d/K, optimized WaveFront / depth-asynchronous verifier. Lockstep is a diagnostic, not the only baseline. Report memory and latency only when physically measured.
- Distinguish `no-refill` admission from `lockstep` output barrier and preserve previous `F-035` negative: unconstrained whole cohorts beat artificial sub-batching.

## Workload and evidence contract

Models: Ouro-1.4B R4; Huginn-3.5B R32 (independent gates; no reused tau/eps). First A/B on existing GSM8K FP32 prompts. Fresh C: GSM8K and MATH-500, then Spec-Bench / code transfer where supported; pre-register prompt offsets/splits, outputs, seeds and threshold regimes before timing. Use native FULL KV first; no lossless claim for s=1 or Ouro protected First+Last. Keep FP32 parity lane distinct from BF16 performance and distinct attention backend.

Record for each trial: commit hashes, model/checkpoint, workload, prefix ID/hash, native reference target, dtype/backend/exit threshold, active D, d, K, AR/SD action, outputs/exactness, accepted length, candidate source depths, actual executed Core invocations/rows, prelude/coda calls, CUDA event stage timings, CPU stalls, wall, repetition, context length, GPU and graph bucket. Result files go in a **new dated directory**. Never overwrite `results_20261112/`.

## Required reports and stop conditions

Write `RESULTS.md` with tables for A/S, K/A, d/A, Joint/K, Joint/d, fixed versus live and prompt CIs; show zero/null/negative cells and number of excluded trial/prefix records. For each measured claim, label it *SOURCE MEASURED*, *DERIVED FROM RECEIPTS*, or *HYPOTHESIS*. Include code/receipt manifest and first parity mismatch. Explicitly say whether specific d crossovers or recurrent-feature gains were falsified.

**GO/NO-GO (research judgment, not an Accepted Decision):** A meaningful d/K specific effect requires non-negligible, repeatable *incremental* gain vs the best simpler policy on unseen prompts at strict target fidelity and physical optimized runtime. No such result exists yet. No automated decision update.
