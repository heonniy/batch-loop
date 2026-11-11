# Dynamic-Depth Self-Speculative Decoding — Dual-Architecture Discovery Plan

**Date:** 2026-11-11  
**Status:** PLAN ONLY — no experiment has been run by this change.  
**Owner:** Looped Transformer inference research  
**Repository evidence:** [H-SPEC-001](https://github.com/heonniy/research-wiki/blob/main/hypotheses/H-SPEC-001.md) and related [F-031](https://github.com/heonniy/research-wiki/blob/main/findings/F-031.md), [F-043](https://github.com/heonniy/research-wiki/blob/main/findings/F-043.md), [F-048](https://github.com/heonniy/research-wiki/blob/main/findings/F-048.md), [F-052](https://github.com/heonniy/research-wiki/blob/main/findings/F-052.md).

## Research framing: discover the problem first

**Question:** How do native per-token recurrent exits change the correctness, performance and best execution decisions of self-speculative decoding (SSD), when compared with a *matched native dynamic-depth autoregressive target*?

Do **not** begin with the conclusion that a different scheduling policy is necessary. Seek both positive and negative observations. Candidate signals:

1. Interaction of draft depth, target exit depth, and speculative acceptance/acceptance length.
2. Dynamic-depth AR speedup versus dynamic-depth SSD speedup; the latter can have lower *relative* speedup while both gain absolute throughput.
3. Physical wavefront occupancy: halted, padded, discarded/rejected, and useful recurrent work; distinction between lower FLOPs and lower wall time.
4. Correctness of commit order, early exit, speculative rejection, KV frontier/rollback, and token-conditional target distribution.
5. Value of policies that adapt the proposal depth, speculation width, and verification/branch priorities to observed halting/acceptance.
6. Model-specific effects of Huginn versus Ouro, including prelude/coda, exit rules, stochastic state initialization, and KV representation.

**Scope warning / novelty boundary:** The WaveFront paper's principal throughput results use a fixed target depth, but its **current public code already contains adaptive-depth AR and SSD** (see exp3 and `R_dyn`). Do not claim first implementation of dynamic WaveFront; instead reproduce its existing code path and show, if evidence supports it, a previously uncharacterized practical failure mode or a measured policy improvement.

## Two primary architecture lanes, start together

| Lane | Checkpoint | Native cap | Exit policy to characterize | Initial KV |
| --- | --- | ---: | --- | --- |
| O | ByteDance/Ouro-1.4B | 4 | Native cumulative hazard gate; threshold sweep calibrated to a non-degenerate depth distribution | Full depth-indexed reference; protected first + last-computed as **approximate** arm |
| H | tomg-group-umd/huginn-0125 (Huginn-3.5B) | 32 | Native Huginn criterion evaluated carefully in FP32; compare WaveFront's relative-L2 criterion rather than assuming equality | Full depth-indexed reference; last-computed KV as **approximate** arm |

**Replication lane:** ByteDance/Ouro-2.6B (same controls, separate calibration). Keep Ouro and Huginn equal-priority despite differences in upstream runtime support. Do not silently replace Huginn by Raven.

**KV policy is an orthogonal axis.** Full depth-indexed is the native parity reference. Huginn last-computed and Ouro protected recurrence-1 + last-computed are user-proposed fidelity arms, *not* proven native-equivalent. The past study F-048/F-052 finds partial rather than complete Ouro accuracy recovery. The WaveFront `kv_budget_s=1` ring-sharing option is NOT synonymous with this specific protected-bank policy; do not label it that way. Record prompt/decode/commit conversion and whether "last" is depth `D_t` actually exited or maximum `R`. Retain exact full KV as a required control; no memory/latency saving claim without physical measurements.

## Codebases: pinned, independently reproducible comparisons

1. [WaveFront public code](https://github.com/summerbro-hhj/wavefront-decoding): **first-priority experimental harness for both architectures.** Fixed target benchmarks: `experiments/exp2_spec_bench/run.sh`; dynamic AR + SSD: `experiments/exp3_accuracy/run.sh`; inspect `src/decoding/scheduler.py`, `src/decoding/{ouro_blocks,huginn_blocks,ar_baseline}.py`. The exp3 code implements per-token halt, freezes deeper KV at exit and can promote a contiguous chain of verifiers. Tests and paper do not by themselves establish native-equivalence or physical gains. **Set `KV_BUDGET_S=0`** for full-cache controls; exp3's shell defaults to `1` (an additional lossy intervention).
2. [LoopSpec public code](https://github.com/kaist-flexml-lab/loopspec): first reproduce official **Ouro** fixed-depth SGLang baseline (including one/two proposals and sampling). It does **not** list Huginn among officially supported checkpoints. Treat Huginn LoopSpec port as an explicitly separate engineering/validation task; WaveFront remains a common two-model comparator. Dynamic verifier is not the released LoopSpec benchmark.
3. [vllm-rlt](https://github.com/InferMatrixLab/vllm-rlt): pin release `v0.2.0` for later integrated serving comparisons. Documented Ouro path includes native exit and *experimental fixed-target-depth* block/pipeline speculation; read `docs/async-speculative.md` and `docs/runtime.md`. Huginn is documented only for synchronous fixed depth, **not** self-speculation or live-exit speculation. Do not assume multi-mode feature combinations work. Keep separate same-engine speed ratios; cross-engine absolute numbers are diagnostics only.
4. [casys-kaist-internal/loopserving.code, branch smcho/decision-placement](https://github.com/casys-kaist-internal/loopserving.code/tree/smcho/decision-placement): **READ ONLY**. At planning time GitHub returned 404 / no access in this connection. Do not guess its implementation or copy proprietary content. If access is granted later, inspect only; never write, clone into a public repo, or mix its private data into artifacts.

Pin source commit, checkpoint revision, tokenizer, runtime/driver/CUDA, attention backend, seed and exact configuration per measurement.

## Experiment phases (run both architecture lanes within each phase)

### E0 — Reference and equality gates

- Fixed AR on each *original author engine* → fixed WaveFront SSD (WFD) and draft-then-verify (DtV); Ouro additionally official LoopSpec. Replicate published directions without requiring exact numeric speedup across different hardware.
- Native adaptive AR baseline at several gate thresholds, using **both actual live exits** and *recorded-depth replay* diagnostic. Confirm depth histograms and gate implementation; Huginn WaveFront relative-L2 must be compared with native criterion (the old BF16 native cosine evaluation collapsed; F-057).
- Equal-output gate for fixed SSD against fixed AR; separate dynamic SSD against **matched dynamic AR** using the **same KV policy** and semantics. Greedy token identity, logits (max abs / KL), prefix validity, correction, hidden/KV frontier, and repeated random-seed sampling. A fixed AR and a dynamic AR are *different targets*, not necessarily identical distributions.
- BF16 batch-shape divergence has already failed in F-013/F-031. At each stage record both strict B=1 parity and practical batch-local / relaxed fidelity. Do not reclassify old failures as passes.
- Failure artifacts: first mismatch token/depth, branch lineage, accepted prefix, emitted logit/KV source, gate output, batch shape and seed. Stop quality/speed claims for a mismatched target until the mismatch is explained.

### E1 — First discovery (without a proposed new algorithm)

A complete matched matrix per lane:
- `AR-fixed`, `WFD-fixed`, `DtV-fixed`, `AR-live`, `WFD-live`, `DtV-live`; `LoopSpec-fixed` for Ouro.
- Additional `WFD-replay` and `DtV-replay` for geometry attribution only, never reported as native live-output speedups.
- Ouro: test first proposal depths `d in {1,2}`. Huginn: `d in {2,4,8}`, plus a small `d=1` diagnostic if numerically/implementation safe. Do not conflate changing draft depth with changing target exit threshold.
- Calibrate **three non-degenerate realized-depth regimes**, separately for each model and task (shallow/mixed/deep); also include forced-full `D=R`. Do not reuse numerical thresholds across incompatible exit criteria.
- `B in {1,8,32}` plus memory-limited and no-refill subsets where real demand exists; `K in {1,2,4,8}` where the engine implements configurable K. WaveFront's natural wave width is not an independently configurable `K`; record its actual active width.
- Primary workloads: GSM8K, MATH-500, Spec-Bench; add BBH and code tasks (HumanEval+/MBPP+) as sensitivity / transfer tests. Use matching prompt templates, cap-hit reporting and actual generated token counts.
- Begin with **32 prompts per model × task smoke**, **100 per model × task pilot**, then **200+ matched prompts per main model/task** after correctness gates; separate fixed prompt pools and held-out policy selection. These are planned, not reported sample sizes.

### E2 — Measure conditional relationships

Store per-token/per-candidate, with a committed/rejected flag:
- `D_t`, exit gate trajectory / latent change, `d`, verifier depth, logits/top-1 entropy and draft-target agreement;
- draft acceptance probability, accepted-prefix length, rejection position, correction frequency and position-dependent acceptance;
- conditional tables `P(accept | D_t, d, task, prefix position)` and acceptance length vs `D_t`, with rejection/censoring and speculative-prefix effects explicitly noted;
- special cases `D_t < d`, `D_t == d`, `D_t > d`: early target halt before draft budget, duplicate draft/verification work, or prolonged verifier;
- live vs replay difference at the **same observed depth trace**; do not assume a speculatively proposed token has the same exit as the finally accepted token.

### E3 — Actual physical work and latency attribution

Measure `useful_recurrent_rows`, `executed_recurrent_rows`, `padded_rows`, `halted_waiting_rows`, `discarded_rejected_rows`, `recomputed_rows`, P/C/LM-head launches, HBM/KV bytes where measurable, kernel count, CUDA Graph captures/fallbacks, peak allocated/reserved HBM, first output/TTFT, TPOT, committed tok/s, and E2E.

Compare:
- fixed-width running until max depth,
- halted-row masking,
- physical compaction, 
- oracle replay with the same accepted trace and depth but **no unverifiable anticipation**,
- real live exits.

Contrast `SD_fixed / AR_fixed` against `SD_live / AR_live` and report both absolute committed tok/s and *relative* speedups. Distinguish smaller logical FLOPs from actual physical Core calls and wall time. If full-batch execution already dominates smaller cohorts (F-020/F-035), do not assume re-grouping is beneficial.

### E4 — KV and state intervention factorial

At matched models/thresholds/dataset and physical engine:
1. Depth-indexed FULL KV (reference).
2. Huginn last-**computed** shared policy; non-native-depth and first-only diagnostics where useful.
3. Ouro protected first-recurrence bank(s) + last-computed; use separate `L0R1` and `ALL-R1` protection; FIRST-only, LAST-only and matched-budget random-bank negative controls.
4. Rejected-branch rollback correctness, mixed-depth reads, committed prefix repair, stale speculative slot invalidation.
5. Prefill FULL with decode-only approximation vs end-to-end shared conversion (explicitly different); compare native vs replay depth and gate behavior.

Do not confuse quality differences caused by KV approximation with effects of speculative scheduling. Initial optimization decisions must be justified on the FULL-KV control, or labeled approximate / non-equivalent.

### E5 — Broader hypothesis search / falsification

- **Counterfactual draft**: keep verifier and KV fixed while sweeping `d`; compare conditional acceptance and wasted verifier work.
- **Target-depth crossover**: compare fixed vs dynamic SD speedups at equal quality where possible, and explain whether changing denominator (faster AR) accounts for the entire apparent regression.
- **Depth variance**: fixed mean depth with differing per-token variance via recorded traces; quantify whether heterogeneity, not average depth, creates idle/padded physical work.
- **Rejection timing**: measure if early halts on speculative descendants cause greater discarded computation, including chains of multiple simultaneously halted tokens and first-rejection positions.
- **Exit gating location**: inspect whether gate/Coda/head calls add repeated overhead or a late dependency; compare real latencies, not just counted loops.
- **Batch/refill sensitivity**: fixed no-refill offline cohorts, refill allowed, short vs long prefixes; verify that any scheduling gain survives optimized CUDA Graph paths.
- **Precision, stochasticity, and architecture**: BF16 vs bounded FP32 diagnostic, Huginn state initialization seeds, criteria calibration, Ouro-1.4B vs Ouro-2.6B, long generated reasoning.
- **Negative results first-class**: absent correlation, already-compact engine, worse than dynamic AR, and accuracy collapse are reportable outcomes.

### E6 — New decision strategy only after E1–E5 show evidence

Try simple baselines before a learned controller:
- fixed `d,K` and threshold; 
- acceptance-only / gate-only controller;
- joint `d,K,D_t` strategy minimizing *observed* expected milliseconds per **committed target-equivalent token**, subject to quality and memory constraints;
- shortest-ready / deepest-ready / age-based branch priority, plus full-cohort and random controls;
- oracle with hindsight **upper bound**, never a live achievable policy.

Use held-out policy thresholds and timings, report scheduler overhead and inferential limits. A new method needs reproducible physical throughput gain over the **best matched dynamic-depth SSD baseline**, not merely over fixed-depth AR.

## Exactness, metrics and stopping decisions

**Scientific claim under consideration (not yet supported):** "Token-adaptive recurrent halting changes the useful speculative-work frontier enough that fixed-depth-optimized draft/verification decisions may become suboptimal in some architectures/workloads."

**Would support:** nontrivial, repeatable correlation/crossover + matched dynamic-AR fidelity + physical waste that simple baselines fail to remove + measured gains of a changed decision policy.

**Would falsify or narrow:** existing WaveFront dynamic halt/compaction already captures the useful headroom; depth variation does not change the optimal decisions; speedup reductions are solely because AR got faster; profiler overhead erases logical-work savings; approximate KV dominates all output changes.

Report each: model revision, dataset/n, B, prompt/output caps, d, K, gate rule/threshold, actual depth distribution, sampling/seed, dtype, KV semantics, runtime commit, warmup/timed repetitions, output-equivalence status, confidence intervals, first divergence, exact wall breakdown. Benchmark `>=2` warmups and `>=5` scored repeats for primary physical settings; do not call replay/analytical cost a measured speedup.

## Execution boundary and publication

This is a **design-only artifact**. No GPU jobs or source modifications were launched by this PR. The current chat environment has no NVIDIA GPU runtime. Actual baseline reproduction, source modifications and benchmark commits require an equipped GPU host and explicit run receipts.

Leave `AGENTS.md` and past results untouched. Produce isolated source adapters under `dynamic_sd/` only after initial reproduction and tests. Keep each runner's original fixed-depth code path pristine and capture upstream revisions. Do not write to `casys-kaist-internal/loopserving.code`. Publish smoke / pilot / main separately with scripts, structured trace, summary, `RESULT_REVIEW.md`, `review_manifest.json` and full negative/OOM/failure cases.

Related existing work `IDEA-002` / `IDEA-003` covers general serving characterization; `H-SPEC-001` covers this distinct speculative verification study. No new wiki Finding or Decision follows from this plan.
