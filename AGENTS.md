# AGENTS.md

## Current priority — Phase 05 regime/architecture/oracle study (2026-10-19)

The user asked for E1–E3 of `phase05/AGENT_TASK.md` and confirmed GPU ownership: they stopped their own load job on GPUs 0–3 for this study (`phase05/GPU_ASSIGNMENT_20261019.json`). GPUs 4–7 belong to other users. E1 (CPU), E2 (H100 stage microbenchmark) and E3 (latency-aware Oracle; replay plus PHYSICAL_FORCED_DEPTH runs) are complete. Start at `phase05/results/20261019/RESULTS.md` and `REVIEW.md`. Nothing in Phase 05 is live-exit performance. The Phase-04 BF16 Gate A remains FAIL, and the predictor block below still applies. The Phase 04R "no GPU" rule is superseded only for this authorized Phase 05 scope.

## Current priority — Phase 04R: quantify BF16 depth-rank stability from existing traces (2026-09-17)

Read in order:
1. `phase04/DEPTH_RANK_STABILITY_CPU_PLAN.md` — current no-GPU analysis contract.
2. `phase04/DEPTH_RANK_STABILITY_AGENT_TASK.md` — bounded executor handoff.
3. `phase04/results/20260917T060824Z_50910e6_bf16_resolution/CAUSE_REVIEW.md` — evidence that BF16 batch shape can perturb live exit decisions.
4. `phase04/results/20260917T060824Z_50910e6_bf16_resolution/PROPOSED_B2_CONTRACT.md` — draft only; NOT active.
5. `phase04/BF16_DIVERGENCE_RESOLUTION_PLAN.md` — prior diagnosis/decision tree.
6. `phase04/GATE_A_PARITY_RULE.md` — original BF16 parity rule; its failures remain failures.
7. `phase04/DEPTH_PATTERN_PREDICTOR_PLAN.md` — predictor work remains blocked pending the research-contract decision.
8. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — provenance and review protocol.

### Current state

Phase 04R reproduced BF16 B1-versus-batched live-depth differences and isolated batch-shape-sensitive numerical divergence in model operators. Fresh N32/T8 receipts show a small number of depth flips across B=2/4/8 and committed row orders. The original BF16 Gate A remains FAIL; FP32 diagnostic success does not retroactively pass it.

The immediate question is now narrower:

> Are exact BF16 exit depths slightly batch-sensitive while the **relative shallow/deep recurrence-demand ordering** remains stable enough for depth-aware batch formation?

Answer this first from existing committed receipts. Do not run a model to answer a question the current traces can already test.

### Immediate no-GPU task

Use CPU only to compute:
- `live_depth - canonical_B1_depth` magnitude and direction;
- per-token-step Kendall tau-b / Spearman ordering stability;
- preserved/tie/inverted request-pair ordering by canonical depth gap;
- request-token live-depth range across the already observed B/order conditions;
- descriptive relation to committed live work/RIC.

The scheduling unit is the same `token_idx` ready pool, so do not flatten different decode steps into one rank correlation. Treat ties explicitly. Do not invent a live Oracle by applying one grouping's observed depths to an unexecuted grouping.

### GPU/process rule

**No GPU execution is authorized or needed for the current task.** Do not launch CUDA/model jobs on GPU 2, 3, 5, 7 or any other device. Do not touch unrelated processes. A later explicit task may authorize fresh confirmation only after the CPU analysis and research-contract decision.

### Hard block retained

Until the CPU rank-stability review is published and the research contract is explicitly decided:
- do not collect new predictor train/validation/test features;
- do not train/select a new predictor;
- do not publish a new FIFO/Oracle speed result;
- do not run B16/B32/B64 just to broaden the evidence;
- do not reinterpret old BF16 failures as passes;
- do not change threshold/q, checkpoint, KV policy, Rmax or primary precision;
- do not activate `PROPOSED_B2_CONTRACT.md` automatically.

Prior Phase-2 frozen/forced-work packing results remain historical evidence under their original scope only.

### Interpretation contract

B1 depth is a **canonical profiling depth** for this analysis, not unique batch-invariant truth. The primary question is whether its relative ordering transfers to live BF16 executions at the already observed legal shapes.

Do not predeclare that all flips are ±1 or harmless. Measure the complete delta histogram first. Do not use a post-hoc universal pass threshold. The review should distinguish:
- neighboring-depth swaps/ties;
- large canonical-gap inversions;
- exact-depth perturbation magnitude;
- cross-B/order range for the same request-token.

If evidence strongly preserves coarse ordering, record `supports ranking-contract discussion`; if mixed, record `mixed evidence`; if large-gap order is materially unstable, record `ordering materially unstable`. None of these statuses activates B2 or restarts predictor work by itself.

### Scope locks retained from Phase 04

- Strict no-refill: membership chosen before a recurrent interval; after start, exited rows are removed only.
- No cross-microbatch survivor merging or after-r1/r2 regrouping.
- Predictor, if later reopened, controls grouping only and never the actual exit decision.
- Ouro-1.4B uses last-exited depth-indexed KV; Huginn-3.5B uses shared physical-layer KV.
- Previous hidden is a predictor feature, not a recurrent warm-start modification.
- Defer independent Prelude/Core/Coda batch sizing, mixed prefill/decode and dynamic initial B until predictor validation.
- No task-quality/GSM8K/perplexity study.

### Publication

Implement deterministic analysis under an additive Phase-04 path and commit source before derived results. Publish results to:

`phase04/results/<timestamp>_<source_sha>_rank_stability_cpu/`

Preserve raw Phase-04R evidence unchanged. Record input SHAs, source SHA, Python/package versions, exact command and row counts. Stop after the CPU result/review commit and request the next research decision.
