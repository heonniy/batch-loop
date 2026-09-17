# AGENTS.md

## Current priority — Phase 04R: resolve BF16 live-exit divergence before predictor research (2026-09-17)

Read in order:
1. `phase04/BF16_DIVERGENCE_RESOLUTION_PLAN.md` — canonical diagnosis/decision tree.
2. `phase04/BF16_DIVERGENCE_AGENT_TASK.md` — bounded executor handoff.
3. `phase04/results/20260917T052341Z_c591671/IMPLEMENTATION_REVIEW.md` — current Gate-A failure evidence.
4. `phase04/GATE_A_PARITY_RULE.md` — original prospective BF16 rule and why old failures remain failures.
5. `phase04/DEPTH_PATTERN_PREDICTOR_PLAN.md` — resumes only after Phase 04R reopening condition.
6. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — provenance and five-axis review.

### Current state

The N32/B8/T8 real live-exit Gate A failed in BF16 for both primary models, while the matched FP32 diagnostic passed. Ouro had 2/256 exit-depth disagreements; Huginn had 7/256 depth disagreements and 3 next-token disagreements. Huginn request/position-keyed h0 receipts matched and structural KV mapping did not fail. Near-threshold scores cross the exit boundary in saved examples.

This is not yet proof of a model/runtime bug and not yet proof of harmless expected BF16 behavior. The immediate task is to localize the **earliest** B1-versus-batched BF16 divergence and classify the cause.

### Hard block

Until the Phase-04R decision is complete:
- do not collect new predictor train/validation/test features;
- do not train/select a new predictor;
- do not publish a new FIFO/Oracle speed result;
- do not reinterpret old BF16 failures as passes;
- do not change threshold/q, checkpoint, KV policy, Rmax or primary precision just to make Gate A pass;
- do not use forced-depth replay to claim live-exit parity.

Prior Phase-2 frozen/forced-work packing results remain historical evidence under their original scope only.

### Diagnosis contract

Use saved mismatch request/token rows first. Freeze exact effective inputs, history, h0 identity where applicable and filler rows. Compare B1 versus controlled packed batches at progressively finer boundaries: prefill/Prelude, recurrent input, each physical layer, first divergent attention/MLP/norm/residual component, attention-visible K/V, final hidden, exit score, Coda/logits.

Then vary only batch width/order/row position/padding/precision in the bounded matrix specified in `BF16_DIVERGENCE_RESOLUTION_PLAN.md`. Add duplicate-target and row-permutation invariants. Localize before broadening to larger B or longer horizons.

### Decision branches

- **Branch A — implementation bug:** fix the demonstrated bug, add CPU/GPU regressions, rerun the original BF16 Gate A unchanged.
- **Branch B — expected BF16 batch-shape numerics:** do not call old Gate A a pass. Quantify depth/next-token flip rate versus B and exit margin on fresh validation data, then prospectively redefine the semantic contract or explicitly treat live depth as batch-dependent.
- **Branch C — unresolved:** publish the bounded negative diagnosis and stop predictor work.

A robust-margin/hysteresis exit policy would change the exit method and is a separate research decision, not a hidden correctness fix.

### Research consequence

If exact live parity is restored under Branch A, resume the original Phase-4 research question: can offline token/policy priors plus previous hidden/current embedding/Huginn Prelude information predict recurrence-depth compatibility before the recurrent interval?

If Branch B shows material batch-dependent depth, the predictor target and Oracle terminology must be updated before collection. B1 traces become a canonical/frozen workload rather than unique live counterfactual truth. Practical scheduling must be evaluated under actual live work; frozen-depth Oracle remains a structural packing upper bound only.

### Scope locks retained from Phase 04

- Strict no-refill: membership chosen before a recurrent interval; after start, exited rows are removed only.
- No cross-microbatch survivor merging or after-r1/r2 regrouping.
- Predictor, when reopened, controls grouping only and never the actual exit decision.
- Ouro-1.4B uses last-exited depth-indexed KV; Huginn-3.5B uses shared physical-layer KV.
- Previous hidden is a predictor feature, not a recurrent warm-start modification.
- Defer independent Prelude/Core/Coda batch sizing, mixed prefill/decode and dynamic initial B until predictor validation.
- No task-quality/GSM8K/perplexity study; same-policy state/KV/gate/logit checks are implementation validation.

### GPU/process ownership

Use only the exact physical-index/UUID/model-role mapping already recorded in `phase04/GPU_ASSIGNMENT_20260917.json`, unless a later explicit authorization supersedes it. Inspect running jobs before launch and do not interrupt unrelated work. No indefinite background sweep. Release devices after each bounded diagnostic.

### Publication

Preserve all prior sources/results. Use additive roots `phase04/results/<timestamp>_<source_sha>_bf16_resolution/`. Publish implementation changes before execution and measured result/review commits separately. Retain raw mismatch receipts, failed runs and blocked cells.

A completed Phase 04R handoff must state the first divergent operation, Branch A/B/C evidence, exact fix or new prospective contract if any, fresh Gate-A result if rerun, and what predictor/speed work remains blocked.
