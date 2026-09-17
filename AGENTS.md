# AGENTS.md

## Current priority — Phase 04: implementation validation, depth patterns, prior + hidden predictors (2026-09-17)

Read in order:
1. `phase04/IMPLEMENTATION_AUDIT.md` — reviewed implementation findings and evidence limits.
2. `phase04/DEPTH_PATTERN_PREDICTOR_PLAN.md` — canonical validation, data-analysis, predictor and timing contract.
3. `phase04/AGENT_TASK.md` — bounded executor handoff.
4. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — provenance, result publication and five-axis review.
5. Prior Phase-2/3 reviews for historical evidence only.

This priority supersedes the earlier Phase-3 instruction to choose candidates using test RIC. Select on validation, freeze choices, then confirm on a fresh test cohort. The earlier selection instruction was itself flawed; do not blame only the executor or silently rewrite historical outcomes.

### Research question

Can information available BEFORE the current token's recurrence — offline token/policy depth priors, previous exit hidden, current token embedding, causal history and Huginn Prelude/initial state — support better depth-compatible microbatch formation in offline no-refill Looped-LM inference?

First establish implementation correctness and measurement integrity, then determine which features explain useful depth differences, then train and confirm practical predictors. The old lightweight tournament is not a proven pre-loop predictability ceiling.

### Scope locks

- Strict no-refill. At the token boundary, choose the microbatch membership; after it starts, remove exited rows only. No new rows or cross-microbatch survivor merging during that interval.
- No after-r1/r2 regrouping or prediction-free repacking pivot.
- Predictor determines grouping, never the actual exit decision.
- Ouro-1.4B: last-exited depth-indexed KV; Huginn-3.5B: shared physical-layer KV.
- Freeze model weights, checkpoint, KV/exit/init policy and thresholds within comparisons.
- Previous hidden is a predictor feature, not a warm-start change to model recurrence.
- Defer independent Prelude/Core/Coda batch-size optimization, mixed prefill/decode and dynamic initial B until predictor validation.
- No task-quality, GSM8K, perplexity or native/full-depth quality comparison. Same-policy state/KV/logit/gate correctness checks remain mandatory.

### Gate A comes before expensive collection

Address the audit's reproducibility, live-versus-replayed exit, timing instrumentation and selection issues. In particular:
- match realized per-request/token Huginn stochastic states across collector, reference and batched runtime; a seed alone is insufficient;
- compare physical compaction with an independent reference under the SAME chosen semantics;
- separate bounded debug/profile instrumentation from scored benchmark execution;
- verify effective tokenized-input split disjointness and every consumed artifact;
- re-establish clean FIFO/Oracle timing under the corrected runtime.

Do not announce that existing GPU forced-depth tests certify live adaptive behavior. Record unexecuted/blocked checks honestly. A source-inspection concern is not automatically a reproduced model bug.

### GPU ownership and existing work

No GPU launch is performed by this planning commit. Before any new model execution, require an explicit authorized model-to-physical-GPU/UUID mapping in the executor's current task or launch manifest.

Historical instructions conflict: the previous AGENTS specified GPU7=Huginn, GPU8=Ouro; Phase-3 source hard-codes GPU7=Huginn, GPU6=Ouro and claims an override. Do not infer permission from that code, invent a correction, or silently fallback to GPU6/GPU8. If authorization or hardware identity is unresolved, stop GPU execution for that lane and continue CPU/source work. Use only the authorized devices, validate CUDA_VISIBLE_DEVICES versus physical identity, and leave all existing jobs untouched. No background soak or indefinite run is authorized.

### Data and selection

Use new versioned result roots. Split by request/conversation/document family and hash actual model-input tokens after templating/truncation, not only raw prompt text. No current/future exit information enters practical predictor features. Canonical IDs are bookkeeping, not predictive features.

The detailed plan starts with a train256/validation128 pilot and bounded expansion to offline priors, raw-hidden probes and prior-plus-context residual prediction. Primary policies remain Huginn threshold0.12/Rmax16/shared and Ouro q0.2/Rmax4/last-exited, with fixed secondary policies and fresh ShareGPT transfer. Never tune thresholds/test cohorts to manufacture recovery.

Validation chooses models, hyperparameters, calibration/grouping settings and physical finalists. Freeze choices before opening new confirmation requests. Existing Phase-3 test results are exploratory selection evidence, not untouched confirmation for new methods.

### Metrics and fairness

Use fixed-B FIFO/Predictor/Oracle comparisons on identical eligible cohorts. RIC is a structural diagnostic, not latency. For equal fixed W and B, logical idle slots = B*RIC-W. N=B and B=1 have no composition RIC opportunity.

Report:
- RIC reduction and Oracle RIC recovery;
- actual executed token-loop work and active tensor widths;
- decode-only wall and throughput;
- full-workload BCT only when prefill and all relevant scheduling are timed;
- feature maintenance, predictor, ordering, KV and compaction overhead.

Use the same Prelude staging for all Huginn strategies. Compare to an Oracle measured on the SAME corrected runtime and cohort, not a historical denominator. Label frozen-depth/input replay, live-exit verification and free generation separately. Recovery with a zero Oracle-FIFO denominator is NA. Preserve negative recovery and failed capacity points.

### EasyBalance inspiration

Borrow ready/deferred/RunSet selection of work BEFORE the recurrent interval. Do not import cross-layer execution, device load-balance formulas, or a no-overhead guarantee. Reordering already fixed microbatches alone cannot reduce their sum of maximum depths; a claimed gain needs a change in actual pre-interval composition or another explicitly measured mechanism.

### Execution and publication protocol

Publish source commits before execution, then result and review commits separately. Preserve all Phase-1/2/3 sources and result roots. Use `phase04/results/<timestamp>_<source_sha>/` for new artifacts. Prefer subjects `phase04: audit ...`, `phase04: implement ...`, `phase04: collect ...`, `phase04: publish ...`, `phase04: review ...`.

The executing agent must provide exact commands, source/model/config hashes, GPU identities, test PASS/FAIL/BLOCKED status, frozen splits and artifact hashes, all candidate outcomes, repeated physical timings and a five-axis review. Never fill unknown metadata from intended plan values.

A completed bounded study must identify the best measured candidate, even if weak, without guaranteeing a positive result or changing the problem after viewing confirmation data.
