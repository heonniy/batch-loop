# Phase 04R — Resolve BF16 live-exit divergence before predictor research

Date: 2026-09-17
Parent evidence: `f84bd0c02855de2848e449f4c4eab45f4cbd2dbf`
Status: resolution plan only. This file does not claim a fix or a passing rerun.

## Why this phase exists

The prospective N32/B8/T8 live Gate A failed in BF16 for both models while the matched FP32 diagnostic passed.

Observed BF16 anchor:

| Model | depth mismatches / 256 | next-token mismatches | B1 / compact recurrent calls | notable first mismatch |
| --- | ---: | ---: | ---: | --- |
| Ouro-1.4B q=0.2 | 2 | 0 | 617 / 615 | request 22, token 6: cumulative gate 0.199307978 vs 0.200557709 crosses q=0.2 |
| Huginn-3.5B threshold=0.12 | 7 | 3 | 2108 / 2105 | request 25, token 1: 0.120288983 vs 0.119749524 gives depth 10 vs 9; an earlier token-0 next-token mismatch also exists |

Huginn request/position-keyed h0 receipts matched and structural KV mappings did not fail. FP32 N32/B8/T8 had zero recorded depth/next-token disagreements for both models. These facts narrow the diagnosis toward BF16 batch-shape numerical effects, but they do **not** yet prove the first layer-level cause or prove that the BF16 runtime is semantically acceptable.

The research cannot simply continue with B1 depth as an unquestioned label. If true live depth changes with batch shape, a B1 trace is a canonical trace, not necessarily the exact counterfactual depth under another grouping.

## Non-negotiable rules

1. Do not collect Phase-4 predictor training data, choose a predictor winner, or publish new FIFO/Oracle speedups until the resolution decision below is completed.
2. Do not change q/threshold, checkpoint, KV policy, Rmax, prompt set, or primary precision solely to obtain a pass.
3. FP32 is a diagnosis, not an automatic replacement deployment configuration.
4. Preserve every prior FAIL and raw receipt. A new protocol may apply prospectively only.
5. No forced-depth replay may be used to claim live-exit parity.
6. Keep strict no-refill and current model-specific KV semantics while diagnosing.
7. Use the already authorized Phase-4 GPU mapping/UUID contract; do not infer new device permissions from historical code.

# R0 — Freeze the failing counterexamples

Extract a small immutable diagnostic packet from the saved Gate-A roots.

Required target rows:
- every Ouro depth-mismatch request/token, including request 22/token 6;
- every Huginn depth or next-token mismatch, including the earliest token-0 divergence and request 25/token 1;
- at least two matched, far-from-threshold controls per model.

For each target save canonical request ID, exact effective input IDs/mask, previous token IDs, exact h0 identity where applicable, KV-policy signature, exit scores/margins at every recurrence, reference/compact active-row order, and source/model/config hashes. Do not regenerate a "similar" prompt as a substitute.

The target packet is the unit used in R1-R4 so changes in data do not masquerade as fixes.

# R1 — Find the earliest numerical divergence, before the exit decision differs

The goal is not to inspect the final 10-30% relative error after trajectories have already separated. Find the first operation where the same logical row begins to differ.

Add bounded hooks/checksums for the target row only. Compare B1 versus a controlled B>1 run at:

1. prefill last hidden/logits;
2. current-token embedding / Prelude output;
3. recurrent input state before recurrence r;
4. each physical Transformer layer input/output;
5. within the first divergent layer: norm output, Q/K/V projections, attention output, output projection, MLP/up-gate-down output and residual additions as applicable;
6. attention-visible K/V actually read for the target row;
7. final recurrent hidden;
8. Ouro lambda/CDF or Huginn relative-difference exit score;
9. Coda/logits for the same token.

Record max-abs and relative-L2 **at the first divergence**, not only maxima after decisions split. Dump a full tensor only for the first divergent component and a small neighborhood; otherwise store hashes/norms to avoid huge receipts.

Run the same hooks in FP32 as a control. The layer-level diagnosis is PASS only if the first BF16 divergence point is identified or the tool can prove the difference exists before the first instrumented point. "Probably BF16" is not a localization result.

# R2 — Separate batch-size, row-order, padding, and implementation effects

Use one frozen target request with fixed history/h0 and fixed filler requests. The target's logical input must not change.

Required matrix for each model, bounded first to the saved mismatch token:

| Axis | Values |
| --- | --- |
| batch width | B=1,2,4,8; add 16/32 only after the bounded matrix |
| target row position | first, middle, last |
| row order | FIFO and a fixed permutation |
| prompt lengths | equal-length control and original mixed-length cohort |
| precision | BF16 primary diagnostic, FP32 control |
| execution | live gate; no forced depth |

Important comparison: keep the target and filler rows identical while only changing the tensor batch shape/order. This tests whether the target trajectory itself changes with kernel shape rather than with logical context.

Also run two invariants:
- duplicate the exact target input into multiple independent rows with identity-distinct bookkeeping and check whether equal logical rows remain numerically equal within the same batch;
- permute rows and restore canonical order, checking for any evidence of cross-row/KV contamination.

Interpretation:
- divergence tied to a specific logical row/KV mapping or permutation suggests an implementation bug;
- small divergence beginning inside a standard BF16 linear/attention kernel, varying with GEMM/batch shape, while bookkeeping/KV identities stay correct suggests ordinary low-precision batch-shape numerics;
- mixed behavior must be reported as unresolved, not forced into either category.

# R3 — Reproduce the first divergent operator in isolation

Once R1 identifies the first component, make the smallest reproducible tensor test from the saved input to that component.

Examples, chosen only after localization:
- Linear/GEMM: same target row alone versus concatenated with the frozen filler rows; compare target output in BF16 and FP32.
- Attention: same target Q/K/V and mask with B1 versus packed batch, using the exact backend/config; verify that target K/V sequence and mask are identical.
- Norm/residual: compare with identical target tensor and dtype.

Do not swap to a different kernel implementation and call that the cause. First reproduce the current backend's behavior. Then, as diagnosis only, compare a more deterministic/high-precision implementation to see whether the discrepancy disappears.

Publish the isolated input hashes, backend flags and exact operator result. This is the point at which "BF16 batch-shape numerical effect" can become an evidence-backed conclusion rather than an inference.

# R4 — Resolution decision tree

## Branch A: implementation bug found

Examples: wrong row/KV selection, padding/position bug, state identity mismatch, duplicated state update, mask contamination, unsupported config path.

1. Fix only the demonstrated bug.
2. Add a CPU/unit regression and a real-GPU counterexample regression.
3. Re-run the frozen R0 packets.
4. Re-run the original prospective BF16 N32/B8/T8 Gate A unchanged.
5. If it passes, expand to B=1/4/8/16/32, row permutations and N64/T32 before reopening predictor collection.

No threshold/tolerance relaxation is permitted as part of Branch A.

## Branch B: expected BF16 batch-shape numerics, no semantic bookkeeping bug found

Do **not** pretend the old exact-depth invariance requirement passed. Instead quantify whether exact live depth is inherently batch-shape dependent under the chosen low-precision execution.

Measure prospectively on fresh validation requests:
- depth-flip rate versus B;
- next-token flip rate versus B;
- exit-margin distribution and fraction within epsilon of threshold;
- conditional flip probability as a function of reference exit margin;
- RIC/work variation across repeated legal batch compositions;
- whether flips are concentrated in threshold-near tokens or also occur far from threshold.

Then choose one of the following research contracts explicitly:

**B1. Stable-region contract.** If flips are rare and confined to a predeclared numerical margin around the exit threshold, define a prospective semantic validation rule that treats these cases as numerically ambiguous, while still requiring exact row/KV bookkeeping and bounded hidden/logit/KV drift. This is a new protocol and must be justified before its confirmation run. It must not retroactively relabel old failures.

**B2. Batch-dependent-depth contract.** If live depths materially depend on batch composition, stop treating B1 depth as a unique ground-truth label. Keep B1/frozen traces only for controlled packing studies. For practical scheduling, evaluate the predictor/scheduler under actual live execution and report the actual work it induces. Any Oracle must be renamed/redefined as a frozen-work oracle or an explicitly computed live counterfactual; do not use the old B1-depth oracle as if it were an exact live oracle.

**B3. Numerically robust exit policy as a separate method.** Only if scientifically justified, study a margin/hysteresis/robust gate policy that reduces threshold flips. This changes the exit policy and potentially the work/model behavior; it is a separate method with its own baseline and cannot be introduced as a hidden implementation fix. Do not start B3 without a research decision after B1/B2 measurements.

## Branch C: unresolved mixed cause

Do not continue to predictor training. Publish the bounded negative diagnosis and the next smallest discriminating experiment. Do not broaden into an open-ended GPU sweep.

# R5 — Fresh Gate A after a justified resolution

Only after Branch A fix or an explicitly approved prospective Branch-B semantic contract:

1. write the new release rule before running the confirmation set;
2. use fresh N32/B8/T8 mixed-length requests, not the R0 diagnostic rows;
3. keep exact checkpoint, exit policy, KV policy and BF16 precision unless the approved branch explicitly defines a new method;
4. preserve exact depth/next-token mismatches and threshold-near receipts;
5. repeat on the second authorized same-model GPU if available;
6. publish PASS/FAIL/BLOCKED independently per model.

The predictor phase reopens only for a model lane whose new Gate A passes under its preregistered contract.

# Consequences for the predictor/oracle study

Do not lose the original research goal. The resolution determines what the predictor target actually is.

If Branch A holds and exact live parity is restored:
- continue the planned `offline token prior + previous hidden/Prelude contextual residual` predictor;
- compare FIFO/Predictor/Oracle on a corrected clean runtime.

If Branch B2 holds:
- distinguish **depth propensity prediction** from exact-depth prediction;
- collect training labels under a canonical declared batching policy and measure transfer to other B/compositions;
- evaluate real scheduling by actual live RIC/work/wall time, not only replayed B1 labels;
- keep frozen-depth Oracle as a structural packing upper bound, clearly labeled as such.

Either way, the next paper-level insight may become stronger: not only that adaptive depth varies across tokens, but also whether low-precision live depth is stable or sensitive to execution shape. Do not make that claim until R1-R4 support it.

# Required artifacts

Create a new result root `phase04/results/<timestamp>_<source_sha>_bf16_resolution/` and preserve:
- `COUNTEREXAMPLE_MANIFEST.json`;
- `first_divergence.csv` plus bounded per-layer receipts;
- `batch_shape_matrix.csv`;
- isolated-operator inputs/hashes and results;
- `CAUSE_REVIEW.md` with Branch A/B/C decision and evidence;
- any fix commit SHA and regression tests;
- prospective replacement Gate-A rule if Branch B is chosen;
- fresh confirmation `runtime_validation.json` and `RESULT_REVIEW.md`.

No predictor data, speed matrix, or new Oracle claim belongs in this root until R5 passes.
