# Experiment review protocol — looped-LM batching

Apply this contract when publishing or reviewing a batch-loop experiment. Keep intended plans, completed GPU execution, saved-data analysis, simulation, and physical speed measurements separate.

## Execution-kind labels

- `exit-trace-collection`: real model execution used to record per-token exit/recurrent depth. It is not itself a scheduling speed experiment.
- `trace-sim`: saved exit depths are regrouped/replayed to estimate no-refill batch shape, slot utilization, or oracle headroom. It cannot establish wall-clock throughput/BCT.
- `saved-data-analysis`: reaggregation/plotting of already committed traces/results. No new model inference.
- `real-no-refill-runtime`: an implementation actually stops exited tokens and executes survivors only, with no intra-loop refill; physical H100 timing is valid only under this label and a bound source/run receipt.
- `refill-reference-runtime`: physical refill implementation used only as a baseline/reference; keep its scheduling semantics explicit.

## Review five axes separately

| Axis | Checks | Example statuses |
| --- | --- | --- |
| Measurement | exact request/token denominator, EOS/cap, depth rule, missing late-token positions, timing source, aggregation | VALID / INSUFFICIENT_DATA / DATA_DEFECT |
| Implementation | exit rule matches model, no-refill semantics, grouping boundary, same weights/KV semantics, runtime really skips work | CHECKED_IN_SCOPE / UNVERIFIED / IMPLEMENTATION_DEFECT |
| Simulator | identical subset/work across strategies, deterministic grouping, capacity/idle arithmetic, correspondence to runtime semantics | CHECKED_IN_SCOPE / TRACE_SIM_ONLY / SIMULATOR_DEFECT |
| Comparison | same prompts/tokens/model revision/threshold/B/context policy; same subset; appropriate refill/fixed-depth references | COMPARABLE / COMPARISON_INCOMPLETE / UNMATCHED |
| Hypothesis | effect size across B/model/threshold/workload, dispersion, counterexamples, practical recoverability | NOT_TESTED / EXPLORATORY_SIGNAL / INCONCLUSIVE / NEGATIVE_IN_SCOPE |

A valid trace simulation is not a valid speed result. A valid implementation is not automatically a scientific GO.

## Required audits

1. Bind model/checkpoint revision, tokenizer, precision, dataset revision/sampling seed, generation cap, prompt cap, H100 identity, PyTorch/Transformers version, and source commit.
2. For Ouro, state whether the public implementation computed all UT steps and the gate was interpreted post hoc. Never describe that as physical early exit.
3. For Huginn, record criterion, threshold, min/max steps, and whether collection followed the implementation's adaptive-compute batching constraints.
4. For grouping comparisons, freeze the eligible token/request subset **before** strategy-specific ordering. Verify identical item IDs and `total_token_loop_work` across strategies.
5. Report how incomplete micro-batches and request termination are handled. A missing B point is not a zero.
6. Separate `slot_utilization` from hardware utilization. The former is a logical recurrent-slot metric.
7. Physical speed claims require synchronized wall-clock timing, warmup, repeated measurements, and actual survivor-only execution.

## Scientific interpretation

For trace work, the causal chain is:

`observed exit-depth heterogeneity -> intra-batch depth divergence -> idle recurrent slots -> oracle regrouping headroom`.

Only the first four links are tested by R0/R1. Claims about GPU time, throughput, BCT, or refill-equivalent speed require R2 physical execution.

A weak predictor does not invalidate oracle headroom; it means the practical scheduler problem remains unsolved. A model/threshold with no valley is negative evidence for that configuration and should remain in the paper/repo.

## Required result handoff

Every completed bounded block must publish or link a `review_manifest.json` containing at least:

```text
result_commit / source_commit / plan_or_priority
execution_kind / hardware / exact_command_or_receipt
model_revisions / dataset_revision / thresholds / batch_sizes
measurement_status + evidence + known_missing_fields
implementation_status + evidence
simulator_status + fairness/subset checks
comparison_status + baselines + denominator checks
hypothesis_status + effect sizes + counterevidence
artifacts + hashes_or_paths
next_action + finite scope
```

UNKNOWN fields stay UNKNOWN. Do not copy intended plan values into a result manifest if the run did not verify them.

## Commit protocol

The agent/model executing a bounded experiment is responsible for publishing a result commit after completion. Use separate commits for plan/source changes and actual results.

Preferred subjects:

- `phase01: plan ...`
- `phase01: implement ...`
- `phase01: publish ...`
- `phase01: review ...`

A result publication must state whether it is trace collection, trace simulation, saved-data analysis, or physical runtime. Preserve negative results and implementation failures rather than retrying until a favorable number appears.
