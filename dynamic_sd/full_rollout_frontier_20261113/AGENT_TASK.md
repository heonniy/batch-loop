# Coding agent: implement and run *after* protocol gate

Read [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) and [REPRODUCTION_PROTOCOL.md](REPRODUCTION_PROTOCOL.md), then inspect existing `dynamic_sd/{dsd_run.py,dsd_batch.py,followup_20261112,decision_characterization_20261113}`. Do not overwrite prior results or reinterpret one-round Oracle as full-rollout throughput.

## Implementation steps (additive, parity-first)

1. Create `run_frontier.py` (or a clearly documented wrapper around pinned WaveFront `exp2/exp3`) that generates a dry-run matrix and produces **actual free-running generation**; no one-round snapshots in the primary frontier. Set action list `AR`, `DtV:d{fixed}:K{1,2,4,8,16}`; WFD(d) independent with no separate K. Record paper prompt config IDs and dataset splits before execution.
2. Implement `validate_frontier_receipts.py` to **reject** any mixed models, different native target or seed, different prompt hash, incomplete K grid, dynamic Dory mislabeled as native, inexact target in strict parity lane, missing stage-time attribution masquerading as observed, mismatch of BF16/FP32 kernel backend.
3. Parse actual MODEL/MODE/SCHEDULER/KV cache flags. On Ouro use fixed R4 and native learned hazard `tau`; on Huginn fixed R32 and **calibrated actual** Huginn native exit criterion (do not silently equate WaveFront relative-L2 to checkpoint cosine). On Dory verify supported *fixed elastic* R1/2/4 and `per_loop` KV reference; no assumed official per-token gate.
4. Start smoke on each model family and **every benchmark class** with n≤32, K∈{2,4,8}; strict FP32 FULL KV AR-vs-DtV/WFD equality on model-compatible targets first. Explicitly block unsupported Dory actions; treat failures as data.
5. Run fixed/full-rollout frontier B1 (AR, each supported K, WFD) for paper-style GSM8K-CoT 3shot, MATH-500 4shot; also WaveFront-style GSM8K 8shot and Spec-Bench, and LoopSpec-style BBH/HumanEval+/MBPP+ as viable. Per model × dataset, n100 pilot then n200 confirmation on separate prompts (complete full Spec-Bench if resources permit). Official LoopSpec R4 Ouro and Raven R32 are separate *reproductions*; Huginn is not Raven.
6. Run matching native live Ouro and Huginn on identical prompts + exact per-target oracle/quality, not forcibly on Dory. Include τ=.4/.1 and ε=.1/.2 *only if those policies are calibrated, nondegenerate and have working correctness*. Compare speedup versus own AR at same depth rule, not fixed AR.
7. Generate fixed/live × K plots, paired bootstrap CIs, per-prompt summaries, selected K and second-best gap; then the **post-hoc** exit/acceptance/stage-cost conditional comparison. Add separated action predictor lanes only after real frontier exists. Keep all negative results and prompt/seed/backend metadata.
8. Run code/tests, a CPU-only `--dry-run` protocol report and manifest validator, then commit first code/test, results later only after actual runs; no claims of GPU completion without source traces and checksum.

## Example upstream references (not automatically executed)

- WaveFront `summerbro-hhj/wavefront-decoding/experiments/exp2_spec_bench/run.sh`: `SCHEDULER=dtv DRAFT_LENGTH=4` for fixed-K; `SCHEDULER=wfd` for separate wavefront; `MODEL=ouro-1.4b T_TOTAL=4 T_DRAFT=1` or `MODEL=huginn-3.5b T_TOTAL=32 T_DRAFT=4`; matched AR must be separately measured.
- LoopSpec `kaist-flexml-lab/loopspec/experiments/110-speed-math.yaml`, `112-speed-coding.yaml`, `113-speed-general.yaml`, `tasks.yaml`: *single-batch* fixed-depth reproduction first; not already adapted for native live.
- Dory model `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`; model paper `P-2611-03998`: fixed R1/2/4 and self-spec baseline d1 K4 with per-loop KV and activation reuse. Do not substitute an arbitrary original model's recurrent core for Dory nGPT.

## PASS / STOP

- FAIL parity → stop speed claim, output diagnostic run; strict exactness and measured engine fidelity are required.
- FAIL live gate correctness → record exact reason and skip, don't forge D trace.
- FAIL near-tied K optimum → report a **plateau**, not a crossover.
- FAIL live specificity/held-out net gain → conclude no verified dynamic-only policy opportunity; do not train an elaborate controller.
