# batch-loop

Research PoC for **offline no-refill batching in depth-adaptive looped LMs** on a single H100.

## Current priority — full three-model headroom + native E2E study

The current committed experiment runs one combined block over:

- `ByteDance/Ouro-1.4B`
- `ByteDance/Ouro-2.6B`
- `tomg-group-umd/huginn-0125` (Huginn-3.5B)
- B = `4, 8, 16, 32, 64`
- Ouro thresholds = `0.2, 0.4, 0.5, 0.7`
- Huginn latent-diff thresholds = `0.5, 0.35, 0.26, 0.12`

The same H100 run measures both the no-refill scheduling opportunity and the actual cost of the complete model:

1. fair FIFO/random/history/oracle recurrent-depth headroom;
2. per-threshold depth distributions;
3. **actual native whole-model E2E BCT, generated tokens/s and requests/s**;
4. H100 recurrent-step latency surfaces;
5. hardware-calibrated no-refill BCT/throughput replay.

Refill is out of scope.

Read before new work:

- [`AGENTS.md`](AGENTS.md) — coding/model-agent execution and commit protocol.
- [`phase01/FULL_E2E_EXPERIMENT_PLAN.md`](phase01/FULL_E2E_EXPERIMENT_PLAN.md) — exact matrix and measurement definitions.
- [`phase01/CURRENT_RESEARCH_PRIORITY.md`](phase01/CURRENT_RESEARCH_PRIORITY.md) — current execution scope.
- [`phase01/EXPERIMENT_REVIEW_PROTOCOL.md`](phase01/EXPERIMENT_REVIEW_PROTOCOL.md) — evidence/publication contract.
- [`phase01/results/20260915_poc/RESULT_REVIEW.md`](phase01/results/20260915_poc/RESULT_REVIEW.md) — first H100 PoC review.

The repository follows the same plan/source -> bounded run -> result publication -> review discipline used in `sub-moe`. The model/coding agent that executes the run must commit compact results and a review manifest instead of leaving only local GPU artifacts.

## Hypothesis

In pure no-refill decoding, a micro-batch starts full but shrinks inside one token-generation interval as tokens exit the recurrent core at different depths. The resulting gray/idle slots form a **recurrent-depth valley**.

The research question is:

> If an offline scheduler forms each micro-batch from tokens with compatible recurrent demand, how much of the no-refill valley disappears without any intra-loop refill, and is that opportunity large enough to matter in actual H100 execution?

Scheduling changes only micro-batch formation at token boundaries. A token that exits is never replaced until the next token boundary.

## Existing evidence

The first result commit (`4ec78c1`) collected real exit-depth traces and simulated no-refill grouping. Ouro-1.4B showed substantial exploratory oracle slot-utilization headroom, Huginn at threshold 0.12 showed little room, and simple history-based predictors recovered only a small fraction of Ouro's oracle gap.

A fairness defect was then identified: the first simulator truncated incomplete remainders after strategy-specific ordering. `analyze_fair.py` fixes this by freezing the exact same subset before reordering. Therefore the original headroom is exploratory until rerun with the fair analyzer.

## Measurements and claim boundary

Trace metric:

```text
slot_utilization = sum(actual token-loop work) / sum(B * max_depth_in_microbatch)
idle_slot_ratio  = 1 - slot_utilization
```

`slot_utilization` is a logical batch-shape metric, not GPU hardware utilization.

The combined run also emits:

```text
native_e2e_all.csv
```

This is **actual H100 whole-model generation wall-clock** using the official/native generation path. It reports BCT, generated tokens/s, requests/s, p50/p95 batch time, tokenization/H2D and HBM peak.

Important model caveat:

- Ouro's pinned public HF implementation still computes every recurrent step and exposes the exit gate post-hoc. Its native E2E numbers are therefore real whole-model baselines, but threshold differences are not physical early-exit/no-refill speedup.
- Huginn exposes adaptive recurrent generation; batched adaptive timing is trusted only when deterministic batch-vs-B1 parity passes.
- `runtime_replay_all.csv` combines real H100 timing with real exit traces to estimate no-refill scheduling cost. It is hardware-calibrated replay, not an integrated survivor-compaction engine.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended machine: **1x H100 80GB**.

## Full experiment

```bash
bash run_h100_full_sweep.sh
```

The runner records the source SHA, environment, model revisions and exact launch command under a unique `results/full_sweep_*` directory.

Key outputs:

```text
results/full_sweep_*/analysis/fair/oracle_vs_fifo.csv
results/full_sweep_*/analysis/depth_distribution.csv
results/full_sweep_*/e2e/native_e2e_all.csv
results/full_sweep_*/calibration/*_fit.csv
results/full_sweep_*/analysis/runtime_replay_all.csv
```

After completion, the executing agent must publish one result-bound commit with `RESULT_REVIEW.md`, `review_manifest.json`, compact CSVs/plots and explicit failures/limitations.
