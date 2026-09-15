# batch-loop

Research PoC for **offline no-refill batching in depth-adaptive looped LMs** on a single H100.

## Current priority — validate the scheduling opportunity fairly

The first H100 result commit (`4ec78c1`) collected real exit-depth traces and then simulated no-refill micro-batch regrouping. Ouro-1.4B showed substantial oracle slot-utilization headroom, while Huginn at threshold 0.12 showed little room and simple causal predictors recovered little of the Ouro oracle gap.

A post-result review identified a fairness defect in the R0 simulator: strategy-specific ordering happened before incomplete-remainder truncation. Therefore the existing headroom is an **exploratory signal**, not yet a final comparison and not a physical speedup result.

The next bounded block is R1 only: fair-subset analysis, a larger Ouro request pool, and Ouro/Huginn threshold sweeps. The already-committed R1 runner/source changes are not themselves new experimental results.

Read before new work:

- [`AGENTS.md`](AGENTS.md) — coding/model-agent execution and commit protocol.
- [`phase01/README.md`](phase01/README.md) — experiment state and research framing.
- [`phase01/CURRENT_RESEARCH_PRIORITY.md`](phase01/CURRENT_RESEARCH_PRIORITY.md) — bounded next block.
- [`phase01/EXPERIMENT_REVIEW_PROTOCOL.md`](phase01/EXPERIMENT_REVIEW_PROTOCOL.md) — evidence/publication contract.
- [`phase01/results/20260915_poc/RESULT_REVIEW.md`](phase01/results/20260915_poc/RESULT_REVIEW.md) — bound review of the first H100 PoC.

The repository follows the same plan/source -> bounded run -> result publication -> review checkpoint discipline used in `sub-moe`. The model/coding agent that executes a bounded block is expected to commit the compact result handoff and manifest rather than leaving only local GPU artifacts.

## Hypothesis

In no-refill decoding, a micro-batch starts full but shrinks inside one token-generation interval as tokens exit the recurrent core at different depths. The resulting gray/idle slots form a **recurrent-depth valley**.

The research question is:

> If an offline scheduler forms each micro-batch from tokens with compatible recurrent demand, how much of the no-refill valley disappears **without any intra-loop refill**?

The execution semantics stay pure no-refill: a token that exits is never replaced until the next token boundary. We only change micro-batch formation before entering the recurrent core.

## Phase-1 measurements

Trace/simulator metric:

```text
slot_utilization = sum(actual token-loop work) / sum(B * max_depth_in_microbatch)
idle_slot_ratio  = 1 - slot_utilization
```

For one micro-batch with depths `[d1, ..., dB]`:

```text
idle_slots = B * max(di) - sum(di)
```

`slot_utilization` is a logical batching metric, **not GPU hardware utilization** and not throughput.

## Models

- `ByteDance/Ouro-1.4B` — 4 recurrent steps, learned exit gate.
- `ByteDance/Ouro-2.6B` — larger model in the same LoopLM family.
- `tomg-group-umd/huginn-0125` — 3.5B recurrent-depth model with adaptive convergence criterion.

Important caveat: the public Ouro Transformers implementation used in R0 computes all UT steps and exposes its learned gate as a post-hoc readout. R0 therefore measures scheduling opportunity from realistic model traces, not real early-exit wall-clock acceleration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended machine: **1x H100 80GB**.

## Existing PoC runner

```bash
bash run_h100_poc.sh smoke
bash run_h100_poc.sh poc
```

The repository also contains the R1 fair-subset / scale / threshold runner committed after R0. Its outputs must be published as a separate result commit before being treated as evidence.

## Result / claim boundary

Current R0 evidence supports only:

- real model exit-depth traces were collected,
- Ouro displayed an exploratory no-refill grouping opportunity in trace simulation,
- Huginn threshold 0.12 displayed little scheduling headroom,
- simple history-based predictors were weak.

It does **not** yet support a throughput, BCT, or end-to-end latency improvement claim.

After a fair positive R1 review, the next phase is to implement physical no-refill survivor-only execution and measure recurrent wall-clock, generated tokens/s, and offline BCT. The intended paper direction remains **offline no-refill micro-batch formation**, not refill/rebatching inside the recurrent interval.
