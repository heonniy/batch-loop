# batch-loop

PoC for **offline no-refill batching in depth-adaptive looped LMs** on a single H100.

## Hypothesis

In no-refill decoding, a micro-batch starts full but shrinks inside one token-generation interval as tokens exit the recurrent core at different depths. The resulting gray/idle slots form a **recurrent-depth valley**.

The PoC asks a narrow question:

> If an offline scheduler forms each micro-batch from tokens with compatible recurrent demand, how much of the no-refill valley disappears **without any intra-loop refill**?

The execution semantics stay pure no-refill: a token that exits is never replaced until the next token boundary. We only change micro-batch formation before entering the recurrent core.

## First go/no-go experiment

1. Collect per-request, per-generated-token exit-depth traces.
2. Reconstruct no-refill micro-batches at each token interval.
3. Compare:
   - `fifo`: request order.
   - `random`: shuffled requests.
   - `oracle`: sort by the current token's true exit depth (upper bound).
   - `prev`: sort by previous-token depth.
   - `recent`: sort by recent mean depth.
   - `seqmean`: sort by cumulative request-level mean depth.
4. Sweep batch sizes: `4,8,16,32,64,128,256` (large values are trace-simulation points; runtime feasibility depends on KV/context length).
5. Repeat across looped architectures.

Primary metric:

```text
slot_utilization = sum(actual token-loop work) / sum(B * max_depth_in_microbatch)
idle_slot_ratio  = 1 - slot_utilization
```

For one micro-batch with depths `[d1, ..., dB]`, the total gray area is

```text
idle_slots = B * max(di) - sum(di)
```

This directly measures the valley targeted by the idea.

## Models

The default H100 sweep uses:

- `ByteDance/Ouro-1.4B` — 4 recurrent steps, learned exit gate.
- `ByteDance/Ouro-2.6B` — same LoopLM family at a larger size.
- `tomg-group-umd/huginn-0125` — 3.5B recurrent-depth model; use latent-state convergence for adaptive depth.

Important trace caveat:

- The public Ouro Transformers implementation computes all UT steps and applies the learned gate as a **post-hoc readout**. `collect-ouro` records that gate's chosen depth. This is sufficient for the initial batch-shape/oracle-headroom PoC, but it is **not** an end-to-end real-early-exit runtime.
- Huginn exposes adaptive generation and per-token compute-step traces. We collect it with batch size 1 per request, as its implementation recommends for faithful adaptive-compute traces.

The purpose of Phase 1 is therefore to validate the scheduling opportunity before implementing a custom no-refill runtime.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended machine: **1x H100 80GB**.

## Quick smoke test

```bash
bash run_h100_poc.sh smoke
```

This collects short traces from Ouro-1.4B and Huginn and runs the scheduler simulation.

## Main PoC

```bash
bash run_h100_poc.sh poc
```

To include the larger Ouro model:

```bash
RUN_OURO_26B=1 bash run_h100_poc.sh poc
```

To additionally run a ShareGPT-like workload:

```bash
RUN_SHAREGPT=1 bash run_h100_poc.sh poc
```

Outputs are written under `results/`:

```text
results/
  traces/*.jsonl
  analysis/*/summary.csv
  analysis/*/predictability.csv
  analysis/*/slot_utilization.png
  analysis/*/idle_slot_ratio.png
  analysis/*/valley_example_b*.png
```

## Manual commands

### Ouro

```bash
python poc.py collect-ouro \
  --model ByteDance/Ouro-1.4B \
  --dataset alpaca \
  --num-requests 128 \
  --max-new-tokens 32 \
  --exit-threshold 0.2 \
  --output results/traces/ouro14_alpaca_q02.jsonl
```

### Huginn

```bash
python poc.py collect-huginn \
  --model tomg-group-umd/huginn-0125 \
  --dataset alpaca \
  --num-requests 64 \
  --max-new-tokens 32 \
  --max-steps 16 \
  --criterion latent-diff \
  --exit-threshold 0.12 \
  --min-steps 1 \
  --output results/traces/huginn_alpaca_d012.jsonl
```

### Analyze valley + grouping headroom

```bash
python poc.py analyze \
  --traces results/traces/*.jsonl \
  --batch-sizes 4,8,16,32,64,128,256 \
  --output-dir results/analysis/poc
```

## What counts as a positive PoC?

The strongest early signal is not predictor accuracy; it is **oracle headroom**.

Continue the topic if, over a meaningful batch-size range and more than one architecture/workload:

- FIFO/random no-refill shows substantial idle-slot valleys, and
- oracle depth grouping recovers a clear fraction of those idle slots, and
- a cheap causal predictor (`prev`, `recent`, or `seqmean`) captures a non-trivial fraction of the oracle gain.

If oracle grouping barely improves slot utilization, there is little reason to build a predictor or a custom runtime.

## Next step after a positive PoC

Only after the trace result is positive:

1. implement real no-refill recurrent execution,
2. measure H100 recurrent-step latency vs active batch size/context length,
3. replace slot utilization with wall-clock scheduling cost,
4. compare against CDB refill as a reference baseline.

The intended paper direction remains **offline no-refill micro-batch formation**, not refill/rebatching inside the recurrent interval.
