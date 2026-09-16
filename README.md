# batch-loop

Research PoC for **offline no-refill batching in depth-adaptive looped LMs** on a single H100.

## Current experiment

The current full experiment asks whether compatible micro-batch formation can reduce the recurrent-depth valley without any intra-loop refill, and whether that opportunity is large enough to matter in real H100 execution.

The fixed matrix is:

- Models: `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`, `tomg-group-umd/huginn-0125`.
- Batch sizes: `4, 8, 16, 32, 64`.
- Ouro thresholds: `0.2, 0.4, 0.5, 0.7`.
- Huginn latent-diff thresholds: `0.5, 0.35, 0.26, 0.12`.
- Workloads: **Alpaca + ShareGPT**.

Alpaca is the short/tightly concentrated workload; ShareGPT is added as the longer/heavier-tailed workload to test whether the headroom survives larger context and KV traffic. Results are stored separately by dataset and then combined into compact review tables.

Default bounded caps are Alpaca `prompt<=512, new_tokens=24` and ShareGPT `prompt<=2048, new_tokens=64`. These are explicit PoC resource bounds and are recorded in each run; they are not presented as the full dataset length distribution.

## What the full sweep measures

1. Real per-token exit-depth traces.
2. Fair FIFO/random/history/oracle no-refill slot utilization and Oracle headroom.
3. **Actual native whole-model H100 E2E/BCT/throughput** for every model/threshold/B/workload.
4. Real H100 fixed-depth latency surfaces at active B=1..64 and contexts up to 2048.
5. Hardware-calibrated no-refill FIFO/oracle/history BCT and throughput replay.

Run everything with:

```bash
bash run_h100_full_sweep.sh
```

See [`phase01/FULL_E2E_EXPERIMENT_PLAN.md`](phase01/FULL_E2E_EXPERIMENT_PLAN.md) for the exact matrix and evidence boundaries.

## Research hypothesis

In pure no-refill decoding, a micro-batch starts full but shrinks inside one token-generation interval as tokens exit the recurrent core at different depths. The gray/idle slots form a **recurrent-depth valley**.

For one micro-batch with depths `[d1, ..., dB]`:

```text
idle_slots = B * max(di) - sum(di)
slot_utilization = sum(di) / (B * max(di))
```

The scheduler changes only micro-batch formation before entering the recurrent core; exited tokens are never replaced inside that recurrent interval.

## Evidence boundary

The first H100 PoC already showed exploratory Ouro-1.4B Oracle headroom, but it was trace-based and the original simulator had a remainder fairness defect that is now fixed by `analyze_fair.py`.

Current evidence labels are intentionally separate:

- `slot_utilization`: trace geometry only.
- `native_e2e_*`: actual whole-model wall-clock model generation.
- `hardware_replay_*`: measured H100 latency surface + observed exit trace, not an integrated compaction engine.

The public Ouro HF path computes all recurrent steps and reads the gate post-hoc, so threshold-dependent native Ouro timing must not be called physical early-exit speedup. Huginn adaptive batched timing is trusted only when parity checks pass.

## Repository protocol

Read [`AGENTS.md`](AGENTS.md) before model/coding-agent work. The repository follows the same plan/source -> bounded run -> result publication -> review discipline as `sub-moe`. The agent that executes a run must commit compact results, `RESULT_REVIEW.md`, and `review_manifest.json`; negative/OOM results are preserved rather than filtered away.
