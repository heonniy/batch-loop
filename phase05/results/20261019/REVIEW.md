# Phase 05 review — E1/E2/E3

## Five-axis review

| Axis | Status | Evidence |
| --- | --- | --- |
| Measurement | VALID with stated limits | E2: 2 warmups + 10 trials per cell; E3: 2 warmups + 5 reps, randomized policy order, all runs retained (440/440 OK). One GPU per benchmark, UUIDs verified against GPU_ASSIGNMENT_20261019.json. |
| Implementation | Correctness gates PASS | E2: 0/149 gate failures (row-0 Core output vs B=1, rel-L2 ≤ 0.03). E3: physical RIC and work asserted equal to the frozen plan in every run. E3 grouping invariants are unit-tested. |
| Simulator / latency model | **REJECTED for absolute claims** | Predicted FIFO / measured FIFO = 1.13–1.27 in all 20 physical cells (rule ±10%). Relative reductions only exploratory. |
| Comparison | Equal frozen work | Same requests, depths, contexts, KV semantics and Rmax across policies. N=B controls and a random equal-work control included. |
| Hypothesis | RQ1 yes (Ouro ≈1:1, Huginn 0.71–1.05); RQ2 launch→KV-path regimes, no weight/compute-bound cell; RQ3 no material gain from latency-aware grouping | See RESULTS.md. |

## Blockers and deviations (kept, not hidden)

- **Cascading spurious OOM in the first E2 attempt** (`results/20261019_e2_83b72bb`, SUPERSEDED.md). A stale cache reference survived a true OOM. Fixed in d6b8674 and re-run in full; the old run is kept.
- **E3 Huginn launch failure** (`results/20261019_e3_7f903d8/failed_attempts`). The phase02 Huginn trace schema differs (KeyError 'family'). Fixed in f6c8c9f and relaunched. The two Ouro lanes started on 7f903d8; that code path did not change, and each lane's manifest records its source SHA.
- **Nsight Compute is not installed**, so there are no DRAM-byte, achieved-FLOP or occupancy counters. Roofline numbers are analytic estimates from safetensors parameter counts and ideal KV bytes. Nsight Systems kernel timelines were captured for the predeclared subset (.nsys-rep kept locally, SHA-256 committed).
- **Ouro-2.6B** has no committed trace, so it is absent from E3 (it is covered by E2 and by E1's phase01 rows).
- **E2 context grid** was extended to L=32–256 to cover E3's real contexts (65–96).
- **Phase-04 BF16 Gate A** remains FAIL. Nothing here is live-exit parity, quality, or live-exit speed.
- **Context:** the phase02 runtime pads every row to the cohort-wide prompt width, so mixed-length cohorts do not occur. Results do not cover a ragged/paged-KV engine.

## Reproduction

Commits on `research/regime-architecture-oracle-20261019`:

| Commit | Content |
| --- | --- |
| 83b72bb | E2 code (superseded run) |
| d6b8674 | E2 fix |
| 7f903d8 | E2 results, Nsight subset, E3 code |
| f6c8c9f | Huginn trace schema fix |
| 97e2e99 | E1 |
| 869a811 | E3 memoization / E2 labels |

Results and docs are in the final commit. Exact commands are in [phase05/README.md](../../README.md). Every run directory carries its RUN_MANIFEST.json (argv, GPU UUID, model revision, prompt/trace hashes, source SHA, torch/CUDA).
