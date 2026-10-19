# E2 — Stage-specific H100 latency and architecture study

**Method** (`scripts/e2_stage_microbench.py`). Each cell uses the unchanged phase02 adapters/KV backends (BF16; Ouro HF-eager with explicit mask and last-exited depth-indexed KV; Huginn SDPA-math head96 wrapper with shared physical-layer KV).
- *Valid state.* `SEED_ROWS=4` rows of real Alpaca text (instruction + output, concatenated and cut to exactly L tokens) are really prefilled. Their KV and masks are then replicated to B rows. Huginn history is allocated at L+slack, so the runtime's amortized capacity doubling never triggers inside the benchmark.
- *One decode step is staged and then rolled back,* so every trial sees cache length L:
  - Prelude (Ouro: embedding only);
  - one full recurrent Core invocation at r=1, including its KV writes;
  - exit gate;
  - exit-time KV fill;
  - compaction B→B/2;
  - Coda Transformer blocks (Huginn only) and LM head, timed separately.
- *Timing.* CUDA events around synchronized calls; 2 warmups + 10 measured trials per cell.
  - Core GPU time, host dispatch and synchronized wall are taken with KV event instrumentation off.
  - KV-update GPU time comes from a separate profiled call on the same state.
- Grid: B ∈ {1,2,4,8,16,32,64,128} × L ∈ {32,64,128,256,512,1024,2048} × {Ouro-1.4B, Ouro-2.6B, Huginn-3.5B}. L=32–256 were added beyond the spec's 128–2048 to cover E3's real contexts (65–96).
- **Correctness gate** (predeclared): row 0's Core output must be finite and within relative L2 ≤ 0.03 of the B=1 output at the same L. The model, KV and mask contract is asserted per cell.
- **Knee B\*** = largest B whose median Core time stays ≤ 1.10 × Core(B=1). The conservative and optimistic bounds use the q25/q75 of the two cells.
- **Bottleneck labels** (`scripts/e2_analyze.py`, stated in its docstring) are assigned only from measured signals: Nsight Systems GPU-busy fraction, measured KV-update share of Core, flatness vs B=1, and achieved bandwidth/FLOP fractions of the H100 peaks (3.35 TB/s, 989 TFLOP/s).
- **Profiler subset** (predeclared): every model at (B=1, L=512) and (B=32, L=512) under Nsight Systems. Nsight Compute is **not installed** on this host, so DRAM bytes, achieved FLOPs and occupancy counters are unavailable.

**Run.** `results/20261019_e2_d6b8674/`. The first attempt `results/20261019_e2_83b72bb/` is kept and **superseded**: after a true OOM, a stale reference to the previous cell's cache caused cascading spurious OOMs (see its SUPERSEDED.md).

**Outputs.**
- `e2_stage_latency.csv`: per cell, medians, IQR, shares, roofline estimates and labels.
- `*/e2_stage_latency_raw.csv`: every trial and every OOM.
- `*/e2_correctness_gate.json`.
- `e2_knee.csv`, `e2_params_by_stage.json`.
- `nsys/e2_nsys_summary.csv`; `.nsys-rep` files are kept locally, SHA-256 in `nsys/nsys_rep_sha256.txt`.
- Plots: `e2_core_latency_vs_B.png`, `e2_stage_breakdown_L512.png`.

**Results.** 168 cells: 149 OK, 19 OOM, **0 correctness-gate failures**. See `results/20261019/RESULTS.md`.
