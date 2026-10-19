# E3 — Architecture-aware (latency-aware) Oracle grouping

**All Oracles read future frozen depths and are not deployable.** Physical runs force those depths (`exit_source="trace"`). They are scheduling-mechanism tests (`PHYSICAL_FORCED_DEPTH`), never live-exit performance.

**Workload.**
- Frozen phase02 B=1 traces (`phase02/oracle_upper_bound_results_20260916/trace_inputs/*.jsonl.gz`, hashes in the manifests). Ouro-1.4B Alpaca/ShareGPT at q ∈ {0.2,0.4,0.5,0.7}; Huginn Alpaca at 0.12/0.26/0.35/0.5.
- Prompts: phase02's seed-1 Alpaca/ShareGPT prompts (first N), padded to the cohort-wide width W ≤ 64. In this runtime every row of a step attends exactly W+t+1 keys, so mixed context lengths do not arise.
- **Ouro-2.6B has no committed trace**, so it is absent from E3 (it is covered by E2).

**Policies** (same requests exactly once per step, full cohorts of B, no refill):
- `fifo`;
- `depth_oracle`: sort by depth, then chunk;
- `latency_oracle`: minimize Σ_m Σ_{r≤max d} T_core(w_{m,r}, L_t) using E2 medians, interpolated linearly in w and L **only inside measured support** (otherwise `NO_MODEL_SUPPORT`, never extrapolated). Solved exactly over depth-count vectors for 2 groups, else by first-improvement swap local search from the depth-sorted grouping, which certifies predicted cost ≤ depth Oracle;
- `random`: equal-work control.

**Replay** (CPU): N ∈ {32,64,128,256}, B ∈ {4,8,16,32,64} plus the B=N control, T ∈ {32,64}, all 12 traces; 2,016 rows in `replay/e3_replay.csv`.

**Physical** (predeclared cells (N,B,T) = (256,8,32), (256,32,32), (64,8,32), (64,32,32), (32,32,32)=N=B control):
- Lanes: Ouro-1.4B Alpaca q0.2 (GPU0), Huginn 0.12 (GPU1), Ouro-1.4B ShareGPT q0.2 (GPU2), Huginn 0.26 (GPU3).
- 2 FIFO warmups + 5 measured repeats per policy, policy order randomized per repeat.
- Core time from CUDA events per recurrent call (KV event instrumentation off, as in E2); synchronized decode wall.
- Per-call receipts are in `e3_call_receipts.jsonl.gz`.
- Every run asserts physical RIC and work == the frozen plan.

**Calibration rule** (predeclared in `scripts/e3_analyze.py`): LATENCY_MODEL claims are accepted for a cell only if the predicted FIFO Core time is within ±10% of the measured FIFO median.

**Outputs.**
- `e3_oracle_comparison.csv`, `e3_call_calibration.json`.
- Plots: `e3_physical_wall_reduction.png`, `e3_call_calibration.png`.
- `physical/*/{e3_physical_runs.csv, RUN_MANIFEST.json, e3_call_receipts.jsonl.gz}`.
- `replay/`, and `failed_attempts/`: the first Huginn launch failed on the trace schema (KeyError 'family'); fixed in f6c8c9f.
