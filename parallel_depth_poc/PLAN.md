# Training-Free Parallel Recurrent Depth — PoC contract

**Date:** 2026-11-16
**Status:** PROTOCOL + CPU REFERENCE IMPLEMENTATION; **NO GPU MEASUREMENTS YET**
**Repository:** heonniy/batch-loop
**Branch:** research/training-free-parallel-depth-poc-20261116
**Base:** `main` at `e7c521d1ea91d1c3577003236f4c58ab40ac6d7b`
**Scope:** Additive `parallel_depth_poc/` only. Do not change the legacy Phase-04 gates, FERRY, dynamic_sd, or original model code.

## Question and falsifiable hypotheses

Can an already-pretrained looped transformer reuse a **frozen** recurrent core to obtain useful depth-parallel execution through a better initial trajectory and cheap iterative correction? This is **not** a new model architecture and must be compared to exact fixed-depth AR, dynamic early exit where available, and inference-time speculative methods.

- **H0 (hardware):** For width `w`, `T_F(w B)` may be smaller than `w*T_F(B)`. But this is only an *oracle batching ceiling*, not a runnable algorithm.
- **H1 (predictability):** Short-horizon depth states can be predicted using no trainable parameters, and their next-token predictions are close to exact sequential states.
- **H2 (correction):** Jacobi and block-Parareal require fewer effective sequential rounds than fixed-depth execution *at matched output quality*. Merely reducing kernel-launch count is not enough.
- **H3 (architecture):** Local perturbation amplification, convergence and actual speed differ between Huginn and Ouro. Do not assume either family contracts.
- **H4 (system):** Large offline batches may lose all depth-parallel headroom due to extra FLOPs and temporary KV/HBM.

## Models and settings

| Lane | Checkpoint | Native recurrent policy | Fixed depths to test |
|---|---|---|---|
| H | `tomg-group-umd/huginn-0125` | Prelude/Core/Coda; native shared physical-layer KV | 8, 16, 32 (verify supported config) |
| O1 | `ByteDance/Ouro-1.4B` | ACT-style native gate; depth-indexed KV | 2, 3, 4 (native cap 4; **never** pretend it supports depth 16) |
| O2 | `ByteDance/Ouro-2.6B` | Verify checkpoint-native max depth and cache semantics | 2, 3, 4 if supported |

GPU experiments (NOT executed by this commit): BF16, prefetch OFF, no-refill offline cohorts. Batch `B in {1,8,16,64,128}` subject to physical memory. Prompt lengths 256 and 1024; generate up to 64 tokens. Hold identical tokenizer, prompts, random seed, exact target depth, EOS policy, and KV policy for each paired comparison. Start with fixed-depth teacher-forced continuation; do not silently switch to dynamic exit. Huginn batch-shape BF16 divergences are known: record B=1 and batched behavior separately.

## E0: correctness contract before any speed claims

1. Inspect the actual core forward and attention/KV interface on **each pinned checkpoint revision**. The research operator is not always a simple stationary `F(h)`: use `F_r(h; KV_r, prefix)` when depth-indexed cache is present.
2. Confirm sequential baseline output matches the original model's fixed-depth logits and cached decode across 16 deterministic prompts. Invariants: row order, causal positions, layer IDs, attention masks, prelude/coda, dtype, cache layouts, and first/last KV semantics.
3. Candidate KV is isolated from **committed** KV; no speculative or incorrect KV is allowed to leak to the next token. Record first mismatching depth/token and fail closed.
4. A solver that has the right hidden state but incorrect KV is **not equivalent**. For a non-native shared-KV arm label it approximate and compare against matched approximate serial baseline.

## E1: oracle hardware headroom (GPU required, future phase)

Record exact sequential states `h_0..h_D` offline. Feed the true input state for every depth to the matching core operator, while preserving **the corresponding per-depth KV context**. One batched call at `w*B` is only valid if a real implementation can batch all the required depth-dependent attention contexts/positions.

Measure median/P10/P90 CUDA-event *core* latency after warm-up, `T_F(B)` and `T_F(w B)`, HBM peak, plus P/C/head and attention traffic. Report:
`S_oracle(w,B) = w*T_F(B) / T_F(w*B)` as a non-deployable ceiling.
If batching across KV banks is unsupported, mark the cell `UNSUPPORTED`; **do not substitute an MLP-only microbenchmark and label it model E2E**.

Do not count already-recorded states as runtime-free in an end-to-end pipeline. Sweeping very large widths without account of hidden and KV storage is invalid.

## E2: offline trajectory diagnostics (CPU or GPU with recorded data)

Capture `[D+1, B, H]` recurrent hidden states at **a single fixed decode position** (one record per token step), and optionally logits `[D+1, B, V]`, per-depth committed/candidate KV metadata and exact token IDs. Save provenance and never mix positions in one trajectory. At minimum compare:
- constant start-state predictor;
- linear extrapolation `h_1 + (r-1)*(h_1-h_0)`;
- previous-token trajectory, aligned by token position (optional, no future leakage);
- oracle reference (non-deployable).
Metrics per depth: relative L2, cosine, logit KL exact→predicted, greedy agreement, max error, and histogram by prompt length/task. `h_1` must be computed with the **real** core. No classifier trained on held-out evaluation requests.

## E3: numerical parallel correction

- Jacobi synchronous sweep: `z_r^{k+1}=F_r(z_{r-1}^{k}; KV_r)` with `z_0=h_0`. Every per-depth call at one iteration must be physically batched to claim wall-time acceleration.
- Block Parareal: partition recurrent depth into blocks, run each block's fine sequential solve with its boundary **in parallel**, then correct the boundaries using an explicitly specified coarse solver `G`. Report the actual cost of G; identity G is a correctness/reference control only.
- The `parallel_depth_poc/solver.py` functions are generic mathematical reference implementations. **They do not implement Huginn/Ouro GPU kernels or model-specific KV handling.** An adapter must implement equivalent batched attention and candidate KV banks before physical speed claims.
- Convergence criterion on held-out prompts: both true final-logit quality and online-available self-consistency/residual. A low update norm does not guarantee a correct prediction.
- Do not use reference exact hidden states at runtime, except in labeled oracle cells. Fallback to the original exact sequential decode if quality/safety gate fails; count its full cost.
- Baselines: native fixed-depth serial; native dynamic early exit (separately); naïve Jacobi, linear-init Jacobi, block-Parareal, oracle states; independent loop skipping (approximate quality arm).

## E4: dynamics and stability

Inject controlled normalized perturbations at `r in {1, floor(D/2), D-1}`, measure `||delta h_{r+k}||/||delta h_r||` for k within range. Use identical KV conditions, isolate BF16 numerical batch effects, summarize median and tails over requests/tokens. A small `||h_r-h_{r-1}||` **does not** imply perturbation contraction. Report teacher-forced final logits as well as trajectories.

## E5: physically integrated decoder (only after E0–E3 gates)

Implement frozen Huginn/Ouro adapters in isolated new files. Compare:
- exact fixed-depth AR;
- native dynamic AR, clearly using native physical early exit only where supported;
- new depth solver with all predictor/correction overhead, coda/head, KV bank creation and commit, fallback, and synchronization;
- best relevant inference-only speculative/parallel decoding baseline if implemented in the *same engine*.

Report TTFT, TPOT, committed tokens/s, end-to-end throughput, GPU core ms per token, actual core calls, executed recurrent rows, peak allocated and reserved HBM, produced token agreement, NLL, GSM8K/MATH-500 accuracy; no-refill cohort size and effective batch occupancy. Compare at *matched quality*; mismatched tokens are not an exact speedup.

## Prioritization / stop conditions

Gate A: If even E1's oracle ceiling at the target offline B is <1.15x (before solver overhead), stop the hardware-oriented direction at that B.

Gate B: If E2 predictions cause large final-logit divergence and E3 needs >=D sequential correction rounds, report negative evidence; do not claim a speedup.

Gate C: If the only apparent gain disappears when full KV and correction/fallback/commit are included, do not make a system-speed claim.

Gate D: If GPU resources are absent, publish code/tests/plan only. Record `not_run` rather than extrapolated numbers. 

## Result schema and provenance

Each run should generate `results/<run_id>/` with `config.json`, `environment.json`, `metrics.jsonl`, `summary.md`, `review_manifest.json`. Include git SHA, model and revision, GPU type, driver, CUDA/PyTorch versions, dtype, input IDs, seeds, attention backend, KV policy, batch width, solver rounds, wall/CUDA timing method, correctness gate status, and all failed/OOM cells. Commit source **before** result publication, keeping raw traces immutable.

## Current status

- [x] New branch created.
- [x] Scoped research plan.
- [ ] Model-specific forward/KV adapter reviewed against official fixed-depth semantics.
- [ ] CPU reference solver unit tests executed.
- [ ] Fixed-depth hidden-state traces collected.
- [ ] E1 real GPU width benchmark.
- [ ] E2/E3/E4 Huginn/Ouro measurements.
- [ ] E5 full no-refill decoder measurements.

No result in this branch should be interpreted as empirical acceleration until GPU evidence and quality gates are present.
