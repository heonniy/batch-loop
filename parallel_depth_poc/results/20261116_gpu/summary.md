# Training-free parallel recurrent depth — GPU results, run `20261116_gpu`

Hardware: 8× H200, burners stopped for every timed run, one job per GPU. BF16 is primary; FP32 is used for parity and exactness. Prefetch is off. Batches are fixed no-refill cohorts with greedy decoding.

Engine: the pinned WaveFront batch blocks, imported unmodified, with FULL depth-indexed KV (one KV slot per recurrent depth, the author-native per-step cache). `wf_engine.py` exposes `step_many(states[R,B,H], depth_ids[R])`. Rows for the same position at different depths write disjoint KV slots, so one physical call evaluates a whole Jacobi sweep, and each row sees its own per-depth context. On commit, the exact serial trajectory is recomputed, which rewrites every slot for that position. Source was committed before any result: `5c3363a2` (runners and job list), `149254d0` (E5 run rule and summarizer definitions) and `e568ea2d` (E5 job list).

Data:
- Receipts: `raw/` (per-job `config.json`, `summary.json` and `metrics.jsonl.gz`; E0/E1 JSON; E5 JSONL).
- Derived tables: `summary.json` (`summarize_gpu.py`).
- Checksums: `review_manifest.json`.
- Job logs: `job_logs.tar.gz`.
- 36 diagnostic jobs plus 8 E5 jobs, all with rc=0.

## Status per plan item

| item | status |
|---|---|
| CPU reference tests (`tests/`) | passed, 12/12 |
| E0 native parity, 16 prompts × 32 tokens, FP32 | **passed**, all 9 (model, D) cells |
| E0 KV isolation (candidate KV never leaks into committed KV) | **passed**, all 27 diagnostic cells |
| E1 oracle width headroom | done; OOM cells listed below |
| E2 predictability | done, 9 cells at B16/L256, plus L1024, B1 and FP32 lanes |
| E3 Jacobi / block-Parareal | done |
| E4 perturbation | done |
| E5 integrated decoder, B=1 | done for h D32, o14 D4 and o26 D4 (pre-registered gate passed) |
| E5 at B>1 | not_run (see Gate C) |
| E5 native dynamic early exit | not_run in this engine; Ouro τ results are in `dynamic_sd/full_rollout_frontier_20261113` |
| Ouro depth > 4 | not run by design (native cap 4) |

Cohorts:
- E2–E4 use GSM8K 3-shot prompts, left-truncated to exactly L=256 tokens. The L=1024 cohort is synthetic: MATH-500 4-shot + GSM8K 8-shot, left-truncated.
- Each run decodes 64 greedy tokens.
- E5 uses held-out GSM8K 3-shot prompts 100–149 at natural length, up to 256 tokens, with lm-eval stop strings.

## E0 — engine vs HF author-native forward (FP32)

The HF forward is teacher-forced on the engine's own greedy tokens. Huginn receives the engine's seeded initial state through `input_states`. Ouro runs with `total_ut_steps=D`, through a call-boundary shim for the installed transformers `create_causal_mask` signature; the shim changes no math.

| model | D | greedy agreement | max \|Δlogit\| | max KL |
|---|---|---:|---:|---:|
| huginn-0125 | 8 / 16 / 32 | 1.000 / 1.000 / 1.000 | 6.6e-5 / 9.8e-5 / 9.8e-5 | ≤3.8e-7 |
| Ouro-1.4B | 2 / 3 / 4 | 1.000 / 1.000 / 1.000 | 1.9e-4 / 1.9e-4 / 9.0e-4 | ≤3.0e-7 |
| Ouro-2.6B | 2 / 3 / 4 | 1.000 / 1.000 / 1.000 | 2.0e-4 / 2.8e-4 / 1.5e-3 | ≤2.8e-7 |

Every diagnostic cell's rollout reproduced a plain fixed-depth AR rollout token for token. The commit-time recompute was bit-identical to the reference trajectory in 100% of steps.

## E1 — oracle ceiling S_oracle(w,B) = w·T_F(B)/T_F(wB), BF16, CUDA events

These are exact-state inputs with real per-depth KV contexts. The ceiling is non-deployable.

| model (D) | L | B=1 | B=8 | B=16 | B=64 | B=128 |
|---|---|---:|---:|---:|---:|---:|
| huginn (w=32) | 256 | 26.9 | 10.1 | 5.95 | 2.11 | **1.49** |
| huginn (w=32) | 1024 | 18.8 | 4.56 | 2.71 | OOM | OOM |
| Ouro-1.4B (w=4) | 256 | 3.96 | 3.84 | 3.57 | 2.82 | 2.03 |
| Ouro-1.4B (w=4) | 1024 | 3.93 | 3.42 | 3.05 | 1.62 | **1.31** |
| Ouro-2.6B (w=4) | 256 | 4.00 | 3.83 | 3.58 | 2.90 | 2.05 |
| Ouro-2.6B (w=4) | 1024 | 3.95 | 3.43 | 3.19 | 1.62 | OOM |

- T_F(1) is 3.5 ms for Huginn (4 core layers), 18 ms for Ouro-1.4B and 35 ms for Ouro-2.6B. Ouro has 24 layers per step, and its core call is launch- and host-bound in this eager engine.
- That is why S_oracle ≈ w at small B: the ceiling partly measures launch overhead, not FLOP headroom.
- The ceiling falls quickly with B and with context length.
- **Gate A (<1.15×):** not triggered at any measured B. The lowest values are 1.31× for Ouro L1024 B128 and 1.49× for Huginn L256 B128. The 1024-token Huginn cohort at B≥64 is OOM.

## E2 — training-free trajectory predictors (B16, L256, BF16; top-1 = greedy token from the predicted state vs exact)

| model D | constant (h1) top-1 @D / KL | linear | prev token | prev_shift |
|---|---|---|---|---|
| huginn 32 | 0.41 / 1.94 | 0.41 / 2.02 (rel-L2 34 at D) | 0.02 / 8.8 | **0.50 / 1.61** |
| huginn 8 | 0.40 / 1.90 | 0.39 / 1.97 | 0.03 / 8.1 | 0.49 / 1.58 |
| Ouro-1.4B 4 | **0.89 / 0.39** | 0.89 / 1.05 | 0.02 / 9.9 | 0.84 / 0.42 |
| Ouro-2.6B 4 | **0.94 / 0.15** | 0.94 / 0.79 | 0.02 / 10.8 | 0.88 / 0.25 |

- No training-free predictor reproduces Huginn's deep states. Relative L2 is ≥0.93 at every depth, and final-logit KL is ≈1.6–2.0.
- Linear extrapolation diverges with depth.
- The previous token's trajectory is useless: top-1 is 2%, because states are token-specific.
- For Ouro, a single real step (h1) already predicts the depth-4 token 89–94% of the time. This is the same fact that makes shallow exits work.

## E3 — Jacobi and block-Parareal on the real core

**Jacobi** is reported over B16/L256 BF16. Effective sequential rounds = 1 (the real h1) + k, against D for serial.

| model D | init | k_stable mean / p50 / p90 (oracle stop) | online gate τ=0.1: rounds / top-1 | τ=0.03: rounds / top-1 | exact after D rounds |
|---|---|---|---|---|---|
| huginn 32 | prev_shift | 2.4 / 1 / 5 | 10.0 / 0.989 | 15.4 / 0.999 | no: BF16 rel 3.7e-2 (FP32 2.6e-6) |
| huginn 32 | constant | 2.4 / 1 / 5 | 10.2 / 0.980 | 15.5 / 0.995 | same |
| huginn 16 | prev_shift | 2.3 / 1 / 5 | 9.6 / 0.981 | 13.9 / 0.994 | — |
| huginn 8 | prev_shift | 1.9 / 1 / 4 | 7.75 / 0.995 | 8 / 0.996 | — |
| Ouro-1.4B 4 | constant | 1.04 / 1 / 1 | 3.3 / 1.000 | 4.0 / 1.000 | yes, bit-exact (BF16) |
| Ouro-2.6B 4 | constant | 1.05 / 1 / 1 | 3.2 / 0.994 | 4.0 / 1.000 | yes |

- `k_stable` is the first round after which the final-depth greedy token always equals the exact token. Finding it **requires the exact token**, so it is not an online rule.
- The online residual gate (relative state update < τ) needs about ⅓ to ½ of D rounds on Huginn. On Ouro it essentially never fires before exactness.
- Huginn's BF16 "Jacobi = serial after D rounds" is **not bit-exact**: the width-32 batched call and the width-1 calls differ numerically (batch-shape effect), so final-token agreement at round D is 0.996. In FP32 the residual difference is 2.6e-6.

**Block-Parareal** (Huginn D=32, B16, L256):

| block b | G | rounds to stable / blocks | sequential calls to stable |
|---|---|---:|---:|
| 2 | identity | 15.9 / 16 | 31.7 |
| 4 | identity | 7.25 / 8 | 29.0 |
| 8 | identity | 3.0 / 4 | 24.3 |
| 2 | one fine step | 9.5 / 16 | 197 |
| 4 | one fine step | 3.6 / 8 | 55 |
| 8 | one fine step | 1.6 / 4 | 24.8 |

- With identity G, Parareal needs about one round per block, i.e. serial-equivalent work. A cheap coarse operator that tracks Huginn's dynamics does not exist without training.
- On Ouro (D=4, b=2), Parareal reaches a stable token after ~1 round but costs 2.1 sequential calls with identity G, or 7 with one-step G, against a serial cost of 4.

## E4 — perturbation dynamics

‖Δh_D‖/‖Δh_r‖, median [max]. Final-token flip rate is in parentheses.

| model D | precision, ε | from r=1 | from r=D/2 | from r=D−1 |
|---|---|---:|---:|---:|
| huginn 32 | FP32, 1e-3 | 5e-4 [9e-3] (0) | 5.7e-3 [0.10] (0) | 0.65 [0.74] (0) |
| huginn 32 | BF16, 1e-1 | 0.14 [0.33] (0.2%) | 0.14 [4.1] (0.4%) | 0.67 [0.77] (1.3%) |
| Ouro-1.4B 4 | FP32, 1e-3 | 0.38 [15] (0) | 0.48 [2.1] (0) | 0.78 [1.8] (0) |
| Ouro-2.6B 4 | FP32, 1e-3 | 0.69 [56] (0) | 0.91 [2.5] (0) | 1.48 [3.6] (0) |
| Ouro-2.6B 4 | BF16, 1e-1 | 0.81 [13.7] (1.9%) | 1.10 [9.4] (1.8%) | 1.53 [7.5] (2.3%) |

- **Huginn contracts strongly in FP32:** a perturbation at depth 1 shrinks about 2000× by depth 32.
- **Ouro does not contract uniformly.** Medians are below 1, but tails reach 15–56×. Ouro-2.6B's last step expands (median 1.48). This confirms H3.
- BF16 at ε=1e-2 shows ratios ≈1.1–2 for every model. That is the BF16 rounding floor, not dynamics; the FP32 rows are the dynamics.
- Unperturbed reruns are bit-identical (noise floor 0).

## E5 — physically integrated decoder, B=1, held-out GSM8K prompts 100–149 (n=50, ≤256 tokens)

Speed is Σtokens/Σwall relative to AR, with a paired prompt bootstrap 95% CI. Wall time includes prefill, every core call, the per-round coda used by the gate, Python row packing and commit.

| model D | arm | speed vs AR [95% CI] | TPOT p50 | core rounds/token | sequence exact to AR | GSM8K strict / flex |
|---|---|---|---:|---:|---:|---|
| huginn 32 | AR | 1.00 (8.8 tok/s) | 112 ms | 32 | 1.00 | 0.22 / 0.30 |
| | jac_exact | **0.81** [0.80, 0.81] | 138 ms | 32 | 0.84 | 0.22 / 0.30 |
| | jac_g0.03 | 1.06 [1.05, 1.07] | 102 ms | 16.7 | 0.66 | 0.18 / 0.26 |
| | jac_g0.1 | **1.57** [1.55, 1.58] | 70 ms | 11.3 | 0.30 | 0.24 / 0.32 |
| | trunc16 (depth 16) | 1.94 [1.93, 1.95] | 58 ms | 16 | 0.08 | 0.18 / 0.26 |
| | trunc8 | 3.62 | 31 ms | 8 | 0.00 | 0.06 / 0.06 |
| Ouro-1.4B 4 | AR | 1.00 (14.5 tok/s) | 69 ms | 4 | 1.00 | 0.62 / 0.76 |
| | jac_exact / g0.03 / g0.1 | 0.99 / 0.98 / 0.98 | 69–70 ms | 4.0 | 0.86–0.88 | 0.60–0.62 / 0.74–0.76 |
| | trunc3 / trunc2 | 1.33 / 1.99 | 51 / 34 ms | 3 / 2 | 0.28 / 0.08 | 0.58 / 0.72 · 0.46 / 0.62 |
| Ouro-2.6B 4 | AR | 1.00 (7.3 tok/s) | 136 ms | 4 | 1.00 | 0.70 / 0.86 |
| | jac_exact / g0.03 / g0.1 | 0.99 / 0.99 / 1.02 | 138 ms | 3.9–4.0 | 0.80–0.82 | 0.70 / 0.86 |
| | trunc3 / trunc2 | 1.33 / 2.00 | 102 / 68 ms | 3 / 2 | 0.32 / 0.20 | 0.72 / 0.86 · 0.58 / 0.76 |

For the parallel-depth arms (`jac_*`), "sequence exact to AR" below 1 is BF16 batch-shape drift (width-D vs width-1 calls) accumulated over up to 256 tokens. Huginn's `jac_exact` matches AR in 84% of sequences, Ouro's in 82–86%. For the `trunc` arms the gap is mostly the shallower depth itself: at depth 8 Huginn reaches 0 exact sequences and 0.06 accuracy.

## Gates (PLAN.md)

- **Gate A (hardware ceiling):** not triggered, but the ceiling falls from ~27× (Huginn B1) to 1.5× (B128, L256) and is OOM for 1024-token Huginn cohorts at B≥64.
- **Gate B (predictability / correction):**
  - **Triggered for Ouro.** The online gate needs the full D−1 rounds, so Jacobi is no faster than serial (E5: 0.98–1.02×).
  - **Huginn:** training-free predictions diverge at the logit level (KL ≈1.6–1.9), but Jacobi correction reaches a stable token in ~2 oracle rounds and ~10–16 online-gated rounds of 31.
- **Gate C (system speed after full costs):**
  - **No exact speedup** for any model. The exact Jacobi arm is 0.81× (Huginn) and 0.99× (Ouro).
  - The only measured gain is Huginn B=1 with the τ=0.1 gate: **1.57× [1.55, 1.58]**, which is **approximate** (30% of sequences token-identical to AR).
  - At comparable or lower fidelity, plain depth truncation (Huginn depth 16) is faster (1.94×, 8% exact).
  - n=50 accuracy differences (0.24 vs 0.18 vs 0.22 strict) are not resolvable.
  - Under the plan's "matched quality" rule this is **not a system-speed claim**.
  - Projections from measured E1 components put the online-gated Huginn variants below 1× already at B=16 (width-32 call 20.7 ms vs 3.8 ms), so E5 at B>1 was not run.

## Deviations and caveats

- Cohorts for E1–E4 are left-truncated to an exact length; the 1024 cohort is synthetic concatenated text. E5 uses natural prompts.
- The E5 Jacobi init (prev_shift for Huginn, constant for Ouro), the τ values and the trunc depths were chosen after E3 and committed before E5 ran (`e568ea2d`). The E5 prompts are disjoint from the diagnostic prompts.
- `k_stable` and every "at k_stable" projection use the exact token. They are oracle upper bounds, not deployable.
- E5 timings come from an eager Python engine. Ouro's core call is launch-bound (17–35 ms), and CUDA graphs or fused kernels would change absolute numbers and possibly the ratios.
- Huginn's BF16 batch-shape sensitivity is visible here too: 1 of 4 B=1 prompts diverged from its B=16 rollout at token 15, and 1 of 4 Ouro-2.6B prompts at token 25.
- No native dynamic-exit arm in E5, no Ouro depth > 4, no Ouro-2.6B E0 beyond D ≤ 4.
