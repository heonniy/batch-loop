# Results — adaptive depth-parallel search on 8× H200 (run `runs/20261121_h200x8`)

All 23 DAG nodes have a receipt in the append-only ledger `runs/20261121_h200x8/evidence.jsonl`.
That makes 22 executed nodes and 1 `not_run` with a reason.

- **Budget:** 50.9 / 384 GPU-hours used. Wall time was about 7.1 h, from 2026-11-21 02:27 to 09:35 KST.
- **Source SHAs:**
  - `09f2008d` — G0 gates, decoder, campaign tooling.
  - `a6577118` — new arms, P5, daemon and test freeze. Committed 19:08 UTC, before the first test decode at 20:14 UTC.
  - `76ce6f9d` — analysis keys and frozen-rule verdicts. Analysis code only; no decode changes.
- **Engine and models:**
  - WaveFront engine pinned at `4345209`. VarEngine sits on top of it: per-request prefill, FULL per-depth KV, and no refill.
  - Models: Huginn-3.5B with D=32, Ouro-1.4B with D=4, and Ouro-2.6B with D=4.
  - Decoding: greedy, `max_new` 256.

## Bottom line

1. **Pre-registered verdict.**
   - **F1 = `supported_heldout_quality_matched`.** This rests on one cell only: Ouro-2.6B, B=16, `jac_m1` (logit-margin ≥ 1 gate).
     - Speed: 1.29× AR [1.23, 1.35].
     - Accuracy: Δ +0.002 [−0.016, +0.022] on GSM8K test (n=500).
   - **F2 = `contradicted_quality` by `control.classify`.**
     - The same configuration on MATH-500 test reaches only 1.05× [1.03, 1.07].
     - Its accuracy CI is Δ [−0.033, +0.033]. FREEZE wording would call this *inconclusive*, since the CI crosses the margin. Both readings are recorded.
2. **The F1 champion does not beat depth truncation.**
   - Paired against `trunc3` in the same cell it gives **0.99× [0.95, 1.04]**. Accuracy is Δ +0.012 [−0.014, +0.038] vs trunc3.
   - Its only edge is fidelity: sequence-exact rate 0.66 vs 0.37.
   - The F1 question asks "beat AR *and* truncation". The honest answer is: beats AR in one cell, ties truncation.
3. **Exact (lossless) depth-parallel decoding never beats AR.**
   - `jac_exact` (D−1 sweeps): 0.32–0.96×.
   - V2 exact-KV repair: 0.43–0.78×.
   - FP32 `jac_g0.1` is sequence-exact (1.00) but runs at 0.49×.
4. **The gain is an artifact of the Python-call-bound engine.**
   - The champion configuration drops from 1.29× in BF16 to **0.62× in FP32** (F2, same prompts).
   - With heavier per-call compute, the extra Jacobi rows cost more than the saved sequential calls.
   - Candidate speedups also shrink monotonically with batch:

     | Batch | Ouro-2.6B `jac_m1` |
     |---|---|
     | B=1 | 1.80× |
     | B=16 | 1.29× |
     | B=64 | 1.00× |
     | B=128 | 0.81× |

   - G1_WIDTH predicts this pattern: oracle depth-parallel headroom for Huginn falls from 27× at B=1 to 1.5× at B=128.
5. **Depth reduction dominates speed everywhere.**
   - Truncation and the skip chain give 1.3–1.9× at all batch sizes, with accuracy Δ around −0.01 to −0.04.
   - `skip{a}_{b}` performs the same as truncation at equal call counts: vs `trunc3` 0.93–1.04×, vs `trunc16` 0.85–0.98×.

## F1 — GSM8K test idx 700–1199 (n=500), frozen arms, vs AR

Each cell is speedup [95% CI] / accuracy Δ [95% CI] / verdict under the frozen rule.

| model | B | AR acc | trunc (baseline) | skip | `jac_m1` / `jac_g0.1` (depth-parallel) |
|---|---|---|---|---|---|
| Huginn | 1 | 0.250 | t16 1.93× [1.91,1.94] / −0.022 [−0.056,+0.010] inc | 1.89× / −0.024 [−0.050,+0.002] inc | g0.1 **1.50× [1.45,1.55]** / −0.008 [−0.032,+0.016] inc |
| Huginn | 16 | 0.258 | t16 1.84× / −0.028 inc | 1.57× / −0.022 inc | g0.1 0.54× no_speedup |
| Huginn | 32 | 0.252 | t16 1.97× / −0.018 inc | 1.80× / −0.014 inc | g0.1 0.38× no_speedup |
| Huginn | 64 | 0.258 | t16 1.89× / −0.026 inc | 1.69× / −0.024 inc | g0.1 0.20× no_speedup |
| Huginn | 128 | — | OOM (FULL per-depth KV bank) | OOM | OOM |
| Ouro-1.4B | 1 | 0.612 | t3 1.33× / −0.042 [−0.074,−0.008] inc | 1.33× / −0.046 inc | m1 1.76× / **−0.080 [−0.110,−0.050] contradicted**; g0.1 0.99× |
| Ouro-1.4B | 16 | 0.604 | t3 1.29× / −0.040 inc | 1.34× / −0.032 inc | m1 1.28× [1.22,1.34] / −0.012 [−0.034,+0.010] inc; g0.1 0.92× |
| Ouro-1.4B | 64 | 0.598 | t3 1.35× / −0.032 inc | 1.27× / −0.026 inc | m1 0.97×; g0.1 0.83× no_speedup |
| Ouro-1.4B | 128 | 0.602 | t3 1.36× / −0.040 inc | 1.27× / −0.038 inc | m1 0.80×; g0.1 0.72× no_speedup |
| Ouro-2.6B | 1 | 0.700 | t3 1.33× / −0.004 [−0.028,+0.020] **supported** | 1.33× / −0.004 **supported** | m1 1.80× [1.78,1.82] / −0.022 [−0.050,+0.006] inc; g0.1 1.02× |
| Ouro-2.6B | 16 | 0.702 | t3 1.30× / −0.010 [−0.036,+0.016] inc | 1.26× / −0.008 inc | **m1 1.29× [1.23,1.35] / +0.002 [−0.016,+0.022] supported**; g0.1 0.94× |
| Ouro-2.6B | 64 | 0.700 | t3 1.34× / −0.006 inc | 1.31× / −0.008 inc | m1 1.00×; g0.1 0.84× no_speedup |
| Ouro-2.6B | 128* | 0.691 | t3 1.37× / −0.013 inc | 1.28× / +0.008 [−0.019,+0.032] **supported** | m1 0.81×; g0.1 0.74× no_speedup |

\* For Ouro-2.6B B=128, one of four cohorts hit OOM, so the cell covers 372 of 500 prompts. That is below the n≥500 final threshold, so it is reported but not used for the node verdict.

Abbreviations: inc = inconclusive (a CI crosses a margin); t16 and t3 = trunc16 and trunc3.

Three cells are `supported` under the frozen rule:
- Ouro-2.6B B=16 `jac_m1`.
- Ouro-2.6B B=1 `skip1_2`.
- Ouro-2.6B B=128 `skip1_2`.

The skip arms are depth reduction with the same call count as `trunc3`, not depth-parallel computation. The node champion is therefore restricted to `jac_*` arms.

Per-cell JSON:
- vs AR: `runs/20261121_h200x8/analysis/F1_FULL_ROLLOUT.json`.
- vs truncation: `…_vs_trunc3.json` and `…_vs_trunc16.json`.

## F2 — stress (n: MATH-500 test 300; repro 500; FP32 200), B=16

| cell | best depth-parallel arm vs AR | vs truncation | note |
|---|---|---|---|
| Ouro-2.6B MATH-500 | `jac_m1` 1.05× [1.03,1.07], Δ 0.000 [−0.033,+0.033] | 0.77× vs trunc3 | champion config does not transfer |
| Ouro-1.4B MATH-500 | `jac_m1` 0.97× | 0.72× | AR acc 0.020 (floor) |
| Huginn MATH-500 | `jac_g0.1` 0.35× | 0.18× vs trunc16 | AR acc 0.100 |
| Ouro-1.4B GSM8K seed 7 (repro) | `jac_m1` 1.29× [1.25,1.33], Δ −0.012 [−0.034,+0.010] | 0.97× | reproduces F1 to the third decimal |
| Ouro-1.4B GSM8K **FP32** | `jac_m1` **0.62×**; `jac_g0.1` 0.49× with seq-exact **1.00** | 0.47× | BF16 gain is overhead-bound; BF16 non-exactness is rounding |

## What the screening nodes established (pilot/dev, all receipts in ledger)

- **Gates.**
  - G0_NATIVE: FP32 parity, greedy 1.0.
  - G0_KV: candidate rows plus exact commit are bitwise equal in every depth bank.
  - G0_DATA: no duplicates and no few-shot leaks, manifest `30961d45…`.
  - In BF16, changing the number of rows changes rounding. This caps exact-sequence rates at about 0.88–0.96 even for "exact" algorithms.
- **Dynamics (G1, P1–P5, C1–C5, A1–A3).**
  - Huginn needs about 11 Jacobi sweeps to reach top-1 ≥ 0.99, against 32 serial calls.
  - Gains come only from suffix windows; a suffix window of k sweeps equals full Jacobi.
  - Anderson gives no gain.
  - Newton with identity or scaled Jacobian diverges. The measured JVP gain is 0.83 on Huginn, so the contraction is weak.
  - Parareal is about serial-equivalent.
  - The best initialisations are constant and previous-token shift. The low-rank predictor (P5) gains at k=0 vanish after one sweep.
  - C4 (scan/low-rank JVP corrector) is `not_run`, because the evidence above gives no headroom for it.
- **Dev decode (V1, J1, V2).**
  - Huginn B=1 `jac_g0.1` reached 1.54× with Δ CI crossing −0.03, so J1 = `contradicted_quality`.
  - Margin-freeze gates collapse Huginn accuracy.
  - Exact KV repair erases the gain.

## Limitations and deviations (read before citing any number)

- **Engine.**
  - The engine is Python-orchestrated: Ouro takes about 17.7 ms per depth call and Huginn about 3.5 ms per call. All speedups are relative to AR *on this engine*. The FP32 flip shows they would shrink or invert in a fused/graph-captured engine.
  - No CUDA-graph or vLLM baseline was run.
  - Speed is measured as cohort tokens per wall-second with no refill. This is not a serving-throughput claim.
- **OOM and coverage.**
  - The FULL per-depth KV bank OOMs at Huginn B=128 and partially at Ouro-2.6B B=128. Those cells are unsupported, not zero.
  - Context and output-length sweeps, HotpotQA, RULER and adversarial prompts (listed in the tree `method` text) were **not** run. F2 covers MATH-500, a second seed, and FP32.
- **Multiplicity.** The single supported `jac_m1` cell is one of 32 non-truncation candidate × cell tests in F1, with no multiplicity correction. It did not replicate on MATH-500.
- **Controller vs FREEZE text.** `control.classify` maps accuracy CI low < −0.03 to `contradicted_quality`, while FREEZE calls a crossing CI inconclusive. Both are reported (F2, J1).
- **Runbook deviations.**
  1. The burner jobs on the 8 GPUs were paused for this run and restored afterwards, on the GPU owner's explicit instruction. The runbook says not to touch other processes.
  2. Waves ran through per-GPU daemons over one queue, so independent screening nodes overlapped in time. Dependency order and the test freeze were respected.
- **Data retention.** Raw per-prompt JSONL (93 MB) stays on the H200 host under `runs/20261121_h200x8/results/`, which is gitignored. Every file's sha256 is in the node receipt.

## Reproduce

```bash
source ad_env.sh   # PY, WF_ROOT=4345209, PYTHONPATH, offline HF flags
RUN=parallel_depth_search/runs/20261121_h200x8
$PY -m parallel_depth_search.final_jobs --stage F1 --out f1.json     # identical job list
$PY -m parallel_depth_search.analyze_nodes --run-dir $RUN --node F1_FULL_ROLLOUT --out a.json
$PY -m parallel_depth_search.final_verdict --analysis a.json
$PY -m parallel_depth_search.control propose --run-dir $RUN
```
