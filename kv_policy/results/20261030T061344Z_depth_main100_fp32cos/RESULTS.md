# Dynamic exit depth as KV-eviction signal — 100 examples/model×dataset (H200x8, 2026-10-30)

Run: 600 requests (ouro-1.4b, ouro-2.6b, huginn-0125 × gsm8k, math500 × 100), 4 candidate prompt
tokens each (last 96 prompt tokens, 16 gold target tokens, B=1 teacher forcing), Ouro q=0.7,
Huginn cosine 1e-3 **evaluated in FP32** (`--huginn-cosine-dtype float32`; model forward BF16), min depth 2.
Code = 3bb55fb + uncommitted patch (FP32 cosine option; prepare_items arg fix). 0 errors, 600/600,
frozen-trace replay max |ΔNLL| = 0.0 for all 600.

Deviation: official Huginn evaluator computes the cosine in model dtype. In BF16 that is
degenerate (smoke: mean depth 31.4, 97% of tokens at R=32) because cosine near 1 is quantised
(~0.004 spacing). FP32 evaluation gives mean depth 18.5, 0.7% at R=32. BF16 smoke kept at
results/20261030T055651Z_depth_smoke; FP32 smoke results/20261030T061129Z_depth_smoke_fp32cos;
pilot (4 ex) results/20261030T061344Z_depth_pilot_fp32cos.

## Depth vs single-token eviction utility (U = NLL(evict) − NLL(retain))

| model | dataset | depth range/mix | Spearman ρ(depth,U) [95% request-bootstrap] | ρ(attention_pre,U) | incr. R² (in-sample) |
|---|---|---|---|---|---|
| huginn-0125 | gsm8k | 10–32, mean 19.5 | 0.064 [-0.045, 0.161] | 0.046 | 0.017 |
| huginn-0125 | math500 | 10–32, mean 17.9 | 0.011 [-0.094, 0.122] | 0.058 | 0.001 |
| ouro-1.4b | gsm8k | d3: 53, d4: 347 | -0.025 [-0.113, 0.058] | 0.038 | 0.001 |
| ouro-1.4b | math500 | d3: 94, d4: 306 | 0.007 [-0.093, 0.109] | 0.000 | 0.000 |
| ouro-2.6b | gsm8k | all 400 at d4 | n/a (no variance) | 0.055 | n/a |
| ouro-2.6b | math500 | all 400 at d4 | n/a (no variance) | -0.045 | n/a |

## Offline 2-of-4 policy diagnostic (mean sum of singleton ΔNLL of the evicted pair; lower = better; not joint NLL)

| model | dataset | depth | reverse_depth | random | recency | H2O proxy | attn-norm |
|---|---|---:|---:|---:|---:|---:|---:|
| huginn | gsm8k | 0.0060 | 0.0275 | 0.0055 | 0.0156 | 0.0252 | 0.0234 |
| huginn | math500 | 0.0202 | 0.0108 | 0.0087 | 0.0156 | 0.0090 | 0.0154 |
| ouro-1.4b | gsm8k | 0.0310 | 0.0429 | 0.0093 | 0.0134 | 0.0253 | 0.0151 |
| ouro-1.4b | math500 | -0.0028 | 0.0026 | -0.0027 | 0.0022 | 0.0026 | -0.0003 |
| ouro-2.6b | gsm8k | 0.0372 | 0.0372 | 0.0178 | 0.0037 | 0.0260 | 0.0128 |
| ouro-2.6b | math500 | 0.0090 | 0.0090 | -0.0003 | 0.0056 | 0.0101 | 0.0041 |

## Verdict (plan decision rules)

- Data-quality gate: passes for Huginn (FP32 cosine) and Ouro-1.4B; **Ouro-2.6B fails the
  "q not degenerate" gate** at q=0.7 (98.7% of trace tokens and 100% of candidates at depth 4).
- Huginn / Ouro-1.4B: **B (negative) / D (inconclusive)** — all ρ CIs include 0, incremental R²
  ≤0.017, depth-ranked eviction does not beat random. Attention proxies are also near zero:
  with 16-token targets single-token eviction effects are small and noisy (mean U 0.0003–0.014).
- Ouro-2.6B: **D (inconclusive)** — needs a different q (pre-registered new run) to get depth variance.
Caveats: B1 gold NLL on possibly left-truncated prompts; singleton oracle, not a serving policy;
no memory measured; in-sample R².
