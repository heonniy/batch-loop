# Phase 05 — consolidated results (E1, E2, E3)

Evidence classes are never mixed. **No number here is live-exit inference performance.** E2 times stages at a fixed state. E3 physical runs force frozen B=1 trace depths. Per-cell table: [summary_table.csv](summary_table.csv); raw data: [E1](../20261019_e1), [E2](../20261019_e2_d6b8674), [E3](../20261019_e3_7f903d8).

## Headline

1. **RQ1 — yes, mostly.** Physical forced-depth gains come from removing whole recurrent Core invocations.
   - Ouro-1.4B: Core time per call is the same under FIFO and Oracle (19.4–22.4 ms) although Oracle cohorts are wider. Core reduction / RIC reduction = 0.87–1.02.
   - Huginn: calls get slightly more expensive when wider at B=32 (6.70→6.93 ms), so tracking drops to 0.71–0.87. At B=8 it is 0.99–1.05.
2. **RQ2 — this runtime is launch/host-dominated at small B·L, then KV-path-dominated.** No cell shows weight-memory-bound or compute-bound behaviour: achieved bandwidth is ≤ 15% of HBM peak and achieved FLOP rate ≤ 2.3% of peak. The architectures differ sharply:
   - Ouro's 24/48-layer Core costs a flat 23–27 / 45–51 ms per call over B=1–32 (L ≤ 1024).
   - Huginn's 4-layer Core is cheap (6.6 ms) but grows with B·L.
3. **RQ3 — no.** The measured-cost (latency-aware) Oracle does not materially beat simple depth sorting.
   - Replay: at most 0.23% of FIFO predicted time better; 0 of 480 modelled cells > 0.5%.
   - Physical: identical RIC in all 20 cells and predicted Core within 0.03%. Measured wall differences of −2.6…+2.3 pp have mixed sign and match the N=B-control noise (−3.0…+2.1 pp).
4. **The LATENCY_MODEL calibration fails the predeclared ±10% rule in every cell.** E2 single-call medians over-predict the E3 steady-state FIFO Core time by 13–27% (per call: measured/predicted 0.76–0.85). Absolute latency-model claims are therefore **rejected**. Relative predicted reductions still track the physical Core reductions within about 3 pp; this is reported as exploratory only.

## E1 — retrospective (CPU, 959 reconciled cells; details in [E1 RESULTS](../20261019_e1/RESULTS.md))

| Cell (phase02/03, PHYSICAL_FORCED_DEPTH, N256/B32, T32) | RIC reduction | Core reduction | Wall reduction | ms/call FIFO→Oracle | Core / RIC tracking |
| --- | ---: | ---: | ---: | --- | ---: |
| Ouro-1.4B Alpaca q0.2 | 29.9% | 29.9% | 30.9% (×1.45) | 20.64→20.66 | 1.00 |
| Ouro-1.4B Alpaca q0.4 | 23.4% | 23.5% | 23.4% | 21.74→21.72 | 1.00 |
| Huginn 0.12 | 20.6% | 17.6% | 16.2% | 6.75→7.01 | 0.85 |
| Huginn 0.5 | 15.3% | 13.5% | 11.0% | 9.92→10.13 | 0.88 |

- Trace-only RIC reduction grows with N/B and depth dispersion (Ouro r = 0.83 with depth CV).
- Phase03 predictors removed only 3–6% of RIC.
- **Missing evidence:** no Ouro-2.6B physical cells or raw traces; only one non-B32 physical cell (3 repeats); no long-context physical cell; live-exit rows are counts without timing.

## E2 — H100 stage latency (PHYSICAL_FORCED_DEPTH stage timing; 168 cells, 19 OOM, 0 gate failures)

Median Core time per **one** recurrent invocation (ms):

| Model | L | B=1 | 8 | 32 | 64 | 128 | B* (≤ +10% of B=1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ouro-1.4B (24 layers, h=2048) | 128 | 23.3 | 23.9 | 24.5 | 26.0 | 26.0 | 32 |
| | 512 | 24.3 | 25.8 | 26.2 | 42.3 | OOM | 32 |
| | 2048 | 24.9 | 26.9 | OOM | OOM | OOM | 8 |
| Ouro-2.6B (48 layers) | 128 | 45.2 | 47.1 | 47.8 | 51.0 | 51.3 | 32 |
| | 512 | 44.6 | 47.6 | 50.6 | OOM | OOM | 16 |
| | 2048 | 45.0 | 46.8 | OOM | OOM | OOM | 8 |
| Huginn-3.5B (4-layer core, h=5280) | 128 | 6.6 | 6.8 | 8.9 | 11.5 | 18.2 | 8 |
| | 512 | 6.6 | 8.5 | 17.9 | 32.0 | 60.4 | 2 |
| | 2048 | 7.1 | 16.8 | 59.9 | OOM | OOM | 1 |

- **Nsight Systems** (predeclared subset, L=512): GPU-busy fraction inside a Core call is 0.15 / 0.16 / 0.25 at B=1 (Ouro-1.4B / 2.6B / Huginn) and 0.65 / 0.66 / 0.64 at B=32.
  - Ouro launches 1,661–3,533 kernels per Core call (mean kernel 3–13 µs); Huginn launches 352.
  - Host dispatch time ≈ Core GPU time at small B.
- **Regime map** (`bottleneck_label` in e2_stage_latency.csv):
  - *launch/host-dominated* in every model at B ≤ 16–32 for L ≤ 1024 (Ouro) and at B ≤ 2–8 (Huginn);
  - *KV-path-dominated* (measured KV-update ≥ 50% of the Core call) for Ouro at B·L ≳ 16k and for Huginn at B·L ≳ 8–16k. Example: Huginn L=512, B=32 spends 10.0 of 17.9 ms in KV updates.
  The phase02 KV backends gather or concatenate the full history on each call, so this is **implementation-level KV traffic**, not an attention-FLOP limit.
  - **No weight-memory-bound or compute-bound cell**: bandwidth fraction ≤ 0.15, FLOP fraction ≤ 0.023.
  - Remaining cells are `unresolved`; no label was guessed.
- **Stage shares per single invocation:**
  - Ouro has no Transformer Prelude/Coda. Embedding (0.07 ms) and LM head (0.18–0.21 ms) are ~1% next to a 23–51 ms Core.
  - Huginn's Prelude (2 layers, 4–37 ms) plus Coda (2 layers, 3.7–31 ms) cost about as much as one 4-layer Core call. Core is 41–49% of Core+boundary for one invocation, and ~88% per token at mean depth 8.6.
- **OOM boundary:** Ouro's depth-indexed KV (Rmax 4 slots per layer) runs out first (1.4B: B=32 at L=2048; 2.6B: B=16 at L=2048). Huginn's shared KV reaches B=128 at L=512.

## E3 — latency-aware Oracle

Physical (PHYSICAL_FORCED_DEPTH, frozen depths; 4 lanes × 110 runs, 440/440 OK, 2 warmups + 5 reps). Decode-wall reduction vs FIFO (median):

| Lane | N256 B8 | N256 B32 | N64 B8 | N64 B32 | N32 B32 (N=B control) |
| --- | --- | --- | --- | --- | --- |
| Ouro-1.4B Alpaca q0.2: depth / latency / random | 28.4 / 28.0 / 1.0 | 26.7 / 29.0 / 1.3 | 24.9 / 24.4 / −1.8 | 14.6 / 14.3 / 0.0 | 0.8 / −1.3 / −0.1 |
| Ouro-1.4B ShareGPT q0.2 | 29.2 / 29.0 / 0.9 | 28.5 / 30.7 / 0.8 | 24.6 / 24.5 / 0.0 | 17.8 / 16.9 / 0.0 | −0.8 / −0.5 / 0.2 |
| Huginn 0.12 | 14.3 / 11.7 / −1.3 | 15.0 / 15.8 / −0.7 | 11.3 / 10.3 / −1.3 | 7.7 / 6.6 / −1.7 | 2.1 / −0.2 / 0.2 |
| Huginn 0.26 | 9.3 / 10.0 / −0.8 | 7.6 / 10.2 / 4.1 | 10.2 / 10.1 / 1.2 | 11.1 / 12.0 / 3.9 | −3.0 / −2.3 / 1.2 |

- Latency and depth Oracle produce identical RIC everywhere. Predicted Core differs by < 0.03%: in this hardware regime the cost of a Core call barely depends on width at these contexts (L = 65–96), so depth sorting is already near-optimal for the measured objective. The unit tests show it is exactly optimal for any width-affine cost.
- Huginn's FIFO→Oracle *predicted* reduction is below its RIC reduction at B=32 (16.0% vs 20.6% at N256/B32), because depth-sorted cohorts are wider and Huginn's Core cost grows with width. The physical Core reduction (17.9%) agrees in direction.
- **Calibration:** predicted FIFO / measured FIFO Core = 1.13–1.27 in all 20 cells → **REJECTED** under the ±10% rule. Likely cause (not proven): E2 times isolated, synchronized single calls, whereas E3 runs back-to-back calls in steady state. The E2 table is reliable for *shape* (B/L scaling) but not for absolute E3 decode time.

## Analysis requested

**Memory/compute/KV-bound regions.**
- On the phase02 runtime nothing is weight-HBM- or compute-bound. The H100 sits mostly idle at small B because the eager HF/PyTorch path launches thousands of tiny kernels per Core call.
- Growth with B appears only when KV gather/concatenate traffic or attention work fills those launch gaps: Ouro above B·L ≈ 16k; Huginn from B·L ≈ 2–4k, because each of its few, wide layers attends with 55 heads × 96.
- The practical "knee" B* is therefore large for Ouro (16–32 up to L=1024) and small for Huginn (1–8 for L ≥ 128).
- These regimes are properties of this implementation: a CUDA-graph or fused-KV engine would shift them, and they were not measured here.

**Ouro vs Huginn architecture.**
- Ouro spends essentially all decode time in a deep, flat-cost Core: 24/48 layers, ~1 ms of launch-bound time per layer. Removing a whole Core invocation saves ~20 ms regardless of how many rows it carried, so RIC reduction converts ~1:1 into time.
- Huginn's 4-layer Core is cheap, wide and B·L-sensitive, and its 2+2 Prelude/Coda layers add a per-cohort cost that grouping cannot remove. Depth grouping widens calls, so Huginn converts only 0.71–0.88 of its RIC reduction into Core time and less into wall time. Its oracle headroom is also smaller, because depth dispersion is lower (CV 0.10–0.13 vs Ouro 0.22–0.36).
- **Cross-model confounds:** width (2048 vs 5280), attention implementation (HF eager vs SDPA math), KV semantics (depth-indexed vs shared), Rmax (4 vs 16) and depth distributions all differ. Layer count alone is not claimed to cause the difference.

**Latency Oracle vs Depth Oracle — improvement potential.**
- On this runtime, near zero. When Core cost is flat or affine in active width, depth sorting minimizes the measured objective.
- A latency-aware objective can only help if per-call cost is strongly non-linear in width (tile/launch steps), or if contexts differ within a cohort. Neither appears at the measured contexts.
- The largest region with width sensitivity is Huginn at B ≥ 32 and long L, where KV traffic dominates. Even there the replay improvement is ≤ 0.23%.
- More promising levers suggested by the data:
  - reducing Core-call fixed cost (CUDA graphs or fusing the ~1,700 kernels per Ouro call), which also makes the RIC lever more valuable;
  - for Huginn, cutting KV-path cost at large B·L.
