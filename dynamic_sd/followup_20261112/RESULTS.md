# Follow-up results — gate cost (Stage 0a), parity (0c), oracle headroom (Stage 1)

Run: 2026-11-12 20:21–22:30 KST, henn-0 (8×H200), burners stopped 20:17 and restored per GPU on queue end (all 8 back by 22:18).
Code: this branch (`research/dynamic-sd-gate-oracle-20261112`) + `counterfactual_actions.py`, `fu_analyze.py`, `make_queues.sh`, `fu_worker.sh`, `fu_watchdog.sh` added in this run.
WaveFront pinned `4345209`. Receipts: `results_20261112/` (gzipped copy of the gitignored `dynamic_sd/results/followup_20261112/`; sha256 in `results_20261112/MANIFEST.sha256`).
Exact commands: `make_queues.sh` → `fu_q/g0..g7`; per-job logs in `results_20261112/logs/`. 0 failed jobs.

**Scope reminder.** Everything here is eager WaveFront B=1. Nothing below is a CUDA-Graph / optimized-engine measurement (Stage 0b, NVTX/Nsight, not done). FP32 runs use the SDPA varlen fallback (FlashAttention has no FP32 path), so FP32 wall times run a different attention kernel than BF16 ones.

## 0c. Quality gate — FULL KV, FP32 exact greedy parity (SD tokens == matched dynamic AR)

| model | exit | n | WFD | DtV |
|---|---|---:|---:|---:|
| Ouro-1.4B | fixed | 16 | 16/16 | 16/16 |
| Ouro-1.4B | live τ=0.4 | 32 | 32/32 | 32/32 |
| Ouro-1.4B | live τ=0.1 | 16 | 16/16 | 16/16 |
| Huginn | fixed | 16 | 16/16 | 16/16 |
| Huginn | **live ε=0.1** | **32** | **32/32** | **32/32** |
| Huginn | live ε=0.2 | 16 | 16/16 | 16/16 |

The previously missing Huginn live FULL FP32 parity (n≥8) now passes, at n=32 (ε=0.1) and n=16 (ε=0.2).

**BF16 first-mismatch diagnosis** (`fu_analyze.py parity`; receipts = 2026-11-11 main n=200 + new n=32 refs). For each SD≠AR sample: did any position before the first mismatch exit at a different depth in SD vs AR (*depth path diverged*), or was the depth path identical (*same-depth numeric flip*)?

| model / exit (BF16, main n=200) | WFD exact | DtV exact | first-mismatch kind (WFD; DtV) |
|---|---:|---:|---|
| Ouro fixed | 53/200 | 63/200 | numeric 147; 137 |
| Ouro τ=0.4 | 66/200 | 67/200 | depth 48 / numeric 86; depth 39 / numeric 94 |
| Ouro τ=0.1 | 48/200 | 49/200 | depth 89 / numeric 63; depth 48 / numeric 103 |
| Huginn fixed | 139/200 | 140/200 | numeric 61; 60 |
| Huginn ε=0.1 | 136/200 | 135/200 | depth 58 / numeric 6; depth 57 / numeric 8 |
| Huginn ε=0.2 | 142/200 | 144/200 | depth 47 / numeric 11; depth 44 / numeric 12 |

- Ouro's BF16 non-exactness is already there at fixed depth (27–32% exact). Exits add few new failures, and most first mismatches stay plain numeric flips.
- For Huginn the overall rate barely changes with exits (≈70% exact either way), but the mechanism switches. Under exits, ~90% of first mismatches start from an exit-depth flip: a near-threshold relative-L2 decision flips under BF16 and the KV/recurrence paths diverge.
- Limitation: depth is only compared at positions SD verified, so a few "numeric" cases may hide an unrecorded depth flip.

## 0a. Matched replay shadow gate (off / gpu / cpu), 5 repeats × 16 prompts, mode order randomized per repeat

All 20 repeat-triples passed `compare_shadow.compare()` (identical tokens, R-calls, rows, widths, depths). Values are paired per-prompt differences of prefill-inclusive wall, divided by generated tokens, pooled over repeats. 95% CI = bootstrap over (repeat, prompt) pairs.

| lane | arm | off ms/tok | gpu−off (gate kernels) | cpu−gpu (readback + Python) | **cpu−off (≈ native gate total)** |
|---|---|---:|---:|---:|---:|
| Ouro τ=0.4 BF16 | AR | 31.8 | −0.19 [−0.44, +0.07] (−0.5%) | +0.67 [+0.38, +0.94] (+2.1%) | **+1.6%** |
| | WFD | 19.6 | −0.23 [−0.41, −0.02] (−1.0%) | +0.48 [+0.24, +0.71] (+2.4%) | **+1.3%** |
| | DtV | 22.8 | −0.13 [−0.39, +0.11] (−0.5%) | +0.47 [+0.23, +0.70] (+2.0%) | **+1.5%** |
| Ouro τ=0.4 FP32 | AR / WFD / DtV | 27.2 / 21.4 / 21.7 | −1.0% / −0.7% / −0.4% (CIs ∋ 0 or ≈0) | +0.7% / +0.8% / +0.4% | **−0.2% / +0.2% / +0.0%** |
| Huginn ε=0.1 BF16 | AR | 41.0 | +1.88 [+1.61, +2.14] (+4.6%) | +0.82 [+0.51, +1.15] (+2.0%) | **+6.6%** |
| | WFD | 23.7 | +0.72 [+0.62, +0.81] (+3.0%) | +0.16 [+0.03, +0.29] (+0.7%) | **+3.7%** |
| | DtV | 27.5 | +1.06 [+0.88, +1.21] (+3.9%) | +0.42 [+0.21, +0.67] (+1.5%) | **+5.4%** |
| Huginn ε=0.1 FP32 | AR / WFD / DtV | 49.2 / 38.2 / 39.6 | +0.2% / +0.6% / +0.9% | +5.9% / +3.2% / +3.0% | **+6.1% / +3.7% / +3.9%** |

Host counters (cpu mode):
- Ouro: ~97–99 µs host time per gate call, of which 28–39 µs is blocking readback.
- Huginn BF16: 106 µs per call (24 µs readback).
- Huginn FP32: 666 µs per call, of which 587 µs is readback. The `.tolist()` blocks while the slower FP32 SDPA-fallback kernels drain, so in FP32 the readback absorbs GPU queue time and is not pure overhead.

Reading:
- **Ouro:** gate cost in this eager harness is ~1.5% (BF16) and ≈0 (FP32). Nearly all of it is the per-row `float(lam[i])` readback/Python, not gate kernels.
- **Huginn is not negligible:** 3.7–6.6% of decode wall in BF16. The relative-L2 gate kernels plus per-token `clone()` are ~3–5%, and the `.tolist()` sync another ~1–2%. AR pays the most, because it runs the gate on every step of a 1-row R call with E[D]≈16.
- The earlier E3 statement "live vs replay ≤3%, gate negligible" holds for Ouro but understates Huginn.
- These are eager numbers with ~10–40 ms/token baselines. In a graph-captured engine the per-step baseline shrinks while a host sync per step stays (or breaks capture), so the *fraction* should be expected to grow. Stage 0b is still required before any production claim, and this run does not falsify the gate-cost concern.

## Stage 1. Same-prefix counterfactual one-round actions (oracle headroom)

`counterfactual_actions.py` method:
- Prompts: gsm8k offset 600+, disjoint from all earlier runs. For each prompt the native AR run is the target; checkpoints fall every 16 tokens.
- At each checkpoint the KV cache is snapshotted. For each action and each of 3 repeats (randomized order), the runner restores the snapshot, runs exactly one round, and times it with cuda-synchronized timing (decode-only, no prefill).
- The committed state is the native AR-live state (live-exit depths and freeze-replicated KV), not a re-prefill. Pass-2 AR must reproduce pass 1 token-for-token: 0 restore failures in all runs.
- Actions: AR (1 token) and DtV(d,K) (1 draft-verify round). WFD has no single decision round and is excluded.

| config (FP32 unless noted) | trials | inexact | best static (vs AR) | in-sample oracle / best static | **cross-fitted oracle / best static [95% CI over prompts]** |
|---|---:|---:|---|---:|---:|
| Ouro fixed | 13,824 | 0 | DtV d1 K8 (1.85×) | 1.177 | **1.176 [1.146, 1.207]** |
| Ouro τ=0.4 | 13,824 | 0 | DtV d1 K8 (1.36×) | 1.221 | **1.220 [1.186, 1.254]** |
| Ouro τ=0.1 | 13,824 | 0 | DtV d1 K2 (1.04×) | 1.262 | **1.257 [1.225, 1.292]** |
| Ouro τ=0.4 BF16 | 13,824 | 183 (1.3%) | DtV d1 K4 (1.47×) | 1.217 | **1.214 [1.190, 1.240]** (491/512 all-exact prefixes) |
| Huginn fixed | 11,520 | 0 | DtV d8 K8 (1.86×) | 1.262 | **1.259 [1.229, 1.296]** |
| Huginn ε=0.1 | 11,520 | 0 | DtV d4 K8 (1.47×) | 1.252 | **1.251 [1.218, 1.287]** |

Cross-fitting picks each prefix's action on 2 repeats and scores it on the held-out repeat. The numbers barely move from in-sample, so timing noise / winner's curse is not what makes the headroom. `oracle_frontier.py` was also run unchanged on the all-exact subsets (`cf/*.oracle_frontier.json`) and gives identical in-sample ratios.

Observations:
1. **Static frontier under exits.**
   - Ouro: exits cut the best static DtV from 1.85× AR (fixed) to 1.36× (τ=0.4) and 1.04× (τ=0.1). This is the E[D]/d collapse from the characterization, reproduced decode-only on new prompts.
   - Huginn: the best draft depth moves from d=8 (fixed) to d=4 (ε=0.1). d=8 loses 11% vs d=4 under exits (1.31× vs 1.47×), confirming the earlier "tuned-at-fixed d is unsafe" finding with a decode-only, same-prefix measurement.
2. **Headroom exists but is not exit-specific.**
   - The one-decision oracle beats the best static action by 18–26% in every config, *including fixed depth* (Ouro 1.18, Huginn 1.26).
   - Exits enlarge the Ouro headroom modestly: 1.18 → 1.22 (τ=0.4) → 1.26 (τ=0.1). Relative to AR, the τ=0.1 oracle is ≈1.31× where static SD gives 1.04×.
   - So most of the frontier is "know K/acceptance in advance", which exists without exits. The exit-specific part is the extra few percent plus the AR-vs-SD choice.
3. **What the oracle picks.** The reference next-token depth D is a non-causal diagnostic, not a feature:
   - Ouro τ=0.1: D=1 → AR is the most frequent pick (111 prefixes, next DtV d1 K2 with 32); D≥2 → DtV d1 K8.
   - Ouro τ=0.4: AR is chosen mainly at D=4.
   - Huginn: d4 K8 dominates at every D.
   - A causal controller would need to predict "the next tokens are shallow/easy", which is exactly the depth-conditional acceptance selection effect seen in the characterization.
4. **BF16 strict target-exactness.** 1.3% of one-round trials (worst DtV d2 K8: 48/1536) already diverge from the BF16 AR reference within a single round.

**Stop rule (PLAN.md).** The frontier shows material headroom (≥1.18× over the best static d/K, CIs well above 1), and all FP32 actions pass target parity, so by the plan's rule Stage 2 is *not* stopped. This is an optimistic upper bound, though:
- The oracle has perfect foresight of greedy acceptance.
- One decision per checkpoint does not compose into a rollout.
- No controller cost is included.
- The headroom is mostly present at fixed depth too.

A Stage 2 controller should therefore be compared against fixed-depth adaptive-K baselines (adaptive K without exits), not only against static d/K.

## Not done in this run
- Stage 0b (NVTX/Nsight, CUDA events, CUDA Graph / optimized common backend, vllm-rlt port).
- Stage 2/3 (no controller or scheduler implemented).
- WFD in the counterfactual grid.
- Huginn BF16 counterfactuals.
- Multi-round rollouts.
