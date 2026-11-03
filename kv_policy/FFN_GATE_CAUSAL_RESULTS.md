# v9 results — Context-KV eviction → recurrent FFN gate → dynamic exit / answer (pilot, 2026-11-03)

Branch research/kv-ffn-gate-depth-causal-20261103 (parent v8b 6fac4edb). Plan: FFN_GATE_DEPTH_CAUSAL_PLAN.md. Code:
kvp/ffn_gate_causal.py, scripts/ffn_gate_causal_sweep.py, scripts/ffn_gate_causal_cross.py (pooling),
scripts/std_ffn_gate_control.py (standard-transformer control, added in this session). CPU tests: 7 (v9) + 12 (v8 runner) pass.
Runs (all complete, 0 errors): results/20261103T143000Z_v9_{huginn-0125,ouro-2.6b,ouro-1.4b}_{controlled_retrieval,hotpotqa_distractor}
(3 clean-only calibration UIDs + 5 evaluation UIDs each, width-4 windows: 4 label-blind + 1 evidence + 1 distractor per prompt),
pooled in results/20261103T143000Z_v9_cross/V9_CROSS_FINDINGS.md; standard control results/20261103T145000Z_v9std_qwen_{controlled_retrieval,
hotpotqa_distractor} (same 5 evaluation UIDs, same windows, Qwen2.5-3B, layer quarters as the analogue of loop groups);
smoke/diagnostics results/20261103T_v9_smoke_ouro26_cr, scripts/diag/v9_*.py. B=1, BF16, full-sized reference KV; no deployment claim.

## Validation receipts
- Clean-branch restoration parity: restoring any FFN activation (gate / up / shifted gate / whole FFN) with the clean trace on the
  clean branch reproduces the logits exactly (max |Δlogp| = 0) on both Ouro checkpoints and Huginn (checked per prompt in the sweep).
- Evicted-branch self-reference parity: restoring the evicted branch with its OWN trace is exact (0) → hook keys (column, loop, layer,
  component) and call counting are correct (scripts/diag/v9_evict_restore_probe2.py).
- First generated token invariant under every post-prefill arm (asserted per prompt).
- Thresholds were chosen on 3 disjoint clean UIDs before any eviction outcome. Ouro: q=0.85/0.70 (1.4B: 3–16% / 53–73% early;
  2.6B: 15–37% / 42–74%). Huginn: the fp32-cosine grid saturates — even the smallest value (1e-4) lets 89–100% of clean queries exit
  early (depths 13–28), so the 25%/60% targets were NOT met; the exit arm is reported as a degenerate-but-variable diagnostic.

## 1. Eviction penalty (gold future NLL, LAST-plane-sharing cache)
| model | CR main (20 windows) | CR evidence (5) | Hotpot main (20) | Hotpot evidence (5) |
|---|---|---|---|---|
| Huginn | 0.003 [−0.002, 0.007] (floor) | 0.34 [0.007, 0.69] | −0.009 (floor) | −0.02 (floor) |
| Ouro-2.6B | −0.017 (floor) | 0.08 [−0.69, 0.90] | 0.034 [−0.001, 0.094] | 0.019 |
| Ouro-1.4B | 0.123 [−0.04, 0.33] | 0.125 | 0.026 | 0.049 |
| Qwen2.5-3B (standard) | 0.133 | 0.327 | 0.010 | −0.002 |
Width-4 windows chosen label-blind are mostly at the BF16 floor in Huginn and Ouro-2.6B; only evidence windows (and Ouro-1.4B, whose
last-plane cache is itself lossy on CR) produce a penalty worth repairing. Interpretation of repairs below is restricted accordingly.

## 2. H1 — does restoring the clean FFN gate repair the damage, beyond up / shifted-gate / whole-FFN controls?
NLL recovered on the evicted branch (positive = helps), mean over windows:
| model / cell | penalty | gate | up | shifted gate | whole FFN | best loop/layer group |
|---|---|---|---|---|---|---|
| Qwen CR main | 0.133 | **0.122 (92%)** | −0.81 | −0.013 | 0.094 | layers 28–36: 0.112 |
| Qwen CR evidence | 0.327 | **0.336 (103%)** | −0.87 | 0.013 | 0.336 | layers 28–36: 0.30 |
| Qwen Hotpot main | 0.010 | 0.014 | 0.018 | −0.000 | 0.012 | 28–36: 0.008 |
| Ouro-1.4B CR main | 0.123 | 0.093 (75%) [−0.08, 0.32] | −8.3 | −1.5 | −5.4 | loops 3–4: 0.148 [0.02, 0.35] |
| Ouro-1.4B CR evidence | 0.125 | 0.032 (25%) | −8.2 | −0.97 | −5.6 | loops 3–4: 0.035 |
| Ouro-1.4B Hotpot main | 0.026 | 0.031 [−0.05, 0.13] | −11.4 | −3.0 | −8.8 | loops 1–2: 0.027 |
| Ouro-2.6B CR evidence | 0.081 | **−0.33 (hurts)** [−0.81, 0.002] | −21.6 | −0.20 | −15.3 | loops 1–2: −0.15; 3–4: −0.27 |
| Ouro-2.6B Hotpot main | 0.034 | 0.027 [−0.01, 0.10] | −18.1 | 0.019 | −18.6 | loops 1–2: 0.037 |
| Huginn CR evidence | 0.337 | **−0.39 (hurts)** [−0.99, 0.01] | −0.37 | −0.048 | −0.057 | loops 25–32: −0.38 (the harm); 1–24: ≈0 |
| Huginn CR main | 0.003 | 0.002 (floor) | −0.069 | 0.001 | 0.002 | — |

Reading:
- In the **standard transformer** the planned signature is present and clean: gate-only restoration recovers ~100% of the eviction
  penalty, the whole-FFN restoration recovers the same amount, the direction-matched shifted-gate control recovers nothing, and the
  up-only restoration is destructive. The repair is concentrated in the last layer quarter (28–36).
- In the **looped models** the same patch is unreliable or harmful. Where the penalty is at the floor (most label-blind windows) there is
  nothing to repair. Where it is large — evidence windows in Huginn and Ouro-2.6B — the gate patch *increases* the damage (−0.39, −0.33),
  and in Huginn the harm comes entirely from patching the last loops (25–32). Only Ouro-1.4B shows a partial repair (75% on CR, CI
  crossing 0), again from the late loops (3–4).
- Whole-FFN / up-branch restoration **compounds** in the looped models: the relative patch norm grows from ~1% at the first site to
  1–5 (Ouro) over the 192 (Ouro) / 128 (Huginn) patched sites of one token, producing +16…+22 nats (Ouro) — a positive-feedback
  mismatch between the restored content path and the evicted residual stream. Gate-only patches stay at 1–3% in every model, and the
  same whole-FFN patch is benign in Qwen (+0.09, 36 sites). This instability is a recurrence effect (the same four Huginn / 48 Ouro
  layers are re-entered 32 / 4 times with the patched state), not a hook bug (self-reference parity is exact).

## 3. H2 — loop-resolved gate response and loop-local restoration
- Eviction changes the recurrent pre-SiLU gate progressively across loops and then plateaus: Huginn evidence windows 0.051 (r=1) →
  0.13 (r=4) → 0.14 (r≥8); Ouro-2.6B evidence 0.06 → 0.22 (r=4); label-blind windows ~0.01 (Huginn) / 0.03–0.10 (Ouro). Sign-flip
  fractions 0.3–7%.
- Restricted-loop gate restoration: the late loops carry the whole effect (Huginn 25–32; Ouro 3–4), whether that effect is a repair
  (Ouro-1.4B) or harm (Huginn, Ouro-2.6B evidence). The standard model shows the same "late stage" concentration (layers 28–36). The
  loop index therefore behaves like depth, not like a loop-specific mechanism.

## 4. H3 — native dynamic exit under eviction (frozen clean-only thresholds)
- Depth changes: Huginn 24–71% of generated queries change exit depth after a single 4-token eviction (evidence windows: Σdepth
  +12 to +13, i.e. deeper); Ouro-1.4B 2–7%; Ouro-2.6B 0–4%. Live-exit vs recorded-replay future NLL are within 0.01 of each other in
  every cell (Huginn CR evidence: live 0.328 vs replay 0.336) → the exit response does not change quality.
- Gate-restored recorded replay does not help where damage exists (Huginn CR evidence −0.38; Ouro-2.6B CR evidence −0.37 at q=0.95,
  +0.11 at q=0.85 — threshold-dependent, i.e. the "cherry-pick" case the plan says to reject). The off-policy gate-restored *live* oracle
  changes 37–67% of Huginn exit decisions and 0–15% of Ouro's, again without a consistent NLL gain.
- Answers: greedy answers under live exit change only for evidence windows (Huginn CR 40% of windows; EM 1.0 → 0.6), i.e. where the
  fixed-depth answer also changes; live-vs-fixed correctness flips in Ouro-2.6B at q=0.85 (80–90% of windows) are caused by the
  threshold itself (FULL-live EM 0.2 vs FULL-fixed 0.8), not by eviction.

## 5. Verdict (plan §Stop/go)
- Gate ≠ up/shift, so the patch is not "generic FFN noise" — but the gate repair is **robust only in the standard transformer**;
  in the looped models it is absent (floor), partial (Ouro-1.4B, CI crossing 0) or harmful (Huginn, Ouro-2.6B evidence), and the
  loop-local effect sits in the late loops exactly as the layer-local effect sits in the late layers of Qwen. → **No loop-specific
  FFN-gate mechanism; NO-GO** for the MoEZip-style claim.
- Exit depth responds to eviction (strongly in Huginn) without a quality effect, and gate restoration changes exit decisions without
  consistent benefit → exit-system observation only, consistent with v7.
- New looped-specific observation worth keeping: **activation restoration is unstable under recurrence** — whole-FFN/up patches that are
  benign in a standard transformer compound to catastrophic loss in Ouro and are harmful at the last loops of Huginn. Any
  activation-level "repair" of an evicted cache in a looped model must be gate-only and early-loop, and even then it is not reliable.
- Limitations: 5 prompts per cell, 4-token windows (most at the floor), Huginn exit calibration saturated, LAST-plane sharing baseline
  for all arms (v8: lossless for prompt tokens), no held-out confirmation, no deployment measurement.
