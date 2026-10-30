# Content → Instruction Recurrent Sensitivity and Future KV Utility
**Date:** 2026-10-30  
**Repository:** heonniy/batch-loop  
**Branch:** research/content-instruction-recurrent-sensitivity-h200x8-20261030  
**Hardware:** H200 × 8 independently assigned workers (NO tensor parallel)  
**Status:** Reference quality instrumentation, runner, and CPU/model-backed tests committed. No GPU evaluation or measured KV-memory savings yet.

## 0. Research question

**RQ:** Does the impact of Content KV on the recurrent state of a later Instruction/Question predict its *actual future decode-time KV eviction damage*, beyond conventional Attention magnitude?

This is not the claim that "a token with larger state change is more important." State changes may be harmless task computation, and attention/output changes occur in dense transformers too. The goal is to TEST:
- H1 (mechanism): Removing a Content KV while processing subsequent Instruction changes Instruction's hidden trajectory, even at similar instruction→content attention.
- H2 (error dynamics): A perturbation at ONLY the last Instruction query's Loop 1 may recover or persist through later recurrences; sustained masking across all Instruction queries/loops is a distinct intervention.
- H3 (utility): Content tokens with high prefill recurrent sensitivity cause larger **post-prefill** future generation NLL damage when evicted.
- H4 (deployability): Online-observable FULL-prefill signals (Instruction→Content loop-attention; the Content source's own state trajectory) beat age+attention alone on **held-out requests**. The counterfactual sensitivity obtained by re-running masked instruction is an expensive ORACLE diagnostic, not a free online feature.
- H5 (dynamic decode): Eviction changes native exit depths; does live-exit quality differ from frozen recorded-replay quality? EXIT FLIP alone is not evidence of harm.

**Critical causal direction:** A Content token can influence an Instruction token only when the Content precedes that Instruction in the autoregressive prompt. If Instruction precedes Content, no backward attention to future Content exists. Our NEW frozen prompt explicitly orders `Background (optional) → Content → Instruction → Answer:`. Only future Generated Token queries observe the post-prefill eviction.

## 1. Models, datasets, protocol

| Item | Frozen setting |
|---|---|
| Models | `huginn-0125` (3.5B), `ouro-1.4b`, `ouro-2.6b`, with pinned revisions in `kvp/specs.py` |
| Data | 100 disjoint questions per **model × dataset**, GSM8K + MATH-500, same question UIDs across models (600 model-question instances per decode mode) |
| GPU | H200 ×8; one B=1 reference model per worker/device; BF16 |
| Prefill | **FULL recurrent depth for every Instruction/Content token**, **token-major** for accurate causal intervention (not a throughput-optimized parallel prefill) |
| Decode | Fixed depth R OR native Dynamic Exit; no-refill; current token always computes its actual loop KV |
| Ouro dynamic | learned gate, cumulative q=0.7, Rmax=4, min=2; checkpoint-specific bias |
| Huginn dynamic | cosine threshold 0.001, Rmax=32, min=2, FP32 cosine arithmetic while backbone remains BF16; this is a diagnosed modified evaluator, not identical BF16-native |
| Target | First 64 gold answer/rationale tokens; no free generated Accuracy measured |
| Prompt | NEW role-specific prompt; Huginn chat template, Ouro plain completion; no old few-shot protocol |
| Content candidates | 2 aligned Content tokens per question; 1 Instruction and 1 unrelated Background control, with exact tokenizer offsets |
| Cache store | FULL-sized broadcast reference. Evictions are Attention read masks, not HBM reclamation |

All prompt tokens are retained. `--max-prompt-tokens 1536` is an explicit **cap that FAILS preparation on an overlong prompt**. Never silently trim the Instruction/Content region, unlike the older last-96-token probe. No use of future answers or Attention in candidate selection.

Run both fixed/dynamic decode to disentangle output sensitivity from Exit decisions, but the experiment may pilot one mode first. At `--decode-modes fixed,dynamic`, the 100-each main cohort has **1,200 jobs** (600 model-question instances × 2 decode modes); each job runs baseline, parity replay, two Content interventions and role controls. Because work is token-major, use H200 resource preflight and pilot gates.

### Role tagging

`kvp/role_data.py` pins the exact Content, Instruction, optional unrelated Background and Other token indices using tokenizer offsets. Cross-boundary subword tokens are **Other** and excluded from interventions. A slow-tokenizer fallback is allowed **only if prefix tokenization proves exact boundary alignment**, otherwise fails closed. Source identity and model revision are frozen in `jobs.json` and `run_manifest.json`.

This design avoids treating GSM8K/MATH500 old few-shot examples or Huginn system delimiters as the user Instruction. Do not compare its absolute NLL/Accuracy directly to earlier different prompt results.

## 2. Paired interventions

For each source Content token `i`, compare different causal effects.

### A: Mechanism — intervention DURING Instruction Prefill

All prior Content is computed at Full recurrence, untouched. For the Instruction/Question token(s) **after** Content:
- **Clean FULL Prefill:** query at every Loop r can attend Content i.
- **Persistent Instruction mask:** every Instruction query's recurrent Attention at all loops masks Content i; prelude/coda are not masked for this arm.
- **One-shot Loop-1 mask:** ONLY the **final Instruction query position** at **Loop 1** masks Content i (all recurrent physical layers); subsequent loops and all earlier Instruction tokens get unmodified KV reads. Thus the final anchor receives exactly one Loop-level intervention and can exhibit recurrent recovery/persistence.

Collect baseline/perturbed hidden for final Instruction query at every r, and compute:

```
E_(i,r) = ||h_instruction,r(mask i) - h_instruction,r(clean)||_2
          / (||h_instruction,r(clean)||_2 + eps)
P_i = E_(i,R) / max(E_(i,1), eps)
```

Keep signed effects via NLL as well; a large hidden norm difference need not harm predictive quality. For Huginn, annotate randomly initialized recurrent state and separately record prelude representation; do not interpret an initial-noise norm as intrinsic Content utility.

### B: Real use case — intervention AFTER full Prefill

Run exactly the same Full-depth Prefill with **no** earlier change.
After the prompt is complete:
- Freeze baseline target token sequence, model revision and decode exit depths.
- Mask Content i from ALL attention layers during **future generated-token processing** (Ouro core; Huginn prelude/core/coda).
- Compare teacher-forced future NLL under identical Recorded Decode trace.
- For dynamic decode, also run a separate native **Live Exit** under the same mask, record ExitFlip, signed depth shift and future NLL. Unlike Replay, this allows actual gate decisions to change.

```
U_i^future = NLL_future(postprefill_evict_i, recorded_depth)
             - NLL_future(clean_FULL_prefill, same recorded_depth)
```

**Important off-by-one:** The first gold answer token is predicted at the LAST prompt position, prior to any post-prefill cache mask, so it must NOT be counted in `NLL_future`. The implementation records all-target NLL too but uses targets 2..64 for the post-prefill utility label. Do not label an unchanged first-token logit as evidence of robustness.

### C: Observation-only prefill signals

On the clean Full prefill, record:
- Per-loop Instruction-query Attention **mass** to Content, earlier Instruction, unrelated Background, and self; omit future positions by causal mask.
- Normalize Content group mass by its token count; different roles have unequal lengths.
- Per-Content-source Instruction Attention profile across loops; mean/max and late-loop can be derived without future-token leakage.
- Each Content source's own clean recurrent hidden-change profile (loop-to-loop differences).
- Instruction anchor's full hidden-state trajectory.

Do NOT use the later Generated Token Attention or the counterfactual `E_(i,r)` as a zero-cost online feature.

### D: Role-specific negative controls

Also evict an earlier **Instruction** and an unrelated **Background** token after full prefill (one each) to compare role sensitivity. These controls do NOT enter the Content→Instruction mechanism test; they serve as separate role-group effects. Content/Instruction/Background sample sizes and position/age differ: report within-role distributions and match age/attention before comparing groups.

## 3. Hypothesis tests and metrics

| Output | What it answers |
|---|---|
| `instruction_attention_by_loop.csv` | Does Instruction→Content mass evolve over recurrence? Any systematic shift relative to Instruction/Background mass? |
| `content_sensitivity.csv` | For each Content source: loop-wise Instruction hidden error, single-shot final recovery, persistent final error, prefill attention, future eviction NLL, live exit shifts |
| `instruction_distractor_controls.csv` | How do Instruction/Background post-prefill KV removal effects differ? |
| `analysis/SUMMARY.json` | Spearman relation of Attention/state changes/counterfactual sensitivity to future NLL damage, fixed-vs-dynamic groups |
| `request_heldout_prediction` | Frozen SHA-based 70/30 **request** split: Attention+Age vs Attention+Age+online clean source-state change; diagnostic Oracle adds masked Instruction final error |
| `completeness.json` | All jobs, failed parity, missing/errors; none silently accepted |

Candidate tokens from same request are correlated; inferential confidence intervals must be **bootstrapped by request**, not by individual token. The current `SUMMARY.json` reports descriptive Spearman and 70/30 held-out OLS R²; rigorous request-bootstrap confidence intervals/seed sensitivity are a follow-up before paper claims.

**GO** only if (i) real Attention-normalized mismatch exists within comparable Content/age bins; (ii) counterfactual Instruction sensitivity predicts future eviction utility; and (iii) at least one **online-observable** clean-prefill recurrent feature adds request-held-out information beyond age+attention. Then and ONLY then implement actual multi-token eviction at matched KV bytes against faithful H2O/SnapKV/StreamingLLM and prompt-protected random; generation Accuracy and actual compact HBM/TPOT/TPS are further stages.

**NO-GO:** Instruction recurrent perturbations mostly vanish, or are unrelated to future damage, or offline Attention/recency already explains all signal. A counterfactual-only oracle improvement with no affordable predictor does **not** establish a practical eviction method.

**INCONCLUSIVE:** dynamic Ouro-2.6B all exits R4, short gold horizon yields almost zero meaningful damage, or native model parity fails. Then use separately frozen q and/or longer target on new data, never silently retune and reuse heldout questions.

## 4. H200 × 8 execution

Use a **new source branch** and a **new run directory for every frozen config/source SHA**. Existing 10/30 exit-depth experiments and FERRY are untouched.

```bash
git fetch origin
git checkout research/content-instruction-recurrent-sensitivity-h200x8-20261030
cd kv_policy

# CPU tensor/unit + tiny HF-backed model parity (must pass).
python -m pytest -q tests/test_role_sensitivity.py tests/test_role_model_parity.py

# Smoke: 1 question/cell, fixed+dynamic, 3 models, 2 datasets.
python scripts/role_sensitivity_sweep.py --prepare \
  --run-dir results/role_smoke_20261030 --n-examples 1 \
  --n-content 1 --n-instruction 1 --n-distractor 1 \
  --max-target-tokens 16
python scripts/role_sensitivity_sweep.py --execute \
  --run-dir results/role_smoke_20261030 --gpu-ids 0
python scripts/role_sensitivity_sweep.py --analyze \
  --run-dir results/role_smoke_20261030

# Pilot: 8 questions per model/dataset, fixed+dynamic,
# new directory; opt-in H200 x8 after smoke.
python scripts/role_sensitivity_sweep.py --prepare \
  --run-dir results/role_pilot8_20261030 --n-examples 8 \
  --n-content 2 --max-target-tokens 64

# Main: 100 per model/dataset; if both decode modes, 1200 jobs total.
# Recommended after pilot quality, duration/peak-HBM and role-label checks.
python scripts/role_sensitivity_sweep.py --prepare \
  --run-dir results/role_main100_20261030 --n-examples 100 \
  --n-content 2 --max-target-tokens 64 \
  --decode-modes fixed,dynamic
python scripts/role_sensitivity_sweep.py --execute \
  --run-dir results/role_main100_20261030 \
  --gpu-ids 0,1,2,3,4,5,6,7
python scripts/role_sensitivity_sweep.py --analyze \
  --run-dir results/role_main100_20261030
```

GPU is NEVER launched by `--prepare` or `--analyze`. Only `--execute` starts model workers. Inspect host GPU ownership and model availability before executing. Atomic output rename and exclusive job claims protect results; failed claims require manual audit. Source code SHA and model revision mismatch aborts resumed runs.

**Technical gates before promotion:**
- Huginn & Ouro tiny-model Full-prefill R-only path matches independent existing Full/strict at fixed depth;
- Huginn FP32 cosine threshold is documented as a deviation from model-dtype native evaluator;
- clean recorded replay has max absolute per-token NLL difference <=1e-4;
- role labels exact and causal; never truncate prompt; role controls present;
- first gold token excluded from postprefill NLL utility;
- single-loop anchor perturbation has NO earlier-Instruction / later-loop reinjection;
- error-free jobs, truthful zero/negative sensitivities, nondegenerate future NLL variance.

**Not yet implemented:** physical compact eviction, throughput measurement, faithful external eviction-policy port, free-generation Accuracy, multi-source simultaneous eviction. No H200 run is claimed by this GitHub-only operation.
