# Huginn-0125 R32 — HF-only quality rerun

## Scope
- Base: research/kv-policy-h200x8-20261022; new branch isolated from prior results.
- **No vllm-rlt. No early exit. No Ouro.** Huginn-0125 BF16, fixed R=32 for both prompt and generated tokens.
- Same frozen 200 GSM8K and 200 MATH-500 items, prompts, parsers and gold-target NLL used previously.
- Policies: FULL (baseline), LAST-convert, AVERAGE-convert (secondary). Prefill computed using per-loop FULL, then prompt cache compressed to final/mean loop state; decode reads this cache. The transient prefill memory is not claimed as saved.
- Generation cap GSM8K **512** (was 256); MATH-500 **2048** (was 1024). These larger caps are not official benchmark defaults; compare with previous length truncation and document comparability limitations. Same stop strings and scorer across policies.
- Test Full/native parity and absence of truncation before interpreting a Full–Last accuracy gap. No claim of official MATH accuracy because substitute prompt remains.
- Main metrics: Accuracy and token-weighted NLL; supporting: fraction truncated, finish reason, parse failures, mean generated length, pairwise FULL-only/LAST-only correct. Do not expand metrics into new sweeping performance study.
- Lower microbatch for H200 memory at R32: GSM8K B8, MATH500 B4. 8 independent workers; no TP.
- Raw data plus run manifests should be committed only after actual H200 execution and completeness checks. Do not edit historical results.

## Run (on authorized H200 8-GPU host)
```bash
git checkout research/huginn-r32-length-h200x8-20261023
cd kv_policy
PYTHON=~/ddn/envs/kvp/bin/python bash scripts/run_huginn_r32_h200x8.sh
```
Script launches 8 workers, runs policy x task x dataset and writes length_audit.md. Verify worker failures and count n=200 for all six generation cells and six NLL cells before reporting success.

## Known issues to report
- Huginn latent initial noise is keyed by UID/position in the reference engine; confirm parity with native model. Record any BF16 deviations.
- R16 baseline is not directly comparable to R32 for absolute accuracy.
- Old MATH-500 Full had 41/200 sequences hitting 1024 tokens; larger cap can affect accuracy. Do not claim every longer completion is a rescued former failure.
- This update is code and a run script only. **No H200 job has been run from this chat.**
