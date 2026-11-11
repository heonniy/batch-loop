# Coding-agent handoff — Dynamic-depth SSD, Ouro + Huginn

**Status: execution specification only. DO NOT describe any step as already executed.**

1. **Clone/pin independent public baselines** on an authorized GPU host: `summerbro-hhj/wavefront-decoding` (both models), `kaist-flexml-lab/loopspec` (Ouro official only), `InferMatrixLab/vllm-rlt` tag `v0.2.0` (later serving integration). Log exact SHAs and model/checkpoint revisions. No modification or cloning of `casys-kaist-internal/loopserving.code`; read-only inspection if permission eventually exists.
2. **Start BOTH model lanes together**, not "finish Ouro then consider Huginn":
   - Ouro-1.4B: native R=4; `d={1,2}`; full KV; calibrate q for shallow/mixed/deep realizations.
   - Huginn-0125: native R=32; `d={2,4,8}`; full KV; check native FP32 cosine against WaveFront relative-L2 `EXIT_HIDDEN_EPS`. Never assume they are numerically identical.
3. **Smoke only 32 prompts per model/dataset** (GSM8K and MATH-500 initially) in author WaveFront. First reproduce `AR-fixed` vs `WFD-fixed` vs `DtV-fixed`; then `AR-live` vs `WFD-live` vs `DtV-live` with the exact same exit/KV settings. Keep comparable generated output length, trace seeds and repeat counts. Reject equality claims if the native dynamic AR reference is not reproduced.
4. **Representative upstream commands** (substitute calibrated Huginn epsilon rather than inventing one):
   ```bash
   CUDA_VISIBLE_DEVICES=0 MODEL=ouro-1.4b DATASET=gsm8k LIMIT=32 T_TOTAL=4 T_DRAFT=1 KV_BUDGET_S=0 DECODE=both bash experiments/exp3_accuracy/run.sh
   CUDA_VISIBLE_DEVICES=0 MODEL=ouro-1.4b DATASET=gsm8k LIMIT=32 T_TOTAL=4 T_DRAFT=1 KV_BUDGET_S=0 EXIT_THRESHOLD=0.2 DECODE=both bash experiments/exp3_accuracy/run.sh
   CUDA_VISIBLE_DEVICES=1 MODEL=huginn-3.5b DATASET=gsm8k LIMIT=32 T_TOTAL=32 T_DRAFT=4 KV_BUDGET_S=0 DECODE=both bash experiments/exp3_accuracy/run.sh
   # After measuring and choosing a non-degenerate epsilon:
   CUDA_VISIBLE_DEVICES=1 MODEL=huginn-3.5b DATASET=gsm8k LIMIT=32 T_TOTAL=32 T_DRAFT=4 KV_BUDGET_S=0 EXIT_HIDDEN_EPS=<calibrated_epsilon> DECODE=both bash experiments/exp3_accuracy/run.sh
   ```
   These are examples of an upstream invocation, not commands that have been executed or known to produce a working parity baseline in this session.
5. **Instrument** (without changing the algorithm first): per-token native depth, draft depth, accepted/rejected and verifier commit status, gate, trace replay, actual recurrent-row count, rollback KV frontier, padding, memory, wall time, model output. Distinguish true dynamic exits from model-replay traces and lossless fixed-target behavior.
6. **Parallel follow-up**: Ouro LoopSpec fixed-depth reproduction; Huginn WaveFront fixed-depth reproduction. Only then adapt LoopSpec's verifier for live halting (Ouro first, Huginn only as a separately validated port) and compare to WaveFront in a shared per-model workload. Pin all attention/backend differences and refrain from cross-engine absolute speedup claims.
7. **KV arms after full-KV parity**: Huginn last-computed; Ouro first-protected+last-computed; FULL for both. Do not treat `kv_budget_s=1` as first+last, and do not call the 1+last policy proven accuracy-preserving.
8. **Stage gates**: E0 parity and fidelity first; E1 physical observation; E2 trace statistics; E3 padded-vs-compacted cost; E4 KV axes; E5 stress; E6 new policy only on demonstrated headroom. Keep non-results and failures, with `RESULT_REVIEW.md` and `review_manifest.json`.
9. **Do not run GPU on this ChatGPT/container runtime**: it has no NVIDIA device. Do not use a GPU or modify unrelated remote processes without a proper execution host/session. Read and honor the repo's historical BF16 parity-failure scope; new study is not retroactive evidence for those experiments.

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for all controls, hypotheses and falsifiers.
