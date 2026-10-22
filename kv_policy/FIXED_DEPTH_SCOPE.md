# Fixed-depth-only scope (2026-10-22)

This scope supersedes the dynamic-depth stages of EXPERIMENT_PLAN_H200X8.md for the first experiment. No learned/live/recorded early exit, Ragged KV, or token-adaptive Ours in this phase.

Models: Ouro-1.4B, Ouro-2.6B and Huginn-0125. Run fixed R=2/4/8 where checkpoint/runtime supports it; explicitly record skipped unsupported depths. Policies: FULL, LAST, AVERAGE, with read/write semantics and current-token/prefill causality tests before accuracy claims. AVERAGE is the mean of K and V over recurrent iterations, not mean of attention outputs. Record source paper correspondence. Validate reference logits and greedy generation in Hugging Face before porting to vllm-rlt. Do not assume SHARED and LAST are identical without parity.

H200 x8: eight independent single-GPU workers, rotating model/policy placements to prevent GPU-id confounds. 8 GPUs authorized on the distinct H200 server only, not the older shared H100 server. GPU execution requires actual access; never terminate other users' processes.

Metrics: accuracy GSM8K/MATH500, general NLL, prefill/decode output parity, active physical KV bytes, reserved HBM, peak GPU use, TTFT, TPOT, throughput, batch-size/context sweeps, raw trials (>=3), warmup and hardware/software provenance. Include accuracy–memory and throughput–memory comparisons. No fabricated results.

Order: (0) semantics/unit tests → (1) HF correctness/quality → (2) vllm-rlt measured performance → (3) synthesis. Next decision whether to pursue adaptive retention after fixed-policy results.
