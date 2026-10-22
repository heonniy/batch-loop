# KV policy characterization — H200 x8 (2026-10-22)

## Goal
Reproduce FULL / LAST / AVERAGE recurrent-KV policies in Ouro and Huginn; measure output quality, KV residency, and real inference performance before any learned retention policy is introduced.

## Source and isolation
- Repository: heonniy/batch-loop, branch research/kv-policy-h200x8-20261022.
- Do not mutate prior phase results or change existing phase04 research contracts. New work stays under kv_policy/.
- Environment: a distinct user-authorized H200 8-GPU server. Existing AGENTS.md restricts old GPU experiments and disallows running on old shared GPUs; do not interpret this plan as permission to run on those older GPUs. Never kill unrelated processes.
- No GPU experiments have been performed by this commit.

## Definition of policy (critical)
- FULL: persistent independent KV per (physical recurrent layer, loop index, historical token).
- LAST: a single retained historical KV representing the last *computed* loop; clarify prefill/decode timing and whether same-token current KV is handled separately. During fixed-depth decoding all recurrent queries access the same retained historical last-loop cache.
- AVERAGE: a single retained historical KV built by averaging corresponding K and V over computed loops. Explicitly document when this average becomes available and how updates avoid seeing future loop states; a two-pass or deferred-write reference implementation is acceptable, but any extra execution is charged in performance results.
- Prelude/coda caches are maintained separately for Huginn. Never count KV units without accounting for recurrent physical layers, K and V, dtype and GQA.
- Do not equate vllm-rlt LAST-EXITED or SHARED with paper LAST without read/write parity tests.
- If policy semantics conflict with a model's training or original paper, report the difference; never silently invent agreement.

## Stage 0: reproducibility and support matrix
- Record exact model revisions, transformers/PyTorch/vllm-rlt commits, dtype, attention backend, hardware SKU, CUDA version.
- Inspect native Huginn and Ouro generation code and upstream paper's LAST/AVERAGE definitions, then write explicit pseudocode and a support table of read/write policies.
- Run deterministic 1-3 token toy parity cases, inspect each loop's K/V and cross-token attention reads (FULL, LAST, AVERAGE). Verify average of K and V independently, correct position indices, no prefill/decode leakage.
- Unit-test KV memory footprint and policy switching, then greedy generation. Stop and report if parity fails.

## Stage 1: HF correctness/quality reference
- Models: ByteDance/Ouro-1.4B, ByteDance/Ouro-2.6B, tomg-group-umd/huginn-0125 (confirm model identifiers before download).
- Fixed depths: R=2,4,8 only where native checkpoints support them; if out of supported range skip and report.
- Policies: FULL/LAST/AVERAGE; optional FIRST as sanity negative control.
- Data: GSM8K, MATH-500 plus held-out general-language NLL and long-context recall (evaluate model applicability). Fixed prompts, sampling greedy, reproducible dataset hashes.
- Compare token-by-token logits/NLL, task accuracy, generation divergence, model execution count. Treat learned early exits separately.
- Baseline FULL parity against native HF generation required before performance study.

## Stage 2: vllm-rlt real serving benchmark
- Prefer vllm-rlt for optimized serving ONLY after correctness; the current upstream supports Ouro fixed/learned exit and Huginn synchronous fixed depth; confirm via pinned commit.
- Implement explicit FULL/LAST/AVERAGE cache policy mapping with both logical and physical accounting. AVERAGE needs update machinery, cannot be modeled as LAST via config toggle.
- Maintain a synchronized reference mode; disable speculative decode, CUDA graphs, continuous refill, and mixed policies initially. Match outputs with HF (precision tolerances documented).
- Measure allocated/active physical KV blocks, pool reserved, max_memory_allocated/max_memory_reserved, nvidia-smi process use, attention traffic when possible, TTFT, TPOT, E2E, tokens/s and peak supported batch.
- Sweep input length 512/2048/8192/32768 within model context limits; generated 64/128, batch 1/8/32/64/128 as memory allows; prefill and decode split.
- Warmup and >=3 repeats; raw per-trial logs, no cherry-picking, note OOM.

## H200 x8 execution
- Use all eight H200 GPUs as eight independent 1-GPU workers initially, NOT tensor parallel for a ~1–4B model. Each worker pins CUDA_VISIBLE_DEVICES to one device and receives independent seeds/config shards.
- Example allocation: GPU0 Ouro1.4 FULL, GPU1 Ouro1.4 LAST, GPU2 Ouro1.4 AVERAGE; GPU3 Ouro2.6 FULL, GPU4 Ouro2.6 LAST, GPU5 Ouro2.6 AVERAGE; GPU6 Huginn FULL, GPU7 Huginn LAST, then rotate GPU6/7 for Huginn AVERAGE and swap policy↔GPU assignments to control for device variance. All 8 GPUs utilized during sweeps.
- Rotate assignment across GPU IDs, randomize run ordering, log power/utilization, and ensure results are not confounded by concurrently measured GPU interference. Parallel mode for throughput campaign; isolate a subset for clean single-job latency comparison when needed, with remaining GPUs on non-conflicting quality jobs.
- Do not assume host can be reached through GitHub tools. Run only when H200 access actually exists.

## Stage 3: dynamic depth (after fixed policy validation)
- Freeze/replay per-token exit traces, use trained model's actual exit gate; do NOT allow policy to change exit depth during controlled replay.
- Baselines FULL/RAGGED best-available/LAST-AVAILABLE; define AVERAGE over only computed depths. Compare correctness under identical exit histories, then live adaptive exit separately.

## Output
- kv_policy/RESULTS.md, raw per-run JSONL/CSV, code, README, reproducibility manifests, hashes, correctness proofs/tests.
- Publish explicit distinction between **logical** KV entries, **active physical** KV pages and **reserved GPU HBM**.
- Success means verifiable policy semantics and quality-memory curves, not a predeclared speedup.
