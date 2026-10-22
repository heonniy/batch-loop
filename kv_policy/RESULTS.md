# Fixed-depth recurrent-KV policies (FULL / LAST / AVERAGE) — Stages 1–3 on H200 x8

Scope: `FIXED_DEPTH_SCOPE.md` (fixed native depth, early exit OFF, GSM8K-200 + MATH-500-200, Stages 1–3).
Generated from receipts by `scripts/render_results.py` (tables) — prose written by the executing agent.
Run directories:

- `results/20261022T150208Z_d7a92d0_stage1_correctness`
- `results/20261022T150648Z_13f1e87_stage2_quality`
- `results/20261022T151156Z_cefd535_stage3_serving`

## 한국어 요약

- **구현/정합성(Stage 1)**: Ouro-1.4B/2.6B(4 UT steps)와 Huginn-0125(16 steps)에 대해 FULL/LAST/AVERAGE 정책을 HF 가중치 위의 참조 엔진으로 구현했다. fp32에서 FULL은 원본 HF forward와 logit 최대 오차 ≤2.3e-3, LAST/AVERAGE는 원본 forward + 정의 기반 dictionary 캐시(독립 참조 구현)와 ≤4.2e-5로 일치. CPU fp64 의미론 테스트 117개 통과(쓰기 시점, 미래 누설 없음, K/V 별도 평균, left-padding 배치 = B1, compact 저장소 = broadcast).
- **정책 정의 문제**: 논문(Ouro §5.4.2)은 LAST/AVERAGE를 "prefill에는 4개 캐시 필요, decoding에서만 재사용, 메모리 4×절감"이라고만 기술한다. 4× 절감과 계획서 정의(모든 recurrent query가 같은 보존 캐시를 읽음)에 부합하는 **convert**(루프별 prefill 후 prompt KV까지 정책 버전으로 변환)를 주 정의로, 프롬프트 KV를 루프별로 유지하는 **decode**(하이브리드)와 프롬프트까지 토큰 단위로 정책을 적용하는 **strict**(vllm-rlt `shared`와 fp32에서 8/8 동일)를 함께 보고한다.
- **품질(Stage 2)**: Ouro는 정책 해석과 무관하게 FULL 대비 손실이 크다. convert: Ouro-1.4B GSM8K 0.820→0.000(LAST)/0.230(AVG), MATH 0.710→0.000/0.025; Ouro-2.6B GSM8K 0.875→0.760/0.195, MATH 0.565→0.030/0.040. decode(하이브리드)에서도 Ouro-1.4B는 GSM8K −13.5%p, MATH −55%p이며 Ouro-2.6B만 −3.5~−6.5%p(CI가 0 근처). **논문 Table 14의 "LAST/AVG ≈ FULL"은 어떤 해석에서도 재현되지 않았다.** Huginn은 모든 정책에서 FULL과 유의한 차이 없음(GSM8K AVG decode만 −7%p, p=0.03).
- **원인 분해**: teacher-forced NLL 증가는 작고(decode: +0.01~0.07 nats/token) 생성 위치가 뒤로 갈수록 커지며, 실패의 대부분은 줄바꿈 반복 퇴화 → 토큰 한도 도달이다. GSM8K는 정답을 쓴 뒤 퇴화하는 경우가 다수, MATH는 답 이전 퇴화가 다수.
- **수치 대조군**: 동일 FULL을 다른 배치 구성/순차 prefill로 실행해도 Δacc는 ±2.5%p 이내, ΔNLL ≤1e-3 — 위 정책 효과는 bf16 노이즈 바닥보다 훨씬 크다.
- **서빙(Stage 3, vllm-rlt 4a35034)**: last_exited=FULL(fp32 8/8), shared=strict LAST(fp32 8/8, 논문식 LAST와는 0/8). KV 블록은 정확히 4×(Ouro) 차이; TPOT는 layout 간 차이 없음(eager sync 디코드는 launch-bound), shared는 prompt 위치별 순차 prefill 때문에 TTFT가 수십 배 증가. AVERAGE·convert·decode-LAST는 vllm-rlt 미구현 → HF compact 저장소로 실제 메모리 측정.
- **발견한 재현성 문제**: (1) 고정한 Ouro 리비전의 원본 `UniversalTransformerCache`는 README 권장 transformers 4.54.1에서 생성 불가(`key_cache` property 충돌); (2) vllm-rlt는 Huginn prefill을 config 깊이(32)로 실행 → 깊이 정렬 후 재측정; (3) Ouro MATH500 공식 프롬프트는 비공개(in-house) → 대체 프롬프트, 논문 수치와 직접 비교 불가.

## 1. Execution context and scope decisions

- Host: 8x NVIDIA H200 (driver 580.126.16), user-space only (no root): tmux, gcc 13 (Triton JIT), nvitop installed via conda-forge.
- `AGENTS.md` (Phase 04R) forbids GPU execution and GSM8K studies, but its scope is the older shared GPU machine and the Phase-04 contract. `kv_policy/EXPERIMENT_PLAN_H200X8.md` / `FIXED_DEPTH_SCOPE.md` on this branch authorise this study on this H200 host, and the user explicitly requested Stages 1–3 here. No Phase-0x result or contract was modified; all new work is under `kv_policy/`.
- Pinned: Ouro-1.4B `574fa66c…`, Ouro-2.6B `1ed04250…` (current HEADs; earlier phases used 1.4B `37bc7e8f…`), huginn-0125 `bb6621b6…`; torch 2.6.0+cu124, transformers 4.54.1, lm_eval 0.4.9.1, math-verify 0.9.0; vllm-rlt `4a350344027960a48384626aa646f1765fda6d4f`. Full environment in each `run_manifest.json` / job meta.
- Fixed depth: Ouro `total_ut_steps=4`, step-4 logits (the checkpoint's own `exit_threshold`/weighted-exit paths are bypassed; its training-time NLL uses expected logits — not used here). Huginn `num_steps=16` = scope's pragmatic reference; the checkpoint's `mean_recurrence` is 32 and the authors' GSM8K command uses 32, so Huginn scores are **not** comparable to its published numbers.
- Huginn initial latent state is random by design; every item uses a deterministic per-(uid, position) truncated-normal state, identical across policies, modes and batch shapes.

## 2. Policy semantics (what exactly was compared)

For a recurrent physical layer, the query of the current token t at loop r attends to its **own** loop-r K/V (all policies) and to history i < t:

| region | FULL | LAST | AVERAGE |
|---|---|---|---|
| history in policy region | K/V of loop r | K/V of the final loop | separate means of K and of V over all loops (fp32 accumulation) |

Policy region by mode (all three implemented in the engine and in the independent reference):

- **convert** — prompt prefilled in parallel with per-loop K/V exactly as FULL (prompt logits identical to FULL), then the prompt K/V is replaced by its policy version; decoding reads policy versions only. Retained KV = 1 plane per recurrent layer (Ouro: 4x smaller). Matches the paper's 4.00x memory claim and the plan's "all recurrent queries access the same retained cache"; **primary reading of the paper**.
- **decode** — only generated / teacher-forced tokens use the policy; prompt keeps all loop versions (hybrid; retained KV = R·P + G planes·tokens).
- **strict** — the prompt is processed token-by-token so prompt tokens also read policy versions (paper: "reuse during prefill degrades >10 points"); identical to vllm-rlt `shared`.

Huginn prelude/coda layers are non-recurrent and always keep ordinary single-version caches. The engine stores every policy in one broadcast layout (same SDPA kernel and shapes for all policies); `CompactPolicyKV` (policy-sized) is verified equal to it and used for memory measurement.

## 3. Stage 1 — correctness


## huginn-0125  (GPU NVIDIA H200 9d1efd41-3a8a-196e-690e-4e0463a802e9, 506s)

| check | cases | max abs logit diff | min top-1 agreement | max KL |
|---|---|---|---|---|
| FULL engine vs native HF, fp32 (decode) | 4 | 0.000118 | 1.0000 | 3.24e-07 |
| FULL engine vs native HF, fp32 (strict) | 4 | 0.000105 | 1.0000 | 4.01e-07 |
| FULL engine vs native HF, bf16 (decode) | 4 | 0.375 | 0.9930 | 0.00765 |
| FULL engine vs native HF, bf16 (strict) | 4 | 0.375 | 0.9930 | 0.0117 |
| policy engine vs dict-cache reference, fp32 (decode) | 6 | 6.56e-06 | 1.0000 | 1.08e-07 |
| policy engine vs dict-cache reference, fp32 (strict) | 3 | 1.05e-05 | 1.0000 | 2.14e-07 |

Policy rows of the reference check (fp32): full/decode/short_prompt=0.0e+00, full/strict/short_prompt=0.0e+00, last/decode/short_prompt=0.0e+00, last/strict/short_prompt=0.0e+00, average/decode/short_prompt=6.6e-06, average/strict/short_prompt=1.0e-05, full/decode/benchmark_prompt=0.0e+00, last/decode/benchmark_prompt=0.0e+00, average/decode/benchmark_prompt=5.8e-06

Greedy 48-token generation, bf16 (first divergence index, None = identical):
- gsm8k-test-0005: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None
- gsm8k-test-0010: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None
- test/algebra/1349.json: native vs engine B1 = 10, native vs engine B4 = 10, engine B1 vs B4 = None
- test/prealgebra/1139.json: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None

bf16 batch-shape sensitivity (B=4 vs B=1, teacher-forced 48 gold tokens): full: max |Δlogprob| 0.192, max |Δsum| 0.556, last: max |Δlogprob| 0.119, max |Δsum| 0.150, average: max |Δlogprob| 0.115, max |Δsum| 0.126

## ouro-1.4b  (GPU NVIDIA H200 6b56b49e-2472-9e55-7d0a-74a087b4132e, 383s)

| check | cases | max abs logit diff | min top-1 agreement | max KL |
|---|---|---|---|---|
| FULL engine vs native HF, fp32 (decode) | 4 | 0.000315 | 1.0000 | 3.54e-07 |
| FULL engine vs native HF, fp32 (strict) | 4 | 0.00234 | 1.0000 | 3.25e-07 |
| FULL engine vs native HF, bf16 (decode) | 4 | 0.5 | 1.0000 | 0.00308 |
| FULL engine vs native HF, bf16 (strict) | 4 | 8.59 | 0.9836 | 0.0138 |
| policy engine vs dict-cache reference, fp32 (decode) | 6 | 1.14e-05 | 1.0000 | 1.63e-07 |
| policy engine vs dict-cache reference, fp32 (strict) | 3 | 2.29e-05 | 1.0000 | 2.72e-07 |

Policy rows of the reference check (fp32): full/decode/short_prompt=0.0e+00, full/strict/short_prompt=0.0e+00, last/decode/short_prompt=0.0e+00, last/strict/short_prompt=0.0e+00, average/decode/short_prompt=1.1e-05, average/strict/short_prompt=2.3e-05, full/decode/benchmark_prompt=0.0e+00, last/decode/benchmark_prompt=0.0e+00, average/decode/benchmark_prompt=7.7e-06

Greedy 48-token generation, bf16 (first divergence index, None = identical):
- gsm8k-test-0005: native vs engine B1 = 31, native vs engine B4 = 31, engine B1 vs B4 = None
- gsm8k-test-0010: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None
- test/algebra/1349.json: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None
- test/prealgebra/1139.json: native vs engine B1 = 4, native vs engine B4 = 4, engine B1 vs B4 = None

bf16 batch-shape sensitivity (B=4 vs B=1, teacher-forced 48 gold tokens): full: max |Δlogprob| 0.309, max |Δsum| 0.235, last: max |Δlogprob| 0.370, max |Δsum| 0.940, average: max |Δlogprob| 0.309, max |Δsum| 0.419

## ouro-2.6b  (GPU NVIDIA H200 4f143625-73b8-2372-5e1a-0bf13de11d61, 715s)

| check | cases | max abs logit diff | min top-1 agreement | max KL |
|---|---|---|---|---|
| FULL engine vs native HF, fp32 (decode) | 4 | 0.00021 | 1.0000 | 4.36e-07 |
| FULL engine vs native HF, fp32 (strict) | 4 | 0.00101 | 1.0000 | 5.75e-07 |
| FULL engine vs native HF, bf16 (decode) | 4 | 0.375 | 0.9989 | 0.00212 |
| FULL engine vs native HF, bf16 (strict) | 4 | 4.12 | 0.9895 | 0.055 |
| policy engine vs dict-cache reference, fp32 (decode) | 6 | 1.14e-05 | 1.0000 | 1.49e-07 |
| policy engine vs dict-cache reference, fp32 (strict) | 3 | 4.2e-05 | 1.0000 | 4.23e-07 |

Policy rows of the reference check (fp32): full/decode/short_prompt=0.0e+00, full/strict/short_prompt=0.0e+00, last/decode/short_prompt=0.0e+00, last/strict/short_prompt=0.0e+00, average/decode/short_prompt=1.1e-05, average/strict/short_prompt=4.2e-05, full/decode/benchmark_prompt=0.0e+00, last/decode/benchmark_prompt=0.0e+00, average/decode/benchmark_prompt=9.5e-06

Greedy 48-token generation, bf16 (first divergence index, None = identical):
- gsm8k-test-0005: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None
- gsm8k-test-0010: native vs engine B1 = 5, native vs engine B4 = None, engine B1 vs B4 = 5
- test/algebra/1349.json: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None
- test/prealgebra/1139.json: native vs engine B1 = None, native vs engine B4 = None, engine B1 vs B4 = None

bf16 batch-shape sensitivity (B=4 vs B=1, teacher-forced 48 gold tokens): full: max |Δlogprob| 0.778, max |Δsum| 0.525, last: max |Δlogprob| 0.778, max |Δsum| 0.852, average: max |Δlogprob| 0.778, max |Δsum| 0.565

CPU tests `pytest_cpu.log` (source_sha=d7a92d0 (run source)): `78 passed, 1 warning in 18.59s`
CPU tests `pytest_cpu_final.log` (source_sha=5096d9e72371facc2e66f7963bc076a211d9126f): `117 passed, 1 warning in 25.72s`


Notes:
- bf16 strict-mode FULL differs from the parallel prefill by up to 8.6 logits on Ouro prompts (fp32: ≤2.3e-3): token-by-token prompt processing is a different bf16 reduction order, not a semantic difference. This motivated the `FULL strict` control in Stage 2.
- The checkpoint's native cached generation cannot run on transformers 4.54.1 (README-recommended): `UniversalTransformerCache.__init__` assigns the read-only `key_cache` property. Native parity therefore uses the no-cache forward (and a plain-object dictionary cache for the policy reference). 4.55.0 fails identically; 4.56.x could not be checked without changing tokenizers (not attempted further).

## 4. Stage 2 — quality (GSM8K-200, MATH-500-200, greedy)

Protocol (frozen before any policy run; identical for all policies of a model; `kvp/prompts.py`):

| model | GSM8K | MATH-500 |
|---|---|---|
| Ouro-1.4B/2.6B | lm-eval `gsm8k_cot`, 3-shot CoT, no chat template (paper Table 16: "strict match, 3-shot CoT, lm-eval-harness") | **not official**: paper uses an unpublished in-house 5-shot prompt; frozen substitute = lm-eval minerva 4-shot |
| Huginn | authors' README lm-eval command: `gsm8k_cot` 8-shot, chat template, fewshot-as-multiturn, system instruction (depth 16 instead of 32) | **not official**: minerva 4-shot + same chat settings |

Max new tokens 256 (GSM8K, lm-eval default) / 1024 (MATH, frozen choice). Subsets: GSM8K `random.Random(20261022).sample(1319, 200)`, MATH-500 subject x level proportional stratification (seed 20261022); IDs and SHA-256 in `subsets.json`. Batches (B=32 GSM8K / 16 MATH, longest-first) are identical across policies. 1080 jobs (5 main cells + 2 controls + 2 convert cells, x gen/NLL x 3 models x 2 datasets), 10,800 generations + 10,800 NLL records, 0 failed jobs, every cell complete (200/200).

Scoring: GSM8K numeric parser ("The answer is N" first, then "####", then `\boxed{}`; commas, `$`, negatives, decimals, fractions canonicalised to exact rationals); MATH balanced-brace `\boxed{}` / minerva "Final Answer" extraction, math_verify equivalence (+ normalised string match), lm-eval minerva `is_equiv` reported as reference. After the parse-failure audit, a fallback (last number / last math RHS / "is X" phrase) is applied **only to generations that terminated on their own**; truncated generations never use it. Format-only accuracy is kept in `summary.csv`. All stored generations are re-scored at analysis time (grader SHA in `summary.md`); raw receipts are unchanged. NLL: teacher-forced log-likelihood of the gold solution written in the elicited answer format (GSM8K rationale without `<<…>>` + "The answer is N."; MATH solution + minerva closing line), prompt tokens masked, token-weighted mean.

### 4.1 Accuracy and NLL by cell

| model | dataset | FULL | LAST conv | AVG conv | LAST dec | AVG dec | LAST strict | AVG strict | FULL strict (ctrl) | FULL alt-batch (ctrl) |
|---|---|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | 0.820 | 0.000 | 0.230 | 0.685 | 0.735 | 0.000 | 0.025 | 0.815 | 0.810 |
| ouro-1.4b | math500 | 0.710 | 0.000 | 0.025 | 0.160 | 0.320 | 0.000 | 0.000 | 0.695 | 0.720 |
| ouro-2.6b | gsm8k | 0.875 | 0.760 | 0.195 | 0.840 | 0.840 | 0.545 | 0.010 | 0.875 | 0.875 |
| ouro-2.6b | math500 | 0.565 | 0.030 | 0.040 | 0.505 | 0.500 | 0.325 | 0.000 | 0.540 | 0.550 |
| huginn-0125 | gsm8k | 0.410 | 0.395 | 0.345 | 0.395 | 0.340 | 0.410 | 0.395 | 0.415 | 0.415 |
| huginn-0125 | math500 | 0.140 | 0.160 | 0.125 | 0.155 | 0.135 | 0.155 | 0.140 | 0.140 | 0.150 |

Teacher-forced token-weighted NLL (nats/token) on the same gold targets:

| model | dataset | FULL | LAST conv | AVG conv | LAST dec | AVG dec | LAST strict | AVG strict | FULL strict (ctrl) | FULL alt-batch (ctrl) |
|---|---|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | 0.7319 | 3.4998 | 1.0203 | 0.7528 | 0.7414 | 6.2638 | 1.5375 | 0.7313 | 0.7318 |
| ouro-1.4b | math500 | 0.5435 | 2.6125 | 1.2609 | 0.6177 | 0.5908 | 5.4176 | 2.3193 | 0.5436 | 0.5434 |
| ouro-2.6b | gsm8k | 0.6959 | 0.8610 | 1.0196 | 0.7045 | 0.6966 | 0.9777 | 3.0756 | 0.6959 | 0.6958 |
| ouro-2.6b | math500 | 0.4646 | 2.1717 | 0.9343 | 0.4943 | 0.4933 | 0.8201 | 4.4869 | 0.4649 | 0.4647 |
| huginn-0125 | gsm8k | 0.8812 | 0.8890 | 0.8655 | 0.8825 | 0.8833 | 0.9022 | 0.8712 | 0.8811 | 0.8814 |
| huginn-0125 | math500 | 0.6376 | 0.6319 | 0.6610 | 0.6323 | 0.6515 | 0.6311 | 0.6547 | 0.6376 | 0.6375 |

### 4.2 Paired differences vs FULL (same items; paired bootstrap 10k, exact McNemar)

| model | dataset | policy | region | Δacc vs FULL [95% CI] | FULL-only / policy-only | McNemar p | ΔNLL/token [95% CI] |
|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | LAST | convert | -0.820 [-0.870, -0.765] | 164 / 0 | 8.6e-50 | +2.7678 [+2.6792, +2.8560] |
| ouro-1.4b | gsm8k | AVERAGE | convert | -0.590 [-0.660, -0.520] | 120 / 2 | 2.8e-33 | +0.2884 [+0.2693, +0.3080] |
| ouro-1.4b | gsm8k | LAST | decode | -0.135 [-0.195, -0.075] | 34 / 7 | 2.5e-05 | +0.0209 [+0.0159, +0.0260] |
| ouro-1.4b | gsm8k | AVERAGE | decode | -0.085 [-0.140, -0.030] | 24 / 7 | 0.0033 | +0.0095 [+0.0064, +0.0126] |
| ouro-1.4b | gsm8k | LAST | strict | -0.820 [-0.870, -0.765] | 164 / 0 | 8.6e-50 | +5.5319 [+5.4326, +5.6319] |
| ouro-1.4b | gsm8k | AVERAGE | strict | -0.795 [-0.850, -0.740] | 159 / 0 | 2.7e-48 | +0.8056 [+0.7774, +0.8334] |
| ouro-1.4b | math500 | LAST | convert | -0.710 [-0.770, -0.650] | 142 / 0 | 3.6e-43 | +2.0690 [+1.9673, +2.1733] |
| ouro-1.4b | math500 | AVERAGE | convert | -0.685 [-0.750, -0.620] | 137 / 0 | 1.1e-41 | +0.7174 [+0.6789, +0.7565] |
| ouro-1.4b | math500 | LAST | decode | -0.550 [-0.620, -0.475] | 112 / 2 | 6.3e-31 | +0.0741 [+0.0608, +0.0889] |
| ouro-1.4b | math500 | AVERAGE | decode | -0.390 [-0.465, -0.315] | 83 / 5 | 2.7e-19 | +0.0473 [+0.0394, +0.0555] |
| ouro-1.4b | math500 | LAST | strict | -0.710 [-0.775, -0.645] | 142 / 0 | 3.6e-43 | +4.8741 [+4.7449, +5.0025] |
| ouro-1.4b | math500 | AVERAGE | strict | -0.710 [-0.770, -0.645] | 142 / 0 | 3.6e-43 | +1.7758 [+1.7057, +1.8420] |
| ouro-2.6b | gsm8k | LAST | convert | -0.115 [-0.170, -0.060] | 28 / 5 | 6.6e-05 | +0.1651 [+0.1349, +0.1987] |
| ouro-2.6b | gsm8k | AVERAGE | convert | -0.680 [-0.745, -0.615] | 137 / 1 | 8e-40 | +0.3237 [+0.3052, +0.3424] |
| ouro-2.6b | gsm8k | LAST | decode | -0.035 [-0.080, +0.005] | 13 / 6 | 0.17 | +0.0086 [+0.0052, +0.0124] |
| ouro-2.6b | gsm8k | AVERAGE | decode | -0.035 [-0.075, +0.005] | 12 / 5 | 0.14 | +0.0007 [-0.0018, +0.0031] |
| ouro-2.6b | gsm8k | LAST | strict | -0.330 [-0.400, -0.260] | 69 / 3 | 2.6e-17 | +0.2819 [+0.2463, +0.3212] |
| ouro-2.6b | gsm8k | AVERAGE | strict | -0.865 [-0.910, -0.815] | 173 / 0 | 1.7e-52 | +2.3797 [+2.3081, +2.4538] |
| ouro-2.6b | math500 | LAST | convert | -0.535 [-0.600, -0.465] | 107 / 0 | 1.2e-32 | +1.7071 [+1.5515, +1.8627] |
| ouro-2.6b | math500 | AVERAGE | convert | -0.525 [-0.595, -0.450] | 107 / 2 | 1.8e-29 | +0.4697 [+0.4375, +0.5044] |
| ouro-2.6b | math500 | LAST | decode | -0.060 [-0.130, +0.010] | 31 / 19 | 0.12 | +0.0297 [+0.0254, +0.0345] |
| ouro-2.6b | math500 | AVERAGE | decode | -0.065 [-0.130, +0.000] | 29 / 16 | 0.072 | +0.0288 [+0.0246, +0.0333] |
| ouro-2.6b | math500 | LAST | strict | -0.240 [-0.320, -0.160] | 63 / 15 | 3.7e-08 | +0.3555 [+0.3161, +0.3975] |
| ouro-2.6b | math500 | AVERAGE | strict | -0.565 [-0.635, -0.495] | 113 / 0 | 1.9e-34 | +4.0224 [+3.8850, +4.1577] |
| huginn-0125 | gsm8k | LAST | convert | -0.015 [-0.065, +0.035] | 15 / 12 | 0.7 | +0.0077 [+0.0037, +0.0120] |
| huginn-0125 | gsm8k | AVERAGE | convert | -0.065 [-0.135, +0.005] | 33 / 20 | 0.098 | -0.0157 [-0.0225, -0.0089] |
| huginn-0125 | gsm8k | LAST | decode | -0.015 [-0.065, +0.035] | 15 / 12 | 0.7 | +0.0012 [-0.0016, +0.0042] |
| huginn-0125 | gsm8k | AVERAGE | decode | -0.070 [-0.130, -0.010] | 26 / 12 | 0.034 | +0.0021 [-0.0023, +0.0068] |
| huginn-0125 | gsm8k | LAST | strict | +0.000 [-0.060, +0.060] | 18 / 18 | 1 | +0.0210 [+0.0158, +0.0268] |
| huginn-0125 | gsm8k | AVERAGE | strict | -0.015 [-0.080, +0.050] | 23 / 20 | 0.76 | -0.0100 [-0.0164, -0.0038] |
| huginn-0125 | math500 | LAST | convert | +0.020 [-0.020, +0.060] | 6 / 10 | 0.45 | -0.0057 [-0.0073, -0.0041] |
| huginn-0125 | math500 | AVERAGE | convert | -0.015 [-0.055, +0.025] | 10 / 7 | 0.63 | +0.0234 [+0.0207, +0.0262] |
| huginn-0125 | math500 | LAST | decode | +0.015 [-0.020, +0.055] | 6 / 9 | 0.61 | -0.0053 [-0.0070, -0.0036] |
| huginn-0125 | math500 | AVERAGE | decode | -0.005 [-0.045, +0.035] | 9 / 8 | 1 | +0.0139 [+0.0117, +0.0161] |
| huginn-0125 | math500 | LAST | strict | +0.015 [-0.025, +0.055] | 7 / 10 | 0.63 | -0.0065 [-0.0081, -0.0049] |
| huginn-0125 | math500 | AVERAGE | strict | +0.000 [-0.035, +0.035] | 7 / 7 | 1 | +0.0171 [+0.0147, +0.0195] |

Numerical controls (same policy, different execution path / batch composition):

| model | dataset | control | Δacc [95% CI] | ΔNLL/token [95% CI] |
|---|---|---|---|---|
| ouro-1.4b | gsm8k | FULL strict (ctrl) | -0.005 [-0.030, +0.020] | -0.0007 [-0.0011, -0.0002] |
| ouro-1.4b | gsm8k | FULL alt-batch (ctrl) | -0.010 [-0.040, +0.020] | -0.0001 [-0.0005, +0.0003] |
| ouro-1.4b | math500 | FULL strict (ctrl) | -0.015 [-0.045, +0.015] | +0.0001 [-0.0001, +0.0004] |
| ouro-1.4b | math500 | FULL alt-batch (ctrl) | +0.010 [-0.025, +0.045] | -0.0002 [-0.0004, +0.0001] |
| ouro-2.6b | gsm8k | FULL strict (ctrl) | +0.000 [-0.015, +0.015] | +0.0000 [-0.0004, +0.0005] |
| ouro-2.6b | gsm8k | FULL alt-batch (ctrl) | +0.000 [+0.000, +0.000] | -0.0001 [-0.0005, +0.0003] |
| ouro-2.6b | math500 | FULL strict (ctrl) | -0.025 [-0.070, +0.020] | +0.0003 [+0.0001, +0.0006] |
| ouro-2.6b | math500 | FULL alt-batch (ctrl) | -0.015 [-0.045, +0.015] | +0.0001 [-0.0001, +0.0004] |
| huginn-0125 | gsm8k | FULL strict (ctrl) | +0.005 [+0.000, +0.015] | -0.0001 [-0.0002, +0.0000] |
| huginn-0125 | gsm8k | FULL alt-batch (ctrl) | +0.005 [-0.020, +0.030] | +0.0002 [-0.0002, +0.0006] |
| huginn-0125 | math500 | FULL strict (ctrl) | +0.000 [+0.000, +0.000] | -0.0000 [-0.0001, +0.0000] |
| huginn-0125 | math500 | FULL alt-batch (ctrl) | +0.010 [-0.015, +0.035] | -0.0001 [-0.0003, +0.0001] |

### 4.3 Comparison with the paper

- FULL baselines: Ouro-1.4B GSM8K 0.820 vs paper 78.92; Ouro-2.6B 0.875 vs 81.58 (n=200, SE ≈ 2.3–2.7 pts; same lm-eval 3-shot protocol, our primary scorer is numeric rather than lm-eval strict-match — lm-eval strict-match is listed as the reference scorer). MATH-500 (0.710 / 0.565 vs 82.40 / 90.85) is **not comparable** (different prompt, unpublished in-house protocol).
- Paper Table 14 (Ouro-1.4B): last-step 78.85 / 80.40, averaged 78.73 / 78.52 vs full 78.92 / 82.40. **Not reproduced** under convert (collapse), decode (−13.5 / −55 pts, p ≤ 3e-5) or strict (collapse). Sampling noise cannot explain the 1.4B gaps (paired CIs far from 0). The paper's implementation is unpublished; possible unexplained differences remain (generation length, extraction, precision, exact cache handling).
- Huginn: consistent with the reported tolerance of recurrent KV sharing — no policy differs significantly from FULL on MATH; GSM8K AVERAGE/decode −7 pts (p = 0.034, CI [−13, −1]) is the only nominally significant cell (12 Huginn policy cells tested; not multiplicity-corrected).

### 4.4 Why Ouro degrades: degeneration and error accumulation (secondary analyses)

Stop rule: every stored generation is cut at the first `"\n\n\n"` and re-scored — the same rule for every cell; with greedy decoding this equals stopping online, so no re-run is needed. `degenerate` = tail is a long newline run or a ≤3-line loop.

| model | dataset | cell | degenerate | hit max tokens | acc main | acc stop-rule | answered→degenerated | degenerated w/o answer |
|---|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | FULL | 0.000 | 0.010 | 0.820 | 0.820 | 0 | 0 |
| ouro-1.4b | gsm8k | LAST conv | 0.120 | 0.930 | 0.000 | 0.000 | 0 | 24 |
| ouro-1.4b | gsm8k | AVG conv | 0.000 | 0.120 | 0.230 | 0.230 | 0 | 0 |
| ouro-1.4b | gsm8k | LAST dec | 0.845 | 0.865 | 0.685 | 0.685 | 117 | 52 |
| ouro-1.4b | gsm8k | AVG dec | 0.960 | 0.975 | 0.735 | 0.735 | 144 | 48 |
| ouro-1.4b | gsm8k | LAST strict | 0.190 | 1.000 | 0.000 | 0.000 | 0 | 38 |
| ouro-1.4b | gsm8k | AVG strict | 0.120 | 0.150 | 0.025 | 0.035 | 2 | 22 |
| ouro-1.4b | math500 | FULL | 0.000 | 0.075 | 0.710 | 0.710 | 0 | 0 |
| ouro-1.4b | math500 | LAST conv | 0.410 | 0.735 | 0.000 | 0.000 | 0 | 82 |
| ouro-1.4b | math500 | AVG conv | 0.270 | 0.835 | 0.025 | 0.020 | 0 | 54 |
| ouro-1.4b | math500 | LAST dec | 0.860 | 0.800 | 0.160 | 0.080 | 12 | 160 |
| ouro-1.4b | math500 | AVG dec | 0.980 | 1.000 | 0.320 | 0.120 | 23 | 173 |
| ouro-1.4b | math500 | LAST strict | 0.090 | 0.910 | 0.000 | 0.000 | 0 | 18 |
| ouro-1.4b | math500 | AVG strict | 0.990 | 1.000 | 0.000 | 0.000 | 0 | 198 |
| ouro-2.6b | gsm8k | FULL | 0.000 | 0.000 | 0.875 | 0.875 | 0 | 0 |
| ouro-2.6b | gsm8k | LAST conv | 0.005 | 0.030 | 0.760 | 0.760 | 0 | 1 |
| ouro-2.6b | gsm8k | AVG conv | 0.025 | 0.345 | 0.195 | 0.200 | 0 | 5 |
| ouro-2.6b | gsm8k | LAST dec | 0.000 | 0.005 | 0.840 | 0.840 | 0 | 0 |
| ouro-2.6b | gsm8k | AVG dec | 0.000 | 0.010 | 0.840 | 0.840 | 0 | 0 |
| ouro-2.6b | gsm8k | LAST strict | 0.000 | 0.035 | 0.545 | 0.545 | 0 | 0 |
| ouro-2.6b | gsm8k | AVG strict | 0.110 | 0.770 | 0.010 | 0.015 | 0 | 22 |
| ouro-2.6b | math500 | FULL | 0.015 | 0.105 | 0.565 | 0.565 | 0 | 3 |
| ouro-2.6b | math500 | LAST conv | 0.240 | 0.875 | 0.030 | 0.030 | 0 | 48 |
| ouro-2.6b | math500 | AVG conv | 0.185 | 0.865 | 0.040 | 0.045 | 2 | 35 |
| ouro-2.6b | math500 | LAST dec | 0.010 | 0.110 | 0.505 | 0.495 | 0 | 2 |
| ouro-2.6b | math500 | AVG dec | 0.030 | 0.185 | 0.500 | 0.500 | 0 | 6 |
| ouro-2.6b | math500 | LAST strict | 0.040 | 0.175 | 0.325 | 0.325 | 0 | 8 |
| ouro-2.6b | math500 | AVG strict | 0.085 | 0.110 | 0.000 | 0.000 | 0 | 17 |
| huginn-0125 | gsm8k | FULL | 0.000 | 0.035 | 0.410 | 0.410 | 0 | 0 |
| huginn-0125 | gsm8k | LAST conv | 0.000 | 0.030 | 0.395 | 0.395 | 0 | 0 |
| huginn-0125 | gsm8k | AVG conv | 0.005 | 0.050 | 0.345 | 0.345 | 0 | 1 |
| huginn-0125 | gsm8k | LAST dec | 0.000 | 0.025 | 0.395 | 0.395 | 0 | 0 |
| huginn-0125 | gsm8k | AVG dec | 0.000 | 0.050 | 0.340 | 0.340 | 0 | 0 |
| huginn-0125 | gsm8k | LAST strict | 0.000 | 0.020 | 0.410 | 0.410 | 0 | 0 |
| huginn-0125 | gsm8k | AVG strict | 0.000 | 0.010 | 0.395 | 0.395 | 0 | 0 |
| huginn-0125 | math500 | FULL | 0.055 | 0.205 | 0.140 | 0.140 | 0 | 11 |
| huginn-0125 | math500 | LAST conv | 0.060 | 0.245 | 0.160 | 0.160 | 0 | 12 |
| huginn-0125 | math500 | AVG conv | 0.040 | 0.270 | 0.125 | 0.125 | 0 | 8 |
| huginn-0125 | math500 | LAST dec | 0.050 | 0.235 | 0.155 | 0.155 | 0 | 10 |
| huginn-0125 | math500 | AVG dec | 0.060 | 0.270 | 0.135 | 0.135 | 0 | 12 |
| huginn-0125 | math500 | LAST strict | 0.045 | 0.270 | 0.155 | 0.155 | 0 | 9 |
| huginn-0125 | math500 | AVG strict | 0.045 | 0.265 | 0.140 | 0.140 | 0 | 9 |

Teacher-forced ΔNLL per token vs FULL by gold-target position (paired on identical gold tokens):

| model | dataset | cell | pos 0-63 | pos 64-127 | pos 128-255 | pos 256-511 | pos 512+ |
|---|---|---|---|---|---|---|---|
| ouro-1.4b | gsm8k | LAST conv | +2.625 (n=200) | +2.971 (n=143) | +3.258 (n=37) | +3.633 (n=1) |  |
| ouro-1.4b | gsm8k | AVG conv | +0.217 (n=200) | +0.402 (n=143) | +0.469 (n=37) | +0.601 (n=1) |  |
| ouro-1.4b | gsm8k | LAST dec | +0.016 (n=200) | +0.031 (n=143) | +0.027 (n=37) | -0.025 (n=1) |  |
| ouro-1.4b | gsm8k | AVG dec | +0.005 (n=200) | +0.017 (n=143) | +0.018 (n=37) | -0.016 (n=1) |  |
| ouro-1.4b | gsm8k | LAST strict | +5.489 (n=200) | +5.592 (n=143) | +5.719 (n=37) | +4.567 (n=1) |  |
| ouro-1.4b | gsm8k | AVG strict | +0.793 (n=200) | +0.835 (n=143) | +0.791 (n=37) | +0.637 (n=1) |  |
| ouro-1.4b | math500 | LAST conv | +2.169 (n=200) | +2.058 (n=193) | +2.030 (n=144) | +2.074 (n=70) | +1.891 (n=21) |
| ouro-1.4b | math500 | AVG conv | +0.463 (n=200) | +0.681 (n=193) | +0.765 (n=144) | +0.868 (n=70) | +1.121 (n=21) |
| ouro-1.4b | math500 | LAST dec | +0.035 (n=200) | +0.068 (n=193) | +0.084 (n=144) | +0.108 (n=70) | +0.102 (n=21) |
| ouro-1.4b | math500 | AVG dec | +0.025 (n=200) | +0.043 (n=193) | +0.055 (n=144) | +0.068 (n=70) | +0.055 (n=21) |
| ouro-1.4b | math500 | LAST strict | +4.944 (n=200) | +4.931 (n=193) | +4.936 (n=144) | +4.798 (n=70) | +4.452 (n=21) |
| ouro-1.4b | math500 | AVG strict | +1.423 (n=200) | +1.724 (n=193) | +1.855 (n=144) | +2.004 (n=70) | +2.244 (n=21) |
| ouro-2.6b | gsm8k | LAST conv | +0.153 (n=200) | +0.196 (n=143) | +0.136 (n=37) | -0.001 (n=1) |  |
| ouro-2.6b | gsm8k | AVG conv | +0.324 (n=200) | +0.324 (n=143) | +0.316 (n=37) | +0.297 (n=1) |  |
| ouro-2.6b | gsm8k | LAST dec | +0.003 (n=200) | +0.019 (n=143) | +0.009 (n=37) | -0.005 (n=1) |  |
| ouro-2.6b | gsm8k | AVG dec | -0.002 (n=200) | +0.005 (n=143) | +0.007 (n=37) | -0.012 (n=1) |  |
| ouro-2.6b | gsm8k | LAST strict | +0.289 (n=200) | +0.285 (n=143) | +0.191 (n=37) | -0.043 (n=1) |  |
| ouro-2.6b | gsm8k | AVG strict | +2.300 (n=200) | +2.492 (n=143) | +2.706 (n=37) | +1.306 (n=1) |  |
| ouro-2.6b | math500 | LAST conv | +1.493 (n=200) | +1.564 (n=193) | +1.809 (n=144) | +1.936 (n=70) | +1.892 (n=21) |
| ouro-2.6b | math500 | AVG conv | +0.431 (n=200) | +0.426 (n=193) | +0.485 (n=144) | +0.536 (n=70) | +0.501 (n=21) |
| ouro-2.6b | math500 | LAST dec | +0.021 (n=200) | +0.032 (n=193) | +0.035 (n=144) | +0.033 (n=70) | +0.027 (n=21) |
| ouro-2.6b | math500 | AVG dec | +0.020 (n=200) | +0.026 (n=193) | +0.033 (n=144) | +0.037 (n=70) | +0.031 (n=21) |
| ouro-2.6b | math500 | LAST strict | +0.353 (n=200) | +0.318 (n=193) | +0.361 (n=144) | +0.375 (n=70) | +0.400 (n=21) |
| ouro-2.6b | math500 | AVG strict | +3.661 (n=200) | +3.801 (n=193) | +4.044 (n=144) | +4.476 (n=70) | +4.638 (n=21) |
| huginn-0125 | gsm8k | LAST conv | +0.013 (n=200) | -0.003 (n=130) | -0.000 (n=23) | -0.017 (n=1) |  |
| huginn-0125 | gsm8k | AVG conv | -0.024 (n=200) | +0.001 (n=130) | +0.002 (n=23) | -0.008 (n=1) |  |
| huginn-0125 | gsm8k | LAST dec | +0.004 (n=200) | -0.005 (n=130) | +0.001 (n=23) | -0.008 (n=1) |  |
| huginn-0125 | gsm8k | AVG dec | +0.002 (n=200) | +0.001 (n=130) | +0.004 (n=23) | -0.035 (n=1) |  |
| huginn-0125 | gsm8k | LAST strict | +0.030 (n=200) | +0.002 (n=130) | +0.001 (n=23) | -0.028 (n=1) |  |
| huginn-0125 | gsm8k | AVG strict | -0.013 (n=200) | -0.004 (n=130) | +0.001 (n=23) | -0.040 (n=1) |  |
| huginn-0125 | math500 | LAST conv | -0.005 (n=200) | -0.008 (n=192) | -0.005 (n=136) | -0.006 (n=67) | -0.002 (n=18) |
| huginn-0125 | math500 | AVG conv | +0.022 (n=200) | +0.027 (n=192) | +0.023 (n=136) | +0.023 (n=67) | +0.020 (n=18) |
| huginn-0125 | math500 | LAST dec | -0.004 (n=200) | -0.007 (n=192) | -0.006 (n=136) | -0.006 (n=67) | -0.004 (n=18) |
| huginn-0125 | math500 | AVG dec | +0.015 (n=200) | +0.016 (n=192) | +0.013 (n=136) | +0.012 (n=67) | +0.010 (n=18) |
| huginn-0125 | math500 | LAST strict | -0.008 (n=200) | -0.008 (n=192) | -0.005 (n=136) | -0.006 (n=67) | -0.001 (n=18) |
| huginn-0125 | math500 | AVG strict | +0.017 (n=200) | +0.020 (n=192) | +0.016 (n=136) | +0.016 (n=67) | +0.015 (n=18) |

Reading: under decode-region policies Ouro's per-token NLL penalty is small at the first positions and grows with position (error accumulation once more policy-region history is read). On GSM8K many LAST/AVG generations write the correct answer and only then fall into newline loops (stop-rule accuracy ≈ main accuracy; the loss is in answers that change); on MATH most degenerate before an answer. convert/strict penalties are large from the first positions (prompt history already in policy form).

### 4.5 Parse-failure audit (user request: fix parse failures)

Before the fix, most failures were genuine "no answer" outputs (truncated at the token limit, mostly degenerate). The audit found one real extraction gap — naturally terminated outputs, mainly Huginn MATH, ending in an un-boxed expression or phrase — fixed with the termination-gated fallback above, plus one fallback bug (an escaped `\$` read as a math delimiter) fixed with a fixture. 40 sampled fallback extractions were inspected: they recover the intended final answer; most of them are still wrong answers. Effect on accuracy is small (e.g. Huginn MATH FULL 0.135 → 0.140); the remaining failures are classified as `truncated_no_answer` / `finished_no_answer` in `summary.csv`. Primary-vs-reference scorer disagreements are listed in `analysis/scorer_audit.json` (GSM8K: lm-eval strict-match requires the exact "The answer is N." phrase and fails e.g. on "$18"; MATH: math_verify vs minerva).

## 5. Stage 3 — serving validation (vllm-rlt) and measured KV residency

Settings: synchronous scheduling, eager (no CUDA graphs), no speculation, Triton attention, block 16, bf16, fixed depth (`min_loops = max_loops = depth`, `exit_threshold = 1`), real frozen GSM8K prompts, 128 new tokens with `ignore_eos`, 1 warm-up + 3 timed repetitions, each configuration on two GPUs with swapped layout order; the other GPUs were running Stage-2 quality jobs at the time (host-side launch contention possible, equal for both layouts).


## Output parity vs HF reference engine (16 GSM8K prompts x 64 greedy tokens, bf16)

| model | layout | gpu | HF reference cell | exact 64-token match | mean first divergence | all depths fixed |
|---|---|---|---|---|---|---|
| huginn-0125 | last_exited | gpu0 | full/decode | 0.688 | 48.4 | False |
| huginn-0125 | last_exited | gpu1 | full/decode | 0.688 | 48.4 | False |
| ouro-1.4b | last_exited | gpu0 | full/decode | 0.938 | 62.2 | True |
| ouro-1.4b | last_exited | gpu1 | full/decode | 0.938 | 62.2 | True |
| ouro-1.4b | shared | gpu0 | last/strict | 0.000 | 0.9 | True |
| ouro-1.4b | shared | gpu1 | last/strict | 0.000 | 0.9 | True |
| ouro-2.6b | last_exited | gpu0 | full/decode | 0.750 | 53.5 | True |
| ouro-2.6b | last_exited | gpu1 | full/decode | 0.750 | 53.5 | True |
| ouro-2.6b | shared | gpu0 | last/strict | 0.000 | 7.4 | True |
| ouro-2.6b | shared | gpu1 | last/strict | 0.000 | 7.4 | True |

## KV pool geometry and per-request footprint (prompt 0, 128 new tokens)

| model | layout | gpu | storage planes | bytes/block | pool blocks | prompt tok | blocks/request | bytes/request | requests fitting pool |
|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | last_exited | gpu0 | 32 | 2703360 | 19065 | 805 | 1888 | 4867.5 MiB | 10 |
| huginn-0125 | last_exited | gpu1 | 32 | 2703360 | 19065 | 805 | 1888 | 4867.5 MiB | 10 |
| ouro-1.4b | last_exited | gpu0 | 4 | 3145728 | 16384 | 318 | 112 | 336.0 MiB | 146 |
| ouro-1.4b | last_exited | gpu1 | 4 | 3145728 | 16384 | 318 | 112 | 336.0 MiB | 146 |
| ouro-1.4b | shared | gpu0 | 1 | 3145728 | 16384 | 318 | 28 | 84.0 MiB | 585 |
| ouro-1.4b | shared | gpu1 | 1 | 3145728 | 16384 | 318 | 28 | 84.0 MiB | 585 |
| ouro-2.6b | last_exited | gpu0 | 4 | 6291456 | 8192 | 318 | 112 | 672.0 MiB | 73 |
| ouro-2.6b | last_exited | gpu1 | 4 | 6291456 | 8192 | 318 | 112 | 672.0 MiB | 73 |
| ouro-2.6b | shared | gpu0 | 1 | 6291456 | 8192 | 318 | 28 | 168.0 MiB | 292 |
| ouro-2.6b | shared | gpu1 | 1 | 6291456 | 8192 | 318 | 28 | 168.0 MiB | 292 |

## Decode performance (1 warm-up + 3 timed reps, 128 tokens, ignore_eos, sync eager triton)

| model | layout | gpu | B | TPOT ms mean [min,max] | decode tok/s | e2e tok/s | TTFT s | peak KV used |
|---|---|---|---|---|---|---|---|---|
| huginn-0125 | last_exited | gpu0 | 8 | 67.32 [67.14, 67.54] | 118.8 | 69.0 | 6.295 | 37.87 GiB (15040 blk) |
| huginn-0125 | last_exited | gpu0 | 32 | 72.90 [72.87, 72.93] | 595.5 | 67.4 | 27.204 | 47.94 GiB (19040 blk) |
| huginn-0125 | last_exited | gpu1 | 8 | 68.20 [68.08, 68.33] | 117.3 | 68.7 | 6.253 | 37.87 GiB (15040 blk) |
| huginn-0125 | last_exited | gpu1 | 32 | 74.04 [73.74, 74.50] | 586.1 | 66.8 | 27.380 | 47.94 GiB (19040 blk) |
| ouro-1.4b | last_exited | gpu0 | 8 | 50.87 [50.71, 50.96] | 157.3 | 151.7 | 0.289 | 2.64 GiB (900 blk) |
| ouro-1.4b | last_exited | gpu0 | 32 | 58.49 [58.45, 58.52] | 550.6 | 478.6 | 1.118 | 10.79 GiB (3684 blk) |
| ouro-1.4b | last_exited | gpu1 | 8 | 50.93 [50.89, 50.96] | 157.1 | 151.6 | 0.286 | 2.64 GiB (900 blk) |
| ouro-1.4b | last_exited | gpu1 | 32 | 56.83 [56.39, 57.14] | 566.8 | 494.2 | 1.059 | 10.79 GiB (3684 blk) |
| ouro-1.4b | shared | gpu0 | 8 | 50.74 [50.62, 50.97] | 157.7 | 47.0 | 15.356 | 0.66 GiB (225 blk) |
| ouro-1.4b | shared | gpu0 | 32 | 56.96 [56.92, 56.99] | 572.1 | 160.1 | 18.344 | 2.70 GiB (921 blk) |
| ouro-1.4b | shared | gpu1 | 8 | 50.30 [50.24, 50.37] | 159.1 | 47.4 | 15.220 | 0.66 GiB (225 blk) |
| ouro-1.4b | shared | gpu1 | 32 | 56.20 [55.77, 56.59] | 579.8 | 162.3 | 18.094 | 2.70 GiB (921 blk) |
| ouro-2.6b | last_exited | gpu0 | 8 | 96.49 [95.60, 97.59] | 82.9 | 79.9 | 0.560 | 5.27 GiB (900 blk) |
| ouro-2.6b | last_exited | gpu0 | 32 | 103.34 [103.05, 103.93] | 311.8 | 269.7 | 2.038 | 21.59 GiB (3684 blk) |
| ouro-2.6b | last_exited | gpu1 | 8 | 97.29 [97.20, 97.43] | 82.2 | 79.3 | 0.556 | 5.27 GiB (900 blk) |
| ouro-2.6b | last_exited | gpu1 | 32 | 105.16 [104.38, 105.78] | 306.4 | 266.3 | 2.005 | 21.59 GiB (3684 blk) |
| ouro-2.6b | shared | gpu0 | 8 | 99.75 [98.47, 100.59] | 80.2 | 23.2 | 31.423 | 1.32 GiB (225 blk) |
| ouro-2.6b | shared | gpu0 | 32 | 103.85 [103.72, 104.03] | 314.2 | 83.4 | 35.913 | 5.40 GiB (921 blk) |
| ouro-2.6b | shared | gpu1 | 8 | 98.43 [97.81, 98.94] | 81.3 | 23.5 | 31.055 | 1.32 GiB (225 blk) |
| ouro-2.6b | shared | gpu1 | 32 | 108.52 [106.48, 110.50] | 300.7 | 79.3 | 37.910 | 5.40 GiB (921 blk) |

## Huginn re-run with depth alignment (config mean_recurrence 32 -> 16, 100 GiB pool)

| gpu | parity exact | mean first div | depths fixed | B | TPOT ms [min,max] | decode tok/s | TTFT s | peak KV |
|---|---|---|---|---|---|---|---|---|
| gpu0 | 0.688 | 51.8 | True | 8 | 58.18 [57.71, 58.73] | 137.5 | 3.30 | 18.93 GiB |
| gpu0 | 0.688 | 51.8 | True | 32 | 86.33 [85.98, 87.00] | 370.7 | 12.98 | 76.42 GiB |
| gpu1 | 0.688 | 51.8 | True | 8 | 57.80 [57.33, 58.31] | 138.4 | 3.22 | 18.93 GiB |
| gpu1 | 0.688 | 51.8 | True | 32 | 85.31 [84.88, 85.88] | 375.1 | 12.84 | 76.42 GiB |

## fp32 layout semantics (ouro-1.4b, backend torch, 24 greedy tokens)

| vllm-rlt layout | HF cell | exact / n | first divergences |
|---|---|---|---|
| last_exited | full/decode | 8 / 8 | [None, None, None, None, None, None, None, None] |
| shared | last/strict | 8 / 8 | [None, None, None, None, None, None, None, None] |
| shared | last/decode | 0 / 8 | [0, 0, 1, 7, 1, 2, 1, 3] |

## Measured KV residency of every policy (HF engine, compact policy-sized store)

First frozen GSM8K batch (B=32, 256 new tokens, no early stop). Bytes are the retained store after prefill (recurrent + non-recurrent K/V); `alloc` includes convert-mode transient prompt planes. The HF engine is a reference implementation, not a serving engine.

| model | policy | mode | P | KV retained GiB | ratio vs FULL | analytic recurrent GiB | KV at alloc GiB | CUDA peak GiB | prefix agreement vs stage-2 |
|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | full | decode | 875 | 48.41 | 1.000 | 45.56 | 48.41 | 66.29 | 1.000 |
| huginn-0125 | last | convert | 875 | 5.70 | 0.118 | 2.85 | 40.94 | 58.86 | 1.000 |
| huginn-0125 | average | convert | 875 | 5.70 | 0.118 | 2.85 | 40.94 | 64.50 | 1.000 |
| huginn-0125 | last | decode | 875 | 38.74 | 0.800 | 35.89 | 38.74 | 56.66 | 1.000 |
| huginn-0125 | average | decode | 875 | 38.74 | 0.800 | 35.89 | 38.74 | 56.66 | 1.000 |
| huginn-0125 | last | strict | 875 | 5.70 | 0.118 | 2.85 | 5.70 | 13.50 | 1.000 |
| huginn-0125 | average | strict | 875 | 5.70 | 0.118 | 2.85 | 5.70 | 13.50 | 1.000 |
| ouro-1.4b | full | decode | 393 | 15.21 | 1.000 | 15.21 | 15.21 | 21.72 | 1.000 |
| ouro-1.4b | last | convert | 393 | 3.80 | 0.250 | 3.80 | 13.01 | 19.59 | 1.000 |
| ouro-1.4b | average | convert | 393 | 3.80 | 0.250 | 3.80 | 13.01 | 19.59 | 1.000 |
| ouro-1.4b | last | decode | 393 | 10.71 | 0.704 | 10.71 | 10.71 | 17.24 | 1.000 |
| ouro-1.4b | average | decode | 393 | 10.71 | 0.704 | 10.71 | 10.71 | 17.24 | 1.000 |
| ouro-1.4b | last | strict | 393 | 3.80 | 0.250 | 3.80 | 3.80 | 6.74 | 1.000 |
| ouro-1.4b | average | strict | 393 | 3.80 | 0.250 | 3.80 | 3.80 | 6.74 | 1.000 |
| ouro-2.6b | full | decode | 393 | 30.42 | 1.000 | 30.42 | 30.42 | 39.23 | 1.000 |
| ouro-2.6b | last | convert | 393 | 7.61 | 0.250 | 7.61 | 26.03 | 34.96 | 1.000 |
| ouro-2.6b | average | convert | 393 | 7.61 | 0.250 | 7.61 | 26.03 | 34.96 | 1.000 |
| ouro-2.6b | last | decode | 393 | 21.42 | 0.704 | 21.42 | 21.42 | 30.28 | 1.000 |
| ouro-2.6b | average | decode | 393 | 21.42 | 0.704 | 21.42 | 21.42 | 30.28 | 1.000 |
| ouro-2.6b | last | strict | 393 | 7.61 | 0.250 | 7.61 | 7.61 | 12.90 | 1.000 |
| ouro-2.6b | average | strict | 393 | 7.61 | 0.250 | 7.61 | 7.61 | 12.90 | 1.000 |


Notes:
- The first Huginn run (table rows `huginn-0125 … gpu0/gpu1` above) is **invalid for Huginn comparisons**: vllm-rlt ran prompt prefill at the config depth 32 (exit depths {16, 32}) and sized the pool with 32 planes, so only 10 requests fitted at B=32. It is kept as a receipt; the depth-aligned re-run is the valid one. Huginn output parity is bounded by different initial-state RNG between the engines (11/16 exact 64-token matches).
- `shared` = strict LAST, so its quality is the collapsed strict-LAST cell of Stage 2 for Ouro. The paper-style policies (convert/decode LAST, all AVERAGE) are not implemented in vllm-rlt; their measured residency comes from the HF compact store.
- TPOT is launch-bound in this eager configuration and does not reflect KV-read bandwidth; the 4x KV reduction shows up as capacity (requests fitting the pool: 146 → 585 for Ouro-1.4B at this prompt length) and as TTFT cost for `shared` (sequential prompt).
- Measured residency matches the analytic accounting exactly (recurrent bytes; Huginn additionally keeps 4 non-recurrent prelude/coda layers at one version, so its convert/strict total is 0.118 of FULL while its recurrent part is 1/16). convert holds the per-loop prompt planes transiently during prefill (`KV at alloc`), so its peak memory is close to FULL for prompt-heavy batches; the saving is in the retained cache. Compact-store generations are token-identical to the Stage-2 broadcast receipts (prefix agreement 1.000 for every cell).
- Stage-3 v1 HF memory receipts in the stage-3 root (`hf_memory_*.json`, without convert cells) are superseded by `hf_memory_v2/`.

## 6. Deviations, failures and open items

- Plan items not done: no learned exits / depth sweep / full-dataset runs (by scope); no AVERAGE / convert / decode-only LAST layout in vllm-rlt (would require new engine machinery); no TTFT/kernel deep-dive.
- Two policy-region variants (convert, decode) were added after the first results; the Stage-2 run manifest records the source SHA of every addendum (`control_jobs_addendum`, `convert_jobs_addendum`), and each job's `source_sha`.
- Ouro MATH-500 and Huginn MATH-500 prompts are frozen substitutes (not official); Huginn depth is 16 (not the authors' 32).
- The paper's Table-14 claim could not be reproduced; the cause is not identified.

## 7. Reproduction

See `README.md` (environment, exact commands). Every run directory contains the exact command lines, job lists, source SHAs, package versions and GPU UUIDs.
