# Fixed-depth recurrent-KV policies (FULL / LAST / AVERAGE) — Stages 1–3 on H200 x8

Scope: `FIXED_DEPTH_SCOPE.md` (fixed native depth, early exit OFF, GSM8K-200 + MATH-500-200, Stages 1–3).
Generated from receipts by `scripts/render_results.py` (tables) — prose written by the executing agent.
Run directories:

{{RUN_DIRS}}

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

{{STAGE1}}

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

{{ACC_MATRIX}}

### 4.2 Paired differences vs FULL (same items; paired bootstrap 10k, exact McNemar)

{{DELTA_TABLE}}

### 4.3 Comparison with the paper

- FULL baselines: Ouro-1.4B GSM8K 0.820 vs paper 78.92; Ouro-2.6B 0.875 vs 81.58 (n=200, SE ≈ 2.3–2.7 pts; same lm-eval 3-shot protocol, our primary scorer is numeric rather than lm-eval strict-match — lm-eval strict-match is listed as the reference scorer). MATH-500 (0.710 / 0.565 vs 82.40 / 90.85) is **not comparable** (different prompt, unpublished in-house protocol).
- Paper Table 14 (Ouro-1.4B): last-step 78.85 / 80.40, averaged 78.73 / 78.52 vs full 78.92 / 82.40. **Not reproduced** under convert (collapse), decode (−13.5 / −55 pts, p ≤ 3e-5) or strict (collapse). Sampling noise cannot explain the 1.4B gaps (paired CIs far from 0). The paper's implementation is unpublished; possible unexplained differences remain (generation length, extraction, precision, exact cache handling).
- Huginn: consistent with the reported tolerance of recurrent KV sharing — no policy differs significantly from FULL on MATH; GSM8K AVERAGE/decode −7 pts (p = 0.034, CI [−13, −1]) is the only nominally significant cell (12 Huginn policy cells tested; not multiplicity-corrected).

### 4.4 Why Ouro degrades: degeneration and error accumulation (secondary analyses)

Stop rule: every stored generation is cut at the first `"\n\n\n"` and re-scored — the same rule for every cell; with greedy decoding this equals stopping online, so no re-run is needed. `degenerate` = tail is a long newline run or a ≤3-line loop.

{{STOPRULE_TABLE}}

Teacher-forced ΔNLL per token vs FULL by gold-target position (paired on identical gold tokens):

{{NLLPOS_TABLE}}

Reading: under decode-region policies Ouro's per-token NLL penalty is small at the first positions and grows with position (error accumulation once more policy-region history is read). On GSM8K many LAST/AVG generations write the correct answer and only then fall into newline loops (stop-rule accuracy ≈ main accuracy; the loss is in answers that change); on MATH most degenerate before an answer. convert/strict penalties are large from the first positions (prompt history already in policy form).

### 4.5 Parse-failure audit (user request: fix parse failures)

Before the fix, most failures were genuine "no answer" outputs (truncated at the token limit, mostly degenerate). The audit found one real extraction gap — naturally terminated outputs, mainly Huginn MATH, ending in an un-boxed expression or phrase — fixed with the termination-gated fallback above, plus one fallback bug (an escaped `\$` read as a math delimiter) fixed with a fixture. 40 sampled fallback extractions were inspected: they recover the intended final answer; most of them are still wrong answers. Effect on accuracy is small (e.g. Huginn MATH FULL 0.135 → 0.140); the remaining failures are classified as `truncated_no_answer` / `finished_no_answer` in `summary.csv`. Primary-vs-reference scorer disagreements are listed in `analysis/scorer_audit.json` (GSM8K: lm-eval strict-match requires the exact "The answer is N." phrase and fails e.g. on "$18"; MATH: math_verify vs minerva).

## 5. Stage 3 — serving validation (vllm-rlt) and measured KV residency

Settings: synchronous scheduling, eager (no CUDA graphs), no speculation, Triton attention, block 16, bf16, fixed depth (`min_loops = max_loops = depth`, `exit_threshold = 1`), real frozen GSM8K prompts, 128 new tokens with `ignore_eos`, 1 warm-up + 3 timed repetitions, each configuration on two GPUs with swapped layout order; the other GPUs were running Stage-2 quality jobs at the time (host-side launch contention possible, equal for both layouts).

{{STAGE3}}

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
