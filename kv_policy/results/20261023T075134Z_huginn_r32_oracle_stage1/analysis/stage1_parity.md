# Huginn R32 Stage 1 — native correctness / official evaluation audit

Run: `results/20261023T075134Z_huginn_r32_oracle_stage1`; base quality run: `results/20261023T060125Z_huginn_r32_h200x8`. Plan: `kv_policy/HUGINN_R32_KV_ORACLE_PLAN.md`.

## 1.1 Official Huginn lm-eval GSM8K (separate from the project runner)

| variant | items | strict-match | flexible-extract | on frozen-200: strict | flexible | project FULL primary | project FULL lm-eval strict / flexible |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_b1_full | 1319 | 0.3503 | 0.4238 | 0.325 | 0.395 | 0.415 (n=200) | 0.340 / 0.410 |

- A_b1_full: batch_sizes=[], seeds random/torch=0/1234; prompt text identical 200/200, prompt token IDs identical 200/200; paired strict-match, official vs project FULL (lm-eval strict on project outputs): {'official_only': 2, 'project_only': 5, 'both': 63, 'neither': 130}; generation length: {'n': 1319, 'ge_256_tokens': 24}

Deviations from the README command (scripts/oracle/s1_official_eval.sh): revision pinned (== HF main), transformers 4.53.3 (4.54.1 breaks the remote HuginnDynamicCache), batch_size=1 (README `auto` chose 64 and OOMed, fixed 16 OOMed; failed logs kept under official/logs/).

Protocol differences vs the project runner that remain by construction: official cap = lm-eval default max_gen_toks 256 (project 512); official initial latent = model `randn` under lm-eval torch seed 1234, consumed sequentially per rank (project: per-(uid, position) seeded draw); native dynamic cache vs project engine; project primary GSM8K scorer adds a last-number fallback for untruncated outputs (official strict-match does not; official flexible-extract takes the last number even when truncated).

## 1.2 FP32 parity: engine FULL vs native HF (FP64 native no-cache = ground truth)

| case | P+cont | native32 vs 64: max|Δ| | engine32 vs 64: max|Δ| | native-cache32 vs 64: max|Δ| | engine vs native32: max|Δ| | KL max | top-1 disagree (margin>1e-3) | first-token top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| short short | 30+26 | 5.34e-05 | 4.29e-05 | 4.29e-05 | 5.15e-05 | 2.43e-10 | 0 (0) | True |
| gsm8k gsm8k-test-0084 | 773+64 | 1.10e-04 | 1.07e-04 | 1.07e-04 | 8.20e-05 | 5.38e-10 | 0 (0) | True |
| gsm8k gsm8k-test-0104 | 797+64 | 8.20e-05 | 1.12e-04 | 1.12e-04 | 7.06e-05 | 3.80e-10 | 0 (0) | True |
| math500 test/algebra/109.json | 779+64 | 8.58e-05 | 9.92e-05 | 9.92e-05 | 8.58e-05 | 1.20e-09 | 0 (0) | True |
| math500 test/algebra/1934.json | 804+64 | 8.77e-05 | 9.92e-05 | 9.92e-05 | 7.92e-05 | 3.27e-10 | 0 (0) | True |

Engine inside a left-padded B4 batch, FP32 (teacher-forced gold logprobs):

- gsm8k-test-0084: max|Δlogprob| B4 vs FP64 3.50e-05, B1 vs FP64 2.27e-05, B4 vs B1 1.52e-05
- gsm8k-test-0104: max|Δlogprob| B4 vs FP64 2.69e-05, B1 vs FP64 3.39e-05, B4 vs B1 2.02e-05
- test/algebra/109.json: max|Δlogprob| B4 vs FP64 2.59e-05, B1 vs FP64 3.98e-05, B4 vs B1 1.60e-05
- test/algebra/1934.json: max|Δlogprob| B4 vs FP64 2.38e-05, B1 vs FP64 2.79e-05, B4 vs B1 7.90e-06

Initial-latent sensitivity of the native model, FP32 (vs paired engine latents):

- short: independent seeded draw: max|Δlogit| 1.30e-01, KL max 4.20e-04, top-1 agree 1.0000; model randn init: max|Δlogit| 1.16e-01, KL max 4.23e-04, top-1 agree 1.0000
- gsm8k-test-0084: independent seeded draw: max|Δlogit| 1.08e+00, KL max 1.21e-03, top-1 agree 0.9988; model randn init: max|Δlogit| 9.46e-01, KL max 1.43e-03, top-1 agree 0.9988
- gsm8k-test-0104: independent seeded draw: max|Δlogit| 8.82e-01, KL max 2.56e-03, top-1 agree 0.9977; model randn init: max|Δlogit| 8.39e-01, KL max 5.54e-04, top-1 agree 0.9965
- test/algebra/109.json: independent seeded draw: max|Δlogit| 3.16e+00, KL max 2.50e-01, top-1 agree 0.9941; model randn init: max|Δlogit| 6.40e-01, KL max 9.39e-03, top-1 agree 0.9964
- test/algebra/1934.json: independent seeded draw: max|Δlogit| 5.89e-01, KL max 7.26e-03, top-1 agree 0.9988; model randn init: max|Δlogit| 5.41e-01, KL max 3.89e-03, top-1 agree 0.9977

**FP32 gate: PASS** (criteria in the header of scripts/oracle/s1_report.py).

## 1.3 BF16 paired greedy generations + teacher-forced comparison

| dataset | n | identical native vs engine B1 | identical engine B1 vs batched | acc native | acc engine B1 | acc engine batched | first-token top-1 agree | TF top-1 agree (mean) | TF KL mean | TF ΔNLL eng−nat (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 32 | 23 | 21 | 16/32 | 16/32 | 16/32 | 32/32 | 0.9975 | 3.03e-04 | -2.57e-04 |
| math500 | 32 | 20 | 19 | 10/32 | 12/32 | 14/32 | 31/32 | 0.9976 | 2.09e-04 | -5.92e-04 |

First greedy divergence index (native vs engine B1), per item: gsm8k-test-0084=6, gsm8k-test-0104=None, gsm8k-test-0121=None, gsm8k-test-0152=None, gsm8k-test-0197=None, gsm8k-test-0300=None, gsm8k-test-0364=None, gsm8k-test-0383=None, gsm8k-test-0407=56, gsm8k-test-0487=91, gsm8k-test-0584=None, gsm8k-test-0712=18, gsm8k-test-0722=None, gsm8k-test-0752=None, gsm8k-test-0790=40, gsm8k-test-0798=41, gsm8k-test-0872=None, gsm8k-test-0876=None, gsm8k-test-0923=None, gsm8k-test-0924=None, gsm8k-test-0941=21, gsm8k-test-0983=None, gsm8k-test-0985=None, gsm8k-test-1026=None, gsm8k-test-1028=None, gsm8k-test-1033=None, gsm8k-test-1046=None, gsm8k-test-1053=None, gsm8k-test-1058=None, gsm8k-test-1119=None, gsm8k-test-1152=20, gsm8k-test-1260=22, 109.json=None, 1934.json=271, 2593.json=None, 305.json=None, 425.json=None, 567.json=None, 769.json=135, 849.json=None, 858.json=10, 873.json=None, 23957.json=None, 508.json=None, 525.json=None, 803.json=22, 183.json=69, 477.json=None, 965.json=None, 967.json=None, 1166.json=None, 207.json=0, 232.json=25, 754.json=337, 860.json=None, 1128.json=44, 1185.json=42, 255.json=None, 368.json=None, 911.json=None, 1423.json=None, 1761.json=11, 505.json=17, 1199.json=None

NaN in any teacher-forced distribution: False; max |sum p − 1|: 2.6e-15

### BF16 batch-shape noise control (engine FULL, B4 vs B1, R32 FULL continuation)

32 items: max |Δlogprob| median 0.077, max 0.144; mean |Δlogprob| mean 5.12e-03; |Δ mean NLL| max 3.88e-03

## 1.4 Item audit

See `analysis/stage1_item_audit.md` / `.csv`.

| dataset | policy | n | acc | hit cap | cap & loop | cap & answer line | ended, no parse | scorer disagreements | loops (any) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | full | 200 | 0.415 | 3 | 3 | 0 | 0 | 16 | 3 |
| gsm8k | last | 200 | 0.445 | 1 | 1 | 0 | 0 | 18 | 1 |
| gsm8k | average | 200 | 0.420 | 2 | 2 | 0 | 0 | 9 | 2 |
| math500 | full | 200 | 0.155 | 47 | 42 | 0 | 1 | 3 | 53 |
| math500 | last | 200 | 0.165 | 52 | 50 | 0 | 1 | 3 | 60 |
| math500 | average | 200 | 0.125 | 49 | 48 | 0 | 1 | 2 | 55 |


## Incidents

- A first FP32-parity pass used a GPU float64 forward as ground truth and showed |Δlogit| ≈ 10 for both the native and the engine FP32 paths. Root cause: CUDA float64 softmax on this torch build (2.6.0+cu124, H200) is wrong for some row lengths (S=837: rows ≥ 31 off by up to 0.2; `repro_cuda_fp64_softmax.txt`, `scripts/oracle/repro_cuda_fp64_softmax.py`). CUDA FP32/BF16 softmax and SDPA agree with CPU float64, so the BF16/FP32 experiment paths are unaffected. The FP64 reference is now computed on CPU; the invalid result is kept as `parity/superseded/fp_parity_gpu_fp64_INVALID.json`.
- Official lm-eval needed transformers 4.53.3 and batch_size=1 (see 1.1).

## Gate

{"fp32_parity_pass": true, "bf16_greedy_items": 64}

