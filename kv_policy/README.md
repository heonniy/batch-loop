# kv_policy — fixed-depth FULL / LAST / AVERAGE recurrent-KV reproduction (H200 x8)

Scope: `FIXED_DEPTH_SCOPE.md` (Stages 1–3 only, fixed native depth, no early exit).
Results and the analysis report: `RESULTS.md`.

## Layout

| path | content |
| --- | --- |
| `kvp/engine.py` | reference engine: Ouro / Huginn backbones on the pinned HF weights, policy KV stores (`PolicyKV` broadcast layout used for all quality runs, `CompactPolicyKV` policy-sized storage) |
| `kvp/reference.py` | independent policy reference: native HF forward + dictionary cache materialising each policy from its definition |
| `kvp/prompts.py` | lm-eval (0.4.9.1) prompt rendering, per-model protocol + provenance, NLL gold-target definition |
| `kvp/grading.py` | GSM8K numeric parser, MATH-500 balanced `\boxed{}` / minerva extraction, math_verify equivalence, lm-eval reference scorers |
| `kvp/data.py` | pinned dataset revisions, seed-fixed subset selection (GSM8K random 200/1319, MATH-500 subject x level stratified 200/500) |
| `kvp/jobs.py` | deterministic batching and per-job execution (one JSONL per job) |
| `tests/` | parser fixtures, policy semantics vs reference (tiny fp64 models), compact-store equivalence |
| `scripts/stage1_parity.py` | Stage 1 on real checkpoints |
| `scripts/make_quality_jobs.py`, `scripts/quality_worker.py`, `scripts/analyze_quality.py` | Stage 2 |
| `scripts/stage3_serving.py` | Stage 3 (vllm-rlt) |
| `results/<UTC>_<sha>_stage{1,2,3}_*` | raw outputs, logs, manifests |

## Environment (as executed)

```bash
# user-space tools (no root on the host)
conda create -p ~/ddn/envs/tools -c conda-forge tmux        # + nvitop (pip), gcc/gxx 13 (conda-forge) for Triton
conda create -p ~/ddn/envs/kvp python=3.11
~/ddn/envs/kvp/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~/ddn/envs/kvp/bin/pip install transformers==4.54.1 "accelerate>=1.2" "datasets>=3.0" "numpy>=2.0,<2.3" pandas scipy \
    sympy huggingface_hub pytest pynvml lm_eval==0.4.9.1 "antlr4-python3-runtime==4.11.0" "math-verify[antlr4_11_0]"
git clone https://github.com/ThinkFlowLab/vllm-rlt ~/vllm-rlt && git -C ~/vllm-rlt checkout 4a350344027960a48384626aa646f1765fda6d4f
~/ddn/envs/kvp/bin/pip install --no-deps -e ~/vllm-rlt && ~/ddn/envs/kvp/bin/pip install "aiohttp>=3.11,<4"
```

Pinned weights: Ouro-1.4B `574fa66c…`, Ouro-2.6B `1ed04250…`, huginn-0125 `bb6621b6…`;
datasets openai/gsm8k `740312ad…`, HuggingFaceH4/MATH-500 `6e4ed1a2…` (full SHAs in `kvp/specs.py`, `kvp/data.py`).
Long jobs were run inside tmux (the host drops interactive sessions).

## Reproduce

```bash
cd kv_policy
PY=~/ddn/envs/kvp/bin/python

# Stage 1 — CPU semantics tests, then real-checkpoint parity (one GPU per model)
CUDA_VISIBLE_DEVICES="" $PY -m pytest -v tests -p no:cacheprovider
CUDA_VISIBLE_DEVICES=0 $PY scripts/stage1_parity.py --model ouro-1.4b   --out results/<dir>/parity_ouro-1.4b.json
CUDA_VISIBLE_DEVICES=1 $PY scripts/stage1_parity.py --model ouro-2.6b   --out results/<dir>/parity_ouro-2.6b.json
CUDA_VISIBLE_DEVICES=2 $PY scripts/stage1_parity.py --model huginn-0125 --out results/<dir>/parity_huginn-0125.json

# Stage 2 — freeze prompts/batches/jobs, then 8 independent single-GPU workers sharing one queue
$PY scripts/make_quality_jobs.py --run-dir results/<dir>
for g in 0 1 2 3 4 5 6 7; do
  tmux new -d -s q$g "CUDA_VISIBLE_DEVICES=$g $PY scripts/quality_worker.py --run-dir results/<dir> --worker $g"
done
$PY scripts/analyze_quality.py --run-dir results/<dir>

# Stage 3 — vllm-rlt (needs a C compiler for Triton: CC=~/ddn/envs/tools/bin/gcc)
CC=~/ddn/envs/tools/bin/gcc CUDA_VISIBLE_DEVICES=0 $PY scripts/stage3_serving.py \
   --model ouro-1.4b --layout last_exited --stage2-dir results/<stage2 dir> --out results/<dir>/serving_....json
```

The exact command lines, source SHAs, package versions and GPU UUIDs of the published runs are in
each run directory (`run_manifest.json`, per-job `job_meta`, `run_gpu.sh`, logs).
