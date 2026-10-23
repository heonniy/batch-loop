#!/usr/bin/env bash
# Stage 1.1 -- official Huginn lm-eval GSM8K protocol, run separately from the project runner.
# Command = seal-rg/recurrent-pretraining README "w/ sys. prompt" invocation (gsm8k_cot, chat template,
# fewshot_as_multiturn, system instruction, bf16, mean_recurrence=32) on the full 1319-item test set, plus:
#   * revision pin (== HF main at run time), --log_samples/--output_path (receipts), 8-way data parallel;
#   * batch_size=1 instead of README "auto". On H200, auto picked 64 and OOMed during generation
#     (logs/A_auto_full.FAILED_auto64_oom.log); a fixed 16 also OOMed on the first (longest) batch
#     (logs/A_b16_full.FAILED_oom.log): the native HuginnDynamicCache keeps per-token tensors for all
#     132 prelude/core/coda slots. Batch 1 also avoids left padding, which the native forward does not
#     mask (attention_mask is ignored and RoPE positions shift by the pad length).
# The frozen GSM8K-200 subset is evaluated as a slice of this run (doc_id = test index).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
OUT="$RUN_DIR/official"
mkdir -p "$OUT/logs"
REV=bb6621b65e90b6a4b9b29ef88dc83866d450470c
MARGS="pretrained=tomg-group-umd/huginn-0125,revision=$REV,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32"
SYS="You are a helpful assistant that can assist users with mathematical reasoning."
# The pinned Huginn remote code's HuginnDynamicCache assigns self.key_cache, which is a read-only
# property from transformers 4.54 on (logs/A_auto_full.FAILED_tf4.54.1.log); native cached generation
# therefore needs transformers 4.53.3 (the version in the checkpoint's generation_config.json).
"$PYTHON" -m pip freeze > "$OUT/pip_freeze.txt"
"$PYTHON" -c "import transformers; assert transformers.__version__ == '4.53.3', transformers.__version__"
ACC=("$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision no --dynamo_backend no -m lm_eval)
if [ ! -e "$OUT/A_b1_full/.done" ]; then
  "${ACC[@]}" --model hf --model_args "$MARGS" --tasks gsm8k_cot --batch_size=1 \
    --apply_chat_template=True --fewshot_as_multiturn --system_instruction="$SYS" \
    --log_samples --output_path "$OUT/A_b1_full" > "$OUT/logs/A_b1_full.log" 2>&1
  touch "$OUT/A_b1_full/.done"
fi
echo "official eval finished: $OUT"
