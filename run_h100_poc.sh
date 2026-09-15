#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
RUN_OURO_26B="${RUN_OURO_26B:-0}"
RUN_SHAREGPT="${RUN_SHAREGPT:-0}"
RUN_ALT_THRESHOLDS="${RUN_ALT_THRESHOLDS:-0}"

if [[ "$MODE" == "smoke" ]]; then
  OURO_N=8
  HUGINN_N=4
  NEW_TOKENS=8
  PROMPT_TOKENS=256
elif [[ "$MODE" == "poc" ]]; then
  # Large offline pool is intentional: one collected trace can simulate many
  # micro-batch sizes. Ouro reaches B=256; Huginn reaches B=128 by default.
  OURO_N=256
  HUGINN_N=128
  NEW_TOKENS=24
  PROMPT_TOKENS=512
else
  echo "usage: $0 [smoke|poc]"
  exit 2
fi

mkdir -p results/traces results/analysis

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY

# Analysis-path sanity check requires no model download.
python poc.py synthetic --tokens 12 --output results/traces/synthetic.jsonl
python poc.py analyze \
  --traces results/traces/synthetic.jsonl \
  --batch-sizes 4,8,16,32,64 \
  --output-dir results/analysis/synthetic

run_dataset () {
  local DATASET="$1"
  local TAG="$2"

  echo "=== Ouro-1.4B / ${DATASET} ==="
  python poc.py collect-ouro \
    --model ByteDance/Ouro-1.4B \
    --dataset "$DATASET" \
    --num-requests "$OURO_N" \
    --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" \
    --exit-threshold 0.2 \
    --output "results/traces/ouro14_${TAG}_q02.jsonl"

  if [[ "$RUN_ALT_THRESHOLDS" == "1" ]]; then
    python poc.py collect-ouro \
      --model ByteDance/Ouro-1.4B \
      --dataset "$DATASET" \
      --num-requests "$OURO_N" \
      --max-new-tokens "$NEW_TOKENS" \
      --max-prompt-tokens "$PROMPT_TOKENS" \
      --exit-threshold 0.5 \
      --output "results/traces/ouro14_${TAG}_q05.jsonl"
  fi

  if [[ "$RUN_OURO_26B" == "1" ]]; then
    echo "=== Ouro-2.6B / ${DATASET} ==="
    python poc.py collect-ouro \
      --model ByteDance/Ouro-2.6B \
      --dataset "$DATASET" \
      --num-requests "$OURO_N" \
      --max-new-tokens "$NEW_TOKENS" \
      --max-prompt-tokens "$PROMPT_TOKENS" \
      --exit-threshold 0.2 \
      --output "results/traces/ouro26_${TAG}_q02.jsonl"
  fi

  echo "=== Huginn-3.5B / ${DATASET} ==="
  python poc.py collect-huginn \
    --model tomg-group-umd/huginn-0125 \
    --dataset "$DATASET" \
    --num-requests "$HUGINN_N" \
    --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" \
    --max-steps 16 \
    --criterion latent-diff \
    --exit-threshold 0.12 \
    --min-steps 1 \
    --output "results/traces/huginn_${TAG}_d012.jsonl"

  if [[ "$RUN_ALT_THRESHOLDS" == "1" ]]; then
    python poc.py collect-huginn \
      --model tomg-group-umd/huginn-0125 \
      --dataset "$DATASET" \
      --num-requests "$HUGINN_N" \
      --max-new-tokens "$NEW_TOKENS" \
      --max-prompt-tokens "$PROMPT_TOKENS" \
      --max-steps 16 \
      --criterion latent-diff \
      --exit-threshold 0.22 \
      --min-steps 1 \
      --output "results/traces/huginn_${TAG}_d022.jsonl"
  fi
}

run_dataset alpaca alpaca

if [[ "$RUN_SHAREGPT" == "1" ]]; then
  run_dataset sharegpt sharegpt
fi

# Large batch sizes are cheap to sweep after trace collection. If there are fewer
# live requests than B at later token positions, those incomplete micro-batches are
# intentionally excluded from the metric.
TRACE_ARGS=(results/traces/ouro14_*.jsonl results/traces/huginn_*.jsonl)
if [[ "$RUN_OURO_26B" == "1" ]]; then
  TRACE_ARGS+=(results/traces/ouro26_*.jsonl)
fi

python poc.py analyze \
  --traces "${TRACE_ARGS[@]}" \
  --batch-sizes 4,8,16,32,64,128,256 \
  --strategies fifo,random,prev,recent,seqmean,oracle \
  --recent-k 4 \
  --output-dir "results/analysis/${MODE}"

echo

echo "Done. Inspect: results/analysis/${MODE}/summary.csv"
echo "Most important first question: oracle_headroom_pp over FIFO across B and models."
