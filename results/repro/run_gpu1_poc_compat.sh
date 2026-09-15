#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$REPRO_DIR/../.." && pwd)"
OURO_REVISION="37bc7e8ff85a90d813dc096383405f521b6a67b0"

export PYTHONPATH="$REPRO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p results/traces results/analysis

OURO_MODEL_PATH="$(python - "$OURO_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download("ByteDance/Ouro-1.4B", revision=sys.argv[1]))
PY
)"

python poc.py synthetic \
  --tokens 12 \
  --output results/traces/synthetic.jsonl

python poc.py analyze \
  --traces results/traces/synthetic.jsonl \
  --batch-sizes 4,8,16,32,64 \
  --output-dir results/analysis/synthetic

python poc.py collect-ouro \
  --model "$OURO_MODEL_PATH" \
  --dataset alpaca \
  --num-requests 256 \
  --max-new-tokens 24 \
  --max-prompt-tokens 512 \
  --exit-threshold 0.2 \
  --output results/traces/ouro14_alpaca_q02.jsonl

python poc.py collect-huginn \
  --model tomg-group-umd/huginn-0125 \
  --dataset alpaca \
  --num-requests 128 \
  --max-new-tokens 24 \
  --max-prompt-tokens 512 \
  --max-steps 16 \
  --criterion latent-diff \
  --exit-threshold 0.12 \
  --min-steps 1 \
  --output results/traces/huginn_alpaca_d012.jsonl

python poc.py analyze \
  --traces results/traces/ouro14_alpaca_q02.jsonl results/traces/huginn_alpaca_d012.jsonl \
  --batch-sizes 4,8,16,32,64,128,256 \
  --strategies fifo,random,prev,recent,seqmean,oracle \
  --recent-k 4 \
  --output-dir results/analysis/poc
