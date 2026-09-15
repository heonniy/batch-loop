#!/usr/bin/env bash
set -euo pipefail

# Phase 1-3 follow-up on a single H100:
# 1) fair-subset simulator
# 2) larger Ouro pool (default 1024 requests)
# 3) exit-threshold sweeps for Ouro + Huginn
#
# This is TRACE + SIMULATION, not a wall-clock early-exit runtime benchmark.

OURO_N="${OURO_N:-1024}"
HUGINN_N="${HUGINN_N:-512}"
NEW_TOKENS="${NEW_TOKENS:-24}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
DATASET="${DATASET:-alpaca}"
SEED="${SEED:-1}"
BATCH_SIZES="${BATCH_SIZES:-16,32,64,128,256,512}"

mkdir -p results/traces_phase123 results/analysis_phase123

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('bf16:', torch.cuda.is_bf16_supported())
PY

collect_ouro () {
  local Q="$1"
  local TAG="${Q//./}"
  local OUT="results/traces_phase123/ouro14_${DATASET}_q${TAG}_n${OURO_N}.jsonl"
  if [[ -s "$OUT" ]]; then
    echo "[skip] $OUT already exists"
    return
  fi
  echo "=== Ouro-1.4B q=${Q}, N=${OURO_N} ==="
  python poc.py collect-ouro \
    --model ByteDance/Ouro-1.4B \
    --dataset "$DATASET" \
    --num-requests "$OURO_N" \
    --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" \
    --exit-threshold "$Q" \
    --seed "$SEED" \
    --output "$OUT"
}

collect_huginn () {
  local Q="$1"
  local TAG="${Q//./}"
  local OUT="results/traces_phase123/huginn_${DATASET}_d${TAG}_n${HUGINN_N}.jsonl"
  if [[ -s "$OUT" ]]; then
    echo "[skip] $OUT already exists"
    return
  fi
  echo "=== Huginn-3.5B convergence=${Q}, N=${HUGINN_N} ==="
  python poc.py collect-huginn \
    --model tomg-group-umd/huginn-0125 \
    --dataset "$DATASET" \
    --num-requests "$HUGINN_N" \
    --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" \
    --max-steps 16 \
    --criterion latent-diff \
    --exit-threshold "$Q" \
    --min-steps 1 \
    --seed "$SEED" \
    --output "$OUT"
}

# Thresholds follow the CDB paper's exit-distribution sweep.
for q in 0.2 0.4 0.5 0.7; do
  collect_ouro "$q"
done

for q in 0.5 0.35 0.26 0.12; do
  collect_huginn "$q"
done

# Fair analysis: all strategies see the exact same request/token subset before reordering.
python analyze_fair.py \
  --traces results/traces_phase123/ouro14_${DATASET}_q*_n${OURO_N}.jsonl \
           results/traces_phase123/huginn_${DATASET}_d*_n${HUGINN_N}.jsonl \
  --batch-sizes "$BATCH_SIZES" \
  --strategies fifo,random,prev,recent,seqmean,oracle \
  --recent-k 4 \
  --seed "$SEED" \
  --output-dir results/analysis_phase123/fair

# Depth-distribution summary per threshold/run.
python - <<'PY'
import glob, json, os
import pandas as pd

rows=[]
for path in sorted(glob.glob('results/traces_phase123/*.jsonl')):
    data=[]
    with open(path) as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    if not data: continue
    df=pd.DataFrame(data)
    counts=df['depth'].value_counts(normalize=True).sort_index()
    rows.append({
        'run': os.path.basename(path),
        'model': df['model'].iloc[0],
        'threshold': str(df['exit_threshold'].iloc[0]),
        'n_records': len(df),
        'n_requests': df['request_id'].nunique(),
        'mean_depth': df['depth'].mean(),
        'std_depth': df['depth'].std(),
        'min_depth': df['depth'].min(),
        'max_depth': df['depth'].max(),
        'depth_distribution': ';'.join(f'{int(k)}:{v:.6f}' for k,v in counts.items()),
    })
out=pd.DataFrame(rows)
out.to_csv('results/analysis_phase123/depth_distribution.csv', index=False)
print(out.to_string(index=False))
PY

echo
echo "Done. Key files:"
echo "  results/analysis_phase123/fair/oracle_vs_fifo.csv"
echo "  results/analysis_phase123/fair/fairness_validation.csv"
echo "  results/analysis_phase123/depth_distribution.csv"
