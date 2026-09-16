#!/usr/bin/env bash
set -euo pipefail

# One bounded single-H100 experiment block over TWO workloads:
#   - Alpaca: short prompt/output workload
#   - ShareGPT: longer/heavier-tailed prompt/output workload
#
# For each workload:
#   1) fair no-refill headroom on real exit traces
#   2) B = 4,8,16,32,64 only
#   3) Ouro-1.4B, Ouro-2.6B, Huginn-3.5B
#   4) threshold sweeps for every model
#   5) REAL native whole-model E2E/BCT/throughput on the H100
#   6) hardware-calibrated no-refill replay using one shared H100 latency surface
#
# Refill is out of scope. Results are separated by dataset.

SEED="${SEED:-1}"
BATCH_SIZES="4,8,16,32,64"
CALIB_BATCH_SIZES="1,2,4,8,16,32,64"
CALIB_CONTEXTS="${CALIB_CONTEXTS:-32,128,512,2048}"
CALIB_DECODE_TOKENS="${CALIB_DECODE_TOKENS:-6}"
CALIB_REPEATS="${CALIB_REPEATS:-3}"

# Dataset-specific bounded defaults. Override explicitly from the environment if needed.
ALPACA_N_REQUESTS="${ALPACA_N_REQUESTS:-256}"
ALPACA_E2E_N_REQUESTS="${ALPACA_E2E_N_REQUESTS:-128}"
ALPACA_NEW_TOKENS="${ALPACA_NEW_TOKENS:-24}"
ALPACA_PROMPT_TOKENS="${ALPACA_PROMPT_TOKENS:-512}"

SHAREGPT_N_REQUESTS="${SHAREGPT_N_REQUESTS:-128}"
SHAREGPT_E2E_N_REQUESTS="${SHAREGPT_E2E_N_REQUESTS:-64}"
SHAREGPT_NEW_TOKENS="${SHAREGPT_NEW_TOKENS:-64}"
SHAREGPT_PROMPT_TOKENS="${SHAREGPT_PROMPT_TOKENS:-2048}"

# Revisions are bound to known compatible code snapshots from prior evidence.
OURO14_REV="${OURO14_REV:-37bc7e8ff85a90d813dc096383405f521b6a67b0}"
OURO26_REV="${OURO26_REV:-f8e38372d72a8aa7f4ea9f14b98739a311589088}"
HUGINN_REV="${HUGINN_REV:-bb6621b65e90b6a4b9b29ef88dc83866d450470c}"

ROOT="results/full_sweep_$(date +%Y%m%d_%H%M%S)"
CALIB_DIR="$ROOT/calibration"
mkdir -p "$CALIB_DIR"

if [[ -d results/repro ]]; then
  export PYTHONPATH="$(pwd)/results/repro${PYTHONPATH:+:$PYTHONPATH}"
fi

python - <<'PY' | tee "$ROOT/environment.txt"
import platform, torch, transformers, numpy, pandas
print('python:', platform.python_version())
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('bf16:', torch.cuda.is_bf16_supported())
    print('device_count:', torch.cuda.device_count())
PY

git rev-parse HEAD | tee "$ROOT/source_commit.txt"
printf '%q ' "$0" "$@" > "$ROOT/launch_command.txt"; echo >> "$ROOT/launch_command.txt"

cat > "$ROOT/workload_contract.txt" <<EOF
alpaca trace_requests=$ALPACA_N_REQUESTS e2e_requests=$ALPACA_E2E_N_REQUESTS max_prompt_tokens=$ALPACA_PROMPT_TOKENS max_new_tokens=$ALPACA_NEW_TOKENS
sharegpt trace_requests=$SHAREGPT_N_REQUESTS e2e_requests=$SHAREGPT_E2E_N_REQUESTS max_prompt_tokens=$SHAREGPT_PROMPT_TOKENS max_new_tokens=$SHAREGPT_NEW_TOKENS
EOF

snapshot () {
  local REPO="$1"
  local REV="$2"
  python - "$REPO" "$REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
}

OURO14_PATH="$(snapshot ByteDance/Ouro-1.4B "$OURO14_REV")"
OURO26_PATH="$(snapshot ByteDance/Ouro-2.6B "$OURO26_REV")"
HUGINN_PATH="$(snapshot tomg-group-umd/huginn-0125 "$HUGINN_REV")"

cat > "$ROOT/model_revisions.txt" <<EOF
ByteDance/Ouro-1.4B $OURO14_REV $OURO14_PATH
ByteDance/Ouro-2.6B $OURO26_REV $OURO26_PATH
tomg-group-umd/huginn-0125 $HUGINN_REV $HUGINN_PATH
EOF

# ---------------------------------------------------------------------------
# Shared physical H100 calibration. Includes a longer context point for ShareGPT.
# OOM/error cells are retained by runtime_calibrate.py instead of silently dropped.
# ---------------------------------------------------------------------------
echo "=== shared H100 latency calibration: Ouro-1.4B ==="
python runtime_calibrate.py calibrate \
  --model "$OURO14_PATH" --family ouro \
  --batch-sizes "$CALIB_BATCH_SIZES" --context-lengths "$CALIB_CONTEXTS" \
  --depths 1,2,3,4 --decode-tokens "$CALIB_DECODE_TOKENS" --repeats "$CALIB_REPEATS" \
  --output "$CALIB_DIR/ouro14.csv"

echo "=== shared H100 latency calibration: Ouro-2.6B ==="
python runtime_calibrate.py calibrate \
  --model "$OURO26_PATH" --family ouro \
  --batch-sizes "$CALIB_BATCH_SIZES" --context-lengths "$CALIB_CONTEXTS" \
  --depths 1,2,3,4 --decode-tokens "$CALIB_DECODE_TOKENS" --repeats "$CALIB_REPEATS" \
  --output "$CALIB_DIR/ouro26.csv"

echo "=== shared H100 latency calibration: Huginn-3.5B ==="
python runtime_calibrate.py calibrate \
  --model "$HUGINN_PATH" --family huginn \
  --batch-sizes "$CALIB_BATCH_SIZES" --context-lengths "$CALIB_CONTEXTS" \
  --depths 4,8,12,16 --decode-tokens "$CALIB_DECODE_TOKENS" --repeats "$CALIB_REPEATS" \
  --output "$CALIB_DIR/huginn35.csv"

run_dataset () {
  local DATASET="$1"
  local N_REQUESTS E2E_N_REQUESTS NEW_TOKENS PROMPT_TOKENS
  if [[ "$DATASET" == "alpaca" ]]; then
    N_REQUESTS="$ALPACA_N_REQUESTS"
    E2E_N_REQUESTS="$ALPACA_E2E_N_REQUESTS"
    NEW_TOKENS="$ALPACA_NEW_TOKENS"
    PROMPT_TOKENS="$ALPACA_PROMPT_TOKENS"
  elif [[ "$DATASET" == "sharegpt" ]]; then
    N_REQUESTS="$SHAREGPT_N_REQUESTS"
    E2E_N_REQUESTS="$SHAREGPT_E2E_N_REQUESTS"
    NEW_TOKENS="$SHAREGPT_NEW_TOKENS"
    PROMPT_TOKENS="$SHAREGPT_PROMPT_TOKENS"
  else
    echo "unknown dataset: $DATASET" >&2
    exit 2
  fi

  local DROOT="$ROOT/$DATASET"
  local TRACE_DIR="$DROOT/traces"
  local ANALYSIS_DIR="$DROOT/analysis"
  local E2E_DIR="$DROOT/e2e"
  mkdir -p "$TRACE_DIR" "$ANALYSIS_DIR" "$E2E_DIR"

  echo "============================================================"
  echo "DATASET=$DATASET N=$N_REQUESTS E2E_N=$E2E_N_REQUESTS PROMPT_CAP=$PROMPT_TOKENS NEW_TOKEN_CAP=$NEW_TOKENS"
  echo "============================================================"

  collect_ouro_model () {
    local MODEL_PATH="$1"
    local TAG="$2"
    for q in 0.2 0.4 0.5 0.7; do
      local QT="${q//./}"
      echo "=== trace $DATASET ${TAG} q=${q} ==="
      python poc.py collect-ouro \
        --model "$MODEL_PATH" --dataset "$DATASET" \
        --num-requests "$N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
        --max-prompt-tokens "$PROMPT_TOKENS" --exit-threshold "$q" \
        --seed "$SEED" --output "$TRACE_DIR/${TAG}_${DATASET}_q${QT}.jsonl"
    done
  }

  collect_huginn_model () {
    for q in 0.5 0.35 0.26 0.12; do
      local QT="${q//./}"
      echo "=== trace $DATASET huginn35 convergence=${q} ==="
      python poc.py collect-huginn \
        --model "$HUGINN_PATH" --dataset "$DATASET" \
        --num-requests "$N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
        --max-prompt-tokens "$PROMPT_TOKENS" --max-steps 16 \
        --criterion latent-diff --exit-threshold "$q" --min-steps 1 \
        --seed "$SEED" --output "$TRACE_DIR/huginn35_${DATASET}_d${QT}.jsonl"
    done
  }

  # A. Exit traces + fair no-refill headroom.
  collect_ouro_model "$OURO14_PATH" ouro14
  collect_ouro_model "$OURO26_PATH" ouro26
  collect_huginn_model

  python analyze_fair.py \
    --traces "$TRACE_DIR"/*.jsonl --batch-sizes "$BATCH_SIZES" \
    --strategies fifo,random,prev,recent,seqmean,oracle --recent-k 4 \
    --seed "$SEED" --output-dir "$ANALYSIS_DIR/fair"

  python - "$TRACE_DIR" "$ANALYSIS_DIR/depth_distribution.csv" <<'PY'
import glob, json, os, sys
import pandas as pd
trace_dir, out_path = sys.argv[1:3]
rows=[]
for path in sorted(glob.glob(os.path.join(trace_dir, '*.jsonl'))):
    data=[]
    with open(path) as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    if not data: continue
    df=pd.DataFrame(data)
    counts=df.depth.value_counts(normalize=True).sort_index()
    rows.append({
        'run': os.path.basename(path), 'dataset': str(df.dataset.iloc[0]),
        'model': str(df.model.iloc[0]), 'family': str(df.family.iloc[0]) if 'family' in df else '',
        'threshold': str(df.exit_threshold.iloc[0]) if 'exit_threshold' in df else '',
        'records': len(df), 'requests': df.request_id.nunique(),
        'mean_depth': df.depth.mean(), 'std_depth': df.depth.std(),
        'min_depth': df.depth.min(), 'max_depth': df.depth.max(),
        'distribution': ';'.join(f'{int(k)}:{v:.6f}' for k,v in counts.items()),
    })
out=pd.DataFrame(rows)
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
PY

  # B. Actual native whole-model wall-clock/BCT/throughput.
  echo "=== native E2E $DATASET Ouro-1.4B ==="
  python e2e_benchmark.py \
    --model "$OURO14_PATH" --family ouro --dataset "$DATASET" \
    --num-requests "$E2E_N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" --thresholds 0.2,0.4,0.5,0.7 \
    --batch-sizes "$BATCH_SIZES" --seed "$SEED" --output "$E2E_DIR/ouro14_native.csv"

  echo "=== native E2E $DATASET Ouro-2.6B ==="
  python e2e_benchmark.py \
    --model "$OURO26_PATH" --family ouro --dataset "$DATASET" \
    --num-requests "$E2E_N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" --thresholds 0.2,0.4,0.5,0.7 \
    --batch-sizes "$BATCH_SIZES" --seed "$SEED" --output "$E2E_DIR/ouro26_native.csv"

  echo "=== native E2E $DATASET Huginn-3.5B ==="
  python e2e_benchmark.py \
    --model "$HUGINN_PATH" --family huginn --dataset "$DATASET" \
    --num-requests "$E2E_N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
    --max-prompt-tokens "$PROMPT_TOKENS" --thresholds 0.5,0.35,0.26,0.12 \
    --batch-sizes "$BATCH_SIZES" --criterion latent-diff --max-steps 16 --min-steps 1 \
    --seed "$SEED" --output "$E2E_DIR/huginn35_native.csv"

  python - "$E2E_DIR" <<'PY'
import glob, os, sys
import pandas as pd
root=sys.argv[1]
frames=[]
for p in sorted(glob.glob(os.path.join(root,'*_native.csv'))):
    d=pd.read_csv(p); d['source_csv']=os.path.basename(p); frames.append(d)
if frames:
    out=pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(root,'native_e2e_all.csv'), index=False)
PY

  # C. Hardware-calibrated no-refill replay using the shared H100 timing surface.
  python runtime_calibrate.py replay \
    --traces "$TRACE_DIR"/ouro14_*.jsonl --fit "$CALIB_DIR/ouro14_fit.csv" \
    --batch-sizes "$BATCH_SIZES" --output "$ANALYSIS_DIR/runtime_replay_ouro14.csv"
  python runtime_calibrate.py replay \
    --traces "$TRACE_DIR"/ouro26_*.jsonl --fit "$CALIB_DIR/ouro26_fit.csv" \
    --batch-sizes "$BATCH_SIZES" --output "$ANALYSIS_DIR/runtime_replay_ouro26.csv"
  python runtime_calibrate.py replay \
    --traces "$TRACE_DIR"/huginn35_*.jsonl --fit "$CALIB_DIR/huginn35_fit.csv" \
    --batch-sizes "$BATCH_SIZES" --output "$ANALYSIS_DIR/runtime_replay_huginn35.csv"

  python - "$ANALYSIS_DIR" <<'PY'
import glob, os, sys
import pandas as pd
root=sys.argv[1]
paths=sorted(glob.glob(os.path.join(root,'runtime_replay_*.csv')))
frames=[pd.read_csv(p) for p in paths]
if frames:
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(root,'runtime_replay_all.csv'), index=False)
PY
}

run_dataset alpaca
run_dataset sharegpt

# Cross-workload compact tables for review. Dataset-specific raw evidence remains separate.
python - "$ROOT" <<'PY'
import glob, os, sys
import pandas as pd
root=sys.argv[1]
for rel, outname in [
    ('*/e2e/native_e2e_all.csv', 'native_e2e_both_workloads.csv'),
    ('*/analysis/runtime_replay_all.csv', 'runtime_replay_both_workloads.csv'),
    ('*/analysis/fair/oracle_vs_fifo.csv', 'oracle_vs_fifo_both_workloads.csv'),
    ('*/analysis/depth_distribution.csv', 'depth_distribution_both_workloads.csv'),
]:
    frames=[]
    for p in sorted(glob.glob(os.path.join(root, rel))):
        d=pd.read_csv(p)
        if 'dataset' not in d.columns:
            d.insert(0, 'dataset', os.path.normpath(p).split(os.sep)[-3 if '/analysis/' in p else -2])
        frames.append(d)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(root, outname), index=False)
PY

echo
echo "FULL ALPACA + SHAREGPT SWEEP FINISHED: $ROOT"
echo "Publish this run as a result commit before changing the method."
echo "Primary review files:"
echo "  $ROOT/oracle_vs_fifo_both_workloads.csv"
echo "  $ROOT/native_e2e_both_workloads.csv"
echo "  $ROOT/runtime_replay_both_workloads.csv"
echo "  $ROOT/depth_distribution_both_workloads.csv"
