#!/usr/bin/env bash
set -euo pipefail

# One bounded single-H100 experiment block:
#   - fair trace headroom
#   - B = 4,8,16,32,64 only
#   - Ouro-1.4B, Ouro-2.6B, Huginn-3.5B
#   - threshold sweeps for every model
#   - real H100 fixed-depth latency calibration
#   - hardware-calibrated throughput/BCT replay for each threshold/strategy
#
# IMPORTANT: the final hardware replay is NOT yet an integrated dynamic
# no-refill engine. It uses real measured GPU latency surfaces plus real exit
# traces. The result publication must preserve this evidence label.

DATASET="${DATASET:-alpaca}"
SEED="${SEED:-1}"
N_REQUESTS="${N_REQUESTS:-256}"
HUGINN_N_REQUESTS="${HUGINN_N_REQUESTS:-256}"
NEW_TOKENS="${NEW_TOKENS:-24}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
BATCH_SIZES="4,8,16,32,64"
CALIB_BATCH_SIZES="1,2,4,8,16,32,64"
CALIB_CONTEXTS="${CALIB_CONTEXTS:-32,128,512}"
CALIB_DECODE_TOKENS="${CALIB_DECODE_TOKENS:-6}"
CALIB_REPEATS="${CALIB_REPEATS:-3}"

# Revisions are bound to known compatible code snapshots from prior evidence.
OURO14_REV="${OURO14_REV:-37bc7e8ff85a90d813dc096383405f521b6a67b0}"
# Ouro-2.6B old pre-cache-update snapshot; override if the local environment
# validates a newer revision against its Transformers version.
OURO26_REV="${OURO26_REV:-f8e38372d72a8aa7f4ea9f14b98739a311589088}"
HUGINN_REV="${HUGINN_REV:-bb6621b65e90b6a4b9b29ef88dc83866d450470c}"

ROOT="results/full_sweep_$(date +%Y%m%d_%H%M%S)"
TRACE_DIR="$ROOT/traces"
ANALYSIS_DIR="$ROOT/analysis"
CALIB_DIR="$ROOT/calibration"
mkdir -p "$TRACE_DIR" "$ANALYSIS_DIR" "$CALIB_DIR"

# Reuse the tracked Huginn compatibility shim from the first successful PoC.
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

collect_ouro_model () {
  local MODEL_PATH="$1"
  local TAG="$2"
  for q in 0.2 0.4 0.5 0.7; do
    local QT="${q//./}"
    echo "=== trace ${TAG} q=${q} ==="
    python poc.py collect-ouro \
      --model "$MODEL_PATH" \
      --dataset "$DATASET" \
      --num-requests "$N_REQUESTS" \
      --max-new-tokens "$NEW_TOKENS" \
      --max-prompt-tokens "$PROMPT_TOKENS" \
      --exit-threshold "$q" \
      --seed "$SEED" \
      --output "$TRACE_DIR/${TAG}_${DATASET}_q${QT}.jsonl"
  done
}

collect_huginn_model () {
  for q in 0.5 0.35 0.26 0.12; do
    local QT="${q//./}"
    echo "=== trace huginn35 convergence=${q} ==="
    python poc.py collect-huginn \
      --model "$HUGINN_PATH" \
      --dataset "$DATASET" \
      --num-requests "$HUGINN_N_REQUESTS" \
      --max-new-tokens "$NEW_TOKENS" \
      --max-prompt-tokens "$PROMPT_TOKENS" \
      --max-steps 16 \
      --criterion latent-diff \
      --exit-threshold "$q" \
      --min-steps 1 \
      --seed "$SEED" \
      --output "$TRACE_DIR/huginn35_${DATASET}_d${QT}.jsonl"
  done
}

collect_ouro_model "$OURO14_PATH" ouro14
collect_ouro_model "$OURO26_PATH" ouro26
collect_huginn_model

# One fair comparison over every trace.  Every strategy receives exactly the
# same eligible token/request subset before ordering.
python analyze_fair.py \
  --traces "$TRACE_DIR"/*.jsonl \
  --batch-sizes "$BATCH_SIZES" \
  --strategies fifo,random,prev,recent,seqmean,oracle \
  --recent-k 4 \
  --seed "$SEED" \
  --output-dir "$ANALYSIS_DIR/fair"

# Per-threshold exit-depth distributions.
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
        'run': os.path.basename(path),
        'model': str(df.model.iloc[0]),
        'family': str(df.family.iloc[0]) if 'family' in df else '',
        'threshold': str(df.exit_threshold.iloc[0]) if 'exit_threshold' in df else '',
        'records': len(df),
        'requests': df.request_id.nunique(),
        'mean_depth': df.depth.mean(),
        'std_depth': df.depth.std(),
        'min_depth': df.depth.min(),
        'max_depth': df.depth.max(),
        'distribution': ';'.join(f'{int(k)}:{v:.6f}' for k,v in counts.items()),
    })
pd.DataFrame(rows).to_csv(out_path, index=False)
print(pd.DataFrame(rows).to_string(index=False))
PY

# REAL H100 timing calibration.  Internal B=1,2 points exist only so replay can
# price shrunk active batches; the paper-facing batch sweep remains 4..64.
python runtime_calibrate.py calibrate \
  --model "$OURO14_PATH" --family ouro \
  --batch-sizes "$CALIB_BATCH_SIZES" --context-lengths "$CALIB_CONTEXTS" \
  --depths 1,2,3,4 --decode-tokens "$CALIB_DECODE_TOKENS" --repeats "$CALIB_REPEATS" \
  --output "$CALIB_DIR/ouro14.csv"

python runtime_calibrate.py calibrate \
  --model "$OURO26_PATH" --family ouro \
  --batch-sizes "$CALIB_BATCH_SIZES" --context-lengths "$CALIB_CONTEXTS" \
  --depths 1,2,3,4 --decode-tokens "$CALIB_DECODE_TOKENS" --repeats "$CALIB_REPEATS" \
  --output "$CALIB_DIR/ouro26.csv"

python runtime_calibrate.py calibrate \
  --model "$HUGINN_PATH" --family huginn \
  --batch-sizes "$CALIB_BATCH_SIZES" --context-lengths "$CALIB_CONTEXTS" \
  --depths 4,8,12,16 --decode-tokens "$CALIB_DECODE_TOKENS" --repeats "$CALIB_REPEATS" \
  --output "$CALIB_DIR/huginn35.csv"

# Hardware-calibrated throughput/BCT replay for EVERY threshold.
python runtime_calibrate.py replay \
  --traces "$TRACE_DIR"/ouro14_*.jsonl \
  --fit "$CALIB_DIR/ouro14_fit.csv" \
  --batch-sizes "$BATCH_SIZES" \
  --output "$ANALYSIS_DIR/runtime_replay_ouro14.csv"

python runtime_calibrate.py replay \
  --traces "$TRACE_DIR"/ouro26_*.jsonl \
  --fit "$CALIB_DIR/ouro26_fit.csv" \
  --batch-sizes "$BATCH_SIZES" \
  --output "$ANALYSIS_DIR/runtime_replay_ouro26.csv"

python runtime_calibrate.py replay \
  --traces "$TRACE_DIR"/huginn35_*.jsonl \
  --fit "$CALIB_DIR/huginn35_fit.csv" \
  --batch-sizes "$BATCH_SIZES" \
  --output "$ANALYSIS_DIR/runtime_replay_huginn35.csv"

python - "$ANALYSIS_DIR" <<'PY'
import glob, os, sys
import pandas as pd
root=sys.argv[1]
paths=sorted(glob.glob(os.path.join(root,'runtime_replay_*.csv')))
frames=[pd.read_csv(p) for p in paths]
if frames:
    out=pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(root,'runtime_replay_all.csv'), index=False)
    compact=out[out.strategy.isin(['fifo','oracle','prev','recent','seqmean'])]
    cols=['run','batch_size','strategy','hardware_replay_bct_ms','hardware_replay_tokens_per_s','bct_speedup_vs_fifo','throughput_speedup_vs_fifo','evidence_kind']
    print(compact[cols].to_string(index=False))
PY

echo
echo "FULL SWEEP FINISHED: $ROOT"
echo "Publish this run as a result commit before changing the method."
echo "Key evidence:"
echo "  $ANALYSIS_DIR/fair/oracle_vs_fifo.csv"
echo "  $ANALYSIS_DIR/depth_distribution.csv"
echo "  $CALIB_DIR/*_fit.csv"
echo "  $ANALYSIS_DIR/runtime_replay_all.csv"
