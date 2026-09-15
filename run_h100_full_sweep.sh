#!/usr/bin/env bash
set -euo pipefail

# One bounded single-H100 experiment block:
#   1) fair no-refill headroom on real exit traces
#   2) B = 4,8,16,32,64 only
#   3) Ouro-1.4B, Ouro-2.6B, Huginn-3.5B
#   4) threshold sweeps for every model
#   5) REAL native whole-model E2E/BCT/throughput on the H100
#   6) recurrent-step latency calibration + hardware-calibrated no-refill replay
#
# Refill is out of scope.
#
# Evidence boundary:
# - native E2E rows are actual wall-clock model.generate measurements;
# - Ouro's public HF code still computes all recurrent steps and exposes exit
#   decisions post-hoc, so its threshold-dependent native E2E is a whole-model
#   baseline, not a physical early-exit speedup;
# - Huginn native adaptive timing is trusted only where deterministic batched-vs-B1
#   parity passes;
# - hardware replay prices the observed no-refill batch-shrink trace using measured
#   H100 timing surfaces; it is not an integrated compaction engine.

DATASET="${DATASET:-alpaca}"
SEED="${SEED:-1}"
N_REQUESTS="${N_REQUESTS:-256}"
HUGINN_N_REQUESTS="${HUGINN_N_REQUESTS:-256}"
E2E_N_REQUESTS="${E2E_N_REQUESTS:-128}"
NEW_TOKENS="${NEW_TOKENS:-24}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
BATCH_SIZES="4,8,16,32,64"
CALIB_BATCH_SIZES="1,2,4,8,16,32,64"
CALIB_CONTEXTS="${CALIB_CONTEXTS:-32,128,512}"
CALIB_DECODE_TOKENS="${CALIB_DECODE_TOKENS:-6}"
CALIB_REPEATS="${CALIB_REPEATS:-3}"

# Revisions are bound to known compatible code snapshots from prior evidence.
OURO14_REV="${OURO14_REV:-37bc7e8ff85a90d813dc096383405f521b6a67b0}"
OURO26_REV="${OURO26_REV:-f8e38372d72a8aa7f4ea9f14b98739a311589088}"
HUGINN_REV="${HUGINN_REV:-bb6621b65e90b6a4b9b29ef88dc83866d450470c}"

ROOT="results/full_sweep_$(date +%Y%m%d_%H%M%S)"
TRACE_DIR="$ROOT/traces"
ANALYSIS_DIR="$ROOT/analysis"
CALIB_DIR="$ROOT/calibration"
E2E_DIR="$ROOT/e2e"
mkdir -p "$TRACE_DIR" "$ANALYSIS_DIR" "$CALIB_DIR" "$E2E_DIR"

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

# ---------------------------------------------------------------------------
# A. REAL exit-depth traces + fair no-refill headroom.
# ---------------------------------------------------------------------------
collect_ouro_model "$OURO14_PATH" ouro14
collect_ouro_model "$OURO26_PATH" ouro26
collect_huginn_model

python analyze_fair.py \
  --traces "$TRACE_DIR"/*.jsonl \
  --batch-sizes "$BATCH_SIZES" \
  --strategies fifo,random,prev,recent,seqmean,oracle \
  --recent-k 4 \
  --seed "$SEED" \
  --output-dir "$ANALYSIS_DIR/fair"

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

# ---------------------------------------------------------------------------
# B. ACTUAL whole-model E2E/BCT/throughput using official/native generation.
#    This answers: how expensive is the complete model at each B/threshold?
# ---------------------------------------------------------------------------
echo "=== native E2E Ouro-1.4B ==="
python e2e_benchmark.py \
  --model "$OURO14_PATH" --family ouro --dataset "$DATASET" \
  --num-requests "$E2E_N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
  --max-prompt-tokens "$PROMPT_TOKENS" \
  --thresholds 0.2,0.4,0.5,0.7 --batch-sizes "$BATCH_SIZES" \
  --seed "$SEED" --output "$E2E_DIR/ouro14_native.csv"

echo "=== native E2E Ouro-2.6B ==="
python e2e_benchmark.py \
  --model "$OURO26_PATH" --family ouro --dataset "$DATASET" \
  --num-requests "$E2E_N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
  --max-prompt-tokens "$PROMPT_TOKENS" \
  --thresholds 0.2,0.4,0.5,0.7 --batch-sizes "$BATCH_SIZES" \
  --seed "$SEED" --output "$E2E_DIR/ouro26_native.csv"

echo "=== native E2E Huginn-3.5B ==="
python e2e_benchmark.py \
  --model "$HUGINN_PATH" --family huginn --dataset "$DATASET" \
  --num-requests "$E2E_N_REQUESTS" --max-new-tokens "$NEW_TOKENS" \
  --max-prompt-tokens "$PROMPT_TOKENS" \
  --thresholds 0.5,0.35,0.26,0.12 --batch-sizes "$BATCH_SIZES" \
  --criterion latent-diff --max-steps 16 --min-steps 1 \
  --seed "$SEED" --output "$E2E_DIR/huginn35_native.csv"

python - "$E2E_DIR" <<'PY'
import glob, os, sys
import pandas as pd
root=sys.argv[1]
frames=[]
for p in sorted(glob.glob(os.path.join(root,'*_native.csv'))):
    d=pd.read_csv(p)
    d['source_csv']=os.path.basename(p)
    frames.append(d)
if frames:
    out=pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(root,'native_e2e_all.csv'), index=False)
    cols=['source_csv','threshold','batch_size','completed_requests','generated_tokens',
          'offline_bct_e2e_s','e2e_generated_tokens_per_s','request_throughput_per_s',
          'parity_ok','adaptive_compute_physically_realized','evidence_kind']
    print(out[cols].to_string(index=False))
PY

# ---------------------------------------------------------------------------
# C. REAL H100 recurrent timing surface + hardware-calibrated no-refill replay.
#    Internal B=1,2 price shrunk survivor batches; paper-facing B is still 4..64.
# ---------------------------------------------------------------------------
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
    out=pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(root,'runtime_replay_all.csv'), index=False)
    compact=out[out.strategy.isin(['fifo','oracle','prev','recent','seqmean'])]
    cols=['run','batch_size','strategy','hardware_replay_bct_ms','hardware_replay_tokens_per_s',
          'bct_speedup_vs_fifo','throughput_speedup_vs_fifo','evidence_kind']
    print(compact[cols].to_string(index=False))
PY

echo
echo "FULL SWEEP FINISHED: $ROOT"
echo "Publish this run as a result commit before changing the method."
echo "Key evidence:"
echo "  $ANALYSIS_DIR/fair/oracle_vs_fifo.csv       # trace headroom"
echo "  $ANALYSIS_DIR/depth_distribution.csv       # threshold/depth distribution"
echo "  $E2E_DIR/native_e2e_all.csv                 # ACTUAL whole-model E2E/BCT/throughput"
echo "  $CALIB_DIR/*_fit.csv                        # ACTUAL H100 latency surface"
echo "  $ANALYSIS_DIR/runtime_replay_all.csv        # no-refill hardware replay"
