#!/usr/bin/env bash
# Parallel Ouro trace + native E2E shard. A shard owns disjoint thresholds.
set -u -o pipefail

MODEL_KEY="${1:?usage: run_ouro_matrix_shard.sh ouro14|ouro26 thresholds}"
THRESHOLDS="${2:?comma-separated thresholds}"
OUT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PY="${PYTHON_BIN:-/home/hwlee/batch-loop/.venv/bin/python}"
export PYTHONPATH="/home/hwlee/batch-loop/results/repro${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
BATCHES="4,8,16,32,64"
SKIP_TRACE_CELLS=",${SKIP_TRACE_CELLS:-},"
mkdir -p "$OUT"/{traces,analysis,e2e,status}

case "$MODEL_KEY" in
  ouro14)
    REV="37bc7e8ff85a90d813dc096383405f521b6a67b0"
    MODEL="/home/hwlee/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/$REV" ;;
  ouro26)
    REV="f8e38372d72a8aa7f4ea9f14b98739a311589088"
    MODEL="/home/hwlee/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B/snapshots/$REV" ;;
  *) echo "unknown model: $MODEL_KEY" >&2; exit 2 ;;
esac

record() {
  local step="$1" rc="$2"
  "$PY" - "$OUT/status/steps.jsonl" "$step" "$rc" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
with p.open("a") as f: f.write(json.dumps({"time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"step":sys.argv[2],"exit_code":int(sys.argv[3])})+"\n")
PY
}

step() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"; local rc=$?
  record "$name" "$rc"
  return 0
}

"$PY" - "$OUT/run_manifest.json" "$MODEL_KEY" "$THRESHOLDS" "$REV" <<'PY'
import datetime,json,os,pathlib,subprocess,sys
def get(*x):
    try:return subprocess.check_output(x,text=True).strip()
    except Exception as e:return f"ERROR:{e}"
pathlib.Path(sys.argv[1]).write_text(json.dumps({
 "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"lane":"ouro-fast-shard",
 "model_key":sys.argv[2],"thresholds":sys.argv[3].split(","),"model_revision":sys.argv[4],
 "source_sha":get("git","rev-parse","HEAD"),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
 "gpu_inventory":get("nvidia-smi","--query-gpu=index,uuid,name,memory.total","--format=csv,noheader"),
 "batch_sizes":[4,8,16,32,64],"datasets":["alpaca","sharegpt"]},indent=2)+"\n")
PY

IFS=',' read -r -a QS <<< "$THRESHOLDS"
for DATASET in alpaca sharegpt; do
  if [[ "$DATASET" == alpaca ]]; then MAX_PROMPT=512; MAX_NEW=64; else MAX_PROMPT=2048; MAX_NEW=128; fi
  TRACE_FILES=()
  for Q in "${QS[@]}"; do
    TAG="${Q//./p}"
    CELL="$DATASET:$Q"
    TRACE="$OUT/traces/${MODEL_KEY}_${DATASET}_q${TAG}.jsonl"
    TRACE_FILES+=("$TRACE")
    if [[ "$SKIP_TRACE_CELLS" == *",$CELL,"* ]]; then
      record "trace_${CELL}_external" 0
      continue
    fi
    START=0
    APPEND_ARGS=()
    RECEIPT="${TRACE%.jsonl}.requests.jsonl"
    if [[ -s "$RECEIPT" ]]; then
      START=$(wc -l < "$RECEIPT")
      APPEND_ARGS=(--append --start-request-id "$START")
    fi
    step "trace_${CELL}" "$PY" poc.py collect-ouro --model "$MODEL" --dataset "$DATASET" \
      --num-requests 256 --max-new-tokens "$MAX_NEW" --max-prompt-tokens "$MAX_PROMPT" \
      --exit-threshold "$Q" --seed 1 --output "$TRACE" "${APPEND_ARGS[@]}"
  done

  EXISTING=()
  for f in "${TRACE_FILES[@]}"; do [[ -s "$f" ]] && EXISTING+=("$f"); done
  if ((${#EXISTING[@]})); then
    step "fair_${DATASET}" "$PY" analyze_fair.py --traces "${EXISTING[@]}" --batch-sizes "$BATCHES" \
      --output-dir "$OUT/analysis/$DATASET"
  fi

  step "native_e2e_${DATASET}" "$PY" e2e_benchmark.py --model "$MODEL" --model-revision "$REV" \
    --family ouro --dataset "$DATASET" --num-requests 256 --max-new-tokens "$MAX_NEW" \
    --max-prompt-tokens "$MAX_PROMPT" --thresholds "$THRESHOLDS" --batch-sizes "$BATCHES" \
    --seed 1 --fixed-output-tokens --no-validate-parity \
    --output "$OUT/e2e/${MODEL_KEY}_${DATASET}_native.csv"
done
record shard_complete 0
