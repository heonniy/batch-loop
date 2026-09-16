#!/usr/bin/env bash
# One isolated Huginn policy/capacity gate lane.
set -u -o pipefail
POLICY="${1:?usage: run_huginn_gate_lane.sh full|latest-m4|latest-m4-compress-s4}"
OUT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PY="${PYTHON_BIN:-/home/hwlee/batch-loop/.venv/bin/python}"
export PYTHONPATH="/home/hwlee/batch-loop/results/repro${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
REV="bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MODEL="/home/hwlee/.cache/huggingface/hub/models--tomg-group-umd--huginn-0125/snapshots/$REV"
mkdir -p "$OUT"/{capacity,parity,status}

run_keep() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"; local rc=$?
  "$PY" - "$OUT/status/steps.jsonl" "$name" "$rc" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
with p.open("a") as f: f.write(json.dumps({"time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"step":sys.argv[2],"exit_code":int(sys.argv[3])})+"\n")
PY
  return 0
}

"$PY" - "$OUT/run_manifest.json" "$POLICY" "$REV" <<'PY'
import datetime,json,os,pathlib,subprocess,sys
def get(*x):
 try:return subprocess.check_output(x,text=True).strip()
 except Exception as e:return f"ERROR:{e}"
pathlib.Path(sys.argv[1]).write_text(json.dumps({"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"lane":"huginn-kv-gate","kv_policy":sys.argv[2],"model_revision":sys.argv[3],"source_sha":get("git","rev-parse","HEAD"),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu":get("nvidia-smi","--query-gpu=index,uuid,name,memory.total","--format=csv,noheader")},indent=2)+"\n")
PY

# Decision cells first so the canonical main lane need not wait for the full surface.
run_keep target_capacity "$PY" runtime_calibrate.py calibrate --model "$MODEL" --family huginn \
  --revision "$REV" --cache-lookup-strategy "$POLICY" --batch-sizes 64 --context-lengths 2048 \
  --depths 16 --decode-tokens 2 --repeats 1 --output "$OUT/capacity/target_b64_l2048_d16.csv"
run_keep parity_probe "$PY" e2e_benchmark.py --model "$MODEL" --model-revision "$REV" --family huginn \
  --dataset alpaca --num-requests 4 --max-new-tokens 16 --max-prompt-tokens 512 --thresholds 0.26 \
  --batch-sizes 4 --criterion latent-diff --max-steps 16 --cache-lookup-strategy "$POLICY" \
  --fixed-output-tokens --output "$OUT/parity/b4_vs_b1.csv"

if [[ "$POLICY" == "latest-m4" ]]; then
  run_keep b1_policy_output_probe "$PY" huginn_policy_probe.py --model "$MODEL" --model-revision "$REV" \
    --policies full,latest-m4 --dataset alpaca --num-requests 4 --max-prompt-tokens 2048 \
    --max-new-tokens 64 --threshold 0.26 --max-steps 16 --output "$OUT/parity/b1_full_vs_latest_m4.json"
fi

run_keep capacity_surface "$PY" runtime_calibrate.py calibrate --model "$MODEL" --family huginn \
  --revision "$REV" --cache-lookup-strategy "$POLICY" --batch-sizes 1,2,4,8,16,32,64 \
  --context-lengths 128,512,2048 --depths 4,8,12,16 --decode-tokens 6 --repeats 2 \
  --output "$OUT/capacity/surface.csv"
