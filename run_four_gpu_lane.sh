#!/usr/bin/env bash
# One isolated main-model lane from FOUR_GPU_REPLAN_20260916.md.
set -u -o pipefail

LANE="${1:?usage: run_four_gpu_lane.sh ouro14|ouro26|huginn}"
OUT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the lane result directory}"
PY="${PYTHON_BIN:-/home/hwlee/batch-loop/.venv/bin/python}"
REPRO="${REPRO_DIR:-/home/hwlee/batch-loop/results/repro}"
export PYTHONPATH="$REPRO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

OURO14_REV="37bc7e8ff85a90d813dc096383405f521b6a67b0"
OURO26_REV="f8e38372d72a8aa7f4ea9f14b98739a311589088"
HUGINN_REV="bb6621b65e90b6a4b9b29ef88dc83866d450470c"
HF_ROOT="${HF_ROOT:-/home/hwlee/.cache/huggingface/hub}"
BATCHES="4,8,16,32,64"
CAL_BATCHES="1,2,4,8,16,32,64"
mkdir -p "$OUT"/{calibration,traces,analysis,e2e,status}
STEPS="$OUT/status/steps.jsonl"

case "$LANE" in
  ouro14)
    FAMILY=ouro; MODEL_REV="$OURO14_REV"; MODEL="$HF_ROOT/models--ByteDance--Ouro-1.4B/snapshots/$MODEL_REV"
    THRESHOLDS="0.2,0.4,0.5,0.7"; DEPTHS="1,2,3,4"; KV="native-reference"; RMAX=4 ;;
  ouro26)
    FAMILY=ouro; MODEL_REV="$OURO26_REV"; MODEL="$HF_ROOT/models--ByteDance--Ouro-2.6B/snapshots/$MODEL_REV"
    THRESHOLDS="0.2,0.4,0.5,0.7"; DEPTHS="1,2,3,4"; KV="native-reference"; RMAX=4 ;;
  huginn)
    FAMILY=huginn; MODEL_REV="$HUGINN_REV"; MODEL="$HF_ROOT/models--tomg-group-umd--huginn-0125/snapshots/$MODEL_REV"
    THRESHOLDS="0.5,0.35,0.26,0.12"; DEPTHS="4,8,12,16"; KV="${HUGINN_KV_POLICY:?set selected HUGINN_KV_POLICY}"; RMAX=16 ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac

record_step() {
  local name="$1" rc="$2"
  "$PY" - "$STEPS" "$name" "$rc" <<'PY'
import datetime, json, pathlib, sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
with p.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "step": sys.argv[2], "exit_code": int(sys.argv[3])})+"\n")
PY
}

step() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"
  local rc=$?
  record_step "$name" "$rc"
  return 0
}

"$PY" - "$OUT/run_manifest.json" "$LANE" "$MODEL" "$MODEL_REV" "$FAMILY" "$KV" "$RMAX" <<'PY'
import datetime, json, os, pathlib, subprocess, sys
path=pathlib.Path(sys.argv[1])
def get(*cmd):
    try: return subprocess.check_output(cmd, text=True).strip()
    except Exception as e: return f"ERROR:{e}"
payload={"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "lane":sys.argv[2],"model_path":sys.argv[3],"model_revision":sys.argv[4],"family":sys.argv[5],
 "kv_policy":sys.argv[6],"rmax":int(sys.argv[7]),"source_sha":get("git","rev-parse","HEAD"),
 "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
 "gpu":get("nvidia-smi","--query-gpu=index,uuid,name,memory.total","--format=csv,noheader"),
 "workloads":{"alpaca":{"trace_requests":256,"e2e_requests":256,"prompt_cap":512,"decode_tokens":64},
              "sharegpt":{"trace_requests":256,"e2e_requests":256,"prompt_cap":2048,"decode_tokens":128}},
 "thresholds":os.environ.get("THRESHOLDS_OVERRIDE")}
path.write_text(json.dumps(payload,indent=2)+"\n")
PY

CAL_ARGS=(calibrate --model "$MODEL" --family "$FAMILY" --revision "$MODEL_REV"
  --batch-sizes "$CAL_BATCHES" --context-lengths "128,512,2048" --depths "$DEPTHS"
  --decode-tokens 6 --repeats 2 --output "$OUT/calibration/${LANE}.csv")
if [[ "$FAMILY" == huginn ]]; then CAL_ARGS+=(--cache-lookup-strategy "$KV"); fi
step calibration "$PY" runtime_calibrate.py "${CAL_ARGS[@]}"

HUGINN_NATIVE_E2E_ALLOWED=1
if [[ "$FAMILY" == huginn ]]; then
  step adaptive_batch_parity_gate "$PY" e2e_benchmark.py --model "$MODEL" --model-revision "$MODEL_REV" \
    --family huginn --dataset alpaca --num-requests 4 --max-new-tokens 16 --max-prompt-tokens 512 \
    --thresholds 0.26 --batch-sizes 4 --criterion latent-diff --max-steps 16 --min-steps 1 \
    --check-every 1 --cache-lookup-strategy "$KV" --fixed-output-tokens \
    --output "$OUT/status/adaptive_batch_parity_gate.csv"
  HUGINN_NATIVE_E2E_ALLOWED=$("$PY" - "$OUT/status/adaptive_batch_parity_gate.csv" <<'PY'
import pathlib, sys
import pandas as pd
p=pathlib.Path(sys.argv[1])
if not p.exists(): print(0)
else:
    d=pd.read_csv(p)
    ok=(not d.empty and d["parity_ok"].astype(str).str.lower().eq("true").all())
    print(1 if ok else 0)
PY
  )
fi

IFS=',' read -r -a QS <<< "$THRESHOLDS"
for DATASET in alpaca sharegpt; do
  if [[ "$DATASET" == alpaca ]]; then MAX_PROMPT=512; MAX_NEW=64; else MAX_PROMPT=2048; MAX_NEW=128; fi
  TRACE_FILES=()
  for Q in "${QS[@]}"; do
    TAG="${Q//./p}"
    TRACE="$OUT/traces/${LANE}_${DATASET}_q${TAG}.jsonl"
    TRACE_FILES+=("$TRACE")
    if [[ "$FAMILY" == ouro ]]; then
      step "trace_${DATASET}_${Q}" "$PY" poc.py collect-ouro --model "$MODEL" --dataset "$DATASET" \
        --num-requests 256 --max-new-tokens "$MAX_NEW" --max-prompt-tokens "$MAX_PROMPT" \
        --exit-threshold "$Q" --seed 1 --output "$TRACE"
    else
      step "trace_${DATASET}_${Q}" "$PY" poc.py collect-huginn --model "$MODEL" --dataset "$DATASET" \
        --num-requests 256 --max-new-tokens "$MAX_NEW" --max-prompt-tokens "$MAX_PROMPT" \
        --criterion latent-diff --exit-threshold "$Q" --max-steps 16 --min-steps 1 --check-every 1 \
        --cache-lookup-strategy "$KV" --seed 1 --output "$TRACE"
    fi
  done
  EXISTING=()
  for f in "${TRACE_FILES[@]}"; do [[ -s "$f" ]] && EXISTING+=("$f"); done
  if ((${#EXISTING[@]})); then
    step "fair_${DATASET}" "$PY" analyze_fair.py --traces "${EXISTING[@]}" --batch-sizes "$BATCHES" \
      --output-dir "$OUT/analysis/${DATASET}"
    if [[ -s "$OUT/calibration/${LANE}_fit.csv" ]]; then
      step "replay_${DATASET}" "$PY" runtime_calibrate.py replay --traces "${EXISTING[@]}" \
        --fit "$OUT/calibration/${LANE}_fit.csv" --batch-sizes "$BATCHES" \
        --output "$OUT/analysis/${DATASET}/runtime_replay.csv"
    fi
  fi

  E2E_ARGS=(--model "$MODEL" --model-revision "$MODEL_REV" --family "$FAMILY" --dataset "$DATASET"
    --num-requests 256 --max-new-tokens "$MAX_NEW" --max-prompt-tokens "$MAX_PROMPT"
    --thresholds "$THRESHOLDS" --batch-sizes "$BATCHES" --seed 1 --fixed-output-tokens
    --output "$OUT/e2e/${LANE}_${DATASET}_native.csv")
  if [[ "$FAMILY" == huginn ]]; then
    E2E_ARGS+=(--criterion latent-diff --max-steps 16 --min-steps 1 --check-every 1 --cache-lookup-strategy "$KV")
  fi
  if [[ "$FAMILY" != huginn || "$HUGINN_NATIVE_E2E_ALLOWED" == 1 ]]; then
    step "native_e2e_${DATASET}" "$PY" e2e_benchmark.py "${E2E_ARGS[@]}"
  else
    record_step "native_e2e_${DATASET}_skipped_parity_failure" 0
  fi
done

"$PY" - "$OUT/integrated_runtime_status.json" "$FAMILY" <<'PY'
import json,pathlib,sys
family=sys.argv[2]
reason=("Ouro public HF gate is post-hoc; no physical survivor-skipping adapter is implemented."
        if family=="ouro" else
        "Integrated label requires survivor compaction plus passing batched-vs-B1 parity; native evidence is kept separate.")
pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"IMPLEMENTATION_OR_CORRECTNESS_STOP","reason":reason},indent=2)+"\n")
PY
record_step lane_complete 0
