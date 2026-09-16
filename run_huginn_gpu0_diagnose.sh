#!/usr/bin/env bash
# Additive GPU0-only Huginn diagnosis. Does not modify/kill the in-flight GPU4-7 runs.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
PY="${PYTHON_BIN:-/home/hwlee/batch-loop/.venv/bin/python}"
REV="${HUGINN_REV:-bb6621b65e90b6a4b9b29ef88dc83866d450470c}"
MODEL="${HUGINN_MODEL:-/home/hwlee/.cache/huggingface/hub/models--tomg-group-umd--huginn-0125/snapshots/$REV}"
ROOT="${OUTPUT_ROOT:-results/huginn_gpu0_diag_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

# Fail closed if the caller accidentally exposes more than the intended GPU.
"$PY" - "$ROOT/run_manifest.json" "$REV" "$MODEL" <<'PY'
import datetime,json,os,pathlib,subprocess,sys

def get(*cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as e:
        return f"ERROR:{e}"

visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
if "," in visible:
    raise SystemExit(f"GPU0 diagnostic requires exactly one visible GPU, got CUDA_VISIBLE_DEVICES={visible}")
manifest={
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "lane": "huginn-gpu0-isolated-diagnosis",
    "cuda_visible_devices": visible,
    "model_revision": sys.argv[2],
    "model_path": sys.argv[3],
    "source_sha": get("git","rev-parse","HEAD"),
    "gpu": get("nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used","--format=csv,noheader"),
    "in_flight_gpu4_7_contract": "DO_NOT_TOUCH",
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
PY

"$PY" huginn_gpu0_diag.py \
  --model "$MODEL" \
  --revision "$REV" \
  --depth 16 \
  --threshold 0.26 \
  --output-dir "$ROOT"

echo "Huginn GPU0 diagnostic complete: $ROOT"
