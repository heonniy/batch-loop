#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PHASE02_PYTHON:-/home/hwlee/batch-loop/.venv/bin/python}"
result_root="${PHASE02_RESULT_ROOT:-${script_dir}/results/phase02_physical_gpu7_huginn_$(date -u +%Y%m%dT%H%M%SZ)_$$}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not executable: $python_bin" >&2
  exit 1
fi
if [[ -e "$result_root" ]]; then
  echo "Result root already exists: $result_root" >&2
  exit 1
fi

echo "Source SHA: $(git -C "$script_dir" rev-parse HEAD)"
echo "Physical GPU 7: $(nvidia-smi -i 7 --query-gpu=index,uuid,name,memory.total --format=csv,noheader)"
echo "Model revision: bb6621b65e90b6a4b9b29ef88dc83866d450470c"
echo "Result root: $result_root"
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$script_dir"
exec "$python_bin" -m phase02_runtime.benchmark --stage m0 "$@" --model huginn --output-dir "$result_root"
