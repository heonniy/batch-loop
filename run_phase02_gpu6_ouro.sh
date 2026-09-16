#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PHASE02_PYTHON:-/home/hwlee/batch-loop/.venv/bin/python}"
result_root="${PHASE02_RESULT_ROOT:-${script_dir}/results/phase02_physical_gpu6_ouro_$(date -u +%Y%m%dT%H%M%SZ)_$$}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not executable: $python_bin" >&2
  exit 1
fi
if [[ -e "$result_root" ]]; then
  echo "Result root already exists: $result_root" >&2
  exit 1
fi

echo "Source SHA: $(git -C "$script_dir" rev-parse HEAD)"
echo "Physical GPU 6: $(nvidia-smi -i 6 --query-gpu=index,uuid,name,memory.total --format=csv,noheader)"
echo "Model revision: 37bc7e8ff85a90d813dc096383405f521b6a67b0"
echo "Result root: $result_root"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$script_dir"
exec "$python_bin" -m phase02_runtime.benchmark --stage m0 "$@" --model ouro --output-dir "$result_root"
