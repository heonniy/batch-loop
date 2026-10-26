#!/usr/bin/env bash
# E1: depth signal characterisation under FULL for Huginn R32, Ouro-1.4B, Ouro-2.6B (8 GPU workers).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR/e1/logs"
test -e "$RUN_DIR/e1/jobs.json" || "$PYTHON" scripts/hof/e1_run.py make-jobs --run-dir "$RUN_DIR"
"$PYTHON" -c "import json,sys,subprocess; sys.path.insert(0,'.'); from kvp.manifest import environment; \
g=lambda *a: subprocess.run(['git',*a],capture_output=True,text=True).stdout.strip(); \
json.dump({'source_sha': g('rev-parse','HEAD'), 'dirty_files': g('status','--porcelain').splitlines(), \
'command': 'scripts/hof/e1_run.sh', 'environment': environment()}, open('$RUN_DIR/e1/manifest.json','w'), indent=1)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/hof/e1_run.py work --run-dir "$RUN_DIR" --worker "$gpu" \
    > "$RUN_DIR/e1/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
ls "$RUN_DIR/e1/errors" 2>/dev/null && echo "WARNING: failed E1 jobs listed above" || true
echo "E1 finished: $RUN_DIR"
