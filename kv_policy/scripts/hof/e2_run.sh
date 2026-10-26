#!/usr/bin/env bash
# E2 localisation jobs on 8 GPU workers (MODELS_FILTER restricts the models a pass may claim).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR/e2/logs"
"$PYTHON" -c "import json,sys,subprocess; sys.path.insert(0,'.'); from kvp.manifest import environment; \
g=lambda *a: subprocess.run(['git',*a],capture_output=True,text=True).stdout.strip(); \
json.dump({'source_sha': g('rev-parse','HEAD'), 'dirty_files': g('status','--porcelain').splitlines(), \
'command': 'scripts/hof/e2_run.sh', 'models_filter': '${MODELS_FILTER:-}', 'environment': environment()}, \
open('$RUN_DIR/e2/manifest_${TAG:-main}.json','w'), indent=1)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/hof/e2_run.py work --run-dir "$RUN_DIR" --worker "$gpu" \
    --models-filter "${MODELS_FILTER:-}" > "$RUN_DIR/e2/logs/worker${gpu}_${TAG:-main}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
ls "$RUN_DIR/e2/errors" 2>/dev/null && echo "WARNING: failed E2 jobs listed above" || true
echo "E2 pass finished (${TAG:-main}): $RUN_DIR"
