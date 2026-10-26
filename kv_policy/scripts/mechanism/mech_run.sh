#!/usr/bin/env bash
# Attention-mechanism pilot: 64 prompts on 8 independent single-GPU workers, then analysis.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR/mechanism/logs"
test -e "$RUN_DIR/mechanism/jobs.json" || "$PYTHON" scripts/mechanism/mech_run.py make-jobs --run-dir "$RUN_DIR"
"$PYTHON" -c "import json,sys,subprocess; sys.path.insert(0,'.'); from kvp.manifest import environment; \
g=lambda *a: subprocess.run(['git',*a],capture_output=True,text=True).stdout.strip(); \
json.dump({'source_sha': g('rev-parse','HEAD'), 'dirty_files': g('status','--porcelain').splitlines(), \
'command': 'scripts/mechanism/mech_run.sh', 'environment': environment()}, open('$RUN_DIR/mechanism/manifest.json','w'), indent=1)"
pids=()
for gpu in ${GPUS:-0 1 2 3 4 5 6 7}; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/mechanism/mech_run.py work --run-dir "$RUN_DIR" --worker "$gpu" \
    > "$RUN_DIR/mechanism/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
ls "$RUN_DIR/mechanism/errors" 2>/dev/null && echo "WARNING: failed mechanism jobs listed above" || true
"$PYTHON" scripts/mechanism/mech_analyze.py --run-dir "$RUN_DIR" > "$RUN_DIR/mechanism/logs/analyze_stdout.log" 2>&1 || echo "analysis failed"
echo "Mechanism pilot finished: $RUN_DIR"
