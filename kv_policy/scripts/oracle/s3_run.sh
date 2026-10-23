#!/usr/bin/env bash
# Stage 3: restoration-oracle pilot on 8 independent single-GPU workers, then analysis.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR/stage3/logs"
test -e "$RUN_DIR/stage3/jobs.json" || "$PYTHON" scripts/oracle/s3_oracle.py make-jobs --run-dir "$RUN_DIR"
"$PYTHON" -c "import json,sys; sys.path.insert(0,'.'); from kvp.manifest import environment; import subprocess; \
json.dump({'source_sha': subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(), \
'command': 'scripts/oracle/s3_run.sh', 'environment': environment()}, open('$RUN_DIR/stage3/manifest.json','w'), indent=1)"
pids=()
for gpu in ${GPUS:-0 1 2 3 4 5 6 7}; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/oracle/s3_oracle.py work --run-dir "$RUN_DIR" --worker "$gpu" \
    > "$RUN_DIR/stage3/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
ls "$RUN_DIR/stage3/errors" 2>/dev/null && echo "WARNING: failed stage-3 jobs listed above"
"$PYTHON" scripts/oracle/s3_analyze.py --run-dir "$RUN_DIR" > "$RUN_DIR/stage3/logs/analyze_stdout.log" 2>&1 || echo "analysis failed"
echo "Stage 3 finished: $RUN_DIR"
