#!/usr/bin/env bash
# Stage 1.2-1.4: parity jobs on 8 independent single-GPU workers, item audit, report.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
BASE_RUN="${BASE_RUN:-results/20261023T060125Z_huginn_r32_h200x8}"
mkdir -p "$RUN_DIR/parity/logs"
test -e "$RUN_DIR/parity/jobs.json" || "$PYTHON" scripts/oracle/s1_parity.py make-jobs --run-dir "$RUN_DIR" --base-run "$BASE_RUN"
pids=()
for gpu in ${GPUS:-0 1 2 3 4 5 6 7}; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/oracle/s1_parity.py work --run-dir "$RUN_DIR" --base-run "$BASE_RUN" \
    --worker "$gpu" > "$RUN_DIR/parity/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON" scripts/oracle/s1_item_audit.py --base-run "$BASE_RUN" --run-dir "$RUN_DIR"
"$PYTHON" scripts/oracle/s1_report.py --run-dir "$RUN_DIR" --base-run "$BASE_RUN" > "$RUN_DIR/logs_report.txt" 2>&1
ls "$RUN_DIR/parity/errors" 2>/dev/null && echo "WARNING: failed parity jobs listed above"
echo "Stage 1 parity finished: $RUN_DIR"
