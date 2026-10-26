#!/usr/bin/env bash
# E2 (separately labelled): Huginn at fixed num_steps R in {4, 8, 16}, FULL vs LAST-CONVERT on the frozen
# GSM8K-200 / MATH500-200 items, caps 512/2048, gen + gold NLL; matches Ouro's R=4 for a depth-matched contrast.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR/logs"
test -e "$RUN_DIR/jobs.json" || "$PYTHON" scripts/make_quality_jobs.py --models huginn-0125-r4,huginn-0125-r8,huginn-0125-r16 \
  --policies full-decode,last-convert --run-dir "$RUN_DIR"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/quality_worker.py --run-dir "$RUN_DIR" --worker "$gpu" \
    > "$RUN_DIR/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON" scripts/analyze_quality.py --run-dir "$RUN_DIR" > "$RUN_DIR/logs/analysis_stdout.log" 2>&1 || echo "analyze_quality failed"
ls "$RUN_DIR/errors" 2>/dev/null && echo "WARNING: failed jobs listed above" || true
echo "E2 Huginn depth quality finished: $RUN_DIR"
