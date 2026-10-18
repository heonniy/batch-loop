#!/usr/bin/env bash
# WP5 lane: Gate-0 ranker wiring audit (wp3_validation) -> scored physical on sealed wp5_test (only if PASS).
# usage: run_wp5_lane.sh <model> <role> <dataset> <lane_name> <wp5_root> <wp3_root>
set -u
MODEL=$1 ROLE=$2 DATASET=$3 LANE=$4 OUT=$5 W3=$6
PY=/home/hwlee/batch-loop/.venv/bin/python
cd /home/hwlee/batch-loop-worktrees/phase05_no_refill || exit 2
COMMON=(--model "$MODEL" --role "$ROLE" --dataset "$DATASET" --feature-root "$OUT/features/$LANE"
        --ranker-root "$OUT/ranker/$LANE" --wp3-feature-root "$W3/features/$LANE" --wp3-selection-root "$W3/selection/$LANE")
$PY -m phase05.wp5_ranker physical "${COMMON[@]}" --output-dir "$OUT/audit/$LANE" --audit-only \
  || { echo "pipeline_exit=gate0_failed"; exit 1; }
$PY -m phase05.wp5_ranker physical "${COMMON[@]}" --output-dir "$OUT/physical/$LANE"
echo "pipeline_exit=$?"
