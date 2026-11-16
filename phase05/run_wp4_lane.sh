#!/usr/bin/env bash
# WP4 lane: collect wp4_confirm -> Gate-0 equivalence audit -> scored physical (only if audit PASS).
# usage: wp4_lane.sh <model> <role> <dataset> <lane_name> <out_root> <wp3_root>
set -u
MODEL=$1 ROLE=$2 DATASET=$3 LANE=$4 OUT=$5 W3=$6
PY=/home/hwlee/batch-loop/.venv/bin/python
cd /home/hwlee/batch-loop-worktrees/phase05_no_refill || exit 2
$PY -m phase05.wp3_predictor collect --model "$MODEL" --role "$ROLE" --dataset "$DATASET" \
  --splits wp4_confirm --output-dir "$OUT/confirm/$LANE" || { echo "pipeline_exit=collect_failed"; exit 1; }
$PY -m phase05.wp4_gpu_serving --model "$MODEL" --role "$ROLE" --dataset "$DATASET" \
  --feature-root "$W3/features/$LANE" --confirm-root "$OUT/confirm/$LANE" --selection-root "$W3/selection/$LANE" \
  --output-dir "$OUT/audit/$LANE" --audit-only || { echo "pipeline_exit=gate0_failed"; exit 1; }
$PY -m phase05.wp4_gpu_serving --model "$MODEL" --role "$ROLE" --dataset "$DATASET" \
  --feature-root "$W3/features/$LANE" --confirm-root "$OUT/confirm/$LANE" --selection-root "$W3/selection/$LANE" \
  --output-dir "$OUT/physical/$LANE"
echo "pipeline_exit=$?"
