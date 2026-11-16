#!/usr/bin/env bash
# WP6 orchestration per GPU and stage. usage: run_wp6.sh <stage> <gpu> <wp6_root> <wp5_root>
#   stage1: data collection (+ no-trajectory ranker training)   stage2: q=0.4 rankers + Gate-0 audits
#   stage3a: transfer/no-traj physical                          stage3b: q=0.4 / EOS physical
set -u
STAGE=$1 GPU=$2 OUT=$3 W5=$4
PY=/home/hwlee/batch-loop/.venv/bin/python
cd /home/hwlee/batch-loop-worktrees/phase05_no_refill || exit 2
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
col() { $PY -m phase05.wp3_predictor collect --role gpu$GPU --full-dim "$@" || echo "STEP_FAILED collect $*"; }
notraj() { $PY -m phase05.wp5_ranker train --feature-root "$W5/features/$1" --output-dir "$OUT/ranker/notraj/$1" --vocab "$2" --drop-trajectory || echo "STEP_FAILED notraj $1"; }
q04() { $PY -m phase05.wp5_ranker train --feature-root "$OUT/features/q04/$1" --output-dir "$OUT/ranker/q04/$1" --vocab 49152 || echo "STEP_FAILED q04 $1"; }
run() { $PY -m phase05.wp6_runner --role gpu$GPU "$@" || echo "STEP_FAILED run $*"; }
case "$STAGE:$GPU" in
  stage1:0) col --model ouro --dataset alpaca --threshold 0.4 --splits wp3_train wp3_validation wp6_q_test --output-dir "$OUT/features/q04/lane_A_ouro_alpaca" ;;
  stage1:2) col --model ouro --dataset sharegpt --threshold 0.4 --splits wp3_train wp3_validation wp6_q_test --output-dir "$OUT/features/q04/lane_C_ouro_sharegpt" ;;
  stage1:1) notraj lane_B_huginn_alpaca 65536 & notraj lane_D_huginn_sharegpt 65536 &
            col --model huginn --dataset alpaca --splits wp6_test --output-dir "$OUT/features/wp6/lane_B_huginn_alpaca"
            col --model huginn --dataset sharegpt --splits wp6_test --output-dir "$OUT/features/wp6/lane_D_huginn_sharegpt"; wait ;;
  stage1:3) notraj lane_A_ouro_alpaca 49152 & notraj lane_C_ouro_sharegpt 49152 &
            col --model ouro --dataset alpaca --splits wp6_test --output-dir "$OUT/features/wp6/lane_A_ouro_alpaca"
            col --model ouro --dataset sharegpt --splits wp6_test --output-dir "$OUT/features/wp6/lane_C_ouro_sharegpt"
            col --model ouro --dataset alpaca --stop-at-eos --splits wp6_eos_test --output-dir "$OUT/features/eos/lane_A_ouro_alpaca"
            col --model ouro --dataset sharegpt --stop-at-eos --splits wp6_eos_test --output-dir "$OUT/features/eos/lane_C_ouro_sharegpt"; wait ;;
  *) echo "unknown $STAGE:$GPU"; exit 2 ;;
esac
echo "stage_exit=$STAGE:$GPU"
