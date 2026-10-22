#!/bin/bash
# usage: run_gpu.sh <gpu> <stage2_dir> <out_dir> <model:layout> ...
set -u
GPU=$1; S2=$2; OUT=$3; shift 3
export CC=$HOME/ddn/envs/tools/bin/gcc CUDA_VISIBLE_DEVICES=$GPU
cd $HOME/batch-loop/kv_policy
for spec in "$@"; do
  m=${spec%%:*}; l=${spec##*:}
  echo "[$(date -u +%FT%TZ)] start $m $l gpu$GPU"
  $HOME/ddn/envs/kvp/bin/python scripts/stage3_serving.py --model $m --layout $l --stage2-dir ../$S2 \
     --out ../$OUT/serving_${m}_${l}_gpu${GPU}.json > ../$OUT/logs/${m}_${l}_gpu${GPU}.log 2>&1
  echo "[$(date -u +%FT%TZ)] exit=$? $m $l gpu$GPU"
done
