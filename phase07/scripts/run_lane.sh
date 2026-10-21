#!/usr/bin/env bash
# Run collection jobs sequentially on one GPU. usage: run_lane.sh <gpu> <out_root> <model:threshold:dataset>...
set -u
GPU=$1 OUT=$2; shift 2
cd /home/hwlee/batch-loop-worktrees/phase07 || exit 2
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for job in "$@"; do
  IFS=: read -r model thr ds <<< "$job"
  name="${model}_t${thr}_${ds}"
  /home/hwlee/batch-loop/.venv/bin/python phase07/scripts/collect.py --model "$model" --threshold "$thr" --dataset "$ds" \
    --output-dir "$OUT/features/$name" > "$OUT/logs/collect_$name.log" 2>&1
  echo "job_done=$name rc=$?"
done
echo "lane_exit=$GPU"
