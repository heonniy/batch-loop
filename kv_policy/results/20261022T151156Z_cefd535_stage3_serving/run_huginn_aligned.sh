#!/bin/bash
# Huginn re-run after depth alignment (source 707c311): config mean_recurrence 32 -> 16, KV pool 100 GiB
export CC=$HOME/ddn/envs/tools/bin/gcc CUDA_VISIBLE_DEVICES=$1
cd $HOME/batch-loop/kv_policy
$HOME/ddn/envs/kvp/bin/python scripts/stage3_serving.py --model huginn-0125 --layout last_exited --stage2-dir ../kv_policy/results/20261022T150648Z_13f1e87_stage2_quality    --kv-gib 100 --out ../kv_policy/results/20261022T151156Z_cefd535_stage3_serving/aligned_serving_huginn-0125_last_exited_gpu$1.json > ../kv_policy/results/20261022T151156Z_cefd535_stage3_serving/logs/aligned_huginn_gpu$1.log 2>&1
echo "exit=$?"
