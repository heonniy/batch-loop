#!/usr/bin/env bash
# Wait for this GPU's stage-1 to finish, then run WP6 jobs sequentially on the same GPU.
# usage: run_wp6_chain.sh <gpu> <wp6_root> <wp5_root> <job>...
set -u
GPU=$1 OUT=$2 W5=$3; shift 3
cd /home/hwlee/batch-loop-worktrees/phase05_no_refill || exit 2
until grep -q "stage_exit=stage1:$GPU" "$OUT/logs/stage1_gpu$GPU.log" 2>/dev/null; do sleep 60; done
for job in "$@"; do
  phase05/run_wp6_scored.sh "$job" "$GPU" "$OUT" "$W5" > "$OUT/logs/job_${job}_gpu$GPU.log" 2>&1
  echo "chain_done=$job"
done
echo "chain_exit=$GPU"
