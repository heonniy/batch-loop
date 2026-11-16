#!/usr/bin/env bash
# WP6 stage 2/3 per GPU: Gate-0 audit(s) for the lane, then the scored physical run only if every audit passes.
# usage: run_wp6_scored.sh <job> <gpu> <wp6_root> <wp5_root>
#   jobs: q04train_A, q04train_C (stage 2); transfer_A|B|C|D (stage 3a); q04_A|C, eos_A|C (stage 3b)
set -u
JOB=$1 GPU=$2 OUT=$3 W5=$4
PY=/home/hwlee/batch-loop/.venv/bin/python
cd /home/hwlee/batch-loop-worktrees/phase05_no_refill || exit 2
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
declare -A NAME=([A]=lane_A_ouro_alpaca [B]=lane_B_huginn_alpaca [C]=lane_C_ouro_sharegpt [D]=lane_D_huginn_sharegpt)
declare -A OTHER=([A]=C [B]=D [C]=A [D]=B)
declare -A MODEL=([A]=ouro [B]=huginn [C]=ouro [D]=huginn)
declare -A DATA=([A]=alpaca [B]=alpaca [C]=sharegpt [D]=sharegpt)
L=${JOB#*_}; LN=${NAME[$L]}; M=${MODEL[$L]}; DS=${DATA[$L]}
R=(--model "$M" --role "gpu$GPU" --dataset "$DS")
audit() { $PY -m phase05.wp6_runner "${R[@]}" --split wp3_validation "$@" || { echo "GATE0_FAILED $JOB"; exit 1; }; }
case "$JOB" in
  q04train_*)
    $PY -m phase05.wp5_ranker train --feature-root "$OUT/features/q04/$LN" --output-dir "$OUT/ranker/q04/$LN" --vocab 49152 ;;
  transfer_*)
    audit --study transfer_notraj --feature-root "$W5/features/$LN" --ranker "notraj=$OUT/ranker/notraj/$LN" \
          --audit-ranker notraj --output-dir "$OUT/audit/notraj/$LN"
    $PY -m phase05.wp6_runner "${R[@]}" --study transfer_notraj --split wp6_test --feature-root "$OUT/features/wp6/$LN" \
        --ranker "same=$W5/ranker/$LN" --ranker "cross=$W5/ranker/${NAME[${OTHER[$L]}]}" \
        --ranker "notraj=$OUT/ranker/notraj/$LN" --output-dir "$OUT/physical/transfer_notraj/$LN" ;;
  q04_*)
    audit --study threshold --threshold 0.4 --feature-root "$OUT/features/q04/$LN" --ranker "q04=$OUT/ranker/q04/$LN" \
          --audit-ranker q04 --output-dir "$OUT/audit/q04/$LN"
    $PY -m phase05.wp6_runner "${R[@]}" --study threshold --threshold 0.4 --split wp6_q_test \
        --feature-root "$OUT/features/q04/$LN" --ranker "q04=$OUT/ranker/q04/$LN" --ranker "q02=$W5/ranker/$LN" \
        --output-dir "$OUT/physical/threshold/$LN" ;;
  eos_*)
    audit --study eos --feature-root "$W5/features/$LN" --ranker "same=$W5/ranker/$LN" --audit-ranker same \
          --audit-random-lengths --output-dir "$OUT/audit/eos/$LN"
    $PY -m phase05.wp6_runner "${R[@]}" --study eos --split wp6_eos_test --use-eos-lengths \
        --feature-root "$OUT/features/eos/$LN" --ranker "same=$W5/ranker/$LN" --output-dir "$OUT/physical/eos/$LN" ;;
  *) echo "unknown job $JOB"; exit 2 ;;
esac
echo "job_exit=$JOB:$?"
