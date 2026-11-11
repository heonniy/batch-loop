#!/bin/bash
# usage: gpu_worker.sh DSD_GPU=<g> <queue_file>
# Runs queue lines sequentially on GPU g; waits for appended lines; stops at "#END"
# or when a burner is running on this GPU (never runs on top of the burner).
G=${1#DSD_GPU=}; Q=$2; n=0
LOGD=/home1/irteam/dsd_work/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=/home/irteam/ddn/envs/wfd/bin/python
cd /home1/irteam/dsd_work/bl/dynamic_sd
while true; do
  pgrep -f "train_lm.py --gpu $G " >/dev/null && { echo "$(date '+%T') burner on gpu$G, worker exit" >> $LOGD/worker$G.log; exit 0; }
  line=$(sed -n "$((n+1))p" $Q)
  if [ -z "$line" ]; then sleep 20; continue; fi
  n=$((n+1))
  [ "$line" = "#END" ] && { echo "$(date '+%T') END" >> $LOGD/worker$G.log; exit 0; }
  [[ "$line" == \#* ]] && continue
  echo "$(date '+%T') START $n: $line" >> $LOGD/worker$G.log
  eval "$PY dsd_run.py $line" > $LOGD/job_g${G}_$n.log 2>&1
  echo "$(date '+%T') DONE $n rc=$?" >> $LOGD/worker$G.log
done
