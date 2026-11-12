#!/bin/bash
# usage: fu_worker.sh DSD_GPU=<g> <queue_file>
# Runs queue lines (full shell commands, cwd = dynamic_sd/) sequentially on GPU g.
#   "WAIT <path>"  blocks until <path> exists (cross-GPU dependency, max 6h)
#   "#END"         restores this GPU's burner (same command as lm_pretrain/run_all.sh) and exits
# Never runs on top of a burner. A failing line is logged and the queue continues.
G=${1#DSD_GPU=}; Q=$2; n=0
LOGD=/home1/irteam/dsd_work/logs_fu
mkdir -p $LOGD
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WF_ROOT=/home1/irteam/dsd_work/wavefront-decoding
export PY=/home/irteam/ddn/envs/wfd/bin/python
cd /home1/irteam/dsd_work/fu/dynamic_sd
start_burner() {
  pgrep -f "train_lm.py --gpu $G " >/dev/null && return
  cd /home1/irteam/lm_pretrain && (env -u CUDA_VISIBLE_DEVICES -u PYTORCH_CUDA_ALLOC_CONF CUDA_DEVICE_ORDER=PCI_BUS_ID \
    setsid nohup /home/irteam/ddn/envs/kvp/bin/python train_lm.py --gpu $G --hours 24 --util_min 0.80 --util_max 0.95 \
    > logs/gpu$G.log 2>&1 < /dev/null &)
  echo "$(date '+%F %T') burner restored gpu$G" >> $LOGD/worker$G.log
}
while true; do
  pgrep -f "train_lm.py --gpu $G " >/dev/null && { echo "$(date '+%T') burner on gpu$G, worker exit" >> $LOGD/worker$G.log; exit 0; }
  line=$(sed -n "$((n+1))p" $Q)
  if [ -z "$line" ]; then sleep 20; continue; fi
  n=$((n+1))
  if [ "$line" = "#END" ]; then echo "$(date '+%T') END" >> $LOGD/worker$G.log; start_burner; exit 0; fi
  [[ "$line" == \#* ]] && continue
  if [[ "$line" == WAIT\ * ]]; then
    f=${line#WAIT }; w=0
    echo "$(date '+%T') WAIT $f" >> $LOGD/worker$G.log
    while [ ! -f "$f" ] && [ $w -lt 21600 ]; do sleep 30; w=$((w+30)); done
    continue
  fi
  echo "$(date '+%T') START $n: $line" >> $LOGD/worker$G.log
  eval "$line" > $LOGD/job_g${G}_$n.log 2>&1
  echo "$(date '+%T') DONE $n rc=$?" >> $LOGD/worker$G.log
done
