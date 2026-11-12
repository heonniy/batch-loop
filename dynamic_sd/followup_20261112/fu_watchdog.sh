#!/bin/bash
# Safety net: if a GPU has neither its fu_worker nor a burner for IDLE_MIN consecutive
# minutes (worker died), restore that GPU's burner (same command as lm_pretrain/run_all.sh).
# Exits once all 8 burners are running.
IDLE_MIN=${IDLE_MIN:-10}
LOG=/home1/irteam/dsd_work/logs_fu/watchdog.log
declare -A idle
for g in 0 1 2 3 4 5 6 7; do idle[$g]=0; done
echo "$(date '+%F %T') fu watchdog up (idle=${IDLE_MIN}m)" >> $LOG
while true; do
  all=1
  for g in 0 1 2 3 4 5 6 7; do
    pgrep -f "train_lm.py --gpu $g " >/dev/null && continue
    all=0
    if pgrep -f "fu_worker.sh DSD_GPU=$g " >/dev/null; then idle[$g]=0; else idle[$g]=$((idle[$g]+1)); fi
    if [ ${idle[$g]} -ge $IDLE_MIN ]; then
      cd /home1/irteam/lm_pretrain && (env -u CUDA_VISIBLE_DEVICES -u PYTORCH_CUDA_ALLOC_CONF CUDA_DEVICE_ORDER=PCI_BUS_ID \
        setsid nohup /home/irteam/ddn/envs/kvp/bin/python train_lm.py --gpu $g --hours 24 --util_min 0.80 --util_max 0.95 \
        > logs/gpu$g.log 2>&1 < /dev/null &)
      echo "$(date '+%F %T') watchdog restored burner gpu$g" >> $LOG
    fi
  done
  [ $all -eq 1 ] && { echo "$(date '+%F %T') all burners running; watchdog exit" >> $LOG; exit 0; }
  sleep 60
done
