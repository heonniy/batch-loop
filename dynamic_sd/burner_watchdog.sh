#!/bin/bash
# Restores the GPU burner (same command as /home1/irteam/lm_pretrain/run_all.sh, per GPU)
# when a GPU has had no dsd experiment process for IDLE_MIN minutes, or at DEADLINE
# (then dsd jobs are stopped first). Never starts a second burner on a GPU.
IDLE_MIN=${IDLE_MIN:-30}
DEADLINE=${DEADLINE:-"09:50"}
BURN_DIR=/home1/irteam/lm_pretrain
PY=/home/irteam/ddn/envs/kvp/bin/python
LOG=/home1/irteam/dsd_work/logs/watchdog.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID
declare -A idle
mkdir -p $BURN_DIR/logs/prev_run3
for g in 0 1 2 3 4 5 6 7; do idle[$g]=0; done

burner_on() { pgrep -f "train_lm.py --gpu $1 " >/dev/null; }
dsd_on() { [ -n "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
start_burner() {
  local g=$1
  burner_on $g && return
  [ -f $BURN_DIR/logs/gpu$g.log ] && [ ! -f $BURN_DIR/logs/prev_run3/gpu$g.log ] && cp $BURN_DIR/logs/gpu$g.log $BURN_DIR/logs/prev_run3/
  cd $BURN_DIR && setsid nohup $PY train_lm.py --gpu $g --hours 24 --util_min 0.80 --util_max 0.95 > logs/gpu$g.log 2>&1 < /dev/null &
  echo "$(date '+%F %T') started burner gpu$g" >> $LOG
}

echo "$(date '+%F %T') watchdog up (idle=${IDLE_MIN}m deadline=$DEADLINE)" >> $LOG
while true; do
  now=$(date +%H:%M)
  all=1
  for g in 0 1 2 3 4 5 6 7; do
    if burner_on $g; then continue; fi
    all=0
    if [[ "$now" > "$DEADLINE" && "$now" < "12:00" ]] || [ -f /home1/irteam/dsd_work/RESTORE_NOW ]; then
      pkill -f "DSD_GPU=$g " ; sleep 5
      pids=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)
      for p in $pids; do grep -q dsd_run /proc/$p/cmdline 2>/dev/null && kill $p; done
      sleep 5; start_burner $g; continue
    fi
    if dsd_on $g; then idle[$g]=0; else idle[$g]=$((idle[$g]+1)); fi
    if [ ${idle[$g]} -ge $IDLE_MIN ]; then start_burner $g; fi
  done
  [ $all -eq 1 ] && { echo "$(date '+%F %T') all burners running; watchdog exit" >> $LOG; exit 0; }
  sleep 60
done
