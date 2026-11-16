#!/bin/bash
# usage: pool_worker.sh <gpu> <jobs_file>   (cwd: repository root)
# Shared job pool: atomically claims the next line (flock <jobs>.lock, counter <jobs>.next) and runs it
# on this GPU. When the pool is empty (or <logdir>/STOP exists) it restores this GPU's burner (same
# command as lm_pretrain/run_all.sh) and exits. Never runs on top of a burner.
# The burner keep-alive must be paused (BURNER_PAUSE file) while workers run.
G=$1; J=$2
LOGD=${LOGD:-/home1/irteam/dsd_work/logs_pd}
mkdir -p $LOGD
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WF_ROOT=/home1/irteam/dsd_work/wavefront-decoding PY=/home/irteam/ddn/envs/wfd/bin/python
cd /home1/irteam/dsd_work/pd
restore() {
  pgrep -f "[t]rain_lm.py --gpu $G " >/dev/null && return
  cd /home1/irteam/lm_pretrain && (env -u CUDA_VISIBLE_DEVICES -u PYTORCH_CUDA_ALLOC_CONF CUDA_DEVICE_ORDER=PCI_BUS_ID \
    setsid nohup /home/irteam/ddn/envs/kvp/bin/python train_lm.py --gpu $G --hours 24 --util_min 0.80 --util_max 0.95 \
    > logs/gpu$G.log 2>&1 < /dev/null &)
  echo "$(date '+%F %T') burner restored gpu$G" >> $LOGD/worker$G.log
}
while true; do
  pgrep -f "[t]rain_lm.py --gpu $G " >/dev/null && { echo "$(date '+%T') burner present on gpu$G, exit" >> $LOGD/worker$G.log; exit 0; }
  [ -f $LOGD/STOP ] && { restore; exit 0; }
  n=$(flock $J.lock bash -c "n=\$(cat $J.next 2>/dev/null || echo 0); n=\$((n+1)); echo \$n > $J.next; echo \$n")
  line=$(sed -n "${n}p" $J)
  if [ -z "$line" ]; then echo "$(date '+%T') pool empty" >> $LOGD/worker$G.log; restore; exit 0; fi
  echo "$(date '+%T') START #$n: $line" >> $LOGD/worker$G.log
  eval "$line" > $LOGD/job_$n.log 2>&1
  echo "$(date '+%T') DONE #$n rc=$? gpu$G" >> $LOGD/worker$G.log
done
