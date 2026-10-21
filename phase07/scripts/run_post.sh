#!/usr/bin/env bash
# Per-GPU post-processing: for each lane, wait for its collection, train probes (E2/E3) and, for Alpaca lanes,
# run the eager BF16 parity check (E0). Optionally, after this GPU's collection lane exits, run E4 physical.
# usage: run_post.sh <gpu> <out_root> <lane_log> <physical_model|none> <lane>...
set -u
GPU=$1 OUT=$2 LANELOG=$3 PHYS=$4; shift 4
cd /home/hwlee/batch-loop-worktrees/phase07 || exit 2
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/hwlee/batch-loop/.venv/bin/python
for lane in "$@"; do
  until [ -f "$OUT/features/$lane/test/manifest.json" ]; do sleep 60; done
  [ -f "$OUT/probes/$lane/probe_results.csv" ] || \
    $PY phase07/scripts/e2_probes.py "$OUT/features" --lane "$lane" --out "$OUT/probes" --device cuda > "$OUT/logs/probes_$lane.log" 2>&1
  echo "probes_done=$lane rc=$?"
  case "$lane" in *_alpaca)
    $PY phase07/scripts/e0_parity_eager.py --features-root "$OUT/features" --lane "$lane" --out "$OUT/e0" > "$OUT/logs/parity_eager_$lane.log" 2>&1
    echo "parity_done=$lane rc=$?";; esac
done
if [ "$PHYS" != none ]; then
  until grep -q lane_exit "$LANELOG"; do sleep 60; done
  IFS=: read -r model lane <<< "$PHYS"
  DIM=$($PY -c "import csv;print(next(r['dim'] for r in csv.DictReader(open('$OUT/probes/$lane/probe_results.csv')) if r['features']=='all_precore' and r['hidden']=='64'))")
  phase07/third_party/continuous-depth-batching/.venv/bin/python phase07/scripts/e4_physical.py --model "$model" --lane "$lane" \
    --features-root "$OUT/features" --predictions "$OUT/probes/$lane/test_predictions.npz" --pred-key "all_precore|h64|w0" \
    --feature-dim "$DIM" --hidden 64 --out "$OUT/e4_physical/$lane" > "$OUT/logs/e4_physical_$lane.log" 2>&1
  echo "physical_done=$lane rc=$?"
fi
echo "post_exit=$GPU"
