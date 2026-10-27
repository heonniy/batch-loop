#!/usr/bin/env bash
# Step 1 pilot (OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md): Stage-1 32+32 items, gen + gold NLL, BF16, caps 512/2048.
# Ouro-1.4B / 2.6B: FULL, LAST-CONVERT, LAST-DECODE (diagnostic alternate semantics only), protected banks
# (kvp.banks, <run>/banks.json); Huginn R32 negative control: FULL, LAST-CONVERT, bankP0.
# Workers re-read jobs.json, so calibration-selected (bankcal) jobs can be appended later (add_jobs.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p "$RUN_DIR/logs"
if [ ! -e "$RUN_DIR/jobs.json" ]; then
  OURO="full-decode,last-convert,last-decode,bankP0-convert,bankNeg1-convert,bankR1all-convert,bankL0R12-convert"
  OURO="$OURO,bankrand0-convert,bankrand1-convert,bankrand2-convert,bankrand3-convert,bankrand4-convert"
  "$PYTHON" scripts/diag/add_jobs.py --run-dir "$RUN_DIR" --models ouro-1.4b,ouro-2.6b --policies "$OURO"
  "$PYTHON" scripts/diag/add_jobs.py --run-dir "$RUN_DIR" --models huginn-0125 --policies full-decode,last-convert,bankP0-convert
fi
"$PYTHON" -c "import json,sys,subprocess; sys.path.insert(0,'.'); from kvp.manifest import environment; \
g=lambda *a: subprocess.run(['git',*a],capture_output=True,text=True).stdout.strip(); \
json.dump({'source_sha': g('rev-parse','HEAD'), 'dirty_files': g('status','--porcelain').splitlines(), \
'command': 'scripts/diag/pilot_run.sh', 'environment': environment()}, open('$RUN_DIR/manifest.json','w'), indent=1)"
pids=()
for gpu in $GPUS; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/quality_worker.py --run-dir "$RUN_DIR" --worker "$gpu" \
    > "$RUN_DIR/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
ls "$RUN_DIR/errors" 2>/dev/null && echo "WARNING: failed jobs listed above" || true
echo "Pilot finished: $RUN_DIR"
