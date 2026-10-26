#!/usr/bin/env bash
# E0 (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md): Ouro-1.4B / Ouro-2.6B FULL vs LAST-CONVERT on the frozen
# GSM8K-200 / MATH500-200 items, caps 512/2048, gen + gold-NLL, 8 independent GPU workers; plus Stage-1
# parity receipts for both Ouro checkpoints. Huginn R32 FULL vs LAST-CONVERT is reused from
# results/20261023T060125Z_huginn_r32_h200x8 (same items, evaluator, caps and policy semantics).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/parity"
test -e "$RUN_DIR/jobs.json" || "$PYTHON" scripts/make_quality_jobs.py --models ouro-1.4b,ouro-2.6b \
  --policies full-decode,last-convert --run-dir "$RUN_DIR"
"$PYTHON" - "$RUN_DIR" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); j = json.loads((p / "jobs.json").read_text())
assert {(x["policy"], x["mode"]) for x in j} == {("full", "decode"), ("last", "convert")}
assert {x["model"] for x in j} == {"ouro-1.4b", "ouro-2.6b"}
(p / "job_check.json").write_text(json.dumps({"jobs": len(j), "policies": ["full-decode", "last-convert"],
                                              "gsm8k_cap": 512, "math500_cap": 2048}, indent=2))
PY
pids=()
for m in ouro-1.4b ouro-2.6b; do
  g=$([ "$m" = ouro-1.4b ] && echo 0 || echo 1)
  [ -e "$RUN_DIR/parity/parity_$m.json" ] || CUDA_VISIBLE_DEVICES=$g "$PYTHON" scripts/stage1_parity.py --model "$m" \
    --out "$RUN_DIR/parity/parity_$m.json" > "$RUN_DIR/logs/parity_$m.log" 2>&1 &
  pids+=("$!")
done
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/quality_worker.py --run-dir "$RUN_DIR" --worker "$gpu" \
    > "$RUN_DIR/logs/worker${gpu}_stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON" scripts/analyze_quality.py --run-dir "$RUN_DIR" > "$RUN_DIR/logs/analysis_stdout.log" 2>&1 || echo "analyze_quality failed"
ls "$RUN_DIR/errors" 2>/dev/null && echo "WARNING: failed jobs listed above" || true
echo "E0 finished: $RUN_DIR"
