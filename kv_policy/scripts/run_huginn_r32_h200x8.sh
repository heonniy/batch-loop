#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)_huginn_r32_h200x8}"
mkdir -p "$RUN_DIR/logs"
test ! -e "$RUN_DIR/jobs.json" || { echo "Already initialized: $RUN_DIR"; exit 2; }
"$PYTHON" scripts/make_quality_jobs.py --models huginn-0125 --run-dir "$RUN_DIR"
"$PYTHON" - "$RUN_DIR" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); j=json.loads((p/"jobs.json").read_text())
expected={("full","decode"),("last","convert"),("average","convert")}
assert {(x["policy"],x["mode"]) for x in j} == expected
assert {x["model"] for x in j}=={"huginn-0125"}
(p/"job_check.json").write_text(json.dumps({"jobs":len(j),"depth":32,"gsm8k_cap":512,"math500_cap":2048},indent=2))
PY
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/quality_worker.py --run-dir "$RUN_DIR" --worker "$gpu" > "$RUN_DIR/logs/worker${gpu}_stdout.log" 2>&1 &
 pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON" scripts/analyze_quality.py --run-dir "$RUN_DIR" > "$RUN_DIR/logs/analysis_stdout.log" 2>&1
"$PYTHON" scripts/audit_huginn_length.py --run-dir "$RUN_DIR" --output "$RUN_DIR/length_audit.md"
echo "Finished: $RUN_DIR"
