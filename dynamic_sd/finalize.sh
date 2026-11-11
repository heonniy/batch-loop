#!/bin/bash
# Run on the server after all queues finished: regenerate fragments from plain JSON,
# collect worker logs, gzip receipts, write manifest.
set -e
cd /home1/irteam/dsd_work/bl/dynamic_sd
find . -name '._*' -delete
mkdir -p characterization_results_20261111/logs
cp /home1/irteam/dsd_work/logs/worker*.log /home1/irteam/dsd_work/logs/watchdog.log characterization_results_20261111/logs/
# keep only the summary/progress lines of job logs (drop HF warnings)
for f in /home1/irteam/dsd_work/logs/job_g*.log; do
  grep -E '^\[dsd\]|^\s+\[|SUMMARY|^\{|Traceback|Error' "$f" > characterization_results_20261111/logs/$(basename "$f") || true
done
bash make_fragments.sh            # tables from plain .json
find characterization_results_20261111 -name '*.json' ! -path 'characterization_results_20261111/_fragments/*' -print0 | xargs -0 gzip -9 -f
bash make_fragments.sh manifest   # manifest now lists .json.gz (sha256 of committed files)
du -sh characterization_results_20261111
