#!/bin/bash
# Regenerates the markdown tables quoted in RESULT_REVIEW.md from the JSON receipts,
# and writes review_manifest.json (path, bytes, sha256 for every receipt).
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/irteam/ddn/envs/wfd/bin/python}
F=characterization_results_20261111/_fragments
mkdir -p $F
if [ "$1" != "manifest" ]; then
$PY dsd_analyze.py table characterization_results_20261111/smoke/*.json* > $F/table_smoke.md
$PY dsd_analyze.py table characterization_results_20261111/pilot/*.json* characterization_results_20261111/main/*.json* > $F/table_heldout.md
$PY dsd_analyze.py ci characterization_results_20261111/pilot/*.json* characterization_results_20261111/main/*.json* > $F/ci.md
$PY dsd_analyze.py straggler characterization_results_20261111/smoke/*_live_*.json* characterization_results_20261111/pilot/*_live_*.json* characterization_results_20261111/main/*_live_*.json* > $F/straggler.md
$PY dsd_analyze.py batch characterization_results_20261111/smoke/batch_*.json* > $F/batch.md
$PY dsd_analyze.py bench characterization_results_20261111/smoke/bench_r_*.json* > $F/bench.md
$PY dsd_analyze.py calib characterization_results_20261111/smoke/obs_*.json* > $F/calib.md
fi
$PY - <<'EOF'
import gzip, hashlib, json, os, glob
items = []
for p in sorted(glob.glob("characterization_results_20261111/**/*.json*", recursive=True)):
    b = open(p, "rb").read()
    meta = {}
    try:
        J = json.loads(gzip.decompress(b) if p.endswith(".gz") else b)
        if "meta" in J:
            meta = {k: J["meta"]["args"].get(k) for k in ("model", "dataset", "exit", "tau", "eps", "T_draft", "K",
                                                           "limit", "offset", "dtype", "sampling", "kv_s")}
            meta["wf_commit"] = J["meta"].get("wf_commit")
            meta["gpu"] = J["meta"].get("gpu")
    except Exception:
        pass
    items.append({"path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), **meta})
json.dump({"run": "dynamic-sd characterization 2026-11-11/12", "host": "henn-0 (8xH200)",
           "receipts": items}, open("review_manifest.json", "w"), indent=1)
print(len(items), "receipts")
EOF
