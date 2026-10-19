# Phase 05 — No-refill depth-grouping regime and architecture study

Specification: [AGENT_TASK.md](AGENT_TASK.md). Executed 2026-10-19 on H100 GPUs 0–3 ([GPU_ASSIGNMENT_20261019.json](GPU_ASSIGNMENT_20261019.json)). The user confirmed ownership by stopping their own load job; GPUs 4–7 (other users) were never touched.

**Start here:** [results/20261019/RESULTS.md](results/20261019/RESULTS.md) (consolidated answers to RQ1–RQ3) and [results/20261019/REVIEW.md](results/20261019/REVIEW.md).

| Experiment | Doc | Results | Evidence classes |
| --- | --- | --- | --- |
| E1 retrospective regime reanalysis (CPU) | [E1_REGIME_REANALYSIS.md](E1_REGIME_REANALYSIS.md) | [results/20261019_e1](results/20261019_e1) | TRACE_ONLY, LATENCY_MODEL (phase01), PHYSICAL_FORCED_DEPTH (phase02/03), PHYSICAL_LIVE_EXIT (phase04 counts) |
| E2 H100 stage microbenchmark | [E2_STAGE_MICROBENCH.md](E2_STAGE_MICROBENCH.md) | [20261019_e2_d6b8674](results/20261019_e2_d6b8674) | PHYSICAL_FORCED_DEPTH (stage timing at fixed state) |
| E3 latency-aware Oracle | [E3_LATENCY_AWARE_ORACLE.md](E3_LATENCY_AWARE_ORACLE.md) | [20261019_e3_7f903d8](results/20261019_e3_7f903d8) | TRACE_ONLY, LATENCY_MODEL (replay), PHYSICAL_FORCED_DEPTH (physical) |

Nothing in this phase is live-exit inference performance. Every physical E3 number forces frozen B=1 trace depths.

## CPU checks (no GPU)

```
export CUDA_VISIBLE_DEVICES=""
/home/hwlee/batch-loop/.venv/bin/python -m unittest phase05/tests/test_e1.py phase05/tests/test_e3.py
/home/hwlee/batch-loop/.venv/bin/python phase05/scripts/e1_regime_reanalysis.py --smoke
```

## GPU commands (one GPU per benchmark; replace N with an authorized index)

```
# E2 (per model: ouro14 | ouro26 | huginn)
CUDA_VISIBLE_DEVICES=N HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python phase05/scripts/e2_stage_microbench.py --model ouro14 --output-dir <dir>/ouro14
# E2 Nsight Systems subset
CUDA_VISIBLE_DEVICES=N nsys profile -t cuda,nvtx -o <dir>/nsys/ouro14_B1_L512 python phase05/scripts/e2_nsys_cell.py --model ouro14 --B 1 --L 512
nsys stats --report nvtx_kern_sum,nvtx_sum --format csv --output <dir>/nsys/ouro14_B1_L512 <dir>/nsys/ouro14_B1_L512.nsys-rep
python phase05/scripts/e2_nsys_summarize.py <dir>/nsys && python phase05/scripts/e2_analyze.py <dir>
# E3
python phase05/scripts/e3_latency_oracle.py replay --e2-root <e2dir> --output-dir <e3dir>/replay           # CPU
CUDA_VISIBLE_DEVICES=N python phase05/scripts/e3_latency_oracle.py physical --trace ouro14_alpaca_q0p2.jsonl.gz --e2-root <e2dir> --output-dir <e3dir>/physical/ouro14_alpaca_q0p2
python phase05/scripts/e3_analyze.py <e3dir> --e2-root <e2dir>
python phase05/scripts/summary_table.py --e1 phase05/results/20261019_e1 --e2 <e2dir> --e3 <e3dir> --out phase05/results/20261019
```
Add `--smoke` to any script for a tiny run.

## Layout deviation

Per-experiment raw results live in dated subdirectories (`20261019_e1`, `20261019_e2_d6b8674`, `20261019_e3_7f903d8`). The consolidated `results/20261019/` holds RUN_MANIFEST.json, RESULTS.md, REVIEW.md, the three required CSVs (copied) and summary_table.csv.
