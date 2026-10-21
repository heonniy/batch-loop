# Phase 07 — Exit-depth predictability and grouping utility (4×H100)

Task contract: [`AGENT_TASK.md`](AGENT_TASK.md). Run `20261021_4a3c11e` results: [`results/20261021_4a3c11e/RESULTS.md`](results/20261021_4a3c11e/RESULTS.md), review [`REVIEW.md`](results/20261021_4a3c11e/REVIEW.md).

## Environments

* Collection, probes, eager parity and CPU analysis use the repository venv at `~/batch-loop/.venv`. Collection goes through the Phase-02 eager adapters (`phase02_runtime`).
* Optimized CDB parity and E4 physical use the upstream engine, pinned and git-ignored:
  ```bash
  git clone https://github.com/kschwethelm/continuous-depth-batching phase07/third_party/continuous-depth-batching
  git -C phase07/third_party/continuous-depth-batching checkout abdfae0adbdd12be8c6123ccb34eee98a38afb39
  cd phase07/third_party/continuous-depth-batching && uv sync   # uv.lock sha256 c71a58bf… (see RUN_MANIFEST)
  ```
  `phase07/cdb_ext/` is the Phase-06 cohort-selection extension (copied from local commit 3e2779e, provenance headers in each file). It adds a `predicted` policy that sorts each wave by a frozen per-request/token score.
* Models are offline Hugging Face snapshots: Ouro-1.4B `37bc7e8f…`, Ouro-2.6B `f8e38372…`, Huginn-3.5B `bb6621b6…`.
* GPUs: [`GPU_ASSIGNMENT_20261021.json`](GPU_ASSIGNMENT_20261021.json) assigns physical 0–3, one lane each. GPUs 4–7 were not used.

## Commands (as run)

```bash
OUT=phase07/results/20261021_4a3c11e
# E0 collection, one lane per GPU (model:threshold:dataset jobs run sequentially)
phase07/scripts/run_lane.sh 0 $OUT ouro14:0.2:alpaca ouro14:0.2:sharegpt      > $OUT/logs/lane_A.log
phase07/scripts/run_lane.sh 1 $OUT ouro26:0.2:alpaca ouro26:0.2:sharegpt      > $OUT/logs/lane_B.log
phase07/scripts/run_lane.sh 2 $OUT huginn:0.12:alpaca huginn:0.12:sharegpt    > $OUT/logs/lane_C.log
phase07/scripts/run_lane.sh 3 $OUT ouro14:0.4:alpaca huginn:0.26:alpaca       > $OUT/logs/lane_D.log
# E2/E3 probes, eager parity and E4 physical, per GPU
phase07/scripts/run_post.sh <gpu> $OUT $OUT/logs/lane_<X>.log <model:lane|none> <lanes...>
# E0 optimized-CDB live parity (CDB venv)
phase07/third_party/continuous-depth-batching/.venv/bin/python phase07/scripts/e0_parity_cdb.py --model ouro14 --threshold 0.2 --out $OUT/e0
phase07/third_party/continuous-depth-batching/.venv/bin/python phase07/scripts/e0_parity_cdb.py --model huginn --threshold 0.12 --out $OUT/e0
# E4 physical, rerun after attempt 1 failed (see e4_physical_failed_attempt_1/README.txt)
phase07/third_party/continuous-depth-batching/.venv/bin/python phase07/scripts/e4_physical.py --model ouro14 --lane ouro14_t0.2_alpaca \
  --features-root $OUT/features --predictions $OUT/probes/ouro14_t0.2_alpaca/test_predictions.npz \
  --pred-key "all_precore|h64|w0" --feature-dim <dim from probe_results.csv> --hidden 64 --out $OUT/e4_physical/ouro14_t0.2_alpaca
# CPU analysis
~/batch-loop/.venv/bin/python phase07/scripts/e1_attribution.py $OUT/features --out $OUT/analysis/e1
~/batch-loop/.venv/bin/python phase07/scripts/e4_utility.py $OUT/probes --out $OUT/analysis/e4
~/batch-loop/.venv/bin/python phase07/scripts/summarize.py $OUT
# tests
~/batch-loop/.venv/bin/python phase07/tests/test_phase07.py   # unittest, 4 tests
```

## Data layout

* `features/<lane>/<split>/meta.npz` is committed. It holds depth, token, next_token, traj and prompt_len, together with `manifest.json`.
* The hidden-state memmaps (`emb`, `prevh`, `pre`, `loop1`, `loop2` `.npy`, 22 GB in total) stay local only. Their SHA-256 values are listed in `features/LOCAL_ONLY_ARRAYS.sha256`.
* `probes/<lane>/` holds per-seed metrics (`probe_results.csv`) and the seed-0 test PDFs (`test_predictions.npz`).
