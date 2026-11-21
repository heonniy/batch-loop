# Experimental execution status — 2026-11-21

**EXECUTED on 8× H200 (run `runs/20261121_h200x8`).** All 23 DAG nodes have ledger receipts: 22 executed and C4 `not_run` with a reason. Total 50.9 / 384 GPU-hours. Results and limitations: [RESULTS.md](RESULTS.md).

- **Final verdicts:**
  - F1_FULL_ROLLOUT = `supported_heldout_quality_matched`. This is one cell: Ouro-2.6B B=16 `jac_m1`, 1.29× vs AR. It **ties** depth truncation.
  - F2_STRESS = `contradicted_quality` by controller rules (inconclusive by FREEZE wording). The gain does not transfer to MATH-500 or to FP32.
- **Exact depth-parallel decoding:** no speedup in any cell.
- **Ledger:** `runs/20261121_h200x8/evidence.jsonl`, with per-node `receipts/` and `analysis/`.
- **Raw per-prompt outputs:** kept on the H200 host and gitignored; sha256 values are in the receipts.
- **Test freeze:** [FREEZE_20261121.md](FREEZE_20261121.md), committed in `a6577118` before any test-split decode.

**Never replace `NOT_RUN` with `completed` based solely on successful CPU CI or generated job manifests.**
