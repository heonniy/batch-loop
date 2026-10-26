# Huginn vs Ouro: FULL vs LAST-CONVERT — E0–E2 results

Plan: `kv_policy/HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md`.

## Reports

| Stage | Report | Notes |
|---|---|---|
| E0 | `analysis/e0_quality.md`, `analysis/e0_pairs.csv`, `analysis/e0_repeats.md` | Ouro runs are in `raw/`, parity in `parity/`. Huginn R32 is reused from `results/20261023T060125Z_huginn_r32_h200x8`. |
| E1 | `analysis/e1_training_signal_report.md`, `analysis/e1_*.csv` | Per-item records `e1/raw/*.json` (1200 files, 182 MB) are NOT committed. Their sha256 are in `e1/raw_sha256.txt`; the files stay on the H200 host. |
| E2 | `analysis/e2_localisation.md`, `analysis/e2_*.csv`, `e2/raw/` | The separately labelled Huginn R4/8/16 depth sweep is in `e2_huginn_depth/` (`analysis/summary.md`). |

## E3

E3 (selective protection) calibration was still running when these results were committed. Its code and results
are not part of this commit.

## Provenance

- Jobs ran from uncommitted working-tree code on top of `12b7ea5`, which is the `source_sha` in the receipts and
  manifests. The E0–E2 code is committed together with these results.
- Test log: `logs/pytest_cpu.log`.
- `kvp/stability.py` originates from the unpushed perturbation-stability work. It is included here because
  `kvp/depthprobe.py` and `kvp/xdepth.py` import it.
- Repository-root `AGENTS.md` blocks GPU use for the Phase-04R task only. This is a separate kv_policy task that
  the user requested.
