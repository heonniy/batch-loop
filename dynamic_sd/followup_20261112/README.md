# Follow-up code — Dynamic Depth SD

This additive directory contains **no measured results**. It does not modify the upstream WaveFront code or previous `dynamic_sd` characterization scripts.

- `gate_shadow.py`: use the existing `dynamic_sd/dsd_run.py` replay runner and add a non-intervening shadow-gate evaluation to every recurrent step. Modes: `off`, `gpu`, `cpu`. It **does not implement CUDA Graph, CPU/GPU event attribution or true native exit replay parity**.
- `compare_shadow.py`: validate that different shadow modes executed the same replay geometry and outputs, then describe paired prefill-inclusive latency deltas; reports limitations.
- `oracle_frontier.py`: CPU-only one-decision counterfactual analysis of **measured** target-equivalent action choices (not a multi-step policy rollout).
- `tests/test_cpu.py`: CPU-only invariant tests using Python stdlib. Execute `python -m unittest discover -s dynamic_sd/followup_20261112/tests -v`.
- `PLAN.md` and `RUNBOOK.md`: Stage 0–3 controls, falsifiers and GPU-host run commands.

This study builds on `dynamic_sd/RESULT_REVIEW.md`. The original source branch and raw receipts are preserved. **No automatic jobs, model downloads or GPU runs on import.**
