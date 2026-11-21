# Coding-agent execution contract — adaptive H200×8 research

This is an **instruction to an agent that is actually running on the assigned research server**, not evidence that any live H200 job has started. User authorized a comprehensive 1–2 day research campaign; all launch and safety requirements still apply. If GPU shell access is unavailable, mark `NOT_RUN` and retain the code/plan only.

## Step 0. Read and reconcile evidence BEFORE starting any experiment
1. Open `../AGENTS.md`: old Phase-04/04R locks remain. New additive study cannot change Phase-04, FERRY, dynamic_sd, or unrelated processes.
2. Read `heonniy/research-wiki/UPDATE_POLICY.md`, `AGENTS.md`, `concepts/C-PAR-001.md`, `questions/RQ-PAR-001.md`, `hypotheses/H-PAR-001.md`, `experiments/EXP-20261116-DEPTH-POC.md`, findings `F-091`–`F-095`.
3. Read the **exact experiment source** `../parallel_depth_poc/RESULTS.md`, `e5_decoder.py`, `wf_engine.py`, and `results/20261116_gpu/review_manifest.json`. Explain candidate-KV carry in E5 and BF16 row-shape mismatch before any new design.
4. Open `LITERATURE.md` and `tree.json`. For the candidate about to be implemented, inspect its wiki paper card AND its primary paper at the cited equation/section. Record (a) authors' verified claim, (b) its assumptions, (c) mapping to depth-indexed KV, (d) alternative baseline and failure reason, (e) paper version and URL, (f) verification depth `abstract/full_text`.
5. Resolve model revisions from author code, prompt IDs and gate checks, baseline throughput/HBM. Never assume P-**** literature abstracts imply verified implementation details.
6. Pin git SHAs and computational environment to a run folder. **Do not write the research wiki** absent a separate checkpoint authorization.

## Step 1. Fail-closed safety
- Check `nvidia-smi -L`, per GPU active processes, CUDA/PyTorch, device IDs, occupied memory. Proceed **only** with idle H200 cards explicitly allocated to this experiment. Never stop/restart GPU burners or terminate someone else's jobs. Existing `parallel_depth_poc/pool_worker.sh` includes burner handling — **do not use it**.
- 8 H200 may live at physical indices other than 0–7; supply observed indices, not assumptions.
- Confirm WaveFront installation (`WF_ROOT`) and cached model revisions; assign one independent job per GPU, **not eight simultaneous copies of a model on one GPU**.
- Verify tensor memory feasibility before B128/L1024; OOM -> shrink or record unsupported, not numeric zero.
- Require explicit command flags `--allow-gpu --authorized` when using `run_jobs`. Simulator outputs and E1 oracle are always labelled `oracle/diagnostic`.
- Stop scheduling if GPU ownership changes, reference parity fails, a process exceeds its memory/time budget, or repeated device fault occurs. Preserve logs and leave unrelated processes alone.

## Step 2. Initial commands (from repository root on the GPU server)
```bash
git checkout research/parallel-depth-adaptive-h200x8-20261121
python -m unittest discover -s parallel_depth_search/tests -v
python -m parallel_depth_search.control validate
RUN_DIR=./parallel_depth_search/runs/20261121_h200x8
python -m parallel_depth_search.control init --run-dir "$RUN_DIR"
python -m parallel_depth_search.control propose --run-dir "$RUN_DIR"
python -m parallel_depth_search.make_jobs --run-dir "$RUN_DIR" --out "$RUN_DIR/stage0.json"
# Read stage0.json, inspect evidence and verify H200 assignments before launch.
python -m parallel_depth_search.run_jobs --plan "$RUN_DIR/stage0.json" \
  --run-dir "$RUN_DIR" --gpu-ids 0,1,2,3,4,5,6,7 --wall-hours 12 \
  --allow-gpu --authorized
```
If idle assigned indices are not 0–7, replace the flag accordingly. The above command is **an example for the on-host agent only**; nothing is launched by committing this file.

## Step 3. Actual scientific adaptive loop (repeat; human/LLM coding agent supervises)
After each batch of *real* jobs:
1. Inspect exit codes, logs, per-run configuration, raw checksums, exact ref vs previous trace, numerical anomalies and oom/unsupported.
2. Aggregate paired comparisons, check quality/CI, check all hidden/KV/rollout metrics, and write a source-backed node-specific `EVIDENCE_NOTE.md` citing primary papers (claim IDs), pinned code revisions, sample IDs, baseline times, quantiles, status of uncertainties, negative controls.
3. Prepare a **real** `receipt.json` with required `node_id, trial_id, state, source_git_sha, command, data_split, model, seed, method, literature_checks[], metrics{}, hardware{}, artifacts{sha256}, gpu_hours`. Use `state=not_run` with a reason if not executed. Attach logs and repeat runs; one node may have many receipts.
4. For *genuine paired free-running decoder* results, compute paired confidence intervals using `python -m parallel_depth_search.paired_metrics --input real_decoder.jsonl --candidate METHOD --out paired_ci.json`. Check non-matching output lengths and cache-parity traces. Report `fidelity_tier=exact` ONLY if per-depth bank and sequence are verified; a quality-noninferior approximate generator can be evaluated without calling it lossless.
5. `python -m parallel_depth_search.control ingest --run-dir "$RUN_DIR" --receipt receipt.json`. This refuses duplicate IDs/ambiguous provenance/final-test leakage and preserves append-only evidence.
6. `python -m parallel_depth_search.rank_candidates --run-dir "$RUN_DIR"` produces split- and hardware-matched quality-aware recommendations with method diversity. It never turns diagnostic-only runs into serving claims.
7. `python -m parallel_depth_search.control propose --run-dir "$RUN_DIR"` to identify eligible or blocked hypotheses. The coding agent **must interpret** each proposed family, related paper and credible confidence interval before selecting the next job list. Do not blindly mark a positive result because a heuristic gate returned low residual.
8. For `needs_investigation`: allocate follow-up with higher n, alternate seed/dtype/row order, validate oracle-vs-online leakage, re-examine paper assumptions and inspect profiler. A small negative pilot is *not* NO-GO.
9. For `manual_nodes`: write the missing actual solver or real B>1 E2E adapter (tests and provenance) on this branch, then register it in `make_jobs.jobs_for`; keep experimental API stable. **Do not fake implementation with a placeholder or synthetic performance.**
10. Keep at least two mechanism-diverse arms alive through mid-stage and reserve >=20% GPU budget for cross-family exploration. Champion selection happens on dev only; final test after freeze.
11. Commit **source before the data run** and then a results commit (manifest, checksums, tables, confidence intervals, failed runs, OOMs and anomalies). Do not commit raw sensitive prompts or enormous GPU logs.

### Critical mandatory milestones
- Before `C3_NEWTON`: verify directional Jacobian-vector accuracy versus exact finite perturbation, cost of surrogate and BF16 stability. Newton identity is a comparator, not a theorem about Huginn.
- Before `C2_ANDERSON`: distinguish triangular Anderson in ParaTAA from ordinary global Anderson; benchmark both, include Gram solve/damping overhead.
- Before selective allocation: compare equal active rows to uniform and random controls, include KV slots for skipped depths and exact fallback.
- Before `V2_KV`: separate `exact serial commit`, `candidate KV`, `selective repair`; compare **newly generated next-token KV** per depth, not only logits.
- Before `F1_FULL_ROLLOUT`: **implement and test complete no-refill B>1 free-running E2E decoder with valid exact/approximate KV modes**. The present gpu_diagnostic is only a teacher-forced **diagnostic**, so it cannot close F1. Include reference native dynamic exit and LoopSpec/WaveFront if verified and feasible in the same engine; otherwise mark `not_implemented`.
- Final heldout n >= 500 per task when possible, paired bootstrap confidence bounds, multiple lengths/seeds/B values and per-model audit. If n insufficient, report inconclusive. No thresholds tuned on heldout questions.

## Branch-selection playbook
- **Low h1 Top-1 but low one-step correctability** -> probe geometric/secant/subspace initial trajectory; retain constant comparator.
- **Low L2 but many Jacobi sweeps** -> test loss/objective misalignment, corruption propagation, numerical precision, and structured correction.
- **Good oracle but poor online** -> threshold calibration on training/dev only, false-stop and margin analysis; keep post-hoc oracle disjoint.
- **Huginn direction-specific contraction** -> test Jacobian-vector distribution; do not assume uniform contraction or that Ouro behaves alike.
- **Poor full-width B64/128 cost** -> prioritize sparse/block correction and exact-anchor lanes, but do not claim FLOP reduction gives GPU speedup automatically.
- **Newton surrogate causes divergence** -> inspect operator norm and direction mismatch, try damping/ridge/shorter blocks; do not reject all structured correction from identity alone.
- **Cache accurate but slower** -> separate serial commit cost from solver cost, design selective exact-repair/fallback and compare truncation; do not hide overhead.
- **Apparent accuracy improvement on 50 math examples** -> require larger held-out sample; it may be noise, not speedup preserving quality.

## Integrity, deliverables and stopping
At completion produce `RESULTS.md` with (i) failure modes and eliminated hypotheses, (ii) still-live directions, (iii) chosen method with exactly matched baselines, (iv) full set of source SHA/models/precision/KV/splits, (v) free-running B1/16/64/128 E2E throughput and quality if executed, (vi) raw record checksums and failed jobs, (vii) GPU-hours and rerun reproducibility, (viii) explicit limitations.

Stop execution at 48 wall hours or 384 GPU-hours maximum, earlier only for safety/authorization. Budget exhaustion means `INCONCLUSIVE_BUDGET` unless a fully verified result supports a narrower claim. An inconclusive hypothesis is not globally NO-GO. A new research wiki Finding/Decision requires separate explicit checkpoint save.
