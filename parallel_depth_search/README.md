# Adaptive depth-parallel inference research (H200 x 8)

**Status: implemented research-control and diagnostic code, NOT an executed H200 campaign.**
Source branch: `research/parallel-depth-adaptive-h200x8-20261121`; parent: PoC `829076bb`.
Do not confuse commit with launched jobs or GPU evidence.

## Research question
Can frozen pretrained Huginn/Ouro recurrent core weights, with **correct per-depth KV**, support prediction, cheap correction, and selective depth allocation such that **actual full decoder throughput** beats optimized fixed-depth AR and simple depth truncation at matched output quality on offline no-refill batches?

This folder adds independent **methodology, adaptive controller, source-backed bibliography, and executable diagnostic methods**. The original `parallel_depth_poc/` and Phase-04 data are not modified.

## What is immediately executable
- `python -m parallel_depth_search.control validate`: validate preregistered hypothesis DAG.
- `python -m parallel_depth_search.control init --run-dir /some/path`: initialize append-only evidence ledger and queue, **does not launch GPU jobs**.
- `python -m parallel_depth_search.control status --run-dir ...`: show eligible hypotheses, missing evidence and experiment status.
- `python -m parallel_depth_search.control propose --run-dir ...`: recommend next tests and explain every choice using score/gate definitions and paper refs.
- `python -m parallel_depth_search.control ingest --run-dir ... --receipt result.json`: hash and record a source-backed result for a node. Refuses malformed, unsupported and duplicate receipts. Distinguishes diagnostic/empirical/oracle levels.
- `python -m parallel_depth_search.gpu_diagnostic --help`: CUDA-only teacher-forced, **non-serving** predictor/corrector probe, explicitly opt-in with `--allow-gpu`.
- `python -m unittest discover -s parallel_depth_search/tests -v`: offline control-plane tests (no GPU).
- `python -m parallel_depth_search.make_jobs --help`: deterministic diagnostic job commands and 8-GPU slot manifest; **does not start them**.

## Critical deployment boundary
The reference `parallel_depth_poc/wf_engine.py` can batch independent depth rows, but E5 `e5_decoder.py` uses candidate KV without an exact `commit()`. Teacher-forced diagnostic corrector measurements with a subsequent exact `commit` **must not be sold as end-to-end acceleration**. Exact KV commit cost must count in real output speed, and `jac_exact` may differ from BF16 AR due to row-shape floating-point effects. Full integrated multi-request decode with optimized AR + correct cache is an explicit **blocked until implemented and parity proven** gate, not a missing checkbox silently assumed passed.

For complete protocol and decision tree: [RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md) and [tree.json](tree.json).
For a coding agent executing on a machine with H200 access: [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md). For references and their limitations: [LITERATURE.md](LITERATURE.md).

## Permissions and lock boundary
Repository-level `AGENTS.md` contains Phase-04/04R locks: do not modify its code/results, interpret BF16 rank outcomes as passed, touch existing burners/processes or run unapproved jobs. This is a new additive study with explicit user scope for H200 experimentation, **not** an instruction to bypass shared-GPU owner authorization or reservation. Start GPU only on a host with the actual allocated devices; never kill or pre-empt other processes. If H200 access is unavailable write `NOT_RUN: no_device_access`.

Research wiki is **read-only** for this task. Follow `heonniy/research-wiki/UPDATE_POLICY.md`, `AGENTS.md`, and reference cards. Do not write wiki findings without an explicit research checkpoint save and source-backed completed runs.
