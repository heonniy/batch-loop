# Dynamic-Depth SD: Runtime Gate Validation -> Oracle -> Online Policy -> Batch Scheduling

Date: 2026-11-12. **Status: source + runnable diagnostics, no GPU experiments run in this branch.**  
Base: `dynamic-sd/characterization-20261111` (kept intact).  
Primary evidence: [EXP-20261111-DSD-CHAR](https://github.com/heonniy/research-wiki/blob/main/experiments/EXP-20261111-DSD-CHAR.md), [F-079](https://github.com/heonniy/research-wiki/blob/main/findings/F-079.md)–[F-085](https://github.com/heonniy/research-wiki/blob/main/findings/F-085.md). Hypotheses H-SPEC-001 and H-SPEC-002 remain Proposed.

## What changed after reviewing E3

The previous `dynamic_sd/RESULT_REVIEW.md` reports live-vs-replay wall-time differences around 3% or less in eager B=1; **this is not a gate-only overhead measurement**:
- `dsd_run.py` AR/WFD/DtV starts timing before prefill; `bench_r.py` explicitly disables gates.
- Native Ouro gate calls `float(lam[i])`; Huginn's relative-L2 gate calls `.tolist()`, both potentially synchronize GPU and CPU.
- AR-live trace replay is not a guaranteed same-target semantic trace for arbitrary speculative descendants.
- No CUDA Graph, gate-only timing, batched WaveFront, or physical compaction control was run.
- Source report's blanket Huginn live FULL FP32 parity is not established in its explicitly listed receipt manifest; do not treat BF16 exactness failures as passes.
- Existing `F-020/F-035` show full-cohort work can beat smaller regrouped batches, so no automatic microbatch/refill win.

## Stage 0 — Gate and runtime validity (prerequisite, highest priority)

**0a. Matched replay shadow-gate diagnostic (implemented).** Start from recorded AR-live depths and run three `dsd_run.py` variants on the SAME model, prompts, dtype, FULL KV, seed, `--exit replay`, `--replay-from`:
1. `off`: replay only; do not compute shadow gate.
2. `gpu`: compute the genuine model gate / Huginn relative-L2 with hidden states produced at each R step, but do not read the score to CPU or alter exiting.
3. `cpu`: same score plus the same `.tolist()`/Python per-row update shape as native. Still **do not** use the shadow score for exit, rollback or KV publication.

Instrument number of gate calls, eligible rows and CPU wall time inside the shadow. Use `compare_shadow.py` to **refuse paired timing comparison** unless the replay samples' tokens, depth/width work and executed recurrent row counts match. Pairwise differences are *diagnostic upper-level totals*, not an isolated kernel event measurement; prefill is still included, and the replay trace is not necessarily an exact native-target execution path. The difference `cpu - gpu` estimates readback/Python costs in an otherwise matched eager replay geometry, not a production latency saving. The `gpu - off` estimate includes added gate kernel launches.

**0b. Required GPU-host follow-up (NOT IMPLEMENTED):** NVTX/Nsight Systems and CUDA Events around the native gate, CPU stalls, CUDA Graph captures/fallbacks, Coda, Prelude, Core and decode-only intervals. Port the three conditions to one **optimized common backend** with fixed graph shapes / same kernel path; do not compare absolute WaveFront-eager TPS to vllm-rlt TPS across engines. On `vllm-rlt v0.2.0`, fixed-depth Ouro self-speculation is documented; native dynamic SD/Huginn SD is **not** a drop-in supported combination. Any port requires a separate parity/semantic gate. No vllm-rlt modification here.

**0c. Quality gate:** On FULL KV, repeat fixed and live FP32 exact greedy checks for Ouro/Huginn, including missing Huginn live FULL n>=8. Identify BF16 first mismatches; continue policy evaluation only with defined target-equivalence contract, reporting strict/relaxed fidelity separately. Use >=2 warmups, >=5 independent timed repeats, isolated GPU, per-repeat bootstrap over prompts, exact model/driver/attention backend source hashes.

**Falsify Gate-cost concern:** under genuinely matched optimized execution, GPU gate and CPU synchronization both remain negligible in decode-only critical-path time across relevant B and prefix lengths. Eager shadow-only results cannot falsify it.

## Stage 1 — Speculate? Choose draft d/K? (implemented offline analysis, not executed)

Generate counterfactual *decision-round* candidates from the SAME committed prefix checkpoint: `action in {AR,WFD,DtV(d,K)}`. Must record `prefix_id`, `prefix_hash`, `target_hash`, `exit_policy`, `kv_policy`, `batch_regime`, `backend`, `precision`, `action`, `wall_ms`, `committed_tokens`, `target_exact`, `measured` and `repeat`. Do not mix divergent continuations or compare a fixed target against dynamic AR; only decisions that match the target for the measured prefix may enter the oracle. Training/calibration/held-out sets are prompt-disjoint.

`oracle_frontier.py` reports:
- best **single static action** based on total measured committed tokens / total measured time across complete prefix groups;
- best action per prefix and its optimistic **one-round** oracle ratio;
- action coverage, parity failures, repeated timing variability.
This **does not** synthesize an achievable multi-token rollout or latency speedup; live policy will have additional CPU/GPU decision cost and new continuations. The optimistic frontier is a headroom screen only.

**Stop rule:** if the oracle frontier offers no material gain over the best static d/K or all candidate actions fail target parity, do not build a learned controller. Never count future exit depth as an online feature.

## Stage 2 — Causal online controller (DESIGN ONLY)

Inputs available before choice: current/past loop states and gate margins, past accepted/rejected events, context length, physical batch occupancy, hardware cost profile. Outputs: AR vs SD, draft loop count `d`, proposal length `K`; no feature from current token's future verifier exit. Compare `fixed best`, simple history-only/acceptance-only heuristics, oracle and causal learned policy. Include predictor/gate inference, memory and scheduling overhead. Prohibit training or selecting on held-out test prompts. For quality, preserve the *matched native dynamic AR* semantics; label any lossy BF16 behavior.

**Stop rule:** if policy cannot beat best fixed baseline at equivalent fidelity including control overhead, no algorithmic gain.

## Stage 3 — Optimized batched scheduling (DESIGN ONLY)

Test B = 1/8/32/64 on same backend, identical admission and sampling, Ouro then Huginn. Separate *no-refill* (cohort admission) from *lockstep* (max-depth token-step barrier). Baselines: no-refill lockstep, depth-asynchronous within the frozen cohort, optional refill allowed, batch AR, batched DtV and **batched WFD only after it exists**. Collect per-round active/padded/committed/rejected rows, graph bucket, gate sync, Core/Prelude/Coda wall and HBM. Measure token generation accuracy, throughput, TTFT/TPOT/E2E and statistical variability.

**Stop rule:** a scheduler only matters if it beats the strongest same-engine, quality-matched dynamic AR/SD baseline rather than simply a weak lockstep control.

## Commands / publication

See [RUNBOOK.md](RUNBOOK.md). All produced results must go to an untracked directory under `dynamic_sd/results/followup_20261112/`, with commands, source SHAs, per-run JSON, GPU device description and failed arms. Do not edit earlier 2026-11-11/12 receipts. No new research Finding or GO/NO-GO is made by committing this plan and code.
