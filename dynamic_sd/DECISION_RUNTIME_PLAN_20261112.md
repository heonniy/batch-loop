# Dynamic-depth SSD: E3 gate validation and adaptive-execution research plan (2026-11-12)

**Status:** experimental plan and code scaffolding; NO NEW GPU EXPERIMENT RESULTS. Source baseline: `dynamic-sd/characterization-20261111` (H200 eager WaveFront), primary report `dynamic_sd/RESULT_REVIEW.md`. This is a new independent experiment. Preserve prior results and Phase-04 restrictions.

## Motivation and three decisions

- **Decision 1 — Speculate or AR?** The native dynamic target may be cheaper than an extra draft.
- **Decision 2 — How deep and wide?** Choose recurrent draft depth `d` and proposal length `K`; best static d stayed d=1 on tested Ouro, d=4 on Huginn even though live exits widened the d=4-vs-8 gap. New best-d switches are **not demonstrated**.
- **Decision 3 — Which work shares a GPU execution step?** Exit/rejection shrinks the active wave, but shrink is not automatically overhead; prove physical idle or padded cost against a tuned dynamic-AR/SD baseline.

Our record IDs: `F-079` speedup denominators, `F-080` post-hoc acceptance-vs-exit signal, `F-081` no argmax-d crossover, `F-082` Ouro-only lockstep batch barrier, `F-083` BF16 output mismatch, `F-084` schedule-sensitive shared KV, `F-085` small natural B1 verifier tail and **no isolated live-gate cost**. Prior `F-020/F-035` warn that smaller microbatches can be slower. New methods must improve on the best *native dynamic* baseline rather than fixed-depth AR.

Adjacent published methods `P-2406-14066` (TurboSpec), `P-2603-01639` (Learning To Draft) and `P-2602-20217` (KnapSpec) already explore adaptive speculation. **Learning To Draft's depth is draft-tree depth; KnapSpec selects standard attention/MLP layers.** Neither directly proves a recurrent-loop controller's improvement.

## Stage 0 — first fix E3's validity

The old `dsd_run.py` timer includes **prefill**; the old `bench_r.py` forcibly disabled exit gates. Thus “Live≈Replay” is a comparison of entire eager paths, **not** a gate-only experiment. In the WaveFront core, Ouro uses `float(lam[i])` and Huginn uses `.tolist()`, potentially synchronizing CPU/GPU. Gate costs must be measured explicitly.

### Four arms, with a controlled three-arm core

`e3_gate_probe.py` (initial **B=1 committed AR teacher-forced trace**, not full speculative WaveFront):

| Arm | Depth decision | Extra gate calculation | GPU→CPU readback |
| --- | --- | --- | --- |
| A `replay` | force the recorded AR-live exit depth | none | none |
| B `shadow_gpu` | same forced depth | post-R GPU gate | none |
| C `shadow_readback` | same forced depth | post-R GPU gate | scalar CPU readback |
| D `live` | native actual gate | native gate in core | native behavior |

A/B/C consume the **same source tokens**, prefix, FULL KV, native model, dtype, forced depths, Core calls and Coda positions; compare only if predicted continuations and actual forced depths match. **B−A and C−B** estimate the shadow implementation's incremental per-position cost. Shadow gate GPU events and host blocked time must be reported separately; the latter is not additive “GPU latency”. **D−A is NOT gate-only latency** unless naturally realized depths/outputs match and even then the dependency graph may differ. D is a sanity diagnostic.

**Important phase boundary:** A/B/C replay a *committed AR* trace, not speculative branch trajectories. Native WFD/DtV gate-overhead claims require an additional branch-lineage trace and identical in-flight speculative state; such an SSD-specific replay/shadow implementation remains **unimplemented** in this initial PR. Do not relabel this probe as an optimized SSD result.

Use the existing `dsd_run.py` `per_sample` records (`tokens_ar`, `depths_ar`, `prefix_len`), source config and `meta.wf_commit`. No synthetic “ideal exit” traces. Source model and tokenizer must be identical, and source file SHA-256 must be preserved in output. Force `kv_budget_s=0` FULL, greedy, same seeds; reject changed prefix lengths. Warmup≥2; scored repetitions≥5; alternate arm order. The output lists every trial, observed depth, predicted argmax match and wall time, plus explicitly marked paired diagnostics.

### E3's physical serving extension (future implementation, not measured by these scripts)

After AR shadow control parity, instrument native WFD/DtV gate call sites and record the *same in-flight speculative branch lineage* under B and C. Compare: Eager PyTorch vs CUDA Graph in **one** engine, Gate GPU-only vs CPU readback, grouped rows vs physical compaction, B=1/8/32, prefill/decode split. Use CUDA events for GPU execution, NVTX/Nsight Systems + CPU wall for host wait. Record Prelude/Core/Coda, gate kernel, `.tolist` sync, `freeze_kv`, scheduled/actually executed/padded rows, HBM allocation, first mismatch, committed target-equivalent tok/s and quality. No CUDA Graph measurement may be claimed until executed.

Native Ouro and Huginn thresholds must be calibrated independently (Ouro τ=0.1/0.4/0.7, Huginn ε=0.05/0.1/0.2, nondegenerate realized D). Huginn relative-L2 used by WaveFront is not a blanket native-cosine equivalence claim. After 32 smoke, use 100 pilot and 200 held-out prompts per relevant model/dataset, repeat primary timings at least five times. Check `F-083`: strict BF16 equality previously failed and cannot be retroactively reported as passed.

**Stage-0 GO:** exact forced-path comparability, a real per-stage cost breakdown, stable measurable gate+host cost under eager and optimized execution. **STOP:** source/replay parity failure, topology drift, or CPU host waits not isolated.

## Stage 1 — whether dynamic choices have real oracle headroom

New `decision_oracle.py` CPU tool accepts multiple **actual measured** `dsd_run.py` JSON/GZ files, not aggregate speedups, and extracts full-request `AR/WFD/DtV × d/K` action timings. Require same model/dataset/exit/τ or ε/precision/prompt cap/sampling/seed/FULL KV/AR target sequence. Non-equivalent outputs are ineligible rather than silently called lossless. The tool reports action coverage, train-selected best globally fixed action, and per-held-out-prompt hindsight oracle. Oracle is an **optimistic upper bound on full-request choice**, not a token-level controller or real physical speedup. A true token-level oracle requires snapshot/replay of *common committed prefixes* with equal continuation semantics.

**Stage-1 GO:** nontrivial held-out headroom over tuned static `d,K` with sufficient identical-output action coverage and repeated timings; **STOP:** best fixed strategy already matches oracle or strict equivalence fails.

## Stage 2 — online controller (design only)

If Stage 1 passes, use only available **pre-decision** signals: partial hidden/exit-gate trajectory, previous acceptance lengths, active rows, context length and measured GPU cost table. Choose AR or speculate and, if chosen, draft d/K. Never use target D observed after verification, future ground truth, or a hindsight trace as a deployable feature. Compare fixed d/K, AR-live, acceptance-only, hardware cost heuristic and learned controller. Include controller compute, CPU/GPU readbacks, latency and quality; evaluate on separate prompts and tasks. **Stage-2 GO:** measured target-equivalent goodput above the best matched dynamic WFD/DtV baseline in a validated engine.

## Stage 3 — batch scheduler (design only)

Do not assume “no-refill = lockstep”. Compare same-engine Ouro/Huginn B=1/8/32/64 at matched exits, draft budgets, KV, precision and correctness: fixed cohort lockstep, physically compacted lockstep and depth-asynchronous ready-row scheduling; refill is a separately labeled option. Existing `dsd_batch.py` executes only batched **Ouro AR/DtV**, not WFD or Huginn. Revisit `F-020/F-035` and count microbatch launch/sync overhead. **Stage-3 GO:** physical end-to-end goodput against optimized dynamic AR + SD, not artificial oracle/analytical row savings.

## Reproduce / handoff

Run *only CPU validation* anywhere without models/GPU:

```bash
python3 -m unittest discover -s dynamic_sd/tests -v
python3 -m py_compile dynamic_sd/e3_gate_probe.py dynamic_sd/gate_probe_common.py dynamic_sd/decision_oracle.py
```

On a separately **authorized GPU host**, after verifying pinned model, tokenizer, datasets, WaveFront SHA and `WF_ROOT`:

```bash
WF_ROOT=/path/to/wavefront-decoding python3 -m dynamic_sd.e3_gate_probe \
  --reference dynamic_sd/characterization_results_20261111/main/ouro14_gsm8k_live_tau0.4_d1.json.gz \
  --limit 8 --max-decode-positions 32 --warmup 2 --repeats 5 \
  --modes replay,shadow_gpu,shadow_readback,live \
  --out /new/output/e3_ouro14_tau04.json
# Repeat separately with huginn_gsm8k_live_eps0.1_d4.json.gz
```

Once **new actually executed** d/K action receipts exist:

```bash
python3 -m dynamic_sd.decision_oracle collect --inputs /new/output/d1.json.gz /new/output/d2.json.gz --out /new/output/actions.json
python3 -m dynamic_sd.decision_oracle analyze --table /new/output/actions.json --out /new/output/oracle.json
```

Save new raw reports under a NEW timestamped results directory only after GPU execution; include SHA/source commit, command, n, hardware/backend, failures and negative cases. Do not update `RESULT_REVIEW.md`, its receipt manifest, wiki cards, `phase04/`, or any Accepted/Rejected Decision. **No GPU work was launched in this PR.**
