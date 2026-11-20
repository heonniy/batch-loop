# 8xH200 adaptive research program: prediction, correction, allocation, verification

**State:** preregistered implementation/protocol; NO new GPU measurements. Target budget: **up to 48 hours of allocated H200 time (384 GPU-hours)** with event- and evidence-driven branch selection. Only actually provisioned GPUs may be used. This is not a promised scheduled run.

## 0. Research objective and guardrails
Frozen pretrained weights. Huginn-3.5B D8/16/24/32, Ouro-1.4B/2.6B D2/3/4. Dory only after checking native adapter, model revision and parity; never manufacture a comparable depth. Primary setting H200 BF16, correct native per-depth FULL KV, fixed-depth no-refill with batch B=1,16,64,128; prompt lengths 256/1024 and greedy output 64/128/256. Separate native dynamic exit and approximate shared-KV as clearly labelled secondary experiments.

Primary outcome: **paired confidence-bounded throughput/TPOT at matched output quality** against optimized fixed-depth AR and depth truncation, including coda/head, prefill, KV, runtime, all correction/fallback cost. Report per-model and per-task. No per-core oracle result qualifies.

Define fidelity tiers:
- `exact`: full sequence same as fixed AR under matched dtype, GPU batch shape, and **cache bank parity**. If exact logits claimed, report numerical error and BF16 reduction caveats.
- `quality_noninferior`: task accuracy/NLL meets prespecified non-inferiority bound on independent held-out data with paired CI; may diverge from AR sequence.
- `approximate`: other output, explicitly labelled. Never use accuracy-only accidental improvement on n50 to justify a fidelity claim.
- `oracle`: uses future exact AR states/tokens and can inform upper bound only.

Non-inferiority examples to be preregistered before final test: math strict accuracy lower CI > -0.03 absolute; perplexity/NLL upper CI < +0.02 nats/token; and task-appropriate retrieval factuality. For exact arms, sequence and KV parity must be **100%** on fixed test with platform numerical tolerance recorded, else label approximate. If n or CI cannot resolve the margin, label `inconclusive`, not `pass`. Never tune thresholds on held-out test.

## 1. Hierarchical hypothesis tree (machine-readable in tree.json)
G0 SCIENTIFIC/ENGINE gate -> G1 HARDWARE & DYNAMICS -> P PREDICT -> C CORRECT -> A ALLOCATE -> V VERIFY -> J JOINT -> F FINAL.
Multiple branches can stay alive; evidence can reopen an inconclusive branch under an explicit alternative mechanism, rather than declaring NO-GO from a short pilot.

### G0 — foundation / confounders
- Pin `batch-loop`, WaveFront, Huginn/Ouro model revisions, tokenizer, torch, CUDA, FA kernel, native KV bank semantics and B/order.
- Resolve old E5 tentative-KV carry and `jac_exact` BF16 disagreement; implement distinct candidate and committed KV. Exactly compare (a) diagnostic oracle teacher forced, (b) online approximate carry, (c) online exact serial commit, (d) hybrid selective commit. Fix broken fidelity prior to any full E2E claim.
- 16-prompt FP32 parity + BF16 B1 and B16 row-permutation checks, depth-wise cache after every token, random-order controls, native AR. Guard pure code bugs and batch numerical sensitivity.
- Freeze data/metric registry: select unrelated train/dev/test question IDs, deduplicate overlaps; preserve independent n >= 500 final per task where possible.

### G1 — core GPU feasibility / dynamical priors
- Repeat E1 width matrix (full physical `step_many`, multiple warmups, CUDA events, 30+ timed repeats, peak allocated/reserved HBM, lock occupancy) and actual AR upper/lower bounds.
- Record full trajectories: `||delta_r||`, cosine delta_r/delta_{r+1}, r-to-D logit KL/Top1, relative L2, finite-difference directional gain (eps ladder), randomized JVP and singular-direction probes where feasible. Separate attention/KV error.
- Test whether depth transitions exhibit stable geometric, low-rank, stationary, late convergence or token-specific behavior. Reject unsupported claims of global contraction.
- Early headroom test is **cost planning only**. A low oracle ceiling may redirect toward sparse correction or smaller batch rather than stop research.

### P — predictor families; fixed corrector for fair comparison
Baseline Constant h1, Linear h1+(r-1)delta, previous trajectory and shifted previous trajectory. Vary exact prefix warmup m=1,2,3,4,8 (m<=D).
Candidates: geometric-delta with robust ratio/clip, secant extrapolation from h1/h2/h3, rank-r recent-delta subspace / DMD / ridge affine dynamics (no learned evaluation-future leakage), polynomial/Anderson extrapolation, selected shallow+late anchors, lightweight correction-aware head (optional post-training lane, frozen F).
Metrics: actual init cost; per-depth L2/cosine, final KL/greedy, post-1/2/4 rounds quality and number of *physical* core rows; worst-10% prompts and held-out transfer. A predictor is better only if `init wall + corrective wall` improves over baselines at quality. Inspect failure clusters instead of discarding an entire family.

### C — corrector solver families; pin predictor init
- Classical full-depth Jacobi/Picard exact convergence reference.
- Damped/relaxed Jacobi; history-aware Anderson with regularized constrained least-squares and finite memory; compare triangular Anderson/ParaTAA structure (not vanilla global Anderson conflation).
- Newton-like triangular correction `delta_r = e_r + A_r delta_{r-1}`. A=0 Jacobi-like reference, A=I, scalar-depth A=alpha_r I (offline calibrated from training split only), diagonal and block-diagonal, low-rank JVP subspace. Test stability/conditioning/precision.
- Parareal coarse G=identity/one-step as negative controls, geometric/semi-linear G candidates; account for ALL coarse/core calls.
- JVP and residual-based error propagation; validate low-cost A_r per-row under fixed KV with independent finite differences; do not use dense full Jacobian in measured decoding.
- Favor cheap corrections only if `cost of A + F rows + required rounds` decreases real GPU time. Report cases where extra CPU sync or Torch kernel launches dominate.

### A — correction allocation and adaptive depth
A1 full-depth correction baseline; A2 contiguous suffix/prefix windows; A3 overlapping depth blocks; A4 residual/gradient influence weighted correction; A5 coarse-fine structured sparse correction; A6 hybrid exact anchors; A7 per-example depth-dependent width/packing if engine supports rowwise masks.
Selection signal must be available online: local residual, margin, inexpensive directional JVP estimates or cheap row-local norm. Compare against same number of randomly chosen depths and uniform windows as strong controls.
**Critical:** altered depth KV must not contaminate other depths or next tokens; skipped states do not imply correct historical slots. Different batch partitions can change BF16 numerical output.

### V — online termination and verification
Test gates: fixed k, residual, max correction update, multi-round argmax stability, logit margin, cheap calibrated error estimator, estimated influence residual, exact fallback.
Separate labels `heuristic`, `empirically calibrated`, `provable under explicit bound`. An argmax-margin certificate applies only if a valid logit error bound is independently verified; it does not certify KV.
Measure false-stop rate, fallback work, loss to AR, gate overhead, exact cache cost, sequence first mismatch and task quality.

### J — joint candidate optimization
Use same pilot data/hardware to select Pareto candidates, then run paired factorials of top 2–3 Predictor x top 2–3 Corrector x top 2–3 Allocation x 2 gates, **not an uncontrolled 10x10x10 Cartesian sweep**. Preserve at least two mechanism-diverse alternatives even when early pilot leader is noisy. Do not optimize decisions on test split.

### F — final audit
Run held-out n>=500 GSM8K and MATH-500 where possible; controlled retrieval/NIAH and HotpotQA with consistent prompt/score; decode lengths 64,128,256 and prompt L256,1024,2048 where memory allows; B1,16,64,128; multiple seeds/row orders/dtypes for numerical checks. Match native baseline, batch occupancy, padding/left truncation, EOS and stopping. Report paired bootstrap 95% CI, mean/p50/p90 TPOT, TTFT, requests/s and output token/s, task accuracy, KL/NLL, sequence/KV parity, HBM, core rows, GPU-hours, failure/OOM/unsupported. Evaluate failure clusters and reproducibility on two runs.

## 2. Resource allocation: maximum 384 GPU-hours, subject to real host availability
| Phase | target GPU-hours | Action when uncertainty remains |
|---|---:|---|
| G0 parity & engine audit | 40 | STOP full evaluation until exactness fixed |
| G1 dynamics/hardware | 44 | add two competing hypotheses if surprising |
| P predictors | 64 | retain baseline plus two diverse mechanisms |
| C correctors | 72 | validate structure JVP before investing in low-rank |
| A selective allocation | 48 | test random/uniform equal-row controls |
| V online gates | 32 | quantify false-stop and cache fallback |
| J joint optimization | 40 | require paired holdouts and avoid champion overfit |
| F final validation | 44 | reassign to robustness if no viable candidate |
**Total: 384 GPU-hours**, an upper limit not a claim of booked resources. Per-GPU slot scheduling, memory caps and timeouts are mandatory. `--allow-gpu` only on an allocated host; no burner stopping/restore script in this project.

## 3. Adaptive decision protocol: explore rather than premature NO-GO
- Pre-register each node before seeing its result: hypothesis, null, alternative, metrics, comparisons, seed, prompt IDs, data split, budget, paper claim, epsilon.
- **First 10–20%** sample/elapsed budget screens for code and catastrophic failure. A screening fail is `diagnose`, not global `NO_GO`.
- `investigate` if n low, CI crosses threshold, baseline inconsistency, OOM, batch-shape numerical drift, or non-monotone quality. Run larger n, alternate seed, fixed-precision diagnostic, code audit and equal-compute controls.
- `promote` if source-backed method & deterministic correctness evidence plus plausible per-B cost and improvement vs at least two baselines; choose next level by evidence and novelty, not intuitive enthusiasm.
- `deprioritize` only with sufficient negative controls and interpretable failure mechanism, logged. A failed pretrained Identity Newton does not imply all corrected dynamics fail.
- Keep **reserve exploration** (>=20% of non-foundation budget) to alternatives, not solely champion optimization.
- Never select/tune using final test; use train for predictors/calibration, dev for adaptive selection, freeze final config before test.
- Budget exhausted -> `inconclusive_budget` with evidence, not scientific NO-GO.

## 4. Evidence, audit and publication
Every trial receipt: `experiment_id`, hypothesis node ID, exact CLI, all Git SHAs, model/tokenizer revisions, GPU id/name, numeric dtype, pinned `WF_ROOT`, prompt IDs/hash and split, seed, KV policy, batch, prompt/output lengths, correction methods/allocations/rounds, metrics, synchronized timing, HBM, baselines, output paths and SHA256, crash reason. Statuses `planned, implemented, executed, failed_infrastructure, failed_correctness, inconclusive, supported, contradicted`; do not infer an executed status from the presence of a job file.

All raw artifacts outside Git initially. Commit source first, then immutable summarized results + checksums + evidence and anomalous traces; keep any sensitive prompt data out of repo. GitHub wiki is read-only until explicit checkpoint authorization. A code agent must re-read the relevant Reference DB paper before each family implementation and cite method equations and limitations in the source review.

## 5. Success and research honesty
A positive claim requires reliable held-out noninferiority or exactness **and** repeated, measured complete-model speedup at the user-targeted offline B. If only B1 works, label B1 only. If only approximate quality works, label approximate and show Pareto curves with truncation and native dynamic exit. If no viable solution, characterize why (predictor quality, Jacobian error, online gate, KV commit, GPU saturation) with evidence, not one early negative PoC.
