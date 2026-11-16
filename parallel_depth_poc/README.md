# Training-Free Parallel Recurrent-Depth PoC

**Status:** New isolated branch with a research plan, numerical reference solvers and
offline trace diagnostics. **No Huginn/Ouro GPU evidence or speedups have been collected.**

Start with [PLAN.md](PLAN.md). This is intentionally independent of existing
Phase-01 to Phase-04, `dynamic_sd/`, and FERRY research. The repository-level
`AGENTS.md` lock remains in force for old Phase-04 claims and GPU jobs. A
new branch does not silently authorize using a GPU or altering existing jobs.

## What is implemented

- `solver.py` — depth-indexed exact sequential reference, training-free
  linear/constant initialization, synchronous depth-Jacobi and block-Parareal
  with explicit coarse function and batched core interface.
- `trace_analysis.py` — relative-state-error and cosine measurements for
  constant, linear and causal previous-token predictors on recorded trajectories.
- `collect_states.py` — opt-in CUDA-only fixed-depth trace collection with the
  existing Phase-02 Huginn/Ouro adapters; no latency measurement.
- `demo.py` — deterministic small CPU mathematical illustration; never report
  its error/speed as a model result.
- `tests/` — reference mathematical tests (CPU only).

The numerical functions are reference implementations. **A batched core API
that handles different recurrent depths and their KV-bank semantics is not
provided**. The existing `phase02_runtime/huginn_adapter.py` and
`phase02_runtime/ouro_adapter.py` provide model-specific single-depth hooks
but mutate their active KV banks. They cannot simply be called in parallel
with predicted hidden states: predicted branches must have isolated cache
writes, and only verified states may be committed.

## CPU-only smoke commands

From repository root in a Python environment with torch and pytest installed:

```bash
python -m pytest parallel_depth_poc/tests -q
python -m parallel_depth_poc.demo
```

To analyze a recorded fixed-depth trajectory file:

```bash
python -m parallel_depth_poc.trace_analysis \
  --trace /path/to/fixed_depth_states.pt \
  --output /path/to/analysis.json
```

The input is a Torch dictionary containing `states` with shape
`[T,D+1,B,H]` or `[D+1,B,H]`. `T` is successive teacher-forced
decode steps, `D+1` includes the initial state before loop 1, `B` is the
persistent cohort and `H` hidden size. Save optional plain metadata as
`metadata`. The loading path uses `weights_only=True`.

Example input creator for **toy** tensors only:

```python
import torch
h0 = torch.randn(3, 2, 4)  # T,B,H
v = torch.randn(3, 2, 4) * 0.1
states = torch.stack([h0 + r * v for r in range(5)], dim=1)
torch.save({"states": states, "metadata": {"kind": "toy"}}, "/tmp/toy_states.pt")
```

**Do not** use an exact future depth state as an inference-time predictor;
the file is only available after running a full sequential reference.

## Optional real-model trace collection (NOT run by this commit)

The new `collect_states.py` uses the existing Phase-02 Huginn or Ouro-1.4B
adapter and records depth-by-depth hidden states. Supply a JSONL file with
`{"prompt":"Question ..."}` for greedy continuation, or exact token IDs via
`{"prompt":"Question ...","decode_token_ids":[123,456,...]}`.
Use one mode across all rows. Confirm GPU availability and permissions first.

```bash
CUDA_VISIBLE_DEVICES=0 python -m parallel_depth_poc.collect_states \
  --family ouro14 --input-jsonl /path/to/2_prompts.jsonl \
  --batch 2 --decode-steps 8 --prompt-cap 256 \
  --output /path/to/ouro14_b2_states.pt --allow-gpu
python -m parallel_depth_poc.trace_analysis \
  --trace /path/to/ouro14_b2_states.pt \
  --output /path/to/ouro14_b2_state_errors.json
```

Use `--family huginn` for the corresponding pinned adapter. This is **E2**
trace collection only, not a GPU speed experiment. Reference parity is still
**UNVALIDATED**. Huginn uses identity-keyed recurrent initialization and
shared physical-layer KV; Ouro-1.4B uses last-exited depth-indexed KV.
Predicted branches must fork KV before E1/E3 speed claims are possible.

## Next GPU integration milestone (not yet done)

1. Pin Huginn and Ouro model revisions and document the exact author-native
   fixed-depth decode semantics. Obtain 16-prompt reference parity **before**
   any new numerical solver measurements.
2. Validate `collect_states.py` against the model author-native fixed-depth
   decode and verify its exact/approximate KV policy. Keep one decode position
   and one batch shape per record; pin RNG.
3. Implement candidate-KV fork / rollback and a genuinely batched
   `step_many(states[R,B,H], depth_ids[R])` that handles depth-indexed
   histories, RoPE, masks, prelude embedding, and current-token KV writes.
   The supplied CPU test operator is *not* sufficient.
4. E1: time exact-state oracle widths with CUDA events, plus full sequential
   baseline and HBM, at B=1,8,16,64,128 where feasible. Label unsupported
   or OOM cells; don't inflate headroom from a pure GEMM benchmark.
5. E2/E3: compare hidden-state prediction error, logit KL, token agreement,
   Jacobi/Parareal rounds, fallback and total physical wall time.
6. E4/E5: perturbation amplification and end-to-end no-refill offline
   generation. Run GSM8K/MATH-500 only after correct KV commit and token
   parity have been established.

Model caps differ. The legacy Ouro-1.4B adapter is `rmax=4`. Its 2/3/4-depth
cells are meaningful; don't run fake 8/16-depth Ouro sweeps. Huginn adapter
currently declares `rmax=16` even if the model supports deeper configured
execution; changing it requires independent validation.

**Result reporting:** write `NOT_RUN` if GPUs or validated cache forks are
missing. CPU synthetic tests prove only numerical identities, not deployable
speedups, model quality or KV-cache correctness.
