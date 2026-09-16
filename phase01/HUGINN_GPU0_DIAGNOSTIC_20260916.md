# Huginn GPU0 isolated diagnosis — 2026-09-16

## Purpose

The four-GPU validation currently running on GPUs 4/5/6/7 must remain untouched.  This diagnostic is additive and is intended only for the otherwise-free GPU0.

The goal is not to start another full sweep.  It answers the specific reasons the current Huginn path differs from the single-H100 results reported by CDB/other Huginn work.

## What we already know

1. Upstream Huginn explicitly warns that batched adaptive-compute evaluation is not B=1-equivalent because the shared batched KV cache is populated up to the deepest sequence in the batch.  Therefore B>1 adaptive `generate_with_adaptive_compute()` is not a correctness reference for our scheduler.
2. `latest-m4` is primarily a missing-KV lookup rule; it does not by itself collapse all recurrence slots.  `latest-m4-compress-s4` is the policy in the current code path that actually remaps/reuses recurrent cache indices.
3. The old capacity probe feeds a dense `B x context` tensor directly to HF `model.generate()`.  At B=64/context=2048 this means a 131,072-token full-prompt forward, which mixes prefill activation pressure with decode-KV capacity.  That is not the same execution contract as CDB's custom serving runtime.

## GPU0 experiment

Run only:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_huginn_gpu0_diagnose.sh
```

This creates a new result root and never reads/writes the current four-GPU result root.

### A. KV-policy memory check

At B=8, context=512, depth=16, one generated token:

- `full`
- `latest-m4`
- `latest-m4-compress-s4`

Expected diagnostic pattern: `full` and `latest-m4` should be similar if all fixed-depth recurrent states exist, while `compress-s4` should lower the peak.  If that pattern does not appear, inspect the pinned remote implementation before any new serving claim.

### B. Is the B64/L2048 OOM already caused by the initial full-prompt forward?

With `latest-m4-compress-s4`, depth=16 and only **one** generated token, sweep:

- B=64, L=512/1024/1536/2048
- L=2048, B=16/32/48/64

Interpretation:

- if `new_tokens=1` already OOMs, later decode-cache growth is not required; the full-prompt/prefill-side peak is sufficient;
- if one token fits but longer decode fails, decode/cache growth is materially involved.

### C. Decode-growth proxy

For B16/L2048 and B32/L1024, compare `new_tokens=1` vs `8` using `compress-s4`.

This is intentionally a proxy, not a perfect phase-separated prefill/decode profiler.  It is enough to decide whether we should next invest in chunked/paged prefill or in decode-cache changes.

### D. B=1 semantic sanity

Use B=1 only, threshold=0.26, Rmax=16, four Alpaca prompts, fixed 32-token decode:

- `latest-m4`
- `latest-m4-compress-s4`

Record exact output hashes and peak HBM.  This is a bounded sanity check for choosing a Huginn KV policy; it is not a quality benchmark and not a throughput result.

## Decision after GPU0

### Case 1 — compress-s4 fits moderate cells and B64/L2048 one-token already OOMs

Treat the old B64/L2048 result as a **monolithic-HF-prefill capacity limit**, not proof that Huginn decode B64 is infeasible.  Next implementation should separate prefill from decode (chunked/paged prefill or prebuilt KV) and then measure a real survivor-compaction no-refill runtime.

### Case 2 — compress-s4 remains close to full/latest-m4 in HBM

Do not proceed to a large Huginn sweep.  First inspect whether the pinned checkpoint/runtime actually honors recurrence-slot compression on the selected path.

### Case 3 — B=1 outputs materially disagree between latest-m4 and compress-s4

Keep `latest-m4` as semantic reference and treat compression as a separate approximation requiring a proper quality study.  Do not silently use it to make high-B cells fit.

## Research contract after this diagnostic

For Huginn, the intended paper path is:

1. collect adaptive exit-depth traces at B=1;
2. freeze one validated KV policy;
3. build/measure a separate no-refill batch runtime that physically removes exited sequences from recurrent work;
4. keep prefill capacity and decode scheduling as separate measurements;
5. never use upstream B>1 adaptive `generate()` parity as the definition of correct no-refill semantics.

The Ouro experiments already running on GPUs 4 and 5, and the existing Huginn gate artifacts on GPUs 6 and 7, remain valid and must not be restarted or overwritten by this diagnostic.
