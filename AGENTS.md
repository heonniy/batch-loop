# AGENTS.md

## Current priority — fair trace characterization, then real no-refill runtime (2026-09-16)

Read in order:
1. `phase01/README.md` — owner-facing experiment state and research question.
2. `phase01/CURRENT_RESEARCH_PRIORITY.md` — bounded next block and stop conditions.
3. `phase01/EXPERIMENT_REVIEW_PROTOCOL.md` — evidence, publication, and review contract.
4. `phase01/results/20260915_poc/RESULT_REVIEW.md` and `review_manifest.json` — bound review of the first H100 PoC.

### What is already evidence

The H100 GPU1 PoC published at result commit `4ec78c1f5c0df7925f9888c7db3d75769e494ee7` collected real model exit-depth traces and then ran an offline scheduling simulator. It supports an **exploratory scheduling-opportunity claim**, not a wall-clock speedup claim.

- Ouro-1.4B showed substantial oracle slot-utilization headroom over FIFO at B=4..64.
- Huginn at convergence threshold 0.12 started near full slot utilization and showed little oracle room.
- Simple causal predictors recovered only a small fraction of Ouro's oracle headroom.
- A comparison defect was identified after publication: each strategy truncated the post-sort remainder independently, so strategy totals were not always based on exactly the same token subset.

Commits `a1e8ced` and `0bb63b7` add the fair-subset analyzer and the next H100 sweep runner. They are **source/plan checkpoints only** until a corresponding result publication is committed.

### New-work execution contract

The next instructed GPU block is Phase01-R1 only:

1. **R1-A Fairness:** freeze the same eligible token/request subset before strategy reordering; assert identical token-loop work across strategies.
2. **R1-B Scale:** Ouro-1.4B request pool >=1024 when practical; sweep B=16,32,64,128,256 and optionally 512.
3. **R1-C Thresholds:** Ouro q={0.2,0.4,0.5,0.7}; Huginn convergence threshold={0.5,0.35,0.26,0.12}, with threshold semantics checked against the model implementation.

Stop after R1 and publish/review the result before implementing a predictor or a custom runtime. Do not automatically expand to a broad Cartesian sweep.

If R1 preserves meaningful headroom, the next separately approved block is Phase01-R2: implement **real no-refill recurrent execution** and measure H100 wall-clock recurrent time, throughput, and offline batch-completion time (BCT). Slot utilization alone must never be presented as runtime speedup.

### Repository / GPU safety

- Use one explicitly assigned H100 only. Never assume a GPU number without checking the owner/run command.
- Do not kill unknown PIDs, reset GPUs, force-push, rewrite historical results, or silently alter model revisions/datasets.
- Do not hot-patch an in-flight source tree. Bind each run to a clean source commit and a separate output root.
- Preserve adverse/negative results. A threshold/model with little headroom is evidence in that measured scope, not a failed experiment to hide.
- Never invent a launch receipt, completed GPU run, wall-clock result, or missing artifact.

### Commit and publication protocol

Follow the same result-bound discipline used in `sub-moe`:

1. **Plan/source checkpoint before a run** — commit the exact plan, runner, model revisions, inputs, metrics, and stop condition.
2. **Execution** — run only the bounded block from that committed source. Save the source SHA and exact command in the run receipt.
3. **Result publication commit** — the coding/model agent that executed the block must commit compact results, a human-readable review, and `review_manifest.json`. Do not leave the owner with only untracked server outputs.
4. **Review checkpoint** — separate implementation/measurement/comparison/hypothesis status. A software PASS is not a scientific GO.
5. **Next plan only after review** — commit the next bounded plan separately from the result publication.

Preferred commit subjects:

- `phase01: plan <bounded experiment>`
- `phase01: implement <runner/instrumentation>`
- `phase01: publish <result block>`
- `phase01: review <result and next decision>`

A result commit message should state what actually completed and avoid claims beyond the measured execution kind.

### Artifact policy

Commit compact CSV/JSON summaries, plots needed for review, receipts, hashes, and manifests. Large raw traces may remain on the server when necessary, but then the result commit must record their verified path, size/hash, and the compact derived evidence. Existing tracked PoC traces/results are preserved; do not rewrite them to make later comparisons look cleaner.
