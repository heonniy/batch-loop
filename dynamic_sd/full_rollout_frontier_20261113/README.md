# AR/SD full-rollout frontier across Huginn / Ouro / Dory

**PLAN ONLY** — source-verified protocol and agent handoff, no new GPU execution. This study responds to the new question: when are AR/SD each faster, and does the optimal speculative length `K` change under native recurrent exits, by task and model?

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): pre-registered comparisons, baseline gates, full-rollout and post-hoc stratification, failure conditions.
- [REPRODUCTION_PROTOCOL.md](REPRODUCTION_PROTOCOL.md): LoopSpec / WaveFront / Dory verified source settings, essential mismatches.
- [AGENT_TASK.md](AGENT_TASK.md): implementation + testing order.

Comes **after** `4f6e9c50` Stage A/B CPU-derived one-round oracle, which is *not* an online execution result. The new measured results directory is absent by design until valid execution is performed. No changes to earlier receipts or algorithms in this PR.
