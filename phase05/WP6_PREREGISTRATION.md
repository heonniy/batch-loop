# Phase 05 WP6 — extensions of the WP5 ranker (preregistration)

Committed **before** any WP6 collection or ranker training. Authorization: user 2026-10-18 ("너가 생각하는 다음 후보들을 다 실험해봐").

## Scope decision: optimized upstream engine (not run)

A read-only audit of github.com/kschwethelm/continuous-depth-batching found three blockers:
- Its no-refill mode forms **one cohort per wave** (resident cap = launch width, `continuous_api.py:820-828`, `scheduler_base.py:203-206`), so decode-boundary regrouping cannot act without a scheduler redesign (≈1–3 days).
- Its Ouro checkpoint is `KristianS7/Ouro-1.4B`, not our ByteDance revision.
- It requires torch 2.11 / transformers 5.9 / FlashAttention-3 in a separate environment.

That study needs its own decision and is **not** part of WP6.

## Shared contract

WP1 contract (BF16, Rmax, KV, h0, teacher-forced B1 tokens, strict no-refill, lockstep), GPU serving without host syncs (`wp4_gpu_serving.gpu_run`), per-run hard checks, B ∈ {8,16,32,64}, 2 cohorts × 256 per lane, 3 repeats, one warmup, rotated and shuffled order, all runs retained. GPUs 0–3 with model-agnostic roles. Fresh request splits (`phase05/wp2_headroom.py:SPLITS`): `wp6_test` [3328,3840), `wp6_q_test` [3840,4352), `wp6_eos_test` [4352,4864).

**Per-ranker GO rule (unchanged):** a ranker policy wins a cell if its headroom vs FIFO is ≥ 5% and all 3 repeats are faster than all 3 FIFO repeats. GO for a model/study if it wins ≥ 2 of 4 B values in each of the 4 dataset×cohort blocks.

## 6a — cross-dataset transfer (Ouro and Huginn, `wp6_test`)

Policies: fifo, history_last, `ranker_same` (WP5 R of this dataset), `ranker_cross` (WP5 R trained on the other dataset; frozen, no refit), `ranker_notraj` (6b), oracle_b1.
- **Transfer GO** = per-ranker rule for `ranker_cross`.
- Transfer retention = mean over cells of headroom(cross)/headroom(same); reported, not deciding.
- 6a also gives a fresh replication of WP5's `ranker_same` result.

## 6b — ranker without trajectory features (same lanes)

R retrained on the WP5 train/validation arrays without the 17 previous-trajectory scalars (`wp5_ranker.py --drop-trajectory`; same grid and selection rule, validation only). Served without the per-recurrence hook.
- **Huginn no-traj GO** = per-ranker rule for `ranker_notraj` on Huginn.
- **Trajectory-cost hypothesis supported** if `ranker_notraj` is faster than `ranker_same` (median) in ≥ 75% of Huginn cells.
- Ouro is reported the same way.

## 6c — another exit threshold (Ouro q=0.4, `wp6_q_test`)

Train/validation re-collected at q=0.4 with full-dim states (same `wp3_train`/`wp3_validation` prompts; new labels). A new R is trained and selected on validation at q=0.4.

Policies: fifo, history_last, `ranker_q04` (new), `ranker_q02` (WP5 R, frozen, transferred), oracle_b1.
- **GO** = per-ranker rule for `ranker_q04`.
- Threshold transfer is reported for `ranker_q02`.

## 6d — EOS-terminated variable length (Ouro, `wp6_eos_test`)

B1 collection stops a request after it generates EOS (ids `<|endoftext|>`, `<|im_end|>`), T_max = 64. In the physical runs a request leaves the ready pool after its EOS step (causal under teacher forcing). Remaining requests form cohorts of B (the last cohort per step may be partial). The finished rows' cache positions become masked placeholders (`kv.py:retire_rows`).

Policies: fifo, history_last, `ranker_same` (WP5 R, frozen), oracle_b1.
- **GO** = per-ranker rule for `ranker_same`.
- Work, RIC and flips are counted over active request-tokens only.

## Gate 0 (untimed, before scoring a lane)

On `wp3_validation` cohort 0, B=8, live GPU-served scores are compared with offline scores on the B1 arrays. **PASS** if mean within-step Spearman ≥ 0.95 and there are no hard-check errors. Audited:
- every new ranker (6b ×4 lanes, 6c ×2);
- the variable-length path, with random lengths imposed, for each 6d lane.

A lane with a failed gate is not scored.

## Engineering smoke (declared)

Unscored smoke (64 requests, T=4) on GPU0 covered:
- EOS collection: 20/64 ended early;
- q=0.4 collection: mean depth 3.17 vs 2.73;
- no-trajectory training;
- random-length Gate-0 audit: PASS, Spearman 0.9997;
- an EOS physical run with five policies: 0 hard-check errors.
