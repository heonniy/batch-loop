# Phase-04R bounded result review

**Outcome:** saved BF16 failures replayed; earliest differing operations isolated in current-backend GPU tests; fresh same-input validation confirms nonzero batch-dependent live depth/work and next-token changes. Branch B is supported for the observed first divergences, with B2 recommended. The original Gate A remains FAIL. The proposed B2 contract is a draft pending explicit approval; no R5 confirmation, predictor collection or new speed measurement occurred.

| Review axis | Judgment | Evidence and limit |
| --- | --- | --- |
| Measurement | VALID for bounded operator/correctness/work diagnosis; INSUFFICIENT for speed or population generalization | Exact saved effective input hash/decision replay; 32 fresh disjoint requests × 8 positions, fixed B2/4/8 and three orderings. Debug timing is not scored. One synthetic short-prompt cohort/model; repeated orderings reuse requests. |
| Implementation | CHECKED IN SCOPE for first-op and selected row/KV/state invariants; original BF16 exact parity FAIL | Ouro equal-input `q_proj` output difference and Huginn equal-QKV/mask SDPA difference reproduced in isolated current-backend calls. Fixed-history merge, h0, duplicate and tested row-order checks pass. No demonstrated row/KV bug in these cases; all paths are not certified. |
| Simulator | TRACE-SIM ONLY for frozen B1-depth regrouping RIC | Fresh live RIC and work are measured from actual gates; frozen-work RIC is separately labeled. No wall-time inference from either. |
| Comparison | COMPARABLE for same-input live policy; unequal-work for speed | Pinned revision/threshold/KV/BF16 and B1-known token IDs within each model. Live depth/work/next tokens sometimes diverge; no equal-work speed attribution. FP32 and isolated operators are controls, not the primary treatment. |
| Hypothesis | Branch-B numerical mechanism SUPPORTED IN SCOPE; predictor hypothesis NOT TESTED | Fresh per-condition depth flips: Ouro 2–3/256, Huginn 2–7/256; actual work varies. A predeclared ±0.005 exit-margin exception misses an Ouro depth flip. No predictor, Oracle recovery or physical speed result. |

The full evidence and failed/blocked cell accounting are in `CAUSE_REVIEW.md`; `review_manifest.json` binds source/result commits, hardware, commands, provenance and artifact paths. Next action: obtain a research decision on `PROPOSED_B2_CONTRACT.md`, then preregister a new rule and run fresh confirmation if approved.
