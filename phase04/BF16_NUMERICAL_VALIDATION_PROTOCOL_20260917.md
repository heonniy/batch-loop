# Predeclared Phase-04R numerical validation (not a replacement Gate A)

Committed before opening the fresh diagnostic cohort. This measures how often the
original BF16 live depth and next-token decisions vary with legal batch shape;
it does not train a predictor, time speed, or relabel the failed Gate A.

- Models/policies: pinned Ouro-1.4B BF16 q=0.2, Rmax4, last-exited KV; pinned
  Huginn-3.5B BF16 threshold=0.12, Rmax16, shared physical-layer KV, request-keyed
  h0 seed1 and configured test-time noise0. No checkpoint, gate, KV or precision changes.
- Hardware: original mapped GPU2/Ouro and GPU7/Huginn, exact UUID single-visible
  checks. No other Phase-4 GPU is needed for this bounded validation.
- Cohort: 32 newly worded synthetic prompts `Validation request 1000..1031`,
  repeated reasoning-task base text 1..4 times by index. Max tokenized prompt
  length64. Hash effective tokenized inputs and reject overlap with the old
  32-request Gate-A cohort. Exactly eight fixed decode positions, no EOS.
- Reference: independent B1 live generation per request. All batched runs use
  its known input token IDs at each boundary but make their own live exit and
  next-token decisions. Report generated-token disagreement; do not feed a
  batched divergent next token into the next boundary in this comparison.
- Batch conditions: B2, B4, B8; FIFO and two fixed permutations from Python
  `random.Random(17)` and `random.Random(29)`. Every request appears exactly
  once per boundary and a started microbatch only shrinks. B1 is the control.
- Denominator: 32 × 8 = 256 request-token positions per model and per batched
  condition. The three orderings reuse these requests; they are not independent
  population samples. Do not call a selected counterexample-grid fraction a
  validation flip rate.
- Near-exit margin epsilon is fixed at absolute 0.005 for both models.
  Conditional bins, based on the B1 minimum absolute margin across its live
  recurrence trajectory, are [0,0.001), [0.001,0.005), [0.005,0.02), [0.02,∞).
- Report exact depth and next-token flip counts per B/ordering, conditional
  depth-flip counts by margin bin, reference and actual recurrent work, actual
  live RIC, and separately the frozen-B1-depth regrouping RIC. These are
  correctness/work measurements, never wall-time speedups.

Only after this result is reviewed may a Branch-B B1/B2 semantic contract be
chosen. Any replacement Gate-A rule must be a separate prospective commit and
must use a fresh confirmation cohort, not these validation prompts.
