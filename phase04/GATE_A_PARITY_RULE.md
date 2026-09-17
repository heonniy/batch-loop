# Phase-4 Gate A: prospective BF16 parity rule

The N4×2 GPU probes through source `c0889f5` were calibration/diagnosis, not
confirmation. Their original elementwise tolerance (atol 0.05, rtol 0.02)
failed for both models, while all observed exit depths and next-token IDs
matched. Same-padding and equal-length controls did not eliminate the drift;
FP32 model diagnostics passed both models at the original tolerance. The
calibration roots and failures remain unchanged.

Before opening the planned N32×8 mixed-length anchor, fix this release rule:

- exact per-request/token exit depth and next-token ID; no force-correcting a
  live gate or overwriting generated labels;
- exact Huginn identity-keyed h0 receipts, KV slot/valid-token shapes, and
  logical request↔KV-row mapping;
- max per-request/recurrence relative L2 error <= 0.03 for hidden, logits,
  prefill logits and attention-visible K/V; max absolute gate-score difference
  <= 0.02;
- report every depth/next-token disagreement, every near-threshold score,
  original elementwise failures, absolute maxima and the first divergent
  recurrence/layer. The 3%/0.02 numbers are prospective BF16 checks based on
  the bounded N4 calibration (~2% relative-L2 maxima), not a relabeling of
  those failed tests.

If the fresh anchor misses any release condition, status is FAIL and that
model's expensive feature collection stops. A passing semantic gate does not
make the BF16 path bitwise equivalent, or turn replay timing into live quality.
The independent row-wise reference and compact runtime must use the same
checkpoint, threshold, h0 identity, token input and chosen KV/exit policy.
