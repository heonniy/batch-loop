# Phase-4 bounded result review

The requested Phase-4 GPU execution reached its mandatory Gate A and **stopped there**. Ouro BF16 on physical GPU2 and Huginn BF16 on physical GPU7 each failed the same-policy B1-versus-compact live-exit N32/B8/T8 anchor. GPU3 and GPU5 supplied N4 BF16 numerical replication controls. No predictor data, candidate tournament, physical strategy matrix, or performance recovery result was generated. The FP32 N32 controls pass and narrow the diagnosis, but are not a new approved primary result.

The complete machine-readable summary is `runtime_validation.json`; raw policy, state-identity and row-level receipts are preserved in sibling timestamped roots. `IMPLEMENTATION_REVIEW.md` gives the divergence details, audit finding disposition, and limits.

## Five-axis judgment

| Axis | Status | Evidence and boundary |
| --- | --- | --- |
| Measurement | VALID for bounded correctness; INSUFFICIENT_DATA for speed | Exactly 32 effective-tokenized distinct synthetic requests × 8 fixed token positions, B8, live depth labels, reference-generated common token inputs. Debug/profile wall clock includes model load and per-step diagnostics; no warmup/repeats or scored FIFO/Oracle matrix. |
| Implementation | FAIL live-adaptive BF16 parity | Ouro 2 depth mismatches and 617/615 recurrent calls; Huginn 7 depth plus 3 next-token mismatches and 2108/2105 calls. Structural KV identities and Huginn h0 hashes match in the anchor. FP32 control passes; precise BF16 first-layer cause remains unproven. |
| Simulator | NOT APPLICABLE | No new trace regrouping or RIC simulation in this bounded result. Do not transfer a trace-only claim into a physical speedup. |
| Comparison | COMPARABLE for same-input correctness only | Same pinned model, threshold, prompt/token IDs and chosen KV policy within each reference/compact pair. Actual live work diverges, so no equal-work physical timing comparison is possible. FP32 is a diagnostic change of precision, not the BF16 treatment. |
| Hypothesis | NOT TESTED for prediction/recovery | Gate A blocks Block B/C. No fresh confirmation winner, Oracle RIC recovery, or wall-time recovery. |

The old forced-depth systems results retain their original limited scope. No new conclusion about whether a causal prior/hidden predictor works follows from this gate failure. Next action is a bounded numerical-divergence diagnosis and a fresh predeclared Gate A rerun after a justified fix; do not start expensive recollection or clean benchmark timing while either lane fails.
