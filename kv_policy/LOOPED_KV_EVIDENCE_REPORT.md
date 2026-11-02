# Does a looped transformer need a different KV-cache policy? — evidence report (v8, 2026-11-03)

Branch research/query-conditioned-kv-recurrence-exit-20261102. Models: Huginn-0125 (R=32 recurrences, 4 shared core
layers + prelude/coda), Ouro-2.6B (R=4, all 48 layers recurrent), control Qwen2.5-3B (36 layers, non-looped). All
measurements are B=1, BF16 forward, full-sized reference KV with read redirection/masks (no HBM or latency claim).
Prompts: frozen Controlled-Retrieval (16 facts, 1 target) and HotpotQA-Distractor excerpts (seed 20261031; dev = 8 UIDs,
holdout = 12 UIDs per dataset), gold "Answer: <x>\n" continuation; plus v6 W1/W2 prompts with the model's own 128-token
greedy continuation for decode-horizon questions. Journal: QUERY_CONDITIONED_JOURNAL.md (E0–E8). Prior results that this
report builds on: v7 QUERY_CONDITIONED_REVIEW.md, v6 V6_PILOT_ASSESSMENT.md, v5 LAST-CONVERT (results/20261026T080106Z_*).

(Sections are filled from the run analyses listed at the end; numbers are quoted from those files.)

## 0. Summary of the evidence

(filled at close-out)

## 1. The plane axis exists only in looped models — and it is where the cheap budget is

## 2. Loop-wise attention is not one attention: the recurrence re-reads the prompt differently at each loop

## 3. Importance signals: which loop (looped) vs which layer (standard)

## 4. Query mismatch and deferred decisions: shared with the standard model (not loop-specific)

## 5. Decode-horizon accumulation: plane compression of generated tokens vs token eviction

## 6. Native early exit and the natural plane budget

## 7. What did NOT differ / negative results

## 8. Practical reading (not a serving claim)

## Provenance
