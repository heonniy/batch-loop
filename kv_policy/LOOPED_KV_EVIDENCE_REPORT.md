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

Source: results/20261102T093000Z_qc_v7_pilotA/analysis/LOOP_PHENOMENOLOGY.md (40 prompts per model, every generated
gold query, head-mean attention over the prompt at each recurrence r, same weights at every r).

- **The attention pattern of one query changes across loops and converges.** Spearman between the Context-attention
  vector at loop r and at loop R: Huginn CR 0.18 (r=1) → 0.68 (r=3) → 0.95 (r=8) → 0.99 (r=16); Hotpot 0.37 → 0.92
  (r=8); Ouro CR 0.51 (r=1) → 0.68 → 0.90 (r=3). Consecutive loops agree at ρ≈0.75 early and >0.99 after r≈8
  (Huginn). A standard transformer has no such object: each layer's attention is a different map, and nothing converges.
- **Evidence is found late.** Huginn's mass on evidence tokens rises from 0.09 (r=1) to 0.30 (r≥8) on CR and 0.25 →
  0.46 on Hotpot; the top-1 attended Context token is evidence in 45% of (query, loop≤R/8) cases vs 66% late. Ouro
  Hotpot 0.38 → 0.43; Ouro CR is the exception (0.21 → 0.16: its first loop already points at the record line).
  Per query, the evidence-attention peak sits at loop ≈0.31–0.37·R for Huginn (45% of CR queries peak in the second
  eighth of the recurrence), at loop 2 for Ouro CR and at loop 4 for Ouro Hotpot.
- **Loop-1 attention is anti-informative about what the consuming query will read.** Spearman between the question
  anchor's loop-1 attention and the consuming generated query's loop-R attention: −0.19 (Huginn CR), 0.18 (Huginn
  Hotpot), 0.20 / 0.42 (Ouro); the anchor's loop-R attention agrees at 0.41 / 0.43 / 0.29 / 0.63. For the generated
  query itself, loop 1 vs loop R agree at only 0.24 (Huginn CR) and 0.40 (Huginn Hotpot); 0.71 / 0.58 for Ouro.
- Sink mass (first 4 tokens) is flat across loops (Huginn ≈0.30–0.34; Ouro ≈0.72); local mass (last 16 prompt tokens)
  falls slightly with r (0.24 → 0.20 Huginn).

Reading: an importance score computed from "the" attention does not exist in a looped model; the loop index is a
parameter of the score, and early-loop attention is a poor proxy for the reads that actually matter.

## 3. Importance signals: which loop (looped) vs which layer (standard)

Within-prompt Spearman between the attention paid to a Context token at loop r (looped) / layer l (standard) and the
future single-token eviction damage (mean KL over the gold continuation), mean over held-out prompts.

| model | query | first loop/layer | mid | late | mean over loops/layers |
|---|---|---|---|---|---|
| Huginn R=32 (holdout) | consuming query t* | r=1: 0.08 | r=8: 0.17 | r≥16: 0.21 | 0.21 |
| Huginn R=32 (holdout) | 2-token lookahead | r=1: −0.005 | r=8: 0.11 | r≥16: 0.19 | 0.19 |
| Huginn R=32 (holdout) | question anchor | r=1: 0.02 | r=8: 0.06 | r≥16: 0.08 | 0.08 |
| Ouro R=4 (holdout) | consuming query t* | r=1: 0.10 | r=2: 0.11 | r=4: 0.16 | 0.15 |
| Ouro R=4 (holdout) | 2-token lookahead | r=1: 0.07 | r=2: 0.10 | r=4: 0.16 | 0.15 |
| Qwen2.5-3B L=36 (holdout) | consuming query t* | l=1: −0.02 | l=12: 0.03 | l=30: 0.08; l=36: 0.02 | 0.09 |
| Qwen2.5-3B L=36 (holdout) | 2-token lookahead | l=1: −0.03 | l=12: 0.02 | l=30: 0.08; l=36: 0.04 | 0.08 |
| Qwen2.5-3B L=36 (holdout) | question anchor | l=1: −0.01 | l=12: 0.03 | l=24: 0.05; l=36: 0.03 | 0.05 |

(Sources: v7 results/20261102T103000Z_qc_v7_pilotB_holdout/analysis/QUERY_CONDITIONED_FINDINGS.md, loop-resolved table;
results/20261103T021500Z_std_qwen_census/analysis/STANDARD_FINDINGS.md, layer-resolved tables; CR cells shown, Hotpot
analogous.)

- In both architectures the first loop/layer carries almost no importance signal and the signal grows with depth. In the
  looped models it **plateaus** after r≈R/2 at ρ≈0.2 (the attention has converged, §2); in the standard model it peaks
  in the late-middle layers (l≈26–34, ρ≈0.09–0.15 on the full curve) and **drops at the last layers** (l=36: 0.01–0.02).
  "Use the last recurrence" is therefore a safe rule for the looped models and a wrong one for the standard model.
- Equal-budget policies agree with the ranking statistic: for Huginn the loop-1 variants of every score are far worse
  than the loop-R variants (holdout CR 10%: la2_r1 ΔNLL 2.26 vs la2_rR 1.14; def4_r1 1.37 vs 0.89; future oracle built
  from loop-1 attention EM 0.17 vs 0.83 with loop R). For Qwen the layer analogue is weaker and non-monotone: qa_early
  (first-quarter layers) is below random at 10% (ΔNLL 2.98 vs 2.16) but qa_mean / qa_late / qa_max_layer beat random by
  0.75–1.2 nats; at 30% qa_mean keeps EM 0.58, qa_late 0.33, qa_last_layer 0.

## 4. Query mismatch and deferred decisions: shared with the standard model — but worse in the looped models

Equal-budget Context-token retention on the SAME 12 held-out UIDs per dataset (gold ΔNLL_future; greedy EM; full planes):

| keep | policy (score) | Huginn CR ΔNLL / EM | Ouro CR ΔNLL / EM | Qwen CR ΔNLL / EM | Huginn Hotpot ΔNLL / EM | Ouro Hotpot | Qwen Hotpot |
|---|---|---|---|---|---|---|---|
| 0.1 | random | 1.46 / 0 | 1.70 / 0 | 2.16 / 0 | 1.22 / 0.08 | 0.96 / 0.25 | 1.15 / 0.25 |
| 0.1 | question-anchor attention (loop-R / layer-mean) | 2.48 / 0 | 2.75 / 0 | 1.58 / 0 | 1.20 / 0.08 | 1.11 / 0.25 | 1.08 / 0.17 |
| 0.1 | 2-token format lookahead | 1.14 / 0 | 0.78 / 0.17 | 0.67 / 0.33 | 0.37 / 0.42 | 0.26 / 0.58 | 0.39 / 0.42 |
| 0.1 | deferred after 4 generated tokens | 0.89 / 0.08 | 0.54 / 0.42 | 0.54 / 0.42 | −0.12 / 0.58 | −0.04 / 0.67 | 0.01 / 0.58 |
| 0.1 | future-query oracle | −0.05 / 0.83 | −0.01 / 1.0 | 0.02 / 1.0 | −0.11 / 0.58 | −0.03 / 0.67 | 0.00 / 0.58 |
| 0.3 | random | 1.06 / 0 | 1.25 / 0 | 1.50 / 0 | 0.91 / 0.11 | 0.83 / 0.42 | 0.95 / 0.17 |
| 0.3 | question-anchor attention | 1.29 / 0 | 1.35 / 0 | 0.49 / 0.58 | 0.77 / 0.17 | 1.16 / 0.33 | 0.95 / 0.17 |
| 0.3 | 2-token format lookahead | 0.63 / 0.25 | 0.06 / 0.92 | 0.32 / 0.58 | 0.19 / 0.58 | 0.15 / 0.67 | 0.20 / 0.50 |
| 0.3 | deferred after 4 generated tokens | 0.41 / 0.33 | −0.01 / 1.0 | 0.10 / 0.92 | −0.05 / 0.58 | −0.01 / 0.67 | 0.01 / 0.58 |
| 0.3 | future-query oracle | −0.03 / 0.83 | −0.00 / 1.0 | 0.01 / 1.0 | −0.04 / 0.58 | −0.01 / 0.67 | 0.02 / 0.67 |

(FULL EM: Huginn 0.92 / 0.58, Ouro 1.0 / 0.58, Qwen 1.0 / 0.58 (CR / Hotpot). Sources: results/20261102T140000Z_qc_v7_pilotB_policy
and results/20261103T021500Z_std_qwen_census, holdout rows; looped scores are the loop-R variants, Qwen scores the layer-mean.)

- **Shared**: in all three models the consuming query is a later generated token whose attention differs from every
  prompt query's; deferring the decision (or using the known "Answer:" format prefix as a lookahead query) recovers most
  of the oracle, and the future-query oracle is near-lossless even at 10%.
- **Different in degree**: prompt-time attention selection is *below random* at 10% in both looped models on CR
  (ΔNLL +1.0 vs random), while the standard model's layer-mean prompt attention is *better than random* (−0.75 nats at
  10%; EM 0.58 at 30% vs 0 for Huginn/Ouro). The diagnostic behind it: Spearman between the question anchor's attention
  and the consuming query's is 0.58 for Qwen vs 0.29–0.43 for the looped models at their last loop (and −0.19–0.42 at
  loop 1). The query mismatch is a general phenomenon; the looped models' prompt query is a worse proxy for the
  consuming query, consistent with §2 (attention is still being re-formed across the recurrence).
- Caveat: the Qwen base model is a different training corpus/architecture family; the comparison is on identical prompts
  and continuations, not a controlled ablation of "loop vs no loop".

## 5. Decode-horizon accumulation: plane compression of generated tokens vs token eviction

## 6. Native early exit and the natural plane budget

## 7. What did NOT differ / negative results

- **No per-source "recurrent sensitivity" of the later query** (v7, dev and holdout, both looped models): recomputing the
  question anchor with a Context token unreadable at one or all loops gives output changes at the BF16 floor (median KL
  1e-5–1e-4) for every attention decile, and the one-loop propagation gain E_R/E_s is an architecture constant (Huginn
  ≈2.0, Ouro ≈1.0–1.2) that adds zero ranking lift over the consuming query's attention (QUERY_CONDITIONED_REVIEW.md §2).
- **The source token's own trajectory** (settle depth, update norms) adds ≤0.025 ranking lift for token eviction in v7;
  its role for *plane* compression is tested in §1 (v8 PLN census).
- **Heavy tail of single-token damage is shared**: the top 5% of Context tokens carry 98–99% of total KL on CR in all
  three models and 79–90% (looped) / 63% (Qwen) on Hotpot; evidence tokens make up 41% (CR) / 54% (Hotpot) of that top
  group in Qwen and 34–45% / 55–64% in the looped models (v7 dev+holdout). Nothing here separates looped from standard.
- **Damage from evicting a single critical token decays along the continuation in all three models** (top-5% sources:
  late/early KL ratio 0.02–0.13 for Huginn/Ouro on W1/W2, 0.02 for Qwen; §5): the models recover from a missing token
  once the answer span has been produced. Accumulation, where it appears, comes from the plane axis (§5), not from
  token eviction.
- **Native exit depth responds to eviction but does not change quality** (v7 H4): Huginn re-evaluated exits change depth
  for 21–33% of generated queries after a single-token eviction (both directions), Ouro for 0.5–2% (almost always deeper);
  live − recorded-replay future NLL ≈ 0 in 7/8 cells.

## 8. Practical reading (not a serving claim)

## Provenance
