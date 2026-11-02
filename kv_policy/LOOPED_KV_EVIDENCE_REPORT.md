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

A prompt token of a looped model owns R KV planes (one per recurrence; Huginn 32, Ouro 4); a later query at loop r reads
plane r. A standard transformer has one KV per token per layer and no such axis. v8 (results/20261103T020000Z_v8_plane_budget,
scripts/plane_budget_sweep.py; validation: identity plane map == reference exactly, all-last map == independent
DepthOracleKV 'last' replay exactly, on both checkpoints) measures what the planes are worth.

**1a. The planes are redundant after the first third of the recurrence.** Cosine of a Context token's K (V) at plane r to
its terminal plane, pooled over 40 prompts × ~400 tokens: Huginn 0.80 (0.46) at r=1, 0.92 (0.75) at r=4, 0.98 (0.93) at
r=8, 0.998 (0.993) at r=16, 1.0 from r=24; 98% of tokens have cos(K_16, K_32) > 0.99. Ouro: K 0.85 / 0.94 / 0.98 at
r=1/2/3, V 0.53 / 0.75 / 0.90 — Ouro's 4 planes differ far more from each other than Huginn's last 24 do. Settle depth of
the token's own state (within 10% of terminal): Huginn mean plane 16.8 of 31 (34% settled by plane 15), Ouro 2.9 of 3
(no token settles before its last plane).

**1b. Equal-plane-budget two-axis matrix (80 units: dev 8 + holdout 12 UIDs per dataset per model; gold ΔNLL_future /
greedy EM; budget = kept planes over Context tokens × R).** Keeping ALL tokens with 10% of their planes is lossless;
keeping 10% of the tokens with all planes is not, under any label-free prompt-time score:

| budget | arm | Huginn CR hold ΔNLL / EM | Huginn Hotpot hold | Ouro CR hold | Ouro Hotpot hold |
|---|---|---|---|---|---|
| FULL | — | 0 / 0.92 | 0 / 0.58 | 0 / 1.0 | 0 / 0.50 |
| 0.1 | plane: last-k (k=3 / 1) | −0.001 / 0.92 | −0.001 / 0.58 | −0.011 / 1.0 | −0.001 / 0.50 |
| 0.1 | plane: evenly spaced k | −0.002 / 1.0 | 0.003 / 0.58 | −0.011 / 1.0 | −0.001 / 0.50 |
| 0.1 | plane: most-attended loops | −0.002 / 0.92 | 0.007 / 0.58 | **0.163 / 0.92** | **0.657 / 0.00** |
| 0.1 | token: question-anchor loop-R attention | 2.48 / 0 | 1.20 / 0.08 | 2.76 / 0 | 1.53 / 0.25 |
| 0.1 | token: 2-token format lookahead | 1.14 / 0 | 0.37 / 0.42 | 0.78 / 0.17 | 0.38 / 0.75 |
| 0.1 | token: random | 1.48 / 0 | 1.18 / 0.08 | 1.92 / 0 | 1.29 / 0.25 |
| 0.1 | token: future-query oracle | −0.05 / 0.83 | −0.11 / 0.58 | −0.01 / 1.0 | −0.08 / 0.75 |
| 0.1 | mixed: deferred-4 tokens (√b) × last planes (√b) | 0.41 / 0.33 | −0.05 / 0.58 | −0.01 / 1.0 | −0.04 / 0.75 |
| 0.3 | plane: last-k (k=10 / 1) | 0.000 / 0.92 | −0.006 / 0.58 | −0.011 / 1.0 | −0.001 / 0.50 |
| 0.3 | token: 2-token format lookahead | 0.63 / 0.25 | 0.19 / 0.58 | 0.06 / 0.92 | 0.41 / 0.75 |
| 0.3 | token: random | 0.95 / 0.08 | 0.96 / 0.17 | 1.15 / 0 | 1.30 / 0.25 |

(dev cells agree; full table: analysis/tax_summary.csv. "Hotpot" FULL EM is low because the gold answers are free-form;
ΔNLL is the primary outcome there.)

- For **Huginn** every late-plane policy (last-k, evenly spaced, first+last, settle-adaptive, most-attended) is at the
  BF16 floor at 10% of the planes (3 of 32); for **Ouro** last-1 / even are at the floor at 1 of 4 planes. The ordering of
  arms is identical on dev and holdout.
- **Which plane matters is model-specific**: Ouro must keep its *terminal* plane — the "most-attended loops" policy,
  which keeps the planes later prompt queries read most (often early loops), costs 0.16–0.92 nats and EM; for Huginn the
  choice is irrelevant. The v5 'first' plane policy is catastrophic in both (HOR arm: KL 0.26–0.5 and growing).
- **Map semantics do not matter** (nearest kept plane vs latest kept plane ≤ r: identical to 3 decimals), because kept
  planes are late and the planes they stand in for are near-identical (1a).
- **Per-token plane compression has no heavy tail** (PLN census, dev, exhaustive: 2816–3677 tokens per cell): collapsing
  ONE token to its last plane gives median KL 8e-5 (Huginn CR), 2.3e-4 (Huginn Hotpot), 1e-4 (Ouro CR), p95 ≤ 1.3e-3; the
  top-5% of tokens carry only 11–21% of the (tiny) total — against 98–99% for single-token *eviction* (§7). No clean
  feature (position, lexical, attention received, settle depth, update norms, future loop profile) ranks it (LOPO ρ ≤ 0.03,
  all CIs cross 0). The plane axis is uniformly cheap; the token axis is sparse and expensive.

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

Runs: results/20261103T023000Z_v8_horizon (v6 B pool, 4+4 prompts/model) + results/20261103T025000Z_v8_horizon_poolA
(v6 A pool, 10+10) on the frozen v6 128-token greedy continuations; standard arm results/20261103T022500Z_std_qwen_v6horizon
+ ..._poolA. Analysis: results/20261103T023000Z_v8_horizon/analysis/HORIZON_FINDINGS.md. Outcome KL(p_full‖p_policy) at
generated step t, binned t 1–8 / 9–32 / 33–128.

| model | arm (30% of Context planes or tokens unless noted) | t 1–8 | t 33–128 | late/early (median over prompts) |
|---|---|---|---|---|
| Huginn W1 | plane last-1, prompt tokens only | 0.0003 | 0.0003 | 1.3 |
| Huginn W1 | plane last-1, prompt + every generated token converted after it finishes | 0.0004 | 0.0004 | 0.9 |
| Huginn W1 | plane first-1, prompt + generated | 0.52 | 1.13 | 1.9 |
| Huginn W1 | exit-truncated planes (what native early exit leaves), prompt + generated | 0.0012 | 0.0011 | 1.1 |
| Huginn W1 | token eviction 30% kept, future-query oracle | 0.27 | 0.26 | 0.8 |
| Huginn W1 | token eviction 30% kept, random | 0.62 | 0.48 | 0.5 |
| Ouro W1 | plane last-1, prompt only | 0.006 | 0.009 | 1.0 |
| Ouro W1 | plane last-1, prompt + generated | 0.032 | 0.098 | **3.2 (85% of prompts grow)** |
| Ouro W1 | exit-truncated, prompt + generated | 0.0007 | 0.0014 | 1.5 |
| Ouro W1 | token eviction 30% kept, future-query oracle | 0.32 | 0.29 | 0.8 |
| Qwen W1 | token eviction 30% kept, future-query oracle | 0.31 | 0.18 | 0.5 |
| Qwen W2 | token eviction 30% kept, deferred-4 | 0.17 | 1.27 | **190** |

- **Single-token damage decays in all three models** (top-5% sources: late/early 0.02–0.13 looped, 0.02–0.09 Qwen):
  once the answer span has been produced the models recover from a missing Context token.
- **Plane compression of generated tokens accumulates only in Ouro** (last-1 on prompt+generated: 0.03 → 0.10 over 128
  tokens, 85% of prompts growing; W2 0.05 → 0.09), consistent with the v5 LAST-CONVERT result on long GSM8K/MATH
  generations (Ouro-2.6B −11 / −52 points; Huginn ±0). Huginn's last plane is a faithful stand-in for every plane of
  every token, prompt or generated, over at least 128 tokens (KL ≈ 3–5e-4, flat).
- **Exit-truncated planes are free in both looped models** (≈1e-3 KL, flat): the planes a token never computed under
  native early exit are not missed by later queries at deeper loops.
- **Lookahead / deferred token policies are short-answer tricks**: on the W2 summarise/explain continuations the
  deferred-4 and 2-token-lookahead token selections lose ground as generation proceeds (Qwen 0.17 → 1.27; Huginn W2
  la2 0.02 → 0.76), because the consuming query keeps moving. This is shared by looped and standard models.

## 6. Native early exit and the natural plane budget

With native early exit (Ouro Q-exit q=0.7; Huginn fp32 cosine 1e-3, frozen on clean data in v7), prompt tokens would stop
at depth d_i and no plane ≥ d_i would exist (no-refill). The "natural" plane budget this implies and its cost:

| model | dataset | natural budget (planes kept / all) | ΔNLL natural (exit-truncated) | ΔNLL last-k at the same count | ΔNLL evenly-spaced k | EM natural / FULL |
|---|---|---|---|---|---|---|
| Huginn | CR holdout | 0.66 | +0.005 | 0.000 | −0.002 | 0.92 / 0.92 |
| Huginn | Hotpot holdout | 0.58 | −0.002 | +0.003 | +0.003 | 0.50 / 0.58 |
| Ouro | CR holdout | 0.82 | 0.000 | −0.005 | 0.000 | 1.0 / 1.0 |
| Ouro | Hotpot holdout | 0.96 | +0.002 | 0.000 | 0.000 | 0.50 / 0.50 |

- The exit-truncated cache is free (per-token PLN: mean KL 9e-5 / 3.5e-4 Huginn, 7e-5 / 2.4e-5 Ouro; only 3–50% of tokens
  have a larger KL than the last-only collapse of the same token), but it is also a **weak** budget: Huginn keeps 58–66% of
  its planes under early exit whereas 10% (last-3) is already lossless. Early exit is therefore not where the KV saving
  is; converting every token to its late planes is, and it composes with exit (an exited token already is a late-plane
  token).
- In v7 the exit depths of generated queries respond to single-token eviction (Huginn 21–33% of queries change depth, Ouro
  0.5–2%, almost always deeper), without a quality effect (§7). Exit and KV policy interact through the *plane count a
  token leaves behind*, not through quality.

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
