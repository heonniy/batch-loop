# Ouro KV Sharing Collapse — Reproduction and Training-free Diagnosis

Status: **plan only**. Do not claim new GPU measurements. Branch based on E0–E2 results at
`kv_policy/results/20261026T080106Z_huginn_ouro_full_last_convert`.
Preserve raw receipts; do not overwrite previous reports.
Hardware: authorized H200 x8; HF reference only, no vllm-rlt.

## Why this experiment now?

Our FULL → LAST-CONVERT results differ sharply:
- Huginn R32: GSM8K 41.5 → 44.5%; MATH500 15.5 → 16.5%.
- Ouro-1.4B R4: GSM8K 83.5 → 0.0%; MATH500 71.0 → 1.5%.
- Ouro-2.6B R4: GSM8K 88.0 → 77.0%; MATH500 56.0 → 3.5%.
These are OUR convert semantics, **not validated as the original Ouro
paper's Last policy**; that paper reports nearly lossless sharing.
B1/FP32 reruns retained most of the observed discrepancy.
E2 found highly concentrated single-call sensitivity at recurrence 1,
physical attention layer 0: final KL 1.29 (Ouro-1.4B), 0.506 (Ouro-2.6B);
other layers/steps were much smaller.

## Questions / falsifiable hypotheses

H1: The first-recurrence, first-physical-attention-layer KV mismatch
accounts for most of Ouro LAST-CONVERT's rollout failure.
H2: Preserving that KV bank (or a minimal set selected by calibration)
restores quality at substantially less than FULL retained KV.
H3: Part of the apparent discrepancy with the original Ouro paper is
different sharing/read/write/prefill semantics and/or official evaluation
protocol, not a direct consequence of ACT training.
Do not assert ACT causes failure from these observational checkpoints.

## Step 0 — exact semantic audit (must finish before interpretation)

- Read `AGENTS.md`, E0/E1/E2 reports, official Ouro implementation and
  original paper's cache-sharing description/appendix.
- Write a small executable toy trace of prompt A,B and generated C,D showing
  exactly which (physical layer, recurrence, historical token) K/V each
  attention query reads, and when writes occur.
- Check FULL and LAST-CONVERT are implemented as defined:
  FULL prefill in both; LAST-CONVERT retains terminal prompt KV and
  terminal generated KV; current token's own per-recurrence K/V stays correct.
  Check RoPE, positions, mask, dtype, storage overwrite, and prompts.
- Compare official Ouro default inference vs reference FULL with FP32
  logit parity and short greedy generations. Do NOT label official
  paper's LAST equivalent absent supporting code/evidence.
- Trace alternate semantics **only for diagnosis**, never conflate their
  scores with the primary FULL/LAST-CONVERT experiment.
- Confirm prompt token IDs, checkpoint revisions, stopping criteria,
  parser, and full-vs-shared scores on a small fixed B1 FP32 subset.
- Record exact limitations if original paper's Last semantics remain unknown.

Artifacts: `analysis/semantics_audit.md`, `analysis/trace_reads_writes.csv`,
`analysis/native_full_parity.csv`.

## Step 1 — localized protection E3 pilot (priority)

With LAST-CONVERT as base, retain the terminal KV bank for all layers,
PLUS one depth-r bank for the selected physical layer(s).
Protect recurrence 1 / physical attention layer 0 as primary hypothesis.
Use correct depth-1 K/V for *all historical tokens* when the current query
is recurrence 1, layer 0; keep terminal history everywhere else.
Prompt prefill remains identical FULL. For generated tokens, capture and
retain the protected early bank causally when created; do not reconstruct
early KV from terminal KV or leak future information.

Compare:
1. FULL (upper reference)
2. LAST-CONVERT (negative baseline)
3. Protect (r=1, layer=0) primary
4. Protect (r=1, layer=1) negative control
5. Protect all layers at r=1 (positive diagnostic)
6. Protect layer=0 at r={1,2} (combination)
7. Random bank(s), same actual byte budget, fixed >=5 seeds
8. Calibration-selected small bank set, selected WITHOUT test outcomes.

Begin with 32 GSM8K + 32 MATH500 item pilot per Ouro checkpoint.
Audit greedy actual generation, NLL, repetition/parse failure, EOS/length,
first divergence and matched-item outcomes. Gate continuation on
pilot correctness and functional mixed-cache unit tests.
Then perform complete frozen GSM8K-200 and MATH500-200 on both Ouro models;
report per-policy paired bootstrap CIs and raw individual outcomes.
Re-evaluate selected sets end-to-end (interactions are not additive).
Keep Huginn R32 as negative-control model, using a small representative set
unless pilot demands otherwise.

## Step 2 — isolate recovery from persistent-context mismatch

From a frozen FULL prefix, compare:
A. one attention-call swap (existing E2 reference);
B. persistent swap in the first recurrence/layer only;
C. full LAST-CONVERT;
D. LAST-CONVERT with first layer/recurrence protected.
Record final hidden-error, logit KL, top-1 agreement, and KL by generated
position; distinguish fixed-prefix teacher forcing from autoregressive
rollout. Repeat selected cases in B1 FP32 to exclude BF16 noise.
Check whether protection prevents a shifted recurrent endpoint rather
than simply reducing one immediate attention error.

## Step 3 — retained KV memory and performance feasibility

Report layer-by-layer and per-token retained bytes of FULL, LAST and
protected policies using the actual bank layout. Distinguish memory after
FULL prefill conversion from **peak during prefill** (Convert does not
remove FULL prefill transient memory), and reference-engine broadcast
allocations from compact theoretical or materialized KV.
Measure correctness first; run bounded compact-cache memory validation
only after confirming token-for-token equivalence.
Do not claim serving speedup without dedicated serving benchmarks.

## Decision rules

- GO: Protection recovers most of the lost NLL/Accuracy on held-out
  items at a clearly lower recurrent KV cost than FULL, and passes
  unit-test parity and independent B1/FP32 controls.
- CONDITIONAL: Recovery occurs only in one checkpoint/dataset, or requires
  several banks; report scope and interactions.
- NO-GO: Targeted protection fails, needs near-FULL memory, or the collapse
  disappears when official inference parity/semantics are corrected.
- If original paper's sharing semantics cannot be independently verified,
  explicitly state unresolved discrepancy rather than attributing it to ACT.

## Reproducibility / deliverables

- Implement under `kv_policy/`, isolated from prior baseline.
- Eight independent H200 GPU workers with atomic claims and resumable jobs.
- Save source SHA, checkpoint/dataset revisions, prompt hashes, seeds,
  precision, actual depth/cache semantics, worker GPU UUID and stop rules.
- Commit tests, per-case results, summary markdown and failure receipts.
- Cite and contrast BeNice! first-layer exception, Ouro original Table 14,
  CDB and previous internal E0/E2 runs. This study identifies execution
  failure structure; a *causal claim about ACT vs stochastic training*
  would require matched-architecture controlled training later.

Execution sequence: semantics audit → 64-item E3 pilot per Ouro checkpoint →
held-out/full 200+200 reruns if gate passes → memory assessment.
