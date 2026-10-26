# Huginn vs Ouro: Full vs Last-Convert KV Sharing

Status: **experiment plan only**. No new GPU experiments or code changes are claimed by this commit.
Repository: heonniy/batch-loop. This branch retains earlier Huginn R32 and Ouro experiments.

## Central question
Why does the pretrained Huginn (stochastic-depth sampling) tolerate cross-depth
KV substitution while the pretrained Ouro (ACT-weighted depth supervision)
appears sensitive under our Last-Convert implementation? This is an
**observational cross-model comparison**; it cannot causally isolate training.

## Fixed evaluation semantics (REQUIRED)

Compare exactly **FULL vs LAST-CONVERT** as the two primary policies:
- FULL: during prefill and decode, current recurrence r reads the past KV
  from the same recurrence r (per physical recurrent layer).
- LAST-CONVERT: execute exactly the same parallel, loop-major FULL prefill;
  at prefill completion, retain only each past prompt token's terminal
  recurrence R KV (per physical recurrent layer). Decode all recurrences
  using past terminal KV. For newly generated tokens, compute current
  recurrence's own K/V normally, retain terminal K/V for later tokens.
- For both, keep prelude/coda KV unchanged. Disable dynamic early exit and
  sampling; greedy generation. Use the exact same tokens, initial latent
  random state, tokenizer, model revision, prompts, score and length caps.

These policies isolate the impact of retained recurrent-depth KV during
decode. **No LAST-STRICT, LAST-DECODE-ONLY or AVERAGE in the main sweep.**
Old alternative semantics can be mentioned only in reproduction caveats.
Note: training uses depth-aligned causal attention, not a persistent
inference-time KV cache; no claim that pretraining explicitly used Last KV.

Models:
- Huginn-0125 pinned checkpoint, FIXED R=32 (trained with depth sampling)
- Ouro-1.4B and Ouro-2.6B pinned checkpoints, FIXED R=4
Use HF reference engine only, BF16 + bounded FP32 parity controls,
authorized H200 x8 independent workers, not vllm-rlt.
Freeze GSM8K-200 and MATH500-200 item identities from prior runs.
Keep accuracy, NLL, parse failures, generation cap hit rate, and per-item
outcomes; use same evaluator and record official protocol deviations.
Main cap: GSM8K=512 and MATH500=2048, contingent on verified OOM-safe batching.
Use smaller per-model per-dataset batches as needed; report numerical noise.
Ouro original paper's Last near-parity conflicts with our Convert result;
do not attribute that disagreement to ACT unless official policy semantics
and parity are established.

## E0 — reproduce Full vs Last-Convert (first priority)

1. Verify exact model revision, depth, Full HF/native numerical parity,
   prompt IDs, initial states, causal masks, and stop criteria.
2. Run paired FULL vs LAST-CONVERT on all frozen items for each model.
3. Record full accuracy, strict/parser variants, token-weighted gold NLL,
   per-problem NLL, generation cap, first greedy divergence, paired
   Full-only/Last-only correct and bootstrap CIs across problems.
4. Report retained and peak cache footprint separately; peak during FULL
   prefill is not reduced by Convert.
5. Audit discrepant failures and repeat noisy cases in FP32/small batches.
6. Gate: if Official Ouro sharing semantics remain ambiguous, label
   observed degradation as 'our LAST-CONVERT policy' specifically.

Artifacts: results/<run>/analysis/e0_quality.md, e0_pairs.csv,
  run_manifest.json, raw receipts and completeness/failed-job audit.

## E1 — training-depth signal characterization (second priority)

**Same FULL policy to collect representations**, so characterization
does not confound different decode histories.

A. Depth-wise prediction quality: collect loop-specific logits and
gold next-token loss at each recurrent depth r. Since intermediate
representations may require a model-specific head/coda path,
ensure the SAME mapping used by each model's training objective
or explicitly mark probes as non-native. Compare model-normalized
loss improvement across relative recurrent depth.

B. Ouro ACT: capture per-position halt logits and the *normalized
exit distribution* p_{i,r} derived using the official ACT algorithm.
Separate raw halting scores, selection probabilities and actual
inference exit depth. Analyze mean/median exit depth, entropy,
per-depth probability mass, per-token variability, relation to
token difficulty and loop-specific NLL. Confirm both checkpoint
families return comparable/meaningful exit data. Exit probabilities
are not stored gradients or direct evidence of past training exposure.

C. Cross-depth KV geometry: per physical layer/head/token compute
normalized K and V distances and cosine similarities for r,s.
Measure separately for prompt and continuation under FULL.
Report Huginn R32 and Ouro R4 individually; do not compare raw
absolute norms across architectures.

D. Association analysis: in Ouro, relate p_{i,r}, entropy and
depth-level NLL to *the effect on future predictions of substituting
the historical KV of token i*. Keep the source-token vs prediction-
position indexing unambiguous. Use stratified paired correlations,
prompt-level bootstrap CIs and negative controls. Correlation is not
proof of ACT causality.

Artifacts: e1_depth_nll.csv, e1_exit_distribution.csv,
 e1_kv_geometry.csv, e1_training_signal_report.md.

## E2 — failure localization (conditional)

If E0 replicates the expected difference and E1 signals are valid:
- Fixed-Q attention intervention Q_t^(r) with historical K/V
  substituted by K/V^(s), causal mask and self KV kept constant.
- For each model, record immediate attention output change and
  continuation from the intervention to final logits; compare
  relative recovery gains normalized for different R. Supplement
  Huginn native-depth R32 with a separately labeled R4/8/16 sweep.
- Measure layer x recurrence sensitivity and single-call
  perturbation decay, matched-amplitude random-control perturbations.
- Distinguish recovery under unchanged KV history from a changed
  fixed point under persistent sharing.

Artifacts: e2_cross_depth_attention.csv, e2_error_propagation.csv.

## E3 — selective protection pilot (conditional)

If E2 shows localized failure in at least one model, retain the
minimum relevant layer x recurrence KV banks while defaulting to
terminal sharing elsewhere. Calibrate on disjoint prompts,
then re-evaluate full rollout Accuracy/NLL on held-out items.
Contrast LAST, FULL, simple first-loop protection, random banks,
and calibrated protection under *equal realized HBM budgets*.
Avoid claiming novelty for first-loop protection alone (BeNice!).
If selective protection requires near-FULL bytes or fails to
generalize, report NO-GO. Distinguish prefill transient peak
from retained decode KV.

## General analysis safeguards

- Training strategy, architecture, depth R, dataset and number of
  model parameters all covary between Huginn and Ouro. E0–E2
  support **association and mechanism hypotheses**, not evidence
  that ACT causes failure or stochastic training causes success.
- Compare FULL-to-LAST deltas within each model; do not interpret
  cross-model raw accuracy or KL differences as training effects.
- Explicitly cite / acknowledge Part II, BeNice!, Ouro original
  near-lossless sharing claim, CDB, and LLA.
- Stop immediately for unexplained parity mismatch and preserve
  failure logs. Stage E0/E1 are mandatory; E2/E3 have explicit
  gate conditions. H200 work must be performed by a connected
  execution agent, not assumed complete from this plan commit.
