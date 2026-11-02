# Query-conditioned recurrent KV sensitivity × dynamic exit (Huginn / Ouro)

Date: 2026-11-02. Study: **v7 exploratory, mechanism-first**.
Branch: research/query-conditioned-kv-recurrence-exit-20261102
Parent: research/loopzip-persistent-trajectory-characterization-20261031 (v6).
Target models: pinned Huginn-0125 (R=32) and Ouro-2.6B (R=4).
Status: **protocol and agent handoff; not a GPU result**. Do not alter/overwrite previous runs.

## 0. The actual research question (not a desired conclusion)

**RQ:** Can the sensitivity of a *later Instruction/Question Query's recurrent computation* to an earlier Context KV explain the future cost of evicting that KV beyond Attention, position, lexical/value/output controls and the *source token's own* recurrent trajectory?

Our goal is **not** to establish that a source with a large normal-prefill update is important; v6 did not support that for Huginn. Instead, ask how a later query changes *conditional on which earlier source is inaccessible*.

Three separate empirical assertions must not be conflated:
1. Different Context KV positions cause different downstream damage (generic KV importance).
2. The same initial/attention-output perturbation at a query has different propagation through repeated core computation (recurrent mechanism).
3. Such query-conditioned information predicts **future post-prefill eviction damage** and improves selection on held-out examples (usefulness; not yet observed).

Dynamic exit adds a fourth **independent** question: does a post-prefill eviction change the number of recurrences used by later generated queries, and does allowing a live depth response compensate for or worsen the output loss?

A positive result in #1 alone is **not** evidence for #2 or #3. A changed exit depth alone is **not** a successful KV policy.

## 1. Existing evidence to account for (do not relabel as new)

Read v6: kv_policy/LOOPZIP_PERSISTENT_RECURRENCE_TRAJECTORY_PLAN.md and
results/20261031T070815Z_loopzip_v6_trajectory_pilot/analysis/V6_PILOT_ASSESSMENT.md,
STAGEA_STAGEC_FINDINGS.md. Also inspect
kvp/loopzip_persistent_eviction.py, kvp/role_sensitivity.py,
kvp/role_span_oracle.py, kvp/depth_eviction_probe.py, kvp/native_exit.py,
and kvp/loopzip_cached_eval.py.

Observed pilot: source-harm heavy tail; source-only recurrent features contributed
Huginn Spearman +0.011 (CI crosses zero), Ouro +0.071 (weak pilot), after
attention/position/lexical controls. Existing one-loop probes generally yielded
tiny KL (~1e-5 median) near BF16 floor; Ouro R4 is especially late-loop sensitive.
Huginn's one-loop hidden error growth is not automatically quality damage.
The v6 pilot also included one-loop controls on generated queries, so **do not
market a rerun of the same mask as a new experiment**. This follow-up changes
the *conditioning query*, comparative question, and prospective predictor tests.

The root AGENTS.md is currently focused on an unrelated Phase-04 scheduler
and disallows unrequested GPU runs in that scope. This is an additive kv_policy
experiment: no Phase-04 modifications and no GPU work performed by plan authoring.
Actual H200 execution must respect the host's GPU allocation/process rules.

## 2. Exact terminology and causal boundaries

Let i denote an earlier aligned **Context source** KV token/window; q a later
prompt **Instruction/Question query** position; r a recurrent loop; t a
generated-token query index; P the number of prompt tokens; R in {32,4}.

- **Source trajectory X_i(r):** the clean source's own h_i(r) update/settling
  statistics. A baseline feature; **not** our proposed mechanism.
- **Clean Query trajectory H_q(r):** the later query's h_q(r) on untouched FULL
  history. It measures normal query computation but does not establish which
  source caused an update.
- **Query-conditioned intervention S_(i|q,s):** independently replay query q
  with source i masked in recurrent Attention **at selected loop s only**, and
  measure its h_q(r>=s) relative to the unchanged FULL replay. Also test
  source i masked at *all* loops of q. The source's own prefill remains clean.
  All such query replays are **offline intervention diagnostics**, not one-pass
  serving features. No alternate prompt history is allowed to contaminate the
  primary post-prefill eviction outcome.
- **Future outcome Y_i:** after an entirely untouched FULL prompt prefill,
  mask the same i from historical KV reads only at generated columns >=P,
  use the SAME frozen continuation and score changes in future NLL, KL and
  later query states. The first generated token predicted by prefill cannot
  change: exclude its loss from post-prefill utility.
- **Live exit T_i(t):** depth taken by generated query t under actual native
  exit with source i masked after FULL prefill.
- **Recorded replay:** force evicted and FULL branches to use the *same*
  per-token depths recorded from FULL-live. Replay is a control for exit
  scheduling, not a claim that live-minus-replay is a pure mediation effect.

Query anchor selection: preregister (a) last aligned Question token when
present, else last aligned Instruction token; (b) last prompt token, which may
be an assistant delimiter/Response marker, as an **explicit separate ablation**.
Do not describe a format marker as an Instruction token. Optional later
queries or averages over aligned Instruction tokens must be labeled independently.
**Future generated queries are only outcomes/diagnostic probes**, never
features available at prompt-completion KV selection.

## 3. Predefined competing hypotheses (null outcomes are welcome)

- H1 (Attention mismatch): Context sources with matched clean q->i Attention
  and position can have different evicted quality damage.
- H2 (query-recurrent mechanism): after controlling the one-loop
  *immediate Attention Output change* (including Value/softmax), certain
  sources have different downstream query error propagation or recovery.
  Persistent all-loop masks alone cannot establish H2 because new damage is
  re-injected each loop.
- H3 (prediction): intervened **query-conditioned sensitivity** ranks
  future post-prefill Y_i better than source-only trajectory + strong
  conventional controls; a further cheap clean-prefill surrogate is
  possible **only if H3 survives held-out prompts**.
- H4 (exit interaction): source eviction alters live native exit depths for
  later queries, beyond repeat/no-mask numerical variation; changed depths
  interact with quality relative to FIXED and recorded-replay controls.
  Direction is not assumed (depth may rise, fall, or remain equal).
- Architecture: do not assume Huginn and Ouro have the same behavior.
  Normalize loop index r/R and compare WITHIN model before pooled reporting.
  Ouro last-loop dominance is an existing prior to beat, not novelty.

## 4. Experimental workflow — required gates, flexible implementation

### Gate A — immutable protocol and parity

Primary datasets: **Controlled Retrieval** (1 correct target among distractors;
evidence annotations for *analysis only*) plus **HotpotQA-Distractor** with
supporting facts (custom excerpt, never call it official full HotpotQA).
Use existing pinned dataset loaders, exact tokenizer-offset roles, paired
question UIDs, no truncation. The older v6 W1/W2 corpus is a useful auxiliary
generalization study and code source, not a replacement for retrieval tasks.

First verify FULL reference/independent replay for BOTH real checkpoints;
the same Huginn seeded initial noise, full-depth token-major prefill,
FULL loop-bank KV, fixed BF16 execution backend and no-refill semantics.
CPU tests before checkpoint smoke. Repeated unmasked runs establish
BF16 KL/NLL/exit-depth floor. Save revisions, code SHA, tokenizer,
dataset checksums, Prompt UID, target IDs, exact commands and GPU ID.

Model config: Huginn pinned R32; Ouro pinned R4; fixed primary first.
Use native Q-exit for Ouro (threshold candidates chosen on clean only)
and cosine evaluator for Huginn (record model-dtype *and* separately
FP32-cosine diagnostic arithmetic; the latter is NOT official BF16).
No claim that manually fixed extra loops imply live exit.

### Gate B — Query-conditioned sensitivity (mechanism, primary novelty probe)

Suggested 5-request/model smoke -> 20 questions/model/dataset pilot.
First use **single source tokens** and at most 8–12 sampled Context
positions per request, label-blind balanced by clean Attention and position;
show sampling denominators. If one-token signals are dominated by BF16
floor, the agent MAY start a new named 4-/8-token contiguous-window cohort,
with **the same number of masked KV tokens in matched arms**. Never
quietly replace or combine budgets.

For source candidate i and anchor q:
1. Obtain FULL q state h_q(0..R) and clean q->i attention at every
   recurrent layer/loop/head if feasible (not source-received aggregate).
2. Recompute q from identical clean *earlier* history with source i
   masked at exactly loop s (ONE-LOOP). Capture h_q(s..R), Coda output
   distribution (Huginn final readout only), and immediate local Attention
   Output shift with q state held fixed. Use per-layer read masks;
   no changing source KV writes or unrelated source positions.
3. Compare with all-loop-in-q persistent mask and plain-only vs rec-only
   masks (Huginn). Repeat at different s; Ouro s=1,2,3,4;
   Huginn a priori s=1,2,4,8,16,24,32. Expand to all 32 only when useful.
4. Measure normalized h error at s and R, vector *direction*,
   output KL/NLL where meaningful, and error gain E(R)/(E(s)+eps),
   but NEVER interpret unstable ratios near numerical floor.
5. Make attention-matched **AND local-output-perturbation-matched**
   within-prompt pairs. Matching should include position/role and equal
   source-window width; report matched/unmatched sample counts. Never
   pick high-gap pairs using future Y or evidence labels for the MAIN arm.

A useful query sensitivity diagnostic may be a summary of the absolute
post-intervention final-state/Coda changes across preregistered s and q
anchors; preserve full r curves rather than defining a single magic scalar.
Source trajectory and clean query trajectory are separate baselines.

### Gate C — Future persistent eviction label and controlled Depth

Starting from the untouched prompt for EACH candidate, compare:
- FULL reference; accumulated source-i eviction across all generated
  queries/layers (primary real no-refill read-mask semantics);
- clean-history isolated generated-query probe (separate counterfactual);
- rec-only vs all-layers if Huginn plain attention explains damage;
- fixed R decode on SAME frozen clean continuation, future target tokens
  only (first target unchanged), with gold NLL and clean-reference KL
  as **different outcomes**. Greedy QA correctness as an additional
  grounded quality check on retrieval tasks.

For the query-conditioned feature, no future generated-query activation,
masked future NLL, evidence label or gold answer is allowed as feature;
those are **outcomes or oracle ablations**. Recomputing q under masks
is a privileged offline diagnostic and must never be advertised as cheap.

Compare within-prompt rankings/source pairs using:
A) mean/last/max/head-/layer-aware clean q->i attention, age/position,
   role and token lexical features;
B) A + source's own clean normal trajectory;
C) A + Value norm / local attention-output perturbation;
D) A+C + query-conditioned intervention features (expensive oracle);
E) optional one-pass clean query-attention × clean query-update surrogate.
Evaluate out-of-sample by **question UID split** (same question across
models/budgets/decodes stays together) and prompt-grouped bootstrap;
report paired lift/CIs, AUROC of high-harm class, Spearman/NDCG,
and positive signed-NLL-damage vs KL separately.

Only call D a recurrent mechanism if it survives the strong local-output
baseline AND the source/loop negative controls. If a cheap E cannot
predict D/Y, an interesting oracle diagnostic is NOT yet a practical method.

### Gate D — Dynamic Depth × KV eviction (explicitly requested)

Use SAME source sets and frozen continuation as C. For each model and
clean-only predeclared exit threshold:
1. FULL-LIVE: normal full prefill, then actual native early exits on
   generated queries; record per-query exit depths and total recurrent work.
2. FULL-REPLAY: exact same recorded FULL-LIVE depths. Assert per-target
   replay-vs-live NLL parity and prompt-depth policy.
3. EVICT-REPLAY: same FULL-LIVE depth trace forced while source i is
   masked only after clean prefill. This isolates a same-depth
   counterfactual, though historical generated KV still changes.
4. EVICT-LIVE: eviction with genuinely re-evaluated exit gate at each
   generated query; do not merely relabel recorded depths.
5. EVICT-FIXED / FULL-FIXED at R for reference.

Report per-query DeltaDepth_i(t)=depth_evict_live(t)-depth_full_live(t),
fractions increased/decreased/unchanged, paired change in sum(depth),
future NLL/KL, exact-match quality, and zero-change/no-mask controls.
If gates saturate at max or minimum depth, show the *actual* clean depth
histogram and mark the exit test uninformative. Threshold alternatives may
be chosen from a CLEAN-only development calibration and must be frozen
before looking at eviction outcomes; do not tune thresholds post hoc to
manufacture depth flips. If Huginn's FP32-cosine diagnostic is needed,
keep a separately labeled result from native BF16.

**Do not interpret EVICT-LIVE minus EVICT-REPLAY as an exact causal
mediated quality component:** the whole per-token history is altered.
A query-local clean-history replay or controlled gate/fixed-KV follow-up
is needed for tighter causal attribution.

### Gate E — ONLY IF B+C+D justify it: held-out KV policy

Test fixed 10/30/50% Context-token retention and identical actual KV
slot budgets for all token-global policies (all loop-bank planes for a
retained token), with attention / attention+value /
source-trajectory / query-diagnostic oracle / feasible one-pass surrogate.
Target accuracy, future NLL, reference KL, and actual kept token counts.
Evidence-locked oracle is interpretation only and NOT a deployable policy.
No physical HBM/TPOT claim from the full-sized masked reference cache.
Strong first/last-loop and loop-wise attention controls may be needed;
do not call merely outperforming mean attention "loop-specific."

## 5. Analysis and go/no-go logic

1. FIRST ask whether selected sources induce measurable, repeatable
   query-conditioned effects above BF16 and no-mask floors; one-loop
   tiny effects in v6 are a serious negative prior.
2. THEN ask whether query responses differ when immediate Attention
   Output perturbation is matched, beyond an architecture-wide late-loop
   sensitivity. Report null if difference vanishes.
3. THEN ask whether Query features predict FUTURE actual Context KV
   eviction better than source trajectory and local controls on held-out
   question UIDs; bootstrap by independent prompts.
4. THEN assess native dynamic exit changes and whether they help, hurt,
   or are unrelated to quality. A depth flip without predictive utility
   is only an exit-system observation.
5. ONLY THEN design a cheap policy; if strong value/local-output baselines
   suffice, state generic importance, not a claimed loop-specific invention.

Minimum scientific reports: per-source query-loop trajectories,
attention/value/local-output matched pairs, fixed future-damage
scatterplots/curves, source-vs-query feature ablation, dynamic
depth-change histograms and live/replay/fixed per-request comparisons,
BF16 floor/parity, prompt-level uncertainty, failures/negative cases.

## 6. Agent autonomy and non-negotiable safeguards

The coding agent is a **research executor**, not a literal script writer.
It can choose a cleaner engine hook, adapt measurements when pilots are
uninformative, propose additional competing hypotheses/negative controls,
and carry out bounded follow-up runs on its allocated machine without
waiting for permission at every minor choice.

**Before each changed experiment**, record (hypothesis, observed failure
or clue, decision, new configuration/UIDs, what comparisons remain valid)
in a concise lab journal; use a new manifest and output directory for
changed sampling, mask width, exit threshold or measurement semantics.
Never retroactively change old frozen experiments, cherry-pick only
positive pairs, contaminate the test split, delete failures, or overwrite
other branches/processes. Do not alter original FERRY work.

An agent may conclude H2/H3/H4 are unsupported. This is a correct result.
Stop or redirect to a narrower defensible question rather than inventing
a recurrence-specific advantage. Report exact commands, code SHA,
parity status, completed/failed jobs and metric uncertainty.

## 7. Executable starting points already present

- Models, pinned revisions: kvp/specs.py and kvp/models.py.
- Token-major prefill/fixed+dynamic+recorded replay: kvp/role_sensitivity.py,
  kvp/role_span_oracle.py, kvp/depth_eviction_probe.py, kvp/native_exit.py.
- Existing source and prompt-query attention: kvp/role_sensitivity.py;
  make sure new measurements are q-specific rather than source averages.
- Clean full-prefill reused across fixed counterfactual decodes:
  kvp/loopzip_cached_eval.py.
- v6 generated-query one-loop / persistent / clean-history diagnostics:
  kvp/loopzip_persistent_eviction.py.
- Pinned Controlled Retrieval and HotpotQA-Distractor workloads:
  kvp/role_evidence_data.py and scripts/role_sensitivity_sweep.py.
- Failure-first attention baselines and budget semantics:
  kvp/loopzip_policy.py.

Suggested initial CPU tests: q anchor is causal and correctly aligned;
single-loop mask changes only target query/loop and preserves prefill
prefix; first target is invariant for post-prefill masks; role/evidence
label permutations do not change MAIN candidate selection; snapshot
replays never leak evicted generated KV across branches; live and
recorded replay parity holds for BOTH models; no-mask exit traces match.
See companion QUERY_CONDITIONED_AGENT_BRIEF.md for an agent-facing handoff.
