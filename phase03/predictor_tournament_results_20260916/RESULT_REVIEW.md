# Phase 03 — Practical predictor tournament review

This packet covers two models at primary and secondary thresholds, with request-disjoint
train/validation/test splits and held-out N256/B32, 32-token physical cohorts. Both lanes used
frozen B=1 live depth labels and physically shrinking survivor batches. The user corrected the
Phase-3 GPU assignment to GPU6=Ouro and GPU7=Huginn, overriding the main-branch GPU8 plan
for this execution; no other GPU was used for model work.

| Model | Best test RIC method | Physical wall winner | Winner RIC recovery | Winner wall recovery | Predictor overhead / decode |
| --- | --- | --- | ---: | ---: | ---: |
| huginn | C8 | C8 | 21.4% | -4.6% | 2.1% |
| ouro | C8 | C8 | 18.5% | 11.4% | 1.9% |

## Evidence and limits

The complete C0–C9 validation and test rows are in `predictor_tournament.csv`. All top-three
methods selected by test RIC were physically measured with two warmups and five measured
repeats each. `physical_matrix.csv` gives medians; raw `summary.csv` and recurrence-event
receipts remain in each model's physical directory. The wall timer includes predictor feature
extraction, CPU head inference, sort, and (for Huginn) the common Prelude staging path.

The work checksum is identical across strategies in each model. Physical labels and input token
IDs are trace-forced to hold recurrent work and token trajectories fixed; this is not live generation
or an output-quality comparison. The generated-next-token match fraction in the physical matrix
quantifies output drift from the B=1 feature collector, despite frozen input tokens.
Dataset revisions remain UNKNOWN (cached Hugging Face datasets); prompt content hashes and
request split IDs are frozen in the feature manifests. The secondary thresholds reuse the same
prompt pool and split IDs, but have newly collected self-consistent features/labels; this is
threshold transfer, not generalization to a new prompt sample. The Phase-2 Oracle ceiling used a different
first-N cohort, so these held-out FIFO/Oracle numbers are the appropriate denominator.
The primary collectors started from an uncommitted worktree: their launch Git HEAD predates the
collector implementation commit. Post-run source-file hashes and that provenance limitation are
recorded in each feature manifest rather than mislabeling the launch HEAD as the source commit.

## Model-specific physical results

### huginn

Held-out STAGED_FIFO/FIFO→Oracle: RIC 3031→2376; median decode wall 23.597→19.608 s. Best measured predictor: C8.
Trace ablation (Oracle RIC recovery): previous depth C0 6.9%; history C1 9.2%; trajectory C2 4.9%; uncertainty C3 5.5%; embedding C4 4.9%; representation C5 5.2%; fused regression C6 15.3%; ordinal C7 18.8%; pairwise C8 20.9%; RIC-aware listwise C9 19.5%.
The best measured physical predictor did not beat staged FIFO/FIFO after overhead.

### ouro

Held-out STAGED_FIFO/FIFO→Oracle: RIC 1021→670; median decode wall 26.196→17.051 s. Best measured predictor: C8.
Trace ablation (Oracle RIC recovery): previous depth C0 3.4%; history C1 6.3%; trajectory C2 4.0%; uncertainty C3 2.6%; embedding C4 8.5%; representation C5 10.8%; fused regression C6 13.4%; ordinal C7 17.7%; pairwise C8 17.9%; RIC-aware listwise C9 17.1%.
The best measured physical predictor recovered less than 25% of Oracle wall headroom.

## Secondary-regime transfer

The primary physical winner and head weights were frozen before the secondary
threshold/q cohort was scored. The full C0–C9 frozen transfer and independently
refit secondary tables are retained; refitting is diagnostic, not the transfer claim.
Secondary physical runs compare that same frozen winner with FIFO and Oracle using
new self-consistent features/labels and the same 2+5 timing protocol.

| Model / regime | Frozen winner | Transfer test RIC recovery | Refit test RIC recovery | Physical RIC recovery | Physical wall recovery |
| --- | --- | ---: | ---: | ---: | ---: |
| huginn / huginn_threshold_0p26_n640_t32 | C8 | 17.9% | 20.8% | 17.2% | -18.7% |
| ouro / ouro_q0p4_n640_t32 | C8 | 20.5% | 20.5% | 20.5% | 9.1% |

## N=B controls

With one complete microbatch, FIFO and Oracle execute the same recurrent invocation
count. Their small wall differences are timing noise/control overhead, not scheduler gain.

| Model | N=B | FIFO/Oracle RIC | FIFO wall (s) | Oracle wall (s) |
| --- | ---: | ---: | ---: | ---: |
| huginn | 32 | 365 / 365 | 2.933 | 2.924 |
| ouro | 32 | 126 / 126 | 3.458 | 3.443 |

## Decision

Pairwise C8 versus Huber regression C6, and listwise C9 versus C8, are shown explicitly
above. On Huginn, the Prelude-only C5 result should be weighed against the staging and
predictor overhead in the physical matrix; representation predictiveness alone is not a
physical speed claim. The best measured candidates and negative competitors are preserved.
Interpret positive wall recovery as recovery of this trace-forced no-refill Oracle
opportunity, not as native quality or production throughput.

Plan decision: **3 — lightweight predictors recover too little physical Oracle room; a new predictor architecture or problem framing is warranted**. The 25% useful-band threshold must be met
on both primary and frozen-head secondary physical cohorts; a single positive
model is not a claim that the method works across both families.

The next finite action is a live-exit quality/trajectory audit; it is outside
this phase's scope.
