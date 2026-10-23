# Huginn R32: HF parity, paired divergence, token-level KV restoration

Status: **experimental plan only; no new GPU run or code implementation claimed.**
Base: `research/huginn-r32-length-h200x8-20261023` (preserve existing raw receipts).
Model: `tomg-group-umd/huginn-0125`, fixed R=32, BF16 evaluation, H200 x8 independent workers.
**Do not use vllm-rlt, early exit, Ouro, or serving throughput in this study.**

## Motivation / existing evidence

The R32 fixed-depth rerun, `kv_policy/results/20261023T060125Z_huginn_r32_h200x8`,
has complete 200-item samples per dataset/policy and no failed jobs. It found:
GSM8K FULL 41.5%, LAST-convert 44.5%, AVG-convert 42.0%;
MATH-500 FULL 15.5%, LAST-convert 16.5%, AVG-convert 12.5%.
LAST–FULL token-weighted teacher-forced ΔNLL is +0.00144 GSM8K and −0.00026 MATH500.
Thus **do not assume FULL is superior to LAST**, and do not use accuracy difference alone as evidence.
MATH-500 has 47/200 FULL and 52/200 LAST generations ending at the 2048-token cap.
An official MATH-500 Huginn evaluation protocol was not available for the preceding rerun.
The previous study did NOT measure fixed-prefix full-distribution KL.

## Experimental invariants

- Freeze model revision, Huginn initial latent state (per UID/position), tokenizer, prompt,
  dataset subsets, numerical dtype, EOS/stop behavior and evaluation parsing.
- Separate **retained KV memory after prefill** from transient peak allocated during prefill.
- Shared-prefix replay never conditions FULL on its own output while LAST uses a different prefix.
- FULL/decode is the depth-indexed baseline. LAST/convert means loop-major FULL prefill,
  retain terminal per-token KV after prefill, then all current recurrent loops read past
  terminal KV. AVERAGE/convert is secondary.
- In all models the current token's per-loop K/V is computed normally; only *historical*
  K/V is replaced. Prelude and Coda layers retain standard KV.
- Never interpret downstream ground-truth-conditioned oracle as deployable.

## Stage 1 — native Huginn correctness / official evaluation audit (gate)

1. Run official Huginn HF + lm-eval GSM8K protocol R32 separately from the project runner.
   Preserve the exact authors' prompt, few-shot selection, tokenizer revision, chat template,
   generation cap, random state and scorer. Compare against project's frozen 200 subset;
   report full official accuracy separately if full dataset used. Do not call two protocols
   identical without prompt-token-ID comparison.
2. Native HF FULL vs existing reference FULL at R32: FP32 logits, first-token top-1,
   KL and max/mean logit error on short and real benchmark prompts, B1 and small batch.
   Verify latent state seed/initialization; use paired identical initial latents if possible.
3. BF16 greedy paired generations (e.g. 32 GSM8K + 32 MATH500 records) and teacher-forced
   logprobs: first differing token, top-1 agreement, NLL, batch-shape noise control.
4. Inspect raw generation, especially 2048 cap, answer format and scorer disagreements.
   Audit COMPLETE outputs and reasons, not just computed percentage.
5. Gate: do not claim native parity unless FP32 agrees within justified tolerance
   and BF16 disagreement is quantified. If mismatch is unexplained, stop downstream
   claims and write a failure report with minimal reproducer.

Artifacts: `analysis/stage1_parity.md`, `stage1_item_audit.csv`,
`official_eval_manifest.json`, scripts and raw receipts.

## Stage 2 — fixed-prefix distribution divergence (primary characterization)

For each selected prompt, generate one frozen FULL continuation with greedy decoding;
use the exact *same prefix* x_{<=t} for FULL and LAST, including FULL prompt and its
recorded continuation. Replay each policy under its own causal cache semantics.
At each eligible position t, record the full-vocabulary log-softmax distributions
p_F(.|x_{<=t}) and p_L(.|x_{<=t}). Compute
  KL_t = sum_v p_F(v) [log p_F(v) - log p_L(v)]
  delta_NLL_t = -log p_L(x_{t+1}) + log p_F(x_{t+1})
  top1_changed_t = 1[argmax p_F != argmax p_L].
Use FP32 accumulation for KL even if model BF16; guard against NaN and report
distribution normalization checks. NLL target here is *FULL-generated next token*,
separate from the previous gold-target teacher-forced NLL. Optionally repeat with
gold continuation and with LAST-generated prefix as sensitivity checks.

Sample selection: all frozen GSM8K-200 and MATH500-200 if practical; prioritize a
stratified pilot retaining BOTH correctly and incorrectly solved items, observed
generation lengths, and MATH difficulty levels. Do not silently filter truncated items.
Report separate counts by 1–64,65–128,129–256,257–512,513–1024,1025–2048.
For long positions report surviving problems, tokens, confidence intervals via
problem-level clustered bootstrap, and censoring. Do not infer increasing error
from changing item composition. Also compute per-request normalized curves and
first-disagreement distributions.

Artifacts: `analysis/stage2_kl_by_position.csv`,
`analysis/stage2_kl_by_item.csv`, `analysis/stage2_kl_report.md`;
per-step logits need not be committed if too large; commit hashes/statistics.

## Stage 3 — token-level restoration oracle (feasibility gate)

Pilot 32–50 frozen prompts (balance GSM8K/MATH, Full-correct/Full-incorrect,
short/long generations) and 16–64 candidate past tokens per prompt.
Evaluate *on the same fixed prefix* and the same future target windows as Stage 2.

For historical position i construct a mixed cache:
- default LAST for each historical token and each recurrent physical layer;
- selected token i reads/writes its per-depth KV (R=32) whenever the current
  recurrent query is at depth r; every other historical token remains terminal.
- compute candidate full cache from the SAME prefix using FULL forward;
  do not claim it can be materialized online without pre-computation.
- verify mixed-cache attention indexing, causal masking and tensor shapes against
  a direct independent slow dictionary reference on toy inputs.
- mixing may change subsequent recurrent query states: rerun forward causally
  for each intervention, never splice output logits or independently blend K/V.
- restore the correct depth-r KV for the selected token at all subsequent queries;
  no future-token leak. Track non-recurrent KV separately.

Score V_i = loss(LAST)-loss(LAST + restore token i) using next-token NLL on
an explicitly frozen horizon. Positive V_i means restoration helped; negative
means it hurt. Report BOTH signed V_i and positive-only/oracle best-of scores.
Candidate-only oracle is an upper bound on the *candidate set*, not a full
global optimum. Single-token benefits need not be additive for k>1;
re-evaluate whole selected sets, do not sum V_i to claim total benefit.

Selection policies with identical byte budget: LAST, RANDOM-k (>=5 seeds),
RECENT-k, KV-CONVERGENCE-k, OFFLINE ORACLE-k, FULL reference. Use
k fractions 0%,5%,10%,25% where feasible, compute actual recurrent layer
KV bytes (1+31f planes per token) plus constant Prelude/Coda KV.
Convergence score is an explicitly specified normalization of K and V
over loops, e.g. average relative consecutive norm difference; also compare
distance-to-terminal. K/V must be measured at equal layers/heads.
On GOLD-conditioned selection describe leakage clearly; perform held-out
target-window analysis before claiming generality.

Report latency/peak memory only as supplementary, not as an optimized serving
performance claim. Stage 3 is expensive: stop after pilot if signed gains
are negligible versus repeated-run numerical noise.

Artifacts: `analysis/stage3_oracle_candidates.csv`,
`analysis/stage3_oracle_pareto.csv`, `analysis/stage3_oracle_report.md`,
reference tests and reproducible manifest.

## Go / No-Go

GO if (a) parity gate passes, (b) at least a meaningful subset has positive
restoration value above BF16 noise, (c) candidate-oracle sets improve loss at
equal KV budgets versus LAST and RANDOM; assess confidence by request bootstrap.
CONDITIONAL if Oracle helps but simple convergence/recency fails: investigate
a predictive importance model, avoid an unsupported online claim.
NO-GO for Huginn token-adaptive retention if the oracle yields no reproducible
gain, or if quality gains cost nearly FULL KV, or if parity never validates.
Do not use arbitrary numerical effect-size thresholds invented after seeing data.

## Engineering and reproducibility

- Implement independently under `kv_policy/` rather than overwriting R32 baseline.
- Before long runs, unit test causal semantics with tiny fake K/V and compare
  FULL=restore-all, LAST=restore-none.
- Use eight independent GPU workers and atomic job claims; resume safely; failed
  jobs must be listed and retried deliberately. Avoid committing lock files.
- Commit complete manifest (git SHA, model/data hashes, environment, seed,
  GPU identity, exact commands), per-item CSV and aggregated results.
- Keep each stage an independent reproducible command and checkpointable
  artifacts. Document any sample reduction due to computational cost.
