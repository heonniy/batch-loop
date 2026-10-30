# Dynamic-depth Token Eviction + Recurrent KV Bank Importance — Follow-up

Branch: `research/dynamic-depth-loop-kv-importance-20261030`.
Source branch: `research/dynamic-depth-kv-eviction-h200x8-20261030`.

## Two independent interventions on the SAME recorded native exit trace

**A. Whole-token eviction** (existing): after prompt completion, hide one
prompt token from all history attention layers and measure future gold NLL.

**B. Depth-bank restoration** (new): after Full-Dynamic prompt computation,
keep all prompt tokens but use one of these read policies on future target
tokens, with the same frozen per-token exit depths:

1. `full`: for query loop `r` read source's `KV[min(r, d_source)]`
   (1-based notation; unexecuted banks use last actually computed bank).
2. `last`: for all query loops read `KV[d_source]`.
3. `firstlast`: query loop 1 reads `KV[1]`, later query loops read
   `KV[d_source]`. It has no extra bank when `d_source=1`.
4. `restore(source i, loop r)`: start with Last; read original
   `KV_i[r]` only at future query loop `r`, in every recurrent
   attention physical layer. All other source/history banks read Last.

A restore is valid ONLY when `1 <= r < d_i`: never invent uncomputed
KV; the terminal bank already exists as Last. The intervention occurs
**after complete prompt prefill**, so prompt token exit depths and
baseline next-token prediction stay identical. Future generated target
tokens are teacher-forced and physically execute only their recorded
depth. No refill/recomputation. The intervention is only an alternative
historical KV READ; it never changes the physically computed source
KV bank or the current token's own loop-specific KV.

**Token source depth vs KV loop importance is a distinct question.**
Do not interpret `source_exit_depth` as `important_bank_loop`.

## Metrics

A (unchanged):
`U_i = NLL(evict token i) - NLL(full dynamic)`.

B:
`Gap = NLL(last) - NLL(full)`
`Recovery(firstlast) = NLL(last) - NLL(firstlast)`
`G_(i,r) = NLL(last) - NLL(restore source i at loop r)`.

Positive `G_(i,r)` indicates a lower gold-NLL under that
single-source/single-recurrence restore.
Negative values are important counterexamples; do NOT clip them.

Output:
- `analysis/dynamic_loop_policy_nll.csv`: paired Full/Last/FirstLast per request.
- `analysis/token_loop_bank_oracle.csv`: source position, executed exit
  depth, restored r, future NLL gain; excludes nonexistent deeper banks.
- Existing `token_oracle.csv`, `depth_utility_summary.json`,
  `offline_policy_oracle_diagnostic.csv` for whole-token deletion.

Full/Last/FirstLast costs here are only **conceptual retained KV bytes**.
The reference cache still allocates Full R planes. A selected extra
`(token, recurrence)` restore is across all recurrent layers, not a
single physical layer bank. Compare against a layer-specific baseline
before claiming a practical optimizer.

## H200 x8 phased experiment

First run a smoke on one GPU and verify model parity + read semantics:
```bash
cd kv_policy
python -m pytest -q tests/test_dynamic_loop_bank.py tests/test_depth_eviction_probe.py tests/test_depth_eviction_model_parity.py
python scripts/depth_eviction_sweep.py --prepare \
  --run-dir results/depth_loop_smoke_20261030 \
  --n-examples 1 --n-candidates 2 --loop-bank
python scripts/depth_eviction_sweep.py --execute \
  --run-dir results/depth_loop_smoke_20261030 --gpu-ids 0
```

Then, CONDITIONAL on smoke success, run a new main experiment on
H200 x8; do not reuse old results because read policy implementations
and source code SHA changed:
```bash
python scripts/depth_eviction_sweep.py --prepare \
  --run-dir results/depth_loop_main100_20261030 \
  --subset main --n-examples 100 --n-candidates 4 \
  --loop-bank --loop-choices 1,2,3,4 \
  --max-prompt-tokens 96 --max-target-tokens 16
python scripts/depth_eviction_sweep.py --execute \
  --run-dir results/depth_loop_main100_20261030 \
  --gpu-ids 0,1,2,3,4,5,6,7
python scripts/depth_eviction_sweep.py --analyze \
  --run-dir results/depth_loop_main100_20261030
```

Initial loop choices 1–4 give all available nonterminal Ouro
recurrences, but only the first few of Huginn's up to R32. After
smoke, add Huginn `--loop-choices 1,2,4,8,16,24,31` in a SEPARATE
new frozen run; do not imply 1–4 exhaust Huginn's loop trajectory.

This reference B1 workload still uses last 96 prompt tokens and first
16 target tokens (gold NLL, B1), NOT full math task Accuracy.
The earlier 100×3×2 study showed near-zero source-depth vs eviction
importance and near-zero Attention importance. That null result can
be a limitation of truncated/short-horizon probes. Conditional followup
is intact prompt and 64-token suffix, on pre-registered smaller cohort
then larger as feasible; keep the earlier null results visible.

## Go/No-Go decision

1. Replay must reproduce Full dynamic, and missing-loop fallback must
   be validated against a tiny independent read oracle.
2. If Full vs Last shows no difference, loop-bank restoration cannot
   be motivated as needed at this task/exit setting.
3. If Full vs Last has a gap, first test whether First+Last closes it.
4. If there is residual gap, compare `G_(i,r)` for first versus
   middle-loop banks, stratified by source exit depth, and distinguish
   source depth from future query depth.
5. Promote only a stable, disjoint held-out improvement at matched
   physical KV bytes; a single-source oracle is NOT a deployed KV
   compression technique.
6. If Ouro-2.6B still has only R4 exits, run a separately frozen
   q-threshold sensitivity sweep; otherwise no depth-importance test
   is identifiable for that checkpoint.

Existing unrelated research branches and FERRY are untouched.
No GPU results are claimed by the GitHub-only commit.
