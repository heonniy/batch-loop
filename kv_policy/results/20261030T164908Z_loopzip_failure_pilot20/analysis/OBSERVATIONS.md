# LoopZip Phase 1/2 Observations — measured runs only

- Run directory: \`20261030T164908Z_loopzip_failure_pilot20\`
- Frozen source SHA: \`16a71d1a1845fab0cc947a23ac004a5685338d2f\`
- Budgets: **10%, 30%, 50% of aligned Context token positions**; other prompt roles unmasked.
- Attention baselines: Mean, Last, Max, Recent+Mean, Random (three seeds).
- Phase 1 complete: **yes**; Phase 2 available: **yes**.
- Full-size R-plane KV read masks, **not physical HBM savings or speedups**.

## Budget-wise failure counts

| Model | Dataset | K% | Requests | Full-correct | All-Attn wrong | Swap probed | NLL swap gain | Greedy rescued | Mean-only gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn-0125 | controlled_retrieval | 10 | 20 | 19 | 19 | 12 | 12 | 0 | 0 |
| huginn-0125 | controlled_retrieval | 30 | 20 | 19 | 13 | 12 | 12 | 0 | 6 |
| huginn-0125 | controlled_retrieval | 50 | 20 | 19 | 8 | 8 | 8 | 0 | 7 |
| huginn-0125 | hotpotqa_distractor | 10 | 20 | 9 | 7 | 7 | 7 | 0 | 1 |
| huginn-0125 | hotpotqa_distractor | 30 | 20 | 9 | 2 | 2 | 2 | 0 | 2 |
| huginn-0125 | hotpotqa_distractor | 50 | 20 | 9 | 1 | 1 | 1 | 0 | 0 |
| ouro-2.6b | controlled_retrieval | 10 | 20 | 20 | 20 | 12 | 12 | 0 | 0 |
| ouro-2.6b | controlled_retrieval | 30 | 20 | 20 | 14 | 12 | 12 | 0 | 6 |
| ouro-2.6b | controlled_retrieval | 50 | 20 | 20 | 11 | 11 | 10 | 0 | 6 |
| ouro-2.6b | hotpotqa_distractor | 10 | 20 | 11 | 5 | 5 | 5 | 0 | 1 |
| ouro-2.6b | hotpotqa_distractor | 30 | 20 | 11 | 4 | 4 | 4 | 0 | 1 |
| ouro-2.6b | hotpotqa_distractor | 50 | 20 | 11 | 0 | 0 | 0 | 0 | 2 |

## Highest residual Attention failures (post-hoc case inventory)

The cases below are sorted by signed future NLL loss of the BEST
among the tested Attention baselines, not by a convenient weak baseline.
This is selection for *mechanism study*, NOT an unbiased population estimate.

| Model | Dataset | UID | K% | Best-Attn ΔNLL | Local-swap gain | Accuracy rescue |
| --- | --- | --- | --- | --- | --- | --- |
| ouro-2.6b | hotpotqa_distractor | hotpotqa-distractor-5ab298fb554299449642c922 | 10 | 3.05886 | 0.07407 | no |
| huginn-0125 | controlled_retrieval | controlled-20261031-00018 | 10 | 2.85070 | 0.13240 | no |
| huginn-0125 | controlled_retrieval | controlled-20261031-00000 | 10 | 2.84027 | 0.02989 | no |
| huginn-0125 | controlled_retrieval | controlled-20261031-00003 | 10 | 2.82455 | 0.45888 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00015 | 10 | 2.80739 | 0.04112 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00013 | 10 | 2.79605 | 0.04719 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00003 | 10 | 2.79593 | 0.03896 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00012 | 10 | 2.78471 | 0.08754 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00019 | 10 | 2.76449 | 0.04512 | no |
| huginn-0125 | controlled_retrieval | controlled-20261031-00006 | 10 | 2.74281 | 0.06240 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00010 | 10 | 2.72945 | 0.87958 | no |
| ouro-2.6b | controlled_retrieval | controlled-20261031-00006 | 10 | 2.71778 | 0.05264 | no |

## Interpretation constraints

- If Mean fails but Last/Max works, this supports **aggregation**
  sensitivity, not a general non-attention signal.
- A positive same-budget swap gain shows a tested better KV set
  for that request, not that the global optimum has been found.
- Greedy answer recovery is only reported if optional greedy
  decoding was actually performed in the swap stage.
- A Phase-1/2 Attention selection gap alone does **NOT** prove
  recurrent error amplification caused the failure. Phase 3
  still needs immediate attention-output and loop-specific controls.
- No observed residual gap is a valid NEGATIVE finding; do not
  broaden budgets or choose more convenient problems post-hoc.
