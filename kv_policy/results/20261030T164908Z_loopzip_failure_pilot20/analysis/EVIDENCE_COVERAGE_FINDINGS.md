# LoopZip v5 Stage 0 — evidence coverage audit (CPU, existing results only)

Audited run: `20261030T164908Z_loopzip_failure_pilot20` (code 16a71d1); 80/80 jobs aligned; alignment failures 0; ambiguous CR subspans 0.

Attention-policy flips (Full correct → policy wrong): **534** rows; with the COMPLETE evidence set retained: **0**.

Rows are budget×policy cells; the same question appears at 3 budgets (not independent). CIs: 1000 request-grouped bootstrap resamples.

| model | dataset | b | policy | full-correct rows | flips | flips w/ complete E | complete-E rate | mean cov(E) | acc given complete E | acc given incomplete E |
|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0.1 | last_attn | 19 | 19 | 0 | 0.00 | 0.21 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.1 | max_attn | 19 | 19 | 0 | 0.00 | 0.27 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.1 | mean_attn | 19 | 19 | 0 | 0.00 | 0.23 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.1 | random | 57 | 57 | 0 | 0.00 | 0.10 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.1 | recent_attn | 19 | 19 | 0 | 0.00 | 0.28 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.3 | last_attn | 19 | 19 | 0 | 0.00 | 0.44 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.3 | max_attn | 19 | 19 | 0 | 0.00 | 0.55 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.3 | mean_attn | 19 | 19 | 0 | 0.00 | 0.46 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.3 | random | 57 | 57 | 0 | 0.00 | 0.31 | — (n=0) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.3 | recent_attn | 19 | 13 | 0 | 0.30 | 0.64 | 1.00 [1.00,1.00] (n=6) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.5 | last_attn | 19 | 14 | 0 | 0.25 | 0.66 | 1.00 [1.00,1.00] (n=5) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.5 | max_attn | 19 | 16 | 0 | 0.00 | 0.71 | — (n=0) | 0.16 [0.00,0.32] |
| huginn-0125 | controlled_retrieval | 0.5 | mean_attn | 19 | 15 | 0 | 0.20 | 0.67 | 1.00 [1.00,1.00] (n=4) | 0.00 [0.00,0.00] |
| huginn-0125 | controlled_retrieval | 0.5 | random | 57 | 55 | 0 | 0.00 | 0.51 | — (n=0) | 0.04 [0.00,0.09] |
| huginn-0125 | controlled_retrieval | 0.5 | recent_attn | 19 | 13 | 0 | 0.30 | 0.73 | 1.00 [1.00,1.00] (n=6) | 0.00 [0.00,0.00] |
| huginn-0125 | hotpotqa_distractor | 0.1 | last_attn | 9 | 7 | 0 | 0.00 | 0.24 | — (n=0) | 0.22 [0.00,0.56] |
| huginn-0125 | hotpotqa_distractor | 0.1 | max_attn | 9 | 7 | 0 | 0.00 | 0.20 | — (n=0) | 0.22 [0.00,0.56] |
| huginn-0125 | hotpotqa_distractor | 0.1 | mean_attn | 9 | 8 | 0 | 0.00 | 0.24 | — (n=0) | 0.11 [0.00,0.33] |
| huginn-0125 | hotpotqa_distractor | 0.1 | random | 27 | 22 | 0 | 0.00 | 0.09 | — (n=0) | 0.19 [0.00,0.41] |
| huginn-0125 | hotpotqa_distractor | 0.1 | recent_attn | 9 | 8 | 0 | 0.00 | 0.21 | — (n=0) | 0.11 [0.00,0.33] |
| huginn-0125 | hotpotqa_distractor | 0.3 | last_attn | 9 | 4 | 0 | 0.00 | 0.49 | — (n=0) | 0.56 [0.22,0.89] |
| huginn-0125 | hotpotqa_distractor | 0.3 | max_attn | 9 | 5 | 0 | 0.00 | 0.42 | — (n=0) | 0.44 [0.11,0.78] |
| huginn-0125 | hotpotqa_distractor | 0.3 | mean_attn | 9 | 4 | 0 | 0.00 | 0.47 | — (n=0) | 0.56 [0.22,0.89] |
| huginn-0125 | hotpotqa_distractor | 0.3 | random | 27 | 21 | 0 | 0.00 | 0.30 | — (n=0) | 0.22 [0.04,0.44] |
| huginn-0125 | hotpotqa_distractor | 0.3 | recent_attn | 9 | 7 | 0 | 0.00 | 0.42 | — (n=0) | 0.22 [0.00,0.56] |
| huginn-0125 | hotpotqa_distractor | 0.5 | last_attn | 9 | 1 | 0 | 0.00 | 0.66 | — (n=0) | 0.89 [0.67,1.00] |
| huginn-0125 | hotpotqa_distractor | 0.5 | max_attn | 9 | 3 | 0 | 0.00 | 0.60 | — (n=0) | 0.67 [0.33,1.00] |
| huginn-0125 | hotpotqa_distractor | 0.5 | mean_attn | 9 | 1 | 0 | 0.00 | 0.64 | — (n=0) | 0.89 [0.67,1.00] |
| huginn-0125 | hotpotqa_distractor | 0.5 | random | 27 | 17 | 0 | 0.00 | 0.50 | — (n=0) | 0.37 [0.19,0.59] |
| huginn-0125 | hotpotqa_distractor | 0.5 | recent_attn | 9 | 1 | 0 | 0.00 | 0.61 | — (n=0) | 0.89 [0.67,1.00] |
| ouro-2.6b | controlled_retrieval | 0.1 | last_attn | 20 | 20 | 0 | 0.00 | 0.19 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.1 | max_attn | 20 | 20 | 0 | 0.00 | 0.23 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.1 | mean_attn | 20 | 20 | 0 | 0.00 | 0.26 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.1 | random | 60 | 60 | 0 | 0.00 | 0.10 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.1 | recent_attn | 20 | 20 | 0 | 0.00 | 0.30 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.3 | last_attn | 20 | 20 | 0 | 0.00 | 0.45 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.3 | max_attn | 20 | 20 | 0 | 0.00 | 0.50 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.3 | mean_attn | 20 | 20 | 0 | 0.00 | 0.50 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.3 | random | 60 | 60 | 0 | 0.00 | 0.28 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.3 | recent_attn | 20 | 14 | 0 | 0.30 | 0.64 | 1.00 [1.00,1.00] (n=6) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.5 | last_attn | 20 | 20 | 0 | 0.00 | 0.67 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.5 | max_attn | 20 | 20 | 0 | 0.00 | 0.66 | — (n=0) | 0.00 [0.00,0.00] |
| ouro-2.6b | controlled_retrieval | 0.5 | mean_attn | 20 | 17 | 0 | 0.10 | 0.68 | 1.00 [1.00,1.00] (n=2) | 0.06 [0.00,0.17] |
| ouro-2.6b | controlled_retrieval | 0.5 | random | 60 | 58 | 0 | 0.00 | 0.48 | — (n=0) | 0.03 [0.00,0.08] |
| ouro-2.6b | controlled_retrieval | 0.5 | recent_attn | 20 | 14 | 0 | 0.30 | 0.74 | 1.00 [1.00,1.00] (n=6) | 0.00 [0.00,0.00] |
| ouro-2.6b | hotpotqa_distractor | 0.1 | last_attn | 11 | 5 | 0 | 0.00 | 0.24 | — (n=0) | 0.55 [0.27,0.82] |
| ouro-2.6b | hotpotqa_distractor | 0.1 | max_attn | 11 | 6 | 0 | 0.00 | 0.23 | — (n=0) | 0.45 [0.18,0.73] |
| ouro-2.6b | hotpotqa_distractor | 0.1 | mean_attn | 11 | 6 | 0 | 0.00 | 0.23 | — (n=0) | 0.45 [0.18,0.73] |
| ouro-2.6b | hotpotqa_distractor | 0.1 | random | 33 | 25 | 0 | 0.00 | 0.10 | — (n=0) | 0.24 [0.03,0.48] |
| ouro-2.6b | hotpotqa_distractor | 0.1 | recent_attn | 11 | 6 | 0 | 0.00 | 0.20 | — (n=0) | 0.45 [0.18,0.73] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | last_attn | 11 | 4 | 0 | 0.00 | 0.50 | — (n=0) | 0.64 [0.36,0.91] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | max_attn | 11 | 5 | 0 | 0.00 | 0.49 | — (n=0) | 0.55 [0.27,0.82] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | mean_attn | 11 | 5 | 0 | 0.00 | 0.49 | — (n=0) | 0.55 [0.27,0.82] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | random | 33 | 16 | 0 | 0.00 | 0.29 | — (n=0) | 0.52 [0.30,0.73] |
| ouro-2.6b | hotpotqa_distractor | 0.3 | recent_attn | 11 | 5 | 0 | 0.00 | 0.44 | — (n=0) | 0.55 [0.27,0.82] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | last_attn | 11 | 0 | 0 | 0.00 | 0.70 | — (n=0) | 1.00 [1.00,1.00] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | max_attn | 11 | 2 | 0 | 0.00 | 0.67 | — (n=0) | 0.82 [0.55,1.00] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | mean_attn | 11 | 2 | 0 | 0.00 | 0.67 | — (n=0) | 0.82 [0.55,1.00] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | random | 33 | 10 | 0 | 0.00 | 0.50 | — (n=0) | 0.70 [0.52,0.85] |
| ouro-2.6b | hotpotqa_distractor | 0.5 | recent_attn | 11 | 3 | 0 | 0.00 | 0.64 | — (n=0) | 0.73 [0.45,1.00] |

## Controlled Retrieval: target location × policy (Full-correct rows)

| model | b | policy | location | n | acc | code complete | acc given code complete |
|---|---|---|---|---|---|---|---|
| huginn-0125 | 0.1 | last_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | last_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | last_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | max_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | max_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | max_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | mean_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | mean_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | mean_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | random | end | 18 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | random | front | 18 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | random | middle | 21 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | recent_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | recent_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.1 | recent_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | last_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | last_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | last_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | max_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | max_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | max_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | mean_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | mean_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | mean_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | random | end | 18 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | random | front | 18 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | random | middle | 21 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | recent_attn | end | 6 | 1.00 | 1.00 | 1.00 |
| huginn-0125 | 0.3 | recent_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.3 | recent_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | last_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | last_attn | front | 6 | 0.83 | 0.83 | 1.00 |
| huginn-0125 | 0.5 | last_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | max_attn | end | 6 | 0.50 | 0.17 | 1.00 |
| huginn-0125 | 0.5 | max_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | max_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | mean_attn | end | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | mean_attn | front | 6 | 0.67 | 0.67 | 1.00 |
| huginn-0125 | 0.5 | mean_attn | middle | 7 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | random | end | 18 | 0.06 | 0.00 | — |
| huginn-0125 | 0.5 | random | front | 18 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | random | middle | 21 | 0.05 | 0.00 | — |
| huginn-0125 | 0.5 | recent_attn | end | 6 | 1.00 | 1.00 | 1.00 |
| huginn-0125 | 0.5 | recent_attn | front | 6 | 0.00 | 0.00 | — |
| huginn-0125 | 0.5 | recent_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | last_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | last_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | last_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | max_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | max_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | max_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | mean_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | mean_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | mean_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | random | end | 18 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | random | front | 21 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | random | middle | 21 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | recent_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | recent_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.1 | recent_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | last_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | last_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | last_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | max_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | max_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | max_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | mean_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | mean_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | mean_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | random | end | 18 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | random | front | 21 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | random | middle | 21 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | recent_attn | end | 6 | 1.00 | 1.00 | 1.00 |
| ouro-2.6b | 0.3 | recent_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.3 | recent_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | last_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | last_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | last_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | max_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | max_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | max_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | mean_attn | end | 6 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | mean_attn | front | 7 | 0.43 | 0.29 | 1.00 |
| ouro-2.6b | 0.5 | mean_attn | middle | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | random | end | 18 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | random | front | 21 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | random | middle | 21 | 0.10 | 0.00 | — |
| ouro-2.6b | 0.5 | recent_attn | end | 6 | 1.00 | 1.00 | 1.00 |
| ouro-2.6b | 0.5 | recent_attn | front | 7 | 0.00 | 0.00 | — |
| ouro-2.6b | 0.5 | recent_attn | middle | 7 | 0.00 | 0.00 | — |

## Phase-2 best swaps: what was dropped / restored

| role | drop | restore | count |
|---|---|---|---|
| control | evidence_other | evidence_other | 1 |
| control | evidence_other | nonevidence | 2 |
| control | none | none | 2 |
| control | nonevidence | evidence_other | 3 |
| control | nonevidence | nonevidence | 6 |
| failure | code | nonevidence | 1 |
| failure | evidence_other | evidence_other | 1 |
| failure | evidence_other | nonevidence | 4 |
| failure | id | nonevidence | 2 |
| failure | none | none | 1 |
| failure | nonevidence | code | 3 |
| failure | nonevidence | evidence_other | 5 |
| failure | nonevidence | nonevidence | 69 |

Best swaps that changed evidence coverage: 20/100; mean gain when coverage rose 0.04, when unchanged 0.07.

Evidence offsets are post-hoc diagnostics; no recurrence-specific claim follows from this audit.
