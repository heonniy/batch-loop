# v8 two-axis KV budget — findings (20261103T053000Z_v8b_gsm_reasoning)

Completeness: 96/96 jobs, errors 0, problems 0, complete=True

## Settle depth of Context tokens (first plane within 10% of the terminal state; pooled over prompts)


## Reasoning arm: zero-shot GSM8K greedy accuracy under plane policies (paired with the model's FULL generation)

lastK@prompt = prompt tokens keep their last K planes; @prompt+generated = every generated token is also converted to its last K planes after it finishes (what a real cache would hold); first1 = first plane only (negative control); token_random50 = half of the question tokens evicted (full planes). Δacc vs FULL [95% problem bootstrap]; 'identical' = greedy token sequence identical to FULL.

| model | arm | n | acc | FULL acc | Δacc [CI] | flexible-extract acc | identical generations | median first divergence (tokens) | mean length |
|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | first1@prompt+generated | 32 | 0 | 0.31 | -0.31 [-0.469,-0.156] | 0 | 0 | 4 | 181 |
| huginn-0125 | full | 32 | 0.31 | 0.31 | 0 [0.000,0.000] | 0.28 | 1 | — | 228 |
| huginn-0125 | last1@prompt | 32 | 0.25 | 0.31 | -0.062 [-0.156,0.000] | 0.22 | 0.062 | 150 | 233 |
| huginn-0125 | last1@prompt+generated | 32 | 0.31 | 0.31 | 0 [-0.125,0.125] | 0.31 | 0.031 | 120 | 224 |
| huginn-0125 | last3@prompt+generated | 32 | 0.34 | 0.31 | 0.031 [0.000,0.094] | 0.31 | 0.031 | 152 | 229 |
| huginn-0125 | token_random50@prompt | 32 | 0.031 | 0.31 | -0.28 [-0.438,-0.125] | 0.031 | 0 | 3 | 224 |
| ouro-1.4b | first1@prompt+generated | 32 | 0.094 | 0.31 | -0.22 [-0.375,-0.062] | 0.12 | 0 | 4 | 279 |
| ouro-1.4b | full | 32 | 0.31 | 0.31 | 0 [0.000,0.000] | 0.28 | 1 | — | 318 |
| ouro-1.4b | last1@prompt | 32 | 0.38 | 0.31 | 0.062 [0.000,0.156] | 0.19 | 0 | 16 | 314 |
| ouro-1.4b | last1@prompt+generated | 32 | 0 | 0.31 | -0.31 [-0.469,-0.156] | 0.25 | 0 | 9 | 308 |
| ouro-1.4b | token_random50@prompt | 32 | 0.031 | 0.31 | -0.28 [-0.469,-0.094] | 0.062 | 0 | 4 | 311 |
| ouro-2.6b | first1@prompt+generated | 32 | 0.12 | 0.34 | -0.22 [-0.406,-0.031] | 0.25 | 0 | 10 | 289 |
| ouro-2.6b | full | 32 | 0.34 | 0.34 | 0 [0.000,0.000] | 0.38 | 1 | — | 309 |
| ouro-2.6b | last1@prompt | 32 | 0.38 | 0.34 | 0.031 [-0.094,0.188] | 0.38 | 0.062 | 39.5 | 315 |
| ouro-2.6b | last1@prompt+generated | 32 | 0.38 | 0.34 | 0.031 [-0.156,0.219] | 0.44 | 0 | 22 | 306 |
| ouro-2.6b | token_random50@prompt | 32 | 0.031 | 0.34 | -0.31 [-0.469,-0.156] | 0.031 | 0 | 9.5 | 308 |
