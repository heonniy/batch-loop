# Loop-wise attention phenomenology (same query, same weights, recurrence 1..R)

Runs: 20261102T093000Z_qc_v7_pilotA, 20261102T103000Z_qc_v7_pilotB_holdout; prompts: huginn-0125/controlled_retrieval: 20, huginn-0125/hotpotqa_distractor: 20, ouro-2.6b/controlled_retrieval: 20, ouro-2.6b/hotpotqa_distractor: 20

## Generated queries (gold continuation): per-loop statistics, mean over queries and prompts

| model | dataset | loop bin | entropy over Context | evidence mass (of Context) | sink mass (first 4) | local mass (last 16) | question mass | ρ(loop r, loop R) | ρ(loop r, loop r−1) | top-1 is evidence | top-8 evidence frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | ≤1/8 | 4.89 | 0.156 | 0.293 | 0.242 | 0.0906 | 0.51 | 0.752 | 0.45 | 0.209 |
| huginn-0125 | controlled_retrieval | 1/8–1/4 | 4.53 | 0.282 | 0.313 | 0.222 | 0.136 | 0.908 | 0.925 | 0.627 | 0.349 |
| huginn-0125 | controlled_retrieval | 1/4–1/2 | 4.48 | 0.293 | 0.331 | 0.2 | 0.131 | 0.982 | 0.99 | 0.674 | 0.383 |
| huginn-0125 | controlled_retrieval | 1/2–3/4 | 4.43 | 0.301 | 0.336 | 0.2 | 0.131 | 0.997 | 0.999 | 0.658 | 0.394 |
| huginn-0125 | controlled_retrieval | 3/4–1 | 4.43 | 0.302 | 0.338 | 0.199 | 0.132 | 1 | 1 | 0.657 | 0.39 |
| huginn-0125 | hotpotqa_distractor | ≤1/8 | 5.13 | 0.331 | 0.307 | 0.247 | 0.104 | 0.587 | 0.736 | 0.577 | 0.428 |
| huginn-0125 | hotpotqa_distractor | 1/8–1/4 | 5.06 | 0.435 | 0.276 | 0.251 | 0.135 | 0.873 | 0.914 | 0.676 | 0.583 |
| huginn-0125 | hotpotqa_distractor | 1/4–1/2 | 4.99 | 0.46 | 0.289 | 0.224 | 0.154 | 0.973 | 0.988 | 0.723 | 0.634 |
| huginn-0125 | hotpotqa_distractor | 1/2–3/4 | 4.97 | 0.472 | 0.297 | 0.215 | 0.152 | 0.996 | 0.999 | 0.741 | 0.644 |
| huginn-0125 | hotpotqa_distractor | 3/4–1 | 4.96 | 0.475 | 0.299 | 0.214 | 0.152 | 1 | 1 | 0.733 | 0.65 |
| ouro-2.6b | controlled_retrieval | 1/8–1/4 | 5.19 | 0.206 | 0.721 | 0.0829 | 0.0609 | 0.512 | — | 0.632 | 0.398 |
| ouro-2.6b | controlled_retrieval | 1/4–1/2 | 5.22 | 0.194 | 0.73 | 0.0811 | 0.0614 | 0.683 | 0.752 | 0.595 | 0.423 |
| ouro-2.6b | controlled_retrieval | 1/2–3/4 | 5.24 | 0.173 | 0.736 | 0.0667 | 0.0484 | 0.904 | 0.81 | 0.55 | 0.382 |
| ouro-2.6b | controlled_retrieval | 3/4–1 | 5.14 | 0.162 | 0.733 | 0.0631 | 0.0426 | 1 | 0.904 | 0.423 | 0.317 |
| ouro-2.6b | hotpotqa_distractor | 1/8–1/4 | 5.55 | 0.381 | 0.708 | 0.0912 | 0.0648 | 0.608 | — | 0.725 | 0.62 |
| ouro-2.6b | hotpotqa_distractor | 1/4–1/2 | 5.33 | 0.407 | 0.712 | 0.0867 | 0.0612 | 0.821 | 0.76 | 0.688 | 0.614 |
| ouro-2.6b | hotpotqa_distractor | 1/2–3/4 | 5.33 | 0.424 | 0.708 | 0.0842 | 0.0697 | 0.942 | 0.905 | 0.703 | 0.656 |
| ouro-2.6b | hotpotqa_distractor | 3/4–1 | 5.27 | 0.435 | 0.714 | 0.081 | 0.0676 | 1 | 0.942 | 0.652 | 0.654 |

## Exact per-loop curves (generated queries): evidence mass and ρ(loop r, loop R)

| model | dataset | loop | evidence mass [CI over prompts] | ρ vs last [CI] | entropy | top-8 evidence frac |
|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 1 | 0.092 [0.069,0.116] | 0.177 [0.152,0.198] | 5.24 | 0.16 |
| huginn-0125 | controlled_retrieval | 2 | 0.145 [0.125,0.164] | 0.438 [0.413,0.459] | 4.91 | 0.191 |
| huginn-0125 | controlled_retrieval | 3 | 0.194 [0.148,0.239] | 0.677 [0.666,0.687] | 4.63 | 0.236 |
| huginn-0125 | controlled_retrieval | 4 | 0.193 [0.154,0.232] | 0.75 [0.740,0.759] | 4.79 | 0.248 |
| huginn-0125 | controlled_retrieval | 6 | 0.274 [0.234,0.311] | 0.903 [0.897,0.908] | 4.57 | 0.354 |
| huginn-0125 | controlled_retrieval | 8 | 0.314 [0.281,0.344] | 0.953 [0.949,0.957] | 4.39 | 0.382 |
| huginn-0125 | controlled_retrieval | 12 | 0.295 [0.268,0.320] | 0.982 [0.980,0.984] | 4.47 | 0.382 |
| huginn-0125 | controlled_retrieval | 16 | 0.296 [0.269,0.323] | 0.992 [0.992,0.993] | 4.47 | 0.385 |
| huginn-0125 | controlled_retrieval | 20 | 0.302 [0.273,0.330] | 0.997 [0.997,0.998] | 4.43 | 0.395 |
| huginn-0125 | controlled_retrieval | 24 | 0.303 [0.276,0.331] | 0.999 [0.999,0.999] | 4.43 | 0.396 |
| huginn-0125 | controlled_retrieval | 28 | 0.302 [0.274,0.331] | 1 [1.000,1.000] | 4.43 | 0.389 |
| huginn-0125 | controlled_retrieval | 32 | 0.301 [0.274,0.330] | 1 [1.000,1.000] | 4.43 | 0.39 |
| huginn-0125 | hotpotqa_distractor | 1 | 0.254 [0.212,0.306] | 0.366 [0.330,0.401] | 5.17 | 0.269 |
| huginn-0125 | hotpotqa_distractor | 2 | 0.312 [0.288,0.334] | 0.556 [0.532,0.580] | 5.25 | 0.424 |
| huginn-0125 | hotpotqa_distractor | 3 | 0.359 [0.321,0.393] | 0.691 [0.673,0.708] | 5.01 | 0.492 |
| huginn-0125 | hotpotqa_distractor | 4 | 0.376 [0.333,0.417] | 0.753 [0.735,0.771] | 5.09 | 0.528 |
| huginn-0125 | hotpotqa_distractor | 6 | 0.425 [0.379,0.467] | 0.859 [0.843,0.873] | 5.04 | 0.593 |
| huginn-0125 | hotpotqa_distractor | 8 | 0.425 [0.381,0.463] | 0.917 [0.909,0.926] | 5.05 | 0.589 |
| huginn-0125 | hotpotqa_distractor | 12 | 0.444 [0.400,0.487] | 0.975 [0.972,0.978] | 4.99 | 0.647 |
| huginn-0125 | hotpotqa_distractor | 16 | 0.454 [0.411,0.495] | 0.99 [0.988,0.991] | 4.98 | 0.647 |
| huginn-0125 | hotpotqa_distractor | 20 | 0.457 [0.413,0.497] | 0.996 [0.996,0.997] | 4.97 | 0.645 |
| huginn-0125 | hotpotqa_distractor | 24 | 0.46 [0.417,0.501] | 0.999 [0.999,0.999] | 4.96 | 0.647 |
| huginn-0125 | hotpotqa_distractor | 28 | 0.461 [0.417,0.501] | 1 [0.999,1.000] | 4.96 | 0.651 |
| huginn-0125 | hotpotqa_distractor | 32 | 0.461 [0.417,0.500] | 1 [1.000,1.000] | 4.96 | 0.649 |
| ouro-2.6b | controlled_retrieval | 1 | 0.206 [0.172,0.238] | 0.512 [0.494,0.530] | 5.19 | 0.398 |
| ouro-2.6b | controlled_retrieval | 2 | 0.194 [0.175,0.214] | 0.683 [0.663,0.703] | 5.22 | 0.423 |
| ouro-2.6b | controlled_retrieval | 3 | 0.173 [0.160,0.185] | 0.904 [0.896,0.912] | 5.24 | 0.382 |
| ouro-2.6b | controlled_retrieval | 4 | 0.162 [0.152,0.172] | 1 [1.000,1.000] | 5.14 | 0.317 |
| ouro-2.6b | hotpotqa_distractor | 1 | 0.377 [0.341,0.411] | 0.603 [0.577,0.630] | 5.55 | 0.62 |
| ouro-2.6b | hotpotqa_distractor | 2 | 0.402 [0.367,0.436] | 0.818 [0.800,0.834] | 5.33 | 0.614 |
| ouro-2.6b | hotpotqa_distractor | 3 | 0.42 [0.387,0.453] | 0.941 [0.936,0.947] | 5.33 | 0.656 |
| ouro-2.6b | hotpotqa_distractor | 4 | 0.431 [0.395,0.466] | 1 [1.000,1.000] | 5.27 | 0.654 |

## Where in the recurrence does a query's evidence attention peak? (fraction of generated queries by peak-loop bin)

| model | dataset | ≤1/8 | 1/8–1/4 | 1/4–1/2 | 1/2–3/4 | 3/4–1 | mean peak loop / R |
|---|---|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0.15 | 0.45 | 0.25 | 0.11 | 0.041 | 0.311 |
| huginn-0125 | hotpotqa_distractor | 0.21 | 0.26 | 0.28 | 0.1 | 0.15 | 0.373 |
| ouro-2.6b | controlled_retrieval | 0 | 0.49 | 0.2 | 0.13 | 0.19 | 0.502 |
| ouro-2.6b | hotpotqa_distractor | 0 | 0.22 | 0.18 | 0.12 | 0.48 | 0.716 |

## Cross-query / cross-loop agreement of Context attention (Spearman over Context tokens)

| model | dataset | anchor(loop R) vs generated t*(loop R) | anchor(loop 1) vs generated t*(loop R) | generated t*: loop 1 vs loop R | generated t=1 (format) vs t* at loop R |
|---|---|---|---|---|---|
| huginn-0125 | controlled_retrieval | 0.411 [0.381,0.441] | -0.191 | 0.241 [0.191,0.293] | 0.608 |
| huginn-0125 | hotpotqa_distractor | 0.43 [0.354,0.507] | 0.182 | 0.403 [0.355,0.453] | 0.408 |
| ouro-2.6b | controlled_retrieval | 0.291 [0.229,0.349] | 0.202 | 0.712 [0.690,0.733] | 0.482 |
| ouro-2.6b | hotpotqa_distractor | 0.628 [0.572,0.683] | 0.42 | 0.583 [0.522,0.643] | 0.624 |
