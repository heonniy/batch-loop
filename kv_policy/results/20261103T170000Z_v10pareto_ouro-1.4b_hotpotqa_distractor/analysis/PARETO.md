# Clean quality-vs-compute under native exit: FULL vs shared KV (ouro-1.4b / hotpotqa_distractor, 8 prompts)

Each row: one threshold, both policies; mean over prompts of realized mean depth (teacher-forced live), future NLL minus the FULL-fixed NLL of the same prompt, KL to FULL-fixed, greedy EM.

| threshold | FULL depth | FULL ΔNLL | FULL KL | FULL EM | SHARED depth | SHARED ΔNLL | SHARED KL | SHARED EM |
|---|---|---|---|---|---|---|---|---|
| fixed R | 4.00 | +0.0000 | 0.00e+00 | 0.12 | 4.00 | -0.2890 | 1.32e-01 | 0.38 |
| 0.995 | 4.00 | +0.0000 | 0.00e+00 | 0.12 | 4.00 | -0.2890 | 1.32e-01 | 0.38 |
| 0.95 | 4.00 | +0.0000 | 0.00e+00 | 0.12 | 4.00 | -0.2890 | 1.32e-01 | 0.38 |
| 0.9 | 4.00 | +0.0000 | 0.00e+00 | 0.12 | 4.00 | -0.2890 | 1.32e-01 | 0.38 |
| 0.85 | 4.00 | +0.0000 | 0.00e+00 | 0.12 | 4.00 | -0.2890 | 1.32e-01 | 0.38 |
| 0.8 | 4.00 | +0.0000 | 0.00e+00 | 0.12 | 3.89 | -0.2909 | 1.32e-01 | 0.38 |
| 0.7 | 3.86 | +0.0011 | 2.91e-04 | 0.12 | 3.46 | -0.2645 | 1.15e-01 | 0.38 |
| 0.6 | 3.74 | +0.0629 | 6.63e-03 | 0.12 | 3.08 | -0.2714 | 1.13e-01 | 0.38 |
| 0.5 | 3.62 | +0.1473 | 1.50e-02 | 0.12 | 2.81 | -0.2368 | 8.95e-02 | 0.25 |
| 0.4 | 3.38 | +0.1426 | 1.71e-02 | 0.12 | 2.62 | -0.2033 | 8.12e-02 | 0.25 |
| 0.3 | 3.21 | +0.1467 | 2.39e-02 | 0.12 | 2.38 | -0.2101 | 8.22e-02 | 0.25 |
| 0.2 | 2.97 | +0.0682 | 3.04e-02 | 0.12 | 2.17 | -0.1753 | 8.96e-02 | 0.25 |

## Matched-depth comparison (FULL KL linearly interpolated at the SHARED point's realized mean depth)

| SHARED threshold | depth | SHARED KL | FULL KL at same depth | SHARED ΔNLL | FULL ΔNLL at same depth |
|---|---|---|---|---|---|
| None | 4.00 | 1.32e-01 | 0.00e+00 | -0.2890 | +0.0000 |
| 0.995 | 4.00 | 1.32e-01 | 0.00e+00 | -0.2890 | +0.0000 |
| 0.95 | 4.00 | 1.32e-01 | 0.00e+00 | -0.2890 | +0.0000 |
| 0.9 | 4.00 | 1.32e-01 | 0.00e+00 | -0.2890 | +0.0000 |
| 0.85 | 4.00 | 1.32e-01 | 0.00e+00 | -0.2890 | +0.0000 |
| 0.8 | 3.89 | 1.32e-01 | 2.26e-04 | -0.2909 | +0.0008 |
| 0.7 | 3.46 | 1.15e-01 | 1.64e-02 | -0.2645 | +0.1441 |
| 0.6 | 3.08 | 1.13e-01 | 2.76e-02 | -0.2714 | +0.1023 |
| 0.5 | 2.81 | 8.95e-02 | out of FULL range | -0.2368 | — |
| 0.4 | 2.62 | 8.12e-02 | out of FULL range | -0.2033 | — |
| 0.3 | 2.38 | 8.22e-02 | out of FULL range | -0.2101 | — |
| 0.2 | 2.17 | 8.96e-02 | out of FULL range | -0.1753 | — |

R = 4. FULL depth range covered: 2.97–4.00.