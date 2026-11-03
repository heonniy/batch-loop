# Clean quality-vs-compute under native exit: FULL vs shared KV (ouro-2.6b / controlled_retrieval, 8 prompts)

Each row: one threshold, both policies; mean over prompts of realized mean depth (teacher-forced live), future NLL minus the FULL-fixed NLL of the same prompt, KL to FULL-fixed, greedy EM.

| threshold | FULL depth | FULL ΔNLL | FULL KL | FULL EM | SHARED depth | SHARED ΔNLL | SHARED KL | SHARED EM |
|---|---|---|---|---|---|---|---|---|
| fixed R | 4.00 | +0.0000 | 0.00e+00 | 1.00 | 4.00 | +0.0549 | 2.87e-02 | 0.88 |
| 0.995 | 4.00 | +0.0000 | 0.00e+00 | 1.00 | 4.00 | +0.0549 | 2.87e-02 | 0.88 |
| 0.95 | 4.00 | +0.0000 | 0.00e+00 | 1.00 | 4.00 | +0.0549 | 2.87e-02 | 0.88 |
| 0.9 | 4.00 | +0.0000 | 0.00e+00 | 1.00 | 4.00 | +0.0549 | 2.87e-02 | 0.88 |
| 0.85 | 4.00 | +0.0000 | 0.00e+00 | 1.00 | 4.00 | +0.0549 | 2.87e-02 | 0.88 |
| 0.8 | 4.00 | +0.0000 | 0.00e+00 | 1.00 | 3.62 | +0.0539 | 2.87e-02 | 0.88 |
| 0.7 | 3.50 | -0.0001 | 2.58e-04 | 1.00 | 3.02 | +0.0416 | 2.65e-02 | 0.88 |
| 0.6 | 3.33 | -0.0003 | 1.84e-03 | 1.00 | 2.62 | +0.0447 | 2.81e-02 | 0.88 |
| 0.5 | 3.28 | -0.0021 | 6.89e-03 | 1.00 | 2.36 | +0.0265 | 2.65e-02 | 0.88 |
| 0.4 | 2.65 | -0.0022 | 8.24e-03 | 1.00 | 2.12 | +0.0273 | 2.72e-02 | 0.88 |
| 0.3 | 2.36 | -0.0009 | 9.44e-03 | 1.00 | 2.06 | +0.0364 | 3.10e-02 | 0.88 |
| 0.2 | 2.32 | -0.0004 | 9.42e-03 | 1.00 | 2.02 | +0.0428 | 3.42e-02 | 0.88 |

## Matched-depth comparison (FULL KL linearly interpolated at the SHARED point's realized mean depth)

| SHARED threshold | depth | SHARED KL | FULL KL at same depth | SHARED ΔNLL | FULL ΔNLL at same depth |
|---|---|---|---|---|---|
| None | 4.00 | 2.87e-02 | 0.00e+00 | +0.0549 | +0.0000 |
| 0.995 | 4.00 | 2.87e-02 | 0.00e+00 | +0.0549 | +0.0000 |
| 0.95 | 4.00 | 2.87e-02 | 0.00e+00 | +0.0549 | +0.0000 |
| 0.9 | 4.00 | 2.87e-02 | 0.00e+00 | +0.0549 | +0.0000 |
| 0.85 | 4.00 | 2.87e-02 | 0.00e+00 | +0.0549 | +0.0000 |
| 0.8 | 3.62 | 2.87e-02 | 1.94e-04 | +0.0539 | -0.0000 |
| 0.7 | 3.02 | 2.65e-02 | 7.44e-03 | +0.0416 | -0.0022 |
| 0.6 | 2.62 | 2.81e-02 | 8.34e-03 | +0.0447 | -0.0021 |
| 0.5 | 2.36 | 2.65e-02 | 9.44e-03 | +0.0265 | -0.0009 |
| 0.4 | 2.12 | 2.72e-02 | out of FULL range | +0.0273 | — |
| 0.3 | 2.06 | 3.10e-02 | out of FULL range | +0.0364 | — |
| 0.2 | 2.02 | 3.42e-02 | out of FULL range | +0.0428 | — |

R = 4. FULL depth range covered: 2.32–4.00.