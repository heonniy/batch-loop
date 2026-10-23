# Huginn R32 generation-length audit

R32, fixed exit. Compare cutoffs 256/512 (GSM8K) and 1024/2048 (MATH).
A generation finishing before the expanded cap is verified; a sequence exceeding an old cap is only potentially rescued, not proof that the old run would have followed the same trajectory.

| Dataset | Policy / mode | n | Acc | Mean tokens | hit extended cap | exceeded old cap | ended without answer |
|---|---|---:|---:|---:|---:|---:|---:|
| gsm8k | average/convert | 200 | 0.420 | 103.4 | 2 | 4 | 2 |
| gsm8k | full/decode | 200 | 0.415 | 105.0 | 3 | 3 | 3 |
| gsm8k | last/convert | 200 | 0.445 | 103.5 | 1 | 2 | 1 |
| math500 | average/convert | 200 | 0.125 | 610.8 | 49 | 49 | 50 |
| math500 | full/decode | 200 | 0.155 | 611.2 | 47 | 47 | 48 |
| math500 | last/convert | 200 | 0.165 | 650.2 | 52 | 52 | 53 |
