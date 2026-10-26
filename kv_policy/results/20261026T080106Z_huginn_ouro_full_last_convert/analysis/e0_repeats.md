## E0 audit: discordant items repeated at B1 (BF16 and FP32)

Up to 6 FULL-only and 6 LAST-only correct items per model x dataset, regenerated at batch size 1. 'same verdict' = B1 correctness equals the batched E0 receipt.

| model | dtype | items | FULL same verdict | LAST-C same verdict | FULL identical tokens | LAST-C identical tokens | discordance kept at B1 (FULL≠LAST correctness) |
|---|---|---:|---:|---:|---:|---:|---:|
| huginn-0125 | bfloat16 | 15 | 11 | 10 | 9 | 7 | 6 |
| huginn-0125 | float32 | 15 | 10 | 8 | 7 | 2 | 5 |
| ouro-1.4b | bfloat16 | 12 | 12 | 12 | 8 | 0 | 12 |
| ouro-1.4b | float32 | 12 | 12 | 12 | 8 | 2 | 12 |
| ouro-2.6b | bfloat16 | 18 | 15 | 17 | 11 | 4 | 14 |
| ouro-2.6b | float32 | 18 | 15 | 15 | 10 | 4 | 14 |
