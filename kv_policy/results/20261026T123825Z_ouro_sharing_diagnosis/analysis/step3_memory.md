# Step 3 — retained KV layout (analytic, BF16)

Per token. Recurrent plane = one (layer, recurrence) K+V slot: 2 x H x D x 2 bytes. Retained = during decode after Convert. Every policy here needs ALL planes for the prompt during the parallel FULL prefill (transient peak unchanged by Convert). The reference engine allocates a FULL-sized broadcast store for every policy, so measured engine peaks are not policy footprints; a compact store was not benchmarked and no serving speed is claimed.

| model | policy | protected banks | retained recurrent planes / token | bytes / token | vs FULL | vs LAST-CONVERT |
|---|---|---|---:|---:|---:|---:|
| ouro-1.4b | FULL | all | 96 | 768.0 KiB | 1.000 | 4.000 |
| ouro-1.4b | LAST-CONVERT | — | 24 | 192.0 KiB | 0.250 | 1.000 |
| ouro-1.4b | bankL0R12 | [[0, 0], [0, 1]] | 26 | 208.0 KiB | 0.271 | 1.083 |
| ouro-1.4b | bankNeg1 | [[1, 0]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-1.4b | bankP0 | [[0, 0]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-1.4b | bankR1all | 24 banks | 48 | 384.0 KiB | 0.500 | 2.000 |
| ouro-1.4b | bankcal | [[0, 0], [1, 0]] | 26 | 208.0 KiB | 0.271 | 1.083 |
| ouro-1.4b | bankrand0 | [[20, 2]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-1.4b | bankrand1 | [[23, 2]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-1.4b | bankrand2 | [[8, 2]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-1.4b | bankrand3 | [[20, 0]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-1.4b | bankrand4 | [[16, 2]] | 25 | 200.0 KiB | 0.260 | 1.042 |
| ouro-2.6b | FULL | all | 192 | 1536.0 KiB | 1.000 | 4.000 |
| ouro-2.6b | LAST-CONVERT | — | 48 | 384.0 KiB | 0.250 | 1.000 |
| ouro-2.6b | bankL0R12 | [[0, 0], [0, 1]] | 50 | 400.0 KiB | 0.260 | 1.042 |
| ouro-2.6b | bankNeg1 | [[1, 0]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankP0 | [[0, 0]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankR1all | 48 banks | 96 | 768.0 KiB | 0.500 | 2.000 |
| ouro-2.6b | bankcal | [[0, 0]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankrand0 | [[30, 0]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankrand1 | [[23, 1]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankrand2 | [[18, 2]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankrand3 | [[24, 2]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| ouro-2.6b | bankrand4 | [[15, 1]] | 49 | 392.0 KiB | 0.255 | 1.021 |
| huginn-0125 | FULL | all | 128 | 2722.5 KiB | 1.000 | 16.500 |
| huginn-0125 | LAST-CONVERT | — | 4 | 165.0 KiB | 0.061 | 1.000 |
| huginn-0125 | bankL0R12 | [[0, 0], [0, 1]] | 6 | 206.2 KiB | 0.076 | 1.250 |
| huginn-0125 | bankNeg1 | [[1, 0]] | 5 | 185.6 KiB | 0.068 | 1.125 |
| huginn-0125 | bankP0 | [[0, 0]] | 5 | 185.6 KiB | 0.068 | 1.125 |
| huginn-0125 | bankR1all | [[0, 0], [1, 0], [2, 0], [3, 0]] | 8 | 247.5 KiB | 0.091 | 1.500 |
| huginn-0125 | bankrand0 | [[1, 24]] | 5 | 185.6 KiB | 0.068 | 1.125 |
| huginn-0125 | bankrand1 | [[2, 14]] | 5 | 185.6 KiB | 0.068 | 1.125 |
| huginn-0125 | bankrand2 | [[1, 12]] | 5 | 185.6 KiB | 0.068 | 1.125 |
| huginn-0125 | bankrand3 | [[2, 21]] | 5 | 185.6 KiB | 0.068 | 1.125 |
| huginn-0125 | bankrand4 | [[2, 27]] | 5 | 185.6 KiB | 0.068 | 1.125 |
