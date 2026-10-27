| family | policy | query | layer | recurrence | reads A | reads B | reads C | prompt-call: B reads A |
|---|---|---|---:|---:|---|---|---|---|
| ouro | full/decode | D | 0 | 1 | r1 | r1 | r1 | r1 |
| ouro | full/decode | D | 0 | 2 | r2 | r2 | r2 | r2 |
| ouro | full/decode | D | 0 | 4 | r4 | r4 | r4 | r4 |
| ouro | full/strict | D | 0 | 1 | r1 | r1 | r1 | r1 |
| ouro | full/strict | D | 0 | 2 | r2 | r2 | r2 | r2 |
| ouro | full/strict | D | 0 | 4 | r4 | r4 | r4 | r4 |
| ouro | last/strict | D | 0 | 1 | r4 | r4 | r4 | r4 |
| ouro | last/strict | D | 0 | 2 | r4 | r4 | r4 | r4 |
| ouro | last/strict | D | 0 | 4 | r4 | r4 | r4 | r4 |
| ouro | first/strict | D | 0 | 1 | r1 | r1 | r1 | r1 |
| ouro | first/strict | D | 0 | 2 | r1 | r1 | r1 | r1 |
| ouro | first/strict | D | 0 | 4 | r1 | r1 | r1 | r1 |
| ouro | last/convert (contrast) | D | 0 | 1 | r4 | r4 | r4 | r1 |
| ouro | last/convert (contrast) | D | 0 | 2 | r4 | r4 | r4 | r2 |
| ouro | last/convert (contrast) | D | 0 | 4 | r4 | r4 | r4 | r4 |
| huginn | full/decode | D | 0 | 1 | r1 | r1 | r1 | r1 |
| huginn | full/decode | D | 0 | 2 | r2 | r2 | r2 | r2 |
| huginn | full/decode | D | 0 | 4 | r4 | r4 | r4 | r4 |
| huginn | full/strict | D | 0 | 1 | r1 | r1 | r1 | r1 |
| huginn | full/strict | D | 0 | 2 | r2 | r2 | r2 | r2 |
| huginn | full/strict | D | 0 | 4 | r4 | r4 | r4 | r4 |
| huginn | last/strict | D | 0 | 1 | r4 | r4 | r4 | r4 |
| huginn | last/strict | D | 0 | 2 | r4 | r4 | r4 | r4 |
| huginn | last/strict | D | 0 | 4 | r4 | r4 | r4 | r4 |
| huginn | first/strict | D | 0 | 1 | r1 | r1 | r1 | r1 |
| huginn | first/strict | D | 0 | 2 | r1 | r1 | r1 | r1 |
| huginn | first/strict | D | 0 | 4 | r1 | r1 | r1 | r1 |
| huginn | last/convert (contrast) | D | 0 | 1 | r4 | r4 | r4 | r1 |
| huginn | last/convert (contrast) | D | 0 | 2 | r4 | r4 | r4 | r2 |
| huginn | last/convert (contrast) | D | 0 | 4 | r4 | r4 | r4 | r4 |
