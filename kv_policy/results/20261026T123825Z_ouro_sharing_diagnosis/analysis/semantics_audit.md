# Step 0 — semantic audit (OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md)

## 1. What the official Ouro code and paper specify

- **Official HF implementation** (pinned `modeling_ouro.py`, Ouro-1.4B `574fa66c…` / Ouro-2.6B `1ed04250…`):
  - `UniversalTransformerCache.update` stores K/V at index `current_ut * num_hidden_layers + layer_idx`. That is a
    separate, depth-indexed cache per (UT step, layer), which is FULL semantics.
  - The released code has **no KV-sharing / reuse option**.
  - The default inference path (`early_exit_threshold = 1.0`) selects the exit step from the cumulative ACT pdf.
  - Every UT step is always computed for every position. The exit only chooses which step's hidden state feeds
    `lm_head`.
- **Official cached generation is not executable at the pinned revision:**
  - On transformers 4.54.1, `UniversalTransformerCache()` fails with "property 'key_cache' … has no setter".
  - On transformers 4.53.3, the modeling code does not import (`GenericForQuestionAnswering`).
  - So no official sharing code path could be run. Native parity below uses the official forward without a cache,
    recomputing the full prefix for every generated token.
- **Paper §5.4.2 / Table 14** (arXiv 2510.25741v2):
  - "during prefilling, all four recurrent steps require their own KV caches"; reuse during prefilling degrades
    GSM8K by more than 10 points.
  - During decoding: "Last-step reuse: Only maintain KV cache from the final (4th) recurrent step"; first-step reuse
    collapses (GSM8K 18.73); last-step and averaged reuse are "nearly identical" to the full cache (78.85 vs 78.92
    GSM8K, 80.40 vs 82.40 MATH-500).
  - The model is not named. The FULL row equals Table 7's "Ouro 1.4B R4" column, so the base 1.4B is likely, but this
    is an inference.
  - The paper does not state whether the **prompt** K/V is converted to the last step after prefill (our
    LAST-CONVERT) or kept per step (the diagnostic reading LAST-DECODE). Its claimed 4.00× memory saving matches
    CONVERT; "prefilling requires its own KV caches" is compatible with either reading.
  - **The paper's Last semantics remain unverified.** We therefore label our collapse as "our LAST-CONVERT" only.
- The paper's evaluation protocol is not reproducible: MATH-500 uses an unpublished in-house 5-shot prompt
  (see E0); GSM8K is lm-eval 3-shot CoT.

## 2. Executable toy trace (tiny 2-layer Ouro, R=4; prompt A,B; generated C,D)

`trace_reads_writes.csv` lists every write and, for every attention query, the (layer, recurrence, token) version
it reads. The trace is produced with the independent dictionary reference. The engine reproduces the reference
logits exactly for each policy: {"FULL": 0.0, "LAST-CONVERT": 0.0, "LAST-DECODE (diagnostic)": 0.0, "PROTECT L0R1": 0.0}.

Decode-time reads (rN = the token's K/V from recurrence N; r4 = terminal):

| policy | query | layer | recurrence | reads A | reads B | reads C |
|---|---|---:|---:|---|---|---|
| FULL | C | 0 | 1 | r1 | r1 |  |
| FULL | C | 0 | 2 | r2 | r2 |  |
| FULL | C | 0 | 4 | r4 | r4 |  |
| FULL | C | 1 | 1 | r1 | r1 |  |
| FULL | C | 1 | 2 | r2 | r2 |  |
| FULL | C | 1 | 4 | r4 | r4 |  |
| FULL | D | 0 | 1 | r1 | r1 | r1 |
| FULL | D | 0 | 2 | r2 | r2 | r2 |
| FULL | D | 0 | 4 | r4 | r4 | r4 |
| FULL | D | 1 | 1 | r1 | r1 | r1 |
| FULL | D | 1 | 2 | r2 | r2 | r2 |
| FULL | D | 1 | 4 | r4 | r4 | r4 |
| LAST-CONVERT | C | 0 | 1 | r4 | r4 |  |
| LAST-CONVERT | C | 0 | 2 | r4 | r4 |  |
| LAST-CONVERT | C | 0 | 4 | r4 | r4 |  |
| LAST-CONVERT | C | 1 | 1 | r4 | r4 |  |
| LAST-CONVERT | C | 1 | 2 | r4 | r4 |  |
| LAST-CONVERT | C | 1 | 4 | r4 | r4 |  |
| LAST-CONVERT | D | 0 | 1 | r4 | r4 | r4 |
| LAST-CONVERT | D | 0 | 2 | r4 | r4 | r4 |
| LAST-CONVERT | D | 0 | 4 | r4 | r4 | r4 |
| LAST-CONVERT | D | 1 | 1 | r4 | r4 | r4 |
| LAST-CONVERT | D | 1 | 2 | r4 | r4 | r4 |
| LAST-CONVERT | D | 1 | 4 | r4 | r4 | r4 |
| LAST-DECODE (diagnostic) | C | 0 | 1 | r1 | r1 |  |
| LAST-DECODE (diagnostic) | C | 0 | 2 | r2 | r2 |  |
| LAST-DECODE (diagnostic) | C | 0 | 4 | r4 | r4 |  |
| LAST-DECODE (diagnostic) | C | 1 | 1 | r1 | r1 |  |
| LAST-DECODE (diagnostic) | C | 1 | 2 | r2 | r2 |  |
| LAST-DECODE (diagnostic) | C | 1 | 4 | r4 | r4 |  |
| LAST-DECODE (diagnostic) | D | 0 | 1 | r1 | r1 | r4 |
| LAST-DECODE (diagnostic) | D | 0 | 2 | r2 | r2 | r4 |
| LAST-DECODE (diagnostic) | D | 0 | 4 | r4 | r4 | r4 |
| LAST-DECODE (diagnostic) | D | 1 | 1 | r1 | r1 | r4 |
| LAST-DECODE (diagnostic) | D | 1 | 2 | r2 | r2 | r4 |
| LAST-DECODE (diagnostic) | D | 1 | 4 | r4 | r4 | r4 |
| PROTECT L0R1 | C | 0 | 1 | r1 | r1 |  |
| PROTECT L0R1 | C | 0 | 2 | r4 | r4 |  |
| PROTECT L0R1 | C | 0 | 4 | r4 | r4 |  |
| PROTECT L0R1 | C | 1 | 1 | r4 | r4 |  |
| PROTECT L0R1 | C | 1 | 2 | r4 | r4 |  |
| PROTECT L0R1 | C | 1 | 4 | r4 | r4 |  |
| PROTECT L0R1 | D | 0 | 1 | r1 | r1 | r1 |
| PROTECT L0R1 | D | 0 | 2 | r4 | r4 | r4 |
| PROTECT L0R1 | D | 0 | 4 | r4 | r4 | r4 |
| PROTECT L0R1 | D | 1 | 1 | r4 | r4 | r4 |
| PROTECT L0R1 | D | 1 | 2 | r4 | r4 | r4 |
| PROTECT L0R1 | D | 1 | 4 | r4 | r4 | r4 |

Checks:
- FULL reads per-recurrence history everywhere.
- LAST-CONVERT reads r4 for every historical token (prompt and generated) at every query recurrence.
- LAST-DECODE keeps the prompt per-recurrence and terminalises generated tokens only.
- PROTECT L0R1 differs from LAST-CONVERT only at (layer 0, recurrence 1).
- In every policy the current token's own K/V is its current-recurrence K/V (the `write` rows): writes happen
  during the token's own forward, and conversion happens only after its last recurrence. No future token is ever
  read.
- RoPE positions, causal mask and dtype are shared with FULL. `tests/test_banks.py` and
  `tests/test_policy_semantics.py` check these against the dictionary reference at 1e-9.

## 3. Native (official forward) vs reference FULL — `native_full_parity.csv`

4 GSM8K + 4 MATH500 pilot items per model; prompt + 64 gold tokens; 48-token greedy decoding.

| model | dtype | max abs logit diff (exit at step 4) | max KL | min top-1 | greedy-48 first divergence | default path == step-4 path |
|---|---|---:|---:|---:|---|---:|
| ouro-1.4b | float32 | 0.000464 | 4.6e-10 | 1.000 | none, none, none, none, none, none, none, none | 8/8 |
| ouro-1.4b | bfloat16 | 9 | 0.0085 | 1.000 | none, none, none, none, none, none, none, none | 8/8 |
| ouro-2.6b | float32 | 0.00021 | 4.4e-10 | 1.000 | none, none, none, none, none, none, none, none | 8/8 |
| ouro-2.6b | bfloat16 | 3 | 6.8 | 0.997 | none, none, none, 33, none, none, none, none | 0/8 |

- **FP32:** the reference FULL engine equals the official forward (|Δlogit| ≤ 5e-4, KL ≤ 5e-10). All 48-token greedy
  generations are identical.
- **Default exit path:** with `early_exit_threshold=1.0`, every position exits at step 4 in FP32.
- **BF16 exception (Ouro-2.6B):** at position 0 (the first prompt token) the gate saturates (λ₁ = 1.0 in BF16), so the
  official default path exits at step 1 and its logits differ there. This affects only that prompt position's
  logits. All four steps are still computed and cached for every position, so generation is unaffected.
- **BF16 numerical deviation:** max |Δlogit| up to about 9 for Ouro-1.4B with top-1 agreement 1.0, the known
  reduction-order noise documented in Stage 1.

## 4. Fixed B1 FP32 subset (4 GSM8K + 4 MATH500 per model, full caps) — `b1_fp32_subset.csv`

| model | FULL | LAST-CONVERT | LAST-DECODE (diagnostic) | PROTECT L0R1 |
|---|---:|---:|---:|---:|
| ouro-1.4b | 8/8 | 0/8 | 6/8 | 6/8 |
| ouro-2.6b | 7/8 | 5/8 | 7/8 | 6/8 |

Prompt token IDs, checkpoint revisions, stop rules and the parser are identical to E0 (`prompts/`, run manifests).
The B1 FP32 outcomes reproduce the batched BF16 pattern: Ouro-1.4B LAST-CONVERT collapses, and both the diagnostic
LAST-DECODE and L0R1 protection keep most items.

## 5. Limitations

- The original paper's Last policy cannot be reproduced from released code, and its prompt-KV handling is
  unspecified. Its Table 14 checkpoint and MATH prompt are also not verifiable.
- "Official default inference" is the official forward evaluated without its (unconstructible) cache. It is not a
  run of the official generate loop.
