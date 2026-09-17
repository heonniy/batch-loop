# Phase-04R BF16 divergence cause review — 2026-09-17

## Decision: Branch B for the observed failures; B2 contract recommended, not yet active

The saved BF16 Gate-A decision disagreements were reproduced from exact tokenized inputs, prior live history and Huginn h0 identities. The first differing operator for each selected model counterexample was isolated with the same target input and the current backend. This establishes a **batch-shape numerical mechanism** for those counterexamples. It does not certify every possible row/KV implementation path, and it does not turn the original Gate-A FAIL into PASS. No base-model fix, threshold change, predictor collection or speed claim was made.

| Model and saved target | First BF16 divergence | Independent current-backend reproduction |
| --- | --- | --- |
| Ouro request 22/token 6 | During **prefill**, first layer `model.layers.0.self_attn.q_proj` (event 2). Target input is bitwise equal; B1 versus N32 BF16 output max-abs 0.03125, relative L2 0.0001462. This precedes any attention KV read or live-exit decision. | Reinvoking the same `Linear` on captured B1 and packed inputs reproduces both original outputs exactly. [`per_layer_receipt.json`](../20260917T055047Z_8e6ce7b_bf16_resolution/ouro_prefill_bf16_N32/per_layer_receipt.json), `isolated_operator_inputs.pt` beside it. |
| Huginn request 7/token 0 | All **768 captured prefill** module outputs are identical for the target. Prelude, h0 and the first core-block path agree. First decode difference is the input of `transformer.core_block.1.attn.proj` (event 37): the preceding SDPA math output differs by max-abs 9.54e-7, relative L2 3.46e-8. | Target Q, K, V and attention bias are bitwise equal. An isolated call to the same `SDPBackend.MATH` operation reproduces both shape-dependent outputs exactly. [`attention_operator_receipt.json`](../20260917T055813Z_b337ed3_bf16_resolution/huginn_sdpa_bf16_r7_t0/attention_operator_receipt.json), `isolated_attention_inputs.pt` beside it. |

The Huginn prefill full-path receipt is [`per_layer_receipt.json`](../20260917T055047Z_8e6ce7b_bf16_resolution/huginn_prefill_bf16_N32_full/per_layer_receipt.json); its 768/768 captured target outputs match. The first decode-module receipt is [`per_layer_receipt.json`](../20260917T055507Z_682a9e7_bf16_resolution/huginn_decode_bf16_r7_t0/per_layer_receipt.json). The attached Huginn attention wrapper, not an altered model implementation, supplied the current-backend Q/K/V/mask and output. Exact imported model-source hashes and CUDA UUIDs are in the receipts.

FP32 is a diagnostic rather than an exact-bitwise oracle: Ouro's first observed FP32 prefill difference is later (`layers.0.mlp.gate_proj`, max-abs 6.20e-6, relative L2 7.59e-7); Huginn's FP32 decode first difference is `prelude.0.attn.Wqkv` (max-abs 1.07e-6, relative L2 3.83e-7). The matched N32 FP32 Gate-A control and the frozen-history FP32 matrix have no observed decision flips. The FP32 Huginn SDPA receipt has small Q/K/V differences; therefore it is not evidence of bitwise FP32 invariance.

## Frozen-history shape/order controls

The old Gate-A mismatch set (Ouro 2 request-token positions; Huginn 9 distinct positions containing 7 depth and 3 next-token field disagreements) plus two far-margin controls per model was frozen in the [`Ouro`](../20260917T055047Z_8e6ce7b_bf16_resolution/ouro_counterexamples/COUNTEREXAMPLE_MANIFEST.json) and [`Huginn`](../20260917T055047Z_8e6ce7b_bf16_resolution/huginn_counterexamples/COUNTEREXAMPLE_MANIFEST.json) packets. They include all 32 effective input IDs/masks and hashes, previous known token IDs, target history signatures, h0 receipts where applicable, gate trajectories/margins and active-row order. Every saved decision disagreement and reference gate trajectory was replayed exactly before these packets were accepted. The first Ouro freeze attempt failed in the manifest writer because Ouro has no h0 field; its failure receipt is retained and the corrected source was committed before the successful runs.

For R2, each target and each unique filler entered the current token with a **frozen per-request B1 live history**. The merged target/filler KV was checked bitwise. No depth was forced. Results are diagnostic grid rows, not independent samples:

| Target, fixed-history BF16 | B1 | B2 | B4 | B8 | FP32 control |
| --- | --- | --- | --- | --- | --- |
| Ouro request 22/token 6 | depth 2 | depth 2 | depth 1 | depth 2 | depth 1 for all measured B; no within-FP32 flip |
| Huginn request 7/token 0 | next token 32 | 18370 | 18370 | 18370 | 18370 for all measured B; no within-FP32 flip |

At a given B, target position and fixed filler permutation did not alter these *discrete* decisions in the measured grid. Huginn target h0 hashes matched B1 across legal groupings. Cloned-target controls (distinct bookkeeping IDs, exact copied input/history and explicit same h0 for Huginn) matched in Prelude embedding, gate scores, depth, next token and final hidden. These checks disfavor the tested row/KV contamination mechanisms; they do not prove all implementations bug-free. The original Ouro B8 exit flip does **not** have to reappear in this R2 B8 cell: R2 freezes B1 histories, whereas the original compact run already accumulated its own batch-shaped prefill and prior-token state.

The frozen 32-input cohort lacks enough *distinct* same-length fillers for some cells: Ouro equal-length B8 and Huginn equal-length B4/B8 are explicitly BLOCKED in the matrices, never filled by duplicated prompts. See the [`Ouro`](../20260917T060412Z_7e8c7a1_bf16_resolution/ouro_matrix_bf16_r22_t6/batch_shape_matrix.csv) and [`Huginn`](../20260917T060412Z_7e8c7a1_bf16_resolution/huginn_matrix_bf16_r7_t0/batch_shape_matrix.csv) grids and duplicate controls beside them. Earlier matrix-harness failures (insufficient fillers and backend-only token initialization) are preserved in their failed roots; they were fixed in source commits before the successful matrix.

## Fresh numerical validation, predeclared before execution

The [`protocol`](../../BF16_NUMERICAL_VALIDATION_PROTOCOL_20260917.md) fixed 32 new synthetic inputs, eight token positions, B2/4/8, FIFO plus permutation seeds 17 and 29, and absolute reference-margin bins before opening the cohort. Effective token IDs are 32/32 unique and disjoint from the saved Gate-A cohort. Each condition has 256 same-input request-token comparisons, live gates, and its own actual work/RIC. Orderings reuse the 256 requests; the nine rows are **not** nine independent cohorts.

| Model | Depth flips per condition | Next-token flips per condition | B1 recurrent work → actual live work | Unique flipped positions across all nine conditions |
| --- | ---: | ---: | ---: | ---: |
| Ouro | 2–3 / 256 (0.78–1.17%) | 0–3 / 256 | 636 → 637–639 | 4 depth; 4 next-token |
| Huginn | 2–7 / 256 (0.78–2.73%) | 3–4 / 256 | 2118 → 2114–2118 | 9 depth; 5 next-token |

Ouro's four unique depth-flipped positions include **one** whose B1 minimum exit margin is 0.0107681, outside the prospectively declared 0.005 near-threshold bin. Huginn's nine unique depth-flipped positions are all within 0.005, but four of its five unique next-token-flipped positions are outside that exit-margin bin. Ouro's four unique next-token-flipped positions have exit margins above 0.10; exit margin does not bound argmax stability. This is why an exit-margin-only B1 stable-region exception is not supported by this validation. The original BF16 FAILs remain FAILs.

Actual live RIC varies with grouping and work: Ouro B2 341–343, B4 181–182, B8 91–94; Huginn B2 1089–1090, B4 557–559, B8 287–289. The corresponding *frozen B1-depth grouping RIC* is a separate saved-label calculation in each row; comparing it to live RIC as an equal-work Oracle is invalid. No synchronized/warmed wall-time strategy comparison was run. Raw [`Ouro`](ouro_fresh_validation/) and [`Huginn`](huginn_fresh_validation/) `per_token_validation.csv`, `margin_bin_summary.csv`, `batch_shape_validation.csv` and manifests retain every positive and negative condition.

## Branch choice and stop

Branch A is not supported for the **first observed differences**: they reproduce in isolated standard model operators with equal logical target input (and Huginn equal attention Q/K/V/mask). Branch B, batch-shape numerical dependence, is supported for these failures. Because a predeclared narrow stable-margin exception misses at least one Ouro depth flip and exit margin cannot account for next-token flips, **B2 (batch-dependent-depth contract) is the recommended research direction**. It must explicitly recast B1 depths as canonical/frozen labels and evaluate practical scheduling under actual induced live work. This recommendation is not itself approval to replace the research contract.

[`PROPOSED_B2_CONTRACT.md`](PROPOSED_B2_CONTRACT.md) is deliberately a draft. No new Gate A was run under it, no predictor phase reopened, and no FIFO/Oracle speed result was produced. The next bounded step requires a research decision on the contract, then a *prospectively committed* release rule and a fresh, disjoint N32/B8/T8 confirmation. Do not change q/threshold, checkpoint, KV policy, Rmax or primary BF16 precision as a hidden fix.
