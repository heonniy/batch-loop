# v9: Context KV Eviction -> Recurrent FFN Gate -> Dynamic Exit / Answer

Date: 2026-11-03. Parent: v8b 6fac4edb.
Status: **implementation and CPU tests committed; GPU execution has NOT occurred in this authoring session**.

## Hypotheses (not desired conclusions)

H1: Keeping Context KV evicted but restoring only FULL-run recurrent FFN gate preactivations reduces the same-token and future output damage, beyond simply restoring the Up branch or a direction-shifted patch.
H2: Gate perturbations at different recurrent loops have different impacts on the final logits / generated-token history, beyond immediate attention-output changes. This is what would distinguish a loop-specific mechanism from generic dense-FFN perturbation.
H3: Context eviction changes native exit depths for future generated queries. Using model-specific CLEAN-only exit-threshold calibration, fixed-depth vs real live exit changes NLL, greedy answer correctness and total recurrent work; gate-restoration may modify the gate decision (oracle diagnostic).

No hypothesis is assumed true. v7 observed tiny one-loop KL near BF16 noise and no robust source-specific recurrent gain. A negative result is valid.

## Verified activation sites

Huginn: transformer.core_block[l].mlp.fc produces concatenated gate and up projections; replace only the first half (before SiLU). Do not patch Prelude/Coda FFNs.

Ouro-1.4B / Ouro-2.6B: model.layers[l].mlp.gate_proj and mlp.up_proj are separate modules; patch only gate_proj output to restore the pre-SiLU gate.

Hooks use an explicit (column, loop, physical-layer, component) key even though the core weights are reused across loops. Restore operators modify activations, not weights.

## Main paired causal arms (same LAST-sharing baseline)

FULL: complete clean prefill, all Context tokens readable, all historical recurrent reads redirected to terminal KV plane.

EVICT: after the SAME full prefill, mask a frozen contiguous Context source window (4 tokens initially) for all generated queries, all loop reads and plain-layer reads; new Generated KV changes persist.

EVICT + GATE: source still masked; for each generated query at each selected loop and layer, replace only the gated preactivation with the matched CLEAN value.

EVICT + UP: source still masked; replace only the other FFN multiplicative branch.

EVICT + SHIFTED-GATE: use a circularly shifted FULL−EVICT gate-difference vector of matched norm. This is a direction control, not a router switch.

EVICT + FFN: restore the entire FFN output as an upper intervention control.

EVICT + SELECTED-LOOP-GATE: restore only Huginn loops 1–8 / 9–16 / 17–24 / 25–32 or Ouro loops 1–2 / 3–4.

Capture FULL and EVICT activation traces by query, recurrent loop and core physical layer; output gate relative L2 change, gate sign-change fraction, FFN output change, final gold-token NLL and Full-to-Evicted KL. Restore-only on FULL with no mask MUST be parity. Output measures must exceed the BF16 no-mask floor before any mechanism claim.

Main source windows: four position-hash-stratified label-blind Context windows, same length. Relevant-evidence and irrelevant-distractor windows are separate annotation-dependent ORACLE diagnostics. Select before looking at eviction damage.

## Dynamic depth: actual gate re-evaluation, not fake replay

Clean-only calibration UIDs are disjoint from evaluation UIDs. Calibration replays full-depth queries with the exact same LAST-sharing historical-KV read map used for evaluation; it gathers exit statistics without performing eviction. Each model chooses two thresholds from a frozen grid, targeting approximately 25% and 60% of clean generated queries exiting early. Report actual depth histogram and fail closed if neither choice yields a useful distribution.

Ouro: native cumulative Q-exit gate, q in [0.25,0.40,0.55,0.70,0.85,0.95], min_depth=2.
Huginn: cosine-distance criterion with FP32 arithmetic over BF16 states, theta in [1e-4,2.5e-4,5e-4,1e-3,2e-3,4e-3,8e-3,1.6e-2], min_depth=2. This is a **FP32 diagnostic**; do not call it literal BF16 native exit.

Per threshold: FULL-FIXED, FULL-LIVE, FULL-REPLAY, EVICT-FIXED, EVICT-LIVE, EVICT-REPLAY and EVICT-GATE-REPLAY. Replay is forced from recorded FULL-LIVE depths; verify exact CLEAN live vs replay output parity. EVICT-GATE-LIVE uses FULL-FIXED clean gate activations at all depths and is explicitly an **off-policy oracle** for whether patching gates changes the native exit decision.

Answer outcome: compute real greedy generations for FULL-FIXED, FULL-LIVE, EVICT-FIXED and EVICT-LIVE separately. Report answer/EM flips in both directions, capped/truncated outputs, actual recurrence depth traces and total work. Do not infer answer quality from teacher-forced NLL. Live-minus-replay does not identify an exact mediation effect, because generated history changes.

## Reproduction commands (after checking allocated H200 GPU)

Working directory: kv_policy

    python -m pytest -q tests/test_ffn_gate_causal.py

    CUDA_VISIBLE_DEVICES=0 python scripts/ffn_gate_causal_sweep.py --calibrate --model huginn-0125 --dataset controlled_retrieval --run-dir results/v9_huginn_cr --n-cal 3 --n-eval 5 --width 4

    CUDA_VISIBLE_DEVICES=0 python scripts/ffn_gate_causal_sweep.py --execute --model huginn-0125 --dataset controlled_retrieval --run-dir results/v9_huginn_cr

    python scripts/ffn_gate_causal_sweep.py --analyze --model huginn-0125 --dataset controlled_retrieval --run-dir results/v9_huginn_cr

Repeat under a **new run directory** for Ouro-2.6B and Ouro-1.4B, and HotpotQA-Distractor. Each GPU process loads one checkpoint once and runs B=1. The existing v5 frozen prompt/target jobs are required.

Sufficient first pilot: 3 clean-only calibration prompts + 5 evaluation prompts per model/dataset; report actual runtime/memory and BF16 floor. Grow to 20 or 100 independent prompts only after the mechanism signal survives gate/up/shift controls and numerical checks. The coding agent can add new width-8 windows, longer W1/W2 free-form continuations or GSM8K, but MUST freeze separate protocol directories, UID splits and journal rationale.

## Stop / go

- Gate patch ≈ Up or shift patch: generic FFN/attention perturbation; not a MoEZip-like special mechanism.
- Gate patch improves NLL but exit/answer do not change: causal FFN contribution is possible, not yet loop-specific.
- Gate repairs depth/answer only under single threshold cherry-picking: not robust; reject.
- Gate repair works at particular recurrent stages across both models / holdout and beats controls: follow up with standard-transformer layer-wise FFN intervention, stronger matching and cheap available predictor.
- Physical KV saving, HBM/TPOT or deployment are NOT measured by the full-sized reference.

This authoring environment can commit GitHub files but cannot execute GPU shell commands on the H200 host. Results should not be claimed before run receipts are published.
