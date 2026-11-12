# GPU-host runbook (commands are illustrative, NOT executed)

Requires checkout of this PR branch, prior WaveFront source at pinned `434520932897fcc1e26898891c7e2feaa6ab099f`, model weights, dataset access, CUDA-capable PyTorch, GPU isolation and experimental approval.

Set:
```bash
export WF_ROOT=/abs/path/wavefront-decoding
export OUT=dynamic_sd/results/followup_20261112
mkdir -p "$OUT"
python -m unittest discover -s dynamic_sd/followup_20261112/tests -v
```

## 0. Obtain native AR-live reference

Use the **same** model, prompt pool and FULL KV for every following arm:
```bash
CUDA_VISIBLE_DEVICES=0 python dynamic_sd/dsd_run.py \
  --model ouro-1.4b --dataset gsm8k --limit 32 \
  --T-total 4 --T-draft 1 --K 4 --exit live --tau 0.4 \
  --arms ar --dtype float32 --kv-s 0 --warmup 2 \
  --out "$OUT/ouro_ar_live.json"
```
For Huginn use `--model huginn-3.5b --T-total 32 --T-draft 4 --eps 0.1`, first confirming the actual native exit definition. `--dtype float32` is a diagnostic; BF16 must be reported separately. Do not interpret the AR-live trace as the true counterfactual exit trajectory for speculative descendants.

## 1. Matched replay, three shadow conditions

```bash
for MODE in off gpu cpu; do
 CUDA_VISIBLE_DEVICES=0 python dynamic_sd/followup_20261112/gate_shadow.py \
   --shadow-mode "$MODE" \
   --model ouro-1.4b --dataset gsm8k --limit 32 \
   --T-total 4 --T-draft 1 --K 4 --exit replay \
   --replay-from "$OUT/ouro_ar_live.json" \
   --arms ar,wfd,dtv --dtype float32 --kv-s 0 --warmup 2 \
   --out "$OUT/ouro_shadow_${MODE}.json"
done
python dynamic_sd/followup_20261112/compare_shadow.py \
 --off "$OUT/ouro_shadow_off.json" --gpu "$OUT/ouro_shadow_gpu.json" \
 --cpu "$OUT/ouro_shadow_cpu.json" \
 --out "$OUT/ouro_shadow_comparison.json"
```
Use the analogous Huginn lane on another reserved GPU. Run each mode >=5 times with interleaved/randomized order; use independent output paths. The script refuses to label a pair matched if the output sequence or executed work differs. `gate_shadow.py` shadows internal `Instr.orig` and keeps `Instr.replay` responsible for exits, which makes the change non-intervening **provided output/geometry checks pass**. Timings retain the original prefill-inclusive timer and eager overhead; GPU/cpu shadow costs are not the native gate's physical standalone latency.

Native Live, Replay and Shadow must be labeled separately. **No physical vllm-rlt / CUDA Graph speedup is tested** by these commands.

## 2. Stage 1 Oracle: create explicit branch-point counterfactuals

Prepare one JSONL line per `(prefix_id, action, repeat)` from an actual *reset-and-run* action experiment. Required fields:
```json
{"prefix_id":"p0","prefix_hash":"sha256:...","target_hash":"sha256:...","exit_policy":"ouro:tau0.4","kv_policy":"FULL","batch_regime":"B1","backend":"wavefront-eager","precision":"fp32","action":"AR","wall_ms":8.0,"committed_tokens":1,"target_exact":true,"measured":true,"repeat":0}
```
`target_hash` identifies the native target/reference semantics; `prefix_hash` must identify the identical **committed** state and token history. A unique `action` includes `d` and `K` if applicable, e.g. `WFD:d1`, `DtV:d1:K4`. **Do not manufacture this file from the existing cross-prompt aggregated E1 table**, and do not use a future exit depth as a deployable predictor.

```bash
python dynamic_sd/followup_20261112/oracle_frontier.py \
  --input "$OUT/counterfactual_actions.jsonl" \
  --out "$OUT/oracle_frontier.json"
```
This script validates paired prefixes, target contract and completeness; it produces a one-step optimistic upper-bound screen, not a runnable online controller or a measured deployment speedup.

## 3. Before scheduling and policy experiments

- Pass FP32 fixed and **live** FULL-KV parity for each model in the chosen backend; diagnose first BF16 mismatch separately.
- Introduce true CUDA Events/NVTX timings for exit-gate kernel, CPU readback, Core/Prelude/Coda, and isolate decode from prefill. Verify same recurrent-call shapes, masks and device writes between A/B. Ensure gated and ungated paths use the same CUDA Graph capture strategy when porting.
- Use no-refill + lockstep as *one* baseline, not the claimed production target. Compare against depth-asynchronous execution before asserting new scheduling gains.
- Publish smoke n32, pilot n100 and main n200 only when their stage gates pass; record raw receipts. No GPU execution has been requested from this chat.
