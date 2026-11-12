#!/bin/bash
# Generates fu_q/g0..g7 for the 2026-11-12 follow-up run:
#   Stage 0a shadow gate (off/gpu/cpu, 5 repeats, FP32 and BF16, Ouro tau0.4 / Huginn eps0.1)
#   Stage 0c FP32 FULL-KV parity (fixed + live, both models)
#   Stage 1  same-prefix counterfactual action trials (oracle input)
# Outputs: dynamic_sd/results/followup_20261112/ (gitignored; receipts copied for commit later).
cd "$(dirname "$0")/.."   # dynamic_sd/
Q=fu_q; mkdir -p $Q; rm -f $Q/g*
OUT=results/followup_20261112
R="\$PY dsd_run.py"
S="\$PY followup_20261112/gate_shadow.py"
C="\$PY followup_20261112/counterfactual_actions.py"
mo() { echo "--model ouro-1.4b --dataset gsm8k --T-total 4 --T-draft 1 --K 4 --dtype $1 --kv-s 0 --warmup 2"; }
mh() { echo "--model huginn-3.5b --dataset gsm8k --T-total 32 --T-draft 4 --K 4 --dtype $1 --kv-s 0 --warmup 2"; }
ref() { echo "$R $1 --limit 32 --exit live $2 --arms ar,wfd,dtv --out $3.tmp && mv $3.tmp $3"; }
shadow() {  # $1=model args $2=ref $3=tag $4..=reps ; mode order randomized per repeat
  local a=$1 f=$2 t=$3; shift 3
  for r in "$@"; do
    for m in $(printf 'off\ngpu\ncpu\n' | shuf --random-source=<(yes $t$r)); do
      echo "$S --shadow-mode $m $a --limit 16 --exit replay --replay-from $f --arms ar,wfd,dtv --out $OUT/shadow/${t}_r${r}_$m.json"
    done
  done; }
mkdir -p $OUT/ref $OUT/shadow $OUT/parity $OUT/cf
O32=$OUT/ref/ouro_live_tau0.4_float32_n32.json;   O16=$OUT/ref/ouro_live_tau0.4_bfloat16_n32.json
H32=$OUT/ref/huginn_live_eps0.1_float32_n32.json; H16=$OUT/ref/huginn_live_eps0.1_bfloat16_n32.json
OACT="AR,DtV:d1:K1,DtV:d1:K2,DtV:d1:K4,DtV:d1:K8,DtV:d2:K1,DtV:d2:K2,DtV:d2:K4,DtV:d2:K8"
HACT="AR,DtV:d2:K2,DtV:d2:K4,DtV:d2:K8,DtV:d4:K2,DtV:d4:K4,DtV:d4:K8,DtV:d8:K2,DtV:d8:K4,DtV:d8:K8"
OCF="--model ouro-1.4b --T-total 4 --dataset gsm8k --offset 600 --limit 32 --actions $OACT"
HCF="--model huginn-3.5b --T-total 32 --dataset gsm8k --offset 600 --limit 24 --actions $HACT"

{ ref "$(mo float32)" "--tau 0.4" $O32;  shadow "$(mo float32)" $O32 ouro_float32 0 1 2 3 4; echo "#END"; } > $Q/g0
{ ref "$(mo bfloat16)" "--tau 0.4" $O16; shadow "$(mo bfloat16)" $O16 ouro_bfloat16 0 1 2 3 4; echo "#END"; } > $Q/g1
{ ref "$(mh float32)" "--eps 0.1" $H32;  shadow "$(mh float32)" $H32 huginn_float32 0 1 2; echo "#END"; } > $Q/g2
{ ref "$(mh bfloat16)" "--eps 0.1" $H16; shadow "$(mh bfloat16)" $H16 huginn_bfloat16 0 1 2; echo "#END"; } > $Q/g3
{ echo "$R $(mo float32) --limit 16 --exit live --tau 0.1 --arms ar,wfd,dtv --out $OUT/parity/ouro_live_tau0.1_float32.json"
  echo "$R $(mo float32) --limit 16 --exit fixed --arms ar,wfd,dtv --out $OUT/parity/ouro_fixed_float32.json"
  echo "$R $(mh float32) --limit 16 --exit fixed --arms ar,wfd,dtv --out $OUT/parity/huginn_fixed_float32.json"
  echo "$R $(mh float32) --limit 16 --exit live --eps 0.2 --arms ar,wfd,dtv --out $OUT/parity/huginn_live_eps0.2_float32.json"
  echo "WAIT $H32"; shadow "$(mh float32)" $H32 huginn_float32 3 4; echo "#END"; } > $Q/g4
{ echo "$C $OCF --exit live --tau 0.4 --out $OUT/cf/ouro_tau0.4.jsonl"
  echo "$C $OCF --exit live --tau 0.1 --out $OUT/cf/ouro_tau0.1.jsonl"
  echo "WAIT $H16"; shadow "$(mh bfloat16)" $H16 huginn_bfloat16 3 4; echo "#END"; } > $Q/g5
{ echo "$C $OCF --exit fixed --out $OUT/cf/ouro_fixed.jsonl"
  echo "$C $HCF --exit fixed --out $OUT/cf/huginn_fixed.jsonl"; echo "#END"; } > $Q/g6
{ echo "$C $HCF --exit live --eps 0.1 --out $OUT/cf/huginn_eps0.1.jsonl"
  echo "$C $OCF --exit live --tau 0.4 --dtype bfloat16 --out $OUT/cf/ouro_tau0.4_bf16.jsonl"; echo "#END"; } > $Q/g7
wc -l $Q/g*
