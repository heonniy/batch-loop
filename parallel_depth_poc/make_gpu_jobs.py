#!/usr/bin/env python3
"""Pre-registered GPU job list, run 20261116_gpu (printed slowest-first, one shell command per line).

E0  FP32 native parity vs HF author code: 16 prompts x 32 greedy tokens, L=256.
      huginn D 8,16,32 | ouro-1.4b D 2,3,4 | ouro-2.6b D 2,3,4
E1  BF16 oracle width headroom, B in {1,8,16,64,128}, L in {256,1024}, widths 1..D (powers of 2 + D).
      huginn D=32 | ouro-1.4b D=4 | ouro-2.6b D=4      (one job per model x L)
E2/E3/E4 online diagnostics (depth_diag), 64 decode steps, greedy, FULL depth-indexed KV:
      bf16_b16_L256 : B=16, L=256 (GSM8K 3-shot, left-truncated)     all 9 (model, D) cells
      bf16_b16_L1024: B=16, L=1024 (synthetic long: MATH 4-shot + GSM8K 8-shot)   huginn32, ouro D4
      bf16_b1_L256  : B=1 (batch-shape lane), prompts 0..3 separately           huginn32, ouro D4
      fp32_b4_L256  : FP32, B=4, 16 decode steps (exactness + small-eps perturbation)  huginn32, ouro D4
Ouro depths never exceed the native cap 4. Huginn depth 32 = its default inference depth.
"""
OUT = "parallel_depth_poc/results/20261116_gpu/raw"
PY = "$PY -m parallel_depth_poc"
CELLS = [("huginn-3.5b", d) for d in (8, 16, 32)] + [("ouro-1.4b", d) for d in (2, 3, 4)] + \
        [("ouro-2.6b", d) for d in (2, 3, 4)]
DEEP = [("huginn-3.5b", 32), ("ouro-1.4b", 4), ("ouro-2.6b", 4)]
short = {"huginn-3.5b": "h", "ouro-1.4b": "o14", "ouro-2.6b": "o26"}

jobs = []   # (est_minutes, cmd)
for m, d in DEEP:
    for L in (1024, 256):
        jobs.append((25 if m.startswith("h") else 10,
                     f"{PY}.e1_oracle_width --model {m} --depth {d} --prompt-lens {L} "
                     f"--out {OUT}/e1/{short[m]}_D{d}_L{L}.json --allow-gpu"))
jobs.append((20, f"{PY}.e0_native_parity --model huginn-3.5b --depths 8,16,32 --out {OUT}/e0/h.json --allow-gpu"))
jobs.append((5, f"{PY}.e0_native_parity --model ouro-1.4b --depths 2,3,4 --out {OUT}/e0/o14.json --allow-gpu"))
jobs.append((5, f"{PY}.e0_native_parity --model ouro-2.6b --depths 2,3,4 --out {OUT}/e0/o26.json --allow-gpu"))
for m, d in CELLS:
    est = {8: 15, 16: 35, 32: 90}.get(d, 8) if m.startswith("h") else 8
    jobs.append((est, f"{PY}.depth_diag --model {m} --depth {d} --batch 16 --prompt-len 256 --cohort gsm8k3 "
                      f"--out {OUT}/diag/bf16_b16_L256/{short[m]}_D{d} --allow-gpu"))
for m, d in DEEP:
    jobs.append((120 if m.startswith("h") else 12,
                 f"{PY}.depth_diag --model {m} --depth {d} --batch 16 --prompt-len 1024 --cohort long "
                 f"--out {OUT}/diag/bf16_b16_L1024/{short[m]}_D{d} --allow-gpu"))
    for off in range(4):
        jobs.append((40 if m.startswith("h") else 4,
                     f"{PY}.depth_diag --model {m} --depth {d} --batch 1 --prompt-len 256 --offset {off} "
                     f"--out {OUT}/diag/bf16_b1_L256/{short[m]}_D{d}_p{off} --allow-gpu"))
    jobs.append((30 if m.startswith("h") else 4,
                 f"{PY}.depth_diag --model {m} --depth {d} --batch 4 --prompt-len 256 --decode-steps 16 "
                 f"--dtype float32 --out {OUT}/diag/fp32_b4_L256/{short[m]}_D{d} --allow-gpu"))

for est, cmd in sorted(jobs, key=lambda x: -x[0]):
    print(cmd)
