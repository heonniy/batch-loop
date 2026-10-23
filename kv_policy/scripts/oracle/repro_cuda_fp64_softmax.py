"""Minimal reproducer: CUDA float64 softmax over a long last dim disagrees with CPU float64.

Found while building the Stage-1 FP64 ground truth: the native Huginn forward on GPU in float64 gave
logits ~4 nats away from FP32 at every position >= 31 of an 837-token prompt. CUDA FP32/BF16
softmax and SDPA kernels agree with CPU float64, so only the GPU-FP64 reference is invalid; the
Stage-1 FP64 reference is therefore computed on CPU (scripts/oracle/s1_parity.py, job "fp").
"""

import torch

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
for S in (31, 32, 100, 512, 837, 1024, 2000):
    torch.manual_seed(S)
    x = torch.randn(1, 55, S, S, dtype=torch.float64)
    x = x.masked_fill(~torch.ones(S, S, dtype=torch.bool).tril(), float("-inf"))
    cpu = torch.softmax(x, -1)
    g64 = torch.softmax(x.cuda(), -1).cpu()
    g32 = torch.softmax(x.float().cuda(), -1).double().cpu()
    bad = ((g64 - cpu).abs().amax((0, 1, 3)) > 1e-6).nonzero().flatten()
    print(f"S={S:5d}  cuda fp64 vs cpu fp64 max err {(g64 - cpu).abs().max().item():.2e} "
          f"(first bad row {bad[:1].tolist()})  cuda fp32 vs cpu fp64 {(g32 - cpu).abs().max().item():.2e}")
