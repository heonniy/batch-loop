### shadow lane `ouro_float32`: matched repeats ['0', '1', '2', '3', '4']; refused []
| arm | off ms/tok | gpu−off ms/tok [95% CI] | % of off | cpu−gpu ms/tok [95% CI] | % of off | cpu−off % |
|---|---:|---:|---:|---:|---:|---:|
| ar | 27.17 | -0.324 [-0.684, +0.012] | -0.97% | +0.200 [+0.040, +0.412] | +0.74% | -0.23% |
| wfd | 21.36 | -0.177 [-0.399, +0.064] | -0.67% | +0.198 [-0.002, +0.414] | +0.84% | +0.17% |
| dtv | 21.65 | -0.126 [-0.341, +0.101] | -0.43% | +0.098 [-0.122, +0.293] | +0.44% | +0.01% |

cpu-mode shadow: 131405 gate calls, 213590 rows; host time inside shadow 98.9 us/call, of which blocking readback 39.3 us/call (gpu-mode host time 59.6 us/call)

### shadow lane `ouro_bfloat16`: matched repeats ['0', '1', '2', '3', '4']; refused []
| arm | off ms/tok | gpu−off ms/tok [95% CI] | % of off | cpu−gpu ms/tok [95% CI] | % of off | cpu−off % |
|---|---:|---:|---:|---:|---:|---:|
| ar | 31.76 | -0.192 [-0.444, +0.065] | -0.53% | +0.672 [+0.384, +0.936] | +2.09% | +1.57% |
| wfd | 19.60 | -0.227 [-0.412, -0.018] | -1.04% | +0.476 [+0.236, +0.713] | +2.36% | +1.32% |
| dtv | 22.84 | -0.132 [-0.385, +0.108] | -0.47% | +0.468 [+0.225, +0.702] | +1.99% | +1.52% |

cpu-mode shadow: 129575 gate calls, 209170 rows; host time inside shadow 97.0 us/call, of which blocking readback 27.8 us/call (gpu-mode host time 67.8 us/call)

### shadow lane `huginn_float32`: matched repeats ['0', '1', '2', '3', '4']; refused []
| arm | off ms/tok | gpu−off ms/tok [95% CI] | % of off | cpu−gpu ms/tok [95% CI] | % of off | cpu−off % |
|---|---:|---:|---:|---:|---:|---:|
| ar | 49.19 | +0.091 [-0.098, +0.297] | +0.20% | +2.878 [+2.646, +3.108] | +5.85% | +6.05% |
| wfd | 38.19 | +0.218 [+0.126, +0.297] | +0.56% | +1.207 [+1.135, +1.278] | +3.17% | +3.73% |
| dtv | 39.57 | +0.390 [+0.245, +0.562] | +0.94% | +1.178 [+0.970, +1.408] | +3.00% | +3.94% |

cpu-mode shadow: 750145 gate calls, 1446890 rows; host time inside shadow 665.5 us/call, of which blocking readback 586.9 us/call (gpu-mode host time 79.3 us/call)

### shadow lane `huginn_bfloat16`: matched repeats ['0', '1', '2', '3', '4']; refused []
| arm | off ms/tok | gpu−off ms/tok [95% CI] | % of off | cpu−gpu ms/tok [95% CI] | % of off | cpu−off % |
|---|---:|---:|---:|---:|---:|---:|
| ar | 41.00 | +1.880 [+1.612, +2.143] | +4.62% | +0.819 [+0.510, +1.148] | +2.02% | +6.64% |
| wfd | 23.74 | +0.715 [+0.618, +0.813] | +3.02% | +0.160 [+0.031, +0.286] | +0.69% | +3.71% |
| dtv | 27.51 | +1.055 [+0.883, +1.212] | +3.89% | +0.420 [+0.207, +0.668] | +1.53% | +5.41% |

cpu-mode shadow: 756495 gate calls, 1468205 rows; host time inside shadow 105.9 us/call, of which blocking readback 24.1 us/call (gpu-mode host time 81.1 us/call)

