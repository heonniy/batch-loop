# Prior work, transfer boundaries, and required reading before branching

**Reading rule:** Inspect the exact GitHub Research Wiki card and primary paper for a selected method before editing the implementation; record `paper_id`, `claim_id`, checked URL/version, mechanistic equivalence, and caveat in a method-specific `EVIDENCE_NOTE.md` or result receipt. A paper's reported improvement on diffusion/other Transformers **is not proof for frozen Huginn/Ouro**.

Start: [C-PAR-001](https://github.com/heonniy/research-wiki/blob/main/concepts/C-PAR-001.md), [RQ-PAR-001](https://github.com/heonniy/research-wiki/blob/main/questions/RQ-PAR-001.md), [H-PAR-001](https://github.com/heonniy/research-wiki/blob/main/hypotheses/H-PAR-001.md), [F-091–095](https://github.com/heonniy/research-wiki/tree/main/findings), [Reference index](https://github.com/heonniy/research-wiki/blob/main/references/by-target/C-PAR-001.md), [Research evidence](https://github.com/heonniy/batch-loop/blob/829076bb/parallel_depth_poc/RESULTS.md).

| Branch | References (wiki ID; primary arXiv) | What they motivate | Mandatory caveat |
|---|---|---|---|
| Same-token trained predictor/corrector | LoopFold `P-2610-35556`; [2610.35556](https://arxiv.org/abs/2610.35556) | Learned J_pred/J_corr, fixed stage/width | Joint from-scratch training; not a frozen-model inference baseline |
| Triangular Jacobi/Picard | `P-2002-03629`; [2002.03629](https://arxiv.org/abs/2002.03629) | Solve h_r=F_r(h_{r-1}) by parallel fixed-point iteration | O(D) exact propagation; no cheap online stopping |
| Diffusion Picard | `P-2305-16317`; [2305.16317](https://arxiv.org/abs/2305.16317) | Future-state speculation + correct | Different diffusion steps, KV not causal LM |
| Triangular Anderson | `P-2402-09970`; [2402.09970](https://arxiv.org/abs/2402.09970) | Stable residual-history extrapolation / ParaTAA | Their iteration count not transferable; compare real solver overhead |
| Block Parareal | `P-2412-08292`; [2412.08292](https://arxiv.org/abs/2412.08292) | Time blocks with cheap coarse map G | Prior E3 identity/one-step coarse often too expensive |
| Structured Newton | `P-2605-17842`; [2605.17842](https://arxiv.org/abs/2605.17842) | Triangular correction e_r + A_r delta_{r-1}, inexpensive A | Trained standard layers; frozen success not established |
| Quasi-Newton RNN | `P-2309-12252`, `P-2407-19115`; [2309.12252](https://arxiv.org/abs/2309.12252), [2407.19115](https://arxiv.org/abs/2407.19115) | Structured Jacobian approximations, stability | Across token time, not depth-indexed KV |
| Parallel cyclic reduction / scans | `P-2309-16318`, `P-2511-01465` | Log-depth composition for affine/linearized recurrences | Only valid if linearization is accurate and cheap |
| Dynamics and predictability | `P-2508-16817`, wiki `C-REC-003`, `F-094` | Directional Jacobian gain, Lyapunov conditioning, stability checks | One finite perturbation is not contraction or LLE |
| Cross-token iterative decoding | `P-2402-02057`, `P-2512-14681` | Rejection/target verification and history | Future *token* prediction, not same-token depth |
| Architecture alternatives | `P-2510-24824` PLT; `P-2510-14961` parallel samplers; WaveFront/LoopSpec wiki cards | Upper neighboring systems baselines | Architecture changes or token/depth diagonal scheduling |

## Mandatory method review template
For each adaptive candidate, before adding `implemented` status:
1. Exact primary paper section, equation, supported claim and source commit/version.
2. Map its operators/states to `F_r(h;KV_r)`; list mismatched assumptions.
3. Complexity: core call width x count, Coda, KV writes, temporary HBM, synchronization, cross-device traffic.
4. Predict *observable failure mode* and negative control; implement regression tests of that failure.
5. State train-free or post-training; frozen-weight constraint; data leakage / test split.
6. Record pilot outcomes, alternative hypotheses and reason for promotion or further refinement. Do not invent novelty for Jacobi, Anderson, Newton or selective correction alone.

**Important methodological distinction:** residual `||z^k-z^{k-1}||` is not a certified bound on `||logits(z^k)-logits(AR)||`. Even correct next-token argmax need not imply safe KV; rate limits and Lipschitz assumptions require direct validation.
