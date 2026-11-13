# Source-verified benchmark reproduction map (2026-11-13)

| Paper/card | Official implementation | Models actually evaluated | Benchmarks and prompt convention | Dynamic per-token target exit? |
|---|---|---|---|---|
| `P-2609-17184` LoopSpec | https://github.com/kaist-flexml-lab/loopspec `experiments/{110,112,113}*.yaml`, `tasks.yaml` | Ouro-1.4B, Ouro-2.6B, Raven Llama/OLMo/TinyLlama checkpoints | GSM8K-CoT 3-shot max-gen 512; MATH-500 4-shot max-gen 2048; BBH 3-shot max-gen 1024; HumanEval+/MBPP+ zero-shot max-gen 1024; server max-concurrent 1; model configs baseline/one-proposal/two-proposal | **No**; fixed target R4/R32. Huginn and Dory not official evaluated models. |
| `P-2609-23033` WaveFront | https://github.com/summerbro-hhj/wavefront-decoding `experiments/exp2_spec_bench/run.sh`, `exp3_accuracy/run.sh` | Huginn-3.5B; Ouro 1.4B/2.6B; extra Parcae and thinking | Spec-Bench IDs 81–560 (480 prompts, six categories), max new tokens 512, max prompt 512; GSM8K and MATH-500 accuracy via exp3 (default GSM8K 8-shot, MATH500 4-shot); fixed R32 Huginn and R4 Ouro; BF16 greedy baseline | Primary reported method fixed; exp3 runner exposes `EXIT_THRESHOLD`/`EXIT_HIDDEN_EPS` as **separate options** requiring validated matching AR. |
| `P-2611-03998` Dory | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 ; https://github.com/vllm-project/vllm/pull/60149 | Dory1.0-4B | Paper steady-state decode benchmark R4, d1, K4, per-loop KV + activation reuse; global elastic budgets R1/2/4. Model accuracy benchmarks differ from LoopSpec/Spec-Bench. | **No verified native per-token exit** in our sources; do not create synthetic native label. |

## Source verification details

Read canonical Wiki paper cards on main: `references/papers/P-2609-17184.md`, `P-2609-23033.md`, `P-2611-03998.md`, original upstream YAML and shell configurations, arXiv abstracts. Source-reported experimental results remain paper-reported; no new reproduction here.

Specific critical mismatch: **LoopSpec is not the same as choosing `K` in DtV.** Its parameter `first:second` denotes proposal depth(s) and a pipelined execution path. WaveFront `SCHEDULER=wfd` ignores `DRAFT_LENGTH`; only its `dtv` uses the explicit `--draft-length` flag. Do not compare apparent K-optima across algorithms unless matching the physical action and cost definition.

Dory manuscript has selectable global loops per checkpoint. It does not demonstrate Huginn/Ouro-style native token-level live halting. It also changes outputs under last-loop vs per-loop KV; use separate matched AR baselines. If GPU-native per-token Dory exit is later implemented/trained, state that it's a **new exit policy**, verify its accuracy/target and label separately.

## Fair comparison / evidence hierarchy

1. Paper-style **fixed target** author reproduction within each author's runtime, don't compare raw wall across GPUs/backends.
2. Same-engine **fixed vs native live** with *each target's own matched AR*, same prompts, target semantics, generation cap; see strict parity.
3. Same-engine **AR vs DtV fixed-K** full rollouts as primary actionable frontier; WFD and LoopSpec as separate baselines. CPU-only one-round result `4f6e9c50` is not a real full-rollout controller.
4. Statistical evidence: paired prompt-bootstrap CI; minimum meaningful K gap >2% and CI excluding 1; show ties/plateaus, per-task and per-model differences, sample sizes.
5. Avoid task accuracy contamination and cherry-picked D tails; show prompt and exit histograms. Never quote a memory, latency or quality saving not physically reported.
