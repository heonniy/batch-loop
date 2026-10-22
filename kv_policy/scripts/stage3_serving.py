"""Stage 3: lightweight vllm-rlt serving validation (fixed depth, synchronous, eager, no speculation).

Layout mapping under test (see kv_policy/STAGE3_NOTES in the report):
  last_exited  ~ FULL   (fixed depth: one plane per loop, no early-exit copies)
  shared       ~ LAST, *strict* variant (prompt processed position-by-position; vllm-rlt docs)
  AVERAGE and paper-style decode-only LAST: not implemented in vllm-rlt -> N/A.

Per (model, layout):
  parity  -- greedy 64-token continuations of the first 16 frozen Stage-2 GSM8K prompts vs the HF
             reference engine run in this process (FULL/decode or LAST/strict), bf16;
  memory  -- KV pool geometry, blocks required/used per request, logical vs physical bytes;
  perf    -- decode TPOT / throughput at B in {8, 32}: 1 warm-up + 3 timed repetitions,
             real GSM8K prompts, 128 output tokens with ignore_eos.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from kvp.jobs import gpu_identity  # noqa: E402
from kvp.manifest import environment  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.specs import MODELS  # noqa: E402

HF_EQUIV = {"last_exited": ("full", "decode"), "shared": ("last", "strict")}
N_PARITY, PARITY_TOKENS, PERF_TOKENS, PERF_BATCHES, REPS = 16, 64, 128, (8, 32), 3


def frozen_prompts(stage2_dir: Path, model_key: str) -> list[list[int]]:
    lines = (stage2_dir / "prompts" / f"{model_key}__gsm8k.jsonl").read_text().splitlines()[1:]
    rows = sorted((json.loads(x) for x in lines), key=lambda r: r["uid"])
    return [r["prompt_ids"] for r in rows]


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def hf_reference(model_key, prompts, policy, mode):
    spec, tok, model = load(model_key)
    run = make_runner(spec, tok, model)
    out = run.generate(prompts, [f"p{i}" for i in range(len(prompts))], policy, mode, PARITY_TOKENS,
                       set(), lambda b, ids: False).gen_ids
    del run, model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def build_engine(model_key, layout, max_seqs, kv_bytes, dtype=torch.bfloat16, backend="triton"):
    from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.models import AutoModelForCausalLM

    spec = MODELS[model_key]
    path = snapshot_download(spec.repo, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(path, revision=None, device="cuda", dtype=dtype)
    override = None
    if spec.family == "huginn" and model.config.total_ut_steps != spec.native_depth:
        # vllm-rlt runs prompt prefill at config depth (mean_recurrence=32) and sizes the KV pool
        # with one plane per config depth; align both with the fixed depth under test.
        override = {"mean_recurrence": [model.config.mean_recurrence, spec.native_depth]}
        model.config = dataclasses.replace(model.config, mean_recurrence=spec.native_depth)
    engine = StageEngine(
        model,
        cache_config=CacheConfig(layout=layout, kv_cache_memory_bytes=kv_bytes),
        scheduler_config=SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=4096, mode="refill"),
        attention_backend=backend,
        exit_config=ExitConfig(mode="ouro"),
        execution_config=ExecutionConfig(async_scheduling=False, cuda_graphs=False, multi_stream=False),
    )
    engine.config_override = override
    return model, engine


def params(spec, max_tokens, ignore_eos):
    from vllm_rlt import SamplingParams

    d = spec.native_depth
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, min_loops=d, max_loops=d,
                          exit_threshold=1.0, ignore_eos=ignore_eos)


def run_batch(engine, prompts, sp, tag):
    """Submit all prompts at once; record per-request token timestamps."""
    ids = [f"{tag}-{i}" for i in range(len(prompts))]
    for rid, p in zip(ids, prompts):
        engine.add_request(rid, list(p), sp)
    times: dict[str, list[float]] = {rid: [] for rid in ids}
    outs = {}
    peak_used = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.has_unfinished_requests():
        step_out = engine.step()
        torch.cuda.synchronize()
        now = time.perf_counter()
        peak_used = max(peak_used, engine.cache_manager.num_used_blocks)
        for o in step_out:
            n_prev = len(times[o.request_id])
            times[o.request_id].extend([now] * (len(o.token_ids) - n_prev))
            if o.finished:
                outs[o.request_id] = o
    wall = time.perf_counter() - t0
    return [outs[r] for r in ids], [times[r] for r in ids], t0, wall, peak_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layout", required=True, choices=["last_exited", "shared"])
    ap.add_argument("--stage2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-gib", type=float, default=48.0)
    args = ap.parse_args()
    spec = MODELS[args.model]
    prompts = frozen_prompts(Path(args.stage2_dir), args.model)
    rep = {"model": args.model, "layout": args.layout, "gpu": gpu_identity(), "environment": environment(),
           "vllm_rlt_commit": None, "hf_equivalent": HF_EQUIV[args.layout]}
    import subprocess

    rep["vllm_rlt_commit"] = subprocess.run(["git", "-C", str(Path.home() / "vllm-rlt"), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()

    # ---------------- parity
    policy, mode = HF_EQUIV[args.layout]
    hf = None
    if spec.family == "ouro":
        hf = hf_reference(args.model, prompts[:N_PARITY], policy, mode)
    else:
        rep["parity_note"] = ("Huginn initial latent noise in vllm-rlt is drawn per request from its own "
                              "generator; the HF engine uses per-(uid, position) seeded noise, so only "
                              "agreement statistics (not exact parity) are meaningful.")
        hf = hf_reference(args.model, prompts[:N_PARITY], policy, mode)
    kv_bytes = int(args.kv_gib * 2**30)
    model, engine = build_engine(args.model, args.layout, max(PERF_BATCHES), kv_bytes)
    cm = engine.cache_manager
    rep["config_override"] = engine.config_override
    rep["pool"] = {"num_blocks": cm.num_blocks, "block_size": cm.block_size, "storage_depths": cm.storage_depths,
                   "bytes_per_block": cm.bytes_per_block, "num_layers_in_cache": cm.num_layers,
                   "num_kv_heads": cm.num_kv_heads, "head_dim": cm.head_dim,
                   "key_cache_shape": list(cm.key_cache.shape),
                   "pool_bytes": 2 * cm.key_cache.numel() * cm.key_cache.element_size(),
                   "memory_plan": str(getattr(engine, "memory_plan", None))}
    outs, _, _, wall, used = run_batch(engine, prompts[:N_PARITY], params(spec, PARITY_TOKENS, True), "parity")
    par = []
    for i, o in enumerate(outs):
        par.append({"i": i, "prompt_tokens": len(prompts[i]), "first_divergence": first_div(hf[i], o.token_ids),
                    "exact_match": hf[i] == o.token_ids, "exit_depths_set": sorted(set(o.exit_depths)),
                    "rlt_tokens": o.token_ids, "hf_tokens": hf[i]})
    rep["parity"] = {"items": par, "exact_match_rate": sum(p["exact_match"] for p in par) / len(par),
                     "mean_first_divergence": statistics.mean(
                         [p["first_divergence"] if p["first_divergence"] is not None else PARITY_TOKENS
                          for p in par]),
                     "all_depths_fixed": all(p["exit_depths_set"] == [spec.native_depth] for p in par)}

    # ---------------- memory (blocks per request at peak, logical bytes)
    mem = []
    for i in range(4):
        p = prompts[i]
        req_blocks = cm.required_blocks(len(p) + PERF_TOKENS - 1)
        mem.append({"prompt_tokens": len(p), "max_tokens": PERF_TOKENS, "required_blocks": req_blocks,
                    "required_bytes": req_blocks * cm.bytes_per_block,
                    "logical_kv_bytes_tokens": (len(p) + PERF_TOKENS - 1) * cm.storage_depths
                    * cm.bytes_per_block // cm.block_size})
    rep["memory_per_request"] = mem
    rep["capacity_requests_at_prompt0_len"] = cm.num_blocks // mem[0]["required_blocks"]

    # ---------------- perf
    perf = []
    for B in PERF_BATCHES:
        batch = (prompts * ((B // len(prompts)) + 1))[:B]
        sp = params(spec, PERF_TOKENS, True)
        run_batch(engine, batch, sp, f"warm{B}")
        for rep_i in range(REPS):
            torch.cuda.reset_peak_memory_stats()
            outs, times, t0, wall, used = run_batch(engine, batch, sp, f"B{B}r{rep_i}")
            ttft = [t[0] - t0 for t in times]
            tpot = [(t[-1] - t[0]) / (len(t) - 1) for t in times if len(t) > 1]
            first_all = max(t[0] for t in times)
            last_all = max(t[-1] for t in times)
            decode_tokens = sum(len(t) - 1 for t in times)
            perf.append({"B": B, "rep": rep_i, "wall_s": wall, "mean_tpot_ms": 1e3 * statistics.mean(tpot),
                         "p50_tpot_ms": 1e3 * statistics.median(tpot), "mean_ttft_s": statistics.mean(ttft),
                         "decode_throughput_tok_s": decode_tokens / max(last_all - first_all, 1e-9),
                         "e2e_throughput_tok_s": sum(len(t) for t in times) / wall,
                         "peak_used_blocks": used, "peak_used_kv_bytes": used * cm.bytes_per_block,
                         "torch_max_allocated": torch.cuda.max_memory_allocated(),
                         "torch_max_reserved": torch.cuda.max_memory_reserved(),
                         "generated_tokens": [len(t) for t in times]})
    rep["perf"] = perf
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    rep["nvidia_smi_apps"] = smi.strip().splitlines()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
