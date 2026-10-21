"""[Copied from Phase-06 phase06/scripts/common.py @ local commit 3e2779e; paths switched to phase07.]

Shared Phase-06 helpers: pins, GPU ownership check, frozen inputs/traces, engine construction.

Run with the pinned upstream environment:
  phase06/third_party/continuous-depth-batching/.venv/bin/python
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "phase07/third_party/continuous-depth-batching"
UPSTREAM_SHA = "abdfae0adbdd12be8c6123ccb34eee98a38afb39"
UPSTREAM_LOCK_SHA256 = "c71a58bfc82f5607aed59b5c86a2fc53bb2880e5090f6ae41b09dbe12beb33fc"
sys.path.insert(0, str(ROOT / "phase07"))

HF = Path.home() / ".cache/huggingface/hub"
MODELS = {  # upstream checkpoints; shared tensors bit-identical to the Phase 05 ByteDance revisions (E0)
    "ouro14": {"path": HF / "models--KristianS7--Ouro-1.4B/snapshots/a27316f37f1f10b3395017c4fec0fea96f28150c",
               "hub": "KristianS7/Ouro-1.4B@a27316f3", "rmax": 4, "same_weights_as": "ByteDance/Ouro-1.4B@37bc7e8f"},
    "ouro26": {"path": HF / "models--KristianS7--Ouro-2.6B/snapshots/ec149213509db556ed696887cd98bc0c9795504f",
               "hub": "KristianS7/Ouro-2.6B@ec149213", "rmax": 4, "same_weights_as": "ByteDance/Ouro-2.6B@f8e38372"},
    "huginn": {"path": HF / "models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c",
               "hub": "tomg-group-umd/huginn-0125@bb6621b6", "rmax": 16, "same_weights_as": "itself"},
}
TRACES = ROOT / "phase02/oracle_upper_bound_results_20260916/trace_inputs"
PROMPTS = ROOT / "phase06/inputs/phase02_prompts_seed1_n256.json"
ATTN = "paged|flash_attention_3"


def gpu_identity() -> dict:
    """Verify the visible GPU is an assigned H100 with no foreign compute process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    q = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=index,uuid,name,memory.used,driver_version",
                                 "--format=csv,noheader"], text=True).strip()
    idx, uuid, name, used, driver = [x.strip() for x in q.split(",")]
    assign = json.loads((ROOT / "phase07/GPU_ASSIGNMENT_20261021.json").read_text())["devices"]
    if assign.get(idx) != uuid or "H100" not in name:
        raise RuntimeError(f"GPU {q} is not in the Phase-06 assignment")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    foreign = [line for line in apps.splitlines() if uuid in line and str(os.getpid()) not in line]
    return {"physical_index": int(idx), "uuid": uuid, "name": name, "driver": driver,
            "memory_used_at_start": used, "other_processes_on_gpu_at_start": foreign}


def env_info() -> dict:
    import torch
    import transformers
    try:
        import flash_attn_interface
        fa3 = getattr(flash_attn_interface, "__version__", "flash-attn-3==3.0.0 (lockfile)")
    except Exception as e:  # recorded, not hidden
        fa3 = f"unavailable: {e}"
    upstream_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=UPSTREAM).strip()
    if upstream_sha != UPSTREAM_SHA:
        raise RuntimeError(f"upstream checkout {upstream_sha} != pinned {UPSTREAM_SHA}")
    lock = hashlib.sha256((UPSTREAM / "uv.lock").read_bytes()).hexdigest()
    if lock != UPSTREAM_LOCK_SHA256:
        raise RuntimeError("upstream uv.lock changed")
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__, "flash_attention": fa3, "attn_implementation": ATTN,
            "upstream_sha": upstream_sha, "upstream_uv_lock_sha256": lock,
            "local_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
            "local_dirty_phase06_sources": subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no", "--", "phase07"], text=True, cwd=ROOT).strip()}


def load_trace(name: str) -> dict:
    """Phase-02 frozen B=1 depth trace -> depth[T, N] (1-indexed) plus metadata."""
    import numpy as np
    path = TRACES / name
    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    n = 1 + max(r["request_id"] for r in rows)
    t = 1 + max(r["token_idx"] for r in rows)
    depth = np.zeros((t, n), dtype=np.int64)
    for r in rows:
        depth[r["token_idx"], r["request_id"]] = r["depth"]
    first = rows[0]
    huginn = first.get("model") == "huginn"
    return {"depth": depth, "family": "huginn" if huginn else "ouro",
            "dataset": "alpaca" if "alpaca" in name else "sharegpt",
            "threshold": first["threshold"] if huginn else first["exit_threshold"],
            "exit_rule": first["exit_source"] if huginn else first["exit_rule"],
            "sha256": hashlib.sha256(gzip.open(path, "rb").read()).hexdigest(), "name": name}


def prompts(dataset: str, tokenizer, n: int, cap: int = 64) -> tuple[list[list[int]], str]:
    texts = json.loads(PROMPTS.read_text())[dataset]["prompts"][:n]
    ids = [tokenizer(t, add_special_tokens=True)["input_ids"][:cap] for t in texts]
    return ids, hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def load_model(key: str):
    import torch
    from looped_cdb.benchmarks.runner import load_serving_model
    from transformers import AutoTokenizer
    m = MODELS[key]
    model = load_serving_model(str(m["path"]), max_recurrent_depth=m["rmax"], attn_implementation=ATTN,
                               dtype=torch.bfloat16, device="cuda", kv_policy="single")
    tok = AutoTokenizer.from_pretrained(str(m["path"]), trust_remote_code=True)
    return model, tok


def build_cdb(model, *, max_num_seqs: int, rmax: int, delayed: bool, num_blocks: int | None = None):
    """Upstream CDB engine: refill=False, CUDA graphs ON (decode and prefill), async batching ON,
    paged FlashAttention-3, synthetic exit replay. ``delayed`` selects the upstream delayed gate
    consumption (min replayable depth 2) versus immediate consumption (depth 1 allowed)."""
    import torch
    from looped_cdb.benchmarks.runner import EngineArgs, build_engine
    args = EngineArgs(backend="cdb", block_size=16, num_blocks=num_blocks, max_num_batched_tokens=8192,
                      max_model_len=1024, max_recurrent_depth=rmax, max_num_seqs=max_num_seqs, min_free_slots=1,
                      min_recurrent_steps=1, no_delay_gate_consumption=not delayed, kv_policy="single",
                      kv_slots_per_layer=None, exit_threshold=None, sync=False, refill=False,
                      min_coda_batch_size=1, safety_margin=0.0, kv_pressure_mode="none", cpu_offload_space=None)
    return build_engine(args, model, dtype=torch.bfloat16, use_cuda_graph=True, use_cuda_graph_prefill=True), args
