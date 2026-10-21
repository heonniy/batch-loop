"""Phase-07 E0 feature/label collection: per-request B=1 live-exit decode on the phase02 survivor runtime.

Each request is decoded alone (B=1, BF16, eager phase02 adapters, model-specific KV semantics) with greedy
next tokens, which also defines the teacher-forced reference token sequence. Per request-token (t = decode
position) the collector stores, aligned to the CURRENT token's recurrence (no lookahead):

  pre-Core (available before the current token's recurrent work):
    token_id, position t, context length, prompt length, request id / prompt hash, past depths (derivable);
    emb   : Ouro token embedding (= Ouro's initial recurrent state) | Huginn scaled token embedding (wte)
    pre   : Huginn final Prelude hidden (Ouro: none — Ouro has no Transformer Prelude)
    prevh : final recurrent state of the PREVIOUS token (zeros at t=0)
  early recurrent trajectory (diagnostic only; NOT available at a no-refill decode boundary):
    loop1, loop2 : hidden after recurrent step 1 / 2 (loop2 zero + mask when the token exited at 1)
    traj[r]      : Ouro per-step hazard sigmoid(gate) | Huginn latent_diff ||h_r-h_{r-1}||/||h_r|| (NaN after exit)
  label: depth (exit step, 1..Rmax) under the fixed threshold; next_token (greedy).
Labels are canonical B=1 depths. Phase-04 showed BF16 batch shape can change live depths: these labels are
NOT batch-invariant truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HF = Path.home() / ".cache/huggingface/hub"
MODELS = {
    "ouro14": ("ouro", HF / "models--ByteDance--Ouro-1.4B/snapshots/37bc7e8ff85a90d813dc096383405f521b6a67b0", 4),
    "ouro26": ("ouro", HF / "models--ByteDance--Ouro-2.6B/snapshots/f8e38372d72a8aa7f4ea9f14b98739a311589088", 4),
    "huginn": ("huginn", HF / "models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c", 16),
}
SPLIT_SEED = 20261021
SPLITS = {"train": (0, 512), "val": (512, 640), "test": (640, 896)}  # request-disjoint, predeclared
T, PROMPT_CAP = 64, 64


def dataset_prompts(dataset: str) -> list[str]:
    import datasets
    from poc import _first_text
    name = {"alpaca": "tatsu-lab/alpaca", "sharegpt": "ewof/sharegpt-instruct-unfiltered-deduped"}[dataset]
    texts = []
    for row in datasets.load_dataset(name, split="train"):
        try:
            texts.append(_first_text(dict(row)))
        except ValueError:
            continue
    pool = sorted(set(texts))  # dedupe -> disjoint prompt hashes across splits
    random.Random(SPLIT_SEED).shuffle(pool)
    return pool


def gpu_identity() -> dict:
    import os
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    q = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=index,uuid,name,memory.used",
                                 "--format=csv,noheader"], text=True).strip()
    idx, uuid, name, used = [x.strip() for x in q.split(",")]
    assign = json.loads((ROOT / "phase07/GPU_ASSIGNMENT_20261021.json").read_text())["devices"]
    if assign.get(idx) != uuid or "H100" not in name:
        raise RuntimeError(f"GPU {q} not in the Phase-07 assignment")
    return {"physical_index": int(idx), "uuid": uuid, "name": name, "memory_used_at_start": used}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--threshold", required=True, type=float)
    p.add_argument("--dataset", required=True, choices=("alpaca", "sharegpt"))
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    import torch
    from phase02_runtime.common import run_interval
    from phase02_runtime.huginn_adapter import HuginnAdapter
    from phase02_runtime.ouro_adapter import OuroAdapter
    torch.set_num_threads(4)
    family, snap, rmax = MODELS[a.model]
    gpu = gpu_identity()
    base = HuginnAdapter.load(1, snap, state_init="native-seeded") if family == "huginn" else OuroAdapter.load(1, snap)
    model, tok = base.model, base.tokenizer
    hidden = int(model.transformer.wte.weight.shape[1] if family == "huginn" else model.model.embed_tokens.weight.shape[1])
    pool = dataset_prompts(a.dataset)
    kind = "huginn_convergence" if family == "huginn" else "ouro_gate"
    a.output_dir.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    for split in a.splits:
        lo, hi = SPLITS[split]
        prompts = pool[lo:hi][: 4 if a.smoke else None]
        steps = 4 if a.smoke else T
        n = len(prompts)
        d = a.output_dir / split
        d.mkdir()
        # Huginn hidden states exceed the float16 range (scaled embeddings), so they are stored as float32.
        dtype = "float32" if family == "huginn" else "float16"
        mm = lambda name: np.lib.format.open_memmap(d / f"{name}.npy", mode="w+", dtype=dtype, shape=(steps, n, hidden))
        arrays = {k: mm(k) for k in (("emb", "pre", "prevh", "loop1", "loop2") if family == "huginn"
                                     else ("emb", "prevh", "loop1", "loop2"))}
        traj = np.full((steps, n, rmax), np.nan, dtype=np.float32)
        depth = np.zeros((steps, n), dtype=np.int16)
        token = np.zeros((steps, n), dtype=np.int32)
        nxt = np.zeros((steps, n), dtype=np.int32)
        enc = tok(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_CAP)
        plen = enc["attention_mask"].sum(dim=1).numpy().astype(np.int32)
        began = time.perf_counter()
        with torch.inference_mode():
            for rid in range(n):
                if family == "huginn":
                    ad = HuginnAdapter(model, tok, 1, state_init="native-seeded", identity_seed=1,
                                       canonical_ids=torch.tensor([rid], device="cuda"), instrumentation="benchmark")
                else:
                    ad = OuroAdapter(model, tok, 1, instrumentation="benchmark")
                cur = ad.prefill(enc["input_ids"][rid:rid + 1].cuda(), enc["attention_mask"][rid:rid + 1].cuda().bool()).reshape(1, 1)
                row = torch.tensor([0], device="cuda")
                prevh = torch.zeros((1, hidden), device="cuda")
                for t in range(steps):
                    ad.begin_token()
                    ad.backend.set_active(row)
                    emb, state = ad.prelude(cur)
                    raw = emb if family == "ouro" else model.transformer.wte(cur) * model.emb_scale
                    arrays["emb"][t, rid] = raw.float().reshape(-1).cpu().numpy()
                    if family == "huginn":
                        arrays["pre"][t, rid] = emb.float().reshape(-1).cpu().numpy()
                    arrays["prevh"][t, rid] = prevh.reshape(-1).cpu().numpy()
                    loops = {}
                    sink: dict = {}

                    def hook(r, ids, after, _ad):
                        if r <= 2:
                            loops[r] = after.float().reshape(-1).cpu().numpy()
                    res = run_interval(ad, emb, state, row, row, exit_source="live", threshold=a.threshold,
                                       instrumentation="benchmark", debug_ids=False, trajectory_sink=sink,
                                       trajectory_kind=kind, debug_hook=hook)
                    ad.backend.set_active(row)
                    logits = ad.coda_logits(res.final_state)
                    nt = logits.argmax(dim=-1).reshape(1, 1)
                    ad.commit_token()
                    dd = int(res.exit_depths[0])
                    depth[t, rid], token[t, rid], nxt[t, rid] = dd, int(cur[0, 0]), int(nt[0, 0])
                    arrays["loop1"][t, rid] = loops[1]
                    if 2 in loops:
                        arrays["loop2"][t, rid] = loops[2]
                    traj[t, rid, :len(sink[0])] = sink[0]
                    prevh = res.final_state.float().reshape(1, -1)
                    cur = nt
                del ad
                if (rid + 1) % 64 == 0:
                    print(json.dumps({"split": split, "done": rid + 1, "of": n,
                                      "elapsed_s": round(time.perf_counter() - began, 1)}), flush=True)
        for arr in arrays.values():
            arr.flush()
        np.savez(d / "meta.npz", depth=depth, token=token, next_token=nxt, traj=traj, prompt_len=plen)
        (d / "manifest.json").write_text(json.dumps({
            "model": a.model, "family": family, "snapshot": str(snap), "rmax": rmax, "threshold": a.threshold,
            "exit_rule": ("Ouro: first r with 1-prod(1-sigmoid(gate)) >= q (cap Rmax)" if family == "ouro" else
                          "Huginn: first r with mean_seq ||h_r-h_{r-1}||/max(||h_r||,1e-12) < tau (cap Rmax)"),
            "kv_policy": base.kv_policy, "dtype": "bfloat16", "dataset": a.dataset, "split": split,
            "split_rule": {"seed": SPLIT_SEED, "pool": "deduplicated prompt texts, shuffled", "range": SPLITS[split]},
            "prompt_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in prompts], "T": steps,
            "prompt_cap": PROMPT_CAP, "teacher_forcing": "B=1 greedy tokens define the reference sequence",
            "source_sha": source_sha, "gpu": gpu, "argv": sys.argv, "elapsed_s": time.perf_counter() - began,
            "arrays": sorted(arrays) + ["meta.npz"], "smoke": a.smoke}, indent=2) + "\n")
        print(json.dumps({"event": "split_done", "split": split, "elapsed_s": round(time.perf_counter() - began, 1)}),
              flush=True)


if __name__ == "__main__":
    main()
