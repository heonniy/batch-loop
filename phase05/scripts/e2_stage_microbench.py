"""E2 — stage-specific H100 latency of the phase02 survivor-compacted runtime.

For each (model, active batch B, context L) one decode step is staged on a valid KV
cache built from a real L-token prefill of `SEED_ROWS` prompts, replicated to B rows
(representative KV, not a synthetic GEMM). Each trial times, with CUDA events around
synchronized stages: Prelude (Ouro: embedding only), one full recurrent Core
invocation (r=1, including its KV writes), the exit gate, the exit-time KV fill,
survivor compaction (B -> B/2), Coda blocks and LM head separately, and the host
dispatch time of Core. The token is then rolled back, so every trial sees the same
cache length L. Evidence class: PHYSICAL_FORCED_DEPTH stage timing (no exit decision
is applied to the schedule).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HF = Path.home() / ".cache/huggingface/hub"
MODELS = {
    "ouro14": ("ouro", HF / "models--ByteDance--Ouro-1.4B/snapshots/37bc7e8ff85a90d813dc096383405f521b6a67b0",
               "37bc7e8ff85a90d813dc096383405f521b6a67b0", 0.2),
    "ouro26": ("ouro", HF / "models--ByteDance--Ouro-2.6B/snapshots/f8e38372d72a8aa7f4ea9f14b98739a311589088",
               "f8e38372d72a8aa7f4ea9f14b98739a311589088", 0.2),
    "huginn": ("huginn", HF / "models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c",
               "bb6621b65e90b6a4b9b29ef88dc83866d450470c", 0.12),
}
B_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
L_GRID = (32, 64, 128, 256, 512, 1024, 2048)
SEED_ROWS = 4
WARMUP, REPS = 2, 10
REL_L2_TOL = 0.03  # predeclared: phase04 semantic tolerance for state outputs
FIELDS = ["model", "B", "L", "trial", "status", "prelude_ms", "core_ms", "core_kv_update_ms", "core_dispatch_ms",
          "core_wall_ms", "gate_ms", "exit_kv_ms", "compaction_ms", "coda_blocks_ms", "lm_head_ms",
          "peak_alloc_gib", "peak_reserved_gib", "free_gib_after_cleanup", "error"]


def gpu_identity() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    q = subprocess.check_output(["nvidia-smi", "-i", visible, "--query-gpu=index,uuid,name,memory.total",
                                 "--format=csv,noheader"], text=True).strip()
    idx, uuid, name, mem = [x.strip() for x in q.split(",")]
    assign = json.loads((ROOT / "phase05/GPU_ASSIGNMENT_20261019.json").read_text())["devices"]
    if idx not in assign or assign[idx] != uuid or "H100" not in name:
        raise RuntimeError(f"GPU {q} is not in the 2026-10-19 assignment")
    return {"physical_index": int(idx), "uuid": uuid, "name": name, "memory_total": mem,
            "CUDA_VISIBLE_DEVICES": visible}


def long_prompts(tokenizer, L: int, rows: int) -> list[list[int]]:
    """Real text: consecutive Alpaca instructions concatenated per row and cut to exactly L tokens."""
    import datasets
    texts = [r["instruction"] + "\n" + r["output"] for r in datasets.load_dataset("tatsu-lab/alpaca", split="train")
             .select(range(4000))]
    out, i = [], 0
    for _ in range(rows):
        ids: list[int] = []
        while len(ids) < L:
            ids += tokenizer(texts[i], add_special_tokens=False)["input_ids"]
            i += 1
        out.append(ids[:L])
    return out


class Bench:
    def __init__(self, key: str):
        import torch
        from phase02_runtime.huginn_adapter import HuginnAdapter
        from phase02_runtime.ouro_adapter import OuroAdapter
        self.key = key
        self.family, self.snapshot, self.revision, self.threshold = MODELS[key]
        if self.family == "huginn":
            base = HuginnAdapter.load(1, self.snapshot, state_init="native-seeded")
        else:
            base = OuroAdapter.load(1, self.snapshot)
        self.model, self.tokenizer = base.model, base.tokenizer
        self.torch = torch

    def adapter(self, rows: int):
        torch = self.torch
        from phase02_runtime.huginn_adapter import HuginnAdapter
        from phase02_runtime.ouro_adapter import OuroAdapter
        if self.family == "huginn":
            return HuginnAdapter(self.model, self.tokenizer, rows, state_init="native-seeded", identity_seed=1,
                                 canonical_ids=torch.arange(rows, device="cuda"), instrumentation="profile")
        return OuroAdapter(self.model, self.tokenizer, rows, instrumentation="profile")

    def build(self, B: int, prompts):
        """Real prefill of len(prompts) rows, replicated to B rows with exactly-sized KV (+slack)."""
        torch = self.torch
        r = len(prompts)
        small = self.adapter(r)
        ids = torch.tensor(prompts, device="cuda")
        small.prefill(ids, torch.ones_like(ids, dtype=torch.bool))
        idx = torch.arange(B, device="cuda") % r
        big = self.adapter(B)
        sb, bb = small.backend, big.backend
        bb.key_valid = sb.key_valid.index_select(0, idx)
        if self.family == "huginn":
            L = sb.history_length
            cap = L + WARMUP + REPS + 4  # sized allocation: no amortized doubling inside the benchmark
            for slot, (k, v) in sb.history.items():
                nk = torch.empty((B, k.shape[1], cap, k.shape[3]), dtype=k.dtype, device="cuda")
                nv = torch.empty_like(nk)
                nk[:, :, :L].copy_(k[:, :, :L].index_select(0, idx))
                nv[:, :, :L].copy_(v[:, :, :L].index_select(0, idx))
                bb.history[slot] = (nk, nv)
            bb.history_length, bb.history_capacity = L, cap
            big.prompt_valid = small.prompt_valid.index_select(0, idx)
        else:
            bb.history = {slot: (k.index_select(0, idx), v.index_select(0, idx)) for slot, (k, v) in sb.history.items()}
        bb.phase = "ready"
        next_tok = small.prefill_logits[:, -1, :].argmax(-1).index_select(0, idx).view(B, 1)
        del small
        return big, next_tok

    def rollback(self, a):
        b = a.backend
        b.key_valid = b.key_valid[:, :-1]
        b.current.clear()
        b.valid_current.clear()
        b.phase = "ready"

    def timed(self, fn):
        torch = self.torch
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        e.synchronize()
        return out, s.elapsed_time(e)

    def coda_parts(self, a, state):
        if self.family == "huginn":
            def blocks():
                freqs = a._freqs(a.backend.active_rows)
                x = self.model.transformer.ln_f(state)
                for i, blk in enumerate(self.model.transformer.coda, 1):
                    x = blk(x, freqs, -i, None, a.backend)
                return self.model.transformer.ln_f(x)
            x, blocks_ms = self.timed(blocks)
            _, head_ms = self.timed(lambda: self.model.lm_head(x).float())
            return blocks_ms, head_ms
        _, head_ms = self.timed(lambda: self.model.lm_head(state).float())
        return 0.0, head_ms  # Ouro: no Coda transformer blocks; final norm is inside Core

    def trial(self, a, tok, B):
        torch = self.torch
        rows = torch.arange(B, device="cuda")
        a.begin_token()
        a.backend.set_active(rows)
        a.backend.pop_update_gpu_ms()
        (emb, state), prelude_ms = self.timed(lambda: a.prelude(tok))
        a.backend.pop_update_gpu_ms()
        a.backend.instrumentation = "benchmark"  # no per-update KV events while timing Core itself
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        a.recurrent_step(state, emb, 1)
        dispatch_ms = (time.perf_counter() - w0) * 1000
        torch.cuda.synchronize()
        core_wall_ms = (time.perf_counter() - w0) * 1000
        after2, core_ms = self.timed(lambda: a.recurrent_step(state, emb, 1))
        a.backend.instrumentation = "profile"  # separate call only to attribute KV-update GPU time
        a.recurrent_step(state, emb, 1)
        torch.cuda.synchronize()
        kv_ms = a.backend.pop_update_gpu_ms()
        mask, gate_ms = self.timed(lambda: a.live_exit(state, after2, rows, 1, self.threshold))
        half = rows[: max(1, B // 2)]
        _, exit_ms = self.timed(lambda: a.backend.on_exit(half, 1))
        keep = torch.arange(max(1, B // 2), device="cuda")
        _, comp_ms = self.timed(lambda: (after2.index_select(0, keep), emb.index_select(0, keep), rows.index_select(0, keep)))
        blocks_ms, head_ms = self.coda_parts(a, after2)
        self.rollback(a)
        return {"prelude_ms": prelude_ms, "core_ms": core_ms, "core_kv_update_ms": kv_ms,
                "core_dispatch_ms": dispatch_ms, "core_wall_ms": core_wall_ms, "gate_ms": gate_ms,
                "exit_kv_ms": exit_ms, "compaction_ms": comp_ms, "coda_blocks_ms": blocks_ms,
                "lm_head_ms": head_ms}, after2[0].float()


def run(args) -> None:
    import torch
    torch.set_num_threads(4)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    gpu = gpu_identity()
    bench = Bench(args.model)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    b_grid = (1, 2, 4) if args.smoke else B_GRID
    l_grid = (32, 128) if args.smoke else L_GRID
    raw = (out / "e2_stage_latency_raw.csv").open("w", newline="")
    writer = csv.DictWriter(raw, fieldnames=FIELDS)
    writer.writeheader()
    gates = []
    prompt_hashes = {}
    for L in l_grid:
        prompts = long_prompts(bench.tokenizer, L, SEED_ROWS)
        prompt_hashes[L] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        reference = None
        for B in b_grid:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            status, err, row0 = "OK", "", None
            a = tok = None  # never let a previous cell's cache survive into this one
            try:
                a, tok = bench.build(B, prompts[: min(SEED_ROWS, B)] if B < SEED_ROWS else prompts)
                if a.backend.key_valid.shape != (B, L):
                    raise AssertionError("KV contract: key_valid shape mismatch")
                for t in range(WARMUP + REPS):
                    res, row0 = bench.trial(a, tok, B)
                    if t >= WARMUP:
                        writer.writerow({"model": args.model, "B": B, "L": L, "trial": t - WARMUP, "status": "OK",
                                         **res, "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                                         "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "error": ""})
            except torch.cuda.OutOfMemoryError as e:
                status, err = "OOM", str(e).splitlines()[0][:240]
            except Exception as e:  # retained as a result
                status, err = "ERROR", f"{type(e).__name__}: {e}"[:240]
            finally:
                a = tok = None
                gc.collect()
                torch.cuda.empty_cache()
            if status != "OK":
                writer.writerow({"model": args.model, "B": B, "L": L, "trial": -1, "status": status, "error": err,
                                 "free_gib_after_cleanup": torch.cuda.mem_get_info()[0] / 2**30,
                                 "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                                 "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
                gates.append({"model": args.model, "B": B, "L": L, "status": status})
            else:
                if B == 1:
                    reference = row0
                rel = (float(torch.linalg.vector_norm(row0 - reference) / torch.linalg.vector_norm(reference))
                       if reference is not None else float("nan"))
                gates.append({"model": args.model, "B": B, "L": L, "status": "PASS" if (bool(torch.isfinite(row0).all())
                              and (rel != rel or rel <= REL_L2_TOL)) else "FAIL", "row0_rel_l2_vs_B1": rel,
                              "tolerance": REL_L2_TOL})
            raw.flush()
            print(json.dumps({"model": args.model, "B": B, "L": L, "status": status, "gate": gates[-1]["status"]}),
                  flush=True)
            torch.cuda.empty_cache()
    raw.close()
    (out / "e2_correctness_gate.json").write_text(json.dumps(gates, indent=2) + "\n")
    (out / "RUN_MANIFEST.json").write_text(json.dumps({
        "experiment": "E2", "evidence_class": "PHYSICAL_FORCED_DEPTH (stage timing at fixed state)",
        "model": args.model, "model_revision": bench.revision, "snapshot": str(bench.snapshot),
        "source_sha": source_sha, "argv": sys.argv, "gpu": gpu, "dtype": "bfloat16",
        "attention": "Ouro: HF eager with explicit additive mask; Huginn: SDPA math (phase02 head96 wrapper)",
        "kv_policy": "Ouro last-exited depth-indexed; Huginn shared physical-layer",
        "B_grid": b_grid, "L_grid": l_grid, "seed_rows": SEED_ROWS, "warmup": WARMUP, "reps": REPS,
        "prompt_source": "tatsu-lab/alpaca train[0:4000] instruction+output concatenated per row, cut to L tokens",
        "prompt_hashes": prompt_hashes, "torch": torch.__version__, "cuda": torch.version.cuda,
        "notes": ["KV of SEED_ROWS real prefilled rows replicated to B rows",
                  "Huginn history allocated at L+slack (no amortized doubling) inside the benchmark",
                  "core_ms: CUDA-event time of one full recurrent Core invocation at r=1 incl. KV writes",
                  "core_dispatch_ms: host time until Core call returns, before synchronize (KV event instrumentation off)",
                  "core_kv_update_ms: KV-update CUDA events from a separate profiled Core call (same state)"]},
        indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
