"""Item preparation, deterministic batching and per-job execution for Stage-2 quality runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from .data import load_subset
from .grading import grade, truncate_at_stops
from .models import eos_ids, load, make_runner
from .prompts import encode_prompt, gold_target, protocol, render
from .specs import MODELS

# B=16 for MATH-500 keeps the FULL-sized reference store of Ouro-2.6B / Huginn strict runs
# (prompt <= 1.7k + 1024 new tokens) inside H200 memory (B=32 peaked at 137.7 GB).
BATCH_SIZE = {"gsm8k": 8, "math500": 4}
QUALITY_POLICIES = [("full", "decode"), ("last", "convert"), ("average", "convert")]
# Controls (added after Stage-1 bf16 parity showed token-by-token prompt numerics differ from the
# parallel prefill): FULL/strict isolates the numerical path of strict cells; FULL/decode with an
# alternative seeded batch composition measures the bf16 batch-shape noise floor.
CONTROL_CELLS = []
PILOT_SELECTION = Path(__file__).resolve().parents[1] / "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json"


SOURCE_SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def prepare_items(model_key: str, dataset: str, tok, batching: str = "main",
                  batch_size: int | None = None, prompt_style: str = "default") -> tuple[list[dict], dict]:
    batch_size = BATCH_SIZE[dataset] if batch_size is None else int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")
    spec = MODELS[model_key]
    docs, meta = load_subset(dataset)
    proto = protocol(spec.family, dataset)
    if prompt_style not in ("default", "reason", "verify", "distract512"):
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    if prompt_style != "default" and spec.family != "huginn":
        raise ValueError("Prompt-style stress tests are Huginn-only")
    if prompt_style == "reason":
        proto = replace(proto, system_instruction=(proto.system_instruction or "") +
            " Please work through the reasoning carefully step by step, then state the final answer.")
    elif prompt_style == "verify":
        proto = replace(proto, system_instruction=(proto.system_instruction or "") +
            " Work through the problem and check any numerical or logical intermediate result before the final answer.")
    items = []
    for d in docs:
        rendered_doc = d
        if prompt_style == "distract512":
            # Content is deliberately irrelevant and contains no problem facts;
            # it is placed in the last QUESTION before lm-eval formatting.
            # This is a controlled context distraction, NOT a new dataset.
            filler = ("Unrelated background: A library hosted a quiet art exhibition with "
                      "wooden frames, clouds in paintings, and a small courtyard. ")
            padding = filler
            while len(tok.encode(padding, add_special_tokens=False)) < 512:
                padding += filler
            key = "question" if dataset == "gsm8k" else "problem"
            rendered_doc = dict(d)
            rendered_doc[key] = (
                "The following unrelated paragraph is background only; "
                "ignore it when solving the actual problem.\n" +
                padding + "\nActual problem:\n" + d[key])
        text = render(rendered_doc, proto, tok)
        ids = encode_prompt(text, tok)
        target = gold_target(d, proto)
        whole = tok.encode(text + target, add_special_tokens=False)
        if whole[: len(ids)] == ids:
            tgt_ids, tgt_mode = whole[len(ids):], "joint_suffix"
        else:
            tgt_ids, tgt_mode = tok.encode(target, add_special_tokens=False), "separate"
        items.append({"uid": d["uid"], "doc": d, "prompt": text, "prompt_ids": ids, "target": target,
                      "target_ids": tgt_ids, "target_tokenization": tgt_mode})
    if batching == "pilot":  # Stage-1 stratified 32-item selection per dataset (diagnosis pilots)
        sel = set(json.loads(PILOT_SELECTION.read_text())[dataset])
        items = [it for it in items if it["uid"] in sel]
    # deterministic, policy-independent batch composition: longest prompts first, chunks of B
    order = sorted(range(len(items)), key=lambda i: (-len(items[i]["prompt_ids"]), items[i]["uid"]))
    if batching == "alt":  # same items, seeded random composition (control only)
        order = list(range(len(items)))
        random.Random(20261023).shuffle(order)
    for rank, i in enumerate(order):
        items[i]["batch"], items[i]["batch_pos"] = divmod(rank, batch_size)
    return items, {"dataset_meta": meta, "protocol": proto.__dict__,
                   "prompt_style": prompt_style}


def batches(items: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for it in sorted(items, key=lambda x: (x["batch"], x["batch_pos"])):
        out.setdefault(it["batch"], []).append(it)
    return out


def gpu_identity() -> dict:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {"host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": props.name, "uuid": str(getattr(props, "uuid", "")), "total_mem": props.total_memory}


class ModelContext:
    def __init__(self, model_key: str):
        self.key = model_key
        self.spec, self.tok, self.model = load(model_key)
        self.runner = make_runner(self.spec, self.tok, self.model)
        self.eos = eos_ids(self.spec, self.tok, self.model)
        self.eos_strings = sorted({self.tok.decode([e], skip_special_tokens=False) for e in self.eos})
        self.items: dict[str, tuple[list[dict], dict]] = {}

    def dataset(self, dataset: str, batching: str = "main", batch_size: int | None = None,
                prompt_style: str = "default"):
        cache_key = (dataset, batching, batch_size, prompt_style)
        if cache_key not in self.items:
            self.items[cache_key] = prepare_items(self.key, dataset, self.tok, batching,
                                                  batch_size, prompt_style)
        return self.items[cache_key]


def run_job(ctx: ModelContext, job: dict, out_dir: Path) -> dict:
    dataset, task, policy, mode, b = job["dataset"], job["task"], job["policy"], job["mode"], job["batch"]
    if policy.startswith("bank"):
        from .banks import REGISTRY, load_registry

        if (ctx.key, policy) not in REGISTRY:
            load_registry(out_dir.parent / "banks.json")
    batching = job.get("batching", "main")
    prompt_style = job.get("prompt_style", "default")
    items, meta = ctx.dataset(dataset, batching, job.get("batch_size"), prompt_style)
    batch = batches(items)[b]
    proto = protocol(ctx.spec.family, dataset)
    uids = [it["uid"] for it in batch]
    prompts = [it["prompt_ids"] for it in batch]
    t0 = time.time()
    records = []
    if task == "gen":
        stops = tuple(proto.until) + tuple(ctx.eos_strings)

        def stop_check(_b, ids, _stops=stops):
            text = ctx.tok.decode(ids, skip_special_tokens=False)
            return any(s in text for s in _stops)

        res = ctx.runner.generate(prompts, uids, policy, mode, proto.max_new_tokens, ctx.eos, stop_check)
        for it, ids, fin in zip(batch, res.gen_ids, res.finish):
            raw = ctx.tok.decode(ids, skip_special_tokens=False)
            text = truncate_at_stops(raw, stops)
            g = grade(dataset, text, it["doc"], fin)
            records.append({"uid": it["uid"], "batch": b, "batch_pos": it["batch_pos"],
                            "prompt_sha256": sha(it["prompt"]), "prompt_tokens": len(it["prompt_ids"]),
                            "gen_token_ids": ids, "n_gen_tokens": len(ids), "finish_reason": fin,
                            "raw_generation": raw, "answer_text": text, **g})
    else:
        targets = [it["target_ids"] for it in batch]
        res = ctx.runner.score(prompts, targets, uids, policy, mode)
        for it, lps in zip(batch, res.target_logprobs):
            records.append({"uid": it["uid"], "batch": b, "batch_pos": it["batch_pos"],
                            "prompt_sha256": sha(it["prompt"]), "prompt_tokens": len(it["prompt_ids"]),
                            "target_sha256": sha(it["target"]), "target_tokens": len(lps),
                            "target_tokenization": it["target_tokenization"],
                            "sum_logprob": float(sum(lps)), "nll_mean": float(-sum(lps) / len(lps)),
                            "token_logprobs": [round(x, 6) for x in lps]})
    job_meta = {**job, "model_revision": ctx.spec.revision, "n_items": len(records),
                "wall_s": time.time() - t0, "source_sha": SOURCE_SHA, "engine_stats": res.stats, "gpu": gpu_identity(),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for r in records:
        r.update({"model": ctx.key, "dataset": dataset, "task": task, "policy": policy, "mode": mode,
                  "batching": batching, "prompt_style": prompt_style})
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = job_name(job)
    tmp = out_dir / f".{stem}.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps({"job_meta": job_meta}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out_dir / f"{stem}.jsonl")
    return job_meta


def job_name(job: dict) -> str:
    alt = "" if job.get("batching", "main") == "main" else f"-{job['batching']}"
    style = "" if job.get("prompt_style", "default") == "default" else f"__style{job['prompt_style']}"
    return (f"{job['model']}__{job['dataset']}__{job['task']}__{job['policy']}-{job['mode']}{alt}"
            f"{style}__b{job['batch']:02d}")
