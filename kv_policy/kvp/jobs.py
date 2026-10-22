"""Item preparation, deterministic batching and per-job execution for Stage-2 quality runs."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch

from .data import load_subset
from .grading import grade, truncate_at_stops
from .models import eos_ids, load, make_runner
from .prompts import encode_prompt, gold_target, protocol, render
from .specs import MODELS

# B=16 for MATH-500 keeps the FULL-sized reference store of Ouro-2.6B / Huginn strict runs
# (prompt <= 1.7k + 1024 new tokens) inside H200 memory (B=32 peaked at 137.7 GB).
BATCH_SIZE = {"gsm8k": 32, "math500": 16}
QUALITY_POLICIES = [("full", "decode"), ("last", "decode"), ("average", "decode"),
                    ("last", "strict"), ("average", "strict")]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def prepare_items(model_key: str, dataset: str, tok) -> tuple[list[dict], dict]:
    spec = MODELS[model_key]
    docs, meta = load_subset(dataset)
    proto = protocol(spec.family, dataset)
    items = []
    for d in docs:
        text = render(d, proto, tok)
        ids = encode_prompt(text, tok)
        target = gold_target(d, proto)
        whole = tok.encode(text + target, add_special_tokens=False)
        if whole[: len(ids)] == ids:
            tgt_ids, tgt_mode = whole[len(ids):], "joint_suffix"
        else:
            tgt_ids, tgt_mode = tok.encode(target, add_special_tokens=False), "separate"
        items.append({"uid": d["uid"], "doc": d, "prompt": text, "prompt_ids": ids, "target": target,
                      "target_ids": tgt_ids, "target_tokenization": tgt_mode})
    # deterministic, policy-independent batch composition: longest prompts first, chunks of B
    order = sorted(range(len(items)), key=lambda i: (-len(items[i]["prompt_ids"]), items[i]["uid"]))
    for rank, i in enumerate(order):
        items[i]["batch"], items[i]["batch_pos"] = divmod(rank, BATCH_SIZE[dataset])
    return items, {"dataset_meta": meta, "protocol": proto.__dict__}


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

    def dataset(self, dataset: str):
        if dataset not in self.items:
            self.items[dataset] = prepare_items(self.key, dataset, self.tok)
        return self.items[dataset]


def run_job(ctx: ModelContext, job: dict, out_dir: Path) -> dict:
    dataset, task, policy, mode, b = job["dataset"], job["task"], job["policy"], job["mode"], job["batch"]
    items, meta = ctx.dataset(dataset)
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
            g = grade(dataset, text, it["doc"])
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
                "wall_s": time.time() - t0, "engine_stats": res.stats, "gpu": gpu_identity(),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for r in records:
        r.update({"model": ctx.key, "dataset": dataset, "task": task, "policy": policy, "mode": mode})
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
    return f"{job['model']}__{job['dataset']}__{job['task']}__{job['policy']}-{job['mode']}__b{job['batch']:02d}"
