#!/usr/bin/env python3
"""CUDA-only source-trace-matched gate probe for Dynamic SD E3.

A=replay (gate disabled), B=shadow_gpu, C=shadow_readback, D=native live.
The A/B/C schedule uses recorded AR-live depths and committed token IDs; only
A/B/C may be differenced. It does NOT time WaveFront/SD or batched serving.
Run from repo root: python3 -m dynamic_sd.e3_gate_probe --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dynamic_sd.gate_probe_common import (
    MODES, load_json, sha256_file, summarize, validate_trace,
)


def sha_json(data):
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, help="Actual dsd_run.py native AR-live JSON/.gz")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--max-decode-positions", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--wf-root", default=os.environ.get(
        "WF_ROOT", os.path.expanduser("~/dsd_work/wavefront-decoding")))
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = arguments()
    if min(a.limit, a.max_decode_positions, a.repeats) <= 0 or a.warmup < 0:
        raise ValueError("limit, max-decode-positions, repeats >0; warmup >=0")
    modes = a.modes.split(",")
    if len(set(modes)) != len(modes) or not set(modes).issubset(MODES) or not set(MODES[:3]).issubset(modes):
        raise ValueError("Modes must be unique and include replay,shadow_gpu,shadow_readback")

    report = load_json(a.reference)
    cfg = report.get("meta", {}).get("args", {})
    required = ("model", "dataset", "exit", "T_total", "seed", "offset",
                "max_prompt_tokens", "dtype", "sampling")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError("Source trace missing configuration: " + repr(missing))
    if cfg["exit"] != "live" or cfg["sampling"] != "greedy" or cfg.get("kv_s") not in (None, 0):
        raise ValueError("Only native LIVE, greedy and FULL KV AR source traces are allowed")
    kind = ("ouro" if str(cfg["model"]).startswith("ouro") else
            "huginn" if str(cfg["model"]).startswith("huginn") else None)
    if kind is None or cfg.get("tau" if kind == "ouro" else "eps") is None:
        raise ValueError("Missing calibrated native Ouro tau / Huginn epsilon")
    ref_rows = report.get("per_sample", [])
    if a.limit > len(ref_rows):
        raise ValueError("limit exceeds AR trace sample count")
    samples = ref_rows[:a.limit]
    T = int(cfg["T_total"])
    for row in samples:
        validate_trace(row, a.max_decode_positions, T)

    wf = Path(a.wf_root).expanduser().resolve()
    if not (wf / "src/decoding/state.py").is_file():
        raise FileNotFoundError("WF_ROOT must be a checked-out WaveFront source directory")
    current_sha = subprocess.run(["git", "-C", str(wf), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    pinned_sha = report.get("meta", {}).get("wf_commit")
    if not pinned_sha or current_sha != pinned_sha:
        raise RuntimeError(f"WaveFront commit mismatch: source={pinned_sha}, current={current_sha}")
    sys.path.insert(0, str(wf))
    import torch
    from experiments.exp2_spec_bench.run import build_blocks, format_prompt, resolve_prompt_format
    from experiments.exp3_accuracy import tasks as ds
    from src.decoding.state import ActiveToken

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for wall/gate event measurements")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(cfg["dtype"])
    if dtype is None:
        raise ValueError("Unsupported reference dtype")

    torch.manual_seed(cfg["seed"])
    blocks, wrapper = build_blocks(cfg["model"], dtype=dtype, device="cuda",
                                    T_total=T, state_init="random", state_seed=cfg["seed"])
    blocks.kv_budget_s = None
    prompt_format = resolve_prompt_format(cfg["model"], "auto")
    prompt_rows = ds.load_dataset_prompts(cfg["dataset"], None, cfg["offset"] + cfg["limit"])
    source_rows = {cfg["offset"] + i: prompt_rows[cfg["offset"] + i]
                   for i in range(a.limit)}
    if any(r["idx"] not in source_rows for r in samples):
        raise ValueError("Reference idx not present in source prompt range")

    def set_native(mode):
        blocks.exit_threshold = cfg.get("tau") if mode == "live" and kind == "ouro" else None
        blocks.exit_hidden_eps = cfg.get("eps") if mode == "live" and kind == "huginn" else None

    def shadow(tok, previous_output, readback):
        """Same GPU gate operation family on the post-R hidden; decision ignored."""
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if kind == "ouro":
            value = torch.sigmoid(blocks.inner.early_exit_gate(
                tok.hidden.view(1, 1, -1))[0, 0, 0].float())
        else:
            old = tok.hidden if previous_output is None else previous_output
            current = tok.hidden.float()
            old = old.float()
            value = (torch.linalg.vector_norm(current - old) /
                     torch.linalg.vector_norm(old).clamp_min(1e-6))
        end.record()
        readback_ms = 0.
        if readback:
            t = time.perf_counter()
            float(value.item())  # deliberate GPU->host dependency for diagnostic C
            readback_ms = (time.perf_counter() - t) * 1000.
        return (start, end), readback_ms

    def probe(sample, mode, repeat):
        set_native(mode)
        # Deterministic reference initialization per arm; not part of decode timer.
        torch.manual_seed(cfg["seed"])
        prompt = format_prompt(wrapper, source_rows[sample["idx"]]["prompt"], prompt_format)
        prompt_ids = wrapper.encode(prompt)[:, :cfg["max_prompt_tokens"]].squeeze(0).cuda()
        state, first_logits = blocks.prefill(prompt_ids.unsqueeze(0), T, T)
        if state.prefix_len != sample["prefix_len"]:
            raise ValueError("Source/current prefix lengths disagree; stop timing")
        cache = state.kv_cache
        n = validate_trace(sample, a.max_decode_positions, T)
        tokens = sample["tokens_ar"][:n]
        expected_next = sample["tokens_ar"][1:n+1]
        outputs, depths, events = [], [], []
        blocked_ms = 0.
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for j, input_id in enumerate(tokens):
            position = state.prefix_len + j
            hidden, static_state = blocks.prelude_forward(cache, input_id, position)
            tok = ActiveToken(token_id=int(input_id), position=position, role="verify")
            tok.hidden, tok.static_state, tok.step = hidden, static_state, 0
            forced_depth = sample["depths_ar"][j]
            prev_out = None
            for _ in range(T):
                blocks.advance_one_step([tok], cache)
                if mode in ("shadow_gpu", "shadow_readback"):
                    event, readback_ms = shadow(tok, prev_out, mode == "shadow_readback")
                    events.append(event)
                    blocked_ms += readback_ms
                    prev_out = tok.hidden
                if mode != "live" and tok.step == forced_depth:
                    if forced_depth < T:
                        tok.should_exit = True
                        blocks._freeze_kv(cache, position, forced_depth)
                    break
                if mode == "live" and tok.should_exit:
                    break
            depths.append(tok.step)
            # Preserve real Huginn CODA/KV path; defer device readback until after clock.
            logits = blocks.coda_logits(state, [tok])[0]
            outputs.append(logits.argmax(dim=-1))
        torch.cuda.synchronize()
        decode_wall_ms = (time.perf_counter() - tick) * 1000.
        computed = torch.stack(outputs).tolist()
        gate_event_ms = sum(start.elapsed_time(end) for start, end in events)
        return {"idx": sample["idx"], "repeat": repeat, "mode": mode,
                "decode_positions": n, "decode_wall_ms": decode_wall_ms,
                "input_digest": sha_json(tokens), "predicted_digest": sha_json(computed),
                "prediction_match_fraction_to_reference": sum(x == y for x, y in zip(
                    computed, expected_next)) / n,
                "depths": depths, "depths_match_reference": depths == sample["depths_ar"][:n],
                "gate_calls": len(events), "gate_event_ms": gate_event_ms,
                "readback_wait_ms": blocked_ms}

    out = []
    for rep in range(-a.warmup, a.repeats):
        # Deterministic rotation reduces a fixed mode-order bias.
        order = modes[rep % len(modes):] + modes[:rep % len(modes)]
        for sample in samples:
            for mode in order:
                one = probe(sample, mode, rep)
                if rep >= 0:
                    out.append(one)
                    print(f"[e3] idx={one['idx']} rep={rep} {mode}: "
                          f"{one['decode_wall_ms']:.2f} ms; depth_match="
                          f"{one['depths_match_reference']}", flush=True)

    summary = summarize(out)
    result = {
        "source_sha256": sha256_file(a.reference), "source_wf_commit": pinned_sha,
        "current_wf_commit": current_sha, "source": str(a.reference),
        "config": cfg, "hardware": {"gpu": torch.cuda.get_device_name(0),
                                   "torch": torch.__version__, "cuda": torch.version.cuda},
        "control": {"modes": modes, "warmup": a.warmup, "repeats": a.repeats,
                    "limit": a.limit, "max_decode_positions": a.max_decode_positions},
        "scope": "B1, eager AR teacher-forced source trace, FULL KV, decode-only; "
                 "not branch-matched speculative WFD/DtV or CUDA Graph",
        "trials": out, "summary": summary,
    }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not summary["valid_triplets"]:
        print("[e3] WARNING: zero valid A/B/C matched triplets; no gate conclusion!", flush=True)
    print(f"[e3] saved {dest}", flush=True)


if __name__ == "__main__":
    main()
