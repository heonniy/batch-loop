#!/usr/bin/env python3
"""Non-intervening gate shadow for dynamic_sd/dsd_run.py replay.

This is a controlled *eager replay diagnostic*, not an isolated CUDA event
benchmark, not a native live exit path, and not guaranteed exact native
speculative target equivalence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

DYNAMIC_SD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DYNAMIC_SD_DIR))
import dsd_run as base  # imports torch and pinned external WaveFront via WF_ROOT

torch = base.torch
_orig_init = base.Instr.__init__
COUNTERS = {"calls": 0, "rows": 0, "shadow_wall_s": 0.0, "shadow_readback_wall_s": 0.0}
SHADOW_MODE = "off"


def _hook_init(self, blocks, T_total, model_kind):
    _orig_init(self, blocks, T_total, model_kind)
    native_advance = self.orig

    @torch.no_grad()
    def advance_with_shadow(toks, cache):
        # This is invoked INSIDE the original Instr._wrapped, so the original
        # trace replay / freeze is still performed *after* this callback.
        native_advance(toks, cache)
        if SHADOW_MODE == "off" or not toks:
            return
        t0 = time.perf_counter()
        if model_kind == "ouro":
            x = torch.stack([t.hidden for t in toks], 0).unsqueeze(0)
            probabilities = torch.sigmoid(blocks.inner.early_exit_gate(x)[0, :, 0].float())
            if SHADOW_MODE == "cpu":
                b0 = time.perf_counter()
                values = probabilities.tolist()
                COUNTERS["shadow_readback_wall_s"] += time.perf_counter() - b0
                for token, value in zip(toks, values):
                    token._shadow_remaining = getattr(token, "_shadow_remaining", 1.0) * (1.0 - value)
        else:
            # With the native exit rule disabled in replay mode, native
            # _advance_core never updates .prev_hidden. Keep an independent
            # per-ActiveToken shadow history instead.
            cur = torch.stack([t.hidden for t in toks], 0).float()
            previous = torch.stack(
                [getattr(t, "_shadow_prev", None) if getattr(t, "_shadow_prev", None) is not None else cur[i]
                 for i, t in enumerate(toks)], 0
            )
            ratio = torch.linalg.vector_norm(cur - previous, dim=1) / (
                torch.linalg.vector_norm(previous, dim=1).clamp_min(1e-6)
            )
            for i, token in enumerate(toks):
                token._shadow_prev = cur[i].clone()
            if SHADOW_MODE == "cpu":
                b0 = time.perf_counter()
                _values = ratio.tolist()
                COUNTERS["shadow_readback_wall_s"] += time.perf_counter() - b0
        COUNTERS["calls"] += 1
        COUNTERS["rows"] += len(toks)
        COUNTERS["shadow_wall_s"] += time.perf_counter() - t0

    self.orig = advance_with_shadow


def main():
    global SHADOW_MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shadow-mode", required=True, choices=("off", "gpu", "cpu"))
    opts, forwarded = parser.parse_known_args()
    SHADOW_MODE = opts.shadow_mode
    if "--exit" not in forwarded or forwarded[forwarded.index("--exit") + 1] != "replay":
        parser.error("shadow gate requires --exit replay (same forced exit trace)")
    if "--replay-from" not in forwarded:
        parser.error("shadow gate requires --replay-from native AR-live JSON")
    if "--out" not in forwarded:
        parser.error("requires --out")
    out_path = Path(forwarded[forwarded.index("--out") + 1])
    if not os.environ.get("WF_ROOT"):
        parser.error("set WF_ROOT to the checked out WaveFront source")
    original_argv = sys.argv
    original_init = base.Instr.__init__
    try:
        base.Instr.__init__ = _hook_init
        sys.argv = [original_argv[0], *forwarded]
        base.main()
    finally:
        base.Instr.__init__ = original_init
        sys.argv = original_argv
    with out_path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    report["shadow_probe"] = {
        "mode": SHADOW_MODE,
        "claim_scope": "eager matched-replay geometry; wall includes prefill; no native live gate decisions",
        **COUNTERS,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("[shadow] " + json.dumps(report["shadow_probe"]), flush=True)


if __name__ == "__main__":
    main()
