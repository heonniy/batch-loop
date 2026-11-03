#!/usr/bin/env python3
"""Standard-transformer control for v9: the same Context-window eviction and the same FFN-activation restorations
(gate-only / up-only / shifted-gate / whole-FFN; layer-quarter groups as the analogue of recurrent loop groups) on
Qwen2.5-3B, same frozen prompts, same label-blind 4-token windows (scripts.ffn_gate_causal_sweep.source_windows), same
evaluation UIDs as the v9 looped cells (first n_cal UIDs skipped, next n_eval evaluated).

Qwen2MLP: down_proj(act(gate_proj(x)) * up_proj(x)) — identical SwiGLU algebra to Ouro, so patch_projection is reused.
Teacher-forced gold continuation; eviction = 2D attention mask zeros on the window for every generated query.

  python scripts/std_ffn_gate_control.py --dataset controlled_retrieval --run-dir results/<dir> --n-cal 3 --n-eval 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffn_gate_causal_sweep import source_windows  # noqa: E402
from kvp.ffn_gate_causal import patch_projection  # noqa: E402
from query_conditioned_sweep import _json, git_sha  # noqa: E402
from standard_kv_census import StdUnit, _docs, _load_std  # noqa: E402

EPS = 1e-8


class StdFFNHooks:
    """Per-(generated column, layer, component) capture / restoration on Qwen2 MLPs (no loops: one call per layer)."""

    def __init__(self, model, *, mode: str, reference=None, selected_layers=None, selected_columns=None):
        self.model, self.mode = model, mode
        self.reference = reference or {}
        self.layers = None if selected_layers is None else frozenset(int(l) for l in selected_layers)
        self.cols = None if selected_columns is None else frozenset(int(c) for c in selected_columns)
        self.trace, self.stats, self.handles = {}, defaultdict(list), []
        self.col, self.patched, self.calls = None, 0, {}

    def begin_column(self, col):
        self.col, self.calls = int(col), {}

    def end_column(self):
        n = len(self.model.model.layers)
        for l in range(n):
            for c in ("gate", "up", "ffn"):
                if self.calls.get((l, c), 0) != 1:
                    raise RuntimeError(f"FFN hook call mismatch at layer {l}/{c}")
        self.col = None

    def _hook(self, layer, component):
        def handler(module, args, output):
            if self.col is None:
                raise RuntimeError("hook without active column")
            self.calls[(layer, component)] = self.calls.get((layer, component), 0) + 1
            if (self.layers is not None and layer not in self.layers) or (self.cols is not None and self.col not in self.cols):
                return output
            key = (self.col, layer, component)
            if self.mode == "capture":
                self.trace[key] = output.detach().clone()
                return output
            action = "ffn" if self.mode == "restore_ffn" else ("up" if self.mode == "restore_up" else "gate")
            if component != action:
                return output
            clean = self.reference[key].to(output.device, output.dtype)
            changed = patch_projection(output, clean, family="ouro", component=component, mode=self.mode)
            self.patched += 1
            self.stats["relative_patch_norm"].append(float(((changed.float() - output.float()).norm() / (output.float().norm() + EPS)).item()))
            return changed
        return handler

    def __enter__(self):
        for l, layer in enumerate(self.model.model.layers):
            self.handles += [layer.mlp.gate_proj.register_forward_hook(self._hook(l, "gate")),
                             layer.mlp.up_proj.register_forward_hook(self._hook(l, "up")),
                             layer.mlp.register_forward_hook(self._hook(l, "ffn"))]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.col = None


class StdGateUnit(StdUnit):
    """StdUnit whose teacher-forced decode notifies the FFN hooks of the current column."""
    hooks: StdFFNHooks | None = None

    def _run(self, evicted=(), record=False, ids=None, greedy=False, max_new=None):
        if greedy or record:
            return super()._run(evicted, record, ids, greedy, max_new)
        cache = self._fresh()
        n = self.n
        mask = torch.ones(1, self.P + n, dtype=torch.long, device=self.device)
        if evicted:
            mask[0, list(evicted)] = 0
        lps = []
        cur = self.targets[0]
        with torch.no_grad():
            for t in range(1, n + 1):
                col = self.P + t - 1
                if self.hooks is not None:
                    self.hooks.begin_column(col)
                o = self.model(input_ids=torch.tensor([[cur]], device=self.device), past_key_values=cache, use_cache=True,
                               attention_mask=mask[:, :col + 1], position_ids=torch.tensor([[col]], device=self.device))
                if self.hooks is not None:
                    self.hooks.end_column()
                cache = o.past_key_values
                lps.append(torch.log_softmax(o.logits[0, -1].float(), -1))
                if t < n:
                    cur = self.targets[t]
        return torch.stack(lps)

    def arm(self, evicted, mode="capture", reference=None, selected_layers=None):
        self.hooks = StdFFNHooks(self.model, mode=mode, reference=reference, selected_layers=selected_layers)
        try:
            with self.hooks:
                sc = self.score(tuple(evicted))
        finally:
            h, self.hooks = self.hooks, None
        return sc | {"trace": h.trace, "patches": h.patched, "patch_stats": dict(h.stats)}


def evaluate(model, tok, eos, doc, width, out_path):
    from kvp.role_data import build_evidence_prompt
    p = build_evidence_prompt(tok, "ouro", doc)
    target_text = "Answer: " + doc["answer"].strip() + "\n"
    targets = tok.encode(target_text, add_special_tokens=False)[:64]
    u = StdGateUnit(model, tok, p, targets, next(model.parameters()).device)
    u.eos = eos
    L = len(model.model.layers)
    clean = u.arm(())
    parity = u.arm((), mode="restore_gate", reference=clean["trace"])
    gap = max(abs(a - b) for a, b in zip(clean["token_losses"], parity["token_losses"]))
    if gap > 1e-5:
        raise RuntimeError(f"clean gate restoration not parity: {gap}")
    cands = source_windows(p, doc["uid"], width)
    q = max(1, L // 4)
    groups = [(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, L)]
    out = {"uid": doc["uid"], "n_prompt": u.P, "n_targets": u.n, "width": width, "full_nll": clean["nll_future"], "n_layers": L,
           "candidates": []}
    t0 = time.time()
    for c in cands:
        src = c["positions"]
        ev = u.arm(src)
        rest = {}
        for mode in ("restore_gate", "restore_up", "shift_gate", "restore_ffn"):
            a = u.arm(src, mode=mode, reference=clean["trace"])
            st = a["patch_stats"].get("relative_patch_norm", [])
            rest[mode] = {"dnll_vs_clean": a["nll_future"] - clean["nll_future"], "recovered": ev["nll_future"] - a["nll_future"],
                          "patches": a["patches"], "relpatch_mean": float(np.mean(st)) if st else None,
                          "relpatch_max": float(np.max(st)) if st else None}
        for lo, hi in groups:
            a = u.arm(src, mode="restore_gate", reference=clean["trace"], selected_layers=range(lo, hi))
            st = a["patch_stats"].get("relative_patch_norm", [])
            rest[f"gate_layers_{lo + 1}_{hi}"] = {"dnll_vs_clean": a["nll_future"] - clean["nll_future"],
                                                  "recovered": ev["nll_future"] - a["nll_future"], "patches": a["patches"],
                                                  "relpatch_mean": float(np.mean(st)) if st else None}
        # trace change by layer quarter (evicted vs clean, unpatched)
        chg = defaultdict(list)
        for k, v in clean["trace"].items():
            col, l, comp = k
            if comp == "gate":
                e = ev["trace"][k]
                chg[min(3, l // q)].append(float(((e.float() - v.float()).norm() / (v.float().norm() + EPS)).item()))
        out["candidates"].append({"source": src, "stratum": c["stratum"], "evict_dnll": ev["nll_future"] - clean["nll_future"],
                                  "evict_kl": float(np.mean(ev["kl_by_t"])), "restorations": rest,
                                  "gate_change_by_layer_quarter": [float(np.mean(chg[i])) if chg[i] else None for i in range(4)]})
    out["elapsed_s"] = time.time() - t0
    _json(out_path, out)
    return out


def summarize(run: Path):
    rows = []
    for f in sorted((run / "raw").glob("*.json")):
        d = json.loads(f.read_text())
        for c in d["candidates"]:
            r = {"uid": d["uid"], "stratum": c["stratum"], "evict_dnll": c["evict_dnll"], "evict_kl": c["evict_kl"]}
            for k, v in c["restorations"].items():
                r[f"rec_{k}"] = v["recovered"]
                r[f"relpatch_{k}"] = v.get("relpatch_mean")
                r[f"relpatchmax_{k}"] = v.get("relpatch_max")
            for i, x in enumerate(c["gate_change_by_layer_quarter"]):
                r[f"gatechg_q{i + 1}"] = x
            rows.append(r)
    md = [f"# Standard-transformer FFN restoration control ({run.name})", "",
          "| stratum | n | ΔNLL evict | gate rec | up rec | shift rec | ffn rec | relpatch gate / up / ffn (mean, max) | gate change by layer quarter |",
          "|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    m = lambda g, k: (statistics.mean(x[k] for x in g if x.get(k) is not None) if any(x.get(k) is not None for x in g) else None)  # noqa: E731
    f = lambda x: "—" if x is None else f"{x:.4g}"  # noqa: E731
    for s, g in sorted(by.items()):
        md.append(f"| {s} | {len(g)} | {f(m(g, 'evict_dnll'))} | {f(m(g, 'rec_restore_gate'))} | {f(m(g, 'rec_restore_up'))} | {f(m(g, 'rec_shift_gate'))} | "
                  f"{f(m(g, 'rec_restore_ffn'))} | {f(m(g, 'relpatch_restore_gate'))}/{f(m(g, 'relpatchmax_restore_gate'))}, "
                  f"{f(m(g, 'relpatch_restore_up'))}/{f(m(g, 'relpatchmax_restore_up'))}, {f(m(g, 'relpatch_restore_ffn'))}/{f(m(g, 'relpatchmax_restore_ffn'))} | "
                  + ", ".join(f(m(g, f'gatechg_q{i}')) for i in range(1, 5)) + " |")
    md += ["", "| stratum | gate layers 1–9 | 10–18 | 19–27 | 28–36 |", "|---|---|---|---|---|"]
    for s, g in sorted(by.items()):
        ks = sorted((k for k in g[0] if k.startswith("rec_gate_layers_")), key=lambda k: int(k.split("_")[3]))
        md.append(f"| {s} | " + " | ".join(f(m(g, k)) for k in ks) + " |")
    (run / "analysis").mkdir(exist_ok=True)
    (run / "analysis" / "STD_FFN_CONTROL.md").write_text("\n".join(md))
    _json(run / "analysis" / "rows.json", rows)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-cal", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=5)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir)
    if a.summarize:
        summarize(run)
        return
    run.mkdir(parents=True, exist_ok=True)
    docs, meta = _docs(a.dataset)
    ev = docs[a.n_cal:a.n_cal + a.n_eval]
    _json(run / "manifest.json", {"model": "qwen2.5-3b", "dataset": a.dataset, "code_sha": git_sha(), "width": a.width,
                                  "evaluation_uids": [d["uid"] for d in ev], "n_cal_skipped": a.n_cal})
    tok, model, eos = _load_std("qwen2.5-3b")
    (run / "raw").mkdir(exist_ok=True)
    for d in ev:
        dest = run / "raw" / (d["uid"].replace("/", "_") + ".json")
        if dest.exists():
            continue
        evaluate(model, tok, eos, d, a.width, dest)
        print("complete", d["uid"], flush=True)
    summarize(run)


if __name__ == "__main__":
    main()
