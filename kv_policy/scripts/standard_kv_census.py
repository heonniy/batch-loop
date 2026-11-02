#!/usr/bin/env python3
"""Non-looped control arm: the same Context-KV eviction census and equal-budget token policies on a STANDARD
transformer (Qwen2.5-3B, pinned), same frozen Controlled-Retrieval / HotpotQA prompts and gold continuations.

What is compared with the looped models (v7/v8):
  * query mismatch: prompt-anchor attention vs 2-token lookahead vs deferred-4 vs future-query oracle token policies;
  * the LAYER analogue of loops: which layer's attention (early / late / last / mean) ranks future damage best;
  * heavy tail / evidence concentration of single-token eviction damage.
There is no plane axis in a standard transformer (one KV per token per layer), which is the point.

HF engine: clean prefill once (eager attention, per-layer attention of the question anchor recorded); every branch
decodes the gold continuation from a fresh copy of the prompt cache with a 2D attention mask whose zeros hide the
evicted prompt positions from every later query at every layer; positions come from cache_position, not the mask.

  --prepare / --execute / --worker / --analyze (standard_kv_analysis.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from query_conditioned_sweep import _json, git_sha  # noqa: E402

PROTOCOL = "standard-transformer-kv-census-v1"
STD_MODELS = {"qwen2.5-3b": ("Qwen/Qwen2.5-3B", "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b")}
V5_SEED, V5_N = 20261031, 20
GREEDY_TOKENS = 24


def _docs(ds: str):
    from kvp.role_evidence_data import load_evidence_workload
    docs, meta = load_evidence_workload(ds, V5_N, V5_SEED, controlled_facts=16, hotpot_context_mode="evidence_excerpt",
                                        hotpot_distractor_sentences=8)
    return sorted(docs, key=lambda d: d["uid"]), meta


def prepare(run: Path, cfg: dict):
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run directory")
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "kv_policy/kvp", "kv_policy/scripts/standard_kv_census.py"],
                      cwd=ROOT.parent, capture_output=True, text=True).stdout.strip():
        raise SystemExit("Commit before freezing a run")
    run.mkdir(parents=True)
    jobs, sources = [], {}
    for ds in cfg["datasets"]:
        docs, meta = _docs(ds)
        sources[ds] = meta
        for split, (lo, hi) in cfg["splits"].items():
            for d in docs[lo:hi]:
                for mk in cfg["models"]:
                    unit = f"{mk}__{ds}__{d['uid']}"
                    jobs.append({"kind": "UNIT", "key": f"UNIT__{unit}", "model": mk, "dataset": ds, "uid": d["uid"], "unit": unit,
                                 "split": split})
    _json(run / "jobs.json", jobs)
    _json(run / "run_manifest.json", {"protocol": PROTOCOL, "code_sha": git_sha(), "config": cfg, "dataset_sources": sources,
                                      "models": {k: {"repo": v[0], "revision": v[1]} for k, v in STD_MODELS.items() if k in cfg["models"]},
                                      "notes": ["Same frozen v5 pilot20 documents (seed 20261031) rebuilt with the standard tokenizer",
                                                "Plain-text prompt (no chat template), gold 'Answer: <x>\\n' continuation; first target excluded",
                                                "Eviction = 2D attention mask zeros on prompt positions for all later queries, all layers",
                                                "No plane axis exists; budgets are token fractions"]})
    return {"jobs": len(jobs)}


# ------------------------------------------------------------------------------------------- engine
class StdUnit:
    def __init__(self, model, tok, prompt, targets, device):
        import torch
        from transformers import DynamicCache
        self.model, self.tok, self.p, self.device = model, tok, prompt, device
        self.ids = torch.tensor([list(prompt.token_ids)], device=device)
        self.P = len(prompt.token_ids)
        self.targets = [int(x) for x in targets]
        with torch.no_grad():
            out = model(input_ids=self.ids, use_cache=True, output_attentions=True)
        qa = max(prompt.question_positions)
        self.anchors = {"question_last": qa, "prompt_last": self.P - 1}
        # per-layer head-mean attention rows of the anchors over the prompt: (L, P)
        self.anchor_attn = {name: torch.stack([a[0, :, col, :].float().mean(0) for a in out.attentions]).cpu().numpy()
                            for name, col in self.anchors.items()}
        self.L = len(out.attentions)
        self.last_prompt_lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        self.prefix = out.past_key_values
        self.DynamicCache = DynamicCache
        self.n = len(self.targets)
        del out
        self.ref_lp, self.gen_attn = self._run(evicted=(), record=True)
        self.ref_ids = self.targets

    def _fresh(self):
        legacy = tuple((k[..., :self.P, :].clone(), v[..., :self.P, :].clone())
                       for k, v in zip(self.prefix.key_cache, self.prefix.value_cache))
        return self.DynamicCache.from_legacy_cache(legacy)

    def _run(self, evicted=(), record=False, ids=None, greedy=False, max_new=None):
        """Teacher-forced (ids=targets) or greedy decode from a fresh prompt cache with `evicted` masked."""
        import torch
        cache = self._fresh()
        n = self.n if not greedy else max_new
        mask = torch.ones(1, self.P + n, dtype=torch.long, device=self.device)
        if evicted:
            mask[0, list(evicted)] = 0
        lps, attn = [], []
        cur = self.targets[0] if not greedy else int(self.last_prompt_lp.argmax())
        out_ids = [cur]
        with torch.no_grad():
            for t in range(1, n + 1):
                col = self.P + t - 1
                o = self.model(input_ids=torch.tensor([[cur]], device=self.device), past_key_values=cache, use_cache=True,
                               attention_mask=mask[:, :col + 1], position_ids=torch.tensor([[col]], device=self.device),
                               output_attentions=record)
                cache = o.past_key_values
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
                lps.append(lp)
                if record:
                    attn.append(torch.stack([a[0, :, 0, :self.P].float().mean(0) for a in o.attentions]).cpu().numpy())
                if greedy:
                    cur = int(lp.argmax())
                    out_ids.append(cur)
                    if cur in self.eos:
                        break
                elif t < n:
                    cur = self.targets[t]
        if greedy:
            return out_ids
        if record:
            import numpy as np
            return torch.stack(lps), np.stack(attn)          # (n, V), (n, L, P)
        return torch.stack(lps)

    def score(self, evicted=()):
        lps = self._run(evicted)
        kl = (self.ref_lp.exp() * (self.ref_lp - lps)).sum(-1)
        losses = [float(-self.last_prompt_lp[self.targets[0]])] + [float(-lps[t - 1][self.targets[t]]) for t in range(1, self.n)]
        ref = [float(-self.last_prompt_lp[self.targets[0]])] + [float(-self.ref_lp[t - 1][self.targets[t]]) for t in range(1, self.n)]
        top1 = (lps.argmax(-1) != self.ref_lp.argmax(-1)).float()
        return {"token_losses": losses, "nll_future": sum(losses[1:]) / max(1, len(losses) - 1),
                "delta_nll_future": (sum(losses[1:]) - sum(ref[1:])) / max(1, len(losses) - 1),
                "kl_by_t": kl.tolist(), "top1_changed": top1.tolist(), "first_target_nll": losses[0]}


def evaluate(model, tok, eos, job, cfg, run: Path):
    import numpy as np
    import torch
    from kvp.role_data import build_evidence_prompt
    from kvp.role_generation import score_free_answer
    raw = run / "raw"
    docs, _ = _docs(job["dataset"])
    doc = next(d for d in docs if d["uid"] == job["uid"])
    p = build_evidence_prompt(tok, "ouro", doc)         # plain text; exact tokenizer-offset roles
    target_text = "Answer: " + doc["answer"].strip() + "\n"
    targets = tok.encode(target_text, add_special_tokens=False)[:64]
    t0 = time.time()
    u = StdUnit(model, tok, p, targets, next(model.parameters()).device)
    u.eos = eos
    t_pre = time.time() - t0
    content = [int(c) for c in p.content_positions]
    P, L, n = u.P, u.L, u.n
    # validation: no-mask == reference; masked differs; repeat identical; first target invariant
    v0 = u.score(())
    ref_nll = [float(-u.last_prompt_lp[targets[0]])] + [float(-u.ref_lp[t - 1][targets[t]]) for t in range(1, n)]
    val = {"nomask_max_abs_nll": max(abs(a - b) for a, b in zip(v0["token_losses"], ref_nll)),
           "nomask_max_kl": max(v0["kl_by_t"])}
    va = u.score((content[len(content) // 3],))
    va2 = u.score((content[len(content) // 3],))
    val["replay_max_abs"] = max(abs(a - b) for a, b in zip(va["token_losses"], va2["token_losses"]))
    val["masked_differs"] = max(va["kl_by_t"]) > 0
    val["first_target_invariant"] = abs(va["first_target_nll"] - ref_nll[0])
    val["pass"] = val["nomask_max_abs_nll"] == 0 and val["nomask_max_kl"] == 0 and val["replay_max_abs"] == 0 \
        and val["masked_differs"] and val["first_target_invariant"] == 0
    if not val["pass"]:
        raise RuntimeError(f"standard-engine validation failed: {val}")
    # exhaustive single-token census
    t1 = time.time()
    cen = {k: np.full((len(content), n), np.nan, dtype="float32") for k in ("kl", "nll_shift", "top1")}
    for si, s in enumerate(content):
        d = u.score((s,))
        cen["kl"][si] = d["kl_by_t"]
        cen["nll_shift"][si, :n - 1] = np.array(d["token_losses"][1:]) - np.array(ref_nll[1:])
        cen["top1"][si] = d["top1_changed"]
    t_cen = time.time() - t1
    # policies at equal token budgets
    ga = u.gen_attn                                        # (n, L, P)
    qa = u.anchor_attn["question_last"]                     # (L, P)
    q4 = max(1, L // 4)
    scores = {"qa_last_layer": qa[-1], "qa_mean": qa.mean(0), "qa_early": qa[:q4].mean(0), "qa_late": qa[-q4:].mean(0),
              "qa_max_layer": qa.max(0),
              "la2_mean": ga[:2].mean(1).max(0), "la2_late": ga[:2, -q4:].mean(1).max(0), "la2_early": ga[:2, :q4].mean(1).max(0),
              "def4_mean": ga[:4].mean(1).max(0), "def4_late": ga[:4, -q4:].mean(1).max(0), "def4_early": ga[:4, :q4].mean(1).max(0),
              "fut_oracle_mean": ga.mean(1).max(0), "fut_oracle_late": ga[:, -q4:].mean(1).max(0),
              "fut_oracle_early": ga[:, :q4].mean(1).max(0)}
    for sd in (11, 29):
        rng = np.random.default_rng(int(hashlib.sha256(f"{job['unit']}/{sd}".encode()).hexdigest()[:8], 16))
        scores[f"random_{sd}"] = rng.random(P)
    gold = doc["answer"]
    g_full = u._run((), greedy=True, max_new=GREEDY_TOKENS)
    rows = []
    t2 = time.time()
    for keep in cfg["retention"]:
        n_keep = max(1, int(round(keep * len(content))))
        for name, sc in scores.items():
            order = sorted(range(len(content)), key=lambda j: (-float(sc[content[j]]), int(content[j])))
            ev = sorted(int(content[j]) for j in order[n_keep:])
            d = u.score(tuple(ev))
            g = u._run(tuple(ev), greedy=True, max_new=GREEDY_TOKENS)
            em = score_free_answer(tok, gold, {"token_ids": g, "decoded_tokens": len(g)})
            rows.append({"retention": keep, "policy": name, "n_evicted": len(ev), "n_kept": len(content) - len(ev),
                         "delta_nll_future": d["delta_nll_future"], "kl_mean": float(np.mean(d["kl_by_t"])),
                         "exact_match": em["exact_match"], "extracted": em["extracted_answer"][:40]})
    np.savez_compressed(raw / f"{job['key']}.npz", sources=np.array(content), qa=qa.astype("float32"),
                        qb=u.anchor_attn["prompt_last"].astype("float32"), gen_attn=ga.astype("float32"),
                        **{f"cen_{k}": v for k, v in cen.items()})
    _json(raw / f"{job['key']}.json", {k: job[k] for k in ("key", "model", "dataset", "uid", "unit", "split")} | {
        "sha": git_sha(), "prompt_ids": list(p.token_ids), "roles": list(p.roles), "content_positions": content,
        "relevant_positions": list(p.relevant_positions), "anchors": u.anchors, "target_ids": targets, "n_layers": L,
        "validation": val, "ref_nll": ref_nll, "policy_rows": rows, "gold_answer": gold,
        "full_exact_match": score_free_answer(tok, gold, {"token_ids": g_full, "decoded_tokens": len(g_full)})["exact_match"],
        "timing": {"prefill_ref_s": t_pre, "census_s": t_cen, "policy_s": time.time() - t2},
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


def _load_std(mk: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev = STD_MODELS[mk]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, (list, tuple)) else [gc.eos_token_id])
    return tok, model, eos


def worker(run: Path, idx: int):
    import torch
    m = json.loads((run / "run_manifest.json").read_text())
    if m["code_sha"] != git_sha():
        raise RuntimeError("Code SHA changed; use a NEW run directory")
    jobs = json.loads((run / "jobs.json").read_text())
    raw, claims, errors = (run / d for d in ("raw", "claims", "errors"))
    for d in (raw, claims, errors):
        d.mkdir(exist_ok=True)
    cur = tok = model = eos = None
    while True:
        pend = [j for j in jobs if not (raw / f"{j['key']}.json").exists() and not (claims / f"{j['key']}.lock").exists()]
        if not pend:
            return
        job = sorted(pend, key=lambda j: (j["model"] != cur, j["key"]))[0]
        try:
            fd = os.open(claims / f"{job['key']}.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"worker={idx} pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        try:
            if cur != job["model"]:
                tok, model, eos = _load_std(job["model"])
                cur = job["model"]
            torch.cuda.reset_peak_memory_stats()
            evaluate(model, tok, eos, job, m["config"], run)
            print(f"done {job['key']}", flush=True)
        except Exception:
            _json(errors / f"{job['key']}.json", {"key": job["key"], "traceback": traceback.format_exc()})
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    for nm in ("prepare", "execute", "analyze"):
        g.add_argument("--" + nm, action="store_true")
    g.add_argument("--worker", type=int)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default="qwen2.5-3b")
    ap.add_argument("--datasets", default="controlled_retrieval,hotpotqa_distractor")
    ap.add_argument("--dev", default="0:8")
    ap.add_argument("--holdout", default="8:20")
    ap.add_argument("--retention", default="0.1,0.3,0.5")
    ap.add_argument("--gpu-ids", default="0")
    a = ap.parse_args()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    run = Path(a.run_dir).resolve()
    if a.prepare:
        rng = lambda s: tuple(int(x) for x in s.split(":"))  # noqa: E731
        cfg = {"models": sp(a.models), "datasets": sp(a.datasets), "splits": {"dev": rng(a.dev), "holdout": rng(a.holdout)},
               "retention": [float(x) for x in sp(a.retention)], "greedy_tokens": GREEDY_TOKENS}
        print(json.dumps(prepare(run, cfg), indent=2))
    elif a.worker is not None:
        worker(run, a.worker)
    elif a.analyze:
        from standard_kv_analysis import analyze
        print(json.dumps(analyze(run), indent=2, default=str))
    else:
        (run / "logs").mkdir(exist_ok=True)
        ps = []
        for i, gpu in enumerate(sp(a.gpu_ids)):
            h = (run / "logs" / f"worker{i}.txt").open("a")
            ps.append((subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--run-dir", str(run)],
                                        cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=h, stderr=subprocess.STDOUT), h))
        codes = [p.wait() for p, _ in ps]
        print(json.dumps({"worker_exit_codes": codes}))
        if any(codes):
            sys.exit(1)


if __name__ == "__main__":
    main()
