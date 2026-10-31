#!/usr/bin/env python3
"""LoopZip v6 (LOOPZIP_PERSISTENT_RECURRENCE_TRAJECTORY_PLAN.md): trajectory census + persistent eviction.

  --preflight   G0 on ONE GPU: real-checkpoint parity gate + timing/memory profile (N sources, 16/128 windows)
  --prepare     CPU: freeze the master manifest (W1/W2 x Stage A / Stage B-C / holdout / reserve) and jobs
  --execute     GPU workers (one model per H200 worker, B=1); jobs with unmet dependencies wait
  --analyze     CPU: completeness (fail closed) + long tables + summaries

Job kinds: A (normal census: prefill + FULL greedy continuation), Bref (Stage-B reference: frozen
continuation, G0 validation, source features), B (chunk of eligible sources: accumulated persistent
eviction over every t and r + clean-history probes at preregistered t).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

PROTOCOL = "loopzip-v6-persistent-trajectory-v1"
SEED = 20261031
CLEAN_TS = (1, 2, 4, 8, 16, 32, 64, 128)
MODELS_V6 = ("ouro-2.6b", "huginn-0125")


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def git_dirty() -> bool:
    return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "kv_policy/kvp",
                                "kv_policy/scripts/loopzip_trajectory_sweep.py"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip())


def _json(path: Path, value):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def _hotpot_frame():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from kvp.role_evidence_data import HOTpot_FILE, HOTpot_REPO, HOTpot_REVISION, HOTpot_SHA256, digest
    path = Path(hf_hub_download(HOTpot_REPO, filename=HOTpot_FILE, repo_type="dataset", revision=HOTpot_REVISION))
    if digest(path.read_bytes()) != HOTpot_SHA256:
        raise RuntimeError("HotpotQA parquet checksum mismatch")
    return pd.read_parquet(path)


def _tokenizers(models):
    from transformers import AutoTokenizer
    from kvp.specs import MODELS
    return {m: (MODELS[m].family, AutoTokenizer.from_pretrained(MODELS[m].repo, revision=MODELS[m].revision,
                                                               trust_remote_code=True)) for m in models}


def _prompt(tok, family, item):
    from kvp.loopzip_trajectory_data import build_trajectory_prompt
    return build_trajectory_prompt(tok, family, item["context"], item["instruction"], item["question"])


# --------------------------------------------------------------------------------------------- prepare
def prepare(run: Path, cfg: dict):
    from kvp.loopzip_persistent_eviction import eligible_sources
    from kvp.loopzip_trajectory_data import candidate_items, fit_item
    from kvp.specs import MODELS

    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run directory")
    if git_dirty():
        raise SystemExit("Commit kvp/ and the sweep script before freezing a manifest")
    run.mkdir(parents=True, exist_ok=True)
    frame = _hotpot_frame()
    toks = _tokenizers(cfg["models"])
    lo, hi = cfg["length_window"]
    nA, nB, nH, nR = cfg["n_stage_a"], cfg["n_stage_b"], cfg["n_holdout"], cfg["n_reserve"]
    manifest, skipped = [], {}
    used_sources = set()
    for w in ("W1", "W2"):
        fitted, skip = [], 0
        for it in candidate_items(frame, w, SEED):
            if it["source_id"] in used_sources:
                continue  # one HotpotQA row feeds at most one prompt across W1/W2 and pools
            f = fit_item(it, toks, lo, hi)
            if f is None:
                skip += 1
                continue
            fitted.append({**it, **f})
            used_sources.add(it["source_id"])
            if len(fitted) >= nA + nB * 4 + nH + nR:
                break
        skipped[w] = skip
        pools = {"A": fitted[:nA]}
        rest = fitted[nA:]
        if w == "W2":  # Stage B balanced over the four operations, chosen by order only
            b, seen = [], set()
            for it in rest:
                if it["op"] not in seen and len(b) < nB:
                    b.append(it)
                    seen.add(it["op"])
        else:
            b = rest[:nB]
        pools["B"] = b
        rest = [it for it in rest if it not in b]
        pools["holdout"] = rest[:nH]
        pools["reserve"] = rest[nH:nH + nR]
        for pool, items in pools.items():
            for it in items:
                manifest.append({"workload": w, "pool": pool, **{k: it[k] for k in (
                    "item_id", "source_id", "titles", "op", "instruction", "question", "context", "n_sent",
                    "lengths")}})
    jobs = []
    for m in cfg["models"]:
        fam, tok = toks[m]
        for it in manifest:
            if it["pool"] not in ("A", "B"):
                continue
            p = _prompt(tok, fam, it)
            elig = eligible_sources(p)
            base = {"model": m, "workload": it["workload"], "item_id": it["item_id"], "uid": f"v6/{it['item_id']}",
                    "prompt_len": len(p.token_ids), "n_eligible": len(elig)}
            if it["pool"] == "A":
                jobs.append({**base, "kind": "A", "key": f"A__{m}__{it['item_id']}"})
            else:
                ref = f"Bref__{m}__{it['item_id']}"
                jobs.append({**base, "kind": "Bref", "key": ref})
                cs = cfg["chunk_size"][m]
                for k in range(0, len(elig), cs):
                    jobs.append({**base, "kind": "B", "key": f"B__{m}__{it['item_id']}__{k // cs:03d}",
                                 "depends": ref, "sources": elig[k:k + cs]})
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (run / "jobs.json").write_text(json.dumps(jobs, indent=1))
    _json(run / "run_manifest.json", {
        "protocol": PROTOCOL, "code_sha": git_sha(), "config": cfg, "seed": SEED,
        "model_revisions": {m: MODELS[m].revision for m in cfg["models"]},
        "skipped_length_ineligible": skipped, "clean_ts": list(CLEAN_TS),
        "pool_counts": {f"{w}/{p}": sum(1 for x in manifest if x["workload"] == w and x["pool"] == p)
                        for w in ("W1", "W2") for p in ("A", "B", "holdout", "reserve")},
        "eligibility_rule": "roles content/instruction/question; 'other' (format/special/boundary) excluded",
        "notes": ["Full-sized reference KV + read masks; no HBM/latency claims",
                  "Primary eviction: all recurrent loops/layers + plain prelude/coda layers, post-prefill",
                  "Greedy FULL continuation to EOS or horizon; early EOS recorded, never dropped",
                  "Stage B chunks teacher-force the frozen Bref continuation"]})
    return {"items": len(manifest), "jobs": len(jobs), "skipped": skipped}


# --------------------------------------------------------------------------------------------- GPU side
def _load(model_key):
    from kvp.models import eos_ids, load, make_runner
    spec, tok, model = load(model_key)
    return tok, make_runner(spec, tok, model).bb, eos_ids(spec, tok, model)


def _trajectory_arrays(rows):
    import numpy as np
    keys = ("h_norm", "U", "F", "C", "cos_init", "cos_term")
    return {k: np.stack([r[k].numpy() for r in rows]).astype("float32") for k in keys}


def run_reference(bb, tok, eos, item, uid, horizon, frozen=None, validate=False, sources=None):
    from kvp.loopzip_persistent_eviction import PersistentEvictionRunner, eligible_sources
    from kvp.specs import MODELS  # noqa: F401
    p = _prompt(tok, bb.family, item)
    t0 = time.time()
    r = PersistentEvictionRunner(bb, p, uid, horizon=horizon)
    t_prefill = time.time() - t0
    ref = r.reference(eos, frozen_ids=frozen)
    t_ref = time.time() - t0 - t_prefill
    out = {"runner": r, "prompt": p, "ref": ref, "t_prefill": t_prefill, "t_ref": t_ref,
           "eligible": eligible_sources(p)}
    if validate:
        el = out["eligible"]
        out["validation"] = r.validate(el[len(el) // 3], el[(2 * len(el)) // 3], t_probe=min(8, ref["n"]))
    return out


def evaluate(bb, tok, eos, job, cfg, run: Path):
    import numpy as np
    import torch
    item = next(x for x in json.loads((run / "manifest.json").read_text()) if x["item_id"] == job["item_id"])
    meta = {k: job[k] for k in ("key", "kind", "model", "workload", "item_id", "uid")} | {"sha": git_sha()}
    raw = run / "raw"
    if job["kind"] in ("A", "Bref"):
        o = run_reference(bb, tok, eos, item, job["uid"], cfg["horizon"], validate=job["kind"] == "Bref")
        r = o["runner"]
        if len(o["prompt"].token_ids) != job["prompt_len"]:
            raise RuntimeError("Prompt length differs from frozen manifest")
        if o.get("validation") and not o["validation"]["pass"]:
            raise RuntimeError(f"G0 validation failed: {o['validation']}")
        arrs = {f"prompt_{k}": v for k, v in _trajectory_arrays(r.prompt_metrics).items()}
        arrs |= {f"gen_{k}": v for k, v in _trajectory_arrays(r.ref_metrics).items()}
        sf = r.source_features()
        arrs |= {f"src_{k}": v.numpy().astype("float32") for k, v in sf.items()}
        lp = r.ref_lp
        arrs["gen_entropy"] = (-(lp.exp() * lp).sum(-1)).cpu().numpy().astype("float32")
        np.savez_compressed(raw / f"{job['key']}.npz", **arrs)
        _json(raw / f"{job['key']}.json", meta | {
            "prompt_ids": list(o["prompt"].token_ids), "roles": list(o["prompt"].roles),
            "eligible": o["eligible"], "ref": o["ref"], "validation": o.get("validation"),
            "ref_state_digest": float(r.ref_H.norm(dim=-1).sum()),
            "timing": {"prefill_s": o["t_prefill"], "reference_s": o["t_ref"]},
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        return
    ref = json.loads((raw / f"{job['depends']}.json").read_text())
    o = run_reference(bb, tok, eos, item, job["uid"], cfg["horizon"], frozen=ref["ref"]["ids"])
    r = o["runner"]
    dig = float(r.ref_H.norm(dim=-1).sum())
    if o["ref"]["frozen_argmax_agreement"] < 1.0 or abs(dig - ref["ref_state_digest"]) > 1e-3 * max(1.0, abs(dig)):
        raise RuntimeError(f"Frozen reference not reproduced: agree={o['ref']['frozen_argmax_agreement']} "
                           f"digest {dig} vs {ref['ref_state_digest']}")
    n, R = o["ref"]["n"], bb.R
    N = len(job["sources"])
    f32 = lambda *s: np.full(s, np.nan, dtype="float32")  # noqa: E731
    A = {k: f32(N, n, R + 1) for k in ("E", "U_ev", "dircos")}
    A |= {k: f32(N, n) for k in ("kl", "max_logprob_diff", "top1_changed", "entropy_shift", "nll_shift")}
    ts = [t for t in CLEAN_TS if t <= n]
    C = {k: f32(N, len(ts), R + 1) for k in ("E", "U_ev", "dircos")}
    C |= {k: f32(N, len(ts)) for k in ("kl", "max_logprob_diff", "top1_changed", "entropy_shift", "nll_shift")}
    times = []
    for si, s in enumerate(job["sources"]):
        t0 = time.time()
        acc = r.accumulated((s,))
        for k in ("E", "U_ev", "dircos"):
            A[k][si] = acc[k].numpy()
        for ti, tk in enumerate(acc["token"]):
            for k in ("kl", "max_logprob_diff", "top1_changed", "entropy_shift", "nll_shift"):
                v = tk[k]
                A[k][si, ti] = np.nan if v is None else float(v)
        cl = r.clean_probes((s,), ts)
        for ti, t in enumerate(ts):
            for k in ("E", "U_ev", "dircos"):
                C[k][si, ti] = cl[t][k].numpy()
            for k in ("kl", "max_logprob_diff", "top1_changed", "entropy_shift", "nll_shift"):
                v = cl[t]["token"][k]
                C[k][si, ti] = np.nan if v is None else float(v)
        times.append(time.time() - t0)
    for k, v in A.items():
        if k != "nll_shift" and np.isnan(v[..., 1:] if v.ndim == 3 else v).any():
            raise RuntimeError(f"NaN in accumulated {k}")
    np.savez_compressed(raw / f"{job['key']}.npz", sources=np.array(job["sources"]), clean_ts=np.array(ts),
                        **{f"acc_{k}": v for k, v in A.items()}, **{f"clean_{k}": v for k, v in C.items()})
    _json(raw / f"{job['key']}.json", meta | {"sources": job["sources"], "n_tokens": n, "clean_ts": ts,
                                              "per_source_s": times, "reference_digest": dig,
                                              "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


def worker(run: Path, idx: int, model_filter: str | None):
    import torch
    m = json.loads((run / "run_manifest.json").read_text())
    if m["code_sha"] != git_sha():
        raise RuntimeError("Code SHA changed; use a NEW run directory")
    jobs = json.loads((run / "jobs.json").read_text())
    if model_filter:
        jobs = [j for j in jobs if j["model"] == model_filter]
    raw, claims, errors = (run / d for d in ("raw", "claims", "errors"))
    for d in (raw, claims, errors):
        d.mkdir(exist_ok=True)
    cur = tok = bb = eos = None
    order = {"Bref": 0, "B": 1, "A": 2}
    while True:
        pend = [j for j in jobs if not (raw / f"{j['key']}.json").exists()
                and not (claims / f"{j['key']}.lock").exists()]
        if not pend:
            return
        ready = [j for j in pend if not j.get("depends") or (raw / f"{j['depends']}.json").exists()]
        if not ready:
            if any((errors / f"{j['depends']}.json").exists() for j in pend if j.get("depends")):
                raise RuntimeError("Dependency failed")
            time.sleep(20)
            continue
        ready.sort(key=lambda j: (j["model"] != cur, order[j["kind"]], j["model"], j["key"]))
        job = ready[0]
        try:
            fd = os.open(claims / f"{job['key']}.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"worker={idx} pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        try:
            if cur != job["model"]:
                tok = bb = None
                torch.cuda.empty_cache()
                tok, bb, eos = _load(job["model"])
                cur = job["model"]
            torch.cuda.reset_peak_memory_stats()
            evaluate(bb, tok, eos, job, m["config"], run)
            print(f"done {job['key']}", flush=True)
        except Exception:
            _json(errors / f"{job['key']}.json", {"key": job["key"], "traceback": traceback.format_exc()})
            raise


# --------------------------------------------------------------------------------------------- preflight
def preflight(out: Path, models, n_sources: int, horizons):
    import torch
    from kvp.loopzip_trajectory_data import candidate_items, fit_item
    frame = _hotpot_frame()
    toks = _tokenizers(MODELS_V6)
    item = None
    for it in candidate_items(frame, "W2", SEED + 1):  # preflight-only seed: not a frozen study prompt
        f = fit_item(it, toks, 128, 192)
        if f:
            item = {**it, **f}
            break
    report = {"protocol": PROTOCOL, "code_sha": git_sha(), "dirty": git_dirty(), "item_id": item["item_id"],
              "gpu": torch.cuda.get_device_name(0), "models": {}}
    for mk in models:
        tok, bb, eos = _load(mk)
        res = {}
        for hz in horizons:
            torch.cuda.reset_peak_memory_stats()
            o = run_reference(bb, tok, eos, item, f"v6-preflight/{item['item_id']}", hz, validate=True)
            r = o["runner"]
            el = o["eligible"]
            srcs = [el[int(k * (len(el) - 1) / max(1, n_sources - 1))] for k in range(n_sources)]
            ta, tc = [], []
            for s in srcs:
                torch.cuda.synchronize()
                t0 = time.time()
                acc = r.accumulated((s,))
                torch.cuda.synchronize()
                ta.append(time.time() - t0)
                t0 = time.time()
                r.clean_probes((s,), [t for t in CLEAN_TS if t <= o["ref"]["n"]])
                torch.cuda.synchronize()
                tc.append(time.time() - t0)
            t0 = time.time()
            r._restore()
            torch.cuda.synchronize()
            res[str(hz)] = {"prompt_len": len(o["prompt"].token_ids), "n_eligible": len(el),
                            "ref_tokens": o["ref"]["n"], "ref_stop": o["ref"]["stop"],
                            "prefill_s": o["t_prefill"], "reference_s": o["t_ref"],
                            "accumulated_s_per_source": ta, "clean_s_per_source": tc,
                            "restore_s": time.time() - t0, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                            "validation": o["validation"],
                            "example_late_kl": [x["kl"] for x in acc["token"][-3:]]}
            per = sum(ta) / len(ta) + sum(tc) / len(tc)
            res[str(hz)]["est_s_per_prompt_all_sources"] = per * len(el) + o["t_prefill"] + o["t_ref"]
            print(json.dumps({mk: {hz: {k: v for k, v in res[str(hz)].items() if k != "validation"}
                                   | {"validation_pass": o["validation"]["pass"]}}}), flush=True)
            del r, o
        report["models"][mk] = res
        del bb
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    _json(out, report)
    return report


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    for n in ("prepare", "execute", "analyze", "preflight"):
        g.add_argument("--" + n, action="store_true")
    g.add_argument("--worker", type=int)
    ap.add_argument("--run-dir")
    ap.add_argument("--out")
    ap.add_argument("--models", default=",".join(MODELS_V6))
    ap.add_argument("--model-filter")
    ap.add_argument("--n-sources", type=int, default=5)
    ap.add_argument("--horizons", default="16,128")
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--n-stage-a", type=int, default=40)
    ap.add_argument("--n-stage-b", type=int, default=4)
    ap.add_argument("--n-holdout", type=int, default=40)
    ap.add_argument("--n-reserve", type=int, default=20)
    ap.add_argument("--chunk-ouro", type=int, default=40)
    ap.add_argument("--chunk-huginn", type=int, default=12)
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--gpu-models", default="", help="optional per-GPU model filter, comma list aligned to --gpu-ids")
    a = ap.parse_args()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    if a.preflight:
        out = Path(a.out or "results/v6_preflight.json").resolve()
        preflight(out, sp(a.models), a.n_sources, [int(x) for x in sp(a.horizons)])
        return
    run = Path(a.run_dir).resolve()
    if a.prepare:
        cfg = {"models": sp(a.models), "horizon": a.horizon, "length_window": [128, 192],
               "n_stage_a": a.n_stage_a, "n_stage_b": a.n_stage_b, "n_holdout": a.n_holdout,
               "n_reserve": a.n_reserve, "chunk_size": {"ouro-2.6b": a.chunk_ouro, "huginn-0125": a.chunk_huginn}}
        print(json.dumps(prepare(run, cfg), indent=2))
    elif a.worker is not None:
        worker(run, a.worker, a.model_filter)
    elif a.analyze:
        from loopzip_trajectory_analysis import analyze
        print(json.dumps(analyze(run), indent=2))
    else:
        ids = sp(a.gpu_ids)
        filt = sp(a.gpu_models) if a.gpu_models else [""] * len(ids)
        (run / "logs").mkdir(exist_ok=True)
        ps = []
        for i, (gpu, mf) in enumerate(zip(ids, filt)):
            h = (run / "logs" / f"worker{i}.txt").open("a")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--run-dir", str(run)]
            if mf:
                cmd += ["--model-filter", mf]
            ps.append((subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                        stdout=h, stderr=subprocess.STDOUT), h))
        codes = [p.wait() for p, _ in ps]
        print(json.dumps({"worker_exit_codes": codes}))
        if any(codes):
            sys.exit(1)


if __name__ == "__main__":
    main()
