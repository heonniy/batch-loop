#!/usr/bin/env python3
"""v8: two-axis KV budgets for looped transformers — which Context TOKENS x which LOOP PLANES to keep.

  --prepare   freeze units (frozen v5 pilot20 prompts; dev = first 8 UIDs, holdout = next 12 per dataset/model) and jobs
  --execute   GPU workers (one model per worker, B=1)
  --worker    one worker
  --analyze   CPU tables + findings (plane_budget_analysis.py)

Job kinds (unit = model x dataset x UID):
  REF  untouched full-depth prefill + FULL fixed-depth gold continuation + v7 validation + plane-map validation;
       stores clean features (per-loop attention received, question-anchor and generated-query attention, settle
       depth of every prompt token's own trajectory).
  PLN  per-token loop-plane compression census: for each Context token alone, replace its R planes by
       last-only / first-only / even-half (dev) or last-only (holdout); gold ΔNLL and KL at fixed depth.
  TAX  equal-plane-budget two-axis policy matrix (budgets 10/30/50% of Context planes): token-only (score x full
       planes), plane-only (all tokens x plane policy), mixed (sqrt budgets); gold ΔNLL/KL + greedy exact match;
       map-semantics ablation (nearest vs latest kept plane).
Budget = planes over Context tokens only; Instruction/Question/format tokens always keep every plane.
Full-sized reference KV + read redirection; no HBM claim.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from query_conditioned_sweep import V5_RUN, UnitCache, _json, _load, _v5_job, git_dirty, git_sha  # noqa: E402

PROTOCOL = "plane-budget-v8-v1"
MODELS = ("ouro-2.6b", "huginn-0125")
DATASETS = ("controlled_retrieval", "hotpotqa_distractor")
SETTLE_TOL = 0.10


def prepare(run: Path, cfg: dict):
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing run directory")
    if git_dirty() or subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "kv_policy/scripts/plane_budget_sweep.py"],
                                     cwd=ROOT.parent, capture_output=True, text=True).stdout.strip():
        raise SystemExit("Commit kvp/ and the sweep scripts before freezing a run")
    run.mkdir(parents=True)
    v5 = json.loads((V5_RUN / "jobs.json").read_text())
    jobs, units = [], []
    for model in cfg["models"]:
        for ds in cfg["datasets"]:
            rows = sorted((j for j in v5 if j["model"] == model and j["dataset"] == ds), key=lambda j: j["uid"])
            for split, (lo, hi) in cfg["splits"].items():
                for j in rows[lo:hi]:
                    unit = f"{model}__{ds}__{j['uid']}"
                    content = j["content_positions"]
                    base = {"model": model, "dataset": ds, "uid": j["uid"], "unit": unit, "v5_key": j["key"], "split": split,
                            "prompt_len": len(j["prompt_ids"]), "n_targets": len(j["target_ids"])}
                    units.append(base | {"n_content": len(content)})
                    refk = f"REF__{unit}"
                    jobs.append(base | {"kind": "REF", "key": refk})
                    jobs.append(base | {"kind": "TAX", "key": f"TAX__{unit}", "depends": refk})
                    variants = cfg["pln_variants"][split]
                    if variants:
                        cs = cfg["chunk_size"][model]
                        for k in range(0, len(content), cs):
                            jobs.append(base | {"kind": "PLN", "key": f"PLN__{unit}__{k // cs:03d}", "depends": refk,
                                                "sources": content[k:k + cs], "variants": variants})
    if len({j["key"] for j in jobs}) != len(jobs):
        raise RuntimeError("Duplicate job keys")
    _json(run / "jobs.json", jobs)
    _json(run / "units.json", units)
    _json(run / "run_manifest.json", {"protocol": PROTOCOL, "code_sha": git_sha(), "config": cfg, "v5_source_run": str(V5_RUN),
                                      "notes": ["Budget counted in loop planes over Context tokens only",
                                                "Token scores are label-free; settle depth and received attention come from the clean prefill",
                                                "Future-query scores (def4/fut_oracle) are deferred/oracle references, not prompt-time policies",
                                                "Full-sized reference KV; reads redirected to kept planes; no HBM claim"]})
    from collections import Counter
    return {"jobs": len(jobs), "units": len(units), "kinds": dict(Counter(j["kind"] for j in jobs))}


def _settle(F, R: int) -> int:
    """First plane index whose state is within SETTLE_TOL of the terminal state (F indexed 0..R; plane r-1)."""
    import numpy as np
    idx = np.where(F[1:] < SETTLE_TOL)[0]
    return int(idx[0]) if len(idx) else R - 1


def unit_features(r, P: int, R: int, n: int):
    import numpy as np
    qa = r.census(r.anchors["question_last"])["attn"].numpy()                              # (R, P)
    recv = r.source_features()["recv_mean"].numpy()                                        # (R, P)
    ga = np.stack([r.local[P + t]["attn"][:, :P].cpu().numpy() for t in range(n)])          # (n, R, P)
    settle = np.array([_settle(m["F"].numpy(), R) for m in r.prompt_metrics], dtype=np.int64)
    U = np.stack([m["U"].numpy() for m in r.prompt_metrics])                                # (P, R+1)
    return qa, recv, ga, settle, U


def token_scores(unit: str, content, P: int, qa, recv, ga):
    import hashlib
    import numpy as np
    sc = {"qa_rR": qa[-1], "recv_rR": recv[-1], "la2_rR": ga[:2, -1].max(0), "def4_rR": ga[:4, -1].max(0),
          "fut_oracle_rR": ga[:, -1].max(0)}
    for sd in (11, 29):
        rng = np.random.default_rng(int(hashlib.sha256(f"{unit}/{sd}".encode()).hexdigest()[:8], 16))
        sc[f"random_{sd}"] = rng.random(P)
    return sc


def evaluate(bb, tok, eos, job, cfg, run: Path, cache: UnitCache):
    import numpy as np
    import torch
    from kvp.plane_policies import plane_map_from_sets, two_axis_arms
    from kvp.role_generation import score_free_answer
    raw = run / "raw"
    meta = {k: job[k] for k in ("key", "kind", "model", "dataset", "uid", "unit", "split")} | {"sha": git_sha()}
    r, p, targets = cache.get(bb, tok, eos, job, cfg, cfg["horizon"])
    R, P, n = bb.R, r.P, len(r.ref_ids)
    dig = float(r.ref_H.norm(dim=-1).sum())
    content = [int(c) for c in p.content_positions]
    qa, recv, ga, settle, U = unit_features(r, P, R, n)
    if job["kind"] == "REF":
        t0 = time.time()
        v7 = r.validate_v7(content[len(content) // 3], content[(2 * len(content)) // 3])
        vp = r.validate_planes()
        if not v7["pass"] or not vp["pass"]:
            raise RuntimeError(f"validation failed: {v7['failures']} {vp['failures']}")
        np.savez_compressed(raw / f"{job['key']}.npz", qa=qa.astype("float32"), recv=recv.astype("float32"),
                            gen_attn=ga.astype("float32"), settle=settle, prompt_U=U.astype("float32"),
                            prompt_F=np.stack([m["F"].numpy() for m in r.prompt_metrics]).astype("float32"))
        _json(raw / f"{job['key']}.json", meta | {"prompt_ids": list(p.token_ids), "roles": list(p.roles),
                                                  "content_positions": content, "anchors": r.anchors, "ref": r.ref,
                                                  "target_ids": targets, "validation_v7": v7, "validation_planes": vp,
                                                  "ref_state_digest": dig, "settle_hist": np.bincount(settle[content], minlength=R).tolist(),
                                                  "timing": {"prefill_s": r.t_prefill, "reference_s": r.t_reference,
                                                             "validation_s": time.time() - t0},
                                                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        return
    ref = json.loads((raw / f"{job['depends']}.json").read_text())
    if abs(dig - ref["ref_state_digest"]) > 1e-3 * max(1.0, dig):
        raise RuntimeError(f"Reference digest drift {dig} vs {ref['ref_state_digest']}")
    full = r.decode((), r.spec)
    if job["kind"] == "PLN":
        variants = {"last": [R - 1], "first": [0], "half": list(np.unique(np.round(np.linspace(0, R - 1, max(1, R // 2))).astype(int)))}
        out = {f"{v}_{k}": np.full(len(job["sources"]), np.nan, dtype="float32")
               for v in job["variants"] for k in ("dnll", "kl_mean", "kl_max", "first_dnll")}
        times = []
        for si, s in enumerate(job["sources"]):
            t0 = time.time()
            for v in job["variants"]:
                m = plane_map_from_sets({int(s): np.array(variants[v])}, R, P, cfg["semantics"])
                d = r.decode((), r.spec, plane_map=m)
                out[f"{v}_dnll"][si] = d["nll_future"] - full["nll_future"]
                out[f"{v}_kl_mean"][si] = float(np.mean(d["kl_by_t"]))
                out[f"{v}_kl_max"][si] = float(np.max(d["kl_by_t"]))
                out[f"{v}_first_dnll"][si] = d["token_losses"][1] - full["token_losses"][1]
            times.append(time.time() - t0)
        np.savez_compressed(raw / f"{job['key']}.npz", sources=np.array(job["sources"]), **out)
        _json(raw / f"{job['key']}.json", meta | {"sources": job["sources"], "variants": job["variants"], "per_source_s": times,
                                                  "reference_digest": dig, "full_nll_future": full["nll_future"]})
        return
    if job["kind"] == "TAX":
        gold = _v5_job(job)["gold_answer"]
        sc = token_scores(job["unit"], content, P, qa, recv, ga)
        g_full = r.greedy((), cfg["greedy_tokens"], eos)
        rows = []
        t0 = time.time()
        for b in cfg["budgets"]:
            arms = two_axis_arms(content, R, b, sc, settle, recv)
            extra = []
            for a in arms:
                if a["axis"] == "plane" and a["plane_policy"] in ("last", "even"):
                    extra.append(a | {"semantics": "latest"})
            for a in arms + extra:
                sem = a.get("semantics", cfg["semantics"])
                m = plane_map_from_sets(a["sets"], R, P, sem)
                d = r.decode(a["evicted"], r.spec, plane_map=m)
                g = r.greedy(a["evicted"], cfg["greedy_tokens"], eos, plane_map=m)
                em = score_free_answer(tok, gold, {"token_ids": g, "decoded_tokens": len(g)})
                rows.append({"budget": b, "axis": a["axis"], "token_score": a["token_score"], "plane_policy": a["plane_policy"],
                             "semantics": sem, "token_frac": a["token_frac"], "planes_per_token": a["planes_per_token"],
                             "n_evicted": len(a["evicted"]), "n_kept": len(content) - len(a["evicted"]),
                             "planes_kept": a["planes"], "planes_target": a["target"], "planes_total": len(content) * R,
                             "nll_future": d["nll_future"], "delta_nll_future": d["nll_future"] - full["nll_future"],
                             "kl_mean": float(np.mean(d["kl_by_t"])), "kl_max": float(np.max(d["kl_by_t"])),
                             "exact_match": em["exact_match"], "extracted": em["extracted_answer"][:40]})
        _json(raw / f"{job['key']}.json", meta | {
            "full": {"nll_future": full["nll_future"], "first_target_nll": full["first_target_nll"],
                     "exact_match": score_free_answer(tok, gold, {"token_ids": g_full, "decoded_tokens": len(g_full)})["exact_match"]},
            "n_content": len(content), "R": R, "rows": rows, "gold_answer": gold, "elapsed_s": time.time() - t0,
            "reference_digest": dig, "settle_hist": np.bincount(settle[content], minlength=R).tolist()})
        return
    raise RuntimeError(job["kind"])


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
    cache = UnitCache()
    order = {"REF": 0, "TAX": 1, "PLN": 2}
    while True:
        pend = [j for j in jobs if not (raw / f"{j['key']}.json").exists() and not (claims / f"{j['key']}.lock").exists()]
        if not pend:
            return
        ready = [j for j in pend if not j.get("depends") or (raw / f"{j['depends']}.json").exists()]
        if not ready:
            if any((errors / f"{j['depends']}.json").exists() for j in pend if j.get("depends")):
                raise RuntimeError("Dependency failed")
            time.sleep(15)
            continue
        ready.sort(key=lambda j: (j["model"] != cur, j["unit"] != cache.unit, order[j["kind"]], j["unit"], j["key"]))
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
                cache = UnitCache()
                torch.cuda.empty_cache()
                tok, bb, eos = _load(job["model"])
                cur = job["model"]
            torch.cuda.reset_peak_memory_stats()
            evaluate(bb, tok, eos, job, m["config"], run, cache)
            print(f"done {job['key']}", flush=True)
        except Exception:
            _json(errors / f"{job['key']}.json", {"key": job["key"], "traceback": traceback.format_exc()})
            cache = UnitCache()
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    for nm in ("prepare", "execute", "analyze"):
        g.add_argument("--" + nm, action="store_true")
    g.add_argument("--worker", type=int)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--model-filter")
    ap.add_argument("--dev", default="0:8")
    ap.add_argument("--holdout", default="8:20")
    ap.add_argument("--budgets", default="0.1,0.3,0.5")
    ap.add_argument("--semantics", default="nearest", choices=("nearest", "latest"))
    ap.add_argument("--pln-dev", default="last,first,half")
    ap.add_argument("--pln-holdout", default="last")
    ap.add_argument("--chunk-ouro", type=int, default=96)
    ap.add_argument("--chunk-huginn", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--greedy-tokens", type=int, default=24)
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--gpu-models", default="")
    a = ap.parse_args()
    sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    run = Path(a.run_dir).resolve()
    if a.prepare:
        rng = lambda s: tuple(int(x) for x in s.split(":"))  # noqa: E731
        cfg = {"models": sp(a.models), "datasets": sp(a.datasets), "splits": {"dev": rng(a.dev), "holdout": rng(a.holdout)},
               "budgets": [float(x) for x in sp(a.budgets)], "semantics": a.semantics,
               "pln_variants": {"dev": sp(a.pln_dev), "holdout": sp(a.pln_holdout)},
               "chunk_size": {"ouro-2.6b": a.chunk_ouro, "huginn-0125": a.chunk_huginn}, "horizon": a.horizon,
               "greedy_tokens": a.greedy_tokens}
        print(json.dumps(prepare(run, cfg), indent=2))
    elif a.worker is not None:
        worker(run, a.worker, a.model_filter)
    elif a.analyze:
        from plane_budget_analysis import analyze
        print(json.dumps(analyze(run), indent=2, default=str))
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
