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

import numpy as np
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
    if cfg.get("v6_pool"):
        from query_conditioned_sweep import V6_RUN
        v6 = json.loads((V6_RUN / "jobs.json").read_text())
        man = {x["item_id"]: x for x in json.loads((V6_RUN / "manifest.json").read_text())}
        limit = cfg.get("v6_per_pool_limit", 0)
        seen = {}
        for ref in (j for j in v6 if j["kind"] in ("Bref", "A") and j["model"] in cfg["models"]
                    and man[j["item_id"]]["pool"] in cfg["v6_pool"]):
            key = (ref["model"], ref["workload"], man[ref["item_id"]]["pool"])
            seen[key] = seen.get(key, 0) + 1
            if limit and seen[key] > limit:
                continue
            unit = f"{ref['model']}__v6{ref['workload']}__{ref['item_id']}"
            base = {"model": ref["model"], "dataset": f"v6_{ref['workload']}", "uid": ref["uid"], "unit": unit, "v6_key": ref["key"],
                    "item_id": ref["item_id"], "split": man[ref["item_id"]]["pool"], "prompt_len": ref["prompt_len"]}
            units.append(base)
            jobs.append(base | {"kind": "HOR", "key": f"HOR__{unit}"})
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


EXIT_THRESH = {"ouro": 0.7, "huginn": 1e-3}   # frozen clean-only thresholds (v7 E2); min depth 2


def clean_exit_depths(r, R: int) -> "list[int]":
    """Native exit depth each PROMPT token would have taken (Ouro Q-exit q=0.7; Huginn fp32 cosine 1e-3)."""
    fam = r.bb.family
    out = []
    for ex in r.prompt_exit:
        if fam == "ouro":
            cum = ex["cumulative"]
            d = next((k for k in range(2, R + 1) if k == R or cum[k - 1] >= EXIT_THRESH["ouro"]), R)
        else:
            dist = ex["distance_fp32"]
            d = next((k for k in range(2, R + 1) if dist[k - 1] < EXIT_THRESH["huginn"] or k == R), R)
        out.append(int(d))
    return out


def generated_exit_depths(r, R: int) -> "list[int]":
    fam = r.bb.family
    out = []
    for ex in r.ref_exit:
        if fam == "ouro":
            cum = ex["cumulative"]
            out.append(next((k for k in range(2, R + 1) if k == R or cum[k - 1] >= EXIT_THRESH["ouro"]), R))
        else:
            dist = ex["distance_fp32"]
            out.append(next((k for k in range(2, R + 1) if dist[k - 1] < EXIT_THRESH["huginn"] or k == R), R))
    return out


def kv_plane_similarity(r, P: int, R: int):
    """Cosine similarity of each prompt token's K and V at plane r to its terminal plane and to the previous plane,
    per recurrent layer (head-averaged): (n_rec, R, P) arrays. Measures how redundant the loop planes are."""
    import torch
    outs = {}
    for name, bufs in (("k", r.kv.rk), ("v", r.kv.rv)):
        to_last, to_prev = [], []
        for li in range(r.n_rec):
            b = bufs[li][:, 0, :, :P, :].float()                     # (R, H, P, D)
            last = b[R - 1:R]
            cl = torch.nn.functional.cosine_similarity(b, last.expand_as(b), dim=-1).mean(1)        # (R, P)
            prev = torch.cat([b[:1], b[:-1]], 0)
            cp = torch.nn.functional.cosine_similarity(b, prev, dim=-1).mean(1)
            to_last.append(cl.cpu().numpy())
            to_prev.append(cp.cpu().numpy())
        outs[f"{name}_cos_to_last"] = np.stack(to_last).astype("float32")
        outs[f"{name}_cos_to_prev"] = np.stack(to_prev).astype("float32")
    return outs


def unit_features(r, P: int, R: int, n: int):
    import numpy as np
    qa_raw = r.census(r.anchors["question_last"])["attn"].numpy()                          # (R, anchor col)
    qa = np.zeros((R, P), dtype=qa_raw.dtype)
    qa[:, :qa_raw.shape[1]] = qa_raw                                                        # (R, P)
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
    from kvp.plane_policies import plane_map_from_sets, plane_sets, two_axis_arms
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
        exit_d = clean_exit_depths(r, R)
        np.savez_compressed(raw / f"{job['key']}.npz", qa=qa.astype("float32"), recv=recv.astype("float32"),
                            gen_attn=ga.astype("float32"), settle=settle, prompt_U=U.astype("float32"),
                            prompt_F=np.stack([m["F"].numpy() for m in r.prompt_metrics]).astype("float32"),
                            exit_depth=np.array(exit_d, dtype=np.int64), **kv_plane_similarity(r, P, R))
        _json(raw / f"{job['key']}.json", meta | {"prompt_ids": list(p.token_ids), "roles": list(p.roles),
                                                  "content_positions": content, "anchors": r.anchors, "ref": r.ref,
                                                  "target_ids": targets, "validation_v7": v7, "validation_planes": vp,
                                                  "ref_state_digest": dig, "settle_hist": np.bincount(settle[content], minlength=R).tolist(),
                                                  "exit_depth_hist": np.bincount(np.array(exit_d)[content], minlength=R + 1).tolist(),
                                                  "timing": {"prefill_s": r.t_prefill, "reference_s": r.t_reference,
                                                             "validation_s": time.time() - t0},
                                                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        return
    if job["kind"] == "HOR":
        return _evaluate_horizon(bb, job, cfg, raw, meta, r, p, content, qa, recv, ga, settle, dig)
    ref = json.loads((raw / f"{job['depends']}.json").read_text())
    if abs(dig - ref["ref_state_digest"]) > 1e-3 * max(1.0, dig):
        raise RuntimeError(f"Reference digest drift {dig} vs {ref['ref_state_digest']}")
    full = r.decode((), r.spec)
    if job["kind"] == "PLN":
        exit_d = clean_exit_depths(r, R)
        variants = {"last": [R - 1], "first": [0], "half": list(np.unique(np.round(np.linspace(0, R - 1, max(1, R // 2))).astype(int)))}
        out = {f"{v}_{k}": np.full(len(job["sources"]), np.nan, dtype="float32")
               for v in job["variants"] for k in ("dnll", "kl_mean", "kl_max", "first_dnll")}
        times = []
        for si, s in enumerate(job["sources"]):
            t0 = time.time()
            for v in job["variants"]:
                if v == "exit":   # the token's own native early exit: planes >= d_i were never computed (no-refill)
                    m = plane_map_from_sets({int(s): np.arange(exit_d[int(s)])}, R, P, "latest")
                else:
                    m = plane_map_from_sets({int(s): np.array(variants[v])}, R, P, cfg["semantics"])
                d = r.decode((), r.spec, plane_map=m)
                out[f"{v}_dnll"][si] = d["nll_future"] - full["nll_future"]
                out[f"{v}_kl_mean"][si] = float(np.mean(d["kl_by_t"]))
                out[f"{v}_kl_max"][si] = float(np.max(d["kl_by_t"]))
                out[f"{v}_first_dnll"][si] = d["token_losses"][1] - full["token_losses"][1]
            times.append(time.time() - t0)
        np.savez_compressed(raw / f"{job['key']}.npz", sources=np.array(job["sources"]), **out)
        _json(raw / f"{job['key']}.json", meta | {"sources": job["sources"], "variants": job["variants"], "per_source_s": times,
                                                  "exit_depth": [exit_d[int(s)] for s in job["sources"]],
                                                  "reference_digest": dig, "full_nll_future": full["nll_future"]})
        return
    if job["kind"] == "TAX":
        gold = _v5_job(job)["gold_answer"]
        sc = token_scores(job["unit"], content, P, qa, recv, ga)
        g_full = r.greedy((), cfg["greedy_tokens"], eos)
        rows = []
        t0 = time.time()
        # "natural" plane budget of a looped model with native early exit: every Context token keeps only the planes it
        # actually computed (0..d_i-1, read with 'latest' semantics), vs last-k / even-k at the SAME total plane count.
        exit_d = np.array(clean_exit_depths(r, R))
        nat_sets = {int(c): np.arange(int(exit_d[c])) for c in content}
        nat_planes = int(sum(len(v) for v in nat_sets.values()))
        k_nat = max(1, int(round(nat_planes / len(content))))
        natural = [("exit_truncated", nat_sets, "latest"),
                   ("last", plane_sets("last", content, R, k_nat), cfg["semantics"]),
                   ("even", plane_sets("even", content, R, k_nat), cfg["semantics"]),
                   ("settle", plane_sets("settle", content, R, k_nat, settle=settle, plane_budget=nat_planes), cfg["semantics"])]
        for pol, sets, sem in natural:
            m = plane_map_from_sets(sets, R, P, sem)
            d = r.decode((), r.spec, plane_map=m)
            g = r.greedy((), cfg["greedy_tokens"], eos, plane_map=m)
            em = score_free_answer(tok, gold, {"token_ids": g, "decoded_tokens": len(g)})
            rows.append({"budget": round(nat_planes / (len(content) * R), 4), "axis": "natural", "token_score": "none",
                         "plane_policy": pol, "semantics": sem, "token_frac": 1.0, "planes_per_token": k_nat, "n_evicted": 0,
                         "n_kept": len(content), "planes_kept": int(sum(len(np.unique(v)) for v in sets.values())),
                         "planes_target": nat_planes, "planes_total": len(content) * R,
                         "nll_future": d["nll_future"], "delta_nll_future": d["nll_future"] - full["nll_future"],
                         "kl_mean": float(np.mean(d["kl_by_t"])), "kl_max": float(np.max(d["kl_by_t"])),
                         "exact_match": em["exact_match"], "extracted": em["extracted_answer"][:40]})
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
            "reference_digest": dig, "settle_hist": np.bincount(settle[content], minlength=R).tolist(),
            "exit_depth_hist": np.bincount(exit_d[content], minlength=R + 1).tolist(), "natural_planes": nat_planes})
        return
    raise RuntimeError(job["kind"])


def _evaluate_horizon(bb, job, cfg, raw, meta, r, p, content, qa, recv, ga, settle, dig):
    """Decode-horizon growth of damage over the frozen v6 128-token greedy continuation: plane compression of prompt
    tokens only vs prompt + generated tokens (converted after each token finishes), token eviction, and mixed arms."""
    import torch
    from kvp.plane_policies import plane_map_from_sets, plane_sets
    from query_conditioned_sweep import V6_RUN
    v6meta = json.loads((V6_RUN / "raw" / f"{job['v6_key']}.json").read_text())
    if r.ref["frozen_argmax_agreement"] < 1.0 or abs(dig - v6meta["ref_state_digest"]) > 1e-3 * max(1.0, dig):
        raise RuntimeError("Frozen v6 reference not reproduced")
    vp = r.validate_planes()
    if not vp["pass"]:
        raise RuntimeError(f"plane validation failed: {vp}")
    R, P, n, T = bb.R, r.P, len(r.ref_ids), r.T
    gen_cols = list(range(P, P + n))
    exit_p = np.array(clean_exit_depths(r, R))
    exit_g = np.array(generated_exit_depths(r, R))
    half = max(1, R // 2)
    sc = token_scores(job["unit"], content, P, qa, recv, ga)
    ident_T = np.tile(np.arange(R)[:, None], (1, T))

    def pmap(sets_prompt, sets_gen=None, sem="nearest"):
        m = plane_map_from_sets(sets_prompt, R, P, sem)
        full = ident_T.copy()
        full[:, :P] = m
        if sets_gen:
            mg = plane_map_from_sets({c - P: v for c, v in sets_gen.items()}, R, n, sem)
            full[:, P:P + n] = mg
        return full

    arms = []
    for scope in ("prompt", "prompt+generated"):
        gen = scope == "prompt+generated"
        for pol, k in (("last", 1), ("even", half), ("first", None), ("exit", None)):
            if pol == "first":
                sp_ = {c: np.array([0]) for c in content}
                sg = {c: np.array([0]) for c in gen_cols} if gen else None
                sem = "nearest"
            elif pol == "exit":
                sp_ = {c: np.arange(int(exit_p[c])) for c in content}
                sg = {c: np.arange(int(exit_g[c - P])) for c in gen_cols} if gen else None
                sem = "latest"
            else:
                sp_ = plane_sets(pol, content, R, k)
                sg = plane_sets(pol, gen_cols, R, k) if gen else None
                sem = "nearest"
            arms.append({"axis": "plane", "scope": scope, "plane_policy": pol, "token_score": "none", "evicted": [],
                         "map": pmap(sp_, sg, sem), "planes_prompt": int(sum(len(np.unique(v)) for v in sp_.values()))})
    n_keep = max(1, int(round(0.3 * len(content))))
    for name in ("random_11", "la2_rR", "def4_rR", "fut_oracle_rR", "qa_rR"):
        order = sorted(range(len(content)), key=lambda j: (-float(sc[name][content[j]]), int(content[j])))
        ev = sorted(int(content[j]) for j in order[n_keep:])
        arms.append({"axis": "token", "scope": "prompt", "plane_policy": "full", "token_score": name, "evicted": ev,
                     "map": None, "planes_prompt": n_keep * R})
    for name in ("la2_rR", "fut_oracle_rR"):
        nk = max(1, int(round(np.sqrt(0.3) * len(content))))
        order = sorted(range(len(content)), key=lambda j: (-float(sc[name][content[j]]), int(content[j])))
        kept = sorted(int(content[j]) for j in order[:nk])
        ev = sorted(set(content) - set(kept))
        sp_ = plane_sets("last", kept, R, max(1, int(round(np.sqrt(0.3) * R))))
        sg = plane_sets("last", gen_cols, R, max(1, int(round(np.sqrt(0.3) * R))))
        arms.append({"axis": "mixed", "scope": "prompt+generated", "plane_policy": "last", "token_score": name, "evicted": ev,
                     "map": pmap(sp_, sg), "planes_prompt": int(sum(len(np.unique(v)) for v in sp_.values()))})
    ref_nll = [-float(r.last_prompt_lp[r.ref_ids[0]])] + [-float(r.ref_lp[i][r.ref_ids[i + 1]]) for i in range(n - 1)]
    rows, kl, nll = [], [], []
    t0 = time.time()
    for a in arms:
        d = r.decode(a["evicted"], r.spec, plane_map=a["map"])
        kl.append(np.array(d["kl_by_t"], dtype="float32"))
        nll.append(np.array(d["token_losses"], dtype="float32") - np.array(ref_nll, dtype="float32"))
        rows.append({k: v for k, v in a.items() if k not in ("map", "evicted")} | {
            "n_evicted": len(a["evicted"]), "nll_future_shift": d["nll_future"] - sum(ref_nll[1:]) / max(1, n - 1),
            "kl_mean": float(np.mean(d["kl_by_t"])), "kl_early8": float(np.mean(d["kl_by_t"][:8])),
            "kl_mid": float(np.mean(d["kl_by_t"][8:32])), "kl_late": float(np.mean(d["kl_by_t"][32:])),
            "planes_total_prompt": len(content) * R})
    np.savez_compressed(raw / f"{job['key']}.npz", kl=np.stack(kl), nll_shift=np.stack(nll), exit_prompt=exit_p, exit_gen=exit_g)
    _json(raw / f"{job['key']}.json", meta | {"rows": rows, "n_tokens": n, "R": R, "n_content": len(content), "validation_planes": vp,
                                              "elapsed_s": time.time() - t0, "reference_digest": dig,
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
    cache = UnitCache()
    order = {"REF": 0, "TAX": 1, "PLN": 2, "HOR": 0}
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
    ap.add_argument("--pln-dev", default="last,first,half,exit")
    ap.add_argument("--pln-holdout", default="last,exit")
    ap.add_argument("--chunk-ouro", type=int, default=96)
    ap.add_argument("--chunk-huginn", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--greedy-tokens", type=int, default=24)
    ap.add_argument("--v6-pool", default="", help="v6 manifest pools for the HOR (decode-horizon) units, e.g. B or A,B")
    ap.add_argument("--v6-per-pool-limit", type=int, default=0, help="max HOR units per (model, workload, pool); 0 = all")
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
               "greedy_tokens": a.greedy_tokens, "v6_pool": sp(a.v6_pool), "v6_per_pool_limit": a.v6_per_pool_limit}
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
