#!/usr/bin/env python3
"""LoopZip v6 Stage C mechanism controls + auxiliary runs, derived from a COMPLETE v6 pilot run.

Job kinds (all small, explicitly labeled diagnostics; none replaces the primary persistent census):
  CTL  per Stage-B prompt: for frozen representative cases (largest / median / smallest census mean-KL
       source x t in {8, 32}), with PRISTINE FULL generated history at query token t:
         all-loop+plain mask (= clean probe), recurrent-only mask (plain reads kept),
         one-loop mask at loop r (recurrent layers of loop r only),
         injection of the one-loop mask's own delta_r at loop r (must reproduce the one-loop mask),
         random-direction injection with ||delta_r|| at loop r (2 seeds): matched-amplitude control.
  RO   Ouro-native intermediate readout (lm_head(h_r)) on every Stage-A Ouro unit's frozen continuation.
  CRref/CR  auxiliary evidence-loss control: exhaustive persistent census on v5 Controlled Retrieval
       prompts (2 per model, frozen v5 UIDs); evidence labels used ONLY for post-hoc stratification.
--prepare / --execute / --worker  (analysis lives in loopzip_trajectory_analysis.analyze_controls)
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

from loopzip_trajectory_sweep import CLEAN_TS, _json, _load, _prompt, git_dirty, git_sha  # noqa: E402

PROTOCOL = "loopzip-v6-stagec-controls-v1"
PROBE_LOOPS = {"ouro-2.6b": (1, 2, 3, 4), "huginn-0125": (1, 2, 4, 8, 16, 24, 32)}
CASE_TS = (8, 32)
SEEDS = (101, 202)
CR_ROLES = ("relevant_content", "irrelevant_content", "instruction", "question")


def prepare(run: Path, cfg: dict):
    import numpy as np
    src = Path(cfg["source_run"])
    if not json.loads((src / "completeness.json").read_text())["complete"]:
        raise SystemExit("Source v6 pilot must be complete")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to overwrite")
    if git_dirty():
        raise SystemExit("Commit code before freezing")
    run.mkdir(parents=True)
    sjobs = json.loads((src / "jobs.json").read_text())
    jobs = []
    for ref in (j for j in sjobs if j["kind"] == "Bref"):
        meta = json.loads((src / "raw" / f"{ref['key']}.json").read_text())
        kl = {}
        for c in (j for j in sjobs if j.get("depends") == ref["key"]):
            z = np.load(src / "raw" / f"{c['key']}.npz")
            for si, s in enumerate(z["sources"]):
                kl[int(s)] = float(np.nanmean(z["acc_kl"][si]))
        order = sorted(kl, key=lambda s: (-kl[s], s))
        picks = {"largest": order[0], "median": order[len(order) // 2], "smallest": order[-1]}
        n = meta["ref"]["n"]
        ts = sorted({t if t <= n else max(x for x in CLEAN_TS if x <= n) for t in CASE_TS})
        jobs.append({"kind": "CTL", "key": f"CTL__{ref['model']}__{ref['item_id']}", "model": ref["model"],
                     "workload": ref["workload"], "item_id": ref["item_id"], "uid": ref["uid"],
                     "ref_key": ref["key"], "cases": [{"label": k, "source": s, "t": t, "census_kl_mean": kl[s]}
                                                      for k, s in picks.items() for t in ts]})
    for a in (j for j in sjobs if j["kind"] == "A" and j["model"].startswith("ouro")):
        jobs.append({"kind": "RO", "key": f"RO__{a['model']}__{a['item_id']}", "model": a["model"],
                     "workload": a["workload"], "item_id": a["item_id"], "uid": a["uid"], "ref_key": a["key"]})
    v5 = json.loads((Path(cfg["v5_run"]) / "jobs.json").read_text())
    for m in cfg["models"]:
        cr = sorted((j for j in v5 if j["model"] == m and j["dataset"] == "controlled_retrieval"),
                    key=lambda j: j["key"])[:cfg["n_cr"]]
        for j in cr:
            from loopzip_failure_screen import make_prompt
            from kvp.loopzip_persistent_eviction import eligible_sources
            elig = eligible_sources(make_prompt(j), CR_ROLES)
            refk = f"CRref__{m}__{j['uid']}"
            base = {"model": m, "workload": "CR", "item_id": j["uid"], "uid": j["uid"], "v5_key": j["key"],
                    "evidence_positions": j["evidence_positions"]}
            jobs.append({**base, "kind": "CRref", "key": refk})
            cs = cfg["chunk"]
            for k in range(0, len(elig), cs):
                jobs.append({**base, "kind": "CR", "key": f"CR__{m}__{j['uid']}__{k // cs:03d}", "depends": refk,
                             "sources": elig[k:k + cs]})
    (run / "jobs.json").write_text(json.dumps(jobs, indent=1))
    _json(run / "run_manifest.json", {"protocol": PROTOCOL, "code_sha": git_sha(), "config": cfg,
                                      "probe_loops": PROBE_LOOPS, "case_ts": CASE_TS, "seeds": SEEDS,
                                      "notes": ["Case selection uses the completed persistent census (labeled)",
                                                "All controls use clean (pristine FULL) generated history",
                                                "Huginn intermediate readout is not model-native: not run",
                                                "CR evidence labels used only for post-hoc stratification"]})
    from collections import Counter
    return {"jobs": len(jobs), "kinds": Counter(j["kind"] for j in jobs)}


def _v5_prompt(cfg, job):
    from loopzip_failure_screen import make_prompt
    v5 = {j["key"]: j for j in json.loads((Path(cfg["v5_run"]) / "jobs.json").read_text())}
    return make_prompt(v5[job["v5_key"]])


def evaluate(bb, tok, eos, job, cfg, run: Path):
    import numpy as np
    import torch
    from kvp.loopzip_persistent_eviction import PersistentEvictionRunner, eligible_sources
    src = Path(cfg["source_run"])
    raw = run / "raw"
    meta = {k: job[k] for k in ("key", "kind", "model", "workload", "item_id", "uid")} | {"sha": git_sha()}
    if job["kind"] in ("CTL", "RO"):
        item = next(x for x in json.loads((src / "manifest.json").read_text()) if x["item_id"] == job["item_id"])
        ref = json.loads((src / "raw" / f"{job['ref_key']}.json").read_text())
        p = _prompt(tok, bb.family, item)
        r = PersistentEvictionRunner(bb, p, job["uid"], horizon=cfg["horizon"], record_prefill_attention=False)
        r.readout = job["kind"] == "RO"
        rr = r.reference(eos, frozen_ids=ref["ref"]["ids"])
        dig = float(r.ref_H.norm(dim=-1).sum())
        if rr["frozen_argmax_agreement"] < 1.0 or abs(dig - ref["ref_state_digest"]) > 1e-3 * max(1.0, abs(dig)):
            raise RuntimeError("Frozen pilot reference not reproduced")
        if job["kind"] == "RO":
            np.savez_compressed(raw / f"{job['key']}.npz",
                                kl_to_final=np.stack([x["kl_to_final"].numpy() for x in r.ref_readout]),
                                top1_agree=np.stack([x["top1_agree"].numpy() for x in r.ref_readout]))
            _json(raw / f"{job['key']}.json", meta | {"n_tokens": rr["n"], "reference_digest": dig})
            return
        rows = []
        R = bb.R
        for c in job["cases"]:
            s, t = c["source"], c["t"]

            def rec(arm, loop=None, seed=None, o=None):
                rows.append({**c, "arm": arm, "loop": loop, "seed": seed, "E": o["E"].tolist(),
                             "dircos": o["dircos"].tolist(), **{k: o["token"][k] for k in
                                                                ("kl", "max_logprob_diff", "top1_changed",
                                                                 "entropy_shift", "nll_shift")}})
            allm = r.control_probe(t, (s,))
            rec("all_loops_plain", o=allm)
            rec("rec_only_all_loops", o=r.control_probe(t, (s,), plain=False))
            for L in PROBE_LOOPS[job["model"]]:
                one = r.control_probe(t, (s,), loops=(L,), plain=False)
                rec("one_loop_mask", L, o=one)
                d = one["delta"][L].clone()
                same = r.control_probe(t, inject=(L, d))
                rec("inject_own_delta", L, o=same)
                # equality is up to BF16 rounding of h_L + delta_L; the gap is reported, not hidden
                amp = float(d.norm())
                for sd in SEEDS:
                    g = torch.Generator(device="cpu").manual_seed(sd * 1000 + L)
                    v = torch.randn(d.numel(), generator=g).to(d.device)
                    v = v / v.norm() * amp
                    rec("inject_random_matched", L, sd, r.control_probe(t, inject=(L, v)))
                for x in rows[-4:]:
                    x["delta_norm"] = amp
        _json(raw / f"{job['key']}.json", meta | {"R": R, "rows": rows, "reference_digest": dig})
        return
    # CR aux census (same code path as Stage B, v5 prompts)
    p = _v5_prompt(cfg, job)
    if job["kind"] == "CRref":
        r = PersistentEvictionRunner(bb, p, job["uid"], horizon=cfg["horizon"])
        ref = r.reference(eos)
        el = eligible_sources(p, CR_ROLES)
        val = r.validate(el[len(el) // 3], el[(2 * len(el)) // 3], t_probe=min(8, ref["n"]))
        if not val["pass"]:
            raise RuntimeError(f"G0 validation failed: {val}")
        sf = r.source_features()
        np.savez_compressed(raw / f"{job['key']}.npz", **{f"src_{k}": v.numpy() for k, v in sf.items()})
        _json(raw / f"{job['key']}.json", meta | {"prompt_ids": list(p.token_ids), "roles": list(p.roles),
                                                  "eligible": el, "ref": ref, "validation": val,
                                                  "evidence_positions": job["evidence_positions"],
                                                  "ref_state_digest": float(r.ref_H.norm(dim=-1).sum())})
        return
    ref = json.loads((raw / f"{job['depends']}.json").read_text())
    r = PersistentEvictionRunner(bb, p, job["uid"], horizon=cfg["horizon"], record_prefill_attention=False)
    rr = r.reference(eos, frozen_ids=ref["ref"]["ids"])
    if rr["frozen_argmax_agreement"] < 1.0:
        raise RuntimeError("Frozen CR reference not reproduced")
    N, n = len(job["sources"]), rr["n"]
    kl = np.full((N, n), np.nan, dtype="float32")
    ER = np.full((N, n), np.nan, dtype="float32")
    for si, s in enumerate(job["sources"]):
        acc = r.accumulated((s,))
        kl[si] = [x["kl"] for x in acc["token"]]
        ER[si] = acc["E"][:, -1].numpy()
    np.savez_compressed(raw / f"{job['key']}.npz", sources=np.array(job["sources"]), acc_kl=kl, acc_E_R=ER)
    _json(raw / f"{job['key']}.json", meta | {"sources": job["sources"], "n_tokens": n})


def worker(run: Path, idx: int):
    import torch
    m = json.loads((run / "run_manifest.json").read_text())
    if m["code_sha"] != git_sha():
        raise RuntimeError("Code SHA changed; use a NEW run directory")
    jobs = json.loads((run / "jobs.json").read_text())
    raw, claims, errors = (run / d for d in ("raw", "claims", "errors"))
    for d in (raw, claims, errors):
        d.mkdir(exist_ok=True)
    cur = tok = bb = eos = None
    order = {"CRref": 0, "CR": 1, "CTL": 2, "RO": 3}
    while True:
        pend = [j for j in jobs if not (raw / f"{j['key']}.json").exists()
                and not (claims / f"{j['key']}.lock").exists()]
        if not pend:
            return
        ready = [j for j in pend if not j.get("depends") or (raw / f"{j['depends']}.json").exists()]
        if not ready:
            time.sleep(20)
            continue
        ready.sort(key=lambda j: (j["model"] != cur, order[j["kind"]], j["model"], j["key"]))
        job = ready[0]
        try:
            fd = os.open(claims / f"{job['key']}.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
        os.close(fd)
        try:
            if cur != job["model"]:
                tok = bb = None
                torch.cuda.empty_cache()
                tok, bb, eos = _load(job["model"])
                cur = job["model"]
            evaluate(bb, tok, eos, job, m["config"], run)
            print(f"done {job['key']}", flush=True)
        except Exception:
            _json(errors / f"{job['key']}.json", {"key": job["key"], "traceback": traceback.format_exc()})
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--worker", type=int)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--source-run")
    ap.add_argument("--v5-run", default="results/20261031T034209Z_loopzip_v5_stage1_pilot20")
    ap.add_argument("--n-cr", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--gpu-ids", default="0")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    if a.prepare:
        cfg = {"source_run": str(Path(a.source_run).resolve()), "v5_run": str(Path(a.v5_run).resolve()),
               "models": ["ouro-2.6b", "huginn-0125"], "n_cr": a.n_cr, "chunk": a.chunk, "horizon": 128}
        print(json.dumps(prepare(run, cfg), indent=1, default=str))
    elif a.worker is not None:
        worker(run, a.worker)
    else:
        ids = [x for x in a.gpu_ids.split(",") if x]
        (run / "logs").mkdir(exist_ok=True)
        ps = []
        for i, gpu in enumerate(ids):
            h = (run / "logs" / f"worker{i}.txt").open("a")
            ps.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i),
                                        "--run-dir", str(run)], cwd=ROOT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                       stdout=h, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in ps]
        print(json.dumps({"worker_exit_codes": codes}))
        sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
