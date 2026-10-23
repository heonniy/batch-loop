"""Stage 3 (HUGINN_R32_KV_ORACLE_PLAN.md): token-level KV restoration oracle, pilot, Huginn R32 BF16.

Per pilot prompt (frozen prefix = prompt + greedy FULL/decode continuation of the R32 quality run):
  cut  = min(CUT_MAX[dataset], n - 2W); history H = columns [0, P + cut)
  window A (selection)  = continuation targets [cut, cut + W)
  window B (held out)   = continuation targets [cut + W, cut + 2W)
Every row is a causal rerun (kvp.oracle.run_rows) of the same prefix with base LAST/convert and a
restore set S (S's columns keep their per-loop K,V; everything else terminal). Loss = mean next-token
NLL of the FULL-generated tokens on a window; KL(p_FULL || p_row) is reported alongside.

phase 0  [FULL, LAST, LAST x 3 independent initial latents]   -> FULL window distributions, noise
                                                                  scale, per-token K/V convergence
phase 1  48 single-token candidates, batches of [LAST anchor + 7] -> V_i = loss(anchor) - loss(LAST+{i})
phase 2  restore sets at k = round(f * |H|), f in {5, 10, 25}%:
         RANDOM (5 seeds), RECENT, CONV-DIST (distance to terminal), CONV-CONS (consecutive change),
         ORACLE+FILL (top candidates by V_i on window A, V_i > 0, rest filled by CONV-DIST order),
         ORACLE-ONLY (positive-V_A candidates only; fewer bytes)          batches of [LAST anchor + 7]
Per-prompt noise scale (fixed before any Stage-3 result): max over the three latent-perturbed LAST
rows of |NLL_w(LAST_alt) - NLL_w(LAST)| on the same window w.
Byte accounting per token: recurrent KV planes 1 (terminal) or R (restored) x n_rec x {K,V} x H x D x 2B,
plus prelude/coda (n_plain x {K,V} x H x D x 2B) for every token.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import load, make_runner  # noqa: E402
from kvp.oracle import claim, convergence_scores, run_rows  # noqa: E402

MODEL = "huginn-0125"
BASE_RUN = "results/20261023T060125Z_huginn_r32_h200x8"
W = 32
CUT_MAX = {"gsm8k": 128, "math500": 256}
MIN_N = {"gsm8k": 96, "math500": 128}
N_PILOT = {"gsm8k": 24, "math500": 24}
N_CAND = 48
BATCH = 8
FRACS = (0.05, 0.10, 0.25)
N_ALT = 3
SEED = 20261023


def gen_records(base: Path, dataset: str) -> dict[str, dict]:
    out = {}
    for f in sorted((base / "raw").glob(f"{MODEL}__{dataset}__gen__full-decode__b*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def select_pilot(base: Path) -> dict[str, list[str]]:
    """Balanced FULL-correct / incorrect x short / long (median split of eligible lengths)."""
    rng = random.Random(SEED)
    sel = {}
    for ds in ("gsm8k", "math500"):
        recs = {u: r for u, r in gen_records(base, ds).items() if r["n_gen_tokens"] >= MIN_N[ds]}
        lens = sorted(r["n_gen_tokens"] for r in recs.values())
        med = lens[len(lens) // 2]
        cells: dict[tuple, list[str]] = {}
        for u, r in sorted(recs.items()):
            cells.setdefault((bool(r["correct"]), r["n_gen_tokens"] >= med), []).append(u)
        per = N_PILOT[ds] // 4
        chosen, rest = [], []
        for key in sorted(cells):
            pool = cells[key][:]
            rng.shuffle(pool)
            chosen += pool[:per]
            rest += pool[per:]
        rng.shuffle(rest)
        chosen += rest[: N_PILOT[ds] - len(chosen)]
        sel[ds] = sorted(chosen)
    return sel


def make_jobs(args):
    d = Path(args.run_dir) / "stage3"
    d.mkdir(parents=True, exist_ok=True)
    sel = select_pilot(Path(args.base_run))
    (d / "pilot_selection.json").write_text(json.dumps(sel, indent=1))
    jobs = []
    for ds, uids in sel.items():
        recs = gen_records(Path(args.base_run), ds)
        for u in uids:
            jobs.append({"name": f"s3__{ds}__{u.replace('/', '_').replace('.json', '')}", "dataset": ds, "uid": u,
                         "cost": len(u) + recs[u]["n_gen_tokens"] + (10000 if ds == "math500" else 0)})
    jobs.sort(key=lambda j: -j["cost"])
    (d / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"{len(jobs)} stage-3 jobs -> {d}")


def bytes_per_token(bb, planes: int) -> int:
    return planes * bb.n_rec * 2 * bb.H * bb.D * 2


def run_job(ctx, job, out: Path, base: Path):
    spec, tok, model, run, items = ctx
    bb = run.bb
    ds, uid = job["dataset"], job["uid"]
    it = items[ds][uid]
    rec = gen_records(base, ds)[uid]
    prompt, full_cont = it["prompt_ids"], rec["gen_token_ids"]
    P, n = len(prompt), len(full_cont)
    cut = min(CUT_MAX[ds], n - 2 * W)
    cont = full_cont[: cut + 2 * W]
    Hn = P + cut
    T = P + len(cont)
    tgt = torch.tensor(cont, device=run.device)
    win = {"A": (cut, cut + W), "B": (cut + W, cut + 2 * W)}
    t0 = time.time()

    def keep_mask(cols) -> list[bool]:
        s = set(cols)
        return [c in s for c in range(T)]

    def batch(rows, store_full=False, keep_kv=False):
        """Returns per-row {"nll": [2W], "kl": [2W]} (window tokens only)."""
        nll, kl = [], []

        def on_step(g, logits):
            if g < cut:
                return
            lp = torch.log_softmax(logits.float(), -1)
            if store_full:
                lpF[g - cut] = lp[0]
            nll.append(-lp[:, tgt[g]])
            ref = lpF[g - cut].double()
            kl.append((ref.exp()[None] * (ref[None] - lp.double())).sum(-1))

        kv = run_rows(run, prompt, cont, rows, on_step, keep_kv=keep_kv)
        nll_t, kl_t = torch.stack(nll, 1).cpu(), torch.stack(kl, 1).cpu()
        res = [{"nll": [round(x, 6) for x in nll_t[b].tolist()], "kl": [float(f"{x:.6g}") for x in kl_t[b].tolist()]}
               for b in range(len(rows))]
        return res, kv

    def wmean(v, w):
        a, b = win[w]
        return sum(v[a - cut: b - cut]) / (b - a)

    # ---- phase 0
    rows0 = [{"base": "last", "keep": "all", "uid": uid}, {"base": "last", "keep": None, "uid": uid}]
    rows0 += [{"base": "last", "keep": None, "uid": f"{uid}#alt{i + 1}"} for i in range(N_ALT)]
    lpF = torch.empty(2 * W, model.lm_head.weight.shape[0], device=run.device)
    r0, kv = batch(rows0, store_full=True, keep_kv=True)
    conv = convergence_scores(kv, 0, Hn)
    del kv
    torch.cuda.empty_cache()
    full_r, last_r, alt_r = r0[0], r0[1], r0[2:]
    noise = {w: max(abs(wmean(a["nll"], w) - wmean(last_r["nll"], w)) for a in alt_r) for w in win}

    # ---- candidates
    rng = random.Random(f"{SEED}:{uid}")
    recent = list(range(Hn - 16, Hn))
    by_dist = sorted(range(Hn), key=lambda c: -float(conv["dist_to_terminal"][c]))
    cands, tags = [], {}
    for c in recent:
        cands.append(c)
        tags[c] = ["recent"]
    for c in by_dist:
        if sum("conv" in tags.get(x, []) for x in cands) >= 16:
            break
        if c not in tags:
            cands.append(c)
        tags.setdefault(c, []).append("conv")
    pool = [c for c in range(Hn) if c not in tags]
    rng.shuffle(pool)
    for c in pool[: N_CAND - len(cands)]:
        cands.append(c)
        tags[c] = ["random"]

    def run_rows_batched(specs):
        """specs: list of (label, cols). Each batch = [LAST anchor] + up to BATCH-1 specs."""
        out = []
        for i in range(0, len(specs), BATCH - 1):
            chunk = specs[i: i + BATCH - 1]
            rows = [{"base": "last", "keep": None, "uid": uid}] + [
                {"base": "last", "keep": keep_mask(cols), "uid": uid} for _, cols in chunk]
            res, _ = batch(rows)
            anchor = res[0]
            out.append({"label": "__anchor__", "batch": i // (BATCH - 1), **anchor})
            for (label, cols), r in zip(chunk, res[1:]):
                row = {"label": label, "batch": i // (BATCH - 1), "k": len(cols), **r}
                for w in win:
                    row[f"V_{w}"] = wmean(anchor["nll"], w) - wmean(r["nll"], w)
                    row[f"VKL_{w}"] = wmean(anchor["kl"], w) - wmean(r["kl"], w)
                out.append(row)
        return out

    # ---- phase 1: single-token restorations
    p1 = run_rows_batched([(f"cand:{c}", [c]) for c in cands])
    single = {int(r["label"].split(":")[1]): r for r in p1 if r["label"] != "__anchor__"}

    # ---- phase 2: restore sets at equal budgets
    by_cons = sorted(range(Hn), key=lambda c: -float(conv["consecutive"][c]))
    oracle_rank = [c for c in sorted(cands, key=lambda c: -single[c]["V_A"]) if single[c]["V_A"] > 0]
    specs = []
    for f in FRACS:
        k = max(1, round(f * Hn))
        for s in range(5):
            specs.append((f"random{s}@{f}", random.Random(f"{SEED}:{uid}:{f}:{s}").sample(range(Hn), k)))
        specs.append((f"recent@{f}", list(range(Hn - k, Hn))))
        specs.append((f"conv_dist@{f}", by_dist[:k]))
        specs.append((f"conv_cons@{f}", by_cons[:k]))
        top = oracle_rank[:k]
        specs.append((f"oracle_fill@{f}", top + [c for c in by_dist if c not in set(top)][: k - len(top)]))
        specs.append((f"oracle_only@{f}", top if top else []))
    p2 = run_rows_batched(specs)

    res = {"job": job, "source_sha": SOURCE_SHA, "gpu": gpu_identity(), "P": P, "n": n, "cut": cut, "W": W,
           "H": Hn, "T": T, "full_correct": rec["correct"], "full_finish": rec["finish_reason"],
           "level": it["doc"].get("level"),
           "bytes": {"rec_terminal_per_token": bytes_per_token(bb, 1), "rec_full_per_token": bytes_per_token(bb, bb.R),
                     "plain_per_token": bb.n_plain * 2 * bb.H * bb.D * 2, "R": bb.R},
           "phase0": {"full": full_r, "last": last_r, "last_alt": alt_r},
           "noise_scale": noise,
           "full_vs_last": {w: wmean(last_r["nll"], w) - wmean(full_r["nll"], w) for w in win},
           "convergence": {k: [round(float(x), 5) for x in v.tolist()] for k, v in conv.items()},
           "candidates": [{"col": c, "in_prompt": c < P, "dist_from_window": Hn - c, "tags": tags[c],
                           "dist_to_terminal": float(conv["dist_to_terminal"][c]),
                           "consecutive": float(conv["consecutive"][c])} for c in cands],
           "phase1": p1, "phase2": p2, "wall_s": time.time() - t0}
    out.write_text(json.dumps(res))


def work(args):
    d = Path(args.run_dir) / "stage3"
    raw = d / "raw"
    raw.mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    log = open(d / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    jobs = json.loads((d / "jobs.json").read_text())
    ctx = None
    for job in jobs:
        if (raw / f"{job['name']}.json").exists() or not claim(d / "claims", job["name"]):
            continue
        t = time.time()
        try:
            if ctx is None:
                spec, tok, model = load(MODEL, dtype=torch.bfloat16)
                items = {ds: {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]} for ds in ("gsm8k", "math500")}
                ctx = (spec, tok, model, make_runner(spec, tok, model), items)
            tmp = raw / f".{job['name']}.tmp"
            run_job(ctx, job, tmp, Path(args.base_run))
            os.replace(tmp, raw / f"{job['name']}.json")
            log.write(f"done {job['name']} {time.time() - t:.1f}s\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (d / "errors").mkdir(exist_ok=True)
            (d / "errors" / f"{job['name']}.json").write_text(json.dumps({"job": job, "traceback": traceback.format_exc()}))
            log.write(f"FAILED {job['name']}\n{traceback.format_exc()}\n")
            torch.cuda.empty_cache()
    log.write("no pending jobs; exit\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-jobs", "work"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-run", default=BASE_RUN)
    ap.add_argument("--worker", type=int, default=0)
    args = ap.parse_args()
    make_jobs(args) if args.cmd == "make-jobs" else work(args)


if __name__ == "__main__":
    main()
