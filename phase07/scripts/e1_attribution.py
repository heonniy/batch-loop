"""E1 — depth attribution, autocorrelation and cheap baselines (CPU; canonical B=1 labels, TRACE_ONLY).

Associations only: no semantic-causality claim is made from token class / position correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, kendall_tau_b_within_step, load, lockstep_ric, point_metrics, prob_metrics, recovery, token_class  # noqa: E402

POS_EDGES = (0, 8, 16, 32, 64)
CTX_EDGES = (0, 32, 64, 96, 129)
B_EVAL = (8, 16, 32)
RMAX = {"ouro": 4, "huginn": 16}


def bucket(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def features(data, family):
    depth = data["depth"].astype(np.int64)
    T, N = depth.shape
    pos = np.repeat(np.arange(T)[:, None], N, 1)
    ctx = data["prompt_len"][None, :] + pos
    cls = token_class(family, data["token"])
    prev = np.vstack([np.zeros((1, N), int), depth[:-1]])
    cum = np.cumsum(depth, 0)
    runmean = np.vstack([np.zeros((1, N)), cum[:-1] / np.arange(1, T)[:, None]])
    mov = np.zeros((T, N))
    for t in range(1, T):
        mov[t] = depth[max(0, t - 4):t].mean(0)
    return {"depth": depth, "pos": pos, "ctx": ctx, "cls": cls, "prev": prev, "runmean": runmean, "mov4": mov,
            "token": data["token"]}


def cond_xent(train_keys, train_y, test_keys, test_y, K, alpha=1.0):
    """Test cross-entropy (bits) of a smoothed train-fit conditional PDF P(D | key)."""
    counts = defaultdict(lambda: np.zeros(K))
    for k, y in zip(train_keys.ravel(), train_y.ravel()):
        counts[k][y - 1] += 1
    marg = np.bincount(train_y.ravel() - 1, minlength=K) + alpha
    marg = marg / marg.sum()
    xs = []
    for k, y in zip(test_keys.ravel(), test_y.ravel()):
        c = counts.get(k)
        p = (c + alpha * K * marg) / (c.sum() + alpha * K) if c is not None else marg
        xs.append(-math.log2(p[y - 1]))
    return float(np.mean(xs))


def lookup_probs(train_keys, train_y, test_keys, K, alpha=1.0):
    counts = defaultdict(lambda: np.zeros(K))
    for k, y in zip(train_keys.ravel(), train_y.ravel()):
        counts[k][y - 1] += 1
    marg = np.bincount(train_y.ravel() - 1, minlength=K) + alpha
    marg = marg / marg.sum()
    out = np.empty(test_keys.shape + (K,))
    for idx, k in np.ndenumerate(test_keys):
        c = counts.get(k)
        out[idx] = (c + alpha * K * marg) / (c.sum() + alpha * K) if c is not None else marg
    return out


def logistic(trF, trY, teF, K, seed=0, epochs=300):
    import torch
    torch.manual_seed(seed)
    mu, sd = trF.mean(0), trF.std(0) + 1e-6
    X, Xt = torch.tensor((trF - mu) / sd, dtype=torch.float32), torch.tensor((teF - mu) / sd, dtype=torch.float32)
    y = torch.tensor(trY - 1)
    lin = torch.nn.Linear(X.shape[1], K)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=epochs, lr=1)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(X), y) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(lin(Xt), -1).numpy()


def cheap_matrix(f):
    T, N = f["depth"].shape
    onehot_cls = np.eye(len(CLASSES))[f["cls"]]
    return np.concatenate([onehot_cls, f["pos"][..., None] / 64, f["ctx"][..., None] / 128, f["prev"][..., None],
                           (f["prev"] > 0)[..., None], f["runmean"][..., None], f["mov4"][..., None]], -1)


def evaluate(name, probs, point, score, te, K, rows, lane, extra=None):
    y = te["depth"]
    r = {"lane": lane, "baseline": name, **(extra or {})}
    if probs is not None:
        r.update(prob_metrics(probs.reshape(-1, K), y.ravel()))
    r.update(point_metrics(point.ravel(), y.ravel()))
    r["kendall_tau_b"] = kendall_tau_b_within_step(y, score)
    fifo = {B: lockstep_ric(y, None, B) for B in B_EVAL}
    orc = {B: lockstep_ric(y, y.astype(float), B) for B in B_EVAL}
    for B in B_EVAL:
        r[f"RIC_recovery_B{B}"] = recovery(fifo[B], lockstep_ric(y, score, B), orc[B])
    rows.append(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features_root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lanes = sorted(p.name for p in a.features_root.iterdir() if (p / "test" / "meta.npz").exists())
    dist_rows, ent_rows, ac_rows, trans_rows, base_rows, var_rows = [], [], [], [], [], []
    cache = {}
    for lane in lanes:
        man = json.loads((a.features_root / lane / "test/manifest.json").read_text())
        fam = man["family"]
        K = RMAX[fam]
        sp = {s: features(load(a.features_root, lane, s), fam) for s in ("train", "val", "test")
              if (a.features_root / lane / s / "meta.npz").exists()}
        cache[lane] = (fam, sp)
        tr, te = sp["train"], sp["test"]
        # distributions
        allf = {k: np.concatenate([sp[s][k] for s in sp], 1) for k in ("depth", "pos", "ctx", "cls")}
        y = allf["depth"].ravel()
        for gname, g in (("all", np.zeros_like(y)), ("pos_bucket", bucket(allf["pos"].ravel(), POS_EDGES)),
                         ("ctx_bucket", bucket(allf["ctx"].ravel(), CTX_EDGES)), ("token_class", allf["cls"].ravel())):
            for gv in np.unique(g):
                m = g == gv
                h = np.bincount(y[m], minlength=K + 1)[1:]
                dist_rows.append({"lane": lane, "group": gname,
                                  "value": CLASSES[gv] if gname == "token_class" else int(gv), "n": int(m.sum()),
                                  "mean_depth": float(y[m].mean()), **{f"p_d{d}": h[d - 1] / m.sum() for d in range(1, K + 1)}})
        # conditional cross-entropy on test (train-fit, smoothed)
        H0 = cond_xent(np.zeros_like(tr["depth"]), tr["depth"], np.zeros_like(te["depth"]), te["depth"], K)
        for cname, key in (("token_class", "cls"), ("pos_bucket", None), ("ctx_bucket", None), ("token_id", "token"),
                           ("prev_depth", "prev")):
            ktr = (bucket(tr["pos"], POS_EDGES) if cname == "pos_bucket" else bucket(tr["ctx"], CTX_EDGES)
                   if cname == "ctx_bucket" else tr[key])
            kte = (bucket(te["pos"], POS_EDGES) if cname == "pos_bucket" else bucket(te["ctx"], CTX_EDGES)
                   if cname == "ctx_bucket" else te[key])
            Hc = cond_xent(ktr, tr["depth"], kte, te["depth"], K)
            ent_rows.append({"lane": lane, "conditioning": cname, "test_xent_bits": Hc, "marginal_xent_bits": H0,
                             "info_gain_bits": H0 - Hc})
        # autocorrelation within requests (+ request-cluster bootstrap CI)
        D = allf["depth"].astype(float)
        rng = np.random.default_rng(0)
        for k in range(1, 9):
            def corr(cols, centered):
                x = D[:-k, cols].copy()
                z = D[k:, cols].copy()
                if centered:
                    mu = D[:, cols].mean(0)
                    x, z = x - mu, z - mu
                return float(np.corrcoef(x.ravel(), z.ravel())[0, 1])
            allc = np.arange(D.shape[1])
            for centered in (False, True):
                boots = [corr(rng.choice(allc, len(allc)), centered) for _ in range(200)]
                ac_rows.append({"lane": lane, "lag": k, "within_request_centered": centered, "corr": corr(allc, centered),
                                "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))})
        # transition matrix
        prev, cur = allf["depth"][:-1].ravel(), allf["depth"][1:].ravel()
        for d0 in range(1, K + 1):
            m = prev == d0
            if m.any():
                h = np.bincount(cur[m], minlength=K + 1)[1:]
                trans_rows.append({"lane": lane, "prev_depth": d0, "n": int(m.sum()),
                                   **{f"p_next_d{d}": h[d - 1] / m.sum() for d in range(1, K + 1)}})
        var_rows.append({"lane": lane, "between_request_variance_share": float(D.mean(0).var() / D.var()),
                         "depth_mean": float(D.mean()), "depth_cv": float(D.std() / D.mean())})
        # baselines (fit on train only)
        marg = np.bincount(tr["depth"].ravel() - 1, minlength=K) / tr["depth"].size
        T_, N_ = te["depth"].shape
        mp = np.broadcast_to(marg, (T_, N_, K))
        evaluate("marginal_pdf/majority", mp, np.full((T_, N_), float(marg.argmax() + 1)), np.zeros((T_, N_)), te, K, base_rows, lane)
        evaluate("last_depth", None, np.where(te["prev"] > 0, te["prev"], marg.argmax() + 1), te["prev"].astype(float), te, K, base_rows, lane)
        evaluate("moving_mean_4", None, np.where(te["mov4"] > 0, te["mov4"], marg @ np.arange(1, K + 1)), te["mov4"], te, K, base_rows, lane)
        evaluate("running_seq_mean", None, np.where(te["runmean"] > 0, te["runmean"], marg @ np.arange(1, K + 1)), te["runmean"], te, K, base_rows, lane)
        lp = lookup_probs(tr["token"], tr["depth"], te["token"], K)
        ev = lp @ np.arange(1, K + 1)
        evaluate("token_id_lookup", lp, ev, ev, te, K, base_rows, lane)
        lg = logistic(cheap_matrix(tr).reshape(-1, len(CLASSES) + 6), tr["depth"].ravel(),
                      cheap_matrix(te).reshape(-1, len(CLASSES) + 6), K).reshape(T_, N_, K)
        ev = lg @ np.arange(1, K + 1)
        evaluate("class+pos+ctx+history_logistic", lg, ev, ev, te, K, base_rows, lane)
    # transfer: cross-workload (same model+threshold), cross-threshold (same model, alpaca)
    for lane in lanes:
        fam, sp = cache[lane]
        K = RMAX[fam]
        for other in lanes:
            if other == lane or cache[other][0] != fam:
                continue
            same_model = lane.split("_t")[0] == other.split("_t")[0]
            if not same_model:
                continue
            tr, te = sp["train"], cache[other][1]["test"]
            T_, N_ = te["depth"].shape
            lg = logistic(cheap_matrix(tr).reshape(-1, len(CLASSES) + 6), tr["depth"].ravel(),
                          cheap_matrix(te).reshape(-1, len(CLASSES) + 6), K).reshape(T_, N_, K)
            ev = lg @ np.arange(1, K + 1)
            evaluate("class+pos+ctx+history_logistic[transfer]", lg, ev, ev, te, K, base_rows, other,
                     {"train_lane": lane})
    for name, rows in (("e1_depth_distribution", dist_rows), ("e1_conditional_entropy", ent_rows),
                       ("e1_autocorrelation", ac_rows), ("e1_transition", trans_rows), ("e1_baselines", base_rows),
                       ("e1_variance", var_rows)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (a.out / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    plot(dist_rows, ac_rows, a.out)
    print(json.dumps({"lanes": lanes, "baseline_rows": len(base_rows)}))


def plot(dist_rows, ac_rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lanes = sorted({r["lane"] for r in ac_rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for i, lane in enumerate(lanes):
        pts = sorted((r["lag"], r["corr"]) for r in ac_rows if r["lane"] == lane and r["within_request_centered"])
        ax.plot(*zip(*pts), marker="o", ms=4, lw=2, color=pal[i % 8], label=lane)
    ax.axhline(0, color="#8a8984", lw=1)
    ax.set_xlabel("lag k (tokens)")
    ax.set_ylabel("within-request (centered) depth autocorrelation")
    ax.legend(frameon=False, fontsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "e1_autocorrelation.png", dpi=150, facecolor="#fcfcfb")
    fig, axes = plt.subplots(1, len(lanes), figsize=(3.4 * len(lanes), 3.4), squeeze=False)
    for ax, lane in zip(axes[0], lanes):
        for i, c in enumerate(CLASSES):
            r = next((r for r in dist_rows if r["lane"] == lane and r["group"] == "token_class" and r["value"] == c), None)
            if r:
                p = [r[k] for k in sorted((k for k in r if k.startswith("p_d")), key=lambda k: int(k[3:]))]
                ax.plot(range(1, len(p) + 1), np.cumsum(p), marker=".", lw=1.6, color=pal[i], label=f"{c} (n={r['n']})")
        ax.set_title(lane, fontsize=7)
        ax.set_xlabel("depth")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("CDF")
    axes[0][-1].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    fig.savefig(out / "e1_depth_cdf_by_token_class.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
