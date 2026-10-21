"""E2 pre-Core probes and E3 early-loop diagnostic probes (frozen model features; CPU/GPU training).

E2 (deployable information set; available before the current token's recurrent work):
  cheap        token class one-hot, position, context length, previous depth, running/moving mean depth
  tok          train-fit token-ID depth PDF (out-of-fold for training rows)
  emb          current-token embedding (Ouro: = initial recurrent state; Huginn: scaled wte)
  prevh        previous token's final recurrent state
  pre          Huginn final Prelude hidden (Ouro has no Transformer Prelude; never treated as equivalent)
E3 (diagnostic upper bound; NOT available at a no-refill decode boundary):
  loop1 (+hazard/latent-diff r1, loop1-emb delta for Ouro) on tokens with D>=2; loop2 (+signals r1..2) on D>=3.
  Compared with a pre-core model trained on the SAME survivor subset and target P(D=r | D>=s).
Models: standardized linear softmax and 2-layer MLP (64 or 128 hidden), Adam(lr 1e-3, wd 1e-4), batch 1024,
<= 40 epochs, early stop (patience 4) on validation NLL, 3 seeds, temperature scaling fit on validation only.
Optional class-weighted loss ablation (inverse frequency) for the best pre-core set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, kendall_tau_b_within_step, load, lockstep_ric, prob_metrics, recovery  # noqa: E402
from e1_attribution import cheap_matrix, features, lookup_probs  # noqa: E402

RMAX = {"ouro": 4, "huginn": 16}
SEEDS = (0, 1, 2)
B_EVAL = (8, 16, 32)


def tok_oof(tr, K, folds=5):
    out = np.zeros(tr["token"].shape + (K,))
    f = np.arange(tr["token"].shape[1]) % folds
    for k in range(folds):
        fit, app = f != k, f == k
        out[:, app] = lookup_probs(tr["token"][:, fit], tr["depth"][:, fit], tr["token"][:, app], K)
    return out


def assemble(raw, f, tokpdf, names, family, stage=0):
    parts = []
    T, N = f["depth"].shape
    for n in names:
        if n == "cheap":
            parts.append(cheap_matrix(f).reshape(T * N, -1))
        elif n == "tok":
            parts.append(tokpdf.reshape(T * N, -1))
        elif n in ("emb", "prevh", "pre", "loop1", "loop2"):
            parts.append(np.asarray(raw[n], dtype=np.float32).reshape(T * N, -1))
        elif n == "sig":  # early-loop scalar signals up to `stage`
            tj = raw["traj"][:, :, :stage].reshape(T * N, stage)
            if family == "ouro":
                cdf = 1 - np.cumprod(1 - tj, axis=1)
                parts.append(np.concatenate([tj, cdf], 1))
            else:
                parts.append(tj)
        elif n == "delta1":
            parts.append((np.asarray(raw["loop1"], np.float32) - np.asarray(raw["emb"], np.float32)).reshape(T * N, -1))
    return np.nan_to_num(np.concatenate(parts, 1).astype(np.float32))


def train_model(Xtr, ytr, Xva, yva, K, hidden, seed, weighted=False, device="cpu"):
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    t = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32, device=device)
    Xt, Xv = t(Xtr), t(Xva)
    yt, yv = torch.tensor(ytr - 1, device=device), torch.tensor(yva - 1, device=device)
    d = Xt.shape[1]
    model = (torch.nn.Linear(d, K) if hidden == 0 else
             torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, K))).to(device)
    w = None
    if weighted:
        cnt = torch.bincount(yt, minlength=K).float().clamp_min(1)
        w = (cnt.sum() / cnt / K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(40):
        model.train()
        for idx in torch.randperm(len(yt), generator=g).split(1024):
            idx = idx.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(Xt[idx]), yt[idx], weight=w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = torch.nn.functional.cross_entropy(model(Xv), yv).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        lv = model(Xv)
    temp = torch.ones(1, requires_grad=True, device=device)
    o2 = torch.optim.LBFGS([temp], max_iter=100)

    def closure():
        o2.zero_grad()
        l = torch.nn.functional.cross_entropy(lv / temp.clamp_min(0.05), yv)
        l.backward()
        return l
    o2.step(closure)
    T_ = float(temp.clamp_min(0.05))

    def predict(X):
        with torch.no_grad():
            return torch.softmax(model(t(X)) / T_, -1).cpu().numpy()
    params = sum(p.numel() for p in model.parameters())
    return predict, {"val_nll": best, "temperature": T_, "params": params, "epochs": ep + 1,
                     "macs_per_token": params}  # dense layers: MACs ~ weight count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features_root", type=Path)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = a.out / a.lane
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((a.features_root / a.lane / "test/manifest.json").read_text())
    fam = man["family"]
    K = RMAX[fam]
    arrs = ("emb", "prevh", "loop1", "loop2") + (("pre",) if fam == "huginn" else ())
    raw = {s: load(a.features_root, a.lane, s, arrs) for s in ("train", "val", "test")}
    f = {s: features(raw[s], fam) for s in raw}
    tokpdf = {"train": tok_oof(f["train"], K),
              "val": lookup_probs(f["train"]["token"], f["train"]["depth"], f["val"]["token"], K),
              "test": lookup_probs(f["train"]["token"], f["train"]["depth"], f["test"]["token"], K)}
    sets = {"cheap": ["cheap"], "cheap+tok": ["cheap", "tok"], "emb": ["emb"], "prevh": ["prevh"],
            "cheap+tok+emb": ["cheap", "tok", "emb"], "cheap+tok+prevh": ["cheap", "tok", "prevh"],
            "all_precore": ["cheap", "tok", "emb", "prevh"] + (["pre"] if fam == "huginn" else [])}
    if fam == "huginn":
        sets["pre"] = ["pre"]
    hiddens = (0, 64, 128)
    if a.quick:
        sets = {k: sets[k] for k in ("cheap", "all_precore")}
        hiddens = (0, 64)
    y = {s: f[s]["depth"] for s in f}
    T, N = y["test"].shape
    rows, preds = [], {}
    fifo = {B: lockstep_ric(y["test"], None, B) for B in B_EVAL}
    orc = {B: lockstep_ric(y["test"], y["test"].astype(float), B) for B in B_EVAL}
    started = time.perf_counter()
    for sname, names in sets.items():
        X = {s: assemble(raw[s], f[s], tokpdf[s], names, fam) for s in raw}
        for h in hiddens:
            for weighted in ((False, True) if sname == "all_precore" and h == 64 else (False,)):
                runs = []
                for seed in SEEDS:
                    pred, info = train_model(X["train"], y["train"].ravel(), X["val"], y["val"].ravel(), K, h, seed,
                                             weighted, a.device)
                    P = pred(X["test"]).reshape(T, N, K)
                    ev = P @ np.arange(1, K + 1)
                    r = {"lane": a.lane, "stage": "E2_precore", "features": sname, "hidden": h, "class_weighted": weighted,
                         "seed": seed, "dim": X["train"].shape[1], **info, **prob_metrics(P.reshape(-1, K), y["test"].ravel()),
                         "mae_expected": float(np.abs(ev - y["test"]).mean()),
                         "kendall_tau_b": kendall_tau_b_within_step(y["test"], ev)}
                    for B in B_EVAL:
                        r[f"RIC_recovery_B{B}"] = recovery(fifo[B], lockstep_ric(y["test"], ev, B), orc[B])
                    runs.append(r)
                    rows.append(r)
                    if seed == 0:
                        preds[f"{sname}|h{h}|w{int(weighted)}"] = P.astype(np.float32)
                print(json.dumps({"set": sname, "hidden": h, "weighted": weighted,
                                  "nll": np.mean([r["nll"] for r in runs]),
                                  "rec16": np.mean([r["RIC_recovery_B16"] for r in runs]),
                                  "t": round(time.perf_counter() - started)}), flush=True)
    # E3: survivor-conditioned early-loop probes vs pre-core on the same subset/target
    pre_names = sets["all_precore"]
    for stage, extra in ((1, ["loop1", "sig"] + (["delta1"] if fam == "ouro" else [])), (2, ["loop1", "loop2", "sig"])):
        smin = stage + 1
        Xs, ys = {}, {}
        for s in raw:
            m = (y[s] >= smin).ravel()
            for tag, names in (("precore", pre_names), ("early", pre_names + extra)):
                Xs[(s, tag)] = assemble(raw[s], f[s], tokpdf[s], names, fam, stage)[m]
            ys[s] = y[s].ravel()[m] - stage  # classes 1..K-stage  => P(D = stage + c | D >= stage + 1)
        Kc = K - stage
        mask_te = (y["test"] >= smin)
        for tag in ("precore", "early"):
            for h in (0, 64):
                for seed in SEEDS[: 1 if a.quick else 3]:
                    pred, info = train_model(Xs[("train", tag)], ys["train"], Xs[("val", tag)], ys["val"], Kc, h, seed,
                                             False, a.device)
                    P = pred(Xs[("test", tag)])
                    ev = P @ np.arange(1, Kc + 1)
                    full = np.full((T, N), np.nan)
                    full[mask_te] = ev
                    tau = kendall_tau_b_within_step(np.where(mask_te, y["test"], 0), np.nan_to_num(full, nan=-1))
                    rows.append({"lane": a.lane, "stage": f"E3_loop{stage}_survivors", "features": tag, "hidden": h,
                                 "class_weighted": False, "seed": seed, "dim": Xs[("train", tag)].shape[1], **info,
                                 **prob_metrics(P, ys["test"]), "mae_expected": float(np.abs(ev - ys["test"]).mean()),
                                 "kendall_tau_b_survivors_with_exited_as_ties": tau, "n_test_survivors": int(mask_te.sum())})
                print(json.dumps({"E3 stage": stage, "set": tag, "hidden": h,
                                  "nll": rows[-1]["nll"], "acc": rows[-1]["acc"]}), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "probe_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "test_predictions.npz", **preds, depth=y["test"], prev=f["test"]["prev"])
    (out / "manifest.json").write_text(json.dumps({"lane": a.lane, "family": fam, "K": K, "sets": sets,
                                                    "hiddens": hiddens, "seeds": SEEDS, "device": a.device,
                                                    "elapsed_s": time.perf_counter() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
