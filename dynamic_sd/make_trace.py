"""Synthetic depth traces from a recorded AR-live trace (E5 depth-variance / correlation tests).
  const   : per-sample mean preserved, minimal variance (floor/ceil Bernoulli)
  shuffle : same per-sample depth multiset, positions permuted (kills depth<->token correlation)
  bimodal : per-sample mean preserved, extremes only (1 or T)
usage: python make_trace.py <live.json> <T> <out_prefix>"""
import json
import random
import sys

src, T, pre = sys.argv[1], int(sys.argv[2]), sys.argv[3]
J = json.load(open(src))
rng = random.Random(0)
for mode in ("const", "shuffle", "bimodal"):
    per = []
    for r in J["per_sample"]:
        dep = r["depths_ar"]
        if not dep:
            per.append({"idx": r["idx"], "prefix_len": r["prefix_len"], "depths_ar": dep})
            continue
        m = sum(dep) / len(dep)
        if mode == "const":
            lo = int(m)
            p = m - lo
            new = [min(T, lo + (1 if rng.random() < p else 0)) for _ in dep]
        elif mode == "shuffle":
            new = dep[:]
            rng.shuffle(new)
        else:
            p = (m - 1) / (T - 1)
            new = [T if rng.random() < p else 1 for _ in dep]
        per.append({"idx": r["idx"], "prefix_len": r["prefix_len"], "depths_ar": new})
    json.dump({"per_sample": per, "source": src, "mode": mode}, open(f"{pre}_{mode}.json", "w"))
    print(mode, "written")
