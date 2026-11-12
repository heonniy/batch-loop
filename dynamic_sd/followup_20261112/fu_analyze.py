#!/usr/bin/env python3
"""Analysis for the 2026-11-12 follow-up receipts (CPU only, stdlib only).

  parity   <json[.gz]...>             exact-match vs AR per arm (+ BF16 first-mismatch diagnosis)
  shadow   <dir> <tag>                pool compare_shadow.compare() over repeats r0..rN of one lane
  oracle   <jsonl> [--out-exact F]    strict-exact filtering, oracle_frontier.analyze,
                                      cross-fitted (selection-bias-free) oracle, bootstrap CIs
Outputs markdown to stdout.
"""
import gzip
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_shadow import compare  # noqa: E402
from oracle_frontier import analyze  # noqa: E402


def load(p):
    b = open(p, "rb").read()
    return json.loads(gzip.decompress(b) if p.endswith(".gz") else b)


def boot(values, stat=statistics.mean, n=2000, seed=0):
    rng = random.Random(seed)
    xs = [stat([rng.choice(values) for _ in values]) for _ in range(n)]
    xs.sort()
    return xs[int(0.025 * n)], xs[int(0.975 * n)]


# ---------------------------------------------------------------------------
def first_mismatch_kind(rec, arm):
    """Classify the first token mismatch of `arm` vs AR.
    gen[j] is produced by processing gen[j-1] at position pl+j-1.
    depth_path_diverged: some position < pl+j had a different exit depth in arm vs AR
    (KV/recurrence differs by an exit decision); else same depth path -> pure numeric flip."""
    fm = rec.get(f"first_mismatch_{arm}")
    if fm is None:
        return None
    pl = rec.get("prefix_len")
    dar = rec.get("depths_ar") or []
    dep = (rec.get(f"ev_{arm}") or {}).get("depth") or {}
    if pl is None or not dep or fm == 0:
        return "unknown"
    for j in range(min(fm, len(dar))):
        pos = str(pl + j)
        if pos in dep and dep[pos] != dar[j]:
            return "depth_path_diverged"
    return "same_depth_numeric"


def parity(paths):
    print("| file | dtype | exit | n | arm | exact (SD==AR) | first-mismatch kind |")
    print("|---|---|---|---:|---|---:|---|")
    for p in paths:
        J = load(p)
        if "meta" not in J:
            continue
        a = J["meta"]["args"]
        arms = [x for x in a["arms"].split(",") if x != "ar"]
        ex = a["exit"] + (f" tau={a['tau']}" if a.get("tau") is not None else "") + \
            (f" eps={a['eps']}" if a.get("eps") is not None else "")
        for arm in arms:
            per = J["per_sample"]
            n = len(per)
            exact = sum(1 for r in per if r.get(f"first_mismatch_{arm}") is None)
            kinds = Counter(k for k in (first_mismatch_kind(r, arm) for r in per) if k)
            ks = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "-"
            print(f"| {Path(p).name} | {a['dtype']} | {ex} | {n} | {arm} | {exact}/{n} | {ks} |")


# ---------------------------------------------------------------------------
def shadow(d, tag):
    d = Path(d)
    reps = sorted({f.name.split("_r")[1].split("_")[0] for f in d.glob(f"{tag}_r*_off.json")}, key=int)
    pooled = defaultdict(lambda: defaultdict(list))
    rel = defaultdict(lambda: defaultdict(list))
    counters = defaultdict(list)
    ok, refused = [], []
    for r in reps:
        files = {m: d / f"{tag}_r{r}_{m}.json" for m in ("off", "gpu", "cpu")}
        if not all(f.exists() for f in files.values()):
            refused.append((r, "missing mode"))
            continue
        reports = {m: load(str(f)) for m, f in files.items()}
        try:
            compare(reports)
        except ValueError as e:
            refused.append((r, str(e)))
            continue
        ok.append(r)
        arms = reports["off"]["meta"]["args"]["arms"].split(",")
        for arm in arms:
            w = {m: [s[f"wall_{arm}"] * 1000 for s in reports[m]["per_sample"]] for m in reports}
            ntok = [len(s[f"tokens_{arm}"]) for s in reports["off"]["per_sample"]]
            for key, (x, y) in {"gpu-off": ("gpu", "off"), "cpu-gpu": ("cpu", "gpu"),
                                "cpu-off": ("cpu", "off")}.items():
                diffs = [a - b for a, b in zip(w[x], w[y])]
                pooled[arm][key] += [dd / t for dd, t in zip(diffs, ntok)]          # ms per token
                rel[arm][key] += [dd / o for dd, o in zip(diffs, w["off"])]          # fraction of off wall
            pooled[arm]["off_ms_per_tok"] += [o / t for o, t in zip(w["off"], ntok)]
        for m in ("gpu", "cpu"):
            counters[m].append(reports[m]["shadow_probe"])
    print(f"### shadow lane `{tag}`: matched repeats {ok}; refused {refused}")
    if not ok:
        return
    print("| arm | off ms/tok | gpu−off ms/tok [95% CI] | % of off | cpu−gpu ms/tok [95% CI] | % of off | cpu−off % |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for arm, P in pooled.items():
        def cell(k):
            lo, hi = boot(P[k])
            return f"{statistics.mean(P[k]):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        pc = {k: 100 * statistics.mean(rel[arm][k]) for k in rel[arm]}
        print(f"| {arm} | {statistics.mean(P['off_ms_per_tok']):.2f} | {cell('gpu-off')} | {pc['gpu-off']:+.2f}% "
              f"| {cell('cpu-gpu')} | {pc['cpu-gpu']:+.2f}% | {pc['cpu-off']:+.2f}% |")
    c = counters["cpu"]
    calls = sum(x["calls"] for x in c)
    print(f"\ncpu-mode shadow: {calls} gate calls, {sum(x['rows'] for x in c)} rows; host time inside shadow "
          f"{1e6 * sum(x['shadow_wall_s'] for x in c) / calls:.1f} us/call, of which blocking readback "
          f"{1e6 * sum(x['shadow_readback_wall_s'] for x in c) / calls:.1f} us/call "
          f"(gpu-mode host time {1e6 * sum(x['shadow_wall_s'] for x in counters['gpu']) / max(1, sum(x['calls'] for x in counters['gpu'])):.1f} us/call)")


# ---------------------------------------------------------------------------
def oracle(path, out_exact=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    meta = {}
    mp = Path(str(path) + ".meta.json")
    if mp.exists():
        meta = json.loads(mp.read_text())
    by_prefix = defaultdict(list)
    for r in rows:
        by_prefix[r["prefix_id"]].append(r)
    inexact = Counter(r["action"] for r in rows if not r["target_exact"])
    keep = {p: t for p, t in by_prefix.items() if all(x["target_exact"] for x in t)}
    print(f"### oracle `{Path(path).name}` ({meta.get('exit_policy', '?')}, {meta.get('args', {}).get('dtype', '?')})")
    print(f"trials {len(rows)}, prefixes {len(by_prefix)} ({len({r['prompt_idx'] for r in rows})} prompts); "
          f"inexact trials by action {dict(inexact) or 0}; prefixes kept (all actions exact) {len(keep)}; "
          f"restore/run stats {meta.get('stats')}")
    if not keep:
        return
    exact_rows = [x for t in keep.values() for x in t]
    if out_exact:
        with open(out_exact, "w") as f:
            for x in exact_rows:
                f.write(json.dumps({k: x[k] for k in ("prefix_id", "prefix_hash", "target_hash", "exit_policy",
                                                    "kv_policy", "batch_regime", "backend", "precision", "action",
                                                    "wall_ms", "committed_tokens", "target_exact", "measured",
                                                    "repeat")}) + "\n")
    res = analyze(exact_rows)
    st = res["static"]
    ar = st["AR"]["tokens_per_ms"]
    print("\n| action | tok/ms | vs AR | mean committed tok/round | mean ms/round |")
    print("|---|---:|---:|---:|---:|")
    n_tr = Counter(x["action"] for x in exact_rows)
    for a_, v in sorted(st.items(), key=lambda kv: -kv[1]["tokens_per_ms"]):
        print(f"| {a_} | {v['tokens_per_ms']:.4f} | {v['tokens_per_ms'] / ar:.2f}× | "
              f"{v['tokens'] / n_tr[a_]:.2f} | {v['wall_ms'] / n_tr[a_]:.1f} |")
    best = res["best_static_action"]
    print(f"\nbest static: **{best}** ({st[best]['tokens_per_ms'] / ar:.2f}× AR). "
          f"in-sample one-decision oracle: {res['optimistic_ratio_vs_best_static']:.3f}× best static")

    # cross-fitted oracle: choose per-prefix action on repeats != r, score on repeat r
    reps = sorted({x["repeat"] for x in exact_rows})
    prefixes = sorted(keep)
    def score(trials):
        return sum(x["committed_tokens"] for x in trials), sum(x["wall_ms"] for x in trials)
    cf_tok = cf_ms = st_tok = st_ms = 0.0
    per_prompt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    choice = Counter()
    for p in prefixes:
        T = keep[p]
        prompt = T[0]["prompt_idx"]
        for r in reps:
            train = [x for x in T if x["repeat"] != r]
            test = [x for x in T if x["repeat"] == r]
            acts = defaultdict(list)
            for x in train:
                acts[x["action"]].append(x)
            pick = max(sorted(acts), key=lambda a_: score(acts[a_])[0] / score(acts[a_])[1])
            choice[pick] += 1
            ttok, tms = score([x for x in test if x["action"] == pick])
            btok, bms = score([x for x in test if x["action"] == best])
            cf_tok += ttok; cf_ms += tms; st_tok += btok; st_ms += bms
            pp = per_prompt[prompt]
            pp[0] += ttok; pp[1] += tms; pp[2] += btok; pp[3] += bms
    ratio = (cf_tok / cf_ms) / (st_tok / st_ms)
    keys = list(per_prompt)
    rng = random.Random(0)
    bs = []
    for _ in range(2000):
        s = [per_prompt[rng.choice(keys)] for _ in keys]
        a0, a1, a2, a3 = (sum(v[i] for v in s) for i in range(4))
        bs.append((a0 / a1) / (a2 / a3))
    bs.sort()
    print(f"cross-fitted one-decision oracle (pick on other repeats, score on held-out repeat): "
          f"**{ratio:.3f}× best static** [95% CI over prompts {bs[50]:.3f}, {bs[1949]:.3f}]")
    print(f"cross-fitted picks: {dict(choice.most_common())}")

    # what does the oracle pick depend on? next-token reference depth D (not causal, diagnostic only)
    byD = defaultdict(Counter)
    for p in prefixes:
        T = keep[p]
        D = (T[0].get("ref_next_depths") or [None])[0]
        byD[D][res["by_prefix"][p]["oracle_action"]] += 1
    print("\nin-sample oracle action by reference depth D of the next token (diagnostic, NOT a causal feature):")
    for D in sorted(byD, key=lambda x: (x is None, x)):
        print(f"- D={D}: {dict(byD[D].most_common(4))}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "parity":
        parity(sys.argv[2:])
    elif cmd == "shadow":
        shadow(sys.argv[2], sys.argv[3])
    elif cmd == "oracle":
        out = sys.argv[sys.argv.index("--out-exact") + 1] if "--out-exact" in sys.argv else None
        oracle(sys.argv[2], out)
