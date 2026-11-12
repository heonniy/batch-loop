#!/usr/bin/env python3
"""Stage 1: measured same-prefix counterfactual action trials (B=1, eager).

For every prompt:
  pass 1  native AR (live exit / fixed) -> reference continuation (the target).
  pass 2  re-run the SAME incremental AR; at every checkpoint k (last committed
          token gen[k], not yet processed, at position prefix_len+k) snapshot the
          KV state, then for each action in the grid and each repeat:
              restore snapshot -> run exactly ONE decision round -> time it
          and finally restore and continue AR. Pass-2 AR must reproduce pass 1
          token-for-token (checks that restore is exact).

The committed state is the native AR-live state (prefix tokens decoded with
their own live exit depths and freeze-replicated KV), NOT a re-prefill.

Actions: AR (one token) | DtV:d{d}:K{K} (one draft-then-verify round).
WFD is a pipelined scheduler without a well-defined single decision round and
is therefore NOT in this grid.

Each trial is one JSONL row in the oracle_frontier.py schema plus diagnostics.
target_exact = committed tokens equal the reference continuation.
Wall time = cuda-synchronized perf_counter around the round only (decode-only,
no prefill), eager execution, B=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

DYNAMIC_SD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DYNAMIC_SD_DIR))
import dsd_run as base  # noqa: E402  (imports torch + pinned WaveFront via WF_ROOT)

torch = base.torch
ActiveToken = base.ActiveToken


def sync():
    torch.cuda.synchronize()


def sha(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# KV snapshot / restore. Every action from checkpoint position p only writes
# positions >= p (prelude/core/coda slots, freeze-replicate) and only truncates
# seqlen; so saving seqlen + the [p, p+W) window of every slot is exact.
# ---------------------------------------------------------------------------
def snapshot(cache, p, W):
    return {"seqlen": dict(cache.seqlen), "seen": cache._seen_tokens, "keys": set(cache.K),
            "data": {s: (cache.K[s][:, p:p + W].clone(), cache.V[s][:, p:p + W].clone()) for s in cache.K}}


def restore(cache, sn, p, W):
    for s in list(cache.K):
        if s not in sn["keys"]:
            del cache.K[s], cache.V[s], cache.seqlen[s]
    for s, (k, v) in sn["data"].items():
        cache.K[s][:, p:p + W].copy_(k)
        cache.V[s][:, p:p + W].copy_(v)
    cache.seqlen.clear()
    cache.seqlen.update(sn["seqlen"])
    cache._seen_tokens = sn["seen"]


# ---------------------------------------------------------------------------
# One decision round
# ---------------------------------------------------------------------------
@torch.no_grad()
def ar_round(blocks, state, last, pos, sampler, T):
    cache = state.kv_cache
    h, ss = blocks.prelude_forward(cache, last, pos)
    tok = ActiveToken(token_id=last, position=pos, role="verify")
    tok.hidden, tok.static_state, tok.step = h, ss, 0
    for _ in range(T):
        blocks.advance_one_step([tok], cache)
        if tok.should_exit:
            break
    logits = blocks.coda_logits(state, [tok])[0]
    return [sampler(logits, pos + 1)], {"depth": tok.step}


@torch.no_grad()
def dtv_round(blocks, state, last, p, sampler, T, d, K, eos):
    """Identical to dsd_run.dtv_live's loop body (one round, live exits)."""
    cache = state.kv_cache
    sources, drafts, qs, dlog = [], [], [], []
    cur_tok, cur_pos = last, p
    for j in range(K + 1):
        h, ss = blocks.prelude_forward(cache, cur_tok, cur_pos)
        tok = ActiveToken(token_id=cur_tok, position=cur_pos, role="verify")
        tok.hidden, tok.static_state, tok.step = h, ss, 0
        for _ in range(d):
            blocks.advance_one_step([tok], cache)
            if tok.should_exit:
                break
        sources.append(tok)
        lg = blocks.coda_logits(state, [tok])[0]
        dlog.append(lg if tok.should_exit else None)
        nx, q = sampler.draft(lg, cur_pos + 1)
        if j < K:
            drafts.append(nx)
            qs.append(q)
        cur_tok, cur_pos = nx, cur_pos + 1
    vt = 0
    while True:
        live = [s for s in sources if not s.should_exit and s.step < T]
        if not live:
            break
        blocks.advance_one_step(live, cache)
        vt += 1
    need = [s for s, g in zip(sources, dlog) if g is None]
    gl = blocks.coda_logits(state, need) if need else []
    it = iter(gl)
    gold_logits = [g if g is not None else next(it) for g in dlog]
    out, n_acc = [], 0
    for i in range(K):
        accepted, gold = sampler.verify(drafts[i], qs[i], gold_logits[i], p + i + 1)
        if accepted:
            n_acc += 1
            out.append(drafts[i])
            if eos is not None and drafts[i] == eos:
                break
        else:
            out.append(gold)
            cache.drop_positions_at_and_after(p + i + 1)
            break
    else:
        out.append(sampler(gold_logits[K], p + K + 1))
    return out, {"n_acc": n_acc, "depths": [s.step for s in sources], "verify_ticks": vt}


def parse_action(a):
    if a == "AR":
        return ("AR", None, None)
    kind, d, K = a.split(":")
    assert kind == "DtV"
    return ("DtV", int(d[1:]), int(K[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--offset", type=int, default=600)
    ap.add_argument("--T-total", type=int, required=True)
    ap.add_argument("--exit", choices=["fixed", "live"], required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--actions", required=True, help="comma list: AR,DtV:d1:K4,...")
    ap.add_argument("--every", type=int, default=16, help="checkpoint spacing (tokens)")
    ap.add_argument("--first", type=int, default=4, help="first checkpoint index")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=1, help="untimed warmup checkpoints")
    ap.add_argument("--out", required=True, help="JSONL of trials (meta -> <out>.meta.json)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    blocks, wrapper = base.build_blocks(a.model, dtype=dtype, device="cuda", T_total=a.T_total,
                                        state_init="random", state_seed=a.seed)
    kind = "ouro" if "ouro" in a.model else "huginn"
    blocks.kv_budget_s = None                      # FULL depth-indexed KV only
    blocks.exit_threshold = None
    blocks.exit_hidden_eps = None
    if a.exit == "live":
        if kind == "ouro":
            blocks.exit_threshold = a.tau
            exit_policy = f"ouro:tau{a.tau}"
        else:
            blocks.exit_hidden_eps = a.eps
            exit_policy = f"huginn:eps{a.eps}"
    else:
        exit_policy = "fixed"
    T = a.T_total
    policy = base.SsdPolicy(T_total=T, T_draft=1, mode="greedy", temperature=1.0, sample_seed=a.seed)
    sampler = policy.make_sampler()
    eos = getattr(wrapper, "eos_token_id", None)
    actions = a.actions.split(",")
    parsed = {x: parse_action(x) for x in actions}
    maxK = max([p[2] for p in parsed.values() if p[2]] + [1])
    W = maxK + 2
    pf = base.resolve_prompt_format(a.model, "auto")
    rows = base.ds.load_dataset_prompts(a.dataset, None, a.offset + a.limit)[a.offset:]
    for r in rows:
        r["prompt"] = base.format_prompt(wrapper, r["prompt"], pf)

    def run_action(name, state, last, pos):
        kind_, d, K = parsed[name]
        if kind_ == "AR":
            return ar_round(blocks, state, last, pos, sampler, T)
        return dtv_round(blocks, state, last, pos, sampler, T, d, K, eos)

    @torch.no_grad()
    def ar_full(ids1):
        gen, depths = [], []
        state, last_logits = blocks.prefill(ids1.unsqueeze(0), T, 1)
        pl = state.prefix_len
        gen.append(sampler(last_logits, pl))
        while len(gen) < a.max_new_tokens and not (eos is not None and gen[-1] == eos):
            nxt, info = ar_round(blocks, state, gen[-1], pl + len(gen) - 1, sampler, T)
            gen.append(nxt[0])
            depths.append(info["depth"])
        return gen, depths, pl

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")
    stats = {"prompts": 0, "checkpoints": 0, "trials": 0, "inexact_trials": 0, "restore_failures": 0}
    t_start = time.time()
    backend = f"wavefront-eager@{os.popen(f'git -C {base.WF_ROOT} rev-parse --short HEAD').read().strip()}"

    @torch.no_grad()
    def process(i, r, record):
        ids1 = wrapper.encode(r["prompt"])[:, : a.max_prompt_tokens].squeeze(0).cuda()
        ref, ref_depths, pl = ar_full(ids1)
        prompt_ids = ids1.tolist()
        target_hash = sha({"model": a.model, "exit": exit_policy, "dtype": a.dtype, "kv": "FULL",
                           "prompt": prompt_ids, "ref": ref})
        ck = set(range(a.first, len(ref) - W, a.every))
        if not record:
            ck = set(sorted(ck)[: a.warmup])
        state, last_logits = blocks.prefill(ids1.unsqueeze(0), T, 1)
        cache = state.kv_cache
        gen = [sampler(last_logits, pl)]
        k = 0
        while True:
            pos = pl + k
            if k in ck:
                sn = snapshot(cache, pos, W)
                order = [(name, rep) for rep in range(a.repeats) for name in actions]
                rng.shuffle(order)
                for name, rep in order:
                    restore(cache, sn, pos, W)
                    sync()
                    t0 = time.perf_counter()
                    toks, info = run_action(name, state, gen[k], pos)
                    sync()
                    wall_ms = (time.perf_counter() - t0) * 1000
                    exact = toks == ref[k + 1: k + 1 + len(toks)]
                    if record:
                        stats["trials"] += 1
                        stats["inexact_trials"] += int(not exact)
                        fh.write(json.dumps({
                            "prefix_id": f"{r.get('idx', a.offset + i)}:{k}",
                            "prefix_hash": sha({"prompt": prompt_ids, "committed": ref[: k + 1]}),
                            "target_hash": target_hash, "exit_policy": exit_policy, "kv_policy": "FULL",
                            "batch_regime": "B1", "backend": backend, "precision": a.dtype,
                            "action": name, "wall_ms": wall_ms, "committed_tokens": len(toks),
                            "target_exact": bool(exact), "measured": True, "repeat": rep,
                            # diagnostics (not used by oracle_frontier.py)
                            "prompt_idx": a.offset + i, "k": k, "pos": pos,
                            "ref_next_depths": ref_depths[k: k + maxK + 1], "info": info,
                        }) + "\n")
                restore(cache, sn, pos, W)
                if record:
                    stats["checkpoints"] += 1
            if len(gen) >= len(ref):
                break
            nxt, _ = ar_round(blocks, state, gen[k], pos, sampler, T)
            gen.append(nxt[0])
            if nxt[0] != ref[k + 1]:
                stats["restore_failures"] += int(record)
                print(f"  !! pass-2 divergence prompt {i} at k={k+1}", flush=True)
                break
            k += 1
        return len(ref)

    if rows and a.warmup:
        process(0, rows[0], record=False)
    for i, r in enumerate(rows):
        n = process(i, r, record=True)
        stats["prompts"] += 1
        fh.flush()
        print(f"  [{i+1}/{len(rows)}] len={n} trials={stats['trials']} inexact={stats['inexact_trials']} "
              f"restore_fail={stats['restore_failures']} ({time.time()-t_start:.0f}s)", flush=True)
    fh.close()
    meta = {"args": vars(a), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "wf_commit": os.popen(f"git -C {base.WF_ROOT} rev-parse HEAD").read().strip(),
            "batch_loop_commit": subprocess.run(["git", "-C", str(DYNAMIC_SD_DIR), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout.strip(),
            "exit_policy": exit_policy, "W": W, "stats": stats,
            "claim_scope": "B=1 eager same-prefix one-round counterfactuals; not a rollout, no controller cost"}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print("[cf] DONE " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
