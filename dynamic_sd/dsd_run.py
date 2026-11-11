"""Dynamic-depth SSD characterization runner (adapter over pristine WaveFront code).

Upstream WaveFront (summerbro-hhj/wavefront-decoding) is imported unmodified.
This file adds, WITHOUT changing upstream algorithms:
  * row/R-call instrumentation of the recurrent step (monkeypatched wrapper)
  * an instrumented copy of the WFD dynamic scheduler (same decisions)
  * a generic AR loop (validated against upstream ar_generate)
  * DtV with live exits (upstream DtV assumes exits off) -- new code, labelled
  * replay mode: forced per-position exit depth from a recorded AR-live trace
  * observe mode: gate trajectory logging without exits (calibration)

Arms: ar | ar_up (upstream ar_generate) | wfd | wfd_up (upstream scheduler) | dtv | dtv_up
Exit modes: fixed (no exit) | live (native gate) | replay (forced depth trace)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter

WF_ROOT = os.environ.get("WF_ROOT", os.path.expanduser("~/dsd_work/wavefront-decoding"))
sys.path.insert(0, WF_ROOT)

import torch  # noqa: E402

from experiments.exp2_spec_bench.run import build_blocks, format_prompt, resolve_prompt_format  # noqa: E402
from experiments.exp3_accuracy import tasks as ds  # noqa: E402
from src.decoding.policy import SsdPolicy  # noqa: E402
from src.decoding.scheduler import generate_wavefront_dynamic, generate_draft_then_verify  # noqa: E402
from src.decoding.state import ActiveToken  # noqa: E402


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
class Ctr:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.rows = 0
        self.width = Counter()


class Instr:
    """Wraps the model's one-step recurrent advance. Counts R calls / rows, tags
    each token with rows_done, optionally forces replay exits, optionally logs
    the gate trajectory (observe mode)."""

    def __init__(self, blocks, T_total, model_kind):
        self.blocks = blocks
        self.T = T_total
        self.kind = model_kind          # "ouro" | "huginn"
        self.ctr = Ctr()
        self.replay = None              # dict position -> forced exit depth
        self.observe = None             # list to append gate records to
        name = "_advance_one_ut" if hasattr(blocks, "_advance_one_ut") else "_advance_core"
        self.name = name
        self.orig = getattr(blocks, name)
        setattr(blocks, name, self._wrapped)

    def _wrapped(self, toks, cache):
        c = self.ctr
        c.calls += 1
        n = len(toks)
        c.rows += n
        c.width[n] += 1
        for t in toks:
            t.rows_done = getattr(t, "rows_done", 0) + 1
        prev = None
        if self.observe is not None and self.kind == "huginn":
            prev = [t.hidden.float().clone() for t in toks]
        self.orig(toks, cache)
        if self.replay is not None:
            for t in toks:
                D = self.replay.get(t.position)
                if D is not None and t.step >= D and t.step < self.T and not t.should_exit:
                    t.should_exit = True
                    self.blocks._freeze_kv(cache, t.position, t.step)
        if self.observe is not None:
            for i, t in enumerate(toks):
                if self.kind == "ouro":
                    self.observe.append((t.position, t.step, 1.0 - t.exit_remaining))
                else:
                    h = t.hidden.float()
                    d = (h - prev[i]).norm()
                    self.observe.append((t.position, t.step,
                                         float(d / prev[i].norm().clamp_min(1e-6)),   # WFD rel (÷prev)
                                         float(d / h.norm().clamp_min(1e-6))))        # native latent-diff (÷cur)


# ---------------------------------------------------------------------------
# Generic AR (full depth or until exit)
# ---------------------------------------------------------------------------
@torch.no_grad()
def ar_generic(blocks, ids1, max_new, eos, sampler, T):
    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(ids1.unsqueeze(0), T, T)
    cache = state.kv_cache
    gen, depths = [], []
    nxt = sampler(last_logits, state.prefix_len)
    gen.append(nxt)
    pos = state.prefix_len
    hit = eos is not None and nxt == eos
    while len(gen) < max_new and not hit:
        h, ss = blocks.prelude_forward(cache, nxt, pos)
        tok = ActiveToken(token_id=nxt, position=pos, role="verify")
        tok.hidden, tok.static_state, tok.step = h, ss, 0
        for _ in range(T):
            blocks.advance_one_step([tok], cache)
            if tok.should_exit:
                break
        depths.append(tok.step)
        logits = blocks.coda_logits(state, [tok])[0]
        nxt = sampler(logits, pos + 1)
        gen.append(nxt)
        pos += 1
        hit = eos is not None and nxt == eos
    sync()
    return gen, time.perf_counter() - t0, {"depths": depths, "prefix_len": state.prefix_len}


# ---------------------------------------------------------------------------
# Instrumented WFD (decision logic identical to upstream generate_wavefront_dynamic)
# ---------------------------------------------------------------------------
@torch.no_grad()
def wfd_instr(blocks, ids1, max_new, policy, eos):
    sampler = policy.make_sampler()
    T, d = policy.T_total, policy.T_draft
    ev = {"depth": {}, "accept": [], "chain": Counter(), "discard_rows": 0,
          "halted_wait_rows": 0, "wave_width": Counter(), "n_ticks": 0,
          "draft_point_missed": 0, "boundary": 0, "n_rej": 0}
    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(ids1.unsqueeze(0), T_total=T, T_draft=d)
    first = sampler(last_logits, state.prefix_len)
    state.committed.append(first)
    state.last_committed_position = state.prefix_len
    tokens = [first]
    if len(state.committed) >= max_new or (eos is not None and first == eos):
        sync()
        return tokens, time.perf_counter() - t0, ev, state.prefix_len
    state.active_p.append(ActiveToken(token_id=first, position=state.prefix_len, role="draft"))
    call_p = True
    while len(state.committed) < max_new:
        rejected = stop = False
        logits_per_pos = []
        if call_p:
            state, _ = blocks.P_dyn(state)
            call_p = False
        alive_r = [t for t in state.active_r if t.is_alive]
        ev["wave_width"][len(alive_r)] += 1
        ev["halted_wait_rows"] += sum(1 for t in alive_r if t.should_exit)
        state, call_c = blocks.R_dyn(state)
        ev["n_ticks"] += 1
        n_verify = 0
        if call_c:
            state, n_verify, logits_per_pos = blocks.C_multi_dyn(state)
        draft_logits = logits_per_pos[n_verify:]
        if n_verify > 0:
            ev["chain"][n_verify] += 1
            verifies = sorted((t for t in state.active_c if t.role == "verify"), key=lambda t: t.position)
            for i, vtok in enumerate(verifies):
                vtok.is_alive = False
                ev["depth"][vtok.position] = vtok.step
                next_pos = vtok.position + 1
                nxt = verifies[i + 1] if i + 1 < n_verify else state.find_active_prc(next_pos)
                if nxt is None:
                    gold = sampler(logits_per_pos[i], next_pos)
                    accepted = False
                else:
                    accepted, gold = sampler.verify(nxt.token_id, nxt.draft_probs, logits_per_pos[i], next_pos)
                if nxt is None:
                    ev["boundary"] += 1
                    committed_tok = gold
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    tokens.append(gold)
                    state.active_p.append(ActiveToken(token_id=gold, position=next_pos, role="draft"))
                    call_p = True
                elif accepted:
                    ev["accept"].append((next_pos, 1, vtok.step, nxt.step, int(nxt.should_exit), i))
                    committed_tok = nxt.token_id
                    state.committed.append(nxt.token_id)
                    state.last_committed_position = next_pos
                    tokens.append(nxt.token_id)
                else:
                    ev["accept"].append((next_pos, 0, vtok.step, nxt.step, int(nxt.should_exit), i))
                    ev["n_rej"] += 1
                    committed_tok = gold
                    rejected = True
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    tokens.append(gold)
                    # rows thrown away: in-flight tokens at >= next_pos and later chain verifies
                    lost = sum(getattr(t, "rows_done", 0) for t in state.active_r
                               if t.is_alive and t.position >= next_pos)
                    lost += sum(getattr(t, "rows_done", 0) for t in verifies[i + 1:])
                    ev["discard_rows"] += lost
                    state.rollback_prc(next_pos)
                    state.active_p.append(ActiveToken(token_id=gold, position=next_pos, role="draft"))
                    call_p = True
                if (eos is not None and committed_tok == eos) or len(state.committed) >= max_new:
                    stop = True
                if stop or rejected or nxt is None:
                    break
        if stop:
            break
        if (not rejected) and draft_logits:
            src = next((t for t in state.active_c if t.role == "draft"), None)
            if src is not None:
                dt, dq = sampler.draft(draft_logits[0], src.position + 1)
                state.active_p.append(ActiveToken(token_id=dt, position=src.position + 1, role="draft",
                                                  draft_probs=dq))
                call_p = True
        state.active_c = []
    sync()
    wall = time.perf_counter() - t0
    ev["tail_rows"] = sum(getattr(t, "rows_done", 0) for t in state.active_r if t.is_alive)
    return tokens, wall, ev, state.prefix_len


# ---------------------------------------------------------------------------
# DtV with optional live/replay exits (NEW; upstream DtV assumes no exits).
# With exits off it reproduces upstream generate_draft_then_verify decisions.
# ---------------------------------------------------------------------------
@torch.no_grad()
def dtv_live(blocks, ids1, max_new, policy, K, eos):
    sampler = policy.make_sampler()
    T, d = policy.T_total, policy.T_draft
    ev = {"rounds": [], "discard_rows": 0, "verify_ticks": 0, "halted_wait_rows": 0,
          "exact_in_draft": 0, "depth": {}}
    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(ids1.unsqueeze(0), T_total=T, T_draft=d)
    first = sampler(last_logits, state.prefix_len)
    tokens = [first]
    if max_new <= 1 or (eos is not None and first == eos):
        sync()
        return tokens, time.perf_counter() - t0, ev, state.prefix_len
    cache = state.kv_cache
    p, last = state.prefix_len, first
    while len(tokens) < max_new:
        Kc = min(K, max_new - len(tokens))
        sources, drafts, qs, dlog = [], [], [], []
        cur_tok, cur_pos = last, p
        for j in range(Kc + 1):
            h, ss = blocks.prelude_forward(cache, cur_tok, cur_pos)
            tok = ActiveToken(token_id=cur_tok, position=cur_pos, role="verify")
            tok.hidden, tok.static_state, tok.step = h, ss, 0
            for _ in range(d):
                blocks.advance_one_step([tok], cache)
                if tok.should_exit:
                    break
            sources.append(tok)
            lg = blocks.coda_logits(state, [tok])[0]
            dlog.append(lg if tok.should_exit else None)   # halted in draft => exact target logit
            if tok.should_exit:
                ev["exact_in_draft"] += 1
            nx, q = sampler.draft(lg, cur_pos + 1)
            if j < Kc:
                drafts.append(nx)
                qs.append(q)
            cur_tok, cur_pos = nx, cur_pos + 1
        # verify: continue unhalted sources in parallel (halted rows masked out)
        vt = 0
        while True:
            live = [s for s in sources if not s.should_exit and s.step < T]
            if not live:
                break
            blocks.advance_one_step(live, cache)
            ev["halted_wait_rows"] += len(sources) - len(live)
            vt += 1
        ev["verify_ticks"] += vt
        need = [s for s, g in zip(sources, dlog) if g is None]
        gl = blocks.coda_logits(state, need) if need else []
        it = iter(gl)
        gold_logits = [g if g is not None else next(it) for g in dlog]
        for s in sources:
            ev["depth"][s.position] = s.step
        n_acc = 0
        stop = rejected = False
        for i in range(Kc):
            accepted, gold = sampler.verify(drafts[i], qs[i], gold_logits[i], p + i + 1)
            if accepted:
                n_acc += 1
                tokens.append(drafts[i])
                if (eos is not None and drafts[i] == eos) or len(tokens) >= max_new:
                    stop = True
                    break
            else:
                rejected = True
                tokens.append(gold)
                ev["discard_rows"] += sum(getattr(s, "rows_done", 0) for s in sources[i + 1:])
                cache.drop_positions_at_and_after(p + i + 1)
                last, p = gold, p + i + 1
                if (eos is not None and gold == eos) or len(tokens) >= max_new:
                    stop = True
                break
        ev["rounds"].append((Kc, n_acc, [s.step for s in sources], vt))
        if not rejected and not stop:
            bonus = sampler(gold_logits[Kc], p + Kc + 1)
            tokens.append(bonus)
            last, p = bonus, p + Kc + 1
            if (eos is not None and bonus == eos) or len(tokens) >= max_new:
                stop = True
        if stop:
            break
    sync()
    return tokens, time.perf_counter() - t0, ev, state.prefix_len


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--T-total", type=int, required=True)
    ap.add_argument("--T-draft", type=int, default=1)
    ap.add_argument("--K", type=int, default=4, help="DtV draft length")
    ap.add_argument("--arms", default="ar,wfd,dtv")
    ap.add_argument("--exit", default="fixed", choices=["fixed", "live", "replay", "observe"])
    ap.add_argument("--tau", type=float, default=None, help="Ouro cumulative-gate threshold")
    ap.add_argument("--eps", type=float, default=None, help="Huginn WFD relative-L2 epsilon")
    ap.add_argument("--replay-from", default=None, help="JSON with AR-live depth traces")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--sampling", default="greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--kv-s", type=int, default=0, help="KV-sharing budget s (0 = FULL depth-indexed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    blocks, wrapper = build_blocks(a.model, dtype=dtype, device="cuda", T_total=a.T_total,
                                   state_init="random", state_seed=a.seed)
    kind = "ouro" if "ouro" in a.model else "huginn"
    blocks.kv_budget_s = a.kv_s or None           # None = FULL depth-indexed KV (reference)
    blocks.exit_threshold = None
    blocks.exit_hidden_eps = None
    if a.exit == "live":
        if kind == "ouro":
            blocks.exit_threshold = a.tau
        else:
            blocks.exit_hidden_eps = a.eps
    elif a.exit == "observe":
        if kind == "ouro":
            blocks.exit_threshold = 1.01          # gate computed, never fires
        # huginn: our wrapper computes both criteria itself
    instr = Instr(blocks, a.T_total, kind)
    replay_all = None
    if a.exit == "replay":
        waited = 0
        while not os.path.exists(a.replay_from) and waited < 4 * 3600:   # produced by another GPU's job
            time.sleep(30)
            waited += 30
        src = json.load(open(a.replay_from))
        replay_all = {r["idx"]: (r["prefix_len"], r["depths_ar"]) for r in src["per_sample"]}

    policy = SsdPolicy(T_total=a.T_total, T_draft=a.T_draft, mode=a.sampling,
                       temperature=a.temperature, sample_seed=a.seed)
    sampler = policy.make_sampler()
    eos = getattr(wrapper, "eos_token_id", None)
    pf = resolve_prompt_format(a.model, "auto")
    rows = ds.load_dataset_prompts(a.dataset, None, a.offset + a.limit)[a.offset:]
    for r in rows:
        r["prompt"] = format_prompt(wrapper, r["prompt"], pf)
    arms = a.arms.split(",")
    print(f"[dsd] {a.model} {a.dataset} n={len(rows)} T={a.T_total} d={a.T_draft} K={a.K} "
          f"exit={a.exit} tau={a.tau} eps={a.eps} arms={arms}", flush=True)

    def run_arm(arm, ids1):
        instr.ctr.reset()
        sync()
        if arm == "ar":
            toks, wall, info = ar_generic(blocks, ids1, a.max_new_tokens, eos, sampler, a.T_total)
            ev = info
            pl = info["prefix_len"]
        elif arm == "ar_up":
            toks, wall = blocks.ar_generate(ids1, max_new_tokens=a.max_new_tokens, eos_token_id=eos,
                                            sampler=sampler)
            ev, pl = {}, None
        elif arm == "wfd":
            toks, wall, ev, pl = wfd_instr(blocks, ids1, a.max_new_tokens, policy, eos)
        elif arm == "wfd_up":
            toks, tr = generate_wavefront_dynamic(blocks, ids1, max_new_tokens=a.max_new_tokens,
                                                  policy=policy, eos_token_id=eos)
            wall, ev, pl = tr.walltime_s, {"acc": tr.n_drafts_accepted, "prop": tr.n_drafts_proposed}, None
        elif arm == "dtv":
            toks, wall, ev, pl = dtv_live(blocks, ids1, a.max_new_tokens, policy, a.K, eos)
        elif arm == "dtv_up":
            toks, tr = generate_draft_then_verify(blocks, ids1, max_new_tokens=a.max_new_tokens,
                                                  policy=policy, draft_length=a.K, eos_token_id=eos)
            wall, ev, pl = tr.walltime_s, {"acc": tr.n_drafts_accepted, "prop": tr.n_drafts_proposed}, None
        else:
            raise ValueError(arm)
        return toks, wall, ev, pl, instr.ctr.calls, instr.ctr.rows, dict(instr.ctr.width)

    # warmup (untimed)
    if rows and a.warmup:
        w = wrapper.encode(rows[0]["prompt"])[:, : a.max_prompt_tokens].squeeze(0).cuda()
        if replay_all is not None:
            instr.replay = {}
        for _ in range(a.warmup):
            for arm in arms:
                run_arm(arm, w) if a.exit != "observe" else None
        instr.replay = None

    per = []
    agg = {arm: Counter() for arm in arms}
    t_start = time.time()
    for i, r in enumerate(rows):
        idx = a.offset + i
        ids1 = wrapper.encode(r["prompt"])[:, : a.max_prompt_tokens].squeeze(0).cuda()
        rec = {"idx": idx, "gold": r["gold"]}
        if replay_all is not None:
            pl0, dep = replay_all[idx]
            instr.replay = {pl0 + k: D for k, D in enumerate(dep)}
        if a.exit == "observe":
            instr.observe = []
            toks, wall, info = ar_generic(blocks, ids1, a.max_new_tokens, eos, sampler, a.T_total)
            rec.update(tokens_ar=toks, prefix_len=info["prefix_len"], observe=instr.observe)
            instr.observe = None
            per.append(rec)
            print(f"  [{i+1}/{len(rows)}] observe n={len(toks)}", flush=True)
            continue
        for arm in arms:
            toks, wall, ev, pl, calls, nrows, width = run_arm(arm, ids1)
            txt = wrapper.tokenizer.decode(toks, skip_special_tokens=True)
            ok, pred, _ = ds.score(a.dataset, txt, r["gold"], r["stop"], r.get("meta"))
            rec[f"tokens_{arm}"] = toks
            rec[f"wall_{arm}"] = wall
            rec[f"correct_{arm}"] = bool(ok)
            rec[f"rcalls_{arm}"] = calls
            rec[f"rows_{arm}"] = nrows
            rec[f"width_{arm}"] = width
            if pl is not None:
                rec["prefix_len"] = pl
            if arm == "ar":
                rec["depths_ar"] = ev["depths"]
            elif arm in ("wfd", "dtv"):
                e = dict(ev)
                e["depth"] = {str(k): v for k, v in e["depth"].items()}
                for k in ("chain", "wave_width"):
                    if k in e:
                        e[k] = dict(e[k])
                rec[f"ev_{arm}"] = e
            else:
                rec[f"ev_{arm}"] = ev
            g = agg[arm]
            g["tok"] += len(toks); g["wall"] += wall; g["correct"] += int(ok)
            g["rcalls"] += calls; g["rows"] += nrows
        base = rec.get("tokens_ar") or rec.get("tokens_ar_up")
        if base is not None:
            for arm in arms:
                o = rec[f"tokens_{arm}"]
                fm = next((k for k in range(min(len(o), len(base))) if o[k] != base[k]), None)
                if fm is None and len(o) != len(base):
                    fm = min(len(o), len(base))
                rec[f"first_mismatch_{arm}"] = fm
        per.append(rec)
        msg = " ".join(f"{arm}:{agg[arm]['tok']/max(agg[arm]['wall'],1e-9):.1f}t/s" for arm in arms)
        print(f"  [{i+1}/{len(rows)}] {msg} ({time.time()-t_start:.0f}s)", flush=True)

    summary = {}
    for arm in arms:
        g = agg[arm]
        if g["wall"] > 0:
            summary[arm] = {"tok_per_s": g["tok"] / g["wall"], "acc": g["correct"] / len(rows),
                            "rcalls_per_tok": g["rcalls"] / max(g["tok"], 1),
                            "rows_per_tok": g["rows"] / max(g["tok"], 1), "tokens": g["tok"],
                            "wall": g["wall"],
                            "exact_match_frac": (sum(1 for x in per if x.get(f"first_mismatch_{arm}") is None)
                                                 / len(per)) if per and f"first_mismatch_{arm}" in per[0] else None}
    if "ar" in summary:
        for arm in arms:
            if arm in summary:
                summary[arm]["speedup_vs_ar"] = summary[arm]["tok_per_s"] / summary["ar"]["tok_per_s"]
    meta = {"args": vars(a), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "wf_commit": os.popen(f"git -C {WF_ROOT} rev-parse HEAD").read().strip()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"meta": meta, "summary": summary, "per_sample": per}, open(a.out, "w"))
    print("[dsd] SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
