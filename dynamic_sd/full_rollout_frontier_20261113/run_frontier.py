#!/usr/bin/env python3
"""S1 full-rollout AR / DtV(d,K) / WFD(d) frontier runner (B=1, eager WaveFront, free-running greedy).

Every arm is a COMPLETE teacher-free generation of the same prompt with the same target
(model, dtype, FULL KV, native exit rule, seed); no one-round snapshots. Reuses the instrumented
loops of dynamic_sd/dsd_run.py unchanged (ar_generic, wfd_instr, dtv_live).

Stop strings (lm-eval `until`) are applied DURING generation through a stateful matcher passed in
place of the EOS id: every loop compares each committed token with `eos` exactly once and in
order, so the matcher sees the committed stream and fires on the token that completes a stop
string (or the real EOS). The call count is checked against the committed length per rollout.

Per rollout: tokens + sha256, exact match / first mismatch vs this prompt's AR (same repeat),
task score, wall, prefill_ms (cuda-synchronised wrapper around blocks.prefill), decode_ms,
R calls / rows / width histogram, peak GPU memory, AR depths, DtV rounds (K, accepted, source
depths, verify ticks), WFD accept events. Uninstrumented stage times are written as null
(missing), never 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import random
import subprocess
import sys
import time
from pathlib import Path

DSD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DSD))
import dsd_run as base  # noqa: E402

torch = base.torch
MISSING_STAGE_FIELDS = ("draft_ms", "verifier_ms", "coda_ms", "exit_gate_ms", "host_sync_ms")


class StopMatcher:
    """Stands in for eos_token_id. `tok == matcher` is evaluated once per committed token."""

    def __init__(self, tokenizer, eos_id, stops, window=16):
        self.tok, self.eos, self.stops, self.window = tokenizer, eos_id, list(stops or []), window
        self.reset()

    def reset(self):
        self.seen = []
        self.fired_at = None

    def __eq__(self, other):
        try:
            other = operator.index(other)
        except TypeError:
            return NotImplemented
        self.seen.append(other)
        if self.fired_at is not None:
            return True
        hit = self.eos is not None and other == self.eos
        if not hit and self.stops:
            tail = self.tok.decode(self.seen[-self.window:], skip_special_tokens=False)
            hit = any(s in tail for s in self.stops)
        if hit:
            self.fired_at = len(self.seen)
        return hit

    __hash__ = None


def sha_tokens(t):
    return hashlib.sha256(json.dumps(t).encode()).hexdigest()


def load_rows(a, wrapper):
    pf = base.resolve_prompt_format(a.model, "auto")
    if a.dataset in ("gsm8k", "math500"):
        rows = base.ds.load_dataset_prompts(a.dataset, a.num_fewshot, a.offset + a.limit)[a.offset:]
        for i, r in enumerate(rows):
            r["prompt"] = base.format_prompt(wrapper, r["prompt"], pf)
            r["id"] = f"{a.dataset}:{a.offset + i}"
        return rows
    if a.dataset == "spec_bench":
        sys.path.insert(0, str(Path(base.WF_ROOT) / "experiments" / "exp2_spec_bench"))
        from run import load_corpus  # WaveFront exp2 loader: downloads/caches question.jsonl
        cache = Path(base.WF_ROOT) / "cache"
        load_corpus("spec_bench", 81, 560, cache)
        ids = [int(x) for x in a.spec_ids.split(",")] if a.spec_ids else None
        rows = []
        for line in open(cache / "spec_bench_question.jsonl"):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid, cat = obj.get("question_id"), obj.get("category", "unk")
            prompt = ((obj.get("turns") or [""])[0] or "").strip()   # same selection as WaveFront exp2
            if qid is None or not (81 <= qid <= 560) or not prompt:
                continue
            if ids is not None and qid not in ids:
                continue
            rows.append({"prompt": base.format_prompt(wrapper, prompt, pf), "gold": None, "stop": [],
                         "meta": {"category": cat, "question_id": qid}, "id": f"spec_bench:{qid}:{cat}"})
        return rows
    raise ValueError(a.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500", "spec_bench"])
    ap.add_argument("--num-fewshot", type=int, default=None, help="gsm8k: 3 = LoopSpec gsm8k-cot, 8 = WaveFront")
    ap.add_argument("--spec-ids", default=None, help="comma list of Spec-Bench question_ids")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--T-total", type=int, required=True)
    ap.add_argument("--d", type=int, required=True, help="fixed draft recurrence d (DtV and WFD)")
    ap.add_argument("--Ks", default="1,2,4,8,16")
    ap.add_argument("--wfd", type=int, default=1)
    ap.add_argument("--exit", choices=["fixed", "live"], required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--stops", type=int, default=1, help="apply the task's lm-eval stop strings during generation")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--protocol", default="", help="free-text protocol id recorded in meta")
    ap.add_argument("--out", required=True, help="JSONL (one line per rollout); meta at <out>.meta.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    blocks, wrapper = base.build_blocks(a.model, dtype=dtype, device="cuda", T_total=a.T_total,
                                        state_init="random", state_seed=a.seed)
    kind = "ouro" if "ouro" in a.model else "huginn"
    blocks.max_seq_len = a.max_seq_len
    if kind == "huginn":
        from src.decoding.huginn_blocks import ensure_freqs_cis
        ensure_freqs_cis(blocks.model, a.max_seq_len)
    blocks.kv_budget_s = None                      # FULL depth-indexed KV
    blocks.exit_threshold = None
    blocks.exit_hidden_eps = None
    if a.exit == "live":
        if kind == "ouro":
            blocks.exit_threshold = a.tau
            rule = {"rule": "ouro early_exit_gate cumulative hazard >= tau", "tau": a.tau}
        else:
            blocks.exit_hidden_eps = a.eps
            rule = {"rule": "WaveFront relative-L2 ||h_t-h_{t-1}||/||h_{t-1}|| < eps (NOT Huginn's own "
                            "LatentDiffExitEvaluator normalisation)", "eps": a.eps}
    else:
        rule = {"rule": "fixed", "R": a.T_total}
    instr = base.Instr(blocks, a.T_total, kind)

    # prefill timing (cuda-synchronised); identical wrapper in every arm
    orig_prefill = blocks.prefill
    PF = {"ms": 0.0}

    def timed_prefill(*args, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_prefill(*args, **kw)
        torch.cuda.synchronize()
        PF["ms"] += (time.perf_counter() - t0) * 1000
        return out
    blocks.prefill = timed_prefill

    policy = base.SsdPolicy(T_total=a.T_total, T_draft=a.d, mode="greedy", temperature=1.0, sample_seed=a.seed)
    sampler = policy.make_sampler()
    eos_id = getattr(wrapper, "eos_token_id", None)
    rows = load_rows(a, wrapper)
    Ks = [int(k) for k in a.Ks.split(",")] if a.Ks else []
    arms = ["AR"] + [f"DtV:d{a.d}:K{k}" for k in Ks] + ([f"WFD:d{a.d}"] if a.wfd else [])

    def run_arm(arm, ids1, matcher):
        instr.ctr.reset()
        matcher.reset()
        PF["ms"] = 0.0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        if arm == "AR":
            toks, wall, ev = base.ar_generic(blocks, ids1, a.max_new_tokens, matcher, sampler, a.T_total)
        elif arm.startswith("DtV"):
            K = int(arm.split(":K")[1])
            toks, wall, ev, _ = base.dtv_live(blocks, ids1, a.max_new_tokens, policy, K, matcher)
        else:
            toks, wall, ev, _ = base.wfd_instr(blocks, ids1, a.max_new_tokens, policy, matcher)
        return toks, wall, ev, PF["ms"], instr.ctr.calls, instr.ctr.rows, dict(instr.ctr.width), \
            torch.cuda.max_memory_allocated(), len(matcher.seen)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stops_for = lambda r: (r.get("stop") or []) if a.stops else []
    if rows and a.warmup:
        w = wrapper.encode(rows[0]["prompt"])[:, : a.max_prompt_tokens].squeeze(0).cuda()
        m = StopMatcher(wrapper.tokenizer, eos_id, stops_for(rows[0]))
        saved = a.max_new_tokens
        a.max_new_tokens = 32
        for _ in range(a.warmup):
            for arm in arms:
                run_arm(arm, w, m)
        a.max_new_tokens = saved
    print(f"[frontier] {a.model} {a.dataset} fs={a.num_fewshot} n={len(rows)} exit={a.exit} {rule} "
          f"dtype={a.dtype} d={a.d} arms={arms}", flush=True)
    fh = out.open("w")
    t_start = time.time()
    n_mismatch_calls = 0
    for i, r in enumerate(rows):
        ids1 = wrapper.encode(r["prompt"])[:, : a.max_prompt_tokens].squeeze(0).cuda()
        matcher = StopMatcher(wrapper.tokenizer, eos_id, stops_for(r))
        for rep in range(a.repeats):
            order = list(arms)
            random.Random(f"{a.seed}:{r['id']}:{rep}").shuffle(order)
            recs = {}
            for arm in order:
                toks, wall, ev, pf_ms, calls, nrows, width, mem, ncmp = run_arm(arm, ids1, matcher)
                if ncmp != len(toks):
                    n_mismatch_calls += 1
                txt = wrapper.tokenizer.decode(toks, skip_special_tokens=True)
                correct = None
                if r.get("gold") is not None:
                    ok, pred, _ = base.ds.score(a.dataset, txt, r["gold"], r["stop"], r.get("meta"))
                    correct = bool(ok)
                rec = {"prompt_id": r["id"], "prompt_hash": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                       "prompt_tokens": int(ids1.shape[0]), "arm": arm, "repeat": rep,
                       "order": order.index(arm), "tokens": toks, "generated_ids_hash": sha_tokens(toks),
                       "tokens_committed": len(toks), "stop_fired": matcher.fired_at is not None,
                       "eos_compare_calls": ncmp, "correct": correct,
                       "measured_wall_ms": wall * 1000, "prefill_ms": pf_ms, "decode_ms": wall * 1000 - pf_ms,
                       "rcalls": calls, "core_rows": nrows, "width": width, "gpu_mem_peak": mem,
                       **{f: None for f in MISSING_STAGE_FIELDS}}
                if arm == "AR":
                    rec["depths"] = ev["depths"]
                elif arm.startswith("DtV"):
                    rec["rounds"] = ev["rounds"]
                    rec["discard_rows"] = ev["discard_rows"]
                    rec["exact_in_draft"] = ev["exact_in_draft"]
                    rec["accepted_count"] = sum(x[1] for x in ev["rounds"])
                    rec["draft_tokens"] = sum(x[0] for x in ev["rounds"])
                    rec["rejected_rounds"] = sum(1 for x in ev["rounds"] if x[1] < x[0])
                else:
                    rec["accept_events"] = ev["accept"]
                    rec["accepted_count"] = sum(x[1] for x in ev["accept"])
                    rec["draft_tokens"] = len(ev["accept"])
                    rec["discard_rows"] = ev["discard_rows"]
                    rec["n_ticks"] = ev["n_ticks"]
                    rec["depth_by_pos"] = {str(k): v for k, v in ev["depth"].items()}
                recs[arm] = rec
            base_t = recs["AR"]["tokens"]
            for arm, rec in recs.items():
                o = rec["tokens"]
                fm = next((k for k in range(min(len(o), len(base_t))) if o[k] != base_t[k]), None)
                if fm is None and len(o) != len(base_t):
                    fm = min(len(o), len(base_t))
                rec["first_mismatch_vs_AR"] = fm
                rec["exact_match_to_AR"] = fm is None
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
        print(f"  [{i+1}/{len(rows)}] AR len={len(recs['AR']['tokens'])} " +
              " ".join(f"{arm.split(':',1)[-1]}:{recs['AR']['measured_wall_ms']/recs[arm]['measured_wall_ms']:.2f}x"
                       f"{'' if recs[arm]['exact_match_to_AR'] else '*'}" for arm in arms if arm != "AR") +
              f" ({time.time()-t_start:.0f}s)", flush=True)
    fh.close()
    git = lambda p: subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"args": vars(a), "arms": arms, "target_exit_rule": rule, "kv_policy": "FULL", "batch_size": 1,
            "backend": f"wavefront-eager ({'SDPA-fallback varlen' if a.dtype == 'float32' else 'FA varlen'})",
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "wf_commit": git(base.WF_ROOT),
            "source_commit": git(DSD), "eos_compare_mismatch_rollouts": n_mismatch_calls,
            "missing_fields": list(MISSING_STAGE_FIELDS),
            "claim_scope": "B=1 eager free-running full rollouts; same target per arm"}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print("[frontier] DONE " + json.dumps({"rollouts_with_eos_compare_mismatch": n_mismatch_calls}), flush=True)


if __name__ == "__main__":
    main()
