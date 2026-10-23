"""Stage 1.2-1.3 (HUGINN_R32_KV_ORACLE_PLAN.md): native Huginn HF vs project engine FULL at R=32.

Job kinds (claimed atomically from <run>/parity/jobs.json by independent single-GPU workers):

* fp       -- FP64 native no-cache forward as ground truth; FP32 native no-cache, FP32 native with its
              own HuginnDynamicCache (parallel prompt + token-by-token continuation), FP32 engine
              FULL/decode at B1 and inside a left-padded B4 batch. Short prompt + real GSM8K / MATH
              prompts with gold continuations. Also the initial-latent sensitivity of the native model
              (paired engine noise vs an independent seeded draw vs the model's own randn init).
* greedy   -- BF16, one frozen item: native HF greedy with its own dynamic cache (paired initial
              latents, project stop rules and caps) vs engine FULL/decode B1 greedy vs the batched
              engine record of the R32 quality run; then both models teacher-forced on the SAME
              native-generated sequence (native = one parallel no-cache forward).
* batch    -- BF16 batch-shape noise control: engine FULL/decode score() of the R32 FULL continuation
              for 4 items in one left-padded batch vs each item at B1.

All initial latents are the engine's deterministic per-(uid, position) draws unless stated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp.grading import grade, truncate_at_stops  # noqa: E402
from kvp.jobs import SOURCE_SHA, gpu_identity, prepare_items  # noqa: E402
from kvp.models import eos_ids, load, make_runner  # noqa: E402
from kvp.prompts import protocol  # noqa: E402

MODEL = "huginn-0125"
BASE_RUN = "results/20261023T060125Z_huginn_r32_h200x8"
N_PER_DATASET = 32
FP_CONT = 64
SELECT_SEED = 20261023


# ----------------------------------------------------------------------------- selection


def base_records(base: Path, dataset: str, policy="full", mode="decode") -> dict[str, dict]:
    out = {}
    for f in sorted((base / "raw").glob(f"{MODEL}__{dataset}__gen__{policy}-{mode}__b*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def select_items(base: Path) -> dict[str, list[str]]:
    """Stratified, seeded: 4 strata x 8 per dataset over the R32 FULL/decode generations.
    GSM8K: {correct, incorrect} x {short, long} split at the median generated length.
    MATH-500: correct-short, correct-long, incorrect-ended (eos/stop), incorrect-hit-cap."""
    rng = random.Random(SELECT_SEED)
    sel = {}
    for ds in ("gsm8k", "math500"):
        recs = base_records(base, ds)
        lens = sorted(r["n_gen_tokens"] for r in recs.values())
        med = lens[len(lens) // 2]
        strata: dict[str, list[str]] = {}
        for uid, r in sorted(recs.items()):
            long_ = r["n_gen_tokens"] >= med
            if ds == "math500" and not r["correct"]:
                key = "incorrect-cap" if r["finish_reason"] == "length" else "incorrect-ended"
            else:
                key = f"{'correct' if r['correct'] else 'incorrect'}-{'long' if long_ else 'short'}"
            strata.setdefault(key, []).append(uid)
        chosen, leftover = [], []
        for key in sorted(strata):
            pool = list(strata[key])
            rng.shuffle(pool)
            chosen += pool[: N_PER_DATASET // 4]
            leftover += pool[N_PER_DATASET // 4:]
        rng.shuffle(leftover)
        chosen += leftover[: N_PER_DATASET - len(chosen)]
        sel[ds] = sorted(chosen)
    return sel


# ----------------------------------------------------------------------------- native helpers


def native_nocache(model, ids, noise):
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    st = None if noise is None else noise[None, : len(ids)].to(dev, next(model.parameters()).dtype)
    return model(x, input_states=st, num_steps=32).logits[0]


def huginn_cache_cls(spec):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module("raven_modeling_minimal.HuginnDynamicCache", spec.repo, revision=spec.revision)


def native_cached_tf(model, cache_cls, prompt, cont, noise):
    """Native dynamic cache: one parallel call for the prompt, then one call per continuation token."""
    cache = cache_cls()
    dt = next(model.parameters()).dtype
    seq = prompt + cont
    calls = [(0, len(prompt))] + [(p, p + 1) for p in range(len(prompt), len(seq))]
    outs = []
    for a, b in calls:
        cpos = torch.arange(a, b, device="cuda")
        out = model(torch.tensor([seq[a:b]], device="cuda"), input_states=noise[None, a:b].to("cuda", dt),
                    num_steps=32, past_key_values=cache, use_cache=True, cache_position=cpos)
        outs.append(out.logits[0])
    return torch.cat(outs, 0)


def native_greedy(model, cache_cls, prompt, noise, max_new, eos, stop_check):
    cache = cache_cls()
    dt = next(model.parameters()).dtype
    P = len(prompt)
    out = model(torch.tensor([prompt], device="cuda"), input_states=noise[None, :P].to("cuda", dt), num_steps=32,
                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, device="cuda"))
    logits = out.logits[0, -1]
    gen, finish = [], "length"
    for t in range(max_new):
        tok = int(logits.argmax())
        gen.append(tok)
        if tok in eos:
            finish = "eos"
            break
        if stop_check(gen):
            finish = "stop_string"
            break
        if t == max_new - 1:
            break
        c = P + t
        out = model(torch.tensor([[tok]], device="cuda"), input_states=noise[None, c:c + 1].to("cuda", dt),
                    num_steps=32, past_key_values=cache, use_cache=True,
                    cache_position=torch.tensor([c], device="cuda"))
        logits = out.logits[0, -1]
    return gen, finish


# ----------------------------------------------------------------------------- metrics


def compare(a: torch.Tensor, b: torch.Tensor, P: int | None = None, ref_margin: torch.Tensor | None = None) -> dict:
    """a, b: [N, V] logits (any float dtype) on the same positions; b is the reference."""
    a64, b64 = a.double(), b.double()
    d = (a64 - b64).abs()
    la, lb = torch.log_softmax(a64, -1), torch.log_softmax(b64, -1)
    kl = (lb.exp() * (lb - la)).sum(-1)  # KL(ref || a)
    agree = a64.argmax(-1) == b64.argmax(-1)
    out = {"n_positions": a.shape[0], "max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
           "top1_agreement": agree.double().mean().item(), "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
           "n_top1_disagree": int((~agree).sum())}
    if ref_margin is not None:  # disagreements where the reference itself is (near-)tied
        out["n_top1_disagree_margin_gt_1e-3"] = int(((~agree) & (ref_margin.to(agree.device) > 1e-3)).sum())
    if P is not None:
        out["first_token_top1_agree"] = bool(agree[P - 1])
        out["first_token_kl"] = kl[P - 1].item()
        out["max_abs_logit_diff_prompt"] = d[:P].max().item()
        out["max_abs_logit_diff_cont"] = d[P:].max().item() if d.shape[0] > P else None
    return out


def top2_margin(logits: torch.Tensor) -> torch.Tensor:
    t = logits.double().topk(2, -1).values
    return t[:, 0] - t[:, 1]


def tf_position_stats(eng: torch.Tensor, nat: torch.Tensor, seq: list[int], P: int) -> dict:
    """Per-position arrays over the continuation (predicting seq[P:] from positions P-1..)."""
    sl = slice(P - 1, len(seq) - 1)
    tgt = torch.tensor(seq[P:], device=eng.device)
    le = torch.log_softmax(eng[sl].double(), -1)
    ln = torch.log_softmax(nat[sl].double().to(eng.device), -1)
    kl = (ln.exp() * (ln - le)).sum(-1)
    nll_e = -le.gather(1, tgt[:, None])[:, 0]
    nll_n = -ln.gather(1, tgt[:, None])[:, 0]
    agree = le.argmax(-1) == ln.argmax(-1)
    bad = (~agree).nonzero()
    return {"kl": kl.float().cpu().tolist(), "nll_engine": nll_e.float().cpu().tolist(),
            "nll_native": nll_n.float().cpu().tolist(), "top1_agree": agree.cpu().tolist(),
            "first_top1_disagreement": int(bad[0]) if len(bad) else None,
            "normalization_max_err": max((le.exp().sum(-1) - 1).abs().max().item(),
                                         (ln.exp().sum(-1) - 1).abs().max().item()),
            "any_nan": bool(torch.isnan(kl).any() or torch.isnan(nll_e).any())}


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ----------------------------------------------------------------------------- jobs


def job_fp(args, job, out):
    rep = {"job": job, "gpu": gpu_identity(), "source_sha": SOURCE_SHA}
    spec, tok, model = load(MODEL, dtype=torch.float32)
    run = make_runner(spec, tok, model)
    cases = [("short", "short", tok.encode(
        "Q: A box has 12 red and 7 blue marbles. Half of the red ones are removed. How many marbles are left?\nA:",
        add_special_tokens=False), tok.encode(" Half of 12 is 6, so 6 red marbles remain. 6 + 7 = 13. "
                                              "The answer is 13.", add_special_tokens=False))]
    sel = json.loads((Path(args.run_dir) / "parity" / "selection.json").read_text())
    for ds in ("gsm8k", "math500"):
        items = {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]}
        for uid in sel[ds][:2]:
            it = items[uid]
            cases.append((ds, uid, it["prompt_ids"], it["target_ids"][:FP_CONT]))
    noise = {uid: run.bb.example_noise(uid, len(p) + len(c)) for _, uid, p, c in cases}

    with torch.inference_mode():
        fp32 = {}
        for ds, uid, p, c in cases:
            r = {"native_nocache": native_nocache(model, p + c, noise[uid]).float().cpu(),
                 "native_cached": native_cached_tf(model, huginn_cache_cls(spec), p, c, noise[uid]).float().cpu(),
                 "engine_b1": run.teacher_forced_logits(p, c, uid, "full", "decode")}
            fp32[uid] = r
        # engine inside a left-padded B4 batch (the 4 real prompts), teacher-forced continuation
        real = [x for x in cases if x[0] != "short"]
        sc = run.score([x[2] for x in real], [x[3] for x in real], [x[1] for x in real], "full", "decode")
        b4_lp = {x[1]: lp for x, lp in zip(real, sc.target_logprobs)}
        # initial-latent sensitivity (native, fp32): paired engine noise vs an independent seeded draw vs randn
        lat = []
        for ds, uid, p, c in cases:
            alt = run.bb.example_noise(uid + "#alt", len(p) + len(c))
            a = native_nocache(model, p + c, alt).float().cpu()
            torch.manual_seed(1234)
            z = native_nocache(model, p + c, None).float().cpu()
            lat.append({"case": uid, "paired_vs_alt_seed": compare(a, fp32[uid]["native_nocache"], len(p)),
                        "paired_vs_model_randn_init": compare(z, fp32[uid]["native_nocache"], len(p))})
        rep["latent_init_sensitivity_fp32"] = lat
    del model, run
    torch.cuda.empty_cache()

    # FP64 ground truth on CPU: CUDA float64 softmax is wrong for some row lengths on this torch build
    # (scripts/oracle/repro_cuda_fp64_softmax.py: S=837 rows >= 31), which invalidated a GPU-FP64 reference.
    torch.set_num_threads(min(64, os.cpu_count() or 1))
    spec, tok, model = load(MODEL, device="cpu", dtype=torch.float64)
    rep["fp64_reference_device"] = "cpu"
    rows = []
    with torch.inference_mode():
        for ds, uid, p, c in cases:
            t = time.time()
            ref = native_nocache(model, p + c, noise[uid].double())
            print(f"fp64 cpu reference {uid}: {time.time() - t:.0f}s", flush=True)
            m = top2_margin(ref)
            r = fp32[uid]
            row = {"dataset": ds, "uid": uid, "P": len(p), "n_cont": len(c),
                   "native_fp32_vs_fp64": compare(r["native_nocache"], ref, len(p), m),
                   "native_cached_fp32_vs_fp64": compare(r["native_cached"], ref, len(p), m),
                   "engine_fp32_vs_fp64": compare(r["engine_b1"], ref, len(p), m),
                   "engine_fp32_vs_native_fp32": compare(r["engine_b1"], r["native_nocache"], len(p), m),
                   "native_cached_vs_nocache_fp32": compare(r["native_cached"], r["native_nocache"], len(p), m)}
            if uid in b4_lp:
                tgt = torch.tensor(c)
                lp_ref = torch.log_softmax(ref[len(p) - 1: len(p) + len(c) - 1], -1).gather(1, tgt[:, None])[:, 0]
                lp_b1 = torch.log_softmax(r["engine_b1"][len(p) - 1: len(p) + len(c) - 1].double(), -1).gather(
                    1, tgt[:, None])[:, 0]
                lp_b4 = torch.tensor(b4_lp[uid], dtype=torch.float64)
                row["engine_b4_fp32_logprob_max_abs_diff_vs_fp64"] = (lp_b4 - lp_ref).abs().max().item()
                row["engine_b1_fp32_logprob_max_abs_diff_vs_fp64"] = (lp_b1 - lp_ref).abs().max().item()
                row["engine_b4_vs_b1_fp32_logprob_max_abs_diff"] = (lp_b4 - lp_b1).abs().max().item()
            rows.append(row)
    rep["fp_parity"] = rows
    out.write_text(json.dumps(rep, indent=1))


def job_greedy(args, job, out, ctx):
    spec, tok, model, run, eos = ctx
    cache_cls = huginn_cache_cls(spec)
    ds, uid = job["dataset"], job["uid"]
    items = {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]}
    it = items[uid]
    proto = protocol("huginn", ds)
    eos_strings = sorted({tok.decode([e], skip_special_tokens=False) for e in eos})
    stops = tuple(proto.until) + tuple(eos_strings)

    def stop_text(ids):
        return any(s in tok.decode(ids, skip_special_tokens=False) for s in stops)

    p = it["prompt_ids"]
    cap = proto.max_new_tokens
    noise = run.bb.example_noise(uid, len(p) + cap + 1)
    t0 = time.time()
    with torch.inference_mode():
        nat_gen, nat_fin = native_greedy(model, cache_cls, p, noise, cap, eos, stop_text)
        t_nat = time.time() - t0
        eng = run.generate([p], [uid], "full", "decode", cap, eos, lambda b, ids: stop_text(ids))
        eng_gen, eng_fin = eng.gen_ids[0], eng.finish[0]
        base = base_records(Path(args.base_run), ds)[uid]
        # teacher-forced on the native-generated sequence (cap at the generated tokens)
        seq = p + nat_gen
        nat_logits = native_nocache(model, seq, noise)
        eng_logits = run.teacher_forced_logits(p, nat_gen[:-1], uid, "full", "decode").to("cuda")
        tf = tf_position_stats(eng_logits, nat_logits, seq, len(p))
        first_tok_logits = compare(eng_logits[len(p) - 1:len(p)], nat_logits[len(p) - 1:len(p)])

    def graded(ids, fin):
        raw = tok.decode(ids, skip_special_tokens=False)
        g = grade(ds, truncate_at_stops(raw, stops), it["doc"], fin)
        return {"n_tokens": len(ids), "finish": fin, "correct": g["correct"], "parse_ok": g["parse_ok"],
                "pred": g.get("pred"), "token_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}

    rep = {"job": job, "gpu": gpu_identity(), "source_sha": SOURCE_SHA, "P": len(p), "cap": cap,
           "native": graded(nat_gen, nat_fin), "engine_b1": graded(eng_gen, eng_fin),
           "engine_batched_r32_run": {"n_tokens": base["n_gen_tokens"], "finish": base["finish_reason"],
                                      "correct": base["correct"], "batch": base["batch"],
                                      "batch_pos": base["batch_pos"]},
           "first_divergence": {"native_vs_engine_b1": first_div(nat_gen, eng_gen),
                                "native_vs_engine_batched": first_div(nat_gen, base["gen_token_ids"]),
                                "engine_b1_vs_engine_batched": first_div(eng_gen, base["gen_token_ids"])},
           "first_token_logits_engine_vs_native": first_tok_logits,
           "teacher_forced_on_native_sequence": tf,
           "native_gen_ids": nat_gen, "engine_b1_gen_ids": eng_gen,
           "native_text": tok.decode(nat_gen), "engine_b1_text": tok.decode(eng_gen),
           "wall_s": {"native_greedy": t_nat, "total": time.time() - t0}}
    out.write_text(json.dumps(rep))


def job_batch(args, job, out, ctx):
    spec, tok, model, run, eos = ctx
    ds = job["dataset"]
    items = {it["uid"]: it for it in prepare_items(MODEL, ds, tok)[0]}
    base = base_records(Path(args.base_run), ds)
    uids = job["uids"]
    prompts = [items[u]["prompt_ids"] for u in uids]
    targets = [base[u]["gen_token_ids"] for u in uids]
    with torch.inference_mode():
        b4 = run.score(prompts, targets, uids, "full", "decode").target_logprobs
        rows = []
        for i, u in enumerate(uids):
            b1 = run.score([prompts[i]], [targets[i]], [u], "full", "decode").target_logprobs[0]
            d = [abs(x - y) for x, y in zip(b1, b4[i])]
            rows.append({"uid": u, "n": len(d), "max_abs_logprob_diff": max(d), "mean_abs_logprob_diff": sum(d) / len(d),
                         "sum_logprob_b1": sum(b1), "sum_logprob_b4": sum(b4[i]),
                         "mean_nll_b1": -sum(b1) / len(b1), "mean_nll_b4": -sum(b4[i]) / len(b4[i])})
    out.write_text(json.dumps({"job": job, "gpu": gpu_identity(), "source_sha": SOURCE_SHA, "rows": rows}))


# ----------------------------------------------------------------------------- driver


def make_jobs(args):
    pdir = Path(args.run_dir) / "parity"
    pdir.mkdir(parents=True, exist_ok=True)
    sel = select_items(Path(args.base_run))
    (pdir / "selection.json").write_text(json.dumps(sel, indent=1))
    base = {ds: base_records(Path(args.base_run), ds) for ds in sel}
    jobs = [{"kind": "fp", "name": "fp_parity"}]
    for ds in ("math500", "gsm8k"):  # long (MATH) first
        for uid in sorted(sel[ds], key=lambda u: -base[ds][u]["n_gen_tokens"]):
            jobs.append({"kind": "greedy", "name": f"greedy__{ds}__{uid.replace('/', '_')}", "dataset": ds, "uid": uid})
        for g in range(0, 16, 4):
            jobs.append({"kind": "batch", "name": f"batch__{ds}__g{g // 4}", "dataset": ds, "uids": sel[ds][g:g + 4]})
    (pdir / "jobs.json").write_text(json.dumps(jobs, indent=0))
    print(f"{len(jobs)} parity jobs -> {pdir}")


def claim(d: Path, name: str) -> bool:
    d.mkdir(exist_ok=True)
    try:
        fd = os.open(d / f"{name}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.environ.get('CUDA_VISIBLE_DEVICES')} {os.getpid()} {time.time()}\n".encode())
    os.close(fd)
    return True


def work(args):
    pdir = Path(args.run_dir) / "parity"
    raw, claims = pdir / "raw", pdir / "claims"
    raw.mkdir(exist_ok=True)
    (pdir / "logs").mkdir(exist_ok=True)
    log = open(pdir / "logs" / f"worker{args.worker}.log", "a", buffering=1)
    log.write(f"start gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} pid={os.getpid()}\n")
    jobs = json.loads((pdir / "jobs.json").read_text())
    ctx = None
    for job in jobs:
        if (raw / f"{job['name']}.json").exists() or not claim(claims, job["name"]):
            continue
        t = time.time()
        try:
            tmp = raw / f".{job['name']}.tmp"
            if job["kind"] == "fp":
                if ctx is not None:
                    ctx = None
                    torch.cuda.empty_cache()
                job_fp(args, job, tmp)
            else:
                if ctx is None:
                    spec, tok, model = load(MODEL, dtype=torch.bfloat16)
                    ctx = (spec, tok, model, make_runner(spec, tok, model), eos_ids(spec, tok, model))
                (job_greedy if job["kind"] == "greedy" else job_batch)(args, job, tmp, ctx)
            os.replace(tmp, raw / f"{job['name']}.json")
            log.write(f"done {job['name']} {time.time() - t:.1f}s\n")
        except Exception:  # noqa: BLE001 - recorded, never silently dropped
            (pdir / "errors").mkdir(exist_ok=True)
            (pdir / "errors" / f"{job['name']}.json").write_text(json.dumps({"job": job, "traceback": traceback.format_exc()}))
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
