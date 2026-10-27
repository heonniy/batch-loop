"""Step 0 semantic audit (OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md).

  trace   (CPU) toy Ouro (2 layers, R=4), prompt A,B + generated C,D: for FULL, LAST-CONVERT,
          LAST-DECODE (diagnostic alternate reading only) and PROTECT(layer 0, recurrence 1), record every
          K/V write and, for every attention query, which (layer, recurrence, token) version it reads.
          Uses the independent dictionary reference (kvp.reference) and checks that the engine
          (kvp.engine / kvp.banks) produces the same logits -> analysis/trace_reads_writes.csv
  native  (GPU) official Ouro forward (no cache: the pinned checkpoint's own cache class cannot be
          constructed on transformers 4.54.1, and the modeling code needs >= 4.54) with its DEFAULT
          inference path (early_exit_threshold=1.0) vs engine FULL: FP32 teacher-forced logits on real
          prompts + gold continuations, the default path's actual exit steps, and 48-token greedy
          generations (native recomputes the full prefix per token) -> analysis/native_full_parity.csv
  subset  (GPU) fixed 4 GSM8K + 4 MATH500 pilot items, B=1 FP32 full-cap greedy generation for FULL,
          LAST-CONVERT, LAST-DECODE, bankP0 with the project evaluator -> analysis/b1_fp32_subset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from kvp import reference as refmod  # noqa: E402


# ----------------------------------------------------------------------------- trace
class TracingStore(refmod._DictPolicyStore):
    def __init__(self, *a, log=None, names=None, **k):
        super().__init__(*a, **k)
        self.log, self.names = log, names

    def _history_version(self, slots, pos, r, kind, layer=-1):
        v = super()._history_version(slots, pos, r, kind, layer)
        label = next((f"r{i + 1}" for i, s in slots.items() if s is v), "avg")
        if kind == "rec":
            self.log.append({"event": "read", "query_tokens": self._q, "layer": layer, "recurrence": r + 1,
                             "hist_token": self.names[pos], "version_read": label})
        return v

    def store_and_read(self, kind, layer, r, k, v):
        S = k.shape[2]
        self._q = "".join(self.names[self.p0 + i] for i in range(S))
        if kind == "rec":
            for i in range(S):
                self.log.append({"event": "write", "query_tokens": self._q, "layer": layer, "recurrence": r + 1,
                                 "hist_token": self.names[self.p0 + i], "version_read": f"r{r + 1} (own, written)"})
            for i in range(S):
                for j in range(i):
                    self.log.append({"event": "read_within_call", "query_tokens": self.names[self.p0 + i], "layer": layer,
                                     "recurrence": r + 1, "hist_token": self.names[self.p0 + j], "version_read": f"r{r + 1}"})
        return super().store_and_read(kind, layer, r, k, v)


def run_trace(out_dir: Path):
    from tests.test_policy_semantics import _tiny
    from kvp.banks import REGISTRY

    model, run = _tiny("ouro", 4)
    run.model_key = "tiny-ouro"
    REGISTRY[("tiny-ouro", "bankP0")] = frozenset({(0, 0)})
    prompt, cont = [5, 17], [99, 3]
    names = ["A", "B", "C", "D"]
    rows, checks = [], []
    for label, pol, mode, banks in (("FULL", "full", "decode", frozenset()), ("LAST-CONVERT", "last", "convert", frozenset()),
                                    ("LAST-DECODE (diagnostic)", "last", "decode", frozenset()),
                                    ("PROTECT L0R1", "last", "convert", frozenset({(0, 0)}))):
        log = []
        orig = refmod._DictPolicyStore

        def factory(policy, R, region_start, keep=frozenset(), banks_=frozenset(), _log=log):
            return TracingStore(policy, R, region_start, keep, banks_, log=_log, names=names)

        refmod._DictPolicyStore = factory
        try:
            ref = refmod.reference_logits(model, "ouro", run.bb.R, prompt, cont, pol, mode, None, banks=banks)
        finally:
            refmod._DictPolicyStore = orig
        eng_pol = "bankP0" if banks else pol
        eng = run.teacher_forced_logits(prompt, cont, "u", eng_pol, mode)
        checks.append({"policy": label, "engine_vs_reference_max_abs_logit_diff": float((eng - ref).abs().max())})
        for r in log:
            phase = "prefill" if r["query_tokens"] in ("AB", "A", "B") else f"decode({r['query_tokens']})"
            rows.append({"policy": label, "phase": phase, **r})
    # de-duplicate (identical rows from the two-token prefill call)
    seen, uniq = set(), []
    for r in rows:
        k = tuple(r.values())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    with open(out_dir / "trace_reads_writes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(uniq[0]))
        w.writeheader()
        w.writerows(uniq)
    (out_dir / "trace_checks.json").write_text(json.dumps(checks, indent=1))
    # compact human view: decode-time reads of historical tokens
    md = ["| policy | query | layer | recurrence | reads A | reads B | reads C |", "|---|---|---:|---:|---|---|---|"]
    for label in [c["policy"] for c in checks]:
        for q in ("C", "D"):
            for layer in (0, 1):
                for rr in (1, 2, 4):
                    got = {r["hist_token"]: r["version_read"] for r in uniq if r["policy"] == label and r["event"] == "read"
                           and r["query_tokens"] == q and r["layer"] == layer and r["recurrence"] == rr}
                    md.append(f"| {label} | {q} | {layer} | {rr} | {got.get('A', '')} | {got.get('B', '')} | {got.get('C', '')} |")
    (out_dir / "trace_table.md").write_text("\n".join(md) + "\n")
    print(json.dumps(checks, indent=1))


# ----------------------------------------------------------------------------- native parity
def native_logits(model, ids, **kw):
    x = torch.tensor([ids], device="cuda")
    return model(x, use_cache=False, **kw).logits[0].float()


def default_exit_steps(model, ids):
    """Exit step chosen by the checkpoint's default threshold path for every position."""
    x = torch.tensor([ids], device="cuda")
    m = model.model
    out = m(x, use_cache=False)
    gates = out[2]
    lam = [torch.sigmoid(g.squeeze(-1).float()) for g in gates]
    rem = torch.ones_like(lam[0])
    pdf = []
    for i, l in enumerate(lam):
        if i < len(lam) - 1:
            pdf.append(l * rem)
            rem = rem * (1 - l)
        else:
            pdf.append(rem)
    cum = torch.cumsum(torch.stack(pdf, -1), -1)
    thr = model.config.early_exit_threshold
    hit = cum >= thr
    step = torch.argmax(hit.float(), -1)
    step[~hit.any(-1)] = len(lam) - 1
    return (step[0] + 1).tolist()


def run_native(out_dir: Path, models):
    from kvp.jobs import prepare_items
    from kvp.models import load, make_runner

    sel = json.loads((Path(__file__).resolve().parents[2] / "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json").read_text())
    rows = []
    for mk in models:
        for dt in (torch.float32, torch.bfloat16):
            spec, tok, model = load(mk, dtype=dt)
            run = make_runner(spec, tok, model)
            for ds in ("gsm8k", "math500"):
                items = {it["uid"]: it for it in prepare_items(mk, ds, tok)[0]}
                for uid in sel[ds][:4]:
                    it = items[uid]
                    p, c = it["prompt_ids"], it["target_ids"][:64]
                    with torch.inference_mode():
                        nat = native_logits(model, p + c)                       # default inference path
                        nat3 = native_logits(model, p + c, exit_at_step=spec.native_depth - 1)
                        eng = run.teacher_forced_logits(p, c, uid, "full", "decode").to("cuda")
                        steps = default_exit_steps(model, p + c)
                        lp_n, lp_e = torch.log_softmax(nat.double(), -1), torch.log_softmax(eng.double(), -1)
                        kl = (lp_n.exp() * (lp_n - lp_e)).sum(-1)
                        # greedy: native recomputes the full prefix per token (official math, no cache)
                        gen_n, ids_ = [], list(p)
                        for _ in range(48):
                            t = int(native_logits(model, ids_)[-1].argmax())
                            gen_n.append(t)
                            ids_.append(t)
                        gen_e = run.generate([p], [uid], "full", "decode", 48, set(), lambda b, x: False).gen_ids[0]
                    div = next((i for i, (a, b) in enumerate(zip(gen_n, gen_e)) if a != b), None)
                    rows.append({"model": mk, "dtype": str(dt).split(".")[-1], "dataset": ds, "uid": uid, "P": len(p),
                                 "max_abs_logit_diff_default_path": float((nat - eng).abs().max()),
                                 "max_abs_logit_diff_exit_at_last": float((nat3 - eng).abs().max()),
                                 "default_path_equals_exit_at_last": bool(torch.equal(nat, nat3)),
                                 "default_exit_step_counts": json.dumps({s: steps.count(s) for s in sorted(set(steps))}),
                                 "max_kl": float(kl.max()), "top1_agreement": float((nat.argmax(-1) == eng.argmax(-1)).float().mean()),
                                 "greedy48_first_divergence": div})
            del model, run
            torch.cuda.empty_cache()
    with open(out_dir / "native_full_parity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("native parity rows", len(rows))


# ----------------------------------------------------------------------------- B1 FP32 subset
def run_subset(out_dir: Path, run_dir: Path, models):
    from kvp.banks import load_registry
    from kvp.grading import grade, truncate_at_stops
    from kvp.jobs import prepare_items
    from kvp.models import eos_ids, load, make_runner
    from kvp.prompts import protocol

    load_registry(run_dir / "pilot" / "banks.json")
    sel = json.loads((Path(__file__).resolve().parents[2] / "results/20261023T075134Z_huginn_r32_oracle_stage1/parity/selection.json").read_text())
    rows = []
    for mk in models:
        spec, tok, model = load(mk, dtype=torch.float32)
        run = make_runner(spec, tok, model)
        eos = eos_ids(spec, tok, model)
        eos_s = sorted({tok.decode([e], skip_special_tokens=False) for e in eos})
        for ds in ("gsm8k", "math500"):
            items = {it["uid"]: it for it in prepare_items(mk, ds, tok)[0]}
            proto = protocol(spec.family, ds)
            stops = tuple(proto.until) + tuple(eos_s)
            for uid in sel[ds][:4]:
                it = items[uid]
                for pol, mode in (("full", "decode"), ("last", "convert"), ("last", "decode"), ("bankP0", "convert")):
                    t0 = time.time()
                    with torch.inference_mode():
                        res = run.generate([it["prompt_ids"]], [uid], pol, mode, proto.max_new_tokens, eos,
                                           lambda b, x: any(s in tok.decode(x, skip_special_tokens=False) for s in stops))
                    ids, fin = res.gen_ids[0], res.finish[0]
                    g = grade(ds, truncate_at_stops(tok.decode(ids, skip_special_tokens=False), stops), it["doc"], fin)
                    rows.append({"model": mk, "dataset": ds, "uid": uid, "policy": f"{pol}-{mode}", "correct": g["correct"],
                                 "parse_ok": g["parse_ok"], "n_gen": len(ids), "finish": fin, "pred": g.get("pred"),
                                 "prompt_tokens": len(it["prompt_ids"]), "wall_s": round(time.time() - t0, 1)})
        del model, run
        torch.cuda.empty_cache()
    with open(out_dir / "b1_fp32_subset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("subset rows", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["trace", "native", "subset"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default="ouro-1.4b,ouro-2.6b")
    args = ap.parse_args()
    out = Path(args.run_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    {"trace": lambda: run_trace(out), "native": lambda: run_native(out, args.models.split(",")),
     "subset": lambda: run_subset(out, Path(args.run_dir), args.models.split(","))}[args.part]()


if __name__ == "__main__":
    main()
